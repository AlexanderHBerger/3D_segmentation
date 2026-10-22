"""Run the released DMT pipeline (github.com/HuXiaoling/DMT_loss @ b50bc35, morse_DMT/: the
dipha graph-recon fork and ComputeGraphReconstruction.cpp) and write goldens_{2d,3d}.npz.

The released code is built outside the repository, in $DMT_REF:
    git clone https://github.com/HuXiaoling/DMT_loss dmt_src && git -C dmt_src checkout b50bc35
    conda create -p mpi_env -c conda-forge openmpi=4 cxx-compiler
    dipha-graph-recon: cmake (MPI from mpi_env) into dmt_src/morse_DMT/dipha-graph-recon/build_local
    g++ -O2 -o dmt_src/morse_DMT/src/graphrecon dmt_src/morse_DMT/src/ComputeGraphReconstruction.cpp
The files between the programs are written as the released scripts write them
(write_dipha_file_3d_revise.py, load_persistence_diagram.m).

The 2D cases include two crops of FIVES labels ($nnUNet_preprocessed/Dataset035_FIVES), the 3D
cases six 64^3 crops of foreground probabilities of an ATM'26 model (--atm_crops, files
<case>_fg.npz with a "logits" array).

    python make_goldens.py 2d|3d [--atm_crops DIR] [--check]
"""
import argparse
import glob
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from baselines._core import load_core  # noqa: E402

REF = Path(os.environ.get("DMT_REF", "dmt_ref"))
DIPHA = REF / "dmt_src/morse_DMT/dipha-graph-recon/build_local/dipha"
RECON = REF / "dmt_src/morse_DMT/src/graphrecon"
WORK = REF / "work"
THR = [0, 25, 51, 76]


def run_reference(vals):
    """Edges (a, b, persistence) of DIPHA and the graph-reconstruction mask per threshold."""
    WORK.mkdir(parents=True, exist_ok=True)
    v3 = vals if vals.ndim == 3 else vals[:, :, None]
    nx, ny, nz = v3.shape
    with open(WORK / "complex.bin", "wb") as f:
        for v in (8067171840, 1, nx * ny * nz, 3, nx, ny, nz):
            np.int64(v).tofile(f)
        v3.transpose(2, 1, 0).astype(np.float64).ravel().tofile(f)     # x fastest
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    np.savetxt(WORK / "vert.txt", np.stack([i.ravel(), j.ravel(), k.ravel(),
                                            v3.transpose(2, 1, 0).ravel()], 1), fmt="%d")
    env = dict(os.environ, PATH=f"{REF / 'mpi_env/bin'}:" + os.environ["PATH"],
               LD_LIBRARY_PATH=str(REF / "mpi_env/lib"))
    for p in ("diagram.bin", "dipha.edges"):
        (WORK / p).unlink(missing_ok=True)
    subprocess.run([str(DIPHA), str(WORK / "complex.bin"), str(WORK / "diagram.bin"),
                    str(WORK / "dipha.edges"), str(nx), str(ny), str(nz)],
                   check=True, env=env, cwd=WORK, stdout=subprocess.DEVNULL)
    raw = open(WORK / "dipha.edges", "rb").read()
    magic, typ, n = struct.unpack("<qqq", raw[:24])
    assert magic == 8067171840 and typ == 2, (magic, typ)
    rec = np.frombuffer(raw[24:], dtype=np.dtype([("a", "<i8"), ("b", "<f8"), ("p", "<f8")]), count=n)
    np.savetxt(WORK / "dipha_edges.txt",
               np.stack([rec["a"], rec["b"].astype(np.int64), rec["p"].astype(np.int64)], 1), fmt="%d")
    masks = {}
    for t in THR:
        out = WORK / f"out_{t}"
        out.mkdir(exist_ok=True)
        subprocess.run([str(RECON), str(WORK / "vert.txt"), str(WORK / "dipha_edges.txt"), str(t),
                        str(out) + "/"], check=True, cwd=WORK, stdout=subprocess.DEVNULL)
        txt = (out / "dimo_vert.txt").read_text().split()
        arr = np.array(txt, dtype=np.int64).reshape(-1, 4) if txt else np.zeros((0, 4), np.int64)
        m = np.zeros(v3.shape, np.uint8)
        m[arr[:, 0], arr[:, 1], arr[:, 2]] = 1
        masks[t] = m.reshape(vals.shape)
    return rec, masks


def quantise(f):
    return np.trunc(-f.astype(np.float32) * 255).astype(np.int32)


def synth_2d(rng, h, w, kind):
    yy, xx = np.mgrid[:h, :w]
    if kind == "blobs":
        m = np.zeros((h, w), bool)
        for _ in range(6):
            cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(3, 9)
            m |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        f = ndimage.gaussian_filter(m.astype(np.float32), 1.5)
    elif kind == "lines":
        m = np.zeros((h, w), bool)
        for _ in range(4):
            y, x = rng.integers(0, h), rng.integers(0, w)
            for _ in range(60):
                m[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)] = True
                y += rng.integers(-1, 2)
                x += rng.integers(-1, 2)
        m = ndimage.binary_dilation(m, iterations=1)
        f = ndimage.gaussian_filter(m.astype(np.float32), 1.0)
        f[h // 2, :] *= 0.2
    elif kind == "noise":
        f = rng.random((h, w)).astype(np.float32)
    elif kind.startswith("fives:"):
        # blurred label with a gap across the vessels and a faint spurious branch
        case, y0, x0 = kind.split(":")[1:]
        root = Path(os.environ["nnUNet_preprocessed"]) / "Dataset035_FIVES"
        seg = np.load(root / f"{case}_seg.npy")[0]
        f = ndimage.gaussian_filter(seg[int(y0):int(y0) + h, int(x0):int(x0) + w].astype(np.float32), 1.2)
        f[h // 2:h // 2 + 3, :] *= 0.15
        f[:, w // 3] = np.maximum(f[:, w // 3], 0.35)
    elif kind == "plateau":
        f = np.zeros((h, w), np.float32)
        f[h // 4: h // 2, w // 4: 3 * w // 4] = 1.0
        f[h // 3, w // 3] = 0.5
    else:
        raise ValueError(kind)
    return np.clip(f + 0.05 * rng.standard_normal((h, w)).astype(np.float32) * (kind != "plateau"), 0, 1)


def synth_3d(rng, shape, kind):
    X, Y, Z = shape
    if kind == "tubes":
        m = np.zeros(shape, bool)
        for _ in range(4):
            p = rng.integers(0, shape)
            for _ in range(80):
                m[tuple(np.clip(p, 0, np.array(shape) - 1))] = True
                p = p + rng.integers(-1, 2, size=3)
        f = ndimage.gaussian_filter(ndimage.binary_dilation(m, iterations=1).astype(np.float32), 1.0)
        f[X // 2, :, :] *= 0.2
    elif kind == "blobs":
        zz, yy, xx = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")
        m = np.zeros(shape, bool)
        for _ in range(6):
            c = rng.integers(0, shape)
            r = rng.integers(2, 6)
            m |= (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= r ** 2
        f = ndimage.gaussian_filter(m.astype(np.float32), 1.2)
    elif kind == "noise":
        f = rng.random(shape).astype(np.float32)
    elif kind == "plateau":
        f = np.zeros(shape, np.float32)
        f[X // 4: X // 2, Y // 4: 3 * Y // 4, 2: Z - 2] = 1.0
        f[X // 3, Y // 3, Z // 2] = 0.5
        return f
    else:
        raise ValueError(kind)
    return np.clip(f + 0.05 * rng.standard_normal(shape).astype(np.float32), 0, 1)


def atm_crops(directory, n_files=3, offsets=((0, 0, 0), (64, 64, 64))):
    import torch
    out = []
    for path in sorted(glob.glob(f"{directory}/*_fg.npz"))[:n_files]:
        p = torch.softmax(torch.from_numpy(np.load(path)["logits"]), 0)[1].numpy().astype(np.float32)
        for o in offsets:
            out.append(np.ascontiguousarray(p[o[0]:o[0] + 64, o[1]:o[1] + 64, o[2]:o[2] + 64]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dim", choices=("2d", "3d"))
    ap.add_argument("--atm_crops", default=None)
    ap.add_argument("--check", action="store_true", help="compare with the stored goldens instead of writing")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    goldens = {}
    if args.dim == "2d":
        core = load_core("topo_baselines_core")
        specs = [("blobs", 40, 56), ("lines", 64, 64), ("noise", 24, 30), ("plateau", 32, 40),
                 ("blobs", 96, 80), ("lines", 128, 96), ("noise", 48, 48),
                 ("fives:train_001_A:700:900", 128, 128), ("fives:train_310_G:1000:600", 160, 120)]
        volumes = [synth_2d(rng, h, w, kind) for kind, h, w in specs]
    else:
        core = load_core("dmt3d_core")
        specs = [("noise", (9, 11, 13)), ("plateau", (16, 20, 12)), ("blobs", (24, 28, 20)),
                 ("tubes", (32, 32, 32)), ("noise", (20, 18, 16)), ("tubes", (48, 40, 36))]
        volumes = [synth_3d(rng, s, k) for k, s in specs]
        if args.atm_crops:
            volumes += atm_crops(args.atm_crops)
    ok = True
    for n, f in enumerate(volumes):
        vals = quantise(f)
        rec, masks = run_reference(vals)
        edges = core.morse_edges(vals)
        a, b, pers, typ, incons = edges[:5]
        enc = np.where(typ == 0, -pers.astype(np.int64) - 1, pers.astype(np.int64))
        ours = dict(zip(zip(a.tolist(), b.tolist()), enc.tolist()))
        ref = dict(zip(zip(rec["a"].tolist(), rec["b"].astype(np.int64).tolist()), rec["p"].astype(np.int64).tolist()))
        same = ours == ref and incons == 0
        for t in THR:
            m, _ = core.morse_graph_batch(vals[None], t, 1)
            same &= bool((m[0] == masks[t]).all())
            goldens[f"case{n}_thr{t}_mask"] = masks[t] if args.dim == "2d" else np.packbits(masks[t])
        if args.dim == "2d":
            goldens[f"case{n}_vals"] = vals
            goldens[f"case{n}_edges"] = np.stack([rec["a"].astype(np.int64), rec["b"].astype(np.int64),
                                                  rec["p"].astype(np.int64)], 1)
        else:
            goldens[f"case{n}_vals"] = vals.astype(np.int16)
            goldens[f"case{n}_shape"] = np.array(vals.shape)
        ok &= same
        print(f"case {n} {'x'.join(map(str, vals.shape))}: {len(ref)} edges, kernel equal: {same}", flush=True)
    goldens["thresholds" if args.dim == "2d" else "thr"] = np.array(THR)
    path = HERE / f"goldens_{args.dim}.npz"
    if args.check:
        stored = np.load(path)
        equal = set(stored.files) == set(goldens) and all(np.array_equal(stored[k], goldens[k]) for k in goldens)
        print("stored goldens reproduced:", equal)
        ok &= equal
    else:
        np.savez_compressed(path, **goldens)
        print("wrote", path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
