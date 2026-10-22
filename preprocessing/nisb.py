"""NISB-B: neuron boundaries in the synthetic EM volumes of NISB (Rieger et al., 2024).

Raw data: the `base` setting of NISB (https://structuralneurobiologylab.github.io/nisb/),
laid out as <raw>/{train,val,test}/<seedN>/data.zarr/{img,seg}. Every volume is
3000 x 3000 x 1350 voxels at 9 x 9 x 20 nm; seg holds the neuron instances, 0 is the
extracellular space (ECS).

    python -m preprocessing.nisb fingerprint --raw <raw>
    python -m preprocessing.nisb preprocess --raw <raw> [--num_shards N --shard i]
    python -m preprocessing.nisb splits

The target is the boundary between objects, where the ECS counts as one more object. It
is built on the interpixel grid: a volume of shape (X, Y, Z) is lifted to
(2X-1, 2Y-1, 2Z-1), original voxel i sits at 2i, and every lifted voxel looks at the 1, 2,
4 or 8 original voxels around it. It is foreground iff their labels disagree. The result
is well composed, and its background components are exactly the 6-connected components
of the partition {instance 1, ..., instance N, ECS}. The image is lifted by the matching
half-sample interpolation, so original intensities stay on the even indices.

Volumes are cut into 10 x 10 x 9 raw tiles of 301 x 301 x 151 voxels that overlap by one
voxel. The overlap is needed: the lifted voxels between two raw tiles would otherwise
belong to neither lifted tile. Each lifted tile (601 x 601 x 301) is one case. seed0-4
are training, seed100 validation and seed101 the test volume.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from preprocessing.topology import n_critical_counts

DATASET = "Dataset031_NISB_B"
VOLUMES = (("train", "seed0"), ("train", "seed1"), ("train", "seed2"), ("train", "seed3"),
           ("train", "seed4"), ("val", "seed100"), ("test", "seed101"))
RAW_SHAPE = (3000, 3000, 1350)
RAW_TILE = (301, 301, 151)
RAW_STRIDE = (300, 300, 150)
SPACING_NM = (9.0, 9.0, 20.0)
LIFTED_SPACING_NM = (4.5, 4.5, 10.0)


# ------------------------------------------------------------------- raw data

class ZarrArray:
    """Read-only zarr v2 array with Blosc chunks, decoded with blosc2 (no zarr needed)."""

    def __init__(self, path):
        self.path = Path(path)
        meta = json.loads((self.path / ".zarray").read_text())
        if (meta["zarr_format"] != 2 or meta["filters"] is not None or meta["order"] != "C"
                or meta["compressor"]["id"] != "blosc"
                or meta.get("dimension_separator", ".") != "."):
            raise ValueError(f"{self.path}: unsupported zarr layout {meta}")
        self.dtype = np.dtype(meta["dtype"])
        self.fill_value = meta.get("fill_value", 0)
        self.n_trailing = len(meta["shape"]) - 3      # img has a trailing singleton axis
        self.shape = tuple(meta["shape"][:3])
        self.chunks = tuple(meta["chunks"][:3])

    def _chunk(self, index) -> np.ndarray:
        import blosc2

        f = self.path / ".".join(str(i) for i in list(index) + [0] * self.n_trailing)
        if not f.exists():
            return np.full(self.chunks, self.fill_value, dtype=self.dtype)
        buf = blosc2.decompress(f.read_bytes())
        if len(buf) != int(np.prod(self.chunks)) * self.dtype.itemsize:
            raise ValueError(f"{f}: unexpected decompressed size {len(buf)}")
        return np.frombuffer(buf, dtype=self.dtype).reshape(self.chunks)

    def read(self, origin, size) -> np.ndarray:
        out = np.empty(tuple(size), dtype=self.dtype)
        ranges = [range(o // c, (o + s - 1) // c + 1)
                  for o, s, c in zip(origin, size, self.chunks)]
        for index in itertools.product(*ranges):
            chunk = self._chunk(index)
            src, dst = [], []
            for o, s, c, i in zip(origin, size, self.chunks, index):
                lo, hi = max(o, i * c), min(o + s, (i + 1) * c)
                src.append(slice(lo - i * c, hi - i * c))
                dst.append(slice(lo - o, hi - o))
            out[tuple(dst)] = chunk[tuple(src)]
        return out


def _open(raw: Path, split: str, name: str, array: str) -> ZarrArray:
    return ZarrArray(Path(raw) / split / name / "data.zarr" / array)


def _single_threaded():
    import blosc2
    blosc2.set_nthreads(1)


# --------------------------------------------------------------------- tiling

def axis_starts(size: int, tile: int, stride: int) -> list[int]:
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def tiles(name: str):
    """(case_id, raw origin) of every tile of a volume, z fastest."""
    grid = [axis_starts(s, t, st) for s, t, st in zip(RAW_SHAPE, RAW_TILE, RAW_STRIDE)]
    for i, origin in enumerate(itertools.product(*grid)):
        yield f"nisb_{name}_{i:03d}", origin


# ----------------------------------------------------------------------- lift

def lift_label(L: np.ndarray) -> np.ndarray:
    """Instance map (X, Y, Z) -> boolean interfaces (2X-1, 2Y-1, 2Z-1)."""
    X, Y, Z = L.shape
    fg = np.empty((2 * X - 1, 2 * Y - 1, 2 * Z - 1), dtype=bool)
    for ox, oy, oz in itertools.product((0, 1), repeat=3):
        # odd lifted coordinates lie between two original voxels on that axis
        shifts = itertools.product(*[(0, 1) if o else (0,) for o in (ox, oy, oz)])
        views = [L[sx:X - ox + sx, sy:Y - oy + sy, sz:Z - oz + sz] for sx, sy, sz in shifts]
        fg[ox::2, oy::2, oz::2] = np.minimum.reduce(views) != np.maximum.reduce(views)
    return fg


def lift_image(a: np.ndarray) -> np.ndarray:
    """Image (X, Y, Z) -> (2X-1, 2Y-1, 2Z-1): new samples are the mean of their two
    neighbours along each axis, original samples are kept."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    for d in range(3):
        a = np.moveaxis(a, d, 0)
        out = np.empty((2 * a.shape[0] - 1,) + a.shape[1:], dtype=np.float32)
        out[::2] = a
        np.add(a[:-1], a[1:], out=out[1::2])
        out[1::2] *= 0.5
        a = np.moveaxis(out, 0, d)
    return a


# ---------------------------------------------------------------- fingerprint

def _volume_histogram(job) -> np.ndarray:
    raw, split, name = job
    _single_threaded()
    img = _open(raw, split, name, "img")
    hist = np.zeros(256, dtype=np.int64)
    grid = [range(-(-s // c)) for s, c in zip(img.shape, img.chunks)]
    for index in itertools.product(*grid):
        lo = [i * c for i, c in zip(index, img.chunks)]
        size = [min(c, s - l) for l, c, s in zip(lo, img.chunks, img.shape)]
        hist += np.bincount(img.read(lo, size).reshape(-1), minlength=256)
    return hist


def _moments(hist: np.ndarray) -> tuple[float, float]:
    values = np.arange(hist.size, dtype=np.float64)
    n = hist.sum()
    mean = float((values * hist).sum() / n)
    return mean, float(np.sqrt(float((hist * (values - mean) ** 2).sum() / n)))


def _percentile(hist: np.ndarray, q: float) -> float:
    return float(np.searchsorted(np.cumsum(hist), q / 100.0 * hist.sum()))


def fingerprint(args):
    """Intensity statistics over every voxel of the training volumes."""
    train = [(str(args.raw), s, n) for s, n in VOLUMES if s == "train"]
    with ProcessPoolExecutor(max_workers=min(args.num_workers, len(train))) as ex:
        hists = list(ex.map(_volume_histogram, train))
    hist = np.sum(hists, axis=0)
    mean, std = _moments(hist)
    out = {"volumes": [n for _, _, n in train],
           "intensity_properties": {"mean": mean, "std": std,
                                    "percentile_00_5": _percentile(hist, 0.5),
                                    "percentile_50": _percentile(hist, 50.0),
                                    "percentile_99_5": _percentile(hist, 99.5)},
           "histogram": hist.tolist()}
    args.fingerprint.parent.mkdir(parents=True, exist_ok=True)
    args.fingerprint.write_text(json.dumps(out, indent=2) + "\n")
    print(f"mean {mean:.4f} std {std:.4f} -> {args.fingerprint}")


# ----------------------------------------------------------------- preprocess

def _save(path: Path, arr: np.ndarray):
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _write_json(path: Path, obj):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def preprocess_tile(job):
    raw, split, name, case_id, origin, out, mean, std = job
    _single_threaded()
    labels = _open(raw, split, name, "seg").read(origin, RAW_TILE)
    image = _open(raw, split, name, "img").read(origin, RAW_TILE)
    # normalising before the lift is the same as after it: the lift is affine
    image = (image.astype(np.float32) - mean) / max(std, 1e-8)
    fg = lift_label(labels)
    image = lift_image(image)
    if n_critical_counts(fg) != (0, 0):
        raise RuntimeError(f"{case_id}: lifted label is not well composed")

    _save(out / f"{case_id}_data.npy", image[np.newaxis].astype(np.float16))
    _write_json(out / f"{case_id}_properties.json", {
        "case_id": case_id, "volume": f"{split}/{name}",
        "raw_origin": [int(o) for o in origin], "raw_shape": list(RAW_TILE),
        "lifted_shape": list(fg.shape), "spacing_nm": list(SPACING_NM),
        "target_spacing_nm": list(LIFTED_SPACING_NM), "fg_fraction": float(fg.mean()),
        "normalization_mean": mean, "normalization_std": std})
    # written last: an existing _seg.npy marks a finished case
    _save(out / f"{case_id}_seg.npy", fg[np.newaxis].astype(np.int8))
    return case_id


def preprocess(args):
    props = json.loads(args.fingerprint.read_text())["intensity_properties"]
    mean, std = float(props["mean"]), float(props["std"])
    volumes = [v for v in VOLUMES if not args.volumes or v[1] in args.volumes]
    jobs = []
    for split, name in volumes:
        for i, (case_id, origin) in enumerate(tiles(name)):
            if args.tiles is not None and i not in args.tiles:
                continue
            if args.overwrite or not (args.out / f"{case_id}_seg.npy").exists():
                jobs.append((str(args.raw), split, name, case_id, origin, args.out, mean, std))
    jobs = jobs[args.shard::args.num_shards]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} tiles -> {args.out}", flush=True)
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for case_id in ex.map(preprocess_tile, jobs):
            print(case_id, flush=True)

    if args.shard == 0:
        _write_json(args.out / "dataset.json", {
            "name": DATASET, "label_variant": "interfaces",
            "description": "NISB boundaries between objects (neurons and ECS) on the "
                           "interpixel grid; the objects are the 6-connected background "
                           "components.",
            "target_spacing": list(LIFTED_SPACING_NM), "spacing_units": "nm",
            "raw_spacing": list(SPACING_NM),
            "labels": {"object interior or ECS": 0, "interface": 1},
            "num_input_channels": 1, "num_classes": 2,
            "intensity_properties": {"mean": mean, "std": std}, "file_ending": ".npy"})


# --------------------------------------------------------------------- splits

def splits(args):
    """One fold with NISB's own partition; the test volume goes to test_cases.json."""
    ids = {s: sorted(c for split, n in VOLUMES if split == s for c, _ in tiles(n))
           for s in ("train", "val", "test")}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "splits_final.json").write_text(
        json.dumps([{"train": ids["train"], "val": ids["val"]}], indent=2) + "\n")
    (args.out / "test_cases.json").write_text(json.dumps(
        {"volumes": [n for s, n in VOLUMES if s == "test"], "case_ids": ids["test"]},
        indent=2) + "\n")
    print({k: len(v) for k, v in ids.items()})


def main():
    default_out = Path(os.environ.get("nnUNet_preprocessed", ".")) / DATASET
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("fingerprint", "preprocess", "splits"))
    ap.add_argument("--raw", type=Path, help="NISB `base` directory")
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--fingerprint", type=Path, default=None,
                    help="default: <out>/fingerprint.json")
    ap.add_argument("--volumes", nargs="*", help="e.g. seed0 seed101 (default: all seven)")
    ap.add_argument("--tiles", type=int, nargs="*", help="tile indices within each volume")
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.fingerprint = args.fingerprint or args.out / "fingerprint.json"
    if args.step != "splits" and args.raw is None:
        ap.error(f"{args.step} needs --raw")
    {"fingerprint": fingerprint, "preprocess": preprocess, "splits": splits}[args.step](args)


if __name__ == "__main__":
    main()
