"""baselines/dmt.py and its kernels against the released DMT pipeline (DMT_loss @ b50bc35).

dmt_reference/goldens_{2d,3d}.npz hold the output of the released binaries (written by
dmt_reference/make_goldens.py): every edge with its persistence and type (2D) and the
Morse-graph pixel set at four thresholds (2D and 3D). The 3D kernel is also checked against a
brute-force reduction in DIPHA's filtration order and against the 2D kernel on single slices.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from baselines._core import load_core                       # noqa: E402
from baselines.dmt import DMTLoss2D, DMTLoss3D, quantise   # noqa: E402

GOLD_2D = np.load(HERE / "dmt_reference/goldens_2d.npz")
GOLD_3D = np.load(HERE / "dmt_reference/goldens_3d.npz")
CASES_2D = sorted({int(k.split("_")[0][4:]) for k in GOLD_2D.files if k.startswith("case")})


def core2():
    return load_core("topo_baselines_core")


def core3():
    return load_core("dmt3d_core")


# ------------------------------------------------------------------------------ 2D
@pytest.mark.parametrize("case", CASES_2D)
def test_2d_every_edge_matches_dipha(case):
    vals = GOLD_2D[f"case{case}_vals"]
    a, b, pers, typ, incons = core2().morse_edges(vals.astype(np.int32))
    assert incons == 0
    enc = np.where(typ == 0, -pers - 1, pers)
    ours = {(int(x), int(y)): int(e) for x, y, e in zip(a, b, enc)}
    theirs = {(int(x), int(y)): int(e) for x, y, e in GOLD_2D[f"case{case}_edges"]}
    assert ours == theirs


@pytest.mark.parametrize("case", CASES_2D)
@pytest.mark.parametrize("thr", [int(t) for t in GOLD_2D["thresholds"]])
def test_2d_morse_graph_matches_graph_reconstruction(case, thr):
    vals = GOLD_2D[f"case{case}_vals"].astype(np.int32)
    gold = GOLD_2D[f"case{case}_thr{thr}_mask"]
    mask, stats = core2().morse_graph_batch(vals[None], thr, 1)
    assert (mask[0] == gold).all()
    assert stats[0, 6] == gold.sum() and stats[0, 7] == 0


def test_quantisation_is_the_released_one():
    f = torch.tensor([[0.0, 0.2, 0.5], [0.999, 1.0, 0.2 + 1e-7]])
    assert quantise(f).tolist() == [[0, -51, -127], [-254, -255, -51]]
    g = torch.tensor([0.0, 0.2, 0.19999, 1.0, 0.5])
    assert quantise(g).tolist() == [int(-float(x) * 255) for x in g]


def test_persistence_fraction_maps_to_integer_threshold():
    assert DMTLoss2D(persistence=0.2).threshold == 51
    assert DMTLoss2D(persistence=0.1).threshold == 26
    assert DMTLoss2D(persistence=0.3).threshold == 76


def test_2d_threads_equal_serial():
    rng = np.random.default_rng(0)
    v = quantise(torch.from_numpy(rng.random((4, 64, 80)).astype(np.float32))).numpy()
    m1, s1 = core2().morse_graph_batch(v, 51, 1)
    m4, s4 = core2().morse_graph_batch(v, 51, 4)
    assert (m1 == m4).all() and (s1 == s4).all()


def test_2d_gradient_lives_on_the_morse_graph():
    torch.manual_seed(0)
    gt = torch.zeros(1, 40, 60, dtype=torch.long)
    gt[:, 18:22, 5:55] = 1
    lg = torch.randn(1, 2, 40, 60) * 0.3
    lg[:, 1] += gt[0].float() * 6 - 3
    lg[:, 1, :, 28:31] = -2.0
    lg.requires_grad_(True)
    fn = DMTLoss2D(persistence=0.2)
    v = fn(lg, gt, with_stats=True)
    mask, _ = fn.morse_mask(lg.detach(), gt)
    assert fn.last_stats["n_mask"] == int(mask.sum()) > 0
    assert mask[0, :, 28:31].any()
    v.backward()
    g = lg.grad[0].abs().sum(0)
    assert (g[~mask[0]] == 0).all() and (g[mask[0]] > 0).all()


def test_reductions():
    torch.manual_seed(1)
    gt = torch.zeros(2, 32, 32, dtype=torch.long)
    gt[:, 10:14, :] = 1
    lg = torch.randn(2, 2, 32, 32)
    lg[:, 1] += gt.float() * 4 - 2
    a, b = DMTLoss2D(reduction="mean"), DMTLoss2D(reduction="crit_mean")
    va, vb = a(lg, gt, with_stats=True), b(lg, gt)
    k, n = a.last_stats["n_mask"], a.last_stats["n_pixels"]
    assert k > 0 and torch.allclose(va * n, vb * k, rtol=1e-5)


def test_no_structure_costs_nothing():
    lg = torch.zeros(1, 2, 16, 16, requires_grad=True)
    fn = DMTLoss2D()
    v = fn(lg, torch.zeros(1, 16, 16, dtype=torch.long), with_stats=True)
    assert float(v.detach()) == 0.0 and fn.last_stats["n_mask"] == 0
    v.backward()
    assert lg.grad is not None


# ------------------------------------------------------------------------------ 3D
def brute_edges(vol):
    """Every edge of a dim-0 or dim-1 pair with its encoded persistence, from a plain Z2
    column reduction of the cubical complex in DIPHA's order (value, dim, cell index)."""
    X, Y, Z = vol.shape
    cx, cy, cz = 2 * X - 1, 2 * Y - 1, 2 * Z - 1
    cells = []
    for tz in range(cz):
        for ty in range(cy):
            for tx in range(cx):
                t = (tx, ty, tz)
                odd = [d for d in range(3) if t[d] % 2]
                verts = []
                for bits in itertools.product((-1, 1), repeat=len(odd)):
                    v = list(t)
                    for d, s in zip(odd, bits):
                        v[d] += s
                    verts.append((v[0] // 2, v[1] // 2, v[2] // 2))
                cells.append((max(vol[v] for v in verts), len(odd), tx + cx * (ty + cy * tz), t))
    cells.sort(key=lambda c: c[:3])
    pos = {c[3]: i for i, c in enumerate(cells)}
    cols = []
    for *_, t in cells:
        bd = set()
        for d in range(3):
            if t[d] % 2:
                for s in (-1, 1):
                    u = list(t)
                    u[d] += s
                    bd.add(pos[tuple(u)])
        cols.append(bd)
    owner, out = {}, {}

    def lat(t):
        return (t[0] // 2) + X * ((t[1] // 2) + Y * (t[2] // 2))

    for j in range(len(cells)):
        c = cols[j]
        while c and max(c) in owner:
            c ^= cols[owner[max(c)]]
        if not c:
            continue
        low = max(c)
        owner[low] = j
        bval, bdim, _, bt = cells[low]
        dval, _, _, dt = cells[j]
        if bdim == 2:
            continue
        e, enc = (dt, -(dval - bval) - 1) if bdim == 0 else (bt, dval - bval)
        d = [k for k in range(3) if e[k] % 2][0]
        lo, hi = list(e), list(e)
        lo[d] -= 1
        hi[d] += 1
        out[(lat(lo), lat(hi))] = int(enc)
    return out


def encoded_3d(vals):
    a, b, pers, typ, inc, _ = core3().morse_edges(vals)
    assert inc == 0
    enc = np.where(typ == 0, -pers.astype(np.int64) - 1, pers.astype(np.int64))
    return dict(zip(zip(a.tolist(), b.tolist()), enc.tolist()))


def test_3d_released_binaries_goldens():
    cases = sorted({k.split("_")[0] for k in GOLD_3D.files if k.startswith("case")},
                   key=lambda s: int(s[4:]))
    assert len(cases) >= 12
    for c in cases:
        vals = GOLD_3D[f"{c}_vals"].astype(np.int32)
        for t in GOLD_3D["thr"]:
            ref = np.unpackbits(GOLD_3D[f"{c}_thr{t}_mask"])[: vals.size].reshape(vals.shape)
            m, st = core3().morse_graph_batch(vals[None], int(t), 1)
            assert (m[0] == ref).all(), (c, int(t))
            assert st[0, 7] == 0


@pytest.mark.parametrize("seed", range(40))
def test_3d_brute_force_reduction(seed):
    rng = np.random.default_rng(seed)
    X, Y, Z = rng.integers(2, 6, size=3)
    vol = rng.integers(-int(rng.choice([2, 3, 5, 50])), 1, size=(X, Y, Z)).astype(np.int32)
    assert encoded_3d(vol) == brute_edges(vol)


def test_3d_matches_2d_kernel_on_slices():
    rng = np.random.default_rng(1)
    for _ in range(60):
        H, W = rng.integers(2, 40, size=2)
        img = rng.integers(-int(rng.choice([2, 5, 255])), 1, size=(H, W)).astype(np.int32)
        for thr in (0, 3, 51):
            m2, s2 = core2().morse_graph_batch(img[None], thr, 1)
            m3, s3 = core3().morse_graph_batch(img[None, :, :, None], thr, 1)
            assert (m2[0] == m3[0, :, :, 0]).all() and (s2[0] == s3[0, :8]).all()


def test_3d_threads_equal_serial():
    rng = np.random.default_rng(2)
    v = rng.integers(-255, 1, size=(5, 12, 10, 9)).astype(np.int32)
    m1, s1 = core3().morse_graph_batch(v, 51, 1)
    m4, s4 = core3().morse_graph_batch(v, 51, 4)
    assert (m1 == m4).all() and (s1 == s4).all()


def test_3d_loss_is_masked_ce_mean():
    torch.manual_seed(0)
    logits = torch.randn(2, 2, 10, 12, 8, requires_grad=True)
    target = (torch.rand(2, 10, 12, 8) > 0.6).long()
    fn = DMTLoss3D()
    loss = fn(logits, target, with_stats=True)
    mask, _ = fn.morse_mask(logits, target)
    sel = logits.permute(0, 2, 3, 4, 1)[mask]
    ref = torch.nn.functional.cross_entropy(sel, target[mask], reduction="sum") / target.numel()
    assert torch.allclose(loss, ref) and fn.last_stats["n_mask"] == int(mask.sum()) > 0
    loss.backward()
    g = logits.grad.abs().sum(1)
    assert (g[~mask] == 0).all()
