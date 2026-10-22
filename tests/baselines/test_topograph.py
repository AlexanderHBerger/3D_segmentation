"""baselines/topograph.py against the reference implementation (topograph_reference/,
AlexanderHBerger/Topograph @ 77ca4c4): identical critical clusters, equal loss values, and the
behaviour the paper describes for regular and critical nodes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from baselines import topograph as T                      # noqa: E402
from topograph_reference import topograph_ref as R        # noqa: E402
from topograph_reference import utils_ref as U            # noqa: E402

FIVES = Path(os.environ.get("nnUNet_preprocessed", "/nonexistent")) / "Dataset035_FIVES"


def ref_groups(pred, gt, eight):
    """Critical clusters as the reference's TopographLoss.forward builds them."""
    bp = torch.from_numpy(pred.astype(np.int64))[None]
    bg = torch.from_numpy(gt.astype(np.int64))[None]
    paired = bp + 2 * bg
    d1, d2 = (-4, 16) if eight else (16, -4)
    paired[paired == 0] = d1
    paired[paired == 3] = d2
    h, v = U.new_compute_diffs(paired)
    dr, dl, sr, sl = U.new_compute_diag_diffs(paired, th=7)
    infos, _ = R._single_sample_class_loss(
        bp[0].numpy(), bg[0].numpy(), h[0].numpy(), v[0].numpy(), dr[0].numpy(),
        dl[0].numpy(), sr[0].numpy(), sl[0].numpy(), 0, use_c=False)
    return infos


def as_set(groups):
    return {frozenset(zip(r.tolist(), c.tolist())) for r, c in groups}


def blobs(rng, h, w, n_blobs=6, r=(2, 7)):
    yy, xx = np.mgrid[:h, :w]
    m = np.zeros((h, w), bool)
    for _ in range(n_blobs):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        rad = rng.integers(*r)
        m |= (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
    return m


def degrade(rng, gt, flip=0.03):
    """Speckle plus one quadrant shifted by a pixel."""
    p = gt.copy()
    p ^= rng.random(gt.shape) < flip
    h, w = gt.shape
    p[: h // 2, : w // 2] = np.roll(gt[: h // 2, : w // 2], 1, axis=1)
    return p.astype(np.uint8)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("eight", [False, True])
def test_critical_clusters_match_reference(seed, eight):
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(24, 80)), int(rng.integers(24, 80))
    gt = blobs(rng, h, w).astype(np.uint8)
    pred = degrade(rng, gt, flip=float(rng.choice([0.01, 0.05, 0.15])))
    ours = as_set(T.critical_regions(pred, gt, eight))
    assert ours == as_set(ref_groups(pred, gt, eight))
    assert all(pred[r, c] != gt[r, c] for grp in ours for (r, c) in grp)


@pytest.mark.skipif(not (FIVES / "train_001_A_seg.npy").exists(), reason="FIVES not available")
@pytest.mark.parametrize("case,off", [("train_001_A", (700, 900)), ("train_310_G", (1000, 600)),
                                      ("train_460_N", (500, 1200))])
@pytest.mark.parametrize("eight", [False, True])
def test_critical_clusters_match_reference_on_fives_crops(case, off, eight):
    seg = np.load(FIVES / f"{case}_seg.npy")[0]
    y, x = off
    gt = seg[y:y + 192, x:x + 192].astype(np.uint8)
    rng = np.random.default_rng(sum(map(ord, case)))
    pred = degrade(rng, gt, flip=0.02)
    ours = as_set(T.critical_regions(pred, gt, eight))
    assert ours == as_set(ref_groups(pred, gt, eight)) and len(ours) > 0


@pytest.mark.parametrize("agg", ["mean", "rms", "max", "min", "sum"])
@pytest.mark.parametrize("include_bg", [False, True])
@pytest.mark.parametrize("eight", [False, True])
def test_loss_value_matches_reference(agg, include_bg, eight):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    B, H, W = 3, 40, 48
    gts = np.stack([blobs(rng, H, W).astype(np.uint8) for _ in range(B)])
    preds = np.stack([degrade(rng, g, 0.05) for g in gts])
    logits = torch.randn(B, 2, H, W) * 0.3
    logits[:, 1] += torch.from_numpy(preds).float() * 5.0 - 2.5
    assert (logits.argmax(1).numpy() == preds).all()
    target_idx = torch.from_numpy(gts.astype(np.int64))
    target_1h = torch.nn.functional.one_hot(target_idx, 2).permute(0, 3, 1, 2).float()
    ref = R.TopographLoss(softmax=True, num_processes=1, include_background=include_bg,
                          use_c=False, sphere=False, eight_connectivity=eight,
                          aggregation=U.AggregationType(agg))
    ours = T.TopographLoss2D(num_processes=1, aggregation=agg, include_background=include_bg,
                             eight_connectivity=eight)
    lr, lo = ref(logits.clone(), target_1h), ours(logits.clone(), target_idx)
    assert lo.item() > 0
    assert abs(lo.item() - lr.item()) < 1e-5 * max(1.0, abs(lr.item()))


def test_gradient_only_on_critical_pixels():
    rng = np.random.default_rng(3)
    gt = blobs(rng, 40, 40).astype(np.uint8)
    pred = degrade(rng, gt, 0.05)
    logits = torch.zeros(1, 2, 40, 40)
    logits[0, 1] = torch.from_numpy(pred).float() * 2 - 1
    logits.requires_grad_(True)
    T.TopographLoss2D(num_processes=1)(logits, torch.from_numpy(gt.astype(np.int64))[None]).backward()
    touched = (logits.grad[0].abs().sum(0) > 0).numpy()
    crit = np.zeros((40, 40), bool)
    for r, c in T.critical_regions(pred, gt):
        crit[r, c] = True
    assert (touched == crit).all()


def _loss(pred, gt, eight=False):
    logits = torch.zeros(1, 2, *gt.shape)
    logits[0, 1] = torch.from_numpy(pred.astype(np.float32)) * 2 - 1
    fn = T.TopographLoss2D(num_processes=1, eight_connectivity=eight)
    return fn(logits, torch.from_numpy(gt.astype(np.int64))[None]).item()


def test_identical_prediction_and_regular_node_cost_nothing():
    gt = np.zeros((20, 20), np.uint8)
    gt[5:15, 8:12] = 1
    assert _loss(gt, gt) == 0.0
    pred = gt.copy()
    pred[5:15, 12] = 1
    assert _loss(pred, gt) == 0.0 and T.critical_regions(pred, gt) == []


def test_unmatched_island_and_gap_are_critical():
    gt = np.zeros((20, 20), np.uint8)
    gt[5:15, 8:12] = 1
    p_pred = torch.sigmoid(torch.tensor(1.0)).item()
    island = gt.copy()
    island[1:3, 1:3] = 1
    assert len(T.critical_regions(island, gt)) == 1
    assert abs(_loss(island, gt) - p_pred) < 1e-6
    gap = gt.copy()
    gap[10, 8:12] = 0
    assert len(T.critical_regions(gap, gt)) == 1
    assert abs(_loss(gap, gt) - p_pred) < 1e-6


def test_eight_connectivity_decides_diagonal_clusters():
    gt = np.zeros((12, 12), np.uint8)
    pred = gt.copy()
    pred[3, 3] = 1
    pred[4, 4] = 1
    assert len(T.critical_regions(pred, gt, eight_connectivity=False)) == 2
    assert len(T.critical_regions(pred, gt, eight_connectivity=True)) == 1
    assert abs(_loss(pred, gt, eight=False) - 2 * _loss(pred, gt, eight=True)) < 1e-6


def test_ignore_index_pixels_are_never_critical():
    gt = np.zeros((20, 20), np.int64)
    gt[5:15, 8:12] = 1
    tgt = gt.copy()
    tgt[1:4, 1:4] = -1
    logits = torch.zeros(1, 2, 20, 20)
    logits[0, 1] = torch.from_numpy(gt).float() * 4 - 2
    logits[0, 1, 1:4, 1:4] = 3.0
    fn = T.TopographLoss2D(num_processes=1, ignore_index=-1)
    assert fn(logits, torch.from_numpy(tgt)[None]).item() == 0.0
    assert fn(logits, torch.from_numpy(gt)[None]).item() > 0.0


def test_pool_equals_serial():
    rng = np.random.default_rng(7)
    gts = np.stack([blobs(rng, 32, 32).astype(np.uint8) for _ in range(4)])
    preds = np.stack([degrade(rng, g, 0.05) for g in gts])
    logits = torch.randn(4, 2, 32, 32)
    logits[:, 1] += torch.from_numpy(preds).float() * 3
    t = torch.from_numpy(gts.astype(np.int64))
    a = T.TopographLoss2D(num_processes=1)(logits, t).item()
    par = T.TopographLoss2D(num_processes=2)
    b = par(logits, t).item()
    par.close()
    assert abs(a - b) < 1e-6 * max(1.0, abs(a))


def test_threshold_variation_is_a_gaussian_threshold_shift():
    """A pixel at p_fg = 0.55 in background is an FP island iff the shifted threshold lies
    below 0.55, i.e. with probability Phi(0.05 / sigma)."""
    from scipy.stats import norm
    logits = torch.zeros(1, 2, 12, 12)
    logits[0, 1] = -3.0
    logits[0, 1, 5, 5] = float(np.log(0.55 / 0.45))
    tgt = torch.zeros(1, 12, 12, dtype=torch.long)
    assert T.TopographLoss2D(num_processes=1, thres_var=0.0)(logits, tgt).item() > 0
    torch.manual_seed(0)
    jitter = T.TopographLoss2D(num_processes=1, thres_var=0.1)
    hits = sum(jitter(logits, tgt).item() > 0 for _ in range(400)) / 400
    assert abs(hits - norm.cdf(0.05 / 0.1)) < 0.06
