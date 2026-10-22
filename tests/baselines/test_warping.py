"""baselines/warping.py against the released warping code (baselines/warping_reference/).

2D: the critical maps equal the ones `warping_ref.warping_loss` builds on the same padded batch,
and the loss is the released formula. 3D: the C++ distance transform and flip loop reproduce
scipy and the released per-sample loop exactly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from baselines import warping as Wp                                  # noqa: E402
from baselines._core import load_core                                # noqa: E402
from baselines.warping_reference import warping_ref as R             # noqa: E402

FIVES = Path(os.environ.get("nnUNet_preprocessed", "/nonexistent")) / "Dataset035_FIVES"


def vessels(rng, h, w, n=6, steps=200, r=2):
    m = np.zeros((h, w), np.uint8)
    for _ in range(n):
        y, x = rng.integers(0, h), rng.integers(0, w)
        for _ in range(steps):
            m[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)] = 1
            y += rng.integers(-1, 2)
            x += rng.integers(-1, 2)
    return ndimage.binary_dilation(m, iterations=r).astype(np.uint8)


def degrade(rng, gt, flip=0.01, cut=True):
    p = gt.copy()
    p ^= (rng.random(gt.shape) < flip).astype(np.uint8)
    if cut:
        h = gt.shape[0]
        p[h // 2: h // 2 + 3, :] = 0
    return p


def logits_for(rng, preds, C=2):
    """Logits whose argmax is `preds`, never exactly 0."""
    B, H, W = preds.shape
    lg = torch.from_numpy(rng.standard_normal((B, C, H, W)).astype(np.float32)) * 0.3
    lg[:, 1] += torch.from_numpy(preds).float() * 6 - 3
    return lg


def reference_maps(logits, targets):
    """The critical maps `warping_loss` builds on the padded batch, read off its CE input."""
    B, C, H, W = logits.shape
    lp = torch.zeros(B, C, H + 2, W + 2)
    lp[:, 0] = 3.0
    lp[:, :, 1:-1, 1:-1] = logits
    yg = F.pad(targets.float(), (1, 1, 1, 1))[:, None]
    captured = {}
    orig = R.RobustCrossEntropyLoss

    class Spy(orig):
        def forward(self, input, target):
            captured["input"] = input.detach().clone()
            return super().forward(input, target)

    R.RobustCrossEntropyLoss = Spy
    try:
        value = R.warping_loss(lp, yg)
    finally:
        R.RobustCrossEntropyLoss = orig
    maps = (captured["input"] != 0).any(1)[:, 1:-1, 1:-1].numpy().astype(np.uint8)
    return maps, float(value)


# ------------------------------------------------------------------------------ 2D
@pytest.mark.parametrize("seed", range(6))
def test_maps_match_released_function(seed):
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(40, 96)), int(rng.integers(40, 96))
    gts = np.stack([vessels(rng, h, w) for _ in range(2)])
    preds = np.stack([degrade(rng, g, flip=float(rng.choice([0.005, 0.02]))) for g in gts])
    lg = logits_for(rng, preds)
    tg = torch.from_numpy(gts).long()
    ref_maps, _ = reference_maps(lg, tg)
    ours, stats = Wp.HomotopyWarpingLoss2D(num_processes=1).critical_mask(lg, tg)
    assert (ours == ref_maps).all() and ours.sum() > 0
    assert int(stats[:, 2].sum()) == int(ours.sum())


def test_errors_on_the_border_run_and_match():
    rng = np.random.default_rng(7)
    gt = vessels(rng, 64, 64)
    gt[0, :] = 1; gt[:, 0] = 1; gt[-1, :] = 1; gt[:, -1] = 1
    pred = gt.copy()
    pred[0, 10:20] = 0; pred[5:9, 0] = 0; pred[-1, 30:40] = 0
    pred[20:23, :] = 0
    lg = logits_for(rng, pred[None])
    tg = torch.from_numpy(gt[None]).long()
    ref_maps, _ = reference_maps(lg, tg)
    ours, _ = Wp.HomotopyWarpingLoss2D(num_processes=1).critical_mask(lg, tg)
    assert (ours == ref_maps).all()


@pytest.mark.skipif(not (FIVES / "train_001_A_seg.npy").exists(), reason="FIVES not available")
@pytest.mark.parametrize("case,off", [("train_001_A", (700, 900)), ("train_310_G", (1000, 600)),
                                      ("train_460_N", (500, 1200))])
def test_maps_match_released_function_on_fives_crops(case, off):
    seg = np.load(FIVES / f"{case}_seg.npy")[0]
    y, x = off
    gt = seg[y:y + 160, x:x + 160].astype(np.uint8)
    rng = np.random.default_rng(sum(map(ord, case)))
    pred = degrade(rng, gt, flip=0.01)
    lg = logits_for(rng, pred[None])
    tg = torch.from_numpy(gt[None]).long()
    ref_maps, _ = reference_maps(lg, tg)
    ours, _ = Wp.HomotopyWarpingLoss2D(num_processes=1).critical_mask(lg, tg)
    assert (ours == ref_maps).all() and ours.sum() > 0


def test_loss_is_the_released_formula():
    rng = np.random.default_rng(3)
    gts = np.stack([vessels(rng, 64, 72) for _ in range(3)])
    preds = np.stack([degrade(rng, g) for g in gts])
    lg = logits_for(rng, preds)
    tg = torch.from_numpy(gts).long()
    fn = Wp.HomotopyWarpingLoss2D(num_processes=1)
    value = fn(lg, tg, with_stats=True)
    maps, _ = fn.critical_mask(lg, tg)
    m = torch.from_numpy(maps)[:, None].float()
    expected = R.RobustCrossEntropyLoss()(lg * m, tg.float()[:, None] * m) * int(maps.sum())
    assert torch.allclose(value, expected, rtol=1e-6, atol=1e-6)
    st = fn.last_stats
    assert st["n_critical"] == int(maps.sum())
    assert abs(st["loss_grad_part"] + st["loss_const"] - float(value)) < 1e-3 * float(value)
    assert st["loss_const"] > 0


def test_pool_equals_serial():
    rng = np.random.default_rng(11)
    gts = np.stack([vessels(rng, 48, 56) for _ in range(4)])
    preds = np.stack([degrade(rng, g) for g in gts])
    lg = logits_for(rng, preds)
    tg = torch.from_numpy(gts).long()
    serial = Wp.HomotopyWarpingLoss2D(num_processes=1)
    pooled = Wp.HomotopyWarpingLoss2D(num_processes=4)
    try:
        a, b = serial(lg, tg), pooled(lg, tg)
        ma, _ = serial.critical_mask(lg, tg)
        mb, _ = pooled.critical_mask(lg, tg)
    finally:
        pooled.close()
    assert (ma == mb).all() and torch.allclose(a, b)


def test_identical_prediction_costs_nothing():
    rng = np.random.default_rng(5)
    gts = np.stack([vessels(rng, 48, 48)])
    fn = Wp.HomotopyWarpingLoss2D(num_processes=1)
    v = fn(logits_for(rng, gts), torch.from_numpy(gts).long(), with_stats=True)
    assert float(v) == 0.0 and fn.last_stats["n_critical"] == 0


def test_cut_vessel_is_critical_and_gradient_lives_there():
    rng = np.random.default_rng(9)
    gt = np.zeros((40, 60), np.uint8)
    gt[18:22, 5:55] = 1
    pred = gt.copy()
    pred[:, 28:31] = 0
    lg = logits_for(rng, pred[None]).requires_grad_(True)
    tg = torch.from_numpy(gt[None]).long()
    fn = Wp.HomotopyWarpingLoss2D(num_processes=1)
    v = fn(lg, tg)
    maps, _ = fn.critical_mask(lg.detach(), tg)
    assert 0 < maps.sum() <= 3 * 4
    assert maps[:, :, :28].sum() == 0 and maps[:, :, 31:].sum() == 0
    v.backward()
    g = lg.grad[0].abs().sum(0).numpy()
    assert (g[maps[0] == 0] == 0).all() and (g[maps[0] == 1] > 0).all()


# ------------------------------------------------------------------------------ 3D
def released_critical_points_3d(pre, gt):
    """The 3D per-sample body of `warping_ref.warping_loss`, with the released kernels."""
    pre_copy, gt_copy = pre.copy(), gt.copy()
    fp = ((pre_copy - gt_copy) == 1).astype(int)
    fn = ((gt_copy - pre_copy) == 1).astype(int)
    # both distances before the first flip: update_simple_point flips in place
    d_fn, d_fp = ndimage.distance_transform_edt(gt_copy) * fn, ndimage.distance_transform_edt(1 - gt_copy) * fp
    gt_warp = R.update_simple_point(d_fp, R.update_simple_point(d_fn, gt_copy))
    d_fp, d_fn = ndimage.distance_transform_edt(pre_copy) * fp, ndimage.distance_transform_edt(1 - pre_copy) * fn
    pre_warp = R.update_simple_point(d_fn, R.update_simple_point(d_fp, pre_copy))
    return np.logical_or(np.not_equal(pre, gt_warp), np.not_equal(gt, pre_warp)).astype(int)


def tubes(rng, shape, n=6, steps=200, r=1):
    m = np.zeros(shape, np.uint8)
    for _ in range(n):
        z, y, x = [rng.integers(0, s) for s in shape]
        for _ in range(steps):
            m[np.clip(z, 0, shape[0] - 1), np.clip(y, 0, shape[1] - 1), np.clip(x, 0, shape[2] - 1)] = 1
            z += rng.integers(-1, 2); y += rng.integers(-1, 2); x += rng.integers(-1, 2)
    return ndimage.binary_dilation(m, iterations=r).astype(np.uint8)


def degrade3d(rng, gt, flip=0.005):
    p = gt.copy()
    p ^= (rng.random(gt.shape) < flip).astype(np.uint8)
    p[gt.shape[0] // 2: gt.shape[0] // 2 + 2] = 0
    return p


@pytest.mark.parametrize("seed", range(4))
def test_edt3d_equals_scipy(seed):
    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in rng.integers(6, 30, 3))
    img = (rng.random(shape) < 0.7).astype(np.uint8)
    img.flat[0] = 0
    ours = load_core().edt3d(img)
    ref = ndimage.distance_transform_edt(img)
    assert ours.dtype == ref.dtype and np.array_equal(ours, ref)


@pytest.mark.parametrize("seed", range(3))
def test_flip_criterion_equals_released(seed):
    rng = np.random.default_rng(seed)
    for _ in range(600):
        patch = (rng.random((3, 3, 3)) < rng.choice([0.2, 0.5, 0.8])).astype(np.uint8)
        vol = np.pad(patch, 1)
        ref = R.decide_simple_point_3D(vol.copy(), 2, 2, 2)
        ours, _ = load_core().warp3d(vol, np.array([2 * 25 + 2 * 5 + 2], np.int64))
        assert np.array_equal(ref, ours)


def test_far_border_truncation_and_index_zero_skip():
    rng = np.random.default_rng(5)
    vol = (rng.random((5, 5, 5)) < 0.5).astype(np.uint8)
    for (x, y, z) in [(4, 4, 4), (4, 2, 2), (2, 4, 1), (0, 2, 2), (2, 0, 2), (1, 1, 0)]:
        ref = R.decide_simple_point_3D(vol.copy(), x, y, z)
        ours, _ = load_core().warp3d(vol, np.array([x * 25 + y * 5 + z], np.int64))
        assert np.array_equal(ref, ours), (x, y, z)


@pytest.mark.parametrize("seed", range(5))
def test_3d_equals_released_loop(seed):
    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in rng.integers(24, 48, 3))
    gt = np.pad(tubes(rng, shape), 1)
    pred = np.pad(degrade3d(rng, tubes(rng, shape) | gt[1:-1, 1:-1, 1:-1]), 1)
    ref = released_critical_points_3d(pred, gt)
    assert np.array_equal(ref, Wp.critical_points_3d(pred, gt)) and ref.sum() > 0


def test_3d_wrapper_pool_and_formula():
    rng = np.random.default_rng(11)
    shape = (24, 32, 28)
    gts = np.stack([tubes(rng, shape) for _ in range(3)])
    preds = np.stack([degrade3d(rng, g) for g in gts])
    lg = torch.from_numpy(rng.standard_normal((3, 2, *shape)).astype(np.float32)) * 0.3
    lg[:, 1] += torch.from_numpy(preds).float() * 6 - 3
    lg.requires_grad_(True)
    tg = torch.from_numpy(gts).long()
    serial = Wp.HomotopyWarpingLoss3D(num_processes=1)
    pooled = Wp.HomotopyWarpingLoss3D(num_processes=3)
    try:
        a, _ = serial.critical_mask(lg, tg)
        c, _ = pooled.critical_mask(lg, tg)
        va, vc = serial(lg, tg, with_stats=True), pooled(lg, tg)
    finally:
        pooled.close()
    ref = np.stack([released_critical_points_3d(np.pad(p, 1), np.pad(g, 1))[1:-1, 1:-1, 1:-1]
                    for p, g in zip(preds, gts)]).astype(np.uint8)
    assert np.array_equal(a, ref) and np.array_equal(a, c)
    assert torch.allclose(va, vc)
    m = torch.from_numpy(a)[:, None].float()
    expected = R.RobustCrossEntropyLoss()(lg.detach() * m, tg.float()[:, None] * m) * int(a.sum())
    assert torch.allclose(va.detach(), expected, rtol=1e-6, atol=1e-6)
    st = serial.last_stats
    assert st["n_critical"] == int(a.sum()) > 0
    assert abs(st["loss_grad_part"] + st["loss_const"] - float(va.detach())) < 1e-3 * float(va.detach())
    va.backward()
    g = lg.grad.abs().sum(1).numpy()
    assert (g[a == 0] == 0).all() and (g[a == 1] > 0).all()
