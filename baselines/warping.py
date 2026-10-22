"""Homotopy warping loss (Hu, "Structure-Aware Image Segmentation with Homotopy Warping",
NeurIPS 2022), run as released in github.com/HuXiaoling/Warping @ f249fa3 (vendored in
warping_reference/).

Per sample, the label is warped towards the binarised prediction and the prediction towards
the label by flipping simple points in distance-transform order; the critical mask M is where
a warped mask still disagrees with its target, and the loss is
CE(logits * M, target * M).mean() * |M|. In 2D the released per-sample code is called
unchanged. In 3D its distance transform and flip loop run in C++ (csrc/topo_baselines_core.cpp)
with identical output; the flipping order is still numpy's argsort of the same array, so ties
resolve as in the released code.

Kept from the released code: the simple-point test counts both foreground and background
components with full connectivity (cv2 / cc3d get the connectivity as a positional argument
that means something else), the argsort is numpy's unstable default, and the loss carries the
gradient-free constant log(C) of every pixel outside M (`last_stats` separates it).
Added: the samples of a batch run in a process pool, every mask is zero-padded by one pixel
before warping and the map is cropped back (the released 2D code crashes on an error pixel in
row or column 0), labels are binarised as class > 0, and ignored pixels are never critical.
"""
from __future__ import annotations

import math
import multiprocessing as mp
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from ._core import load_core
from .warping_reference import warping_ref as R


def critical_points_2d(pre: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """The 2D per-sample body of `warping_ref.warping_loss`. pre, gt: (H, W) uint8."""
    pre_copy = pre.copy()
    gt_copy = gt.copy()
    false_positive = ((pre_copy - gt_copy) == 1).astype(int)
    false_negative = ((gt_copy - pre_copy) == 1).astype(int)

    false_negative_distance_gt = ndimage.distance_transform_edt(gt_copy) * false_negative
    false_positive_distance_gt = ndimage.distance_transform_edt(1 - gt_copy) * false_positive
    gt_warp = R.update_simple_point(false_negative_distance_gt, gt_copy)
    gt_warp = R.update_simple_point(false_positive_distance_gt, gt_warp)

    false_positive_distance_pre = ndimage.distance_transform_edt(pre_copy) * false_positive
    false_negative_distance_pre = ndimage.distance_transform_edt(1 - pre_copy) * false_negative
    pre_warp = R.update_simple_point(false_positive_distance_pre, pre_copy)
    pre_warp = R.update_simple_point(false_negative_distance_pre, pre_warp)

    return np.logical_or(np.not_equal(pre, gt_warp), np.not_equal(gt, pre_warp)).astype(int)


def _released_order(distance: np.ndarray) -> np.ndarray:
    """The visiting order of `warping_ref.update_simple_point` as flat indices."""
    n = len(np.nonzero(distance)[0])
    return np.argsort(-distance, axis=None)[:n][::-1].astype(np.int64)


def critical_points_3d(pre: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """The 3D per-sample body of `warping_ref.warping_loss` with the C++ kernels.
    pre, gt: (D, H, W) uint8."""
    core = load_core()
    pre_copy = np.ascontiguousarray(pre, dtype=np.uint8)
    gt_copy = np.ascontiguousarray(gt, dtype=np.uint8)
    false_positive = ((pre_copy - gt_copy) == 1).astype(int)
    false_negative = ((gt_copy - pre_copy) == 1).astype(int)

    false_negative_distance = core.edt3d(gt_copy) * false_negative
    false_positive_distance = core.edt3d((1 - gt_copy).astype(np.uint8)) * false_positive
    gt_warp, _ = core.warp3d(gt_copy, _released_order(false_negative_distance))
    gt_warp, _ = core.warp3d(gt_warp, _released_order(false_positive_distance))

    false_positive_distance_pre = core.edt3d(pre_copy) * false_positive
    false_negative_distance_pre = core.edt3d((1 - pre_copy).astype(np.uint8)) * false_negative
    pre_warp, _ = core.warp3d(pre_copy, _released_order(false_positive_distance_pre))
    pre_warp, _ = core.warp3d(pre_warp, _released_order(false_negative_distance_pre))

    return np.logical_or(np.not_equal(pre_copy, gt_warp), np.not_equal(gt_copy, pre_warp)).astype(int)


def _worker(args):
    pre, gt = args
    fn = critical_points_2d if pre.ndim == 2 else critical_points_3d
    crit = fn(pre, gt)[(slice(1, -1),) * pre.ndim].astype(np.uint8)
    n_fp = int(((pre == 1) & (gt == 0)).sum())
    n_fn = int(((gt == 1) & (pre == 0)).sum())
    return crit, n_fp, n_fn, int(crit.sum())


def critical_points_batch(pred: np.ndarray, gt: np.ndarray, pool=None):
    """Critical maps (B, *S) uint8 and per-sample (n_fp, n_fn, n_crit) of binary (B, *S) masks."""
    pred = np.asarray(pred, dtype=np.uint8)
    gt = np.asarray(gt, dtype=np.uint8)
    tasks = [(np.pad(pred[b], 1), np.pad(gt[b], 1)) for b in range(pred.shape[0])]
    out = [_worker(t) for t in tasks] if pool is None else list(pool.imap(_worker, tasks, chunksize=1))
    crit = np.stack([o[0] for o in out])
    stats = np.array([o[1:] for o in out], dtype=np.int64)
    return crit, stats


class _HomotopyWarpingLoss(nn.Module):
    ndim = None
    size_key = None

    def __init__(self, num_processes: int = 8, ignore_index: int = -1):
        super().__init__()
        self.num_processes = int(num_processes)
        self.ignore_index = int(ignore_index)
        self._pool = None
        self.last_stats: Optional[dict] = None
        self.ce_loss = R.RobustCrossEntropyLoss()

    def _get_pool(self):
        if self.num_processes <= 1:
            return None
        if self._pool is None:
            self._pool = mp.get_context("fork").Pool(self.num_processes)
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_pool"] = None
        return d

    def critical_mask(self, logits: torch.Tensor, targets: torch.Tensor):
        """(critical map (B, *S) uint8 numpy, per-sample stats (B, 3))."""
        with torch.no_grad():
            pre = (torch.argmax(torch.softmax(logits.float(), dim=1), dim=1) > 0).to(torch.uint8)
            gt = (targets > 0).to(torch.uint8)
        return critical_points_batch(pre.cpu().numpy(), gt.cpu().numpy(), self._get_pool())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, with_stats: bool = False):
        if logits.dim() != self.ndim + 2 or targets.dim() != self.ndim + 1:
            raise ValueError(f"expects logits (B, C, *S) and targets (B, *S) with {self.ndim} spatial axes")
        B, C = logits.shape[:2]
        crit_np, stats = self.critical_mask(logits, targets)
        valid = targets != self.ignore_index
        gt = (targets > 0).to(torch.float32).unsqueeze(1)
        m = torch.from_numpy(crit_np).to(logits.device).unsqueeze(1)
        m = m * valid.unsqueeze(1).to(m.dtype)
        n_crit = int(m.sum().item())
        y_pred = logits.float()
        loss = self.ce_loss(y_pred * m, gt * m) * n_crit
        if with_stats:
            n = int(targets.numel())
            with torch.no_grad():
                if n_crit > 0:
                    mb = m[:, 0].bool()
                    ce_sum = torch.nn.functional.cross_entropy(
                        y_pred.movedim(1, -1)[mb], targets[mb].long(), reduction="sum").item()
                else:
                    ce_sum = 0.0
            self.last_stats = {
                "n_fp": int(stats[:, 0].sum()), "n_fn": int(stats[:, 1].sum()),
                "n_critical": n_crit, self.size_key: n, "n_batch": B,
                "loss": float(loss.item()), "loss_grad_part": float(ce_sum * n_crit / n),
                "loss_const": float((n - n_crit) / n * math.log(C) * n_crit),
            }
        return loss


class HomotopyWarpingLoss2D(_HomotopyWarpingLoss):
    """Warping loss on (B, C, H, W) logits and (B, H, W) class targets."""
    ndim = 2
    size_key = "n_pixels"


class HomotopyWarpingLoss3D(_HomotopyWarpingLoss):
    """Warping loss on (B, C, D, H, W) logits and (B, D, H, W) class targets."""
    ndim = 3
    size_key = "n_voxels"

    def __init__(self, num_processes: int = 8, ignore_index: int = -1):
        super().__init__(num_processes, ignore_index)
        load_core()
