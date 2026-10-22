"""Validation metrics: soft and hard Dice, Betti matching error and calibration."""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

BM_DIMS = ("b0", "b1", "b2")


def prepare_targets(targets, num_classes, ignore_index=-1):
    """(valid mask, targets clamped to the class range)."""
    valid = targets != ignore_index
    return valid, torch.clamp(targets, 0, max(num_classes, 2) - 1).long()


def dice_coefficient(logits, targets, valid, smooth=1e-5):
    """Soft Dice per foreground class, averaged over the batch."""
    n_cls = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=n_cls).movedim(-1, 1).float()
    v = valid.unsqueeze(1).expand_as(probs)
    probs, one_hot = (probs * v).reshape(probs.shape[0], n_cls, -1), (one_hot * v).reshape(probs.shape[0], n_cls, -1)
    inter = (probs * one_hot).sum(2)
    union = probs.sum(2) + one_hot.sum(2)
    return ((2.0 * inter + smooth) / (union + smooth))[:, 1:].mean(0)


def hard_dice_coefficient(pred, targets, valid, smooth=1e-5):
    """Dice of the argmax prediction per foreground class, averaged over the batch."""
    n_cls = max(targets.max().item() + 1, pred.max().item() + 1, 2)
    p = F.one_hot(pred, num_classes=n_cls).movedim(-1, 1).float()
    t = F.one_hot(targets.long(), num_classes=n_cls).movedim(-1, 1).float()
    v = valid.unsqueeze(1).expand_as(p)
    p, t = (p * v).reshape(p.shape[0], n_cls, -1), (t * v).reshape(t.shape[0], n_cls, -1)
    inter = (p * t).sum(2)
    union = p.sum(2) + t.sum(2)
    return ((2.0 * inter + smooth) / (union + smooth))[:, 1:].mean(0)


# --------------------------------------------------------------- Betti matching error

def _count(result, attrs, n_dims):
    counts = [0] * n_dims
    for attr in attrs:
        for d in range(n_dims):
            counts[d] += len(np.asarray(getattr(result, attr)[d]))
    return counts


def betti_matching_error(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Betti matching between two binary masks (2D or 3D), computed on the sparse cubical
    complex, which gives the same matching as the dense one for binary inputs.

    bm_error is the number of features of either mask without a matched partner. Bars
    that are still alive when the background enters (`essential`) count as features.
    """
    from sparse_bm import sparse_matching

    res = sparse_matching(pred.astype(np.float64), gt.astype(np.float64), tau_mask=0.5)[0]
    n = pred.ndim
    matched = _count(res, ("input1_matched_birth_coordinates",
                           "input1_matched_essential_birth_coordinates"), n)
    unm_pred = _count(res, ("input1_unmatched_birth_coordinates",
                            "input1_unmatched_essential_birth_coordinates"), n)
    unm_gt = _count(res, ("input2_unmatched_birth_coordinates",
                          "input2_unmatched_essential_birth_coordinates"), n)
    out = {"bm_error": sum(unm_pred) + sum(unm_gt)}
    for d, name in enumerate(BM_DIMS[:n]):
        out[f"bm_matched_{name}"] = matched[d]
        out[f"bm_unmatched_pred_{name}"] = unm_pred[d]
        out[f"bm_unmatched_gt_{name}"] = unm_gt[d]
    return out


def _bm_one(args):
    return betti_matching_error(*args)


def batch_betti_matching_error(pred, gt, valid, num_workers=None) -> dict:
    """Mean Betti matching error over a batch of label maps (foreground = label > 0)."""
    gt_bin = torch.logical_and(gt > 0, valid).cpu().numpy()
    pred_bin = torch.logical_and(pred > 0, valid).cpu().numpy()
    args = [(pred_bin[b], gt_bin[b]) for b in range(len(gt_bin))]
    if len(args) > 1:
        with ProcessPoolExecutor(max_workers=num_workers or min(len(args), os.cpu_count())) as ex:
            rows = list(ex.map(_bm_one, args))
    else:
        rows = [_bm_one(a) for a in args]
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


# -------------------------------------------------------------------- calibration

def calibration_errors(confidences, accuracies, num_bins=15):
    """(ECE, MCE) over equal-width confidence bins."""
    edges = torch.linspace(0, 1, num_bins + 1, device=confidences.device)
    ece = torch.zeros(1, device=confidences.device)
    mce = torch.zeros(1, device=confidences.device)
    for i in range(num_bins):
        in_bin = (confidences > edges[i]) & (confidences <= edges[i + 1])
        prop = in_bin.float().mean()
        if prop.item() > 0:
            gap = torch.abs(confidences[in_bin].mean() - accuracies[in_bin].mean())
            ece += gap * prop
            mce = torch.max(mce, gap)
    return ece.item(), mce.item()


def calibration_metrics(logits, targets, targets_clamped, valid, num_bins=15, ignore_index=-1):
    """ECE, MCE, NLL and Brier score per sample over the valid voxels, averaged."""
    rows = []
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        n_cls = logits.shape[1]
        for b in range(logits.shape[0]):
            m = valid[b].reshape(-1)
            if m.sum() == 0:
                continue
            p = probs[b].movedim(0, -1).reshape(-1, n_cls)[m]
            t = targets_clamped[b].reshape(-1)[m]
            conf, pred = p.max(dim=1)
            ece, mce = calibration_errors(conf, pred.eq(t).float(), num_bins)
            nll = F.cross_entropy(logits[b:b + 1], targets[b:b + 1], ignore_index=ignore_index).item()
            brier = ((p - F.one_hot(t, n_cls).float()) ** 2).sum().item() / int(m.sum())
            rows.append((ece, mce, nll, brier))
    if not rows:
        return {"ece": 0.0, "mce": 0.0, "nll": 0.0, "brier_score": 0.0}
    return dict(zip(("ece", "mce", "nll", "brier_score"), np.mean(rows, axis=0).tolist()))


# ------------------------------------------------------------------------ aggregation

class MetricsCalculator:
    """Accumulates validation metrics, weighting every sample equally.

    The Betti matching error can be restricted to every `betti_sample_stride`-th sample
    and at most `betti_max_samples` samples, for datasets with many validation patches.
    """

    def __init__(self, ignore_index=-1, compute_betti_matching=False, compute_calibration=False,
                 betti_num_workers=None, betti_max_samples=None, betti_sample_stride=1):
        self.ignore_index = ignore_index
        self.compute_betti_matching = compute_betti_matching
        self.compute_calibration = compute_calibration
        self.betti_num_workers = betti_num_workers
        self.betti_max_samples = betti_max_samples
        self.betti_sample_stride = max(1, int(betti_sample_stride or 1))
        self.reset()

    def reset(self):
        self.dice, self.hard_dice, self.counts = [], [], []
        self.topo, self.topo_counts = [], []
        self.calib, self.calib_counts = [], []
        self._seen = self._scored = 0

    def update(self, logits, targets) -> dict:
        b = logits.shape[0]
        valid, tc = prepare_targets(targets, logits.shape[1], self.ignore_index)
        pred = logits.argmax(dim=1)
        dice = dice_coefficient(logits, tc, valid)
        hard = hard_dice_coefficient(pred, tc, valid)
        self.dice.append(dice)
        self.hard_dice.append(hard)
        self.counts.append(b)
        out = {"dice_mean": dice.mean().item(), "dice_hard": hard.mean().item()}

        if self.compute_betti_matching:
            first, self._seen = self._seen, self._seen + b
            idx = []
            for i in range(b):
                if (first + i) % self.betti_sample_stride:
                    continue
                if self.betti_max_samples is not None and self._scored >= self.betti_max_samples:
                    break
                idx.append(i)
                self._scored += 1
            if idx:
                bm = batch_betti_matching_error(pred[idx], tc[idx], valid[idx], self.betti_num_workers)
                self.topo.append(bm)
                self.topo_counts.append(len(idx))
                out.update(bm)

        if self.compute_calibration:
            cal = calibration_metrics(logits, targets, tc, valid, ignore_index=self.ignore_index)
            self.calib.append(cal)
            self.calib_counts.append(b)
            out.update(cal)
        return out

    def compute(self) -> dict:
        if not self.dice:
            return {}
        w = torch.tensor(self.counts, dtype=self.dice[0].dtype, device=self.dice[0].device)
        w = (w / w.sum()).view(-1, 1)
        out = {"dice_mean": (torch.stack(self.dice) * w).sum(0).mean().item(),
               "dice_hard": (torch.stack(self.hard_dice) * w).sum(0).mean().item()}
        for rows, counts in ((self.topo, self.topo_counts), (self.calib, self.calib_counts)):
            if rows:
                n = sum(counts)
                out.update({k: sum(r[k] * c for r, c in zip(rows, counts)) / n for k in rows[0]})
        return out
