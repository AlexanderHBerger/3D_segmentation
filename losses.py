"""Loss functions. Every loss returns `(loss, components)` and is built by
`get_loss_function(config)` from `LOSSES`.

    dice_ce          Dice + cross-entropy
    cldice           Dice + CE, with Dice replaced by (1 - alpha) Dice + alpha clDice
    skeleton_recall  Dice + CE + weight * Skeleton Recall
    sparse_bm        Dice + CE + weight * sparseBM (Betti matching on sparse cubical
                     complexes, https://github.com/AlexanderHBerger/sparse-cubical-filtration)

The arguments of each loss (`training.loss_kwargs`) are those of its constructor.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class DiceLoss(nn.Module):
    """Soft Dice over the foreground classes, ignoring `ignore_index` voxels. With
    cldice_alpha > 0 the loss is (1 - alpha) Dice + alpha clDice (Shit et al., 2021)."""

    def __init__(self, smooth=1e-5, ignore_index=-1, cldice_alpha=0.0):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.cldice_alpha = cldice_alpha
        if cldice_alpha > 0:
            self.soft_skeletonize = SoftSkeletonize(num_iter=10, checkpoint_iters=True)

    def forward(self, predictions, targets):
        valid = targets != self.ignore_index
        n_cls = predictions.shape[1]
        one_hot = torch.zeros_like(predictions)
        one_hot.scatter_(1, torch.clamp(targets, 0, n_cls - 1).unsqueeze(1).long(), 1.0)
        probs = F.softmax(predictions, dim=1)
        valid = valid.unsqueeze(1).expand_as(probs)
        probs = probs * valid
        one_hot = one_hot * valid

        p = probs[:, 1:].contiguous().view(probs.shape[0], n_cls - 1, -1)
        t = one_hot[:, 1:].contiguous().view(one_hot.shape[0], n_cls - 1, -1)
        tp = (p * t).sum(dim=2)
        fp = ((1.0 - t) * p).sum(dim=2)
        fn = (t * (1.0 - p)).sum(dim=2)
        dice = 1.0 - ((2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)).mean()
        if self.cldice_alpha <= 0:
            return dice, {"dice": dice}

        skel_pred = self.soft_skeletonize(probs[:, 1:])
        with torch.no_grad():
            skel_true = self.soft_skeletonize(one_hot[:, 1:])
        tprec = (torch.sum(skel_pred * one_hot[:, 1:]) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * probs[:, 1:]) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)
        scaled_dice = (1.0 - self.cldice_alpha) * dice
        scaled_cl = self.cldice_alpha * cl
        return scaled_dice + scaled_cl, {"dice": scaled_dice, "cldice": scaled_cl}


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=1.0, ignore_index=-1, cldice_alpha=0.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index, cldice_alpha=cldice_alpha)

    def forward(self, predictions, targets, **batch):
        components, total = {}, 0.0
        if self.ce_weight > 0:
            components["ce"] = self.ce_weight * self.ce(predictions, targets.long())
            total = total + components["ce"]
        if self.dice_weight > 0:
            dice, parts = self.dice(predictions, targets)
            components.update({k: self.dice_weight * v for k, v in parts.items()})
            total = total + self.dice_weight * dice
        return total, components


# --------------------------------------------------------------------- Skeleton Recall

def tubed_skeleton_numpy(label: np.ndarray, tube_radius: int = 2) -> np.ndarray:
    """The Skeleton Recall target (Kirchhoff et al., 2024): skeletonise the label,
    dilate it `tube_radius` times and keep the class of each voxel. Computed in the data
    loader workers, as in the reference implementation."""
    from skimage.morphology import dilation, skeletonize

    out = np.zeros_like(label)
    if not (label > 0).any():
        return out
    skel = skeletonize(label > 0).astype(label.dtype)
    for _ in range(tube_radius):
        skel = dilation(skel)
    return skel * label


class SkeletonRecallLoss(nn.Module):
    """1 - soft recall of the prediction on the tubed skeleton of the label."""

    def __init__(self, smooth=1e-5, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, predictions, targets, skel):
        n_cls = predictions.shape[1]
        valid = (targets != self.ignore_index).unsqueeze(1)
        if skel.dim() == predictions.dim():
            skel = skel.squeeze(1)
        tube = torch.zeros_like(predictions)
        tube.scatter_(1, torch.clamp(skel, 0, n_cls - 1).long().unsqueeze(1), 1.0)
        tube = (tube * valid)[:, 1:]
        probs = (F.softmax(predictions, dim=1) * valid)[:, 1:]
        num = (tube * probs).flatten(2).sum(2)
        den = tube.flatten(2).sum(2)
        present = den > 0   # classes absent from a sample would contribute a recall of 1
        if not bool(present.any()):
            return predictions.sum() * 0.0
        return 1.0 - ((num + self.smooth) / (den + self.smooth))[present].mean()


class DiceCESkeletonRecallLoss(DiceCELoss):
    """`tube_radius` is used by the data loader, which computes the skeleton target."""

    def __init__(self, weight=1.0, tube_radius=2, **kw):
        super().__init__(**kw)
        self.skelrecall_weight = weight
        self.skeleton_recall = SkeletonRecallLoss(ignore_index=kw.get("ignore_index", -1))

    def forward(self, predictions, targets, skel=None, **batch):
        total, components = super().forward(predictions, targets)
        if self.skelrecall_weight > 0:
            term = self.skelrecall_weight * self.skeleton_recall(predictions, targets, skel)
            components["skelrecall"] = term
            total = total + term
        return total, components


class SoftSkeletonize(nn.Module):
    """Soft skeleton of clDice. With checkpoint_iters the iterations are recomputed in
    the backward pass, which keeps the memory of a 3D batch manageable."""

    def __init__(self, num_iter=10, checkpoint_iters=False):
        super().__init__()
        self.num_iter = num_iter
        self.checkpoint_iters = checkpoint_iters

    @staticmethod
    def soft_erode(img):
        if img.dim() == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)

    @staticmethod
    def soft_dilate(img):
        if img.dim() == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def _step(self, img, skel):
        img = self.soft_erode(img)
        delta = F.relu(img - self.soft_open(img))
        return img, skel + F.relu(delta - skel * delta)

    def forward(self, img):
        skel = F.relu(img - self.soft_open(img))
        for _ in range(self.num_iter):
            if self.checkpoint_iters and (img.requires_grad or skel.requires_grad):
                img, skel = grad_checkpoint(self._step, img, skel, use_reentrant=False)
            else:
                img, skel = self._step(img, skel)
        return skel


# ------------------------------------------------------------------------- sparse BM

class DiceCESparseBMLoss(DiceCELoss):
    """Dice + CE + weight * sparseBM on the foreground probability. `tau` is the mask
    threshold on 1 - p, `clamp_confident` and the penalties are those of
    sparse_bm.SparseBettiLossBatched.

    The loss runs in three stages so that the training loop can overlap the CPU matching
    with GPU work (`config.training.interleaved`): `stage_forward` (GPU), `stage_masks`
    (builds the sparse complexes, synchronises the forward) and `stage_finish` (matching
    on the CPU, loss assembly). `forward` runs them back to back.
    """

    def __init__(self, weight=0.01, tau=0.8, clamp_confident=0.95,
                 penalty_matched="se", penalty_unmatched="se", **kw):
        super().__init__(**kw)
        from sparse_bm import SparseBettiLossBatched

        self.betti_weight = weight
        self.make_term = lambda n_dims: SparseBettiLossBatched(
            dims=tuple(range(n_dims)), tau_mask=tau, mask_backend="auto", capture_stats=True,
            clamp_confident=clamp_confident, penalty_matched=penalty_matched,
            penalty_unmatched=penalty_unmatched)
        self.term, self._term_dims = None, None

    def forward(self, predictions, targets, **batch):
        state = self.stage_forward(predictions, targets)
        self.stage_masks(state)
        return self.stage_finish(state)

    def stage_forward(self, predictions, targets, **batch):
        loss, components = super().forward(predictions, targets)
        state = {"loss": loss, "components": components, "device": predictions.device}
        if self.betti_weight > 0:
            n_dims = predictions.dim() - 2
            if self._term_dims != n_dims:
                self.term, self._term_dims = self.make_term(n_dims), n_dims
            state["pred_fg"] = F.softmax(predictions, dim=1)[:, 1].contiguous()
            state["target_fg"] = (targets == 1).float()
            state["logits"] = predictions
        return state

    def stage_masks(self, state):
        if "pred_fg" in state:
            state["masks"] = self.term.build_masks(state["pred_fg"], state["target_fg"])
        return state

    def stage_finish(self, state):
        total, components = state["loss"], state["components"]
        if "masks" not in state:
            return total, components
        term = self.betti_weight * self.term.loss_from_masks(
            state["pred_fg"], *state["masks"], logits=state["logits"])
        components["betti_matching"] = term
        stats = self.term.last_stats or {}
        dev = state["device"]
        if "fg_fraction" in stats:
            components["betti_fg_fraction"] = torch.tensor(float(stats["fg_fraction"]), device=dev)
        if "n_pockets" in stats:
            components["betti_n_pockets"] = torch.tensor(float(stats["n_pockets"]), device=dev)
        for d, v in sorted((stats.get("per_dim") or {}).items()):
            components[f"betti_matching_b{int(d)}"] = torch.tensor(self.betti_weight * float(v), device=dev)
        return total + term, components


# ---------------------------------------------------------------------------- registry

class DiceCEClDiceLoss(DiceCELoss):
    def __init__(self, alpha=0.5, **kw):
        super().__init__(cldice_alpha=alpha, **kw)


LOSSES = {
    "dice_ce": DiceCELoss,
    "cldice": DiceCEClDiceLoss,
    "skeleton_recall": DiceCESkeletonRecallLoss,
    "sparse_bm": DiceCESparseBMLoss,
}


def get_loss_function(config) -> nn.Module:
    """LOSSES[training.loss_function](**training.loss_kwargs) on top of Dice + CE."""
    t = config.training
    if t.loss_function not in LOSSES:
        raise ValueError(f"unknown loss {t.loss_function!r}; available: {sorted(LOSSES)}")
    return LOSSES[t.loss_function](dice_weight=t.dice_weight, ce_weight=t.ce_weight,
                                   ignore_index=t.ignore_index, **t.loss_kwargs)
