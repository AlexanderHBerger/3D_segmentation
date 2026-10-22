"""Per-case test metrics on the binary foreground (label > 0, ignored voxels count as
background in both masks). Connectivity is that of the cubical complex the Betti matching
uses: foreground 6-connected and background 26-connected in 3D (4 and 8 in 2D).

    dice_hard   Dice
    bm_error    Betti matching error, and per dimension (metrics.betti_matching_error)
    cldice      clDice with hard skeletons (Shit et al., 2021)
    voi_fg      variation of information of the foreground components
    voi_bg      the same for the background components (for boundary segmentation)
    nsd         normalised surface Dice with a tolerance of one voxel
"""
import numpy as np

from metrics import betti_matching_error


def binarize(mask, label):
    valid = label != -1
    return (mask > 0) & valid, (label > 0) & valid


def dice(pred, gt, smooth=1e-5):
    return float((2.0 * np.logical_and(pred, gt).sum() + smooth) / (pred.sum() + gt.sum() + smooth))


def cldice(pred, gt):
    from skimage.morphology import skeletonize

    if not gt.any() or not pred.any():
        return float("nan")
    sp, sg = skeletonize(pred), skeletonize(gt)
    if not sp.any() or not sg.any():
        return float("nan")
    prec = float((sp & gt).sum()) / float(sp.sum())
    sens = float((sg & pred).sum()) / float(sg.sum())
    return 0.0 if prec + sens <= 0 else 2 * prec * sens / (prec + sens)


def voi(pred, gt, foreground=True):
    """H(pred | gt) + H(gt | pred) of the connected components of the foreground (or of
    the background)."""
    from scipy.ndimage import label
    from skimage.metrics import variation_of_information

    p, g = (pred, gt) if foreground else (~pred, ~gt)
    structure = None if foreground else np.ones((3,) * pred.ndim)
    p_lab, n_p = label(p, structure=structure)
    g_lab, n_g = label(g, structure=structure)
    if n_p + (not p.all()) < 2 or n_g + (not g.all()) < 2:
        return float("nan")
    split, merge = variation_of_information(g_lab.astype(np.int32), p_lab.astype(np.int32))
    return float(split + merge)


def nsd(pred, gt, tolerance=1.0):
    import torch
    from monai.metrics import compute_surface_dice

    if not gt.any() or not pred.any():
        return float("nan")
    p = torch.from_numpy(np.stack([~pred, pred]).astype(np.uint8))[None]
    g = torch.from_numpy(np.stack([~gt, gt]).astype(np.uint8))[None]
    return float(compute_surface_dice(p, g, class_thresholds=[tolerance], include_background=False).flatten()[0])


def case_metrics(mask, label) -> dict:
    pred, gt = binarize(np.asarray(mask), np.asarray(label))
    out = {"dice_hard": dice(pred, gt), "gt_empty": not gt.any()}
    out.update(betti_matching_error(pred, gt))
    out.update(cldice=cldice(pred, gt), voi_fg=voi(pred, gt, True), voi_bg=voi(pred, gt, False),
               nsd=nsd(pred, gt))
    return out
