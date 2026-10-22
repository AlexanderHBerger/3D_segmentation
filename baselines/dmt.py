"""DMT loss (Hu, Wang, Fuxin, Samaras, Chen, "Topology-Aware Segmentation Using Discrete Morse
Theory", ICLR 2021) in 2D and 3D, computing what the released pipeline
github.com/HuXiaoling/DMT_loss @ b50bc35, folder morse_DMT/, computes.

The likelihood f of each foreground class is quantised to int(-f * levels) and its Morse graph
M_f is the union of the 1-stable manifolds of all saddles with persistence above
`persistence * levels`. The loss is the cross entropy of the logits on the pixels of M_f
(union over classes), summed and divided by the number of pixels ("mean", the paper's L_bce
after the Hadamard mask) or by |M_f| ("crit_mean").

The released pipeline is four programs glued by files (DIPHA persistence, conversion, graph
reconstruction). csrc/topo_baselines_core.cpp (2D) and csrc/dmt3d_core.cpp (3D) compute the
same edge persistences in DIPHA's filtration order (value, dimension, cell index), so ties on
the 8-bit plateaus resolve identically, and the same path tracing. tests/baselines/test_dmt.py
compares both against output of the released binaries.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._core import load_core

REDUCTIONS = ("mean", "crit_mean")
STATS_2D = ("n_dim0_edges", "n_dim1_edges", "n_critical", "n_critical_dim0", "n_vf_edges",
            "n_trees", "n_mask", "n_inconsistent")
STATS_3D = STATS_2D + ("n_dim2_pairs", "n_additions")


def quantise(likelihood: torch.Tensor, levels: int = 255) -> torch.Tensor:
    """The released int(-f * levels), truncation toward zero."""
    return torch.trunc(-likelihood.float() * float(levels)).to(torch.int32)


class _DMTLoss(nn.Module):
    ndim = None
    core = None
    STATS = None
    size_key = None

    def __init__(self, persistence: float = 0.2, levels: int = 255, reduction: str = "mean",
                 n_threads: int = 8, ignore_index: int = -1):
        super().__init__()
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
        if not 0.0 <= persistence <= 1.0:
            raise ValueError("persistence is a fraction of the likelihood range")
        if int(levels) < 1:
            raise ValueError("levels must be >= 1")
        self.persistence = float(persistence)
        self.levels = int(levels)
        self.threshold = int(round(self.persistence * self.levels))
        self.reduction = reduction
        self.n_threads = int(n_threads)
        self.ignore_index = ignore_index
        self.last_stats: dict = {}

    def morse_mask(self, logits: torch.Tensor, targets: torch.Tensor):
        """(mask (B, *S) bool, stats (B, len(STATS)) int64 summed over the foreground classes)."""
        B, C = logits.shape[:2]
        with torch.no_grad():
            p = torch.softmax(logits.float(), dim=1)
            valid = targets != self.ignore_index
            union = None
            stats = np.zeros((B, len(self.STATS)), np.int64)
            for c in range(1, C):
                vals = np.ascontiguousarray(quantise(p[:, c], self.levels).cpu().numpy())
                m, st = load_core(self.core).morse_graph_batch(vals, self.threshold, self.n_threads)
                m = torch.from_numpy(m).to(logits.device).bool()
                union = m if union is None else (union | m)
                stats += st
            if union is None:
                union = torch.zeros_like(valid)
            return union & valid, stats

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, with_stats: bool = False):
        if targets.dim() == logits.dim() and targets.shape[1] == 1:
            targets = targets[:, 0]
        if logits.dim() != self.ndim + 2 or targets.dim() != self.ndim + 1:
            raise ValueError(f"expects logits (B, C, *S) and targets (B, *S) with {self.ndim} spatial axes")
        mask, stats = self.morse_mask(logits, targets)
        k = int(mask.sum().item())
        n = int((targets != self.ignore_index).sum().item())
        if with_stats:
            self.last_stats = {name: int(stats[:, i].sum()) for i, name in enumerate(self.STATS)}
            self.last_stats.update({self.size_key: n, "n_batch": logits.shape[0],
                                    "threshold": self.threshold})
        if k == 0 or n == 0:
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits.movedim(1, -1)[mask].float(), targets[mask].long(), reduction="sum")
        return ce / n if self.reduction == "mean" else ce / k


class DMTLoss2D(_DMTLoss):
    """DMT loss on (B, C, H, W) logits and (B, H, W) class targets."""
    ndim = 2
    core = "topo_baselines_core"
    STATS = STATS_2D
    size_key = "n_pixels"


class DMTLoss3D(_DMTLoss):
    """DMT loss on (B, C, D, H, W) logits and (B, D, H, W) class targets. The volume axes are
    the released writer's (x, y, z), x = axis 0 fastest in DIPHA's cell index."""
    ndim = 3
    core = "dmt3d_core"
    STATS = STATS_3D
    size_key = "n_voxels"
