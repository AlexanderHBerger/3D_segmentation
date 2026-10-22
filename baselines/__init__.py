"""Topological baseline losses used for comparison: homotopy warping, DMT and Topograph.

Each loss takes (B, C, *S) logits and (B, *S) class targets and returns the topological term
alone; the caller adds it to the pixel loss with its weight.
"""
from .dmt import DMTLoss2D, DMTLoss3D
from .topograph import TopographLoss2D
from .warping import HomotopyWarpingLoss2D, HomotopyWarpingLoss3D

__all__ = ["DMTLoss2D", "DMTLoss3D", "HomotopyWarpingLoss2D", "HomotopyWarpingLoss3D",
           "TopographLoss2D"]
