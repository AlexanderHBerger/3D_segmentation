"""The competitor losses as Dice + CE + weight * term, in the form of losses.LOSSES."""
import os
import sys

from losses import DiceCELoss, DiceCESparseBMLoss


class _DiceCEPlusTerm(DiceCELoss):
    name = None
    stats = ()

    def __init__(self, weight, dice_weight=1.0, ce_weight=1.0, ignore_index=-1):
        super().__init__(dice_weight=dice_weight, ce_weight=ce_weight, ignore_index=ignore_index)
        self.weight = float(weight)

    def term(self, predictions):
        raise NotImplementedError

    def forward(self, predictions, targets, **batch):
        total, components = super().forward(predictions, targets)
        if self.weight > 0:
            term = self.term(predictions)
            value = self.weight * term(predictions, targets, with_stats=True)
            components[self.name] = value
            st = term.last_stats or {}
            for key, scaled in self.stats:
                components[f"{self.name}_{key}"] = (self.weight if scaled else 1.0) * float(st.get(key, 0))
            total = total + value
        return total, components


class DiceCEWarpingLoss(_DiceCEPlusTerm):
    """Homotopy warping (Hu, NeurIPS 2022)."""
    name = "warping"
    stats = (("loss_grad_part", True), ("loss_const", True), ("n_critical", False))

    def __init__(self, weight=0.0, num_processes=8, **kw):
        super().__init__(weight, **kw)
        self.num_processes = num_processes
        self._terms = {}

    def term(self, predictions):
        nd = predictions.dim() - 2
        if nd not in self._terms:
            from baselines.warping import HomotopyWarpingLoss2D, HomotopyWarpingLoss3D
            cls = HomotopyWarpingLoss2D if nd == 2 else HomotopyWarpingLoss3D
            self._terms[nd] = cls(num_processes=self.num_processes, ignore_index=self.ce.ignore_index)
        return self._terms[nd]


class DiceCEDMTLoss(_DiceCEPlusTerm):
    """DMT (Hu et al., ICLR 2021)."""
    name = "dmt"
    stats = (("n_mask", False), ("n_critical", False))

    def __init__(self, weight=0.0, persistence=0.2, levels=255, reduction="mean", num_threads=8, **kw):
        super().__init__(weight, **kw)
        self.kw = dict(persistence=persistence, levels=levels, reduction=reduction, n_threads=num_threads)
        self._terms = {}

    def term(self, predictions):
        nd = predictions.dim() - 2
        if nd not in self._terms:
            from baselines.dmt import DMTLoss2D, DMTLoss3D
            cls = DMTLoss2D if nd == 2 else DMTLoss3D
            self._terms[nd] = cls(ignore_index=self.ce.ignore_index, **self.kw)
        return self._terms[nd]


class DiceCETopographLoss(_DiceCEPlusTerm):
    """Topograph (Lux et al., ICLR 2025), 2D."""
    name = "topograph"

    def __init__(self, weight=0.0, num_processes=8, aggregation="mean", thres_var=0.0, **kw):
        super().__init__(weight, **kw)
        from baselines.topograph import TopographLoss2D
        self._term = TopographLoss2D(num_processes=num_processes, aggregation=aggregation,
                                     thres_var=thres_var, ignore_index=self.ce.ignore_index)

    def term(self, predictions):
        if predictions.dim() != 4:
            raise ValueError("Topograph is implemented in 2D only")
        return self._term


class DiceCEDenseBMLoss(DiceCESparseBMLoss):
    """Betti matching on the dense cubical complex (Stucki et al.), as in the ATM 64^3
    comparison. This uses the dense mode of the unpublished development version of the
    sparse matcher: set SPARSE_PH_DENSE to its `scripts` directory and
    SPARSE_PH_DENSE_BUILD to the directory of its `betti_matching` build."""

    def __init__(self, weight=0.01, tau=0.8, clamp_confident=0.95,
                 penalty_matched="se", penalty_unmatched="se", **kw):
        DiceCELoss.__init__(self, **kw)
        for var in ("SPARSE_PH_DENSE_BUILD", "SPARSE_PH_DENSE"):
            if var not in os.environ:
                raise RuntimeError(f"dense_bm needs {var}, see the docstring")
            sys.path.insert(0, os.environ[var])
        from sparse_betti_loss import SparseBettiLossBatched

        self.betti_weight = weight
        self.make_term = lambda n_dims: SparseBettiLossBatched(
            sparse_mode="dense", dims=tuple(range(n_dims)), tau_mask=tau, dim1_mode="rim",
            mask_backend="auto", capture_stats=True, clamp_confident=clamp_confident,
            penalty_matched=penalty_matched, penalty_unmatched=penalty_unmatched)
        self.term, self._term_dims = None, None


BASELINE_LOSSES = {
    "warping": DiceCEWarpingLoss,
    "dmt": DiceCEDMTLoss,
    "topograph": DiceCETopographLoss,
    "dense_bm": DiceCEDenseBMLoss,
}
