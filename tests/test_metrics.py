import numpy as np
import pytest
import torch

from metrics import MetricsCalculator


def test_perfect_prediction():
    target = (torch.rand(2, 16, 16, 16) > 0.7).long()
    logits = torch.stack([10 - 20 * target, 20 * target - 10], 1).float()
    m = MetricsCalculator(compute_calibration=True)
    m.update(logits, target)
    out = m.compute()
    assert out["dice_hard"] == pytest.approx(1.0) and out["ece"] < 1e-3


def test_betti_matching_error():
    pytest.importorskip("sparse_bm")
    from metrics import betti_matching_error

    gt = np.zeros((20, 20, 20), bool)
    gt[5:15, 5:15, 5:15] = True
    assert betti_matching_error(gt, gt)["bm_error"] == 0
    pred = gt.copy()
    pred[2, 2, 2] = True                    # one extra component
    assert betti_matching_error(pred, gt)["bm_unmatched_pred_b0"] == 1
