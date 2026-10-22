import numpy as np
import pytest
import torch

from config import Config
from losses import get_loss_function, tubed_skeleton_numpy


def _batch(shape=(2, 2, 24, 24, 16), seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(shape, generator=g, requires_grad=True)
    target = (torch.rand((shape[0], *shape[2:]), generator=g) > 0.7).long()
    return logits, target


@pytest.mark.parametrize("name,kwargs", [("dice_ce", {}), ("cldice", {"alpha": 0.5})])
def test_losses_are_finite_and_differentiable(name, kwargs):
    c = Config()
    c.training.loss_function, c.training.loss_kwargs = name, kwargs
    logits, target = _batch()
    loss, components = get_loss_function(c)(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert set(components) >= {"ce", "dice"}


def test_skeleton_recall():
    c = Config()
    c.training.loss_function, c.training.loss_kwargs = "skeleton_recall", {"weight": 1.0}
    logits, target = _batch()
    skel = torch.from_numpy(np.stack([tubed_skeleton_numpy(t.numpy()) for t in target]))
    loss, components = get_loss_function(c)(logits, target, skel=skel)
    assert torch.isfinite(loss) and "skelrecall" in components


def test_sparse_bm_matches_the_package():
    sparse_bm = pytest.importorskip("sparse_bm")
    c = Config()
    c.training.loss_function = "sparse_bm"
    c.training.loss_kwargs = {"weight": 0.5, "tau": 0.8}
    logits, target = _batch(shape=(2, 2, 32, 32))
    total, components = get_loss_function(c)(logits, target)
    term = sparse_bm.SparseBettiLossBatched(tau_mask=0.8, clamp_confident=0.95, penalty_matched="se")
    expected = 0.5 * term(torch.softmax(logits, 1)[:, 1], (target == 1).float(), logits=logits)
    assert torch.allclose(components["betti_matching"], expected)
