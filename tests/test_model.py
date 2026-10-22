import torch

from config import Config
from model import create_model, plan_strides


def test_isotropic_strides():
    assert plan_strides((128, 128, 128)) == [[1, 1, 1]] + [[2, 2, 2]] * 5


def test_anisotropic_strides_wait_for_the_fine_axes():
    strides = plan_strides((224, 224, 24), spacing=(1.5, 1.5, 10.0))
    assert strides[1] == [2, 2, 1]
    assert all(s[2] == 1 for s in strides[:3])


def test_forward_2d_and_3d():
    for patch in ((64, 64), (32, 32, 32)):
        c = Config()
        c.data.patch_size = patch
        c.model.model_size = "S"
        out = create_model(c)(torch.zeros(1, 1, *patch))
        assert out.shape == (1, 2, *patch)
