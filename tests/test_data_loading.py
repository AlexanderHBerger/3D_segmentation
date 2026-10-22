import json

import numpy as np
import torch

from config import Config
from data_loading import create_data_loaders


def _dataset(tmp_path, shape, n=4):
    rng = np.random.default_rng(0)
    for i in range(n):
        seg = (rng.random(shape) > 0.8).astype(np.int8)[None]
        np.save(tmp_path / f"case{i}_data.npy", rng.standard_normal((1, *shape)).astype(np.float32))
        np.save(tmp_path / f"case{i}_seg.npy", seg)
        np.save(tmp_path / f"case{i}_foreground_coords.npy", np.argwhere(seg[0] > 0))
    ids = [f"case{i}" for i in range(n)]
    (tmp_path / "splits_final.json").write_text(json.dumps([{"train": ids[:3], "val": ids[3:]}]))


def _config(tmp_path, patch):
    c = Config()
    c.data.data_path = str(tmp_path)
    c.data.patch_size = patch
    c.training.batch_size = 2
    c.num_workers = 0
    c.pin_memory = False
    if len(patch) == 2:
        c.augmentation.rotation_prob = c.augmentation.scaling_prob = c.augmentation.simulate_low_res_prob = 0.0
        c.augmentation.mirror_axes = (0, 1)
    return c


def _check(tmp_path, shape, patch):
    _dataset(tmp_path, shape)
    train, val = create_data_loaders(_config(tmp_path, patch), 0)
    batch = next(iter(train))
    assert batch["image"].shape == (2, 1, *patch) and batch["label"].shape == (2, 1, *patch)
    first = [b["image"] for b in val]
    second = [b["image"] for b in val]
    assert all(torch.equal(a, b) for a, b in zip(first, second)), "validation patches must be fixed"


def test_3d(tmp_path):
    _check(tmp_path, (40, 36, 30), (32, 32, 24))


def test_2d(tmp_path):
    _check(tmp_path, (48, 40), (32, 32))
