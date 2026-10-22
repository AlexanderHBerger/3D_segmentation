"""Patch loading from preprocessed data.

A dataset directory contains `splits_final.json` (a list with one {"train": [...],
"val": [...]} entry per fold) and per case `<case>_data.npy` (C, *spatial) float,
`<case>_seg.npy` (1, *spatial) int and optionally `<case>_foreground_coords.npy`
(np.argwhere(seg[0] > 0)). 2D and 3D data are told apart by len(patch_size).

Each loader worker reads its own share of the cases, one volume at a time, and yields
several patches per volume. Validation patches are fixed: their random state is seeded
from the case id.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from torch.utils.data import DataLoader, IterableDataset

import transforms_2d
import transforms_3d
from losses import tubed_skeleton_numpy


def load_splits(data_path, fold):
    splits = json.loads((Path(data_path) / "splits_final.json").read_text())
    if fold >= len(splits):
        raise ValueError(f"fold {fold} not in {data_path}/splits_final.json ({len(splits)} folds)")
    return splits[fold]["train"], splits[fold]["val"]


class PatchDataset(IterableDataset):
    def __init__(self, case_ids, data_path, patch_size, patches_per_volume, is_training,
                 transforms=None, seed=0, foreground_oversample_percent=0.33,
                 oversize_factor=1.25, val_center_mode="foreground", crop_align=1,
                 channel_indices=None, target_spacing=None, cache_volumes=False,
                 skeleton_tube_radius=None):
        super().__init__()
        self.case_ids = list(case_ids)
        self.data_path = Path(data_path)
        self.patch_size = tuple(int(s) for s in patch_size)
        self.ndim = len(self.patch_size)
        self.patches_per_volume = int(patches_per_volume)
        self.is_training = is_training
        self.transforms = transforms
        self.seed = seed
        self.channel_indices = None if channel_indices is None else list(channel_indices)
        self.skeleton_tube_radius = skeleton_tube_radius
        self.cache = {} if cache_volumes else None
        if self.ndim == 3:
            self.affine = np.diag([*target_spacing, 1.0]) if target_spacing is not None else np.eye(4)
        crop = transforms_3d.OversizedCrop if self.ndim == 3 else transforms_2d.Crop2D

        # Training: in 3D (and in 2D with oversize_factor > 1) an oversized patch is sampled
        # before augmentation and its centre is cut out afterwards. Otherwise the patch is
        # sampled after augmentation.
        self.pre_crop = None
        if is_training and (self.ndim == 3 or oversize_factor > 1.0):
            self.pre_crop = crop(tuple(int(s * oversize_factor) for s in self.patch_size),
                                 foreground_oversample_percent, align=crop_align)
            self.final_crop = crop(self.patch_size, 0.0, center_mode="volume", align=crop_align)
        elif is_training:
            self.final_crop = crop(self.patch_size, foreground_oversample_percent, align=crop_align)
        else:
            self.final_crop = crop(self.patch_size, 0.0, center_mode=val_center_mode, align=crop_align)

        self.executor = ThreadPoolExecutor(max_workers=1)
        # Shared memory, so that set_epoch() reaches the (persistent) worker processes.
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()

    def __len__(self):
        return len(self.case_ids) * self.patches_per_volume

    def set_epoch(self, epoch):
        self._epoch[0] = int(epoch)

    def _load(self, case_id):
        if self.cache is not None and case_id in self.cache:
            return self.cache[case_id]
        image = torch.from_numpy(np.load(self.data_path / f"{case_id}_data.npy")).float()
        if self.channel_indices is not None:
            image = image[self.channel_indices]
        label = torch.from_numpy(np.load(self.data_path / f"{case_id}_seg.npy")).long()
        fg = self.data_path / f"{case_id}_foreground_coords.npy"
        out = (image, label, np.load(fg) if fg.exists() else None)
        if self.cache is not None:
            self.cache[case_id] = out
        return out

    def _subject(self, image, label, fg, case_id):
        s = {"image": image, "label": label, "case_id": case_id}
        if fg is not None:
            s["foreground_coords"] = fg
        if self.ndim == 2:
            return s
        s["image"] = tio.ScalarImage(tensor=image, affine=self.affine)
        s["label"] = tio.LabelMap(tensor=label, affine=self.affine)
        return tio.Subject(**s)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id, n_workers = (0, 1) if info is None else (info.id, info.num_workers)
        case_ids = list(self.case_ids[worker_id::n_workers])
        if not case_ids:
            return
        if self.is_training:
            np.random.default_rng(self.seed + worker_id + int(self._epoch[0]) * 1000).shuffle(case_ids)

        volume = self._load(case_ids[0])
        for i, case_id in enumerate(case_ids):
            nxt = self.executor.submit(self._load, case_ids[i + 1]) if i + 1 < len(case_ids) else None
            base = self._subject(*volume, case_id)

            if self.is_training:
                # Vary the number of patches per volume, and offset the first volume per
                # worker, so that the workers do not all load a new volume at once.
                lo, hi = max(1, self.patches_per_volume // 2), self.patches_per_volume + 1
                n = lo + worker_id % (hi - lo) if i == 0 and worker_id > 0 and hi > lo else np.random.randint(lo, hi)
            else:
                h = int.from_bytes(hashlib.md5(case_id.encode()).digest()[:4], "little") % 100_000_000
                torch.manual_seed(self.seed + h)
                np.random.seed(self.seed + h)
                n = self.patches_per_volume

            for patch_idx in range(n):
                if self.pre_crop is not None:
                    subject = self.pre_crop(base)
                elif self.ndim == 3:
                    base["patch_index"] = patch_idx
                    subject = base
                else:
                    subject = dict(base, patch_index=patch_idx)
                if self.transforms is not None and len(self.transforms) > 0:
                    subject = self.transforms(subject)
                subject = self.final_crop(subject)

                image = subject["image"].data if self.ndim == 3 else subject["image"]
                label = (subject["label"].data if self.ndim == 3 else subject["label"]).long()
                batch = {"image": image, "label": label, "case_id": case_id}
                if self.skeleton_tube_radius is not None:
                    tube = tubed_skeleton_numpy(np.ascontiguousarray(label[0].numpy()),
                                                self.skeleton_tube_radius)
                    batch["skel"] = torch.from_numpy(tube).long().unsqueeze(0)
                yield batch

            if nxt is not None:
                volume = nxt.result()


def create_data_loaders(config, fold):
    train_ids, val_ids = load_splits(config.data.data_path, fold)
    val_ids = val_ids[::max(1, config.data.val_subsample_stride)]
    if config.data.max_samples:
        train_ids, val_ids = train_ids[:config.data.max_samples], val_ids[:config.data.max_samples]
    d, t = config.data, config.training
    transforms = (transforms_3d if len(d.patch_size) == 3 else transforms_2d).get_training_transforms(config)
    common = dict(data_path=d.data_path, patch_size=d.patch_size, seed=config.seed,
                  crop_align=d.crop_align, channel_indices=d.channel_indices,
                  target_spacing=d.target_spacing, cache_volumes=d.cache_volumes,
                  skeleton_tube_radius=(t.loss_kwargs.get("tube_radius", 2)
                                        if t.loss_function == "skeleton_recall" else None))
    train = PatchDataset(train_ids, patches_per_volume=t.patches_per_volume, is_training=True,
                         transforms=transforms, oversize_factor=d.oversize_factor,
                         foreground_oversample_percent=t.oversample_foreground_percent, **common)
    val = PatchDataset(val_ids, patches_per_volume=t.patches_per_volume_val, is_training=False,
                       val_center_mode=d.val_crop_center, **common)
    print(f"fold {fold}: {len(train_ids)} training and {len(val_ids)} validation cases", flush=True)
    persistent = config.num_workers > 0
    train_loader = DataLoader(train, batch_size=t.batch_size, num_workers=config.num_workers,
                              pin_memory=config.pin_memory, drop_last=True, persistent_workers=persistent)
    val_loader = DataLoader(val, batch_size=t.batch_size, num_workers=max(1, config.num_workers // 2),
                            pin_memory=config.pin_memory, persistent_workers=persistent)
    return train_loader, val_loader
