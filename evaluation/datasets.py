"""Test and validation cases of the datasets, read from $nnUNet_preprocessed.

Every case has a `unit`, the independent observation over which statistics are computed
(e.g. a patient with several scans), and optionally a `group` (e.g. a pathology class).
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Case:
    case: str
    unit: str
    data: Path
    seg: Path
    group: Optional[str] = None


def root() -> Path:
    return Path(os.environ["nnUNet_preprocessed"])


def load(case: Case):
    """(image (C, *spatial) float32, label (*spatial) int64 with -1 = ignore)."""
    image = np.load(case.data).astype(np.float32, copy=False)
    seg = np.load(case.seg)
    if seg.ndim == image.ndim:
        seg = seg[0]
    return image, seg.astype(np.int64, copy=False)


def _case(folder: Path, cid: str, unit=None, group=None) -> Case:
    return Case(cid, unit or cid, folder / f"{cid}_data.npy", folder / f"{cid}_seg.npy", group)


def _all(folder: Path):
    return sorted(p.name[:-len("_seg.npy")] for p in folder.glob("*_seg.npy"))


def _val(folder: Path, fold: int):
    return json.loads((folder / "splits_final.json").read_text())[fold]["val"]


def nisb_test_tiles(case_ids, n=12, grid=(10, 10, 9)):
    """An evenly spread interior sublattice of the 10 x 10 x 9 tiling of a volume."""
    k = max(1, int(np.ceil(n ** (1.0 / 3.0) - 1e-9)))
    axes = [sorted({int(round(p)) for p in np.linspace(0, g - 1, k + 2)[1:-1]}) for g in grid]
    idx = sorted(90 * x + 9 * y + z for x in axes[0] for y in axes[1] for z in axes[2])
    if len(idx) > n:
        idx = [idx[i] for i in sorted(set(np.linspace(0, len(idx) - 1, n).round().astype(int).tolist()))]
    return [case_ids[i] for i in idx]


def atm(split, fold=0):
    if split == "test":
        folder = root() / "Dataset038_ATM_test"
        return [_case(folder, c) for c in _all(folder)]
    folder = root() / "Dataset038_ATM"
    return [_case(folder, c) for c in _val(folder, fold)]


def nisb_b(split, fold=0):
    folder = root() / "Dataset031_NISB_B"
    if split == "test":
        ids = json.loads((folder / "test_cases.json").read_text())["case_ids"]
    else:
        ids = _val(folder, fold)
    ids = sorted(ids, key=lambda c: int(c.rsplit("_", 1)[1]))
    return [_case(folder, c) for c in nisb_test_tiles(ids)]


def bratsmets(split, fold=0):
    folder = root() / "Dataset036_BraTSMET"
    ids = json.loads((folder / "test_cases.json").read_text())["cases"] if split == "test" else _val(folder, fold)
    return [_case(folder, c, unit=c.rsplit("-", 1)[0]) for c in ids]   # unit = patient


def mmwhs_ct(split, fold=0):
    folder = root() / "Dataset033_MMWHS_CT"
    if split == "test":
        ids = json.loads((folder / "test_cases.json").read_text())["test"]
        return [_case(folder / "test", c) for c in ids]
    return [_case(folder, c) for c in _val(folder, fold)]


def fives(split, fold=0):
    if split == "test":
        folder = root() / "Dataset035_FIVES_test"
        ids = _all(folder)
    else:
        folder = root() / "Dataset035_FIVES"
        ids = _val(folder, fold)
    return [_case(folder, c, group=c.rsplit("_", 1)[-1]) for c in ids]   # group = pathology


def acdc2d(split, fold=0):
    if split == "test":
        folder = root() / "Dataset040_ACDC2D_test"
        ids = json.loads((folder / "test_cases.json").read_text())["case_ids"]
    else:
        folder = root() / "Dataset040_ACDC2D"
        ids = _val(folder, fold)
    cases = []
    for c in ids:
        props = json.loads((folder / f"{c}_properties.json").read_text())
        cases.append(_case(folder, c, unit=props["patient"], group=props.get("group")))   # unit = patient
    return cases


DATASETS = {"atm": atm, "nisb_b": nisb_b, "bratsmets": bratsmets, "mmwhs_ct": mmwhs_ct,
            "fives": fives, "acdc2d": acdc2d}
