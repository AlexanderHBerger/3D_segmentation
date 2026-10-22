"""FIVES: 800 colour fundus images (2048x2048) with retinal vessel labels, 600 train / 200 test,
four pathology classes A (AMD), D (diabetic retinopathy), G (glaucoma), N (normal).

Unzip the archive from figshare (it contains train/ and test/, each with Original/ and
"Ground truth"/), then run from the repository root:

    python -m preprocessing.fives preprocess --raw <unzipped archive>
    python -m preprocessing.fives splits

This writes Dataset035_FIVES (training images) and Dataset035_FIVES_test under
$nnUNet_preprocessed. Images keep all three channels and are z-scored per image and channel
over the field of view. Labels are filled to well-composedness. The two training images with
an empty label (train_447_G, train_448_G) are excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preprocessing.topology import fill_stats

NAME = "Dataset035_FIVES"
CLASSES = {"A": "amd", "D": "diabetic_retinopathy", "G": "glaucoma", "N": "normal"}
SHAPE = (2048, 2048)
FOV_THRESHOLD = 5
N_FOLDS = 5
_NAME_RE = re.compile(r"^(\d+)_([ADGN])\.png$")


@dataclass(frozen=True)
class Case:
    case_id: str          # split prefix is needed: train/1_A.png and test/1_A.png both exist
    split: str
    cls: str
    image: Path
    label: Path


def discover_cases(raw: Path) -> list[Case]:
    cases = []
    for split in ("train", "test"):
        img_dir, lbl_dir = raw / split / "Original", raw / split / "Ground truth"
        imgs = {p.name: p for p in img_dir.glob("*.png")}
        lbls = {p.name: p for p in lbl_dir.glob("*.png")}
        if set(imgs) != set(lbls):
            raise ValueError(f"{split}: images and labels do not pair up")
        for name in sorted(imgs):
            m = _NAME_RE.match(name)
            if m is None:
                raise ValueError(f"unexpected file name {split}/{name}")
            cases.append(Case(f"{split}_{int(m.group(1)):03d}_{m.group(2)}", split,
                              m.group(2), imgs[name], lbls[name]))
    return sorted(cases, key=lambda c: c.case_id)


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "PA") and not (np.array(im)[..., -1] == 255).all():
        raise ValueError(f"{path} has a non-opaque alpha channel")
    arr = np.array(im.convert("RGB"))
    if arr.shape != (*SHAPE, 3):
        raise ValueError(f"{path}: unexpected shape {arr.shape}")
    return arr


def read_label(path: Path) -> np.ndarray:
    arr = read_rgb(path)
    if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])):
        raise ValueError(f"{path}: label channels differ")
    return arr[..., 0] > 127


def fov_mask(image: np.ndarray) -> np.ndarray:
    """Max over channels above a low threshold, largest 8-connected component, holes filled."""
    from scipy.ndimage import binary_fill_holes, label

    lab, n = label(image.max(axis=2) > FOV_THRESHOLD, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        raise ValueError("empty field of view")
    m = lab > 0
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == int(sizes.argmax())
    return binary_fill_holes(m)


def normalize(image: np.ndarray, fov: np.ndarray) -> tuple[np.ndarray, dict]:
    """(3, H, W) float32 z-scored per channel over the FOV; outside the FOV each channel is set
    to its minimum inside the FOV."""
    mean, std = np.empty(3), np.empty(3)
    for c in range(3):
        v = image[..., c][fov].astype(np.float64)
        mean[c], std[c] = v.mean(), v.std()
    if not np.all(std > 1e-6):
        raise ValueError(f"degenerate channel std {std}")
    norm = (image.astype(np.float64) - mean) / std
    norm[~fov] = norm.reshape(-1, 3)[fov.reshape(-1)].min(axis=0)
    data = np.ascontiguousarray(np.moveaxis(norm, -1, 0).astype(np.float32))
    return data, {"mean": mean.tolist(), "std": std.tolist()}


def preprocess_case(case: Case, out_dir: Path) -> dict:
    image = read_rgb(case.image)
    raw = read_label(case.label)
    if not raw.any():
        return {"case_id": case.case_id, "excluded": "empty label"}
    fov = fov_mask(image)
    fr = fill_stats(raw)
    data, stats = normalize(image, fov)
    np.save(out_dir / f"{case.case_id}_data.npy", data)
    props = {"case_id": case.case_id, "pathology": case.cls, "normalization": stats,
             "fill_added": fr.n_added}
    (out_dir / f"{case.case_id}_properties.json").write_text(json.dumps(props, indent=1))
    np.save(out_dir / f"{case.case_id}_seg.npy", fr.filled[np.newaxis].astype(np.int8))
    return props


def _worker(args):
    return preprocess_case(*args)


def dataset_json(name: str, n_cases: int, excluded: list[dict]) -> dict:
    return {"name": name, "n_cases": n_cases, "excluded_cases": excluded,
            "channel_names": {"0": "R", "1": "G", "2": "B"}, "num_input_channels": 3,
            "labels": {"background": 0, "vessel": 1}, "stored_shape": list(SHAPE),
            "target_spacing": [1.0, 1.0], "spacing_units": "pixel"}


def preprocess(raw: Path, out_root: Path, workers: int, cases: list[str] | None = None) -> None:
    todo = discover_cases(raw)
    if cases:
        todo = [c for c in todo if c.case_id in set(cases)]
    for split, name in (("train", NAME), ("test", NAME + "_test")):
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(c, out_dir) for c in todo if c.split == split]
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_worker, jobs, chunksize=2))
        excluded = [r for r in res if "excluded" in r]
        if not cases:
            (out_dir / "dataset.json").write_text(json.dumps(
                dataset_json(name, len(res) - len(excluded), excluded), indent=1))
        print(f"{name}: {len(res) - len(excluded)} written, excluded "
              f"{[r['case_id'] for r in excluded]}")


def case_ids(d: Path) -> list[str]:
    return sorted(p.name[:-len("_seg.npy")] for p in d.glob("*_seg.npy"))


def splits(out_root: Path) -> None:
    """Five folds, stratified by pathology class: within a class, images in id order are
    assigned round robin."""
    root = out_root / NAME
    by_cls = defaultdict(list)
    for cid in case_ids(root):
        by_cls[cid.rsplit("_", 1)[-1]].append(cid)
    fold_of = {cid: i % N_FOLDS for cls in sorted(by_cls) for i, cid in enumerate(sorted(by_cls[cls]))}
    folds = [{"train": sorted(c for c, k in fold_of.items() if k != f),
              "val": sorted(c for c, k in fold_of.items() if k == f)} for f in range(N_FOLDS)]
    (root / "splits_final.json").write_text(json.dumps(folds, indent=2) + "\n")
    (root / "test_cases.json").write_text(json.dumps(
        {"case_ids": case_ids(out_root / (NAME + "_test"))}, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("preprocess", "splits"))
    ap.add_argument("--raw", type=Path, help="unzipped FIVES archive")
    ap.add_argument("--out", type=Path, default=os.environ.get("nnUNet_preprocessed"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cases", nargs="*", help="only these case ids (no dataset.json)")
    args = ap.parse_args()
    if args.out is None:
        ap.error("set --out or $nnUNet_preprocessed")
    if args.step == "preprocess":
        preprocess(args.raw, Path(args.out), args.workers, args.cases)
    else:
        splits(Path(args.out))


if __name__ == "__main__":
    main()
