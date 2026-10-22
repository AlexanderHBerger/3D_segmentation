"""ACDC 2D: the left-ventricular myocardium of the ACDC cine MRI, slice by slice.

Download the ACDC dataset (2020 release with test labels) from
https://humanheart-project.creatis.insa-lyon.fr/database ("database" folder with training/ and
testing/ patient directories), then run from the repository root:

    python -m preprocessing.acdc2d preprocess --raw <database dir>
    python -m preprocessing.acdc2d splits

This writes Dataset040_ACDC2D (100 training patients) and Dataset040_ACDC2D_test (50 test
patients) under $nnUNet_preprocessed. One case is one acquired short-axis slice of the ED or
ES frame. Each volume is clipped to its [0.5, 99.5] percentiles and z-scored as a whole, then
cropped/padded around the image centre to 224x224 in plane (z is padded to 24 and the padded
slices are dropped). The label is myocardium (raw label 2) filled to 2D well-composedness per
slice. Nothing is resampled.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from preprocessing.topology import fill

NAME = "Dataset040_ACDC2D"
ROI_SHAPE = (224, 224, 24)
MYO = 2                     # raw labels: 0 background, 1 RV cavity, 2 myocardium, 3 LV cavity
CLIP_PERCENTILES = (0.5, 99.5)
GROUPS = ("DCM", "HCM", "MINF", "NOR", "RV")
N_FOLDS = 5


def parse_info(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = (s.strip() for s in line.split(":", 1))
            out[k] = v
    return out


def enumerate_volumes(raw: Path, split: str) -> list[dict]:
    """Every ED and ES frame of `raw/<split>/patientPPP`."""
    vols = []
    for pdir in sorted((raw / split).iterdir()):
        if not re.fullmatch(r"patient\d{3}", pdir.name):
            continue
        info = parse_info(pdir / "Info.cfg")
        for phase in ("ED", "ES"):
            stem = pdir / f"{pdir.name}_frame{int(info[phase]):02d}"
            vols.append({"patient": pdir.name, "phase": phase, "group": info["Group"],
                         "image": Path(f"{stem}.nii.gz"), "label": Path(f"{stem}_gt.nii.gz")})
    return vols


def centred_slices(shape, out_shape) -> tuple[list[slice], list[slice]]:
    """Crop/pad plan centred on the image (not the label)."""
    src, dst = [], []
    for n_in, n_out in zip(shape, out_shape):
        if n_in >= n_out:
            start = (n_in - n_out) // 2
            src.append(slice(start, start + n_out))
            dst.append(slice(0, n_out))
        else:
            start = (n_out - n_in) // 2
            src.append(slice(0, n_in))
            dst.append(slice(start, start + n_in))
    return src, dst


def normalize(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    img = np.clip(img, *np.percentile(img, CLIP_PERCENTILES))
    return ((img - img.mean()) / max(float(img.std()), 1e-8)).astype(np.float32)


def preprocess_volume(vol: dict, out_dir: Path) -> list[str]:
    import nibabel as nib

    image = np.asarray(nib.load(str(vol["image"])).dataobj)
    label = np.rint(np.asarray(nib.load(str(vol["label"])).dataobj)).astype(np.int8)
    if label.shape != image.shape or not (label == MYO).any():
        raise ValueError(f"{vol['image']}: shape mismatch or no myocardium")

    norm = normalize(image)
    src, dst = centred_slices(image.shape, ROI_SHAPE)
    data = np.full(ROI_SHAPE, norm.min(), dtype=np.float32)
    data[tuple(dst)] = norm[tuple(src)]
    raw = np.zeros(ROI_SHAPE, dtype=np.int8)
    raw[tuple(dst)] = label[tuple(src)]
    if (raw == MYO).sum() != (label == MYO).sum():
        raise ValueError(f"{vol['image']}: myocardium outside the {ROI_SHAPE} crop")

    ids = []
    for z in range(dst[2].start, dst[2].stop):
        sid = f"acdc2d_{vol['patient'][7:]}_{vol['phase']}_z{z:02d}"
        filled, _, _ = fill(raw[:, :, z] == MYO)
        np.save(out_dir / f"{sid}_data.npy", np.ascontiguousarray(data[None, :, :, z]))
        np.save(out_dir / f"{sid}_seg.npy", filled.astype(np.int8)[None])
        props = {k: vol[k] for k in ("patient", "phase", "group")}
        (out_dir / f"{sid}_properties.json").write_text(json.dumps(props, indent=1))
        ids.append(sid)
    return ids


def _worker(args):
    return preprocess_volume(*args)


def preprocess(raw: Path, out_root: Path, workers: int, patients: list[str] | None = None) -> None:
    for split, name in (("training", NAME), ("testing", NAME + "_test")):
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        vols = [v for v in enumerate_volumes(raw, split) if not patients or v["patient"] in patients]
        with ProcessPoolExecutor(workers) as ex:
            ids = sorted(i for res in ex.map(_worker, [(v, out_dir) for v in vols]) for i in res)
        if not patients:
            (out_dir / "dataset.json").write_text(json.dumps({
                "name": name, "n_cases": len(ids), "channel_names": {"0": "cine MRI"},
                "labels": {"background": 0, "lv_myocardium": 1}, "stored_shape": list(ROI_SHAPE[:2]),
                "target_spacing": [1.0, 1.0], "spacing_units": "pixel"}, indent=1))
        if split == "testing" and not patients:
            (out_dir / "test_cases.json").write_text(json.dumps({"case_ids": ids}, indent=1))
        print(f"{name}: {len(vols)} volumes, {len(ids)} slices")


def splits(out_root: Path) -> None:
    """Five folds over patients (both frames and all slices of a patient stay together),
    stratified by the pathology group: within a group, patients in id order are assigned
    round robin."""
    root = out_root / NAME
    group, slices = {}, defaultdict(list)
    for p in sorted(root.glob("*_properties.json")):
        props = json.loads(p.read_text())
        group[props["patient"]] = props["group"]
        slices[props["patient"]].append(p.name[:-len("_properties.json")])
    fold_of = {}
    for g in GROUPS:
        for i, patient in enumerate(sorted(p for p, gg in group.items() if gg == g)):
            fold_of[patient] = i % N_FOLDS
    folds = [{"train": sorted(s for p, k in fold_of.items() if k != f for s in slices[p]),
              "val": sorted(s for p, k in fold_of.items() if k == f for s in slices[p])}
             for f in range(N_FOLDS)]
    (root / "splits_final.json").write_text(json.dumps(folds, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("preprocess", "splits"))
    ap.add_argument("--raw", type=Path, help="ACDC directory containing training/ and testing/")
    ap.add_argument("--out", type=Path, default=os.environ.get("nnUNet_preprocessed"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--patients", nargs="*", help="only these patients, e.g. patient001 (no dataset.json)")
    args = ap.parse_args()
    if args.out is None:
        ap.error("set --out or $nnUNet_preprocessed")
    if args.step == "preprocess":
        preprocess(args.raw, Path(args.out), args.workers, args.patients)
    else:
        splits(Path(args.out))


if __name__ == "__main__":
    main()
