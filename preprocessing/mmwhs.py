"""MMWHS CT: binary LV myocardium (label code 205) on a 1.25 mm isotropic grid.

The 20 labelled CT volumes of the MM-WHS 2017 challenge (ct_train.zip); the official test
volumes have no public labels. Four volumes are held out for testing and the remaining
16 form four folds.

    unzip ct_train.zip -d <raw>                     # -> <raw>/ct_train/ct_train_1001_image.nii.gz ...
    python -m preprocessing.mmwhs splits     --raw <raw>
    python -m preprocessing.mmwhs stats      --raw <raw>
    python -m preprocessing.mmwhs preprocess --raw <raw>

Per volume: RAS orientation, intensity clip and z-score with statistics pooled over the
training volumes (on the native grid), resampling to 1.25 mm, removal of annotation specks
below 27 voxels and of all but the largest component, an add-only well-composedness fill
that avoids the other annotated structures where it can, and padding to the patch size.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from preprocessing.common import (apply_pad, fill_in_bbox, load_nifti_ras, output_dir, pad_plan,
                                  resample_image, resample_label, save_case, write_json)

DATASET = "Dataset033_MMWHS_CT"
CODES = (0, 205, 420, 500, 550, 600, 820, 850)
TARGET_CODE = 205                    # LV myocardium
TARGET_SPACING = (1.25, 1.25, 1.25)
PATCH_SIZE = (128, 128, 128)
SPECK_MIN_VOXELS = 27
STRAY_MAX_FRACTION = 0.005
CLIP_PERCENTILES = (0.5, 99.5)
INTENSITY_STRIDE = 4
N_TEST, N_FOLDS = 4, 4


def enumerate_cases(raw: Path) -> list[tuple[str, str, Path, Path]]:
    """(case_id, subject, image, label), sorted by subject."""
    out = []
    for image in sorted((raw / "ct_train").glob("ct_train_*_image.nii.gz")):
        subject = re.fullmatch(r"ct_train_(\d+)_image\.nii\.gz", image.name).group(1)
        out.append((f"mmwhs_ct_{subject}", subject, image,
                    image.with_name(f"ct_train_{subject}_label.nii.gz")))
    if not out:
        raise SystemExit(f"no ct_train_*_image.nii.gz under {raw / 'ct_train'}")
    return out


def assign_splits(subjects) -> dict:
    """Test subjects at a fixed stride through the sorted ids, the rest dealt round-robin
    into the folds."""
    ids = sorted(str(s) for s in subjects)
    stride = len(ids) // N_TEST
    test = [ids[2 + k * stride] for k in range(N_TEST)]
    trainval = [s for s in ids if s not in set(test)]
    folds = [[] for _ in range(N_FOLDS)]
    for i, s in enumerate(trainval):
        folds[i % N_FOLDS].append(s)
    return {"test": test, "trainval": trainval, "fold_val": folds}


def load_case(image_path: Path, label_path: Path):
    lab, zooms = load_nifti_ras(label_path)
    img, izooms = load_nifti_ras(image_path)
    if lab.shape != img.shape or not np.allclose(zooms, izooms, atol=1e-5):
        raise ValueError(f"{image_path}: image and label geometry disagree")
    unknown = sorted(set(np.unique(lab).astype(int).tolist()) - set(CODES))
    if unknown:
        raise ValueError(f"{label_path}: unknown label codes {unknown}")
    return img, lab, zooms


# ------------------------------------------------------------------ label clean-up

def _components(mask: np.ndarray):
    import cc3d

    lab = cc3d.connected_components(mask.astype(np.uint8), connectivity=6, binary_image=True)
    counts = np.bincount(lab.reshape(-1))
    counts[0] = 0
    return lab, counts


def remove_specks(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m.copy()
    lab, counts = _components(m)
    return np.isin(lab, np.flatnonzero(counts >= SPECK_MIN_VOXELS))


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m.copy()
    lab, counts = _components(m)
    keep = int(counts.argmax())
    if (counts.sum() - counts[keep]) / counts.sum() > STRAY_MAX_FRACTION:
        raise ValueError("more than 0.5 % of the target lies outside its largest component")
    return lab == keep


# ------------------------------------------------------------------------ steps

def _intensity_sample(args):
    image_path, label_path = args
    img, lab, _ = load_case(image_path, label_path)
    vals = np.asarray(img, dtype=np.float64)
    s = INTENSITY_STRIDE
    return vals[::s, ::s, ::s].reshape(-1), vals[lab == TARGET_CODE][::s]


def stats(raw: Path, out: Path, num_workers: int) -> None:
    """Clip window and z-score statistics, pooled over the training volumes. The clip is
    only used if it leaves the myocardium intact (< 0.1 % pooled, < 1 % per volume)."""
    cases = enumerate_cases(raw)
    train = set(assign_splits([c[1] for c in cases])["trainval"])
    jobs = [(c[2], c[3]) for c in cases if c[1] in train]
    with ProcessPoolExecutor(num_workers) as ex:
        samples = list(ex.map(_intensity_sample, jobs))
    pooled = np.concatenate([s for s, _ in samples])
    lo, hi = (float(x) for x in np.percentile(pooled, CLIP_PERCENTILES))
    outside = [int(((m < lo) | (m > hi)).sum()) for _, m in samples]
    n_myo = [int(m.size) for _, m in samples]
    pooled_frac = sum(outside) / max(1, sum(n_myo))
    worst = max(o / max(1, n) for o, n in zip(outside, n_myo))
    clip = bool(pooled_frac < 1e-3 and worst < 1e-2)
    values = np.clip(pooled, lo, hi) if clip else pooled
    norm = {"clip": clip, "clip_lo": lo if clip else None, "clip_hi": hi if clip else None,
            "mean": float(values.mean()), "std": float(values.std()), "n_volumes": len(jobs)}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "normalization.json", norm)
    print(norm)


def preprocess_case(args) -> str:
    case_id, image_path, label_path, out, norm = args
    img, lab, zooms = load_case(image_path, label_path)
    x = np.asarray(img, dtype=np.float64)
    if norm["clip"]:
        x = np.clip(x, norm["clip_lo"], norm["clip_hi"])
    x = ((x - norm["mean"]) / max(norm["std"], 1e-8)).astype(np.float32)
    image = resample_image(x, zooms, TARGET_SPACING).astype(np.float32)
    lab = resample_label(lab, zooms, TARGET_SPACING)

    myo = keep_largest_component(remove_specks(lab == TARGET_CODE))
    if not myo.any():
        raise ValueError(f"{case_id}: no myocardium left")
    filled, n_added = fill_in_bbox(myo, protect=(lab != 0) & (lab != TARGET_CODE))

    plan = pad_plan(filled.shape, PATCH_SIZE)
    image = apply_pad(image, plan, float(image.min()))
    seg = apply_pad(filled, plan, False).astype(np.int8)
    props = {"case_id": case_id, "native_spacing": list(zooms), "spacing": list(TARGET_SPACING),
             "pad": [list(p) for p in plan], "fill_voxels_added": int(n_added)}
    save_case(out, case_id, image[None].astype(np.float32), seg[None], props)
    return case_id


def preprocess(raw: Path, out: Path, num_workers: int, only=None) -> None:
    norm = json.loads((out / "normalization.json").read_text())
    cases = enumerate_cases(raw)
    test = set(assign_splits([c[1] for c in cases])["test"])
    jobs = [(cid, img, lab, out / "test" if subj in test else out, norm)
            for cid, subj, img, lab in cases if not only or cid in only]
    with ProcessPoolExecutor(num_workers) as ex:
        for cid in ex.map(preprocess_case, jobs):
            print(cid, flush=True)


def splits(raw: Path, out: Path) -> None:
    cases = enumerate_cases(raw)
    case_of = {subj: cid for cid, subj, _, _ in cases}
    sp = assign_splits(case_of)
    folds = []
    for val_subjects in sp["fold_val"]:
        val = sorted(case_of[s] for s in val_subjects)
        folds.append({"train": sorted(case_of[s] for s in sp["trainval"] if case_of[s] not in val),
                      "val": val})
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "splits_final.json", folds)
    write_json(out / "test_cases.json", {"test": sorted(case_of[s] for s in sp["test"])})
    write_json(out / "dataset.json", {
        "name": DATASET, "channel_names": {"0": "CT"},
        "labels": {"background": 0, "LV_myocardium": 1},
        "target_spacing": list(TARGET_SPACING), "spacing_units": "mm",
        "patch_size": list(PATCH_SIZE), "n_cases": len(cases)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("splits", "stats", "preprocess"))
    ap.add_argument("--raw", type=Path, required=True, help="directory containing ct_train/")
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()
    out = output_dir(DATASET, args.out_root)
    if args.step == "splits":
        splits(args.raw, out)
    elif args.step == "stats":
        stats(args.raw, out, args.num_workers)
    else:
        preprocess(args.raw, out, args.num_workers, args.cases)


if __name__ == "__main__":
    main()
