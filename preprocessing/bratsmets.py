"""BraTS-METS 2025: binary tumour core (labels 1 and 3) on a 1 mm isotropic grid.

    python -m preprocessing.bratsmets preprocess --raw <MICCAI-LH-BraTS2025-MET-Challenge-TrainingData>
    python -m preprocessing.bratsmets splits

The target is the tumour core (non-enhancing core and enhancing tumour), so that each
metastasis is one component. Cases with anisotropy above 2 are skipped and the rest are
resampled to 1 mm. Plain nearest-neighbour resampling can erase small lesions, so every
lesion that disappears gets its voxel closest to the lesion centroid back. Labels are then
made well composed by an add-only fill. The four sequences (t1c, t1n, t2f, t2w) are
z-scored per volume over the brain (the non-zero voxels of the skull-stripped image), and
the volume is cropped to the brain.

Cases without tumour core are stored but left out of the splits. 15 % of the patients are
held out for testing (test_cases.json); the rest form five patient-grouped folds.
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from preprocessing.common import (apply_pad, bbox_slices, fill_in_bbox, load_nifti_ras,
                                  nearest_indices, output_dir, pad_plan, resample_image,
                                  resample_label, resample_shape, save_case, write_json)

DATASET = "Dataset036_BraTSMET"
CHANNELS = ("t1c", "t1n", "t2f", "t2w")
TARGET_CODES = (1, 3)
TARGET_SPACING = (1.0, 1.0, 1.0)
MAX_ANISOTROPY = 2.0
BRAIN_MARGIN = 4
PATCH_SIZE = (128, 128, 128)
TEST_FRACTION, N_FOLDS, SPLIT_SEED = 0.15, 5, 42


def patient_of(case_id: str) -> str:
    return re.fullmatch(r"(BraTS-MET-\d+)-\d+", case_id).group(1)


def enumerate_cases(raw: Path) -> list[Path]:
    return [d for d in sorted(p for p in raw.iterdir() if p.is_dir())
            if (d / f"{d.name}-seg.nii.gz").exists()
            and all((d / f"{d.name}-{c}.nii.gz").exists() for c in CHANNELS)]


def zscore_brain(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    vals = img[mask]
    mean, std = float(vals.mean()), float(vals.std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"degenerate brain statistics: mean={mean} std={std}")
    out = np.zeros_like(img)
    out[mask] = (img[mask] - mean) / std
    return out.astype(np.float32)


def resample_lesions(mask: np.ndarray, zooms) -> tuple[np.ndarray, int]:
    """Nearest-neighbour resampling of the lesion map that keeps every lesion (26-connected
    component): a lesion with no voxel left gets back the output voxel of its member voxel
    closest to its centroid. Returns (mask, number of restored lesions)."""
    import cc3d

    out_shape = resample_shape(mask.shape, zooms, TARGET_SPACING)
    if not mask.any():
        return np.zeros(out_shape, dtype=bool), 0
    lab_in = cc3d.connected_components(mask.astype(np.uint8), connectivity=26, binary_image=True)
    idx = [nearest_indices(mask.shape[d], zooms[d], out_shape[d], TARGET_SPACING[d])
           for d in range(3)]
    lab_out = lab_in[np.ix_(*idx)]
    out = lab_out > 0
    survived = set(int(v) for v in np.unique(lab_out) if v)
    lost = [c for c in range(1, int(lab_in.max()) + 1) if c not in survived]
    for c in lost:
        pts = np.argwhere(lab_in == c)
        anchor = pts[np.argmin(((pts - pts.mean(axis=0)) ** 2).sum(axis=1))]
        out[tuple(int(np.clip(np.floor((anchor[d] + 0.5) * (zooms[d] / TARGET_SPACING[d])),
                              0, out_shape[d] - 1)) for d in range(3))] = True
    return out, len(lost)


def preprocess_case(args) -> str:
    case_dir, out = args
    cid = case_dir.name
    seg_raw, zooms = load_nifti_ras(case_dir / f"{cid}-seg.nii.gz")
    if max(zooms) / min(zooms) > MAX_ANISOTROPY:
        return f"{cid} skipped (anisotropy)"
    resample = tuple(round(z, 6) for z in zooms) != TARGET_SPACING

    # Each channel is normalised over its own non-zero support; the brain mask used for the
    # crop is the union over channels, because sequences can differ in field of view.
    chans = []
    brain = np.zeros(seg_raw.shape, dtype=bool)
    for name in CHANNELS:
        data, _ = load_nifti_ras(case_dir / f"{cid}-{name}.nii.gz")
        if data.shape != seg_raw.shape:
            raise ValueError(f"{cid}: {name} shape {data.shape} != seg shape {seg_raw.shape}")
        m = np.asarray(data) != 0
        brain |= m
        arr = zscore_brain(data, m)
        if resample:
            arr = resample_image(arr, zooms, TARGET_SPACING).astype(np.float32)
        chans.append(arr)

    if not brain.any():
        raise ValueError(f"{cid}: empty brain mask")
    codes = np.rint(np.asarray(seg_raw)).astype(np.int8)
    target = np.isin(codes, TARGET_CODES)
    n_restored = 0
    if resample:
        brain = resample_label(brain.astype(np.uint8), zooms, TARGET_SPACING).astype(bool)
        target, n_restored = resample_lesions(target, zooms)
    target, n_added = fill_in_bbox(target)

    sl = bbox_slices(brain, BRAIN_MARGIN)
    if int(target[sl].sum()) != int(target.sum()):
        raise ValueError(f"{cid}: lesion voxels outside the brain mask")
    target = target[sl]
    plan = pad_plan(target.shape, PATCH_SIZE)
    image = np.stack([apply_pad(c[sl], plan, 0.0) for c in chans], axis=0).astype(np.float32)
    seg = apply_pad(target, plan, False)[None].astype(np.int8)
    props = {"case_id": cid, "native_spacing": list(zooms), "resampled": resample,
             "crop_origin": [s.start for s in sl], "pad": [list(p) for p in plan],
             "lesions_restored": n_restored, "fill_voxels_added": int(n_added)}
    save_case(out, cid, image, seg, props)
    return cid


def preprocess(raw: Path, out: Path, num_workers: int, only=None) -> None:
    jobs = [(d, out) for d in enumerate_cases(raw) if not only or d.name in only]
    with ProcessPoolExecutor(num_workers) as ex:
        for msg in ex.map(preprocess_case, jobs, chunksize=1):
            print(msg, flush=True)


def splits(out: Path) -> None:
    """Patient-grouped: a seeded permutation of the patients, the first 15 % held out and
    the rest dealt round-robin into the folds. Only cases with tumour core are used."""
    cases = sorted(p.name[: -len("_seg.npy")] for p in out.glob("*_seg.npy"))
    by_patient: dict[str, list[str]] = {}
    for cid in cases:
        if np.load(out / f"{cid}_foreground_coords.npy", mmap_mode="r").shape[0] > 0:
            by_patient.setdefault(patient_of(cid), []).append(cid)
    patients = sorted(by_patient)
    ordered = [patients[i] for i in np.random.default_rng(SPLIT_SEED).permutation(len(patients))]
    n_test = max(1, int(round(TEST_FRACTION * len(ordered))))
    test, trainval = sorted(ordered[:n_test]), ordered[n_test:]
    folds = []
    for f in range(N_FOLDS):
        val_p = set(trainval[f::N_FOLDS])
        folds.append({"train": sorted(c for p in trainval if p not in val_p for c in by_patient[p]),
                      "val": sorted(c for p in val_p for c in by_patient[p])})
    write_json(out / "splits_final.json", folds, indent=1)
    write_json(out / "test_cases.json", {
        "unit": "patient", "patients": test,
        "cases": sorted(c for p in test for c in by_patient[p])}, indent=1)
    write_json(out / "dataset.json", {
        "name": DATASET, "channel_names": dict(enumerate(CHANNELS)),
        "labels": {"background": 0, "tumour_core": 1}, "target_codes": list(TARGET_CODES),
        "target_spacing": list(TARGET_SPACING), "spacing_units": "mm",
        "patch_size": list(PATCH_SIZE), "n_cases": len(cases)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("preprocess", "splits"))
    ap.add_argument("--raw", type=Path, help="directory with one folder per case")
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()
    out = output_dir(DATASET, args.out_root)
    if args.step == "preprocess":
        preprocess(args.raw, out, args.num_workers, args.cases)
    else:
        splits(out)


if __name__ == "__main__":
    main()
