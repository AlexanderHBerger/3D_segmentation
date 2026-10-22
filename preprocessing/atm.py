"""ATM'26 Track 1 (binary airway segmentation) on the native acquisition grid.

    unzip ATM26-Task-1-Binary-Airway-Segmentation-TrainBatch.zip -d <raw>   # imagesTr/, labelsTr/
    python -m preprocessing.atm splits     --raw <raw>
    python -m preprocessing.atm stats      --raw <raw>
    python -m preprocessing.atm preprocess --raw <raw>

The three volumes with anisotropy above 2 are excluded. The remaining 296 are split into
53 test volumes (Dataset038_ATM_test) and five folds, stratified by voxel spacing. Nothing
is resampled, since resampling changes the topology of the thin airway labels. Labels are
made well composed by an add-only fill, which joins the branches that are only diagonally
connected. Volumes are cropped to the air inside the body (an image-only rule), clipped to
[-1024, 600] HU and z-scored with statistics over the body voxels of the training volumes.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from preprocessing.common import (apply_pad, bbox, fill_in_bbox, output_dir, pad_plan,
                                  save_case, write_json)

DATASET = "Dataset038_ATM"
PATCH_SIZE = (128, 128, 128)
HU_CLIP = (-1024.0, 600.0)
BODY_HU = -300.0
AIR_HU = -400.0
CROP_MARGIN = 4
MAX_ANISOTROPY = 2.0
N_FOLDS = 5
TEST_FRACTION = 0.15
SPACING_ROUND = 0.05


def enumerate_cases(raw: Path) -> dict[str, tuple[Path, Path]]:
    images = {re.match(r"(ATM_\d{3})", p.name).group(1): p
              for p in sorted((raw / "imagesTr").glob("ATM_*.mha"))}
    labels = {re.match(r"(ATM_\d{3})", p.name).group(1): p
              for p in sorted((raw / "labelsTr").glob("ATM_*.mha"))}
    if not images or set(images) != set(labels):
        raise SystemExit(f"images and labels under {raw} do not pair up")
    return {cid: (images[cid], labels[cid]) for cid in sorted(images)}


# --------------------------------------------------------------------- reading

def _ras_axes(direction_lps) -> tuple[list[int], list[int]]:
    """Permutation and signs that bring an ITK image (LPS direction cosines) to RAS. Only
    signed permutations are accepted, so no interpolation is ever needed."""
    d_ras = np.diag([-1.0, -1.0, 1.0]) @ np.asarray(direction_lps, dtype=np.float64).reshape(3, 3)
    axis_of, sign_of = [], []
    for row in d_ras:
        k = int(np.argmax(np.abs(row)))
        if not np.isclose(abs(row[k]), 1.0, atol=1e-4) or np.abs(np.delete(row, k)).max() > 1e-4:
            raise ValueError(f"direction is not a signed permutation:\n{d_ras}")
        axis_of.append(k)
        sign_of.append(1 if row[k] > 0 else -1)
    if sorted(axis_of) != [0, 1, 2]:
        raise ValueError(f"degenerate direction matrix:\n{d_ras}")
    return [axis_of.index(r) for r in range(3)], sign_of


def zooms_of(path: Path) -> tuple[float, float, float]:
    """RAS voxel spacing, from the header only."""
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    perm, _ = _ras_axes(reader.GetDirection())
    spacing = reader.GetSpacing()
    return tuple(float(spacing[perm[r]]) for r in range(3))


def load_mha(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """A MetaImage as an (x, y, z) array in RAS order, and its spacing."""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    data = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))
    perm, sign = _ras_axes(img.GetDirection())
    data = np.transpose(data, perm)
    for r in range(3):
        if sign[r] < 0:
            data = np.flip(data, axis=r)
    spacing = img.GetSpacing()
    return np.ascontiguousarray(data), tuple(float(spacing[perm[r]]) for r in range(3))


def load_label(path: Path) -> np.ndarray:
    data, _ = load_mha(path)
    if set(np.unique(data).tolist()) - {0, 1}:
        raise ValueError(f"{path}: label values outside {{0, 1}}")
    return data.astype(bool)


def body_mask(image: np.ndarray) -> np.ndarray:
    """Largest 26-connected component above -300 HU, holes filled per axial slice (a 3D
    fill would leave the trachea open where it leaves the top of the scan)."""
    import cc3d
    from scipy.ndimage import binary_fill_holes

    solid = np.asarray(image) >= BODY_HU
    if not solid.any():
        return np.ones(solid.shape, dtype=bool)
    lab = cc3d.connected_components(solid.astype(np.uint8), connectivity=26, binary_image=True)
    counts = np.bincount(lab.reshape(-1))
    counts[0] = 0
    solid = lab == int(counts.argmax())
    for k in range(solid.shape[2]):
        solid[:, :, k] = binary_fill_holes(solid[:, :, k])
    return solid


def fill_wellcomposed(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """The trachea can touch the array face, so the fill runs on a copy padded by one
    background voxel. No voxel can be added in that ring."""
    if not mask.any():
        return mask.copy(), 0
    filled, n_added = fill_in_bbox(np.pad(mask, 1), margin=2)
    ring = filled.copy()
    ring[(slice(1, -1),) * 3] = False
    assert not ring.any(), "the fill added voxels outside the volume"
    return np.ascontiguousarray(filled[(slice(1, -1),) * 3]), n_added


# ------------------------------------------------------------------------ steps

def build_splits(zooms: dict[str, tuple]) -> dict:
    """Each spacing stratum is dealt round-robin over one test slot and the folds, with a
    per-stratum offset so small strata do not all land in the same fold. No RNG."""
    anis = {cid: max(z) / min(z) for cid, z in zooms.items()}
    excluded = sorted(c for c, a in anis.items() if a > MAX_ANISOTROPY)

    def stratum(z):
        r = lambda v: round(round(float(v) / SPACING_ROUND) * SPACING_ROUND, 3)  # noqa: E731
        return f"{r(z[0]):.2f}x{r(z[2]):.2f}"

    by_stratum = defaultdict(list)
    for cid, z in zooms.items():
        if cid not in excluded:
            by_stratum[stratum(z)].append(cid)
    test_slots = max(1, round(TEST_FRACTION * (N_FOLDS + 1) / (1 - TEST_FRACTION)))
    wheel = ["test"] * test_slots + [f"fold{f}" for f in range(N_FOLDS)]
    assign = {}
    for si, s in enumerate(sorted(by_stratum)):
        for i, cid in enumerate(sorted(by_stratum[s])):
            assign[cid] = wheel[(i + si) % len(wheel)]
    trainval = sorted(c for c, a in assign.items() if a != "test")
    folds = []
    for f in range(N_FOLDS):
        val = sorted(c for c, a in assign.items() if a == f"fold{f}")
        folds.append({"train": sorted(c for c in trainval if c not in set(val)), "val": val})
    return {"test": sorted(c for c, a in assign.items() if a == "test"), "trainval": trainval,
            "folds": folds, "excluded": excluded}


def splits(raw: Path, out: Path) -> None:
    cases = enumerate_cases(raw)
    zooms = {cid: zooms_of(label) for cid, (_, label) in cases.items()}
    sp = build_splits(zooms)
    used = [zooms[c] for c in sp["test"] + sp["trainval"]]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "splits_final.json", sp["folds"])
    write_json(out / "test_cases.json", {"test": sp["test"], "excluded": sp["excluded"]})
    write_json(out / "dataset.json", {
        "name": DATASET, "channel_names": {"0": "CT"}, "labels": {"background": 0, "airway": 1},
        "target_spacing": None, "spacing_is_per_case": True, "resampled": False,
        "nominal_spacing": [float(np.median([z[d] for z in used])) for d in range(3)],
        "spacing_units": "mm", "patch_size": list(PATCH_SIZE), "n_cases": len(used)})
    print(f"{len(sp['trainval'])} train/val, {len(sp['test'])} test, excluded {sp['excluded']}")


def _body_sums(image_path: Path) -> tuple[float, float, int]:
    image, _ = load_mha(image_path)
    v = np.clip(np.asarray(image[body_mask(image)], dtype=np.float64), *HU_CLIP)
    return float(np.add.reduce(v)), float(np.add.reduce(np.square(v))), int(v.size)


def stats(raw: Path, out: Path, num_workers: int) -> None:
    cases = enumerate_cases(raw)
    train = sorted({c for f in json.loads((out / "splits_final.json").read_text())
                    for c in f["train"] + f["val"]})
    s = ss = 0.0
    n = 0
    with ProcessPoolExecutor(num_workers) as ex:
        for a, b, k in ex.map(_body_sums, [cases[c][0] for c in train]):
            s, ss, n = s + a, ss + b, n + k
    mean = s / n
    norm = {"clip_hu": list(HU_CLIP), "mean": float(mean),
            "std": float(np.sqrt(max(0.0, ss / n - mean * mean))), "n_volumes": len(train)}
    write_json(out / "normalization.json", norm)
    print(norm)


def preprocess_case(args) -> str:
    case_id, image_path, label_path, out, norm = args
    label = load_label(label_path)
    if not label.any():
        raise ValueError(f"{case_id}: empty label")
    image, zooms = load_mha(image_path)
    if image.shape != label.shape:
        raise ValueError(f"{case_id}: image and label shapes differ")
    target, n_added = fill_wellcomposed(label)

    # Crop to the air inside the body, falling back to the whole body if a lumen voxel is
    # brighter than the air threshold.
    body = body_mask(image)
    shape = target.shape
    sl = None
    for region in ((image < AIR_HU) & body, body):
        bb = bbox(region)
        if bb is None:
            continue
        sl = tuple(slice(max(0, lo - CROP_MARGIN), min(shape[d], hi + CROP_MARGIN))
                   for d, (lo, hi) in enumerate(bb))
        if int(target[sl].sum()) == int(target.sum()):
            break
        sl = None
    if sl is None:
        sl = tuple(slice(0, s) for s in shape)
    target = np.ascontiguousarray(target[sl])
    x = np.clip(np.asarray(np.ascontiguousarray(image[sl]), dtype=np.float32), *HU_CLIP)
    data = ((x - np.float32(norm["mean"])) / np.float32(norm["std"])).astype(np.float32)

    # Padding takes the normalised value of -1024 HU (air), not 0 (soft tissue).
    plan = pad_plan(target.shape, PATCH_SIZE)
    data = apply_pad(data, plan, float((HU_CLIP[0] - norm["mean"]) / norm["std"]))[None]
    seg = apply_pad(target, plan, False)[None].astype(np.int8)
    props = {"case_id": case_id, "spacing": list(zooms), "crop_origin": [s.start for s in sl],
             "pad": [list(p) for p in plan], "fill_voxels_added": int(n_added)}
    save_case(out, case_id, data.astype(np.float32), seg, props)
    return case_id


def preprocess(raw: Path, out: Path, num_workers: int, only=None) -> None:
    cases = enumerate_cases(raw)
    norm = json.loads((out / "normalization.json").read_text())
    test = set(json.loads((out / "test_cases.json").read_text())["test"])
    trainval = {c for f in json.loads((out / "splits_final.json").read_text())
                for c in f["train"] + f["val"]}
    test_out = out.with_name(out.name + "_test")
    jobs = [(cid, *cases[cid], test_out if cid in test else out, norm)
            for cid in sorted(test | trainval) if not only or cid in only]
    with ProcessPoolExecutor(num_workers) as ex:
        for cid in ex.map(preprocess_case, jobs):
            print(cid, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("splits", "stats", "preprocess"))
    ap.add_argument("--raw", type=Path, required=True, help="directory with imagesTr/ and labelsTr/")
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
