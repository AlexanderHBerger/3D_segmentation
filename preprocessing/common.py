"""Helpers shared by the ATM'26, BraTS-METS and MMWHS preprocessing."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from preprocessing.topology import fill_stats


def output_dir(name: str, out_root: Path | None = None) -> Path:
    root = out_root or os.environ.get("nnUNet_preprocessed")
    if root is None:
        raise SystemExit("set nnUNet_preprocessed or pass --out_root")
    return Path(root) / name


# ------------------------------------------------------------------------ IO

def load_nifti_ras(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """A NIfTI volume in RAS+ voxel order (an exact permutation and flip) and its zooms."""
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(path)))
    if "".join(nib.aff2axcodes(img.affine)) != "RAS":
        raise ValueError(f"{path}: canonicalisation did not reach RAS")
    return np.asanyarray(img.dataobj), tuple(float(z) for z in img.header.get_zooms()[:3])


def save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def write_json(path: Path, obj, indent: int = 2) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=indent) + "\n")
    os.replace(tmp, path)


def foreground_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(mask, dtype=bool)).astype(np.int32)


def save_case(out_dir: Path, case_id: str, data: np.ndarray, seg: np.ndarray,
              properties: dict) -> None:
    """data (C, *spatial) float32, seg (1, *spatial) int8. The seg is written last, so a
    case whose seg exists is complete."""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npy(out_dir / f"{case_id}_data.npy", data)
    save_npy(out_dir / f"{case_id}_foreground_coords.npy", foreground_coords(seg[0]))
    write_json(out_dir / f"{case_id}_properties.json", properties)
    save_npy(out_dir / f"{case_id}_seg.npy", seg)


# ----------------------------------------------------------------- resampling

def resample_shape(shape, zooms, target_spacing) -> tuple[int, ...]:
    return tuple(max(1, int(round(n * float(a) / float(b))))
                 for n, a, b in zip(shape, zooms, target_spacing))


def sample_positions(n_out: int, sp_in: float, sp_out: float) -> np.ndarray:
    """Centres of the output voxels in input index coordinates. Output voxel j covers
    [j, j + 1) * sp_out, so this is not scipy.ndimage.zoom's align-corners grid."""
    return (np.arange(n_out, dtype=np.float64) + 0.5) * (float(sp_out) / float(sp_in)) - 0.5


def nearest_indices(n_in: int, sp_in: float, n_out: int, sp_out: float) -> np.ndarray:
    idx = np.floor(sample_positions(n_out, sp_in, sp_out) + 0.5).astype(np.int64)
    return np.clip(idx, 0, n_in - 1)


def resample_label(label: np.ndarray, zooms, target_spacing) -> np.ndarray:
    """Nearest neighbour by index selection: every output voxel copies one input voxel."""
    lab = np.asarray(label)
    out_shape = resample_shape(lab.shape, zooms, target_spacing)
    idx = [nearest_indices(lab.shape[d], zooms[d], out_shape[d], target_spacing[d])
           for d in range(3)]
    return lab[np.ix_(*idx)]


def resample_image(image: np.ndarray, zooms, target_spacing) -> np.ndarray:
    """Separable linear interpolation, edge voxels replicated at the border."""
    img = np.asarray(image, dtype=np.float64)
    out_shape = resample_shape(img.shape, zooms, target_spacing)
    for d in range(3):
        n_in = img.shape[d]
        pos = np.clip(sample_positions(out_shape[d], zooms[d], target_spacing[d]), 0.0, n_in - 1.0)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_in - 1)
        w = (pos - i0).reshape([-1 if k == d else 1 for k in range(3)])
        img = (1.0 - w) * np.take(img, i0, axis=d) + w * np.take(img, i1, axis=d)
    return img


# --------------------------------------------------------- crop, fill and pad

def bbox(mask: np.ndarray) -> tuple[tuple[int, int], ...] | None:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    out = []
    for ax in range(m.ndim):
        idx = np.flatnonzero(m.any(axis=tuple(i for i in range(m.ndim) if i != ax)))
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return tuple(out)


def bbox_slices(mask: np.ndarray, margin: int = 2) -> tuple[slice, ...] | None:
    bb = bbox(mask)
    if bb is None:
        return None
    return tuple(slice(max(0, lo - margin), min(mask.shape[d], hi + margin))
                 for d, (lo, hi) in enumerate(bb))


def fill_in_bbox(mask: np.ndarray, protect: np.ndarray | None = None, margin: int = 2):
    """Well-composedness fill on the bounding box of the mask (the fill never adds a voxel
    outside the bounding box of a configuration's foreground). Returns (filled, n_added)."""
    m = np.asarray(mask, dtype=bool)
    sl = bbox_slices(m, margin=margin)
    if sl is None:
        return m.copy(), 0
    res = fill_stats(m[sl], protect=None if protect is None else np.asarray(protect, dtype=bool)[sl])
    if res.added_coords.size:
        local = res.added_coords
        if (local.min(axis=0) == 0).any() or \
                (local.max(axis=0) == np.array([s.stop - s.start - 1 for s in sl])).any():
            raise AssertionError("the fill wrote on the crop border; increase the margin")
    out = m.copy()
    out[sl] = res.filled
    return out, res.n_added


def pad_plan(shape, min_shape) -> tuple[tuple[int, int], ...]:
    """Symmetric padding so that every axis reaches min_shape. Volumes are padded here,
    with label 0, so that the data loader never pads labels with the ignore index."""
    plan = []
    for n, m in zip(shape, min_shape):
        deficit = max(0, int(m) - int(n))
        plan.append((deficit // 2, deficit - deficit // 2))
    return tuple(plan)


def apply_pad(arr: np.ndarray, plan, pad_value) -> np.ndarray:
    if not any(lo or hi for lo, hi in plan):
        return arr
    return np.pad(arr, plan, mode="constant", constant_values=pad_value)
