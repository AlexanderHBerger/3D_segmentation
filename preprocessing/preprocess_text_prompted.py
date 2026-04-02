"""
Preprocessing script for text-prompted segmentation dataset.

Preprocesses Dataset018_MetastasisCollectionPrompts by:
1. Reorienting to canonical (RAS) orientation
2. Resampling images to isotropic spacing
3. Cropping to nonzero region (matching VoxTell's crop_to_nonzero with hole-filling)
4. Global ZScore normalization on cropped volume (matching VoxTell's inference pipeline)
5. Resampling and cropping instance labels (_cc.nii.gz) and atlas labels (_atlas.nii.gz)
6. Saving as .npz files with 'data', 'seg', 'seg_cc', 'seg_atlas' keys

The normalization matches VoxTell (Rokuss et al., CVPR 2026) exactly:
crop to nonzero first, then global ZScore on the entire cropped volume.
This ensures compatibility with pretrained VoxTell weights.

Usage:
    python preprocessing/preprocess_text_prompted.py \\
        --raw_data_dir /path/to/Dataset018_MetastasisCollectionPrompts \\
        --output_dir /path/to/preprocessed \\
        --target_spacing 1.0 1.0 1.0 \\
        --num_workers 8
"""

import argparse
import json
import sys
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, binary_fill_holes, gaussian_filter1d
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def resample_nifti(nifti_img, target_spacing, interpolation='linear'):
    """Resample a NIfTI image to target spacing.

    For cubic interpolation (order=3), applies per-axis anti-aliasing
    (Gaussian low-pass filter) before downsampling to prevent aliasing.
    """
    original_spacing = nifti_img.header.get_zooms()[:3]
    data = nifti_img.get_fdata()

    zoom_factors = [
        orig / target
        for orig, target in zip(original_spacing, target_spacing)
    ]

    if interpolation == 'nearest':
        resampled = zoom(data, zoom_factors, order=0)
    else:
        # Anti-alias before downsampling: apply Gaussian blur along axes
        # where zoom_factor < 1 (i.e., we are shrinking that axis)
        for axis, zf in enumerate(zoom_factors):
            if zf < 1.0:
                # sigma = 0.5 / zoom_factor gives half-Nyquist of target grid
                sigma = 0.5 / zf
                data = gaussian_filter1d(data, sigma=sigma, axis=axis)
        resampled = zoom(data, zoom_factors, order=3)

    return resampled


def crop_to_nonzero(data_4d, nonzero_threshold=1e-5):
    """
    Crop a (C, H, W, D) array to the nonzero bounding box with hole-filling.

    Matches VoxTell/nnUNet's crop_to_nonzero: creates a nonzero mask from
    channel 0, fills holes with binary_fill_holes, computes tight bounding box.

    Uses a small threshold instead of exact != 0 to handle floating-point
    artifacts from cubic resampling interpolation.

    Returns:
        cropped_data: (C, H', W', D') cropped array
        bbox: tuple of slices for restoring original space
        nonzero_mask_cropped: (1, H', W', D') bool mask within the crop
    """
    # Build nonzero mask across all channels (matches nnUNet create_nonzero_mask)
    # Use threshold to ignore tiny interpolation artifacts from resampling
    nonzero_mask = np.abs(data_4d[0]) > nonzero_threshold
    for c in range(1, data_4d.shape[0]):
        nonzero_mask |= np.abs(data_4d[c]) > nonzero_threshold
    nonzero_mask = binary_fill_holes(nonzero_mask)

    # Compute bounding box
    coords = np.argwhere(nonzero_mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1  # exclusive upper bound
    bbox = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))

    # Crop
    cropped = data_4d[(slice(None),) + bbox]
    mask_cropped = nonzero_mask[bbox][None]  # (1, H', W', D')

    return cropped, bbox, mask_cropped


def preprocess_case(
    case_id: str,
    raw_dir: Path,
    output_dir: Path,
    target_spacing: tuple,
):
    """Preprocess a single case.

    Pipeline:
    1. Load NIfTI and reorient to canonical (RAS)
    2. Resample to target_spacing
    3. Resample all labels the same way
    4. Crop everything to nonzero bounding box (with hole-filling)
    5. Global ZScore normalize the cropped image
    6. Save as .npz
    """
    # File paths
    image_file = raw_dir / 'imagesTr' / f'{case_id}_0000.nii.gz'
    label_file = raw_dir / 'labelsTr' / f'{case_id}.nii.gz'
    cc_file = raw_dir / 'labelsTr' / f'{case_id}_cc.nii.gz'
    atlas_file = raw_dir / 'labelsTr' / f'{case_id}_atlas.nii.gz'

    if not image_file.exists():
        print(f"WARNING: Image not found for {case_id}, skipping")
        return False

    # --- Step 1-2: Load, reorient, resample image ---
    img_nii = nib.load(str(image_file))
    img_canonical = nib.as_closest_canonical(img_nii)
    image = resample_nifti(img_canonical, target_spacing, interpolation='linear')
    # Clamp to 0: cubic interpolation creates negative undershoot at
    # brain/background boundaries. Raw MRI data is non-negative, so any
    # negative values are pure artifacts. Clamping ensures background
    # stays at 0 before ZScore, making it the volume minimum after.
    image = np.clip(image, 0, None)
    image = image[np.newaxis].astype(np.float32)  # (1, H, W, D)

    # --- Step 2b: Load, reorient, resample all labels ---
    labels = {}
    for suffix, label_path in [('seg', label_file), ('seg_cc', cc_file), ('seg_atlas', atlas_file)]:
        if label_path.exists():
            lbl_nii = nib.load(str(label_path))
            lbl_canonical = nib.as_closest_canonical(lbl_nii)
            lbl = resample_nifti(lbl_canonical, target_spacing, interpolation='nearest')
            # Round before int16 cast: some NIfTI files store labels as
            # scaled uint16 (e.g., 0.999... instead of 1), which would
            # truncate to 0 without rounding.
            labels[suffix] = np.rint(lbl)[np.newaxis].astype(np.int16)  # (1, H, W, D)

    # --- Step 3: Crop to nonzero region ---
    image_cropped, bbox, nonzero_mask = crop_to_nonzero(image)

    # Apply same crop to all labels
    labels_cropped = {}
    for key, lbl in labels.items():
        labels_cropped[key] = lbl[(slice(None),) + bbox]

    # --- Step 4: Global ZScore normalization (VoxTell-compatible) ---
    mean_val = image_cropped.mean()
    std_val = image_cropped.std()
    if std_val > 1e-8:
        image_cropped = (image_cropped - mean_val) / std_val
    else:
        image_cropped = image_cropped - mean_val

    # --- Step 5: Save ---
    save_dict = {'data': image_cropped}
    save_dict.update(labels_cropped)

    # Save bounding box for restoring original space at inference time
    bbox_array = np.array([(s.start, s.stop) for s in bbox], dtype=np.int32)
    save_dict['crop_bbox'] = bbox_array

    # Compute foreground coordinates (relative to cropped volume)
    if 'seg' in labels_cropped:
        fg_mask = labels_cropped['seg'][0] > 0
        if fg_mask.any():
            fg_coords = np.argwhere(fg_mask)
            np.save(output_dir / f'{case_id}_foreground_coords.npy', fg_coords)

    np.savez_compressed(output_dir / f'{case_id}.npz', **save_dict)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Dataset018 for text-prompted segmentation"
    )
    parser.add_argument(
        '--raw_data_dir', type=str, required=True,
        help='Path to raw dataset (Dataset018_MetastasisCollectionPrompts)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--target_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help='Target isotropic spacing'
    )
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of parallel workers'
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_spacing = tuple(args.target_spacing)

    # Copy dataset.json and splits_final.json
    for json_file in ['dataset.json', 'splits_final.json']:
        src = raw_dir / json_file
        dst = output_dir / json_file
        if src.exists() and not dst.exists():
            import shutil
            # Resolve symlinks
            src_resolved = src.resolve()
            shutil.copy2(str(src_resolved), str(dst))
            print(f"Copied {json_file}")

    # Get all case IDs from image files
    image_files = sorted((raw_dir / 'imagesTr').glob('*_0000.nii.gz'))
    case_ids = [f.name.replace('_0000.nii.gz', '') for f in image_files]
    print(f"Found {len(case_ids)} cases to preprocess")

    # Filter already processed
    existing = set(f.stem for f in output_dir.glob('*.npz'))
    remaining = [c for c in case_ids if c not in existing]
    print(f"Skipping {len(case_ids) - len(remaining)} already processed cases")
    print(f"Processing {len(remaining)} remaining cases")

    if not remaining:
        print("All cases already preprocessed!")
        return

    # Process cases
    process_fn = partial(
        preprocess_case,
        raw_dir=raw_dir,
        output_dir=output_dir,
        target_spacing=target_spacing,
    )

    if args.num_workers <= 1:
        results = []
        for case_id in tqdm(remaining, desc="Preprocessing"):
            results.append(process_fn(case_id))
    else:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_fn, remaining),
                total=len(remaining),
                desc="Preprocessing"
            ))

    success = sum(1 for r in results if r)
    print(f"\nPreprocessed {success}/{len(remaining)} cases to {output_dir}")


if __name__ == '__main__':
    main()
