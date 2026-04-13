"""Precompute distance fields from brain region atlas for spatial prior loss.

For each preprocessed .npz file that contains seg_atlas, compute one distance
transform per atlas region (that contains at least one lesion) and save as
{case_id}_distance_fields.npz.

The distance field for a region is: sigmoid((EDT(~region) - 3*sigma) / sigma)
- 0 inside the region
- ~1 far from the region
- smooth transition controlled by sigma

Usage:
    python precompute_distance_fields.py \
        --data_dir /path/to/preprocessed \
        --sigma 20.0 --num_workers 8
"""
import argparse
import glob
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


def compute_distance_fields_for_case(npz_path: str, sigma: float) -> dict:
    """Compute normalized distance fields for atlas regions containing lesions.

    Args:
        npz_path: Path to preprocessed .npz file.
        sigma: Controls sigmoid falloff (in voxels).

    Returns:
        Dict mapping str(atlas_label) -> float16 distance field array,
        or None if no atlas/seg_cc available.
    """
    data = np.load(npz_path)

    if 'seg_atlas' not in data or 'seg_cc' not in data:
        return None

    atlas = data['seg_atlas'][0]  # (H, W, D), int16
    seg_cc = data['seg_cc'][0]    # (H, W, D), int16

    # Find atlas regions that contain at least one lesion voxel
    lesion_mask = seg_cc > 0
    if not lesion_mask.any():
        return None

    atlas_at_lesions = atlas[lesion_mask]
    # Exclude background (0) and CSF-like regions
    unique_regions = set(atlas_at_lesions[atlas_at_lesions > 0].tolist())

    if not unique_regions:
        return None

    fields = {}
    for region_label in unique_regions:
        region_mask = atlas == region_label
        if not region_mask.any():
            continue

        # EDT from outside the region
        distance = distance_transform_edt(~region_mask).astype(np.float32)

        # Normalize: sigmoid -> 0 inside, ~1 far away
        normalized = 1.0 / (1.0 + np.exp(-(distance - 3 * sigma) / sigma))

        # Store as float16 to save space
        fields[str(region_label)] = normalized.astype(np.float16)

    return fields


def process_case(args):
    """Worker function for parallel processing."""
    npz_path, sigma, output_dir = args
    case_id = os.path.splitext(os.path.basename(npz_path))[0]
    output_path = os.path.join(output_dir, f"{case_id}_distance_fields.npz")

    # Skip if already computed
    if os.path.exists(output_path):
        return case_id, 'skipped'

    try:
        fields = compute_distance_fields_for_case(npz_path, sigma)
    except Exception as e:
        return case_id, f'error: {e}'

    if fields is None or len(fields) == 0:
        return case_id, 'no_regions'

    np.savez_compressed(output_path, **fields)
    return case_id, f'{len(fields)} regions'


def main():
    parser = argparse.ArgumentParser(description='Precompute distance fields from brain atlas')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with preprocessed .npz files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as data_dir)')
    parser.add_argument('--sigma', type=float, default=20.0,
                        help='Sigmoid falloff in voxels (default: 20.0)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir

    npz_files = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
    # Filter out existing distance field files
    npz_files = [f for f in npz_files if not f.endswith('_distance_fields.npz')]
    print(f"Found {len(npz_files)} preprocessed cases")
    print(f"Sigma: {args.sigma}, Workers: {args.num_workers}")
    print(f"Output: {output_dir}")

    work_items = [(f, args.sigma, output_dir) for f in npz_files]

    stats = {'computed': 0, 'skipped': 0, 'no_regions': 0, 'errors': 0}

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_case, item): item[0] for item in work_items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing distance fields"):
            case_id, result = future.result()
            if result == 'skipped':
                stats['skipped'] += 1
            elif result == 'no_regions':
                stats['no_regions'] += 1
            elif result.startswith('error'):
                stats['errors'] += 1
                print(f"  Error: {case_id}: {result}")
            else:
                stats['computed'] += 1

    print(f"\nDone: {stats['computed']} computed, {stats['skipped']} skipped, "
          f"{stats['no_regions']} no regions, {stats['errors']} errors")


if __name__ == '__main__':
    main()
