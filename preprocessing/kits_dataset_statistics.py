"""
KiTS23 Dataset Statistics Extraction Script (Optimized Version)

Extracts essential statistics from the KiTS23 dataset:
- Shapes, spacings, orientations
- Intensity percentiles (0.5%, 99.5%) from ALL foreground voxels (kidney+tumor+cyst)
- Class distributions
- Number of kidneys per patient (via connected component analysis)

Usage:
    python kits_dataset_statistics.py [--debug] [--num_workers N]
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional
import json
import argparse
import logging
from tqdm import tqdm
from collections import Counter
from scipy.ndimage import label as scipy_label
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset paths
KITS_DATASET_PATH = Path("/vol/miltank/users/bergeral/kits23/dataset")
OUTPUT_PATH = Path("/vol/miltank/users/bergeral/3D_segmentation/kits_statistics.json")

# Label definitions for KiTS23
LABEL_NAMES = {
    0: "background",
    1: "kidney", 
    2: "tumor",
    3: "cyst"
}


def get_case_list(dataset_path: Path) -> List[str]:
    """Get sorted list of case IDs from dataset."""
    cases = sorted([
        d.name for d in dataset_path.iterdir() 
        if d.is_dir() and d.name.startswith('case_')
    ])
    return cases


def analyze_single_case(case_path: Path) -> Optional[Dict]:
    """
    Analyze a single case and return essential statistics.
    Loads full volume for accuracy - assumes sufficient memory.
    """
    case_id = case_path.name
    
    imaging_path = case_path / "imaging.nii.gz"
    seg_path = case_path / "segmentation.nii.gz"
    
    if not imaging_path.exists() or not seg_path.exists():
        logger.warning(f"{case_id}: Missing files")
        return None
    
    try:
        # Load data
        img_nib = nib.load(imaging_path)
        seg_nib = nib.load(seg_path)
        
        image = img_nib.get_fdata().astype(np.float32)
        seg = seg_nib.get_fdata().astype(np.int8)
        
        # Basic info
        shape = img_nib.shape
        spacing = tuple(np.round(img_nib.header.get_zooms()[:3], 4).tolist())
        orientation = tuple(nib.aff2axcodes(img_nib.affine))
        
        # Label counts
        label_counts = {
            0: int((seg == 0).sum()),
            1: int((seg == 1).sum()),
            2: int((seg == 2).sum()),
            3: int((seg == 3).sum())
        }
        
        # Foreground mask: ALL foreground classes (kidney=1, tumor=2, cyst=3)
        foreground_mask = seg >= 1
        
        # Get intensity samples from ALL foreground (for clipping percentiles)
        foreground_intensities = image[foreground_mask]
        num_fg_voxels = len(foreground_intensities)
        
        # Subsample if too many voxels (keep max 100k for percentile calculation)
        if num_fg_voxels > 100000:
            indices = np.random.choice(num_fg_voxels, 100000, replace=False)
            foreground_samples = foreground_intensities[indices].tolist()
        elif num_fg_voxels > 0:
            foreground_samples = foreground_intensities.tolist()
        else:
            foreground_samples = []  # Case has no foreground
        
        # Connected component analysis to count kidneys
        # Use the combined kidney+tumor+cyst mask
        labeled_array, num_components = scipy_label(foreground_mask)
        
        # Get component sizes and filter small noise
        component_sizes = []
        for comp_id in range(1, num_components + 1):
            size = int((labeled_array == comp_id).sum())
            if size > 1000:  # Filter noise (less than 1000 voxels)
                component_sizes.append(size)
        
        num_kidneys = len(component_sizes)
        component_sizes.sort(reverse=True)
        
        # Check which kidney has cyst
        has_cyst = label_counts[3] > 0
        
        stats = {
            'case_id': case_id,
            'shape': list(shape),
            'spacing': list(spacing),
            'orientation': list(orientation),
            'label_counts': label_counts,
            'has_tumor': label_counts[2] > 0,
            'has_cyst': has_cyst,
            'num_kidneys': num_kidneys,
            'kidney_sizes': component_sizes[:2],  # Keep top 2
            'foreground_samples': foreground_samples,
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"{case_id}: Error - {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_aggregate_statistics(case_stats: List[Dict]) -> Dict:
    """Compute aggregate statistics from all cases."""
    if len(case_stats) == 0:
        logger.error("No cases!")
        return {}
    
    # Collect arrays
    shapes = np.array([s['shape'] for s in case_stats])
    spacings = np.array([s['spacing'] for s in case_stats])
    orientations = [s['orientation'] for s in case_stats]
    
    # Combine all foreground samples for percentile calculation
    all_fg_samples = []
    for s in case_stats:
        all_fg_samples.extend(s['foreground_samples'])
    all_fg_samples = np.array(all_fg_samples)
    
    # Compute percentiles (only if we have foreground samples)
    percentiles = [0.5, 1, 5, 50, 95, 99, 99.5]
    if len(all_fg_samples) > 0:
        fg_percentiles = {f"p{p}": float(np.percentile(all_fg_samples, p)) for p in percentiles}
        fg_mean = float(all_fg_samples.mean())
        fg_std = float(all_fg_samples.std())
    else:
        fg_percentiles = {f"p{p}": 0.0 for p in percentiles}
        fg_mean = 0.0
        fg_std = 1.0
    
    # Label distribution
    total_labels = {0: 0, 1: 0, 2: 0, 3: 0}
    for s in case_stats:
        for label, count in s['label_counts'].items():
            total_labels[int(label)] += count
    
    total_voxels = sum(total_labels.values())
    label_pct = {k: v / total_voxels * 100 for k, v in total_labels.items()}
    
    # Kidney counts
    kidney_counts = [s['num_kidneys'] for s in case_stats]
    cases_1_kidney = sum(1 for n in kidney_counts if n == 1)
    cases_2_kidneys = sum(1 for n in kidney_counts if n >= 2)
    
    # Cases with pathology
    cases_with_tumor = sum(1 for s in case_stats if s['has_tumor'])
    cases_with_cyst = sum(1 for s in case_stats if s['has_cyst'])
    
    aggregate = {
        'num_cases': len(case_stats),
        
        # Shape
        'shape_min': shapes.min(axis=0).tolist(),
        'shape_max': shapes.max(axis=0).tolist(),
        'shape_median': np.median(shapes, axis=0).tolist(),
        
        # Spacing with percentiles
        'spacing_min': spacings.min(axis=0).tolist(),
        'spacing_max': spacings.max(axis=0).tolist(),
        'spacing_median': np.median(spacings, axis=0).tolist(),
        'spacing_p10': np.percentile(spacings, 10, axis=0).tolist(),
        'spacing_p25': np.percentile(spacings, 25, axis=0).tolist(),
        'spacing_p75': np.percentile(spacings, 75, axis=0).tolist(),
        'spacing_p90': np.percentile(spacings, 90, axis=0).tolist(),
        
        # Orientation
        'orientations': dict(Counter([str(o) for o in orientations])),
        
        # Foreground intensity (kidney+tumor+cyst - for clipping)
        'foreground_intensity': {
            'mean': fg_mean,
            'std': fg_std,
            'percentiles': fg_percentiles,
            'num_samples': len(all_fg_samples),
        },
        
        # Labels
        'label_counts': total_labels,
        'label_percentages': label_pct,
        
        # Pathology
        'cases_with_tumor': cases_with_tumor,
        'cases_with_cyst': cases_with_cyst,
        
        # Kidneys
        'cases_with_1_kidney': cases_1_kidney,
        'cases_with_2_kidneys': cases_2_kidneys,
        'kidney_count_distribution': dict(Counter(kidney_counts)),
    }
    
    # Determine recommended target spacing
    # Check if median spacing is highly anisotropic (ratio > 3)
    spacing_median = np.median(spacings, axis=0)
    spacing_p10 = np.percentile(spacings, 10, axis=0)
    anisotropy_ratio = np.max(spacing_median) / np.min(spacing_median)
    
    if anisotropy_ratio > 3:
        # Highly anisotropic - use 10th percentile only for the low-res axis
        # Use median for the high-resolution axes
        low_res_axis = int(np.argmax(spacing_median))
        recommended_spacing = spacing_median.copy()
        recommended_spacing[low_res_axis] = spacing_p10[low_res_axis]
        recommended_spacing = recommended_spacing.tolist()
        spacing_strategy = f"hybrid: p10 for axis {low_res_axis}, median for others (anisotropic, ratio={anisotropy_ratio:.2f})"
    else:
        # Reasonably isotropic - use median for all axes
        recommended_spacing = spacing_median.tolist()
        spacing_strategy = f"median (isotropic dataset, ratio={anisotropy_ratio:.2f})"
    
    aggregate['anisotropy_ratio'] = float(anisotropy_ratio)
    aggregate['spacing_strategy'] = spacing_strategy
    aggregate['recommended_target_spacing'] = recommended_spacing
    aggregate['recommended_clip_values'] = [fg_percentiles['p0.5'], fg_percentiles['p99.5']]
    aggregate['recommended_global_mean'] = fg_mean
    aggregate['recommended_global_std'] = fg_std
    
    return aggregate


def main():
    parser = argparse.ArgumentParser(description='Extract KiTS23 dataset statistics')
    parser.add_argument('--debug', action='store_true', help='Run on first 10 cases only')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--output', type=str, default=str(OUTPUT_PATH))
    parser.add_argument('--dataset', type=str, default=str(KITS_DATASET_PATH))
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    
    cases = get_case_list(dataset_path)
    logger.info(f"Found {len(cases)} cases")
    
    if args.debug:
        cases = cases[:50]
        logger.info(f"Debug mode: processing {len(cases)} cases")
    
    # Process cases in parallel
    case_stats = []
    case_paths = [dataset_path / case_id for case_id in cases]
    
    if args.num_workers > 1:
        logger.info(f"Using {args.num_workers} workers for parallel processing")
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(analyze_single_case, path): path.name 
                      for path in case_paths}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing"):
                case_id = futures[future]
                try:
                    stats = future.result()
                    if stats:
                        case_stats.append(stats)
                except Exception as e:
                    logger.error(f"{case_id}: Failed - {e}")
    else:
        # Sequential processing
        for case_path in tqdm(case_paths, desc="Analyzing"):
            stats = analyze_single_case(case_path)
            if stats:
                case_stats.append(stats)
    
    logger.info(f"Processed {len(case_stats)}/{len(cases)} cases")
    
    # Aggregate
    aggregate = compute_aggregate_statistics(case_stats)
    
    # Print summary
    print("\n" + "="*60)
    print("KiTS23 DATASET STATISTICS")
    print("="*60)
    print(f"Cases: {aggregate['num_cases']}")
    
    print(f"\nSpacing statistics:")
    print(f"  Min:    {aggregate['spacing_min']} mm")
    print(f"  P10:    {aggregate['spacing_p10']} mm")
    print(f"  Median: {aggregate['spacing_median']} mm")
    print(f"  P90:    {aggregate['spacing_p90']} mm")
    print(f"  Max:    {aggregate['spacing_max']} mm")
    print(f"  Anisotropy ratio: {aggregate['anisotropy_ratio']:.2f}")
    
    print(f"\nShape range: {aggregate['shape_min']} - {aggregate['shape_max']}")
    print(f"\nOrientations: {aggregate['orientations']}")
    
    print(f"\nForeground Intensity (kidney+tumor+cyst):")
    fg = aggregate['foreground_intensity']
    print(f"  Mean: {fg['mean']:.1f}, Std: {fg['std']:.1f}")
    print(f"  Num samples: {fg['num_samples']}")
    print(f"  Percentiles: {fg['percentiles']}")
    
    print(f"\nRecommended parameters:")
    print(f"  Clip values [p0.5, p99.5]: {aggregate['recommended_clip_values']}")
    print(f"  Target spacing: {aggregate['recommended_target_spacing']}")
    print(f"  Spacing strategy: {aggregate['spacing_strategy']}")
    
    print(f"\nLabel distribution:")
    for label, pct in aggregate['label_percentages'].items():
        print(f"  {label} ({LABEL_NAMES[label]}): {pct:.3f}%")
    
    print(f"\nCases with tumor: {aggregate['cases_with_tumor']}")
    print(f"Cases with cyst: {aggregate['cases_with_cyst']}")
    print(f"\nKidney detection:")
    print(f"  1 kidney: {aggregate['cases_with_1_kidney']} (nephrectomy patients)")
    print(f"  2 kidneys: {aggregate['cases_with_2_kidneys']}")
    print("="*60)
    
    # Save (exclude samples)
    output_data = {
        'aggregate': aggregate,
        'per_case': [{k: v for k, v in s.items() if k != 'foreground_samples'} 
                     for s in case_stats]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
