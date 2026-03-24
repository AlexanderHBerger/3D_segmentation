"""
Preprocess the complete dataset for training and save to nnUNet_preprocessed directory.

This script:
1. Loads the dataset split
2. Preprocesses ALL training samples (from all folds)
3. Saves preprocessed data to a new directory: Dataset015_MetastasisCollection_FastFull
4. Computes and saves global dataset statistics
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import sys

from config import DataConfig
from fast_preprocessing import preprocess_case

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset_split(preprocessed_data_path: Path) -> Dict:
    """Load the dataset split from nnUNet preprocessed directory."""
    splits_file = preprocessed_data_path / "splits_final.json"
    
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    return splits


def get_image_and_seg_paths(
    case_id: str, 
    raw_data_path: Path
) -> Tuple[Path, Path]:
    """Get paths to image and segmentation files."""
    image_path = raw_data_path / "imagesTr" / f"{case_id}_0000.nii.gz"
    seg_path = raw_data_path / "labelsTr" / f"{case_id}.nii.gz"
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")
    
    return image_path, seg_path


def compute_dataset_statistics(
    output_dir: Path,
    case_ids: List[str]
) -> Dict:
    """
    Compute global dataset statistics for normalization and inspection.
    
    Statistics computed:
    - Shape statistics (min, max, mean, median, std)
    - Intensity statistics (range, percentiles)
    - Foreground statistics (lesion volume, count)
    - Memory statistics (file sizes)
    """
    logger.info("Computing global dataset statistics...")
    
    stats = {
        'n_cases': len(case_ids),
        'case_ids': case_ids,
        'shapes': [],
        'intensity_ranges': [],
        'lesion_volumes': [],
        'lesion_counts': [],
        'file_sizes_mb': [],
        'foreground_percentages': [],
    }
    
    for case_id in tqdm(case_ids, desc="Computing statistics"):
        try:
            # Load preprocessed data
            data = np.load(output_dir / f"{case_id}_data.npy")
            seg = np.load(output_dir / f"{case_id}_seg.npy")
            
            # Shape statistics
            stats['shapes'].append(data.shape[1:])  # Exclude channel dimension
            
            # Intensity statistics
            stats['intensity_ranges'].append((float(data.min()), float(data.max())))
            
            # Lesion statistics
            lesion_mask = seg == 1
            lesion_volume = lesion_mask.sum()
            stats['lesion_volumes'].append(int(lesion_volume))
            
            # Count number of lesions (connected components)
            from scipy.ndimage import label as scipy_label
            labeled, num_lesions = scipy_label(lesion_mask[0])
            stats['lesion_counts'].append(int(num_lesions))
            
            # Foreground percentage
            foreground_mask = seg >= 0  # Everything except -1
            foreground_pct = foreground_mask.sum() / seg.size * 100
            stats['foreground_percentages'].append(float(foreground_pct))
            
            # File size
            data_file = output_dir / f"{case_id}_data.npy"
            seg_file = output_dir / f"{case_id}_seg.npy"
            total_size_mb = (data_file.stat().st_size + seg_file.stat().st_size) / (1024 * 1024)
            stats['file_sizes_mb'].append(float(total_size_mb))
            
        except Exception as e:
            logger.warning(f"Failed to compute statistics for {case_id}: {e}")
            continue
    
    # Compute aggregate statistics
    stats['shape_stats'] = {
        'min': [int(x) for x in np.min(stats['shapes'], axis=0)],
        'max': [int(x) for x in np.max(stats['shapes'], axis=0)],
        'mean': [float(x) for x in np.mean(stats['shapes'], axis=0)],
        'median': [int(x) for x in np.median(stats['shapes'], axis=0)],
        'std': [float(x) for x in np.std(stats['shapes'], axis=0)],
    }
    
    stats['intensity_stats'] = {
        'global_min': float(np.min([r[0] for r in stats['intensity_ranges']])),
        'global_max': float(np.max([r[1] for r in stats['intensity_ranges']])),
        'mean_min': float(np.mean([r[0] for r in stats['intensity_ranges']])),
        'mean_max': float(np.mean([r[1] for r in stats['intensity_ranges']])),
    }
    
    stats['lesion_stats'] = {
        'min_volume': int(np.min(stats['lesion_volumes'])),
        'max_volume': int(np.max(stats['lesion_volumes'])),
        'mean_volume': float(np.mean(stats['lesion_volumes'])),
        'median_volume': float(np.median(stats['lesion_volumes'])),
        'total_volume': int(np.sum(stats['lesion_volumes'])),
        'min_count': int(np.min(stats['lesion_counts'])),
        'max_count': int(np.max(stats['lesion_counts'])),
        'mean_count': float(np.mean(stats['lesion_counts'])),
    }
    
    stats['memory_stats'] = {
        'total_size_gb': float(np.sum(stats['file_sizes_mb']) / 1024),
        'mean_size_mb': float(np.mean(stats['file_sizes_mb'])),
        'median_size_mb': float(np.median(stats['file_sizes_mb'])),
    }
    
    stats['foreground_stats'] = {
        'mean_foreground_pct': float(np.mean(stats['foreground_percentages'])),
        'min_foreground_pct': float(np.min(stats['foreground_percentages'])),
        'max_foreground_pct': float(np.max(stats['foreground_percentages'])),
    }
    
    return stats


def main():
    # Configuration
    data_config = DataConfig()
    
    # Paths (hardcoded)
    raw_data_path = Path("/ministorage/ahb/data/nnUNet_raw/Dataset015_MetastasisCollection")
    preprocessed_base = Path("/ministorage/ahb/data/nnUNet_preprocessed")
    preprocessed_data_path = Path("/ministorage/ahb/data/nnUNet_preprocessed/Dataset015_MetastasisCollection")
    output_dir = preprocessed_base / "Dataset015_MetastasisCollection_FastFull"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load dataset splits
    logger.info("Loading dataset splits...")
    splits = load_dataset_split(preprocessed_data_path)
    
    # Get all training cases from all folds
    all_train_cases = []
    for fold_data in splits:
        all_train_cases.extend(fold_data['train'])
    
    # Remove duplicates and sort
    all_train_cases = sorted(list(set(all_train_cases)))
    
    logger.info(f"Total training cases available: {len(all_train_cases)}")
    logger.info(f"Will preprocess ALL {len(all_train_cases)} cases")
    
    # Use all cases
    selected_cases = all_train_cases
    
    # Preprocess each case
    successful_cases = []
    failed_cases = []
    
    logger.info("Starting preprocessing...")
    for i, case_id in enumerate(tqdm(selected_cases, desc="Preprocessing")):
        try:
            # Get file paths
            image_path, seg_path = get_image_and_seg_paths(case_id, raw_data_path)
            
            # Preprocess
            properties = preprocess_case(
                image_path=image_path,
                seg_path=seg_path,
                output_dir=output_dir,
                case_id=case_id,
                data_config=data_config
            )
            
            successful_cases.append(case_id)
            
        except Exception as e:
            logger.error(f"Failed to preprocess {case_id}: {e}")
            failed_cases.append((case_id, str(e)))
            continue
    
    logger.info(f"Preprocessing complete: {len(successful_cases)} successful, {len(failed_cases)} failed")
    
    if failed_cases:
        logger.warning("Failed cases:")
        for case_id, error in failed_cases:
            logger.warning(f"  {case_id}: {error}")
    
    # Compute dataset statistics
    if successful_cases:
        stats = compute_dataset_statistics(output_dir, successful_cases)
        
        # Save statistics
        stats_file = output_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved dataset statistics to: {stats_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("DATASET STATISTICS SUMMARY")
        print("="*80)
        print(f"Number of cases: {stats['n_cases']}")
        print(f"\nShape statistics (X, Y, Z):")
        print(f"  Min:    {stats['shape_stats']['min']}")
        print(f"  Max:    {stats['shape_stats']['max']}")
        print(f"  Mean:   {stats['shape_stats']['mean']}")
        print(f"  Median: {stats['shape_stats']['median']}")
        print(f"\nIntensity statistics (normalized):")
        print(f"  Global range: [{stats['intensity_stats']['global_min']:.2f}, {stats['intensity_stats']['global_max']:.2f}]")
        print(f"  Mean range:   [{stats['intensity_stats']['mean_min']:.2f}, {stats['intensity_stats']['mean_max']:.2f}]")
        print(f"\nLesion statistics:")
        print(f"  Volume range:  {stats['lesion_stats']['min_volume']} - {stats['lesion_stats']['max_volume']} voxels")
        print(f"  Mean volume:   {stats['lesion_stats']['mean_volume']:.1f} voxels")
        print(f"  Count range:   {stats['lesion_stats']['min_count']} - {stats['lesion_stats']['max_count']} lesions")
        print(f"  Mean count:    {stats['lesion_stats']['mean_count']:.2f} lesions")
        print(f"  Total volume:  {stats['lesion_stats']['total_volume']} voxels")
        print(f"\nMemory statistics:")
        print(f"  Total size:    {stats['memory_stats']['total_size_gb']:.2f} GB")
        print(f"  Mean per case: {stats['memory_stats']['mean_size_mb']:.2f} MB")
        print(f"\nForeground statistics:")
        print(f"  Mean foreground: {stats['foreground_stats']['mean_foreground_pct']:.1f}%")
        print(f"  Range: {stats['foreground_stats']['min_foreground_pct']:.1f}% - {stats['foreground_stats']['max_foreground_pct']:.1f}%")
        print("="*80)
        
        # Save case list
        case_list_file = output_dir / "case_list.txt"
        with open(case_list_file, 'w') as f:
            for case_id in successful_cases:
                f.write(f"{case_id}\n")
        logger.info(f"Saved case list to: {case_list_file}")
        
        # Create a simple dataset.json compatible file
        dataset_info = {
            "name": "Dataset015_MetastasisCollection_FastFull",
            "description": "Fast preprocessed complete dataset for MedNeXt training",
            "n_cases": len(successful_cases),
            "case_ids": successful_cases,
            "target_spacing": data_config.target_spacing,
            "num_channels": 1,
            "num_classes": 2,
            "labels": {
                "background": 0,
                "metastasis": 1
            }
        }
        
        dataset_file = output_dir / "dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        logger.info(f"Saved dataset info to: {dataset_file}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
