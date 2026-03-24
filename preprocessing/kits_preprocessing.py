"""
KiTS23 Preprocessing Pipeline for Kidney Tumor Segmentation

This script preprocesses the KiTS23 dataset following nnUNet methodology with
adaptations for CT data.

Key differences from BraTS (MRI) preprocessing:
1. CT has meaningful intensity values (Hounsfield Units) - use percentile clipping
2. Background is not zero (air = -1024 HU typically)
3. Extract ONE ROI per patient containing ALL foreground (both kidneys + tumors + cysts)
4. Only process patients that have at least one tumor (label 2)

Pipeline steps (optimized - ROI extraction before resampling for speed):
1. Load CT volume and segmentation
2. Check if patient has any tumors - skip if not
3. Reorient to canonical orientation (RAS+)
4. Find bounding box around ALL foreground (kidney + tumor + cyst) in original space
5. Create cubic bounding box with safety margin
6. Crop to ROI (much smaller volume for faster resampling)
7. Resample ROI to target spacing (median of dataset)
8. Clip intensities to [p0.5, p99.5] percentiles (based on tumor voxel statistics)
9. Z-score normalize using global dataset statistics (mean and std from all foreground voxels)
10. Create label mask: tumor=1, rest=0 (kidney/cyst/background)
11. Save as numpy arrays

Usage:
    python kits_preprocessing.py [--debug] [--num_workers N]
"""

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from scipy.ndimage import binary_dilation
import logging
import argparse
import json
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Dataset paths
DEFAULT_KITS_DATASET_PATH = Path("/vol/miltank/users/bergeral/kits23/dataset")
DEFAULT_OUTPUT_PATH = Path("/vol/miltank/users/bergeral/nnUNet_preprocessed/Dataset018_KiTS_lowres")
STATISTICS_PATH = Path("/vol/miltank/users/bergeral/3D_segmentation/kits_statistics.json")
DEFAULT_SPLITS_PATH = Path("/vol/miltank/users/bergeral/3D_segmentation/kits_splits.json")

# Target spacing (median of dataset - will be loaded from statistics)
DEFAULT_TARGET_SPACING = (0.8125, 0.8125, 3.0)  # Corrected for RAS+ (H, W, Slice)

# Intensity clipping percentiles (from dataset statistics)
DEFAULT_CLIP_LOW = -54.0   # p0.5 of foreground intensities  
DEFAULT_CLIP_HIGH = 299.0  # p99.5 of foreground intensities

# Global normalization statistics (from dataset statistics)
DEFAULT_GLOBAL_MEAN = 103.0995562154447
DEFAULT_GLOBAL_STD = 73.34347241820662

# ROI extraction parameters
FOREGROUND_DILATION_ITERATIONS = 5  # Dilate foreground mask (in voxels at target spacing)
DEFAULT_MARGIN_MM = 45  # Additional margin in mm to ensure no foreground touches border

# Interpolation orders
DATA_INTERPOLATION_ORDER = 3  # Cubic for image
SEG_INTERPOLATION_ORDER = 0   # Nearest neighbor for segmentation

# Output settings
DATA_DTYPE = np.float32
SEG_DTYPE = np.int8
SAVE_NIFTI_DEBUG = False  # Save NIfTI files for visual inspection

# Label definitions
# Original KiTS labels: 0=background, 1=kidney, 2=tumor, 3=cyst
# Output labels: 0=background (kidney+cyst+background), 1=tumor


def load_preprocessing_params() -> Dict:
    """Load preprocessing parameters from statistics file."""
    params = {
        'target_spacing': DEFAULT_TARGET_SPACING,
        'clip_low': DEFAULT_CLIP_LOW,
        'clip_high': DEFAULT_CLIP_HIGH,
        'global_mean': DEFAULT_GLOBAL_MEAN,
        'global_std': DEFAULT_GLOBAL_STD,
    }
    
    if STATISTICS_PATH.exists():
        try:
            with open(STATISTICS_PATH, 'r') as f:
                stats = json.load(f)
            
            agg = stats.get('aggregate', {})
            if 'recommended_target_spacing' in agg:
                ts = agg['recommended_target_spacing']
                # Statistics are from raw data (Slice, H, W) but we process in RAS+ (H, W, Slice)
                # So we need to swap the axes to match RAS+
                params['target_spacing'] = (ts[1], ts[2], ts[0])
            if 'recommended_clip_values' in agg:
                params['clip_low'] = agg['recommended_clip_values'][0]
                params['clip_high'] = agg['recommended_clip_values'][1]
            if 'recommended_global_mean' in agg:
                params['global_mean'] = agg['recommended_global_mean']
            if 'recommended_global_std' in agg:
                params['global_std'] = agg['recommended_global_std']
                
            logger.info(f"Loaded preprocessing params from {STATISTICS_PATH}")
            logger.info(f"  Target spacing: {params['target_spacing']}")
            logger.info(f"  Clip range: [{params['clip_low']:.1f}, {params['clip_high']:.1f}]")
            logger.info(f"  Global normalization: mean={params['global_mean']:.2f}, std={params['global_std']:.2f}")
        except Exception as e:
            logger.warning(f"Could not load statistics: {e}. Using defaults.")
    else:
        logger.warning(f"Statistics file not found: {STATISTICS_PATH}. Using defaults.")
    
    return params


def get_case_list(dataset_path: Path, subset_file: Optional[Path] = None, split_name: Optional[str] = None) -> List[str]:
    """Get sorted list of case IDs, optionally filtered by a subset.
    
    Args:
        dataset_path: Path to the dataset
        subset_file: Optional JSON file with splits
        split_name: Optional split name (e.g., 'train', 'val', 'test')
    
    Returns:
        List of case IDs to process
    """
    if subset_file is not None and split_name is not None:
        # Load subset from splits file
        if not subset_file.exists():
            raise FileNotFoundError(f"Subset file not found: {subset_file}")
        
        with open(subset_file, 'r') as f:
            splits = json.load(f)
        
        if split_name not in splits['splits']:
            raise ValueError(f"Split '{split_name}' not found in {subset_file}")
        
        subset_cases = splits['splits'][split_name]
        logger.info(f"Loaded {len(subset_cases)} cases from split '{split_name}' in {subset_file}")
        return sorted(subset_cases)
    else:
        # Get all cases
        return sorted([
            d.name for d in dataset_path.iterdir()
            if d.is_dir() and d.name.startswith('case_')
        ])


def case_has_tumor_fast(case_path: Path) -> bool:
    """
    Quickly check if a case has any tumor annotations by looking at instances folder.
    This avoids loading the full segmentation volume for cases without tumors.
    """
    instances_dir = case_path / "instances"
    
    if not instances_dir.exists():
        # No instances folder - need to check full segmentation
        return True  # Be conservative, assume it might have tumors
    
    # Check for any tumor instance files (tumor_*.nii.gz)
    tumor_files = list(instances_dir.glob("tumor_*.nii.gz"))
    return len(tumor_files) > 0


def get_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Get bounding box from binary mask."""
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    
    bbox = tuple((int(c.min()), int(c.max() + 1)) for c in coords)
    return bbox


def crop_to_bbox(data: np.ndarray, bbox: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    """Crop 3D array to bounding box."""
    return data[
        bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]
    ]


def add_margin_to_bbox(
    bbox: Tuple[Tuple[int, int], ...],
    volume_shape: Tuple[int, ...],
    spacing: Tuple[float, float, float],
    margin_mm: float
) -> Tuple[Tuple[int, int], ...]:
    """
    Add physical margin to bounding box.
    
    Args:
        bbox: Original bounding box as ((min0, max0), (min1, max1), (min2, max2))
        volume_shape: Shape of the full volume (to avoid going out of bounds)
        spacing: Voxel spacing in mm
        margin_mm: Margin to add on all sides in mm
        
    Returns:
        Bounding box with margin, clipped to volume bounds
    """
    new_bbox = []
    for i in range(3):
        # Calculate margin in voxels for this dimension
        margin_voxels = int(np.ceil(margin_mm / spacing[i]))
        
        new_min = bbox[i][0] - margin_voxels
        new_max = bbox[i][1] + margin_voxels
        
        # Clip to volume bounds
        new_min = max(0, new_min)
        new_max = min(volume_shape[i], new_max)
        
        new_bbox.append((new_min, new_max))
    
    return tuple(new_bbox)


def preprocess_patient_roi(
    image: np.ndarray,
    seg: np.ndarray,
    params: Dict,
    case_id: str,
    target_spacing: Tuple[float, float, float]
) -> Optional[Dict]:
    """
    Preprocess a patient's ROI after resampling.
    
    Applies intensity clipping, normalization, and creates output segmentation.
    Assumes the ROI has already been extracted and resampled.
    
    Args:
        image: Resampled CT ROI volume
        seg: Resampled segmentation ROI (0=bg, 1=kidney, 2=tumor, 3=cyst)
        params: Preprocessing parameters
        case_id: Patient identifier
        target_spacing: Target voxel spacing
        
    Returns:
        Dict with 'image', 'seg', 'properties' or None if invalid
    """
    # Clip intensities using global percentiles
    image = np.clip(image, params['clip_low'], params['clip_high'])
    
    # Z-score normalization using global dataset statistics
    # CT data has consistent Hounsfield Units, so global statistics are appropriate
    global_mean = params['global_mean']
    global_std = params['global_std']
    
    image = (image - global_mean) / global_std
    
    # Create output segmentation:
    # - Tumor (2) -> 1
    # - Everything else (background, kidney, cyst) -> 0
    seg_out = np.zeros_like(seg, dtype=SEG_DTYPE)
    seg_out[seg == 2] = 1  # Tumor = 1
    
    # Add channel dimension
    image = image[np.newaxis, ...].astype(DATA_DTYPE)
    seg_out = seg_out[np.newaxis, ...]
    
    properties = {
        'shape': image.shape,
        'num_tumor_voxels': int((seg_out == 1).sum()),
        'num_kidney_voxels': int((seg == 1).sum()),
        'num_cyst_voxels': int((seg == 3).sum()),
        'num_background_voxels': int((seg_out == 0).sum()),
        'normalization_mean': global_mean,
        'normalization_std': global_std,
    }
    
    return {
        'image': image,
        'seg': seg_out,
        'properties': properties
    }


def extract_roi_bbox(
    seg: np.ndarray,
    spacing: Tuple[float, float, float],
    volume_shape: Tuple[int, ...],
    case_id: str,
    margin_mm: float
) -> Optional[Tuple[Tuple[int, int], ...]]:
    """
    Extract bounding box around all foreground in original space.
    
    Args:
        seg: Segmentation volume (0=bg, 1=kidney, 2=tumor, 3=cyst)
        spacing: Voxel spacing in mm
        volume_shape: Shape of the volume
        case_id: Case identifier for logging
        
    Returns:
        Bounding box as ((min0, max0), (min1, max1), (min2, max2)) or None
    """
    # Create foreground mask (all non-background: kidney + tumor + cyst)
    foreground_mask = seg >= 1
    
    if not foreground_mask.any():
        logger.warning(f"  {case_id}: No foreground found!")
        return None
    
    # Dilate foreground mask to create the region of interest
    dilated_foreground = binary_dilation(foreground_mask, iterations=FOREGROUND_DILATION_ITERATIONS)
    
    # Get bounding box around dilated foreground
    bbox_dilated = get_bbox_from_mask(dilated_foreground)
    if bbox_dilated is None:
        logger.warning(f"  {case_id}: Empty dilated mask")
        return None
    
    # Add margin
    bbox = add_margin_to_bbox(bbox_dilated, volume_shape, spacing, margin_mm)
    
    # Verify that ALL original foreground is contained within the bbox
    fg_coords = np.where(foreground_mask)
    for dim in range(3):
        fg_min, fg_max = fg_coords[dim].min(), fg_coords[dim].max()
        box_min, box_max = bbox[dim]
        if fg_min < box_min or fg_max >= box_max:
            logger.error(f"  {case_id}: Foreground cut off in dim {dim}! "
                        f"FG: [{fg_min}, {fg_max}], Box: [{box_min}, {box_max}]")
            return None
    
    return bbox


def preprocess_case(
    case_path: Path,
    output_dir: Path,
    params: Dict,
    margin_mm: float = DEFAULT_MARGIN_MM
) -> Optional[str]:
    """
    Preprocess a single case, creating one ROI per patient.
    Only processes patients that have at least one tumor.
    
    Optimized pipeline: Extract ROI first, then resample (much faster).
    
    Returns:
        Case ID if saved, None otherwise
    """
    case_id = case_path.name
    timings = {}
    total_start = time.time()
    
    imaging_path = case_path / "imaging.nii.gz"
    seg_path = case_path / "segmentation.nii.gz"
    
    if not imaging_path.exists() or not seg_path.exists():
        logger.warning(f"{case_id}: Missing files, skipping")
        return None
    
    # Fast check: does this case have any tumors?
    if not case_has_tumor_fast(case_path):
        logger.info(f"{case_id}: No tumor instances found, skipping")
        return None
    
    try:
        # Load data
        t0 = time.time()
        img_nib = nib.load(imaging_path)
        seg_nib = nib.load(seg_path)
        timings['load_headers'] = time.time() - t0
        
        original_spacing = img_nib.header.get_zooms()[:3]
        original_shape = img_nib.shape
        
        logger.info(f"{case_id}: shape={original_shape}, spacing={tuple(f'{s:.2f}' for s in original_spacing)}")
        
        # Step 1: Canonical orientation (fast, just changes affine)
        t0 = time.time()
        img_canonical = nib.as_closest_canonical(img_nib)
        seg_canonical = nib.as_closest_canonical(seg_nib)
        timings['canonical'] = time.time() - t0
        
        # Load segmentation to find ROI and verify tumor presence
        t0 = time.time()
        seg_data = seg_canonical.get_fdata().astype(np.int8)
        timings['load_seg'] = time.time() - t0
        
        if not (seg_data == 2).any():
            logger.info(f"{case_id}: No tumors in segmentation, skipping")
            return None
        
        # Step 2: Extract ROI bounding box in original space (BEFORE resampling)
        t0 = time.time()
        current_spacing = img_canonical.header.get_zooms()[:3]
        bbox = extract_roi_bbox(seg_data, current_spacing, seg_data.shape, case_id, margin_mm)
        timings['extract_bbox'] = time.time() - t0
        
        if bbox is None:
            return None
        
        logger.info(f"  ROI bbox: {bbox}, size: {[b[1]-b[0] for b in bbox]}")
        
        # Step 3: Crop to ROI in original space
        t0 = time.time()
        img_data = img_canonical.get_fdata().astype(np.float32)
        timings['load_img'] = time.time() - t0
        
        t0 = time.time()
        img_roi = crop_to_bbox(img_data, bbox)
        seg_roi = crop_to_bbox(seg_data, bbox)
        timings['crop'] = time.time() - t0
        
        # Free memory from full volumes
        del img_data, seg_data
        
        # Create new NIfTI objects for the ROI with updated affine
        # Adjust affine to account for the crop offset
        roi_affine = img_canonical.affine.copy()
        roi_offset = np.array([bbox[0][0], bbox[1][0], bbox[2][0], 0])
        # Convert voxel offset to world coordinates
        roi_affine[:3, 3] = roi_affine[:3, 3] + roi_affine[:3, :3] @ roi_offset[:3]
        
        img_roi_nib = nib.Nifti1Image(img_roi, roi_affine)
        seg_roi_nib = nib.Nifti1Image(seg_roi.astype(np.float32), roi_affine)
        
        logger.info(f"  Cropped ROI: {img_roi.shape} (was {original_shape})")
        
        # Step 4: Resample ROI to target spacing
        t0 = time.time()
        target_spacing = params['target_spacing']
        anisotropy_ratio = np.max(current_spacing) / np.min(current_spacing)
        
        if anisotropy_ratio > 3:
            # Highly anisotropic - resample in two steps
            low_res_axis = np.argmax(current_spacing)
            logger.info(f"  Anisotropic (ratio={anisotropy_ratio:.1f}), low-res axis: {low_res_axis}")
            
            # Step 1: Resample out-of-plane axis with nearest neighbor
            intermediate_spacing = list(current_spacing)
            intermediate_spacing[low_res_axis] = target_spacing[low_res_axis]
            
            img_intermediate = resample_to_output(
                img_roi_nib, voxel_sizes=intermediate_spacing, order=0
            )
            
            # Step 2: Resample in-plane with high-order interpolation
            img_resampled = resample_to_output(
                img_intermediate, voxel_sizes=target_spacing, order=DATA_INTERPOLATION_ORDER
            )
            
            seg_intermediate = resample_to_output(
                seg_roi_nib, voxel_sizes=intermediate_spacing, order=SEG_INTERPOLATION_ORDER
            )
            seg_resampled = resample_from_to(
                seg_intermediate, img_resampled, order=SEG_INTERPOLATION_ORDER
            )
        else:
            img_resampled = resample_to_output(
                img_roi_nib, voxel_sizes=target_spacing, order=DATA_INTERPOLATION_ORDER
            )
            seg_resampled = resample_from_to(
                seg_roi_nib, img_resampled, order=SEG_INTERPOLATION_ORDER
            )
        timings['resample'] = time.time() - t0
        
        # Get numpy arrays
        t0 = time.time()
        image = img_resampled.get_fdata().astype(np.float32)
        seg = np.round(seg_resampled.get_fdata()).astype(np.int8)
        timings['get_resampled_data'] = time.time() - t0
        
        logger.info(f"  Resampled ROI: {image.shape}, spacing: {target_spacing}")
        
        # Double-check tumor presence after resampling
        if not (seg == 2).any():
            logger.info(f"  {case_id}: Tumors lost during resampling, skipping")
            return None
        
        # Step 5: Apply intensity normalization
        t0 = time.time()
        result = preprocess_patient_roi(image, seg, params, case_id, target_spacing)
        timings['normalize'] = time.time() - t0
        
        if result is None:
            return None
        
        # Add bbox info to properties
        result['properties']['bbox_original'] = [[int(b[0]), int(b[1])] for b in bbox]
        result['properties']['original_shape'] = [int(x) for x in original_shape]
        
        # Save
        t0 = time.time()
        np.save(output_dir / f"{case_id}_data.npy", result['image'])
        np.save(output_dir / f"{case_id}_seg.npy", result['seg'])
        
        # Save foreground coordinates for faster sampling during training
        foreground_mask = result['seg'][0] > 0
        foreground_coords = np.argwhere(foreground_mask)
        np.save(output_dir / f"{case_id}_foreground_coords.npy", foreground_coords)

        # Save properties
        with open(output_dir / f"{case_id}_properties.json", 'w') as f:
            props = result['properties'].copy()
            props['target_spacing'] = [float(x) for x in target_spacing]
            props['original_spacing'] = [float(x) for x in original_spacing]
            # Ensure all values are JSON serializable
            for key, value in props.items():
                if isinstance(value, (np.floating, np.float32, np.float64)):
                    props[key] = float(value)
                elif isinstance(value, (np.integer, np.int32, np.int64)):
                    props[key] = int(value)
                elif isinstance(value, list):
                    props[key] = [float(x) if isinstance(x, (np.floating, np.float32, np.float64)) else 
                                  int(x) if isinstance(x, (np.integer, np.int32, np.int64)) else x 
                                  for x in value]
            json.dump(props, f, indent=2)
        timings['save'] = time.time() - t0
        
        timings['total'] = time.time() - total_start
        
        # Log timing breakdown
        timing_str = ", ".join([f"{k}={v:.2f}s" for k, v in timings.items()])
        logger.info(f"  Timings: {timing_str}")
        
        logger.info(f"  Saved: shape={result['image'].shape}, "
                   f"tumor_voxels={result['properties']['num_tumor_voxels']}, "
                   f"kidney_voxels={result['properties']['num_kidney_voxels']}, "
                   f"cyst_voxels={result['properties']['num_cyst_voxels']}")
        
        # Optionally save NIfTI for debugging
        if SAVE_NIFTI_DEBUG:
            save_debug_nifti(result, output_dir, case_id, target_spacing)
        
        return case_id
        
    except Exception as e:
        logger.error(f"{case_id}: Error - {e}")
        import traceback
        traceback.print_exc()
        return None


def save_debug_nifti(result: Dict, output_dir: Path, case_id: str, spacing: Tuple):
    """Save NIfTI files for visual debugging."""
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    
    img_nib = nib.Nifti1Image(result['image'][0], affine)
    nib.save(img_nib, output_dir / f"{case_id}_image.nii.gz")
    
    seg_nib = nib.Nifti1Image(result['seg'][0].astype(np.float32), affine)
    nib.save(seg_nib, output_dir / f"{case_id}_seg.nii.gz")


def process_case_wrapper(args):
    """Wrapper for multiprocessing."""
    case_path, output_dir, params, margin_mm = args
    return preprocess_case(case_path, output_dir, params, margin_mm)


def main():
    parser = argparse.ArgumentParser(description='Preprocess KiTS23 dataset')
    parser.add_argument('--debug', action='store_true', help='Process only 5 cases')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_PATH), help='Output directory')
    parser.add_argument('--dataset', type=str, default=str(DEFAULT_KITS_DATASET_PATH), help='Dataset path')
    parser.add_argument('--save_nifti', action='store_true', help='Save NIfTI files for debugging')
    parser.add_argument('--margin', type=float, default=DEFAULT_MARGIN_MM, help='ROI margin in mm (default: 45mm)')
    parser.add_argument('--subset_file', type=str, default=None, help='JSON file with train/val/test splits')
    parser.add_argument('--split', type=str, default=None, help='Split to process (e.g., train, val, test)')
    args = parser.parse_args()
    
    global SAVE_NIFTI_DEBUG
    SAVE_NIFTI_DEBUG = args.save_nifti
    
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load preprocessing parameters
    params = load_preprocessing_params()
    
    # Get case list (optionally filtered by subset)
    subset_file = Path(args.subset_file) if args.subset_file else None
    cases = get_case_list(dataset_path, subset_file, args.split)
    logger.info(f"Found {len(cases)} cases")
    if args.split:
        logger.info(f"Processing split: {args.split}")
    logger.info(f"Using margin: {args.margin}mm")
    
    if args.debug:
        # For debug, pick cases that are likely to have tumors
        # First few cases might not have tumors, so we'll process more and stop at 5 saved
        cases = cases[:15]  # Check first 15
        logger.info(f"Debug mode: checking up to {len(cases)} cases to find 5 with tumors")
    
    # Process cases
    saved_cases = []
    skipped_no_tumor = 0
    
    if args.num_workers > 1:
        # Parallel processing
        logger.info(f"Using {args.num_workers} workers")
        tasks = [(dataset_path / case_id, output_dir, params, args.margin) for case_id in cases]
        
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_case_wrapper, task): task[0].name 
                      for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                case_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        saved_cases.append(result)
                    else:
                        skipped_no_tumor += 1
                except Exception as e:
                    logger.error(f"{case_id}: Failed - {e}")
                
                # Early stop in debug mode
                if args.debug and len(saved_cases) >= 5:
                    break
    else:
        # Sequential processing
        for case_id in tqdm(cases, desc="Processing"):
            result = preprocess_case(dataset_path / case_id, output_dir, params, args.margin)
            if result:
                saved_cases.append(result)
            else:
                skipped_no_tumor += 1
            result = preprocess_case(dataset_path / case_id, output_dir, params)
            if result:
                saved_cases.append(result)
            else:
                skipped_no_tumor += 1
            
            # Early stop in debug mode
            if args.debug and len(saved_cases) >= 5:
                break
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Preprocessing complete!")
    logger.info(f"  Cases checked: {len(saved_cases) + skipped_no_tumor}")
    logger.info(f"  Cases saved (with tumors): {len(saved_cases)}")
    logger.info(f"  Cases skipped (no tumors): {skipped_no_tumor}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 60)
    
    # Save list of processed cases
    with open(output_dir / "case_list.txt", 'w') as f:
        for case_id in sorted(saved_cases):
            f.write(f"{case_id}\n")
            f.write(f"{case_id}\n")
    
    # Save preprocessing parameters used
    with open(output_dir / "preprocessing_params.json", 'w') as f:
        json.dump({
            'target_spacing': list(params['target_spacing']),
            'clip_range': [params['clip_low'], params['clip_high']],
            'normalization': 'global z-score (mean and std from dataset statistics)',
            'global_mean': params['global_mean'],
            'global_std': params['global_std'],
            'foreground_dilation_iterations': FOREGROUND_DILATION_ITERATIONS,
            'margin_mm': args.margin,
            'bounding_box_type': 'cubic',
            'num_cases_saved': len(saved_cases),
            'split': args.split if args.split else 'all',
            'output_labels': {
                '0': 'background (kidney + cyst + background)',
                '1': 'tumor'
            }
        }, f, indent=2)


if __name__ == "__main__":
    main()
