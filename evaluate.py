"""
Evaluation script for segmentation predictions.

Computes metrics comparing predictions to ground truth labels.
Outputs per-sample CSV and aggregated JSON with statistics.

Usage:
    python evaluate.py \
        --predictions /path/to/predictions \
        --labels /path/to/labels \
        --output /path/to/output_dir
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import nibabel as nib

# Import calibration metrics from training module
from metrics import compute_calibration_metrics, prepare_targets_with_mask
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Metric Functions (adapted from metrics.py for numpy arrays)
# =============================================================================

def compute_dice_score(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Compute Dice score between binary prediction and target.
    
    Args:
        prediction: Binary prediction array (0 or 1)
        target: Binary target array (0 or 1)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice score (float between 0 and 1)
    """
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(prediction, target).sum()
    pred_sum = prediction.sum()
    target_sum = target.sum()
    
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return float(dice)


def compute_iou_score(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Compute IoU (Jaccard) score between binary prediction and target.
    
    Args:
        prediction: Binary prediction array (0 or 1)
        target: Binary target array (0 or 1)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score (float between 0 and 1)
    """
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def compute_sensitivity(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Compute sensitivity (recall, true positive rate).
    
    Args:
        prediction: Binary prediction array
        target: Binary target array
        smooth: Smoothing factor
        
    Returns:
        Sensitivity score
    """
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    
    true_positives = np.logical_and(prediction, target).sum()
    actual_positives = target.sum()
    
    sensitivity = (true_positives + smooth) / (actual_positives + smooth)
    return float(sensitivity)


def compute_specificity(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Compute specificity (true negative rate).
    
    Args:
        prediction: Binary prediction array
        target: Binary target array
        smooth: Smoothing factor
        
    Returns:
        Specificity score
    """
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    
    true_negatives = np.logical_and(~prediction, ~target).sum()
    actual_negatives = (~target).sum()
    
    specificity = (true_negatives + smooth) / (actual_negatives + smooth)
    return float(specificity)


def compute_precision(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Compute precision (positive predictive value).
    
    Args:
        prediction: Binary prediction array
        target: Binary target array
        smooth: Smoothing factor
        
    Returns:
        Precision score
    """
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    
    true_positives = np.logical_and(prediction, target).sum()
    predicted_positives = prediction.sum()
    
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    return float(precision)


def compute_volume_difference(
    prediction: np.ndarray,
    target: np.ndarray,
    voxel_volume: float = 1.0
) -> Tuple[float, float, float]:
    """
    Compute volume metrics.
    
    Args:
        prediction: Binary prediction array
        target: Binary target array
        voxel_volume: Volume of a single voxel in mm³
        
    Returns:
        Tuple of (predicted_volume, target_volume, volume_difference) in mm³
    """
    pred_volume = float(prediction.sum() * voxel_volume)
    target_volume = float(target.sum() * voxel_volume)
    volume_diff = pred_volume - target_volume
    
    return pred_volume, target_volume, volume_diff


def compute_hausdorff_distance_95(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute 95th percentile Hausdorff distance.
    
    Args:
        prediction: Binary prediction array
        target: Binary target array
        spacing: Voxel spacing in mm
        
    Returns:
        HD95 in mm (or inf if either mask is empty)
    """
    from scipy.ndimage import distance_transform_edt
    
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    
    # Handle empty cases
    if not prediction.any() or not target.any():
        return float('inf')
    
    # Compute distance transforms
    pred_surface = prediction & ~binary_erosion_3d(prediction)
    target_surface = target & ~binary_erosion_3d(target)
    
    # If surfaces are empty (single voxel cases), use the full mask
    if not pred_surface.any():
        pred_surface = prediction
    if not target_surface.any():
        target_surface = target
    
    # Distance from prediction surface to target
    dt_target = distance_transform_edt(~target, sampling=spacing)
    distances_pred_to_target = dt_target[pred_surface]
    
    # Distance from target surface to prediction
    dt_pred = distance_transform_edt(~prediction, sampling=spacing)
    distances_target_to_pred = dt_pred[target_surface]
    
    # Combine and compute 95th percentile
    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    hd95 = np.percentile(all_distances, 95)
    
    return float(hd95)


def binary_erosion_3d(mask: np.ndarray) -> np.ndarray:
    """Simple 3D binary erosion using scipy."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask)


# =============================================================================
# File Loading Utilities
# =============================================================================

def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    """
    Load NIfTI file and return data, affine, and spacing.
    
    Returns:
        Tuple of (data, affine, spacing)
    """
    nii = nib.load(path)
    data = nii.get_fdata()
    affine = nii.affine
    spacing = tuple(nii.header.get_zooms()[:3])
    return data, affine, spacing


def find_matching_files(
    pred_folder: Path,
    label_folder: Path,
    pred_suffix: str = ".nii.gz",
    label_suffix: str = ".nii.gz"
) -> List[Tuple[str, Path, Path]]:
    """
    Find matching prediction and label files.
    
    Args:
        pred_folder: Folder containing predictions
        label_folder: Folder containing labels
        pred_suffix: Suffix for prediction files
        label_suffix: Suffix for label files
        
    Returns:
        List of (case_name, pred_path, label_path) tuples
    """
    matches = []
    
    # Get all prediction files (excluding logits and probabilities)
    pred_files = sorted(pred_folder.glob(f"*{pred_suffix}"))
    pred_files = [f for f in pred_files if '_logits' not in f.name and '_probabilities' not in f.name]
    
    for pred_path in pred_files:
        # Extract case name
        case_name = pred_path.name.replace(pred_suffix, '')
        
        # Try to find matching label
        label_path = label_folder / f"{case_name}{label_suffix}"
        
        if label_path.exists():
            matches.append((case_name, pred_path, label_path))
        else:
            logger.warning(f"No matching label found for {case_name}")
    
    return matches


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_sample(
    case_name: str,
    pred_path: Path,
    label_path: Path,
    pred_folder: Path,
    num_bins: int = 15
) -> Dict[str, float]:
    """
    Evaluate a single sample.
    
    Args:
        case_name: Name of the case
        pred_path: Path to binary prediction
        label_path: Path to ground truth label
        pred_folder: Folder containing predictions (for logits/probs)
        num_bins: Number of bins for calibration metrics
        
    Returns:
        Dictionary of metrics for this sample
    """
    metrics = {'case_name': case_name}
    
    # Load binary prediction and label
    pred_data, pred_affine, pred_spacing = load_nifti(pred_path)
    label_data, label_affine, label_spacing = load_nifti(label_path)
    
    # Create mask for valid pixels (exclude -1 from prediction, which indicates outside brain)
    # Prediction values: -1 = outside brain (ignore), 0 = background, >0 = foreground classes
    valid_mask = (pred_data >= 0)
    num_excluded = (~valid_mask).sum()
    if num_excluded > 0:
        logger.info(f"  {case_name}: Excluding {num_excluded} pixels outside brain mask ({num_excluded/pred_data.size*100:.1f}%)")
    
    # Apply mask to create binary predictions only on valid regions
    pred_binary = ((pred_data > 0) & valid_mask).astype(np.uint8)
    label_binary = ((label_data > 0) & valid_mask).astype(np.uint8)
    
    # For metrics that need full arrays, we'll apply the mask differently
    # Store original label for volume calculations
    label_binary_full = (label_data > 0).astype(np.uint8)
    
    # Compute voxel volume
    voxel_volume = np.prod(label_spacing)
    
    # Check shape match
    if pred_data.shape != label_data.shape:
        logger.warning(f"Shape mismatch for {case_name}: pred {pred_data.shape} vs label {label_data.shape}")
        metrics['error'] = 'shape_mismatch'
        return metrics
    
    # Store number of valid pixels for reference
    metrics['num_valid_pixels'] = int(valid_mask.sum())
    metrics['num_excluded_pixels'] = int(num_excluded)
    
    # ==========================================================================
    # 1. Dice from binary mask (only on valid pixels)
    # ==========================================================================
    metrics['dice_binary'] = compute_dice_score(pred_binary, label_binary)
    
    # ==========================================================================
    # 2. Dice from argmax of logits (if available)
    # ==========================================================================
    logits_path = pred_folder / f"{case_name}_logits.nii.gz"
    if logits_path.exists():
        logits_data, _, _ = load_nifti(logits_path)
        # Logits are in (X, Y, Z, C) format from our inference script
        if logits_data.ndim == 4:
            pred_from_logits = ((logits_data.argmax(axis=-1) > 0) & valid_mask).astype(np.uint8)
        else:
            pred_from_logits = ((logits_data > 0) & valid_mask).astype(np.uint8)
        metrics['dice_from_logits'] = compute_dice_score(pred_from_logits, label_binary)
    else:
        metrics['dice_from_logits'] = np.nan
    
    # ==========================================================================
    # 3. Dice from argmax of probabilities (if available)
    # ==========================================================================
    probs_path = pred_folder / f"{case_name}_probabilities.nii.gz"
    if probs_path.exists():
        probs_data, _, _ = load_nifti(probs_path)
        # Probs are in (X, Y, Z, C) format from our inference script
        if probs_data.ndim == 4:
            pred_from_probs = ((probs_data.argmax(axis=-1) > 0) & valid_mask).astype(np.uint8)
        else:
            pred_from_probs = ((probs_data > 0) & valid_mask).astype(np.uint8)
        metrics['dice_from_probs'] = compute_dice_score(pred_from_probs, label_binary)
    else:
        metrics['dice_from_probs'] = np.nan
    
    # ==========================================================================
    # Additional metrics (using binary prediction on valid region only)
    # ==========================================================================
    metrics['iou'] = compute_iou_score(pred_binary, label_binary)
    metrics['sensitivity'] = compute_sensitivity(pred_binary, label_binary)
    metrics['specificity'] = compute_specificity(pred_binary, label_binary)
    metrics['precision'] = compute_precision(pred_binary, label_binary)
    
    # Volume metrics (use full label for target volume to get actual lesion volume)
    pred_vol, _, _ = compute_volume_difference(pred_binary, label_binary, voxel_volume)
    target_vol_full = float(label_binary_full.sum() * voxel_volume)
    vol_diff = pred_vol - target_vol_full
    metrics['predicted_volume_mm3'] = pred_vol
    metrics['target_volume_mm3'] = target_vol_full
    metrics['volume_difference_mm3'] = vol_diff
    metrics['volume_ratio'] = pred_vol / max(target_vol_full, 1e-5)
    
    # Hausdorff distance
    try:
        metrics['hd95'] = compute_hausdorff_distance_95(pred_binary, label_binary, label_spacing)
    except Exception as e:
        logger.warning(f"HD95 computation failed for {case_name}: {e}")
        metrics['hd95'] = np.nan
    
    # ==========================================================================
    # Calibration metrics (if logits available - uses metrics.py implementation)
    # ==========================================================================
    logits_path = pred_folder / f"{case_name}_logits.nii.gz"
    if logits_path.exists():
        try:
            # Load logits if not already loaded
            if 'logits_data' not in dir():
                logits_data, _, _ = load_nifti(logits_path)
            
            # Convert to PyTorch format expected by compute_calibration_metrics
            # Input: (X, Y, Z, C) numpy -> (1, C, X, Y, Z) torch
            if logits_data.ndim == 4:
                if logits_data.shape[-1] == 2:  # (X, Y, Z, C) format
                    logits_torch = torch.from_numpy(logits_data).float()
                    logits_torch = logits_torch.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, X, Y, Z)
                elif logits_data.shape[0] == 2:  # (C, X, Y, Z) format
                    logits_torch = torch.from_numpy(logits_data).float().unsqueeze(0)  # (1, C, X, Y, Z)
                else:
                    raise ValueError(f"Cannot determine logits format: {logits_data.shape}")
            else:
                raise ValueError(f"Expected 4D logits array, got shape {logits_data.shape}")
            
            # Convert targets to PyTorch format
            # Create target with -1 for invalid regions (outside brain mask)
            target_for_cal = label_data.copy().astype(np.int64)
            target_for_cal[~valid_mask] = -1  # Mark invalid regions
            target_torch = torch.from_numpy(target_for_cal).long().unsqueeze(0)  # (1, X, Y, Z)
            
            # Get number of classes
            num_classes = logits_torch.shape[1]
            
            # Use prepare_targets_with_mask from metrics.py
            valid_mask_torch, targets_clamped, num_valid_voxels = prepare_targets_with_mask(
                target_torch, num_classes, ignore_index=-1
            )
            
            # Compute calibration metrics using the training module's function
            cal_metrics = compute_calibration_metrics(
                logits_torch, target_torch, targets_clamped, 
                valid_mask_torch, num_valid_voxels,
                num_bins=num_bins, ignore_index=-1
            )
            metrics.update(cal_metrics)
        except Exception as e:
            logger.warning(f"Calibration metrics failed for {case_name}: {e}")
            import traceback
            traceback.print_exc()
            metrics['ece'] = np.nan
            metrics['mce'] = np.nan
            metrics['brier_score'] = np.nan
            metrics['nll'] = np.nan
    else:
        metrics['ece'] = np.nan
        metrics['mce'] = np.nan
        metrics['brier_score'] = np.nan
        metrics['nll'] = np.nan
    
    # Count lesion statistics (using masked predictions and full label)
    metrics['num_pred_voxels'] = int(pred_binary.sum())  # Predicted lesion voxels (within valid region)
    metrics['num_target_voxels'] = int(label_binary.sum())  # Target lesion voxels (within valid region)
    metrics['num_target_voxels_full'] = int(label_binary_full.sum())  # Target lesion voxels (full volume)
    metrics['has_lesion'] = int(label_binary_full.sum() > 0)  # Whether the case has any lesion
    
    return metrics


def evaluate_sample_wrapper(args: Tuple) -> Dict:
    """
    Wrapper for evaluate_sample to work with ProcessPoolExecutor.
    
    Args:
        args: Tuple of (case_name, pred_path, label_path, pred_folder, num_bins)
        
    Returns:
        Dictionary of metrics for this sample
    """
    case_name, pred_path, label_path, pred_folder, num_bins = args
    try:
        return evaluate_sample(case_name, pred_path, label_path, pred_folder, num_bins)
    except Exception as e:
        return {'case_name': case_name, 'error': str(e)}


def aggregate_metrics(df: pd.DataFrame) -> Dict:
    """
    Aggregate metrics across all samples.
    
    Args:
        df: DataFrame with per-sample metrics
        
    Returns:
        Dictionary with aggregated statistics
    """
    # Numeric columns to aggregate (exclude case_name and error)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    aggregated = {
        'num_samples': len(df),
        'num_samples_with_lesion': int(df['has_lesion'].sum()) if 'has_lesion' in df.columns else len(df),
        'metrics': {}
    }
    
    for col in numeric_cols:
        if col in ['has_lesion']:
            continue
            
        values = df[col].dropna()
        if len(values) == 0:
            continue
            
        aggregated['metrics'][col] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'median': float(values.median()),
            'min': float(values.min()),
            'max': float(values.max()),
            'count': int(len(values))
        }
    
    # Add summary of key metrics at top level for convenience
    key_metrics = ['dice_binary', 'dice_from_logits', 'dice_from_probs', 'iou', 
                   'sensitivity', 'precision', 'hd95', 'ece', 'nll']
    
    aggregated['summary'] = {}
    for metric in key_metrics:
        if metric in aggregated['metrics']:
            aggregated['summary'][metric] = {
                'mean': aggregated['metrics'][metric]['mean'],
                'std': aggregated['metrics'][metric]['std'],
                'median': aggregated['metrics'][metric]['median']
            }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation predictions against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=True,
        help='Folder containing prediction files'
    )
    
    parser.add_argument(
        '--labels', '-l',
        type=str,
        required=True,
        help='Folder containing ground truth label files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--pred_suffix',
        type=str,
        default='.nii.gz',
        help='Suffix for prediction files'
    )
    
    parser.add_argument(
        '--label_suffix',
        type=str,
        default='.nii.gz',
        help='Suffix for label files'
    )
    
    parser.add_argument(
        '--num_bins',
        type=int,
        default=15,
        help='Number of bins for ECE/MCE computation'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=12,
        help='Number of parallel workers (default: 12)'
    )
    
    args = parser.parse_args()
    
    pred_folder = Path(args.predictions)
    label_folder = Path(args.labels)
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    num_workers = args.num_workers
    
    # Find matching files
    logger.info(f"Finding matching files...")
    logger.info(f"  Predictions: {pred_folder}")
    logger.info(f"  Labels: {label_folder}")
    
    matches = find_matching_files(
        pred_folder, label_folder,
        args.pred_suffix, args.label_suffix
    )
    
    if len(matches) == 0:
        logger.error("No matching files found!")
        return
    
    logger.info(f"Found {len(matches)} matching samples")
    logger.info(f"Using {num_workers} parallel workers")
    
    # Prepare arguments for parallel processing
    eval_args = [
        (case_name, pred_path, label_path, pred_folder, args.num_bins)
        for case_name, pred_path, label_path in matches
    ]
    
    # Evaluate samples in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(evaluate_sample_wrapper, arg): arg[0] for arg in eval_args}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            case_name = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {case_name}: {e}")
                results.append({'case_name': case_name, 'error': str(e)})
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save per-sample CSV
    csv_path = output_folder / 'per_sample_metrics.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved per-sample metrics to: {csv_path}")
    
    # Aggregate and save JSON
    aggregated = aggregate_metrics(df)
    json_path = output_folder / 'aggregated_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"Saved aggregated metrics to: {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Number of samples: {aggregated['num_samples']}")
    print(f"Samples with lesions: {aggregated['num_samples_with_lesion']}")
    print()
    
    if 'summary' in aggregated:
        print("Key Metrics (mean ± std, median):")
        print("-"*60)
        for metric, values in aggregated['summary'].items():
            print(f"  {metric:20s}: {values['mean']:.4f} ± {values['std']:.4f} (median: {values['median']:.4f})")
    
    # Verify dice consistency
    print()
    print("Dice Verification (should be identical):")
    print("-"*60)
    for metric in ['dice_binary', 'dice_from_logits', 'dice_from_probs']:
        if metric in aggregated.get('metrics', {}):
            m = aggregated['metrics'][metric]
            print(f"  {metric:20s}: {m['mean']:.6f} ± {m['std']:.6f}")
    
    print("="*60)


if __name__ == "__main__":
    main()
