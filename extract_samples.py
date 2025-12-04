"""
Extract samples for visual inspection.

Creates folders with image, label, prediction, and probability maps for:
1. Samples with largest difference between runs
2. Samples with dice score close to zero

Usage:
    python extract_samples.py \
        --run1_eval /path/to/run1/evaluation \
        --run2_eval /path/to/run2/evaluation \
        --run1_pred /path/to/run1/inference \
        --run2_pred /path/to/run2/inference \
        --images /path/to/images \
        --labels /path/to/labels \
        --output /path/to/output
"""
import argparse
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def copy_sample_files(
    case_name: str,
    output_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    run1_pred_dir: Path,
    run2_pred_dir: Path,
    run1_name: str,
    run2_name: str
):
    """
    Copy all relevant files for a sample to the output directory.
    
    Args:
        case_name: Name of the case (without extension)
        output_dir: Directory to copy files to
        images_dir: Directory containing input images
        labels_dir: Directory containing ground truth labels
        run1_pred_dir: Directory containing run1 predictions
        run2_pred_dir: Directory containing run2 predictions
        run1_name: Name of run1
        run2_name: Name of run2
    """
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and copy image (might have _0000 suffix)
    image_patterns = [
        f"{case_name}_0000.nii.gz",
        f"{case_name}.nii.gz"
    ]
    for pattern in image_patterns:
        image_path = images_dir / pattern
        if image_path.exists():
            shutil.copy(image_path, case_dir / "image.nii.gz")
            break
    
    # Copy label
    label_path = labels_dir / f"{case_name}.nii.gz"
    if label_path.exists():
        shutil.copy(label_path, case_dir / "label.nii.gz")
    
    # Copy run1 files
    run1_pred = run1_pred_dir / f"{case_name}.nii.gz"
    run1_prob = run1_pred_dir / f"{case_name}_probabilities.nii.gz"
    run1_logits = run1_pred_dir / f"{case_name}_logits.nii.gz"
    
    if run1_pred.exists():
        shutil.copy(run1_pred, case_dir / f"pred_{run1_name}.nii.gz")
    if run1_prob.exists():
        shutil.copy(run1_prob, case_dir / f"prob_{run1_name}.nii.gz")
    if run1_logits.exists():
        shutil.copy(run1_logits, case_dir / f"logits_{run1_name}.nii.gz")
    
    # Copy run2 files
    run2_pred = run2_pred_dir / f"{case_name}.nii.gz"
    run2_prob = run2_pred_dir / f"{case_name}_probabilities.nii.gz"
    run2_logits = run2_pred_dir / f"{case_name}_logits.nii.gz"
    
    if run2_pred.exists():
        shutil.copy(run2_pred, case_dir / f"pred_{run2_name}.nii.gz")
    if run2_prob.exists():
        shutil.copy(run2_prob, case_dir / f"prob_{run2_name}.nii.gz")
    if run2_logits.exists():
        shutil.copy(run2_logits, case_dir / f"logits_{run2_name}.nii.gz")


def main():
    parser = argparse.ArgumentParser(
        description="Extract samples for visual inspection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--run1_eval',
        type=str,
        required=True,
        help='Path to run1 evaluation directory (containing per_sample_metrics.csv)'
    )
    
    parser.add_argument(
        '--run2_eval',
        type=str,
        required=True,
        help='Path to run2 evaluation directory (containing per_sample_metrics.csv)'
    )
    
    parser.add_argument(
        '--run1_pred',
        type=str,
        required=True,
        help='Path to run1 inference/predictions directory'
    )
    
    parser.add_argument(
        '--run2_pred',
        type=str,
        required=True,
        help='Path to run2 inference/predictions directory'
    )
    
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path to input images directory'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to ground truth labels directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for extracted samples'
    )
    
    parser.add_argument(
        '--run1_name',
        type=str,
        default='run1',
        help='Name for run1'
    )
    
    parser.add_argument(
        '--run2_name',
        type=str,
        default='run2',
        help='Name for run2'
    )
    
    parser.add_argument(
        '--top_n',
        type=int,
        default=15,
        help='Number of samples to extract for each category'
    )
    
    parser.add_argument(
        '--zero_threshold',
        type=float,
        default=0.05,
        help='Threshold for "close to zero" dice score'
    )
    
    args = parser.parse_args()
    
    # Load CSVs
    run1_csv = Path(args.run1_eval) / "per_sample_metrics.csv"
    run2_csv = Path(args.run2_eval) / "per_sample_metrics.csv"
    
    print(f"Loading {run1_csv}...")
    df1 = pd.read_csv(run1_csv)
    print(f"Loading {run2_csv}...")
    df2 = pd.read_csv(run2_csv)
    
    # Merge on case_name
    df1 = df1.rename(columns={col: f"{col}_{args.run1_name}" if col != 'case_name' else col 
                               for col in df1.columns})
    df2 = df2.rename(columns={col: f"{col}_{args.run2_name}" if col != 'case_name' else col 
                               for col in df2.columns})
    
    merged = pd.merge(df1, df2, on='case_name', how='outer')
    
    col1 = f"dice_binary_{args.run1_name}"
    col2 = f"dice_binary_{args.run2_name}"
    
    # Compute difference
    merged['diff'] = merged[col1] - merged[col2]
    merged['abs_diff'] = merged['diff'].abs()
    merged['max_dice'] = merged[[col1, col2]].max(axis=1)
    merged['min_dice'] = merged[[col1, col2]].min(axis=1)
    
    # Setup directories
    output_dir = Path(args.output)
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    run1_pred_dir = Path(args.run1_pred)
    run2_pred_dir = Path(args.run2_pred)
    
    # ==========================================================================
    # 1. Samples with largest difference
    # ==========================================================================
    largest_diff_dir = output_dir / "largest_difference"
    largest_diff_dir.mkdir(parents=True, exist_ok=True)
    
    largest_diff = merged.nlargest(args.top_n, 'abs_diff')
    
    print(f"\n📊 Extracting {args.top_n} samples with LARGEST DIFFERENCE...")
    print("-" * 80)
    
    # Save info CSV
    info_cols = ['case_name', col1, col2, 'diff', 'abs_diff']
    largest_diff[info_cols].to_csv(largest_diff_dir / "samples_info.csv", index=False)
    
    for _, row in tqdm(largest_diff.iterrows(), total=len(largest_diff), desc="Copying files"):
        case_name = row['case_name']
        winner = args.run1_name if row['diff'] > 0 else args.run2_name
        print(f"  {case_name}: {args.run1_name}={row[col1]:.4f}, {args.run2_name}={row[col2]:.4f}, diff={row['diff']:+.4f} ({winner} wins)")
        
        copy_sample_files(
            case_name=case_name,
            output_dir=largest_diff_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            run1_pred_dir=run1_pred_dir,
            run2_pred_dir=run2_pred_dir,
            run1_name=args.run1_name,
            run2_name=args.run2_name
        )
    
    # ==========================================================================
    # 2. Samples with dice close to zero (in both runs)
    # ==========================================================================
    near_zero_dir = output_dir / "near_zero_dice"
    near_zero_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to samples where BOTH runs have low dice
    near_zero = merged[merged['max_dice'] < args.zero_threshold].nsmallest(args.top_n, 'max_dice')
    
    if len(near_zero) < args.top_n:
        # If not enough samples with both runs near zero, take samples where at least one is near zero
        near_zero_any = merged[merged['min_dice'] < args.zero_threshold].nsmallest(args.top_n, 'min_dice')
        near_zero = near_zero_any
    
    print(f"\n❌ Extracting {len(near_zero)} samples with DICE CLOSE TO ZERO...")
    print("-" * 80)
    
    # Save info CSV
    near_zero[info_cols].to_csv(near_zero_dir / "samples_info.csv", index=False)
    
    for _, row in tqdm(near_zero.iterrows(), total=len(near_zero), desc="Copying files"):
        case_name = row['case_name']
        print(f"  {case_name}: {args.run1_name}={row[col1]:.4f}, {args.run2_name}={row[col2]:.4f}")
        
        copy_sample_files(
            case_name=case_name,
            output_dir=near_zero_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            run1_pred_dir=run1_pred_dir,
            run2_pred_dir=run2_pred_dir,
            run1_name=args.run1_name,
            run2_name=args.run2_name
        )
    
    print(f"\n✅ Done! Extracted samples saved to: {output_dir}")
    print(f"  - Largest difference: {largest_diff_dir}")
    print(f"  - Near zero dice: {near_zero_dir}")


if __name__ == "__main__":
    main()
