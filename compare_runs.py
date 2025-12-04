"""
Compare evaluation results between two runs.

Identifies:
1. Samples with extremely good dice scores
2. Samples with extremely bad dice scores  
3. Samples with large differences between runs

Usage:
    python compare_runs.py \
        --run1 /path/to/run1/evaluation/per_sample_metrics.csv \
        --run2 /path/to/run2/evaluation/per_sample_metrics.csv \
        --run1_name iygryek2 \
        --run2_name q5524wpi
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_and_merge(csv1: Path, csv2: Path, name1: str, name2: str) -> pd.DataFrame:
    """Load two CSVs and merge on case_name."""
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    
    # Rename columns to distinguish runs
    df1 = df1.rename(columns={col: f"{col}_{name1}" if col != 'case_name' else col 
                               for col in df1.columns})
    df2 = df2.rename(columns={col: f"{col}_{name2}" if col != 'case_name' else col 
                               for col in df2.columns})
    
    # Merge on case_name
    merged = pd.merge(df1, df2, on='case_name', how='outer')
    
    return merged


def analyze_runs(merged: pd.DataFrame, name1: str, name2: str, 
                 metric: str = 'dice_binary',
                 top_n: int = 20,
                 diff_threshold: float = 0.1) -> dict:
    """
    Analyze differences between runs.
    
    Args:
        merged: Merged DataFrame with both runs
        name1: Name of first run
        name2: Name of second run
        metric: Metric to analyze (default: dice_binary)
        top_n: Number of top/bottom samples to show
        diff_threshold: Threshold for "large" difference
        
    Returns:
        Dictionary with analysis results
    """
    col1 = f"{metric}_{name1}"
    col2 = f"{metric}_{name2}"
    
    # Check columns exist
    if col1 not in merged.columns or col2 not in merged.columns:
        print(f"Warning: Columns {col1} or {col2} not found")
        print(f"Available columns: {merged.columns.tolist()}")
        return {}
    
    # Compute difference
    merged['diff'] = merged[col1] - merged[col2]
    merged['abs_diff'] = merged['diff'].abs()
    
    results = {
        'metric': metric,
        'run1': name1,
        'run2': name2,
    }
    
    # Overall statistics
    results['stats'] = {
        f'{name1}_mean': merged[col1].mean(),
        f'{name1}_std': merged[col1].std(),
        f'{name1}_median': merged[col1].median(),
        f'{name2}_mean': merged[col2].mean(),
        f'{name2}_std': merged[col2].std(),
        f'{name2}_median': merged[col2].median(),
        'mean_diff': merged['diff'].mean(),
        'std_diff': merged['diff'].std(),
    }
    
    # Best samples (highest dice in either run)
    merged['max_dice'] = merged[[col1, col2]].max(axis=1)
    merged['min_dice'] = merged[[col1, col2]].min(axis=1)
    
    # Extremely good samples (both runs have high dice)
    good_samples = merged.nlargest(top_n, 'min_dice')[
        ['case_name', col1, col2, 'diff']
    ].copy()
    results['extremely_good'] = good_samples
    
    # Extremely bad samples (both runs have low dice)
    bad_samples = merged.nsmallest(top_n, 'max_dice')[
        ['case_name', col1, col2, 'diff']
    ].copy()
    results['extremely_bad'] = bad_samples
    
    # Largest positive difference (run1 >> run2)
    run1_better = merged.nlargest(top_n, 'diff')[
        ['case_name', col1, col2, 'diff']
    ].copy()
    results['run1_much_better'] = run1_better
    
    # Largest negative difference (run2 >> run1)
    run2_better = merged.nsmallest(top_n, 'diff')[
        ['case_name', col1, col2, 'diff']
    ].copy()
    results['run2_much_better'] = run2_better
    
    # Samples with large absolute difference
    large_diff = merged[merged['abs_diff'] > diff_threshold].sort_values('abs_diff', ascending=False)[
        ['case_name', col1, col2, 'diff']
    ].copy()
    results['large_difference'] = large_diff
    
    return results


def print_results(results: dict, name1: str, name2: str):
    """Pretty print the analysis results."""
    metric = results.get('metric', 'dice_binary')
    
    print("\n" + "="*80)
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"Metric: {metric}")
    print("="*80)
    
    # Overall statistics
    stats = results.get('stats', {})
    print("\n📊 OVERALL STATISTICS")
    print("-"*80)
    print(f"  {name1:20s}: mean={stats.get(f'{name1}_mean', 0):.4f} ± {stats.get(f'{name1}_std', 0):.4f}, median={stats.get(f'{name1}_median', 0):.4f}")
    print(f"  {name2:20s}: mean={stats.get(f'{name2}_mean', 0):.4f} ± {stats.get(f'{name2}_std', 0):.4f}, median={stats.get(f'{name2}_median', 0):.4f}")
    print(f"  Mean difference    : {stats.get('mean_diff', 0):.4f} ± {stats.get('std_diff', 0):.4f}")
    
    col1 = f"{metric}_{name1}"
    col2 = f"{metric}_{name2}"
    
    # Extremely good samples
    print("\n✅ EXTREMELY GOOD SAMPLES (high dice in both runs)")
    print("-"*80)
    good = results.get('extremely_good', pd.DataFrame())
    if not good.empty:
        for _, row in good.head(10).iterrows():
            print(f"  {row['case_name']:50s} | {name1}: {row[col1]:.4f} | {name2}: {row[col2]:.4f} | diff: {row['diff']:+.4f}")
    
    # Extremely bad samples
    print("\n❌ EXTREMELY BAD SAMPLES (low dice in both runs)")
    print("-"*80)
    bad = results.get('extremely_bad', pd.DataFrame())
    if not bad.empty:
        for _, row in bad.head(10).iterrows():
            print(f"  {row['case_name']:50s} | {name1}: {row[col1]:.4f} | {name2}: {row[col2]:.4f} | diff: {row['diff']:+.4f}")
    
    # Run1 much better
    print(f"\n🔵 {name1.upper()} MUCH BETTER (largest positive difference)")
    print("-"*80)
    r1_better = results.get('run1_much_better', pd.DataFrame())
    if not r1_better.empty:
        for _, row in r1_better.head(10).iterrows():
            print(f"  {row['case_name']:50s} | {name1}: {row[col1]:.4f} | {name2}: {row[col2]:.4f} | diff: {row['diff']:+.4f}")
    
    # Run2 much better
    print(f"\n🟢 {name2.upper()} MUCH BETTER (largest negative difference)")
    print("-"*80)
    r2_better = results.get('run2_much_better', pd.DataFrame())
    if not r2_better.empty:
        for _, row in r2_better.head(10).iterrows():
            print(f"  {row['case_name']:50s} | {name1}: {row[col1]:.4f} | {name2}: {row[col2]:.4f} | diff: {row['diff']:+.4f}")
    
    # Large differences
    print(f"\n⚠️  SAMPLES WITH LARGE ABSOLUTE DIFFERENCE (>0.1)")
    print("-"*80)
    large = results.get('large_difference', pd.DataFrame())
    if not large.empty:
        print(f"  Total: {len(large)} samples")
        for _, row in large.head(20).iterrows():
            winner = name1 if row['diff'] > 0 else name2
            print(f"  {row['case_name']:50s} | {name1}: {row[col1]:.4f} | {name2}: {row[col2]:.4f} | diff: {row['diff']:+.4f} ({winner} wins)")
    else:
        print("  No samples with difference > 0.1")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results between two runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--run1', '-r1',
        type=str,
        required=True,
        help='Path to first run CSV (per_sample_metrics.csv)'
    )
    
    parser.add_argument(
        '--run2', '-r2',
        type=str,
        required=True,
        help='Path to second run CSV (per_sample_metrics.csv)'
    )
    
    parser.add_argument(
        '--run1_name', '-n1',
        type=str,
        default='run1',
        help='Name for first run'
    )
    
    parser.add_argument(
        '--run2_name', '-n2',
        type=str,
        default='run2',
        help='Name for second run'
    )
    
    parser.add_argument(
        '--metric', '-m',
        type=str,
        default='dice_binary',
        help='Metric to analyze'
    )
    
    parser.add_argument(
        '--top_n', '-n',
        type=int,
        default=20,
        help='Number of top/bottom samples to show'
    )
    
    parser.add_argument(
        '--diff_threshold', '-t',
        type=float,
        default=0.1,
        help='Threshold for "large" difference'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output CSV with merged comparison (optional)'
    )
    
    args = parser.parse_args()
    
    # Load and merge
    print(f"Loading {args.run1}...")
    print(f"Loading {args.run2}...")
    
    merged = load_and_merge(
        Path(args.run1), Path(args.run2),
        args.run1_name, args.run2_name
    )
    
    print(f"Merged {len(merged)} samples")
    
    # Analyze
    results = analyze_runs(
        merged, args.run1_name, args.run2_name,
        metric=args.metric,
        top_n=args.top_n,
        diff_threshold=args.diff_threshold
    )
    
    # Print results
    print_results(results, args.run1_name, args.run2_name)
    
    # Save merged CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"\nSaved merged comparison to: {output_path}")


if __name__ == "__main__":
    main()
