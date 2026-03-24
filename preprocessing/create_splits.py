"""
Create train/val/test splits for KiTS dataset.
Split ratios: test=50%, train=40%, val=10%
Balanced by: original spacing (axis 0) and tumor percentage
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def load_statistics(stats_path: str) -> dict:
    """Load the kits_statistics.json file."""
    with open(stats_path, 'r') as f:
        return json.load(f)

def extract_features_for_stratification(per_case_stats: list) -> list:
    """Extract features used for stratified splitting."""
    data = []
    for case in per_case_stats:
        case_id = case['case_id']
        
        # Extract spacing (axis 0 - the z/slice spacing which varies most)
        spacing_z = case['spacing'][0]
        
        # Calculate tumor percentage (label 2 is tumor in KiTS)
        label_counts = case['label_counts']
        total_voxels = sum(int(v) for v in label_counts.values())
        tumor_voxels = int(label_counts.get('2', 0))
        tumor_percentage = (tumor_voxels / total_voxels * 100) if total_voxels > 0 else 0
        
        # Also get kidney (label 1) percentage for reference
        kidney_voxels = int(label_counts.get('1', 0))
        kidney_percentage = (kidney_voxels / total_voxels * 100) if total_voxels > 0 else 0
        
        # Cyst presence
        has_cyst = case.get('has_cyst', False)
        
        data.append({
            'case_id': case_id,
            'spacing_z': spacing_z,
            'tumor_percentage': tumor_percentage,
            'kidney_percentage': kidney_percentage,
            'has_cyst': has_cyst,
            'total_voxels': total_voxels,
            'tumor_voxels': tumor_voxels
        })
    
    return data

def get_percentile(values: list, p: float) -> float:
    """Calculate percentile value."""
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]

def assign_bins(values: list, n_bins: int) -> list:
    """Assign values to quantile-based bins."""
    sorted_with_idx = sorted(enumerate(values), key=lambda x: x[1])
    n = len(values)
    bins = [0] * n
    for rank, (orig_idx, val) in enumerate(sorted_with_idx):
        bin_idx = min(int(rank * n_bins / n), n_bins - 1)
        bins[orig_idx] = bin_idx
    return bins

def create_stratification_bins(data: list, n_spacing_bins: int = 5, n_tumor_bins: int = 5) -> list:
    """Create combined stratification bins based on spacing and tumor percentage."""
    spacing_values = [d['spacing_z'] for d in data]
    tumor_values = [d['tumor_percentage'] for d in data]
    
    spacing_bins = assign_bins(spacing_values, n_spacing_bins)
    tumor_bins = assign_bins(tumor_values, n_tumor_bins)
    
    # Combine into stratification groups
    strat_groups = [f"{s}_{t}" for s, t in zip(spacing_bins, tumor_bins)]
    
    return strat_groups

def stratified_split(data: list, strat_groups: list, ratios: dict, random_state: int = 42) -> dict:
    """
    Perform stratified splitting of data.
    
    Args:
        data: List of case dictionaries
        strat_groups: List of stratification group labels
        ratios: Dict with 'train', 'val', 'test' ratios
        random_state: Random seed
    
    Returns:
        Dictionary with split assignments
    """
    random.seed(random_state)
    
    # Group indices by stratification group
    group_indices = defaultdict(list)
    for idx, group in enumerate(strat_groups):
        group_indices[group].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Split each stratification group proportionally
    for group, indices in group_indices.items():
        # Shuffle indices within group
        shuffled = indices.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_test = max(1, round(n * ratios['test']))
        n_val = max(1, round(n * ratios['val'])) if n > 2 else 0
        n_train = n - n_test - n_val
        
        # Handle edge cases
        if n == 1:
            # Single sample goes to test (largest set)
            test_indices.extend(shuffled)
        elif n == 2:
            # Two samples: one to test, one to train
            test_indices.append(shuffled[0])
            train_indices.append(shuffled[1])
        else:
            test_indices.extend(shuffled[:n_test])
            val_indices.extend(shuffled[n_test:n_test + n_val])
            train_indices.extend(shuffled[n_test + n_val:])
    
    splits = {
        'train': [data[i]['case_id'] for i in train_indices],
        'val': [data[i]['case_id'] for i in val_indices],
        'test': [data[i]['case_id'] for i in test_indices]
    }
    
    return splits, {'train': train_indices, 'val': val_indices, 'test': test_indices}

def verify_split_balance(data: list, splits: dict, split_indices: dict) -> None:
    """Print statistics to verify the split is balanced."""
    print("=" * 60)
    print("SPLIT VERIFICATION")
    print("=" * 60)
    
    for split_name in ['train', 'val', 'test']:
        indices = split_indices[split_name]
        split_data = [data[i] for i in indices]
        n_cases = len(split_data)
        percentage = n_cases / len(data) * 100
        
        print(f"\n{split_name.upper()} SET ({n_cases} cases, {percentage:.1f}%)")
        print("-" * 40)
        
        if n_cases == 0:
            print("  No cases in this split")
            continue
        
        # Spacing statistics
        spacing_values = [d['spacing_z'] for d in split_data]
        print(f"  Spacing Z-axis:")
        print(f"    Mean: {sum(spacing_values)/len(spacing_values):.3f}")
        print(f"    Min:  {min(spacing_values):.3f}")
        print(f"    Max:  {max(spacing_values):.3f}")
        
        # Spacing distribution
        spacing_dist = {'<=1.5': 0, '1.5-2.5': 0, '2.5-3.5': 0, '3.5-4.5': 0, '>4.5': 0}
        for s in spacing_values:
            if s <= 1.5:
                spacing_dist['<=1.5'] += 1
            elif s <= 2.5:
                spacing_dist['1.5-2.5'] += 1
            elif s <= 3.5:
                spacing_dist['2.5-3.5'] += 1
            elif s <= 4.5:
                spacing_dist['3.5-4.5'] += 1
            else:
                spacing_dist['>4.5'] += 1
        spacing_pct = {k: round(v/n_cases, 3) for k, v in spacing_dist.items()}
        print(f"    Distribution: {spacing_pct}")
        
        # Tumor percentage statistics
        tumor_values = [d['tumor_percentage'] for d in split_data]
        print(f"  Tumor Percentage:")
        print(f"    Mean: {sum(tumor_values)/len(tumor_values):.4f}%")
        print(f"    Min:  {min(tumor_values):.6f}%")
        print(f"    Max:  {max(tumor_values):.4f}%")
        
        # Cyst presence
        cyst_count = sum(1 for d in split_data if d['has_cyst'])
        cyst_ratio = cyst_count / n_cases * 100
        print(f"  Has Cyst: {cyst_ratio:.1f}%")

def save_splits(splits: dict, output_path: str) -> None:
    """Save the splits to a JSON file."""
    output = {
        'description': 'KiTS dataset train/val/test split',
        'ratios': {'train': 0.4, 'val': 0.1, 'test': 0.5},
        'stratified_by': ['spacing_z', 'tumor_percentage'],
        'splits': splits,
        'counts': {k: len(v) for k, v in splits.items()}
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSplits saved to: {output_path}")

def main():
    # Paths
    stats_path = Path(__file__).parent / 'kits_statistics.json'
    output_path = Path(__file__).parent / 'kits_splits.json'
    
    # Load statistics
    print("Loading dataset statistics...")
    stats = load_statistics(stats_path)
    
    # Extract features
    print(f"Processing {stats['aggregate']['num_cases']} cases...")
    data = extract_features_for_stratification(stats['per_case'])
    
    spacing_values = [d['spacing_z'] for d in data]
    tumor_values = [d['tumor_percentage'] for d in data]
    
    print(f"\nDataset Overview:")
    print(f"  Total cases: {len(data)}")
    print(f"  Spacing Z range: {min(spacing_values):.2f} - {max(spacing_values):.2f}")
    print(f"  Tumor % range: {min(tumor_values):.6f} - {max(tumor_values):.4f}")
    
    # Create stratification groups
    strat_groups = create_stratification_bins(data)
    
    # Create splits
    print("\nCreating stratified splits...")
    ratios = {'train': 0.4, 'val': 0.1, 'test': 0.5}
    splits, split_indices = stratified_split(data, strat_groups, ratios, random_state=42)
    
    # Verify balance
    verify_split_balance(data, splits, split_indices)
    
    # Save splits
    save_splits(splits, output_path)
    
    print("\n" + "=" * 60)
    print("Split creation complete!")
    print(f"  Train: {len(splits['train'])} cases (40%)")
    print(f"  Val:   {len(splits['val'])} cases (10%)")
    print(f"  Test:  {len(splits['test'])} cases (50%)")
    print("=" * 60)

if __name__ == '__main__':
    main()
