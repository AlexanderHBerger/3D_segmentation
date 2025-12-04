"""
Visualize preprocessed data to verify correctness.

For each sample, creates one 2D image per lesion showing:
- Left: Pure axial slice (MRI image)
- Right: Slice with mask overlay (alpha=0.5)

The slice shown is the center slice of each lesion (maximum lesion area).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
import argparse
from tqdm import tqdm


def find_individual_lesions(seg_volume, label=1):
    """
    Find individual lesions using connected component analysis.
    
    Args:
        seg_volume: Segmentation volume (C, D, H, W) or (D, H, W)
        label: Lesion label to find (default 1)
    
    Returns:
        List of tuples (lesion_id, best_slice_idx, lesion_volume, lesion_mask) for each lesion
    """
    # Remove channel dimension if present
    if seg_volume.ndim == 4:
        seg_volume = seg_volume[0]
    
    # Find lesions
    lesion_mask = (seg_volume == label)
    
    if not lesion_mask.any():
        return []
    
    # Connected component analysis to separate individual lesions
    labeled_lesions, num_lesions = ndimage.label(lesion_mask)
    
    lesions_info = []
    for lesion_id in range(1, num_lesions + 1):
        # Get mask for this specific lesion
        single_lesion_mask = (labeled_lesions == lesion_id)
        
        # Find best axial slice (with maximum lesion area)
        best_slice_idx = None
        max_area = 0
        for slice_idx in range(seg_volume.shape[2]):
            lesion_area = single_lesion_mask[:, :, slice_idx].sum()
            if lesion_area > max_area:
                max_area = lesion_area
                best_slice_idx = slice_idx
        
        # Calculate total lesion volume
        lesion_volume = single_lesion_mask.sum()
        
        lesions_info.append((lesion_id, best_slice_idx, lesion_volume, single_lesion_mask))
    
    # Sort by volume (largest first)
    lesions_info.sort(key=lambda x: x[2], reverse=True)
    
    return lesions_info


def create_overlay(image_slice, mask_slice, alpha=0.5, mask_color='red'):
    """
    Create overlay of mask on image.
    
    Args:
        image_slice: 2D image array
        mask_slice: 2D binary mask array
        alpha: Transparency for mask overlay
        mask_color: Color for mask ('red', 'green', 'blue', 'yellow')
    
    Returns:
        RGB image with overlay
    """
    # Normalize image to [0, 1]
    image_normalized = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
    
    # Create RGB image (grayscale)
    rgb_image = np.stack([image_normalized] * 3, axis=-1)
    
    # Create colored mask
    if mask_color == 'red':
        mask_rgb = np.array([1.0, 0.0, 0.0])
    elif mask_color == 'green':
        mask_rgb = np.array([0.0, 1.0, 0.0])
    elif mask_color == 'blue':
        mask_rgb = np.array([0.0, 0.0, 1.0])
    elif mask_color == 'yellow':
        mask_rgb = np.array([1.0, 1.0, 0.0])
    else:
        mask_rgb = np.array([1.0, 0.0, 0.0])  # Default to red
    
    # Apply overlay where mask is True
    mask_bool = mask_slice > 0
    for c in range(3):
        rgb_image[mask_bool, c] = (1 - alpha) * rgb_image[mask_bool, c] + alpha * mask_rgb[c]
    
    return rgb_image


def visualize_sample(data_path, seg_path, output_dir, case_id):
    """
    Create visualizations for all lesions in a single sample.
    
    Args:
        data_path: Path to data .npy file
        seg_path: Path to segmentation .npy file
        output_dir: Directory to save output images
        case_id: Case identifier for title
    
    Returns:
        Number of lesion visualizations created
    """
    # Load data
    data = np.load(data_path)
    seg = np.load(seg_path)
    
    # Remove channel dimension
    if data.ndim == 4:
        data = data[0]
    if seg.ndim == 4:
        seg = seg[0]
    
    # Find individual lesions
    lesions_info = find_individual_lesions(seg, label=1)
    
    if not lesions_info:
        print(f"  WARNING: No lesions found in {case_id}")
        return 0
    
    # Create visualization for each lesion
    num_visualizations = 0
    for lesion_id, best_slice_idx, lesion_volume, lesion_mask in lesions_info:
        # Extract the axial slice (axis 2)
        image_slice = data[:, :, best_slice_idx]
        mask_slice = lesion_mask[:, :, best_slice_idx].astype(float)
        
        # Create overlay
        overlay = create_overlay(image_slice, mask_slice, alpha=0.5, mask_color='red')
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot pure image
        axes[0].imshow(image_slice, cmap='gray', origin='lower')
        axes[0].set_title(f'MRI Image (Slice {best_slice_idx})')
        axes[0].axis('off')
        
        # Plot overlay
        axes[1].imshow(overlay, origin='lower')
        axes[1].set_title(f'Overlay (Lesion {lesion_id}: {lesion_volume} voxels)')
        axes[1].axis('off')
        
        # Add overall title
        fig.suptitle(f'{case_id} - Lesion {lesion_id}/{len(lesions_info)}\nShape: {data.shape}, Slice: {best_slice_idx}/{data.shape[2]}', 
                     fontsize=12, y=0.98)
        
        # Create output path with lesion ID
        output_path = output_dir / f"{case_id}_lesion{lesion_id:02d}_visualization.png"
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        num_visualizations += 1
    
    return num_visualizations


def main():
    parser = argparse.ArgumentParser(description='Visualize preprocessed data')
    parser.add_argument('--data_dir', type=str, 
                        default='/ministorage/ahb/data/nnUNet_preprocessed/Dataset017_SmallBrats_Fast',
                        help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, 
                        default='/ministorage/ahb/scratch/segmentation_model/visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None = all)')
    parser.add_argument('--case_ids', nargs='+', default=None,
                        help='Specific case IDs to process (e.g., BRATS_Mets_BraTS-MET-00003-000)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data directory
    data_dir = Path(args.data_dir)
    
    # Find all data files
    data_files = sorted(data_dir.glob('*_data.npy'))
    
    if args.case_ids:
        # Filter to specific case IDs
        data_files = [f for f in data_files if any(case_id in f.stem for case_id in args.case_ids)]
        print(f"Processing {len(data_files)} specified cases")
    else:
        print(f"Found {len(data_files)} cases in {data_dir}")
    
    if args.max_samples:
        data_files = data_files[:args.max_samples]
        print(f"Processing first {len(data_files)} samples")
    
    # Process each sample
    total_visualizations = 0
    samples_processed = 0
    for data_file in tqdm(data_files, desc='Processing'):
        # Get case ID
        case_id = data_file.stem.replace('_data', '')
        
        # Get corresponding segmentation file
        seg_file = data_file.parent / f"{case_id}_seg.npy"
        
        if not seg_file.exists():
            print(f"  WARNING: Segmentation file not found for {case_id}")
            continue
        
        # Generate visualizations for all lesions in this sample
        num_lesions = visualize_sample(data_file, seg_file, output_dir, case_id)
        if num_lesions > 0:
            total_visualizations += num_lesions
            samples_processed += 1
    
    print(f"\nCompleted: {samples_processed} samples processed, {total_visualizations} lesion visualizations created")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
