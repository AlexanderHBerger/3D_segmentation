"""
Comprehensive visualization of all augmentations on the same samples.

This script:
1. Loads 10 random samples from preprocessed nnUNet data
2. Applies each augmentation individually and in combination
3. Visualizes the results with multiple slices in all 3 directions
4. Shows both raw images and overlays with labels
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import torch
import torchio as tio
from typing import List, Tuple, Dict, Any, Optional
import random

from config import Config, AugmentationConfig
from transforms_torchio import (
    RotationScalingTransform,
    GammaTransformWithRetainStats,
    ContrastTransformWithPreserveRange,
    BrightnessTransform,
    IntensityShift
)


class AugmentationVisualizer:
    """Visualize individual augmentations on fixed samples"""
    
    def __init__(self, output_dir: str, num_samples: int = 1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.samples = []
        
    def load_samples(self, data_path: str):
        """Load random samples from preprocessed data"""
        data_path = Path(data_path)
        
        # Find all preprocessed files (using _data.npy pattern)
        image_files = sorted(list(data_path.glob("*_data.npy")))
        
        if len(image_files) == 0:
            raise ValueError(f"No preprocessed files found in {data_path}")
        
        print(f"Found {len(image_files)} image files")
        
        # Randomly select samples
        random.seed(42)  # Fixed seed for reproducibility
        selected_files = random.sample(image_files, min(self.num_samples, len(image_files)))
        
        print(f"Loading {len(selected_files)} samples...")
        
        for img_file in selected_files:
            # Load image (already has channel dimension: (C, H, W, D))
            image = np.load(img_file)
            
            # Load corresponding label (replace _data.npy with _seg.npy)
            label_file = img_file.parent / img_file.name.replace("_data.npy", "_seg.npy")
            if label_file.exists():
                label = np.load(label_file)
            else:
                print(f"Warning: No label found for {img_file.name}")
                label = np.zeros_like(image)
            
            # Convert to torch tensors (already have correct shape)
            image_tensor = torch.from_numpy(image).float()  # (C, H, W, D)
            label_tensor = torch.from_numpy(label).long().squeeze(0)  # Remove channel dim from label
            
            # Create TorchIO subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image_tensor),
                label=tio.LabelMap(tensor=label_tensor[np.newaxis, ...])  # Add back for TorchIO
            )
            
            self.samples.append({
                'subject': subject,
                'name': img_file.stem.replace('_data', ''),
                'original_image': image_tensor.clone(),
                'original_label': label_tensor.clone()
            })
        
        print(f"Loaded {len(self.samples)} samples successfully")
    
    def get_slice_indices_from_original(self, original_label: np.ndarray) -> Dict[str, int]:
        """Find meaningful slices in each direction from ORIGINAL label (before augmentation)"""
        h, w, d = original_label.shape
        slices = {}
        
        # Find slices with foreground in original label
        # Axial (along depth)
        axial_fg = np.sum(original_label, axis=(0, 1))
        axial_slices = np.where(axial_fg > 0)[0]
        if len(axial_slices) > 0:
            slices['axial'] = axial_slices[len(axial_slices) // 2]
        else:
            slices['axial'] = d // 2
        
        # Coronal (along width)
        coronal_fg = np.sum(original_label, axis=(0, 2))
        coronal_slices = np.where(coronal_fg > 0)[0]
        if len(coronal_slices) > 0:
            slices['coronal'] = coronal_slices[len(coronal_slices) // 2]
        else:
            slices['coronal'] = w // 2
        
        # Sagittal (along height)
        sagittal_fg = np.sum(original_label, axis=(1, 2))
        sagittal_slices = np.where(sagittal_fg > 0)[0]
        if len(sagittal_slices) > 0:
            slices['sagittal'] = sagittal_slices[len(sagittal_slices) // 2]
        else:
            slices['sagittal'] = h // 2
        
        return slices
    
    def visualize_augmentation(self, aug_name: str, transforms: List[Tuple[Any, str]], config: Config):
        """Visualize a specific augmentation with multiple strengths on first sample"""
        print(f"\nVisualizing augmentation: {aug_name}")
        
        # Use only the first sample
        sample = self.samples[0]
        
        # We'll create one figure with 4 columns (one per strength) and 2 rows (image, image+label)
        num_strengths = len(transforms)
        fig, axes = plt.subplots(2, num_strengths, figsize=(5 * num_strengths, 10))
        if num_strengths == 1:
            axes = axes.reshape(-1, 1)  # Ensure 2D array
        
        fig.suptitle(f'{aug_name} - Sample {sample["name"]}\nShowing {num_strengths} different strength levels', 
                    fontsize=16, fontweight='bold')
        
        # Process each strength level
        for strength_idx, (transform, strength_label) in enumerate(transforms):
            # Create augmented version starting from original
            subject_aug = tio.Subject(
                image=tio.ScalarImage(tensor=sample['original_image'].clone()),
                label=tio.LabelMap(tensor=sample['original_label'].clone()[np.newaxis, ...])
            )
            
            # Apply oversize-crop first (matching training pipeline)
            oversize_transform = tio.CropOrPad(
                target_shape=tuple(int(s * 1.15) for s in config.data.patch_size),
                padding_mode="minimum"
            )
            subject_aug = oversize_transform(subject_aug)
            
            # Apply the specific augmentation
            if transform is not None:
                subject_aug = transform(subject_aug)
            
            # Final crop to patch size
            crop_transform = tio.CropOrPad(
                target_shape=config.data.patch_size,
                padding_mode="minimum"
            )
            subject_aug = crop_transform(subject_aug)
            
            # Extract augmented data
            aug_image = subject_aug['image'].data.squeeze().numpy()  # (H, W, D)
            aug_label = subject_aug['label'].data.squeeze().numpy()  # (H, W, D)
            
            # Get axial slice from middle of label
            slices = self.get_slice_indices_from_original(aug_label)
            slice_idx = slices['axial']
            slice_idx = min(slice_idx, aug_image.shape[2]-1)
            
            aug_slice = aug_image[:, :, slice_idx]
            aug_label_slice = aug_label[:, :, slice_idx]
            
            # For the first strength (baseline), compute vmin/vmax
            if strength_idx == 0:
                vmin_global = aug_slice.min()
                vmax_global = aug_slice.max()
            
            # Row 1: Image only
            ax = axes[0, strength_idx]
            ax.imshow(aug_slice.T, cmap='gray', origin='lower', vmin=vmin_global, vmax=vmax_global)
            ax.set_title(f'{strength_label}\nmin={aug_slice.min():.2f}, max={aug_slice.max():.2f}')
            ax.axis('off')
            
            # Row 2: Image with label overlay
            ax = axes[1, strength_idx]
            ax.imshow(aug_slice.T, cmap='gray', origin='lower', vmin=vmin_global, vmax=vmax_global)
            label_mask = np.ma.masked_where(aug_label_slice.T == 0, aug_label_slice.T)
            ax.imshow(label_mask, cmap='Reds', origin='lower', alpha=0.5)
            fg_count = np.sum(aug_label_slice > 0)
            ax.set_title(f'+ Label\nFG pixels: {fg_count}')
            ax.axis('off')
        
        # Save figure
        output_path = self.output_dir / aug_name / f'{sample["name"]}_strength_comparison.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {output_path.relative_to(self.output_dir)}")


def main():
    """Main visualization script"""
    # Load configuration
    config = Config()
    aug_config = config.augmentation
    
    # Create visualizer
    output_dir = Path(config.output_dir) / "augmentation_visualization"
    visualizer = AugmentationVisualizer(output_dir=output_dir, num_samples=1)
    
    # Load samples
    visualizer.load_samples(config.data.data_path)
    
    # Define ALL augmentations to visualize with multiple strength levels
    # Format: (name, [(transform, strength_label), ...])
    augmentations: List[Tuple[str, List[Tuple[Any, str]]]] = []
    
    from math import sqrt
    
    # 1. Baseline (no augmentation at different "levels")
    augmentations.append((
        "0_baseline",
        [
            (None, "Original"),
            (None, "Original"),
            (None, "Original"),
            (None, "Original"),
        ]
    ))
    
    # 2. Rotation (varying angles)
    rotation_strengths = [
        ((0, 0, 0, 0, -5, 5), "Weak: Z=(-5°, 5°)"),
        ((0, 0, 0, 0, -15, 15), "Medium: Z=(-15°, 15°)"),
        ((0, 0, 0, 0, -30, 30), "Strong: Z=(-30°, 30°)"),
        ((-15, 15, -15, 15, -30, 30), "Very Strong: All axes"),
    ]
    rotation_transforms = [
        (RotationScalingTransform(
            rotation_range=angles,
            scale_range=(1.0, 1.0),
            rotation_p=1.0,
            scaling_p=0.0
        ), label)
        for angles, label in rotation_strengths
    ]
    augmentations.append(("1_rotation", rotation_transforms))
    
    # 3. Scaling (varying scale factors)
    scaling_transforms = [
        (RotationScalingTransform(
            rotation_range=(0, 0, 0, 0, 0, 0),
            scale_range=(0.95, 1.05),
            rotation_p=0.0,
            scaling_p=1.0
        ), "Weak: (0.95, 1.05)"),
        (RotationScalingTransform(
            rotation_range=(0, 0, 0, 0, 0, 0),
            scale_range=(0.85, 1.15),
            rotation_p=0.0,
            scaling_p=1.0
        ), "Medium: (0.85, 1.15)"),
        (RotationScalingTransform(
            rotation_range=(0, 0, 0, 0, 0, 0),
            scale_range=(0.7, 1.3),
            rotation_p=0.0,
            scaling_p=1.0
        ), "Strong: (0.7, 1.3)"),
        (RotationScalingTransform(
            rotation_range=(0, 0, 0, 0, 0, 0),
            scale_range=(0.5, 1.5),
            rotation_p=0.0,
            scaling_p=1.0
        ), "Very Strong: (0.5, 1.5)"),
    ]
    augmentations.append(("2_scaling", scaling_transforms))
    
    # 4. Elastic Deformation (varying displacement)
    elastic_transforms = [
        (tio.RandomElasticDeformation(
            num_control_points=(7, 7, 7),
            max_displacement=2,
            image_interpolation='linear',
            label_interpolation='nearest',
            p=1.0
        ), "Weak: max_disp=2"),
        (tio.RandomElasticDeformation(
            num_control_points=(7, 7, 7),
            max_displacement=5,
            image_interpolation='linear',
            label_interpolation='nearest',
            p=1.0
        ), "Medium: max_disp=5"),
        (tio.RandomElasticDeformation(
            num_control_points=(7, 7, 7),
            max_displacement=10,
            image_interpolation='linear',
            label_interpolation='nearest',
            p=1.0
        ), "Strong: max_disp=10"),
        (tio.RandomElasticDeformation(
            num_control_points=(7, 7, 7),
            max_displacement=20,
            image_interpolation='linear',
            label_interpolation='nearest',
            p=1.0
        ), "Very Strong: max_disp=20"),
    ]
    augmentations.append(("3_elastic_deformation", elastic_transforms))
    
    # 5. Gaussian Noise (varying variance)
    noise_transforms = [
        (tio.RandomNoise(mean=0, std=(0.01, 0.02), p=1.0), "Weak: std=(0.01, 0.02)"),
        (tio.RandomNoise(mean=0, std=(0.02, 0.05), p=1.0), "Medium: std=(0.02, 0.05)"),
        (tio.RandomNoise(mean=0, std=(0.05, 0.1), p=1.0), "Strong: std=(0.05, 0.1)"),
        (tio.RandomNoise(mean=0, std=(0.1, 0.2), p=1.0), "Very Strong: std=(0.1, 0.2)"),
    ]
    augmentations.append(("4_gaussian_noise", noise_transforms))
    
    # 6. Gaussian Blur (varying sigma)
    blur_transforms = [
        (tio.RandomBlur(std=(0.2, 0.5), p=1.0), "Weak: sigma=(0.2, 0.5)"),
        (tio.RandomBlur(std=(0.5, 1.0), p=1.0), "Medium: sigma=(0.5, 1.0)"),
        (tio.RandomBlur(std=(1.0, 1.5), p=1.0), "Strong: sigma=(1.0, 1.5)"),
        (tio.RandomBlur(std=(1.5, 2.0), p=1.0), "Very Strong: sigma=(1.5, 2.0)"),
    ]
    augmentations.append(("5_gaussian_blur", blur_transforms))
    
    # 7. Brightness (varying multipliers)
    brightness_transforms = [
        (tio.Lambda(BrightnessTransform((0.9, 1.1)), p=1.0, include=['image']), "Weak: (0.9, 1.1)"),
        (tio.Lambda(BrightnessTransform((0.75, 1.25)), p=1.0, include=['image']), "Medium: (0.75, 1.25)"),
        (tio.Lambda(BrightnessTransform((0.5, 1.5)), p=1.0, include=['image']), "Strong: (0.5, 1.5)"),
        (tio.Lambda(BrightnessTransform((0.3, 1.7)), p=1.0, include=['image']), "Very Strong: (0.3, 1.7)"),
    ]
    augmentations.append(("6_brightness", brightness_transforms))
    
    # 8. Contrast (varying ranges)
    contrast_transforms = [
        (tio.Lambda(ContrastTransformWithPreserveRange((0.9, 1.1), True), p=1.0, include=['image']), "Weak: (0.9, 1.1)"),
        (tio.Lambda(ContrastTransformWithPreserveRange((0.75, 1.25), True), p=1.0, include=['image']), "Medium: (0.75, 1.25)"),
        (tio.Lambda(ContrastTransformWithPreserveRange((0.5, 1.5), True), p=1.0, include=['image']), "Strong: (0.5, 1.5)"),
        (tio.Lambda(ContrastTransformWithPreserveRange((0.3, 2.0), True), p=1.0, include=['image']), "Very Strong: (0.3, 2.0)"),
    ]
    augmentations.append(("7_contrast", contrast_transforms))
    
    # 9. Low Resolution (varying downsampling)
    lowres_transforms = [
        (tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.0, 1.5), image_interpolation='linear', p=1.0), 
         "Weak: downsample=(1.0, 1.5)"),
        (tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.5, 2.0), image_interpolation='linear', p=1.0), 
         "Medium: downsample=(1.5, 2.0)"),
        (tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(2.0, 3.0), image_interpolation='linear', p=1.0), 
         "Strong: downsample=(2.0, 3.0)"),
        (tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(3.0, 4.0), image_interpolation='linear', p=1.0), 
         "Very Strong: downsample=(3.0, 4.0)"),
    ]
    augmentations.append(("8_simulate_low_res", lowres_transforms))
    
    # 10. Gamma (varying ranges)
    gamma_transforms = [
        (tio.Lambda(GammaTransformWithRetainStats((0.9, 1.1), 0.0, True), p=1.0, include=['image']), 
         "Weak: gamma=(0.9, 1.1)"),
        (tio.Lambda(GammaTransformWithRetainStats((0.7, 1.3), 0.0, True), p=1.0, include=['image']), 
         "Medium: gamma=(0.7, 1.3)"),
        (tio.Lambda(GammaTransformWithRetainStats((0.5, 2.0), 0.0, True), p=1.0, include=['image']), 
         "Strong: gamma=(0.5, 2.0)"),
        (tio.Lambda(GammaTransformWithRetainStats((0.3, 3.0), 0.0, True), p=1.0, include=['image']), 
         "Very Strong: gamma=(0.3, 3.0)"),
    ]
    augmentations.append(("9_gamma", gamma_transforms))
    
    # 11. Gamma with invert (fixed gamma, show invert effect)
    gamma_invert_transforms = [
        (tio.Lambda(GammaTransformWithRetainStats((0.7, 1.3), 0.0, True), p=1.0, include=['image']), 
         "No Invert: p=0.0"),
        (tio.Lambda(GammaTransformWithRetainStats((0.7, 1.3), 0.5, True), p=1.0, include=['image']), 
         "Sometimes Invert: p=0.5"),
        (tio.Lambda(GammaTransformWithRetainStats((0.7, 1.3), 1.0, True), p=1.0, include=['image']), 
         "Always Invert: p=1.0"),
        (tio.Lambda(GammaTransformWithRetainStats((0.5, 2.0), 1.0, True), p=1.0, include=['image']), 
         "Strong + Invert: p=1.0"),
    ]
    augmentations.append(("10_gamma_with_invert", gamma_invert_transforms))
    
    print(f"\nTotal augmentations to visualize: {len(augmentations)}")
    print("Augmentation list:")
    for name, transforms in augmentations:
        print(f"  - {name}: {len(transforms)} strength levels")
    
    # Visualize each augmentation
    for aug_name, transforms in augmentations:
        visualizer.visualize_augmentation(aug_name, transforms, config)
    
    print(f"\n✅ Visualization complete! Results saved to: {output_dir}")
    print(f"\nTo view results, check:")
    print(f"  {output_dir}/")
    
    # Create index HTML for easy viewing
    create_index_html(output_dir, augmentations, visualizer.num_samples)


def create_index_html(output_dir: Path, augmentations: List[Tuple[str, List[Tuple[Any, str]]]], num_samples: int):
    """Create an HTML index for easy browsing"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Augmentation Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        .aug-section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; }
        .strength-info { color: #888; font-size: 0.9em; margin: 5px 0 10px 0; font-family: monospace; }
        .sample-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 15px; margin: 15px 0; }
        .sample-card { border: 1px solid #ddd; padding: 10px; border-radius: 4px; }
        .sample-card img { width: 100%; height: auto; border-radius: 4px; }
        .sample-title { font-weight: bold; margin-bottom: 5px; }
    </style>
</head>
<body>
    <h1>Augmentation Visualization Results</h1>
    <p>Showing effects of each augmentation at 4 different strength levels on the same sample (axial view only).</p>
"""
    
    for aug_name, transforms in augmentations:
        strength_labels = [label for _, label in transforms]
        html += f'    <div class="aug-section">\n'
        html += f'        <h2>{aug_name.replace("_", " ").title()}</h2>\n'
        html += f'        <div class="strength-info">Strength levels: {", ".join(strength_labels)}</div>\n'
        html += f'        <div class="sample-grid">\n'
        
        aug_dir = output_dir / aug_name
        if aug_dir.exists():
            images = sorted(list(aug_dir.glob("*.png")))
            for img_path in images:
                rel_path = img_path.relative_to(output_dir)
                html += f'            <div class="sample-card">\n'
                html += f'                <div class="sample-title">{img_path.stem}</div>\n'
                html += f'                <img src="{rel_path}" alt="{img_path.stem}">\n'
                html += f'            </div>\n'
        
        html += f'        </div>\n'
        html += f'    </div>\n'
    
    html += """</body>
</html>"""
    
    index_path = output_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(html)
    
    print(f"\n📄 Created index.html: {index_path}")
    print(f"   Open this file in a browser to view all results")


if __name__ == "__main__":
    main()
