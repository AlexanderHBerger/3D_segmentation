"""
Optimized data augmentation using TorchIO (replaces custom transforms)

TorchIO is specifically designed for 3D medical image augmentation and follows
nnUNet-style best practices exactly. This implementation matches the nnU-Net paper
specification for all augmentations.

Key advantages:
- Optimized C++/CUDA implementations for speed
- Built-in support for medical image formats (NIfTI, DICOM)
- Proper handling of intensity and spatial transforms
- Exact replication of nnU-Net augmentation strategy

nnU-Net Augmentation Pipeline (as per paper):
1. Rotation + Scaling (combined, single interpolation) - p=0.2 each
2. Elastic Deformation
3. Gamma Augmentation
4. Inverted Gamma
5. Gaussian Noise
6. Gaussian Blur
7. Brightness
8. Contrast
9. Simulate Low Resolution
10. Mirroring
"""
import torchio as tio
import torch
import numpy as np
from typing import Optional, Tuple
from math import sqrt
import time


# Picklable transform classes for multiprocessing compatibility
# These must be module-level classes that can be pickled

class RotationScalingTransform:
    """
    Combined rotation and scaling transform with independent probabilities.
    
    nnUNet applies rotation with p=0.2 and scaling with p=0.2 independently,
    giving: 64% none, 16% rotation only, 16% scaling only, 4% both.
    
    Uses -1 padding for labels to mark invalid regions.
    """
    def __init__(self, rotation_range, scale_range, rotation_p=0.2, scaling_p=0.2):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.rotation_p = rotation_p
        self.scaling_p = scaling_p
    
    def __call__(self, subject):
        # Randomly decide whether to apply rotation and/or scaling
        apply_rotation = np.random.random() < self.rotation_p
        apply_scaling = np.random.random() < self.scaling_p
        
        if not apply_rotation and not apply_scaling:
            return subject
        
        # Set parameters based on what to apply
        degrees = self.rotation_range if apply_rotation else (0, 0, 0, 0, 0, 0)
        scales = self.scale_range if apply_scaling else (1.0, 1.0)
        
        # Create transform with -1 padding for labels
        transform = tio.RandomAffine(
            scales=scales,
            degrees=degrees,
            translation=0,
            isotropic=True,
            image_interpolation='linear',
            label_interpolation='nearest',
            default_pad_value='mean',  # Use 0 for images (background value)
            default_pad_label=-1,  # Use -1 for labels (ignore)
            p=1.0,
            copy=False
        )
        
        return transform(subject)


class GammaTransformWithRetainStats:
    """
    Gamma transformation with retain_stats=True (matching nnUNet exactly).
    
    Implementation follows batchgeneratorsv2/transforms/intensity/gamma.py:
    1. Optionally invert image (multiply by -1)
    2. Store mean and std if retain_stats=True
    3. Apply gamma transformation (rescale to [0,1], apply power, rescale back)
    4. Restore original mean and std if retain_stats=True
    5. Optionally invert back
    """
    def __init__(self, gamma_range=(0.7, 1.5), p_invert_image=1.0, retain_stats=True):
        self.gamma_range = gamma_range
        self.p_invert_image = p_invert_image
        self.retain_stats = retain_stats
    
    def __call__(self, img):
        """
        Apply gamma transformation with statistics preservation.
        
        Args:
            img: torch.Tensor of shape (C, H, W, D)
        
        Returns:
            Transformed tensor with same statistics if retain_stats=True
        """
        # Sample gamma value
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        
        # Decide whether to invert
        invert = np.random.random() < self.p_invert_image
        
        # Process each channel
        for c in range(img.shape[0]):
            if invert:
                img[c] *= -1
            
            if self.retain_stats:
                # Store original statistics
                mean = torch.mean(img[c])
                std = torch.std(img[c])
            
            # Apply gamma transformation
            minm = torch.min(img[c])
            rnge = torch.max(img[c]) - minm
            img[c] = torch.pow(((img[c] - minm) / torch.clamp(rnge, min=1e-7)), gamma) * rnge + minm
            
            if self.retain_stats:
                # Restore original statistics
                mn_here = torch.mean(img[c])
                std_here = torch.std(img[c])
                img[c] -= mn_here
                img[c] *= (std / torch.clamp(std_here, min=1e-7))
                img[c] += mean
            
            if invert:
                img[c] *= -1
        
        return img


class ContrastTransformWithPreserveRange:
    """
    Contrast transformation for normalized images.
    
    For normalized brain images (mean≈0 in brain tissue):
    - Simply multiply all values by the contrast factor
    - Negative values become more negative, positive more positive
    - This enhances contrast around 0 (the brain tissue mean)
    """
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True):
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
    
    def __call__(self, img):
        """
        Apply contrast transformation.
        
        Args:
            img: torch.Tensor of shape (C, H, W, D)
        
        Returns:
            Transformed tensor with preserved range if preserve_range=True
        """
        multiplier = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        # Store original range if needed
        if self.preserve_range:
            min_val, max_val = img.min(), img.max()
        
        # Contrast: simple multiplication (no shift)
        # For normalized images, this enhances contrast around 0
        img = img * multiplier
        
        # Clamp to original range
        if self.preserve_range:
            img = torch.clamp(img, min=min_val, max=max_val)
        
        return img


class BrightnessTransform:
    """
    Brightness transformation for normalized images.
    
    For normalized images with negative values:
    1. Shift to positive range (subtract minimum)
    2. Multiply by brightness factor
    3. Shift back (add original minimum)
    
    This brightens/darkens the entire image uniformly.
    """
    def __init__(self, brightness_range=(0.7, 1.3)):
        self.brightness_range = brightness_range
    
    def __call__(self, img):
        """
        Apply brightness transformation.
        
        Args:
            img: torch.Tensor of shape (C, H, W, D)
        
        Returns:
            Transformed tensor
        """
        multiplier = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        
        # Process each channel
        for c in range(img.shape[0]):
            min_val = img[c].min()
            # Shift to positive, multiply, shift back
            img[c] = (img[c] - min_val) * multiplier + min_val
        
        return img
    
class OversizedCrop:
    """
    Oversized crop transform for training augmentation.
    
    This transform:
    1. If volume is smaller than target: Use CropOrPad to pad to target size
    2. If volume is larger: Random crop to oversized patch size
       - With foreground oversampling probability, centers on foreground
       - Otherwise, uniform random crop
    
    This ensures:
    - Small volumes are padded to the required size
    - Large volumes get diverse crops for training
    - Border artifacts are reduced by oversampling before augmentation
    """
    def __init__(
        self,
        target_size: Tuple[int, int, int],
        foreground_oversample_percent: float = 0.0,
        padding_mode: str = 'minimum'
    ):
        """
        Args:
            target_size: Target oversized patch size (z, y, x)
            foreground_oversample_percent: Probability of sampling foreground patches
            padding_mode: Padding mode for CropOrPad ('minimum', 'mean', 'constant')
        """
        self.target_size = target_size
        self.foreground_oversample_percent = foreground_oversample_percent
        self.padding_mode = padding_mode
    
    def __call__(self, subject):
        """
        Apply oversized crop or padding.
        
        Args:
            subject: TorchIO Subject with 'image' and 'label' keys
        
        Returns:
            Transformed subject
        """
        # Get current volume shape (excluding channel dimension)
        # TorchIO shape: (C, W, H, D) where W=width, H=height, D=depth
        current_shape_whd = np.array(subject['image'].shape[1:])  # (W, H, D)

        target_shape_whd = np.array(self.target_size)  # (W, H, D)
        
        # For small volumes: use CropOrPad (center crop/pad)
        if np.any(current_shape_whd < target_shape_whd):
            transform = tio.CropOrPad(
                target_shape=tuple(target_shape_whd.tolist()),
                padding_mode=self.padding_mode,
                copy=False
            )
            print("Using CropOrPad for small volume.", flush=True)
            return transform(subject)
        
        # For large volumes: random crop
        # Decide whether to use foreground or uniform sampling
        use_foreground = (
            self.foreground_oversample_percent > 0 and
            np.random.random() < self.foreground_oversample_percent
        )

        if use_foreground:
            # Sample from foreground
            start_w, start_h, start_d  = self._sample_foreground_crop(subject, current_shape_whd, target_shape_whd)
        else:
            # Sample uniformly
            start_w, start_h, start_d  = self._sample_uniform_crop(current_shape_whd, target_shape_whd)
        
        # Calculate how much to crop from each side
        # W dimension (width, index 0)
        w_ini = int(start_w)
        w_fin = int(current_shape_whd[0] - (start_w + target_shape_whd[0]))
        
        # H dimension (height, index 1)
        h_ini = int(start_h)
        h_fin = int(current_shape_whd[1] - (start_h + target_shape_whd[1]))
        
        # D dimension (depth, index 2)
        d_ini = int(start_d)
        d_fin = int(current_shape_whd[2] - (start_d + target_shape_whd[2]))
        
        # Ensure all crop values are non-negative (clamp to 0)
        w_ini = max(0, w_ini)
        w_fin = max(0, w_fin)
        h_ini = max(0, h_ini)
        h_fin = max(0, h_fin)
        d_ini = max(0, d_ini)
        d_fin = max(0, d_fin)

        # Apply crop
        new_subject = {}
        new_subject["image"] = subject['image'][w_ini:w_ini + target_shape_whd[0], h_ini:h_ini + target_shape_whd[1], d_ini:d_ini + target_shape_whd[2]]
        new_subject["label"] = subject['label'][w_ini:w_ini + target_shape_whd[0], h_ini:h_ini + target_shape_whd[1], d_ini:d_ini + target_shape_whd[2]]
    
        new_subject["case_id"] = subject['case_id']
        if 'foreground_coords' in subject:
                new_subject['foreground_coords'] = subject['foreground_coords']

        new_subject = tio.Subject(**new_subject)
        del subject  # Free memory

        return new_subject

    def _sample_uniform_crop(self, volume_shape: np.ndarray, target_shape: np.ndarray) -> np.ndarray:
        """
        Sample a random crop location uniformly.
        
        Args:
            volume_shape: Shape of the volume (W, H, D) in TorchIO ordering
            target_shape: Size of patch to crop (W, H, D)
        
        Returns:
            Crop coordinates [w_start, h_start, d_start] in (W, H, D) ordering
        """
        # Compute valid range for crop start (ensure crop fits in volume)
        valid_start = np.maximum(0, volume_shape - target_shape)
        
        # Sample random start location
        start_coords = np.array([
            np.random.randint(0, max(1, valid_start[i] + 1))
            for i in range(3)
        ])
        
        return start_coords
    
    def _sample_foreground_crop(
        self,
        subject,
        volume_shape: np.ndarray,
        target_shape: np.ndarray
    ) -> np.ndarray:
        """
        Sample a crop centered on foreground (nnUNet style).
        
        Args:
            subject: TorchIO Subject with label
            volume_shape: Shape of the volume (W, H, D) in TorchIO ordering
            target_shape: Size of patch to crop (W, H, D)
        
        Returns:
            Crop coordinates [w_start, h_start, d_start] in (W, H, D) ordering
        """
        # Check if precomputed foreground coordinates are available
        if 'foreground_coords' in subject:
            print("Using precomputed foreground coordinates for sampling.", flush=True)
            foreground_coords = subject['foreground_coords']
        else:
            print("Computing foreground coordinates for sampling.", flush=True)
            # Get label data
            label = subject['label'].data  # (C, W, H, D)
            
            # Find all foreground voxel coordinates
            foreground_mask = label[0] > 0  # Remove channel dimension
            foreground_coords = torch.nonzero(foreground_mask, as_tuple=False).numpy()
        
        if len(foreground_coords) == 0:
            # No foreground, fall back to uniform sampling
            return self._sample_uniform_crop(volume_shape, target_shape)
        
        # Sample a random foreground voxel
        random_idx = np.random.randint(0, len(foreground_coords))
        center_coord = foreground_coords[random_idx]
        
        # Compute crop start (center crop on foreground voxel)
        start_coords = center_coord - target_shape // 2
        
        # Ensure crop stays within volume bounds
        start_coords = np.maximum(0, start_coords)
        start_coords = np.minimum(start_coords, volume_shape - target_shape)
        
        return start_coords


class IdentityTransform:
    def __call__(self, x):
        print(f"Shape: {x.shape}, Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
        return x
    
print_stats = IdentityTransform()

def get_training_transforms(config, oversize_factor: float = 1.25) -> tio.Compose:
    """
    Create TorchIO training transforms exactly matching nnU-Net augmentation strategy.
    
    This implementation follows the nnU-Net trainer order precisely:
    0. Oversize-and-crop (to reduce border artifacts)
    1. SpatialTransform (rotation + scaling combined, no elastic deformation by default)
    2. GaussianNoise
    3. GaussianBlur
    4. MultiplicativeBrightness
    5. Contrast
    6. SimulateLowResolution
    7. Gamma with invert (p=0.1, p_invert_image=1)
    8. Gamma without invert (p=0.3, p_invert_image=0)
    9. MirrorTransform
    
    Args:
        config: Configuration object with augmentation parameters
        oversize_factor: Factor to oversample patches (e.g., 1.25 = 25% larger)
    
    Returns:
        Composed TorchIO transforms for training
    """
    transforms = []
    
    # 0. Oversize-Crop strategy (before augmentation to reduce border artifacts)
    # This handles both padding for small volumes and random cropping for large ones
    oversized_patch_size = tuple(int(s * oversize_factor) for s in config.data.patch_size)
    # transforms.append(
    #     OversizedCrop(
    #         target_size=oversized_patch_size,
    #         foreground_oversample_percent=config.training.oversample_foreground_percent,
    #         padding_mode='minimum'
    #     )
    # )
    
    # ===== SPATIAL TRANSFORMS =====
    
    # 1. SpatialTransform: Combined Rotation + Scaling (SINGLE interpolation for efficiency)
    # nnU-Net: p_rotation=0.2, p_scaling=0.2, p_elastic_deform=0
    if config.augmentation.rotation_prob > 0 or config.augmentation.scaling_prob > 0:
        rotation_range = (config.augmentation.rotation_x[0], config.augmentation.rotation_x[1],
                         config.augmentation.rotation_y[0], config.augmentation.rotation_y[1],
                         config.augmentation.rotation_z[0], config.augmentation.rotation_z[1])
        scale_range = config.augmentation.scale_range
        
        transforms.append(
            RotationScalingTransform(
                rotation_range=rotation_range,
                scale_range=scale_range,
                rotation_p=config.augmentation.rotation_prob,
                scaling_p=config.augmentation.scaling_prob
            )
        )
    
    # Elastic Deformation (disabled in nnU-Net: p_elastic_deform=0, but included for flexibility)
    if config.augmentation.elastic_deform_prob > 0:
        transforms.append(
            tio.RandomElasticDeformation(
                num_control_points=(7, 7, 7),
                max_displacement=config.augmentation.elastic_deform_sigma[1] / 10,
                image_interpolation='linear',
                label_interpolation='nearest',
                p=config.augmentation.elastic_deform_prob,
                copy=False
            )
        )
    
    # ===== INTENSITY TRANSFORMS (only applied to images, not labels) =====
    
    # 2. GaussianNoise - nnU-Net: apply_probability=0.1
    if config.augmentation.gaussian_noise_prob > 0:
        transforms.append(
            tio.RandomNoise(
                mean=0,
                std=(sqrt(config.augmentation.gaussian_noise_variance[0]), 
                     sqrt(config.augmentation.gaussian_noise_variance[1])),
                p=config.augmentation.gaussian_noise_prob,
                copy=False
            )
        )
    
    # 3. GaussianBlur - nnU-Net: apply_probability=0.2
    if config.augmentation.gaussian_blur_prob > 0:
        transforms.append(
            tio.RandomBlur(
                std=config.augmentation.gaussian_blur_sigma,
                p=config.augmentation.gaussian_blur_prob,
                copy=False
            )
        )
    
    # 4. MultiplicativeBrightness - nnU-Net: apply_probability=0.15
    if config.augmentation.brightness_prob > 0:
        brightness = BrightnessTransform(config.augmentation.brightness_range)
        transforms.append(
            tio.Lambda(
                brightness,
                p=config.augmentation.brightness_prob,
                include=['image'],
                copy=False
            )
        )
    
    # 5. Contrast - nnU-Net: apply_probability=0.15
    if config.augmentation.contrast_prob > 0:
        contrast = ContrastTransformWithPreserveRange(
            contrast_range=config.augmentation.contrast_range,
            preserve_range=config.augmentation.contrast_preserve_range
        )
        transforms.append(
            tio.Lambda(
                contrast,
                p=config.augmentation.contrast_prob,
                include=['image'],
                copy=False
            )
        )
    
    # 6. SimulateLowResolution - nnU-Net: apply_probability=0.25
    if config.augmentation.simulate_low_res_prob > 0:
        down_range = config.augmentation.low_res_scale_range
        downsampling_min = max(1.0, 1.0 / down_range[1])
        downsampling_max = 1.0 / down_range[0]
        transforms.append(
            tio.RandomAnisotropy(
                axes=(0, 1, 2),
                downsampling=(downsampling_min, downsampling_max),
                image_interpolation='linear',
                p=config.augmentation.simulate_low_res_prob,
                copy=False
            )
        )
    
    # 7. Gamma with invert - nnU-Net: apply_probability=0.1, p_invert_image=1
    if config.augmentation.gamma_prob > 0:
        gamma_with_invert = GammaTransformWithRetainStats(
            gamma_range=config.augmentation.gamma_range,
            p_invert_image=1.,
            retain_stats=config.augmentation.gamma_retain_stats
        )
        
        transforms.append(
            tio.Lambda(
                gamma_with_invert,
                p=config.augmentation.gamma_prob,
                include=['image'],
                copy=False
            )
        )
    
    # 8. Gamma without invert - nnU-Net: apply_probability=0.3, p_invert_image=0
    if config.augmentation.gamma_no_invert_prob > 0:
        gamma_no_invert = GammaTransformWithRetainStats(
            gamma_range=config.augmentation.gamma_range,
            p_invert_image=0.0,
            retain_stats=config.augmentation.gamma_retain_stats
        )
        
        transforms.append(
            tio.Lambda(
                gamma_no_invert,
                p=config.augmentation.gamma_no_invert_prob,
                include=['image'],
                copy=False
            )
        )
    
    # 9. MirrorTransform - nnU-Net applies at the end (p=0.5 per axis)
    if config.augmentation.mirror_prob > 0:
        transforms.append(
            tio.RandomFlip(
                axes=config.augmentation.mirror_axes,
                flip_probability=config.augmentation.mirror_prob,
                copy=False
            )
        )
    
    # Note: Final center crop to target patch size is done in the Dataset class
    # after all augmentations to remove border artifacts
    
    print(f"Training augmentation pipeline created:")
    print(f"  - Oversized patch size: {oversized_patch_size} (oversize factor: {oversize_factor})")
    print(f"  - Foreground oversampling: {config.training.oversample_foreground_percent*100:.0f}%")
    print(f"  - Number of augmentation transforms: {len(transforms)}")
    
    # Compose all transforms
    return tio.Compose(transforms)


def get_validation_transforms(config) -> Optional[tio.Compose]:
    """
    Create minimal validation transforms (no augmentation).
    
    For validation, no transforms are needed since the final center crop/pad
    is done in the Dataset class.
    
    Args:
        config: Configuration object
    
    Returns:
        None (no transforms needed for validation, final crop done in Dataset)
    """
    # Note: Final center crop to target patch size is done in the Dataset class
    return None


def get_inference_transforms(config) -> Optional[tio.Compose]:
    """
    Create inference transforms (test-time augmentation if needed).
    
    Args:
        config: Configuration object
    
    Returns:
        Optional transforms for inference
    """
    # For inference, typically no transforms
    # Could add test-time augmentation here if desired
    return None


# Convenience function for creating TorchIO subjects from numpy arrays
def create_subject(image_data, label_data=None, image_path=None, label_path=None):
    """
    Create a TorchIO Subject from data.
    
    Args:
        image_data: Numpy array or path to image
        label_data: Numpy array or path to label (optional)
        image_path: Path to image file (for metadata)
        label_path: Path to label file (for metadata)
    
    Returns:
        tio.Subject
    """
    subject_dict = {}
    
    if image_path is not None:
        subject_dict['image'] = tio.ScalarImage(image_path)
    else:
        subject_dict['image'] = tio.ScalarImage(tensor=image_data)
    
    if label_data is not None:
        if label_path is not None:
            subject_dict['label'] = tio.LabelMap(label_path)
        else:
            subject_dict['label'] = tio.LabelMap(tensor=label_data)
    
    return tio.Subject(**subject_dict)
