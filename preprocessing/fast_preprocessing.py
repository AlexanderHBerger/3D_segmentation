"""
Fast preprocessing pipeline following nnUNet methodology.

This script implements the core preprocessing steps:
1. Canonical orientation (RAS+)
2. Brain mask creation (connected component analysis on non-zero intensities)
3. Resampling to target spacing
4. Normalization using brain pixels only (AFTER resampling)
5. Cropping to brain region
6. Label mask creation with -1 for non-brain tissue

Saves as compressed numpy arrays for fast loading during training.
"""

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from scipy.ndimage import label as scipy_label, binary_fill_holes, binary_dilation
import logging

from config import DataConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Hardcoded preprocessing parameters (not in DataConfig)
INTENSITY_THRESHOLD = 0.0  # Threshold for detecting background tissue (0.0 for zero-background MRI)
BRAIN_MASK_DILATION_PIXELS = 3  # Dilate brain mask by N pixels to include border regions
DATA_INTERPOLATION_ORDER = 3  # Cubic interpolation for image data
SEG_INTERPOLATION_ORDER = 0  # Nearest neighbor for segmentation
DATA_DTYPE = np.float32  # Data type for image storage
SEG_DTYPE = np.int8  # Data type for segmentation storage

# Output format options
USE_COMPRESSION = False  # Save as .npz (compressed) vs .npy (raw numpy array)
SAVE_NIFTI_FILES = False  # Save NIfTI files for visual inspection
SAVE_PROPERTIES = False  # Save per-case properties to .npz file





def fill_holes_in_segmentation(seg: np.ndarray) -> np.ndarray:
    """
    Fill holes in segmentation mask to remove single pixel noise.
    
    For each unique label (except 0 and -1), perform binary hole filling.
    This removes isolated background pixels surrounded by the label.
    
    Args:
        seg: Segmentation mask of shape (X, Y, Z)
        
    Returns:
        Segmentation with holes filled
    """
    seg_filled = seg.copy()
    unique_labels = np.unique(seg)
    
    # Process each label separately (skip background 0 and ignore -1)
    for label in unique_labels:
        if label <= 0:
            continue
            
        # Create binary mask for this label
        label_mask = (seg == label)
        
        # Fill holes in this label
        label_filled = binary_fill_holes(label_mask)
        
        # Update segmentation (only fill where it was previously background)
        newly_filled = label_filled & ~label_mask
        seg_filled[newly_filled] = label
        
        if newly_filled.sum() > 0:
            logger.info(f"    Filled {newly_filled.sum()} pixels for label {label}")
    
    return seg_filled


def create_brain_mask(image: np.ndarray) -> np.ndarray:
    """
    Create brain mask using improved connected component analysis.
    
    Algorithm:
    1. Threshold image to detect background (intensities <= threshold)
    2. Find connected components of background regions
    3. Keep only the LARGEST connected component as background (air outside head)
    4. Everything else becomes foreground (brain tissue, CSF, ventricles, etc.)
    
    This is more robust than binary_fill_holes because it handles:
    - Small air pockets inside the brain
    - CSF and ventricles
    - Partial volume effects
    - Any enclosed regions
    
    WARNING: This assumes background pixels have near-zero intensity (typical for MRI).
    For clinical data with non-zero backgrounds, adjust INTENSITY_THRESHOLD constant.
    
    Args:
        image: Image data of shape (X, Y, Z)
        
    Returns:
        Brain mask of shape (X, Y, Z) with True for brain, False for background
    """
    # Get background mask (pixels below or at threshold)
    background_mask = image <= INTENSITY_THRESHOLD
    
    # Calculate percentage of background pixels
    background_percentage = background_mask.sum() / background_mask.size * 100
    
    # Warn if very few background pixels (suggests non-zero background)
    if background_percentage < 5:
        logger.warning(
            f"WARNING: Only {background_percentage:.1f}% of pixels are at or below threshold {INTENSITY_THRESHOLD}! "
            f"This suggests a non-zero background. Consider adjusting INTENSITY_THRESHOLD constant. "
            f"Image range: [{image.min():.2f}, {image.max():.2f}]"
        )
    
    # Find connected components of background regions
    labeled_background, num_components = scipy_label(background_mask)
    
    if num_components == 0:
        # No background found - entire image is foreground
        logger.warning("No background regions found! Using entire volume as brain.")
        brain_mask = np.ones_like(background_mask, dtype=bool)
        return brain_mask
    
    # Find the largest connected component (should be air outside head)
    component_sizes = np.bincount(labeled_background.ravel())
    
    # Component 0 is the unlabeled region, skip it
    if len(component_sizes) > 1:
        component_sizes[0] = 0
        largest_component_label = np.argmax(component_sizes)
        largest_component_size = component_sizes[largest_component_label]
    else:
        # Only one component (background)
        largest_component_label = 1 if num_components > 0 else 0
        largest_component_size = component_sizes[0] if len(component_sizes) > 0 else 0
    
    # Create brain mask: everything that is NOT the largest background component
    largest_background_component = (labeled_background == largest_component_label)
    brain_mask = ~largest_background_component
    
    brain_percentage = brain_mask.sum() / brain_mask.size * 100
    logger.info(
        f"  Brain mask: {brain_percentage:.1f}% of volume, "
        f"{num_components} background components, largest: {largest_component_size} voxels"
    )
    
    return brain_mask


def dilate_brain_mask(
    brain_mask: np.ndarray
) -> np.ndarray:
    """
    Dilate brain mask by N pixels in all directions.
    
    This ensures we include border regions and partial volume effects
    near the brain boundary, which may contain lesions or important features.
    
    Args:
        brain_mask: Binary brain mask of shape (X, Y, Z)
        
    Returns:
        Dilated brain mask
    """
    if BRAIN_MASK_DILATION_PIXELS <= 0:
        return brain_mask
    
    # Binary dilation with iterations = BRAIN_MASK_DILATION_PIXELS
    # This expands the mask by approximately BRAIN_MASK_DILATION_PIXELS in all directions
    dilated_mask = binary_dilation(brain_mask, iterations=BRAIN_MASK_DILATION_PIXELS)
    
    logger.info(f"  Dilated brain mask by {BRAIN_MASK_DILATION_PIXELS} pixels: "
                f"{brain_mask.sum()} -> {dilated_mask.sum()} voxels "
                f"({dilated_mask.sum() - brain_mask.sum()} added)")
    
    return dilated_mask


def get_bbox_from_mask(mask: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    """
    Get bounding box from binary mask.
    
    Args:
        mask: Binary mask of shape (X, Y, Z)
        
    Returns:
        Tuple of (min, max) for each dimension
    """
    bbox = []
    for axis in range(mask.ndim):
        # Get indices where mask is True along this axis
        nonzero_indices = np.where(mask.sum(axis=tuple(i for i in range(mask.ndim) if i != axis)) > 0)[0]
        
        if len(nonzero_indices) == 0:
            raise ValueError(f"No foreground pixels found along axis {axis}")
            
        bbox.append((int(nonzero_indices[0]), int(nonzero_indices[-1] + 1)))
    
    return tuple(bbox)


def crop_to_bbox(
    data: np.ndarray, 
    bbox: Tuple[Tuple[int, int], ...]
) -> np.ndarray:
    """
    Crop 3D data to bounding box.
    
    Args:
        data: Array of shape (X, Y, Z)
        bbox: Bounding box as tuple of (min, max) for each spatial dimension
        
    Returns:
        Cropped data
    """
    return data[
        bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]
    ]


def normalize_image(
    image: np.ndarray,
    brain_mask: np.ndarray
) -> np.ndarray:
    """
    Normalize image using z-score normalization.
    
    Always uses brain mask for normalization (following nnUNet).
    
    Args:
        image: Image data of shape (X, Y, Z)
        brain_mask: Brain mask of shape (X, Y, Z)
        
    Returns:
        Normalized image
    """
    # Use only brain pixels for statistics (following nnUNet and DataConfig.use_mask_for_norm)
    mask_data = image[brain_mask]
    
    # Z-score normalization
    mean = mask_data.mean()
    std = mask_data.std()
    
    if std < 1e-8:
        logger.warning(f"Image has std={std:.2e}, setting to 1.0")
        std = 1.0
    
    # Normalize all pixels
    normalized = (image - mean) / std
    
    return normalized.astype(np.float32)


def preprocess_case(
    image_path: Union[str, Path],
    seg_path: Optional[Union[str, Path]],
    output_dir: Union[str, Path],
    case_id: str,
    data_config: DataConfig
) -> Dict:
    """
    Preprocess a single case with simplified pipeline.
    
    Steps:
    1. Load image and segmentation
    2. Move to canonical orientation (RAS+) using as_closest_canonical
    3. Create brain mask (connected component analysis on non-zero intensities)
    4. Normalize using brain pixels (BEFORE resampling, following nnUNet)
    5. Resample to target spacing using resample_to_output
    6. Crop to brain region
    7. Create label mask with -1 for non-brain
    8. Save as compressed numpy arrays
    
    Args:
        image_path: Path to input image
        seg_path: Path to segmentation (can be None)
        output_dir: Output directory
        case_id: Case identifier
        data_config: Data configuration from config.py
        
    Returns:
        Dictionary with preprocessing metadata
    """
    logger.info(f"Processing {case_id}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    img_nib = nib.load(image_path)
    original_spacing = img_nib.header.get_zooms()[:3]  # (X, Y, Z) order from NIfTI
    original_shape = img_nib.shape
    
    if seg_path is not None:
        seg_nib = nib.load(seg_path)
        
        # Check for non-integer values in segmentation (e.g. from scaling factors)
        # This is critical because casting 0.999 to int gives 0, losing the mask
        seg_data = seg_nib.get_fdata()
        if not np.all(seg_data == np.round(seg_data)):
            logger.warning(f"WARNING: Segmentation {case_id} contains non-integer values! Rounding to nearest integer.")
            unique_vals = np.unique(seg_data)
            if len(unique_vals) < 20:
                logger.warning(f"  Unique values before rounding: {unique_vals}")
            else:
                logger.warning(f"  Unique values range before rounding: [{np.min(seg_data)}, {np.max(seg_data)}]")
                
            # Round data and create new NIfTI image to ensure subsequent steps use integers
            seg_data = np.round(seg_data)
            seg_nib = nib.Nifti1Image(seg_data, seg_nib.affine, seg_nib.header)
    else:
        seg_nib = None
    
    # Store original properties
    properties = {
        'original_spacing': original_spacing,
        'original_shape': original_shape,
        'original_affine': img_nib.affine.copy(),
    }

    logger.info(f"  Original shape: {original_shape}, spacing: {original_spacing}")
    
    # Step 1: Canonical orientation (RAS+)
    logger.info(f"  1. Canonical orientation (RAS+)")
    img_canonical = nib.as_closest_canonical(img_nib)
    if seg_nib is not None:
        seg_canonical = nib.as_closest_canonical(seg_nib)
    
    properties['shape_after_canonical'] = img_canonical.shape
    properties['spacing_after_canonical'] = img_canonical.header.get_zooms()[:3]
    
    # Step 2: Create brain mask BEFORE resampling (on clean data)
    logger.info(f"  2. Creating brain mask on original data (before resampling)")
    image_canonical_data = img_canonical.get_fdata().astype(np.float32)
    brain_mask_canonical = create_brain_mask(image_canonical_data)
    
    # Dilate brain mask to include border regions
    logger.info(f"  2b. Dilating brain mask by {BRAIN_MASK_DILATION_PIXELS} pixels")
    brain_mask_canonical = dilate_brain_mask(brain_mask_canonical)
    
    # Convert brain mask to NIfTI for resampling
    brain_mask_nib = nib.Nifti1Image(
        brain_mask_canonical.astype(np.uint8),
        affine=img_canonical.affine,
        header=img_canonical.header
    )
    
    # Step 4: Resample normalized image, segmentation, AND brain mask together
    logger.info(f"  4. Resample to target spacing {data_config.target_spacing}")
    
    # Check for anisotropy (nnUNet approach)
    current_spacing = img_canonical.header.get_zooms()[:3]
    if np.max(current_spacing) / np.min(current_spacing) > 3:
        logger.info("    Data is highly anisotropic. Using separate resampling for out-of-plane axis.")
        low_res_axis = np.argmax(current_spacing)

        logger.info(f"    Low resolution axis: {low_res_axis} (spacing {current_spacing[low_res_axis]:.2f})")
        
        # Create intermediate spacing: target for in-plane, original for out-of-plane
        intermediate_spacing = list(current_spacing)
        intermediate_spacing[low_res_axis] = data_config.target_spacing[low_res_axis]

        logger.info(f"    Intermediate spacing for resampling: {intermediate_spacing}")

        # 1. Resample out-of-plane with nearest neighbor
        img_intermediate = resample_to_output(
            img_canonical,
            voxel_sizes=intermediate_spacing,
            order=0  # Nearest neighbor for out-of-plane
        )

        # 2. Resample in-plane with high order
        img_resampled = resample_to_output(
            img_intermediate,
            voxel_sizes=data_config.target_spacing,
            order=DATA_INTERPOLATION_ORDER
        )
        
    else:
        img_resampled = resample_to_output(
            img_canonical, 
            voxel_sizes=data_config.target_spacing, 
            order=DATA_INTERPOLATION_ORDER
        )
    
    # Resample brain mask and segmentation to match the image grid exactly
    # This ensures pixel-perfect alignment even if image was resampled in two steps
    logger.info(f"    Resampling mask and segmentation to match image geometry")
    
    brain_mask_resampled = resample_from_to(
        brain_mask_nib,
        img_resampled,
        order=0  # Nearest neighbor for binary mask
    )
    
    if seg_nib is not None:
        seg_resampled = resample_from_to(
            seg_canonical, 
            img_resampled,
            order=SEG_INTERPOLATION_ORDER  # Nearest neighbor for segmentation
        )

    # Get numpy arrays
    image = img_resampled.get_fdata().astype(np.float32)
    brain_mask = brain_mask_resampled.get_fdata().astype(bool)
    
    # Step 4b: Normalize AFTER resampling
    logger.info(f"  4b. Normalizing using brain mask (after resampling)")
    image = normalize_image(image, brain_mask)
    
    if seg_nib is not None:
        seg = seg_resampled.get_fdata()
    else:
        seg = None
    
    properties['shape_after_resampling'] = image.shape
    properties['spacing_after_resampling'] = img_resampled.header.get_zooms()[:3]
    
    # Step 5: Crop to brain region
    logger.info(f"  5. Cropping to brain region")
    bbox = get_bbox_from_mask(brain_mask)
    
    image = crop_to_bbox(image, bbox)
    brain_mask = crop_to_bbox(brain_mask, bbox)
    
    if seg is not None:
        seg = crop_to_bbox(seg, bbox)
    
    properties['bbox_used_for_cropping'] = bbox
    properties['shape_after_cropping'] = image.shape
    
    # Step 6: Fill holes in segmentation to remove single pixel noise
    logger.info(f"  6. Filling holes in segmentation")
    if seg is not None:
        seg = fill_holes_in_segmentation(seg)
    
    # Step 7: Create label mask with -1 for non-brain tissue
    logger.info(f"  7. Creating label mask with -1 for non-brain")
    if seg is not None:
        # Mark pixels outside brain as -1
        seg_final = seg.copy()
        seg_final[(seg == 0) & (~brain_mask)] = -1
        
        # Convert to appropriate dtype
        if np.max(seg_final) <= 127 and np.min(seg_final) >= -128:
            seg_final = seg_final.astype(SEG_DTYPE)
        else:
            seg_final = seg_final.astype(np.int16)
    else:
        # No segmentation, create empty mask
        seg_final = np.zeros_like(image, dtype=SEG_DTYPE)
        seg_final[~brain_mask] = -1
    
    # Add channel dimension for consistency with training code
    image = image[np.newaxis, ...]
    seg_final = seg_final[np.newaxis, ...]
    
    # Step 8: Save as numpy arrays
    logger.info(f"  8. Saving preprocessed data")
    
    if USE_COMPRESSION:
        np.savez_compressed(
            output_dir / f"{case_id}_data.npz",
            data=image
        )
        np.savez_compressed(
            output_dir / f"{case_id}_seg.npz",
            seg=seg_final
        )
    else:
        np.save(output_dir / f"{case_id}_data.npy", image)
        np.save(output_dir / f"{case_id}_seg.npy", seg_final)
    
    # Optionally save properties
    if SAVE_PROPERTIES:
        np.savez(
            output_dir / f"{case_id}_properties.npz",
            **{k: v for k, v in properties.items() if isinstance(v, (np.ndarray, list, tuple, int, float, str))}
        )
    
    # Optionally save NIfTI files for inspection
    if SAVE_NIFTI_FILES:
        logger.info(f"  8b. Saving NIfTI files for inspection")
        
        # Create affine matrix for cropped data
        # We need to adjust the affine to account for the cropping offset
        cropped_affine = img_resampled.affine.copy()
        # Adjust translation based on bbox
        offset = np.array([bbox[0][0], bbox[1][0], bbox[2][0], 0])
        cropped_affine[:, 3] = img_resampled.affine @ np.append(offset[:3], 1)
        
        # Save preprocessed image (remove channel dimension for NIfTI)
        image_nifti = nib.Nifti1Image(
            image[0].astype(np.float32),  # Remove channel dimension
            affine=cropped_affine
        )
        nib.save(image_nifti, output_dir / f"{case_id}_image.nii.gz")
        logger.info(f"    Saved: {case_id}_image.nii.gz")
        
        # Save segmentation with -1 for background
        seg_nifti = nib.Nifti1Image(
            seg_final[0].astype(np.int8),  # Remove channel dimension
            affine=cropped_affine
        )
        nib.save(seg_nifti, output_dir / f"{case_id}_segmentation.nii.gz")
        logger.info(f"    Saved: {case_id}_segmentation.nii.gz")
    
    # Statistics
    logger.info(f"  Final shape: {image.shape}")
    logger.info(f"  Final spacing: {data_config.target_spacing}")
    logger.info(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
    if seg is not None:
        unique_labels = np.unique(seg_final)
        logger.info(f"  Unique labels: {unique_labels}")
        for label in unique_labels:
            count = (seg_final == label).sum()
            pct = count / seg_final.size * 100
            logger.info(f"    Label {label}: {count} voxels ({pct:.1f}%)")
    
    properties['final_shape'] = image.shape
    properties['final_spacing'] = data_config.target_spacing
    properties['image_range'] = (float(image.min()), float(image.max()))
    if seg is not None:
        properties['unique_labels'] = np.unique(seg_final).tolist()
    
    return properties

if __name__ == "__main__":
    # See test_preprocessing.py for example usage
    print("Use run_preprocessing.py for batch processing or test_preprocessing.py for testing")
    print("Example usage:")
    print("  from config import DataConfig")
    print("  data_config = DataConfig()")
    print("  preprocess_case(image_path, seg_path, output_dir, case_id, data_config)")
