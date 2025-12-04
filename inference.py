"""
Inference script for MedNeXt segmentation model.

Implements sliding window inference following nnUNet methodology but adapted
to our specific preprocessing pipeline (normalize before resampling).

Usage:
    python inference.py \
        --input_folder /path/to/images \
        --output_folder /path/to/predictions \
        --checkpoint /path/to/checkpoint.pth \
        --fold 0
"""
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nibabel as nib
from nibabel.processing import resample_to_output
from scipy.ndimage import label as scipy_label, binary_dilation, binary_fill_holes
from monai.inferers import sliding_window_inference
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from config import Config, get_config
from model import create_model
from fast_preprocessing import (
    create_brain_mask,
    dilate_brain_mask,
    normalize_image,
    get_bbox_from_mask,
    crop_to_bbox,
    DATA_INTERPOLATION_ORDER
)
from nibabel.processing import resample_from_to

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Preprocessing Functions (Using functions from fast_preprocessing.py)
# =============================================================================


def preprocess_case_for_inference(
    image_path: str,
    target_spacing: Tuple[float, float, float],
    verbose: bool = False,
    save_preprocessed: bool = False,
    output_dir: Optional[Path] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Preprocess a single case for inference.
    1. Load NIfTI
    2. Convert to canonical orientation (RAS+)
    3. Create brain mask on canonical data
    4. Dilate brain mask
    5. Resample image (with anisotropy handling) and brain mask to target spacing
    6. Normalize AFTER resampling using brain mask
    7. Convert to tensor
    
    Args:
        image_path: Path to input NIfTI file
        target_spacing: Target voxel spacing (x, y, z)
        verbose: Print detailed info
        save_preprocessed: Save preprocessed image as NIfTI for debugging
        output_dir: Directory to save preprocessed images (if save_preprocessed=True)
        
    Returns:
        preprocessed_tensor: Shape (1, H, W, D) - ready for model
        metadata: Dict with original image info and preprocessing steps for reverting
    """
    if verbose:
        logger.info(f"Preprocessing {Path(image_path).name}")
    
    # 1. Load image
    nii_img = nib.load(image_path)
    original_affine = nii_img.affine.copy()
    original_shape = nii_img.shape
    original_spacing = nii_img.header.get_zooms()[:3]
    
    if verbose:
        logger.info(f"  Original: shape={original_shape}, spacing={original_spacing}")
    
    # 2. Convert to canonical orientation (RAS+)
    if verbose:
        logger.info(f"  Converting to canonical orientation (RAS+)")
    img_canonical = nib.as_closest_canonical(nii_img)
    
    # 3. Create brain mask on canonical data (BEFORE resampling)
    if verbose:
        logger.info(f"  Creating brain mask on canonical data")
    image_canonical_data = img_canonical.get_fdata().astype(np.float32)
    brain_mask_canonical = create_brain_mask(image_canonical_data)
    
    # 4. Dilate brain mask to include border regions
    if verbose:
        logger.info(f"  Dilating brain mask")
    brain_mask_canonical = dilate_brain_mask(brain_mask_canonical)
    
    # Convert brain mask to NIfTI for resampling
    brain_mask_nib = nib.Nifti1Image(
        brain_mask_canonical.astype(np.uint8),
        affine=img_canonical.affine,
        header=img_canonical.header
    )
    
    # 5. Resample to target spacing (with anisotropy handling like training)
    if verbose:
        logger.info(f"  Resampling to target spacing {target_spacing}")
    
    current_spacing = img_canonical.header.get_zooms()[:3]
    
    # Check for anisotropy (nnUNet approach - same as training)
    if np.max(current_spacing) / np.min(current_spacing) > 3:
        if verbose:
            logger.info(f"    Data is highly anisotropic. Using two-stage resampling.")
        
        low_res_axis = np.argmax(current_spacing)
        
        # Create intermediate spacing: target for in-plane, original for out-of-plane
        intermediate_spacing = list(current_spacing)
        intermediate_spacing[low_res_axis] = target_spacing[low_res_axis]
        
        if verbose:
            logger.info(f"    Low res axis: {low_res_axis}, intermediate spacing: {intermediate_spacing}")
        
        # 1. Resample out-of-plane with nearest neighbor
        img_intermediate = nib.processing.resample_to_output(
            img_canonical,
            voxel_sizes=intermediate_spacing,
            order=0  # Nearest neighbor for out-of-plane
        )
        
        # 2. Resample in-plane with cubic interpolation
        img_resampled = nib.processing.resample_to_output(
            img_intermediate,
            voxel_sizes=target_spacing,
            order=DATA_INTERPOLATION_ORDER
        )
    else:
        # Single-stage resampling
        img_resampled = nib.processing.resample_to_output(
            img_canonical,
            voxel_sizes=target_spacing,
            order=DATA_INTERPOLATION_ORDER
        )
    
    # Resample brain mask to match image grid exactly
    if verbose:
        logger.info(f"    Resampling brain mask to match image geometry")
    brain_mask_resampled = resample_from_to(
        brain_mask_nib,
        img_resampled,
        order=0  # Nearest neighbor for binary mask
    )
    
    # Get numpy arrays
    image = img_resampled.get_fdata().astype(np.float32)
    brain_mask = brain_mask_resampled.get_fdata().astype(bool)
    resampled_affine = img_resampled.affine
    resampled_shape = image.shape
    
    # 6. Normalize AFTER resampling (CRITICAL: matches training)
    if verbose:
        logger.info(f"  Normalizing using brain mask (after resampling)")
    image = normalize_image(image, brain_mask)
    
    # 7. Crop to brain region (matches training pipeline)
    if verbose:
        logger.info(f"  Cropping to brain region")
    bbox = get_bbox_from_mask(brain_mask)
    
    image_cropped = crop_to_bbox(image, bbox)
    brain_mask_cropped = crop_to_bbox(brain_mask, bbox)
    cropped_shape = image_cropped.shape
    
    if verbose:
        logger.info(f"  Cropped: shape={cropped_shape} (was {resampled_shape})")
    
    if verbose:
        logger.info(f"  Preprocessed: shape={('1',) + cropped_shape}")
    
    # Save preprocessed image for debugging if requested
    if save_preprocessed and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        case_name = Path(image_path).name.replace('.nii.gz', '').replace('_0000', '')
        
        # Save normalized + resampled image
        preprocessed_path = output_dir / f"{case_name}_preprocessed.nii.gz"
        preprocessed_nii = nib.Nifti1Image(image, affine=resampled_affine)
        nib.save(preprocessed_nii, preprocessed_path)
        
        # Save brain mask (cropped)
        mask_path = output_dir / f"{case_name}_brain_mask.nii.gz"
        # Create affine for cropped data
        cropped_affine = resampled_affine.copy()
        offset = np.array([bbox[0][0], bbox[1][0], bbox[2][0], 0])
        cropped_affine[:, 3] = resampled_affine @ np.append(offset[:3], 1)
        mask_nii = nib.Nifti1Image(brain_mask_cropped.astype(np.uint8), affine=cropped_affine)
        nib.save(mask_nii, mask_path)
        
        if verbose:
            logger.info(f"  Saved preprocessed image to: {preprocessed_path}")
            logger.info(f"  Saved brain mask to: {mask_path}")
    
    # 8. Convert to tensor
    image_tensor = torch.from_numpy(image_cropped).float()
    image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
    
    # Store properties for reverting (only essential info, not full NIfTI)
    properties = {
        'original_affine': original_affine,
        'original_shape': original_shape,
        'original_spacing': original_spacing,
        'resampled_affine': resampled_affine,
        'resampled_shape': resampled_shape,
        'brain_mask_resampled': brain_mask,  # Store full resampled version (not cropped)
        'bbox_used_for_cropping': bbox,  # Store bbox for padding back
        'cropped_shape': cropped_shape,
        'brain_mask_cropped': brain_mask_cropped  # Store cropped brain mask
    }
    
    return image_tensor, properties


# =============================================================================
# Sliding Window Inference (Using MONAI)
# =============================================================================

def create_predictor_wrapper(model: torch.nn.Module) -> Callable:
    """
    Create a predictor function compatible with MONAI's sliding_window_inference.
    
    Handles deep supervision output (returns only the first output if model returns tuple/list).
    
    Args:
        model: The segmentation model
        
    Returns:
        Predictor function that takes batched input and returns predictions
    """
    def predictor(patch_data: torch.Tensor) -> torch.Tensor:
        """
        Predict on a batch of patches.
        
        Args:
            patch_data: Batched patches of shape (N, C, H, W, D)
            
        Returns:
            Predictions of shape (N, num_classes, H, W, D)
        """
        output = model(patch_data)
        # Handle deep supervision - take first output (highest resolution)
        if isinstance(output, (list, tuple)):
            output = output[0]
        return output
    
    return predictor


# =============================================================================
# Postprocessing (Revert preprocessing to get final segmentation)
# =============================================================================

def postprocess_prediction(
    predicted_logits: torch.Tensor,
    properties: Dict[str, Any],
    save_preprocessed: bool = False,
    output_dir: Optional[Path] = None,
    case_name: Optional[str] = None,
    filter_to_brain: bool = False,
    save_logits: bool = False,
    save_probabilities: bool = False
) -> nib.Nifti1Image:
    """
    Postprocess prediction to original image space.
    
    Steps (reverse of preprocessing):
    1. Pad logits back to full resampled size (reverse of cropping)
       - Background class logits outside brain: set to dtype max (high confidence background)
       - Foreground class logits outside brain: set to dtype min (low confidence foreground)
    2. Apply argmax to get segmentation
    3. Set pixels outside brain mask to -1 (for evaluation)
    4. Resample back to original spacing
    5. Optionally save logits and/or probabilities in original space
    
    For probabilities: softmax is applied BEFORE resampling to preserve probability
    distributions. Each channel is resampled independently with linear interpolation.
    After resampling, probabilities are re-normalized to sum to 1.
    
    Args:
        predicted_logits: Predicted logits of shape (num_classes, X, Y, Z) - cropped
        properties: Properties dict from preprocessing
        save_preprocessed: Save intermediate prediction for debugging
        output_dir: Directory to save intermediate prediction
        case_name: Case name for saving files
        filter_to_brain: Filter to brain mask before resampling (now always done via -1)
        save_logits: Save logits in original image space
        save_probabilities: Save probability maps (via softmax) in original image space
        
    Returns:
        Final segmentation in original image space (with -1 outside brain mask)
    """
    logger.info(f"  Postprocessing prediction")
    
    # Step 0: Pad logits back to full resampled size (reverse of cropping)
    bbox = properties.get('bbox_used_for_cropping')
    resampled_shape = properties['resampled_shape']
    brain_mask_full = properties['brain_mask_resampled']
    num_classes = predicted_logits.shape[0]
    
    if bbox is not None:
        logger.info(f"    Padding logits from cropped {tuple(predicted_logits.shape[1:])} to full {resampled_shape}")
        
        # Create full-size logits array
        # Use float32 to get proper min/max values
        predicted_logits_np = predicted_logits.cpu().numpy().astype(np.float32)
        logits_dtype_info = np.finfo(np.float32)
        
        # Convert resampled_shape to tuple if it's a list/array
        resampled_shape_tuple = tuple(resampled_shape)
        
        # Initialize full logits with appropriate values for background outside brain:
        # - Background class (0): high logit (dtype max) -> high confidence for background
        # - Foreground classes (1+): low logit (dtype min) -> low confidence for foreground
        full_logits = np.zeros((num_classes,) + resampled_shape_tuple, dtype=np.float32)
        full_logits[0, :, :, :] = logits_dtype_info.max  # Background class gets max value
        full_logits[1:, :, :, :] = logits_dtype_info.min  # Foreground classes get min value
        
        # Copy cropped logits into the correct position
        full_logits[
            :,
            bbox[0][0]:bbox[0][1],
            bbox[1][0]:bbox[1][1],
            bbox[2][0]:bbox[2][1]
        ] = predicted_logits_np
        
        # Convert back to tensor
        predicted_logits = torch.from_numpy(full_logits)
        
        logger.info(f"    Padded logits shape: {predicted_logits.shape}")
    
    # Step 1: Convert logits to segmentation
    # Argmax to get class labels (no softmax needed, just argmax)
    segmentation = torch.argmax(predicted_logits, dim=0).cpu().numpy()
    
    logger.info(f"    After argmax: shape={segmentation.shape}")
    
    # Step 1a: Set pixels outside brain mask to -1 (for evaluation)
    # Use int8 to support -1 value
    segmentation = segmentation.astype(np.int8)
    segmentation[~brain_mask_full] = -1
    logger.info(f"    Set {(~brain_mask_full).sum()} pixels outside brain mask to -1")
    
    # Step 1b: Compute probabilities via softmax BEFORE any resampling
    # This preserves proper probability distributions
    if save_probabilities:
        probabilities_resampled = torch.softmax(predicted_logits, dim=0).cpu().numpy().astype(np.float32)
        logger.info(f"    Computed softmax probabilities: shape={probabilities_resampled.shape}")
        # Verify probabilities sum to 1 in resampled space
        prob_sum = probabilities_resampled.sum(axis=0)
        logger.info(f"    Probability sum in resampled space - min: {prob_sum.min():.6f}, max: {prob_sum.max():.6f}")
    
    # Step 1c: Get logits as numpy if needed
    if save_logits:
        logits_resampled = predicted_logits.cpu().numpy().astype(np.float32)
        logger.info(f"    Logits shape: {logits_resampled.shape}")
    
    # Note: filter_to_brain is now always implicitly done via the -1 labeling above
    # The old filter_to_brain behavior (set to 0) is replaced by setting to -1
    
    # Step 2: Create NIfTI image in resampled space
    # Use int8 to preserve -1 values
    seg_nib_resampled = nib.Nifti1Image(
        segmentation.astype(np.int8),
        affine=properties['resampled_affine']
    )
    
    # Save intermediate prediction (in resampled space) if requested
    if save_preprocessed and output_dir is not None and case_name is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pred_resampled_path = output_dir / f"{case_name}_prediction_resampled.nii.gz"
        nib.save(seg_nib_resampled, pred_resampled_path)
        
        logger.info(f"    Saved resampled prediction to: {pred_resampled_path}")
    
    # Step 3: Resample back to original space
    # Create a reference NIfTI with original properties for resampling
    original_ref = nib.Nifti1Image(
        np.zeros(properties['original_shape'], dtype=np.int8),
        affine=properties['original_affine']
    )
    
    seg_nib_original = resample_from_to(
        seg_nib_resampled,
        original_ref,
        order=0,  # Nearest neighbor for labels
        mode='constant',
        cval=-1  # Use -1 for areas outside the resampled region (should be rare)
    )
    
    logger.info(f"    After resample to original: shape={seg_nib_original.get_fdata().shape}")
    
    # Step 4: Save logits in original space if requested
    if save_logits and output_dir is not None and case_name is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_classes = logits_resampled.shape[0]
        original_shape = tuple(properties['original_shape'])
        logits_original = np.zeros(
            (num_classes,) + original_shape, 
            dtype=np.float32
        )
        
        # Resample each logit channel independently with linear interpolation
        for c in range(num_classes):
            logit_channel_nib = nib.Nifti1Image(
                logits_resampled[c],
                affine=properties['resampled_affine']
            )
            logit_channel_original = resample_from_to(
                logit_channel_nib,
                original_ref,
                order=1,  # Linear interpolation for continuous values
                mode='constant',
                cval=0.0
            )
            logits_original[c] = logit_channel_original.get_fdata().astype(np.float32)
        
        # Save as 4D NIfTI (last dimension is class channel)
        # Transpose from (C, X, Y, Z) to (X, Y, Z, C) for NIfTI convention
        logits_4d = np.transpose(logits_original, (1, 2, 3, 0))
        logits_nib = nib.Nifti1Image(logits_4d, affine=properties['original_affine'])
        
        logits_path = output_dir / f"{case_name}_logits.nii.gz"
        nib.save(logits_nib, logits_path)
        logger.info(f"    Saved logits to: {logits_path}")
    
    # Step 5: Save probabilities in original space if requested
    if save_probabilities and output_dir is not None and case_name is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_classes = probabilities_resampled.shape[0]
        original_shape = tuple(properties['original_shape'])
        probabilities_original = np.zeros(
            (num_classes,) + original_shape, 
            dtype=np.float32
        )
        
        # Resample each probability channel independently with linear interpolation
        for c in range(num_classes):
            prob_channel_nib = nib.Nifti1Image(
                probabilities_resampled[c],
                affine=properties['resampled_affine']
            )
            prob_channel_original = resample_from_to(
                prob_channel_nib,
                original_ref,
                order=1,  # Linear interpolation for continuous values
                mode='constant',
                cval=0.0
            )
            probabilities_original[c] = prob_channel_original.get_fdata().astype(np.float32)
        
        # Re-normalize probabilities to sum to 1 after resampling
        # This is necessary because linear interpolation can break the sum-to-1 property
        prob_sum = probabilities_original.sum(axis=0, keepdims=True)
        # Avoid division by zero (set to 1 where sum is 0)
        prob_sum = np.where(prob_sum == 0, 1.0, prob_sum)
        probabilities_original = probabilities_original / prob_sum
        
        # Verify probabilities are valid after normalization
        final_prob_sum = probabilities_original.sum(axis=0)
        prob_min = probabilities_original.min()
        prob_max = probabilities_original.max()
        logger.info(f"    Probability map validation after resampling and normalization:")
        logger.info(f"      Sum - min: {final_prob_sum.min():.6f}, max: {final_prob_sum.max():.6f}")
        logger.info(f"      Values - min: {prob_min:.6f}, max: {prob_max:.6f}")
        
        # Check if probabilities are valid (sum to ~1 and in [0,1] range)
        is_valid = np.allclose(final_prob_sum, 1.0, atol=1e-3) and prob_min >= 0 and prob_max <= 1
        if is_valid:
            logger.info(f"      ✓ Probability map is valid!")
        else:
            logger.warning(f"      ⚠ Probability map may have issues!")
        
        # Save as 4D NIfTI (last dimension is class channel)
        # Transpose from (C, X, Y, Z) to (X, Y, Z, C) for NIfTI convention
        probabilities_4d = np.transpose(probabilities_original, (1, 2, 3, 0))
        probabilities_nib = nib.Nifti1Image(probabilities_4d, affine=properties['original_affine'])
        
        probabilities_path = output_dir / f"{case_name}_probabilities.nii.gz"
        nib.save(probabilities_nib, probabilities_path)
        logger.info(f"    Saved probabilities to: {probabilities_path}")
    
    return seg_nib_original


# =============================================================================
# PyTorch Dataset for Parallel Data Loading
# =============================================================================

class InferenceDataset(Dataset):
    """
    PyTorch Dataset for parallel preprocessing during inference.
    
    Each worker independently loads and preprocesses images, enabling
    parallel CPU preprocessing while the GPU performs inference.
    
    Supports two modes:
    1. Raw mode: Load NIfTI and preprocess on-the-fly
    2. Preprocessed mode: Load already-preprocessed .npy files
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        target_spacing: Tuple[float, float, float],
        verbose: bool = False,
        save_preprocessed: bool = False,
        output_dir: Optional[Path] = None,
        use_preprocessed: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of paths to input images (NIfTI or .npy)
            target_spacing: Target voxel spacing for resampling
            verbose: Print preprocessing details
            save_preprocessed: Save preprocessed images
            output_dir: Directory to save preprocessed images
            use_preprocessed: If True, load from .npy files instead of NIfTI
        """
        self.image_paths = image_paths
        self.target_spacing = target_spacing
        self.verbose = verbose
        self.save_preprocessed = save_preprocessed
        self.output_dir = output_dir
        self.use_preprocessed = use_preprocessed
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any], Path]:
        """
        Load and preprocess a single case.
        
        Returns:
            preprocessed_tensor: Shape (1, H, W, D)
            properties: Metadata for postprocessing
            image_path: Original image path
        """
        image_path = self.image_paths[idx]
        
        if self.use_preprocessed:
            # Load preprocessed .npy file
            data = np.load(image_path)
            preprocessed_tensor = torch.from_numpy(data).float()
            
            # For preprocessed data, we don't have properties for reverting
            # Create minimal properties dict
            properties = {
                'preprocessed': True,
                'original_path': str(image_path)
            }
        else:
            # Preprocess the case on-the-fly
            preprocessed_tensor, properties = preprocess_case_for_inference(
                str(image_path),
                target_spacing=self.target_spacing,
                verbose=self.verbose,
                save_preprocessed=self.save_preprocessed,
                output_dir=self.output_dir
            )
        
        return preprocessed_tensor, properties, image_path


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Since each image can have different dimensions after preprocessing,
    we don't stack them. Just return lists.
    
    Args:
        batch: List of (tensor, properties, path) tuples
        
    Returns:
        tensors: List of preprocessed tensors
        properties_list: List of property dicts
        paths: List of image paths
    """
    tensors = [item[0] for item in batch]
    properties_list = [item[1] for item in batch]
    paths = [item[2] for item in batch]
    
    return tensors, properties_list, paths


# =============================================================================
# Background Post-Processing
# =============================================================================

def postprocess_and_save(
    predicted_logits: torch.Tensor,
    properties: Dict[str, Any],
    output_path: Path,
    case_name: str,
    save_preprocessed: bool = False,
    filter_to_brain: bool = False,
    save_logits: bool = False,
    save_probabilities: bool = False,
    pbar: Optional[tqdm] = None,
    pbar_lock: Optional[Lock] = None
) -> None:
    """
    Post-process prediction and save to disk (designed to run in background thread).
    
    Args:
        predicted_logits: Predicted logits tensor
        properties: Preprocessing metadata
        output_path: Path to save final segmentation
        case_name: Case name for intermediate files
        save_preprocessed: Save intermediate predictions
        filter_to_brain: Filter to brain mask
        save_logits: Save logits in original image space
        save_probabilities: Save probability maps (via softmax) in original image space
        pbar: Progress bar to update
        pbar_lock: Lock for thread-safe progress bar updates
    """
    try:
        # Postprocess
        seg_nib = postprocess_prediction(
            predicted_logits,
            properties,
            save_preprocessed=save_preprocessed,
            output_dir=output_path.parent if (save_preprocessed or save_logits or save_probabilities) else None,
            case_name=case_name,
            filter_to_brain=filter_to_brain,
            save_logits=save_logits,
            save_probabilities=save_probabilities
        )
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(seg_nib, output_path)
        
    except Exception as e:
        logger.error(f"Error in post-processing {case_name}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Update progress bar in thread-safe manner
        if pbar is not None and pbar_lock is not None:
            with pbar_lock:
                pbar.update(1)


# =============================================================================
# Main Inference Function
# =============================================================================

class Predictor:
    """
    Predictor class for inference.
    
    Usage:
        predictor = Predictor(checkpoint_path, config, device='cuda')
        predictor.predict_from_folder(input_folder, output_folder)
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        config: Optional[Config] = None,
        device: str = 'cuda',
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        verbose: bool = True,
        save_preprocessed: bool = False,
        sw_batch_size: int = 1,
        filter_to_brain: bool = False,
        num_workers: int = 4,
        num_postprocess_workers: int = 2,
        use_preprocessed: bool = False,
        save_logits: bool = False,
        save_probabilities: bool = False
    ):
        """
        Initialize predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration (if None, will try to load from checkpoint)
            device: Device to run on
            tile_step_size: Sliding window step size (0.5 = 50% overlap)
            use_gaussian: Use Gaussian weighting
            verbose: Print progress
            save_preprocessed: Save preprocessed images for debugging
            sw_batch_size: Batch size for sliding window inference
            filter_to_brain: Filter final segmentation to brain mask
            num_workers: Number of parallel workers for data loading (default: 4)
            num_postprocess_workers: Number of background threads for post-processing (default: 2)
            use_preprocessed: Load preprocessed .npy files instead of raw NIfTI (default: False)
            save_logits: Save logits in original image space (default: False)
            save_probabilities: Save probability maps (via softmax) in original image space (default: False)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.verbose = verbose
        self.save_preprocessed = save_preprocessed
        self.sw_batch_size = sw_batch_size
        self.filter_to_brain = filter_to_brain
        self.num_workers = num_workers
        self.num_postprocess_workers = num_postprocess_workers
        self.use_preprocessed = use_preprocessed
        self.save_logits = save_logits
        self.save_probabilities = save_probabilities

        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load or use provided config
        if config is None:
            if 'config' in checkpoint:
                self.config = checkpoint['config']
                logger.info("Loaded config from checkpoint")
            else:
                logger.warning("No config in checkpoint, using default config")
                self.config = get_config()
        else:
            self.config = config
        
        # Create model
        logger.info("Creating model")
        self.model = create_model(self.config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError("Could not find model weights in checkpoint")
        
        logger.info(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set thread settings for optimal performance
        if self.device.type == 'cuda':
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        
        logger.info(f"Model ready on {self.device}")
    
    def predict_case(
        self,
        image_path: Path,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Predict a single case using MONAI's sliding window inference.
        
        Args:
            image_path: Path to input NIfTI image
            output_path: Path to save output (optional)
            
        Returns:
            Segmentation in original image space
        """
        logger.info(f"\nPredicting {image_path.name}")
        
        # Preprocess
        image, properties = preprocess_case_for_inference(
            str(image_path),
            target_spacing=self.config.data.target_spacing,
            verbose=self.verbose,
            save_preprocessed=self.save_preprocessed,
            output_dir=output_path.parent if output_path is not None else None
        )
        
        # Add batch dimension for MONAI
        image_tensor = image.unsqueeze(0)  # (1, C, H, W, D)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Create predictor wrapper for deep supervision handling
        predictor = create_predictor_wrapper(self.model)
        
        # Predict using MONAI's sliding_window_inference
        with torch.no_grad():
            if self.verbose:
                logger.info(f"  Running sliding window inference:")
                logger.info(f"    Input shape: {image_tensor.shape}")
                logger.info(f"    Patch size: {self.config.data.patch_size}")
                logger.info(f"    Overlap: {1.0 - self.tile_step_size}")
            
            predicted_logits = sliding_window_inference(
                inputs=image_tensor,
                roi_size=self.config.data.patch_size,
                sw_batch_size=self.sw_batch_size,
                predictor=predictor,
                overlap=1.0 - self.tile_step_size,  # MONAI uses overlap, not step size
                mode="gaussian" if self.use_gaussian else "constant",
                sigma_scale=0.125,  # Same as nnUNet (1/8)
                padding_mode="constant",  # Edge padding like we were using
                cval=0.0,
                sw_device=self.device,
                device=self.device,
                progress=self.verbose
            )
        
        # Remove batch dimension
        predicted_logits = predicted_logits[0]  # (num_classes, H, W, D)
        
        if self.verbose:
            logger.info(f"    Predicted logits shape: {predicted_logits.shape}")
        
        # Get case name for saving intermediate files
        case_name = image_path.name.replace('.nii.gz', '').replace('_0000', '')
        
        # Postprocess
        seg_nib = postprocess_prediction(
            predicted_logits,
            properties,
            save_preprocessed=self.save_preprocessed,
            output_dir=output_path.parent if output_path is not None else None,
            case_name=case_name,
            filter_to_brain=self.filter_to_brain,
            save_logits=self.save_logits,
            save_probabilities=self.save_probabilities
        )
        
        # Save if output path provided
        if output_path is not None:
            logger.info(f"  Saving to {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            nib.save(seg_nib, output_path)
        
        return seg_nib
    
    def predict_from_folder(
        self,
        input_folder: Path,
        output_folder: Path,
        file_pattern: str = "*.nii.gz",
        max_samples: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Predict all cases in a folder using parallel data loading and background post-processing.
        
        Uses:
        1. PyTorch DataLoader with multiple workers for parallel preprocessing
        2. ThreadPoolExecutor for background post-processing (resample + save)
        
        This creates a 3-stage pipeline:
        - Stage 1 (CPU): Parallel preprocessing in DataLoader workers
        - Stage 2 (GPU): Inference on preprocessed data
        - Stage 3 (CPU): Background post-processing and saving in thread pool
        
        Args:
            input_folder: Folder with input images
            output_folder: Folder to save predictions
            file_pattern: Glob pattern for input files
            max_samples: If specified, randomly sample this many files (for debugging)
            random_seed: Random seed for sampling
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all input files
        input_files = sorted(input_folder.glob(file_pattern))
        
        if len(input_files) == 0:
            raise ValueError(f"No files found matching {file_pattern} in {input_folder}")
        
        logger.info(f"\nFound {len(input_files)} files total")
        
        # Randomly sample if max_samples is specified
        if max_samples is not None and max_samples < len(input_files):
            import random
            random.seed(random_seed)
            input_files = random.sample(input_files, max_samples)
            logger.info(f"Randomly sampled {max_samples} files for processing (seed={random_seed})")
        
        logger.info(f"Processing {len(input_files)} files:")
        logger.info(f"  - Preprocessing workers: {self.num_workers}")
        logger.info(f"  - Post-processing workers: {self.num_postprocess_workers}")
        
        # Create dataset and dataloader for parallel preprocessing
        dataset = InferenceDataset(
            image_paths=input_files,
            target_spacing=self.config.data.target_spacing,
            verbose=False,  # Disable per-image verbose in workers
            save_preprocessed=self.save_preprocessed,
            output_dir=output_folder if self.save_preprocessed else None,
            use_preprocessed=self.use_preprocessed
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Process one image at a time (different sizes)
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        # Create thread pool for background post-processing
        postprocess_executor = ThreadPoolExecutor(max_workers=self.num_postprocess_workers)
        postprocess_futures = []
        
        # Progress tracking
        pbar = tqdm(total=len(input_files), desc="Inference progress")
        pbar_lock = Lock()
        
        try:
            # Process each batch (preprocessed in parallel)
            for tensors_batch, properties_batch, paths_batch in dataloader:
                # Process each item in the batch (batch_size=1, so just one item)
                for image_tensor, properties, input_path in zip(tensors_batch, properties_batch, paths_batch):
                    # Get case name by removing _0000 suffix (matches labelsTs naming)
                    case_name = input_path.name.replace('.nii.gz', '').replace('_0000', '')
                    output_path = output_folder / f"{case_name}.nii.gz"
                    
                    try:
                        # Add batch dimension for MONAI
                        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W, D)
                        
                        # Move to device
                        image_tensor = image_tensor.to(self.device)
                        
                        # Create predictor wrapper for deep supervision handling
                        predictor = create_predictor_wrapper(self.model)
                        
                        # Predict using MONAI's sliding_window_inference
                        with torch.no_grad():
                            predicted_logits = sliding_window_inference(
                                inputs=image_tensor,
                                roi_size=self.config.data.patch_size,
                                sw_batch_size=self.sw_batch_size,
                                predictor=predictor,
                                overlap=1.0 - self.tile_step_size,
                                mode="gaussian" if self.use_gaussian else "constant",
                                sigma_scale=0.125,
                                padding_mode="constant",
                                cval=0.0,
                                sw_device=self.device,
                                device=self.device,
                                progress=False  # Disable per-image progress
                            )
                        
                        # Remove batch dimension and move to CPU for post-processing
                        predicted_logits = predicted_logits[0].cpu()  # (num_classes, H, W, D)
                        
                        # Check if we can do postprocessing
                        if self.use_preprocessed or properties.get('preprocessed', False):
                            # Preprocessed data - save logits directly without resampling
                            logger.warning(f"Preprocessed data mode: saving argmax segmentation without resampling to original space")
                            segmentation = torch.argmax(predicted_logits, dim=0).cpu().numpy().astype(np.uint8)
                            
                            # Save directly
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            seg_nii = nib.Nifti1Image(
                                segmentation,
                                affine=properties.get('resampled_affine', np.eye(4))
                            )
                            nib.save(seg_nii, output_path)
                            
                            # Update progress
                            with pbar_lock:
                                pbar.update(1)
                        else:
                            # Submit post-processing to background thread pool
                            # This allows inference to continue on next sample while post-processing happens
                            future = postprocess_executor.submit(
                                postprocess_and_save,
                                predicted_logits=predicted_logits,
                                properties=properties,
                                output_path=output_path,
                                case_name=case_name,
                                save_preprocessed=self.save_preprocessed,
                                filter_to_brain=self.filter_to_brain,
                                save_logits=self.save_logits,
                                save_probabilities=self.save_probabilities,
                                pbar=pbar,
                                pbar_lock=pbar_lock
                            )
                            postprocess_futures.append(future)
                        
                    except Exception as e:
                        logger.error(f"Error during inference for {input_path.name}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Still update progress bar for failed cases
                        with pbar_lock:
                            pbar.update(1)
            
            # Wait for all post-processing to complete
            logger.info("\nWaiting for background post-processing to complete...")
            for future in as_completed(postprocess_futures):
                try:
                    future.result()  # Will raise exception if post-processing failed
                except Exception as e:
                    logger.error(f"Post-processing error: {e}")
            
        finally:
            # Clean up
            pbar.close()
            postprocess_executor.shutdown(wait=True)
        
        logger.info(f"\n✓ All predictions complete! Results saved to {output_folder}")


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MedNeXt segmentation model inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input_folder', '-i',
        type=str,
        required=True,
        help='Input folder with NIfTI images'
    )
    
    parser.add_argument(
        '--output_folder', '-o',
        type=str,
        required=True,
        help='Output folder for predictions'
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on'
    )
    
    parser.add_argument(
        '--tile_step_size',
        type=float,
        default=0.5,
        help='Sliding window step size (0.5 = 50%% overlap)'
    )
    
    parser.add_argument(
        '--sw_batch_size',
        type=int,
        default=1,
        help='Batch size for sliding window inference (default: 1)'
    )
    
    parser.add_argument(
        '--no_gaussian',
        action='store_true',
        help='Disable Gaussian weighting for sliding window'
    )
    
    parser.add_argument(
        '--file_pattern',
        type=str,
        default='*.nii.gz',
        help='Glob pattern for input files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Randomly sample this many files for processing (for debugging)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    
    parser.add_argument(
        '--save_preprocessed',
        action='store_true',
        help='Save preprocessed images as NIfTI files for debugging'
    )

    parser.add_argument(
        '--filter_to_brain',
        action='store_true',
        help='Filter final segmentation to brain mask (set outside brain to background)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=6,
        help='Number of parallel workers for data loading (default: 6)'
    )
    
    parser.add_argument(
        '--num_postprocess_workers',
        type=int,
        default=6,
        help='Number of background threads for post-processing (default: 6)'
    )
    
    parser.add_argument(
        '--use_preprocessed',
        action='store_true',
        help='Load preprocessed .npy files instead of raw NIfTI images'
    )
    
    parser.add_argument(
        '--save_logits',
        action='store_true',
        help='Save logits in original image space (4D NIfTI with class channels)'
    )
    
    parser.add_argument(
        '--save_probabilities',
        action='store_true',
        help='Save probability maps (via softmax) in original image space (4D NIfTI with class channels)'
    )
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = Predictor(
        checkpoint_path=Path(args.checkpoint),
        config=None,  # Will load from checkpoint
        device=args.device,
        tile_step_size=args.tile_step_size,
        use_gaussian=not args.no_gaussian,
        verbose=args.verbose,
        save_preprocessed=args.save_preprocessed,
        sw_batch_size=args.sw_batch_size,
        filter_to_brain=args.filter_to_brain,
        num_workers=args.num_workers,
        num_postprocess_workers=args.num_postprocess_workers,
        use_preprocessed=args.use_preprocessed,
        save_logits=args.save_logits,
        save_probabilities=args.save_probabilities
    )
    
    # Run prediction
    predictor.predict_from_folder(
        input_folder=Path(args.input_folder),
        output_folder=Path(args.output_folder),
        file_pattern=args.file_pattern,
        max_samples=args.max_samples,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
