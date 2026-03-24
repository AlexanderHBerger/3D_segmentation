from __future__ import annotations

import sys
import torch
import torch.nn.functional as F
import numpy as np

# Add Betti-Matching-3D to path
sys.path.insert(0, '/vol/miltank/users/bergeral/vesuvius/metrics_impl/topological-metrics-kaggle/external/Betti-Matching-3D/build')
import betti_matching

import typing
if typing.TYPE_CHECKING:
    from typing import Tuple, List, Dict, Optional
    LossOutputName = str
    from jaxtyping import Float
    from torch import Tensor


ENCOUNTERED_NONCONTIGUOUS = False


def bounds_from_coords(valid_coords: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """Compute bounding box from precomputed valid coordinates.
    
    This is much faster than computing bounds from a mask tensor since it only
    requires min/max operations on an (N, 3) array.
    
    Args:
        valid_coords: (N, 3) array of valid voxel coordinates in (W, H, D) order
        
    Returns:
        Tuple of ((w_min, w_max), (h_min, h_max), (d_min, d_max)) inclusive bounds,
        or None if valid_coords is None or empty.
    """
    if valid_coords is None or len(valid_coords) == 0:
        return None
    
    w_min, h_min, d_min = valid_coords.min(axis=0)
    w_max, h_max, d_max = valid_coords.max(axis=0)
    
    return ((int(w_min), int(w_max)), (int(h_min), int(h_max)), (int(d_min), int(d_max)))


def find_valid_region_bounds(
    valid_mask: torch.Tensor,
    precomputed_bounds: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Find the bounding box of the valid region in a 3D mask.
    
    Since valid regions are always a connected cube, we find min/max along each axis.
    Uses efficient reduction operations instead of nonzero() for better performance.
    
    Args:
        valid_mask: (W, H, D) boolean/float tensor where valid regions are > 0.5
        precomputed_bounds: Optional precomputed bounds to skip computation.
                           If provided, returns these bounds directly.
        
    Returns:
        Tuple of ((w_min, w_max), (h_min, h_max), (d_min, d_max)) inclusive bounds
    """
    # Use precomputed bounds if available
    if precomputed_bounds is not None:
        return precomputed_bounds
    
    # Create binary mask
    binary_mask = (valid_mask > 0.5)
    
    # Check if there are any valid voxels
    if not binary_mask.any():
        # No valid region, return full volume bounds
        return ((0, valid_mask.shape[0] - 1), 
                (0, valid_mask.shape[1] - 1), 
                (0, valid_mask.shape[2] - 1))
    
    # Find bounds using efficient max/argmax operations along each axis
    # For each dimension, project the mask and find first/last True values
    
    # W dimension (dim 0): project onto W axis by checking if any valid in H,D plane
    w_any = binary_mask.any(dim=2).any(dim=1)  # Shape: (W,)
    w_indices = torch.where(w_any)[0]
    w_min, w_max = w_indices[0].item(), w_indices[-1].item()
    
    # H dimension (dim 1): project onto H axis
    h_any = binary_mask.any(dim=2).any(dim=0)  # Shape: (H,)
    h_indices = torch.where(h_any)[0]
    h_min, h_max = h_indices[0].item(), h_indices[-1].item()
    
    # D dimension (dim 2): project onto D axis
    d_any = binary_mask.any(dim=1).any(dim=0)  # Shape: (D,)
    d_indices = torch.where(d_any)[0]
    d_min, d_max = d_indices[0].item(), d_indices[-1].item()
    
    return ((w_min, w_max), (h_min, h_max), (d_min, d_max))


def sample_random_crop(spatial_dims: Tuple[int, ...], 
                       crop_size: int, 
                       valid_mask: Optional[torch.Tensor] = None,
                       valid_bounds: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None) -> Tuple[int, int, int]:
    """Sample a random crop position that maximizes overlap with valid regions.
    
    Args:
        spatial_dims: (W, H, D) spatial dimensions of the volume
        crop_size: Size of the cubic crop
        valid_mask: Optional (B, 1, W, H, D) or (W, H, D) mask of valid regions
        valid_bounds: Optional precomputed bounds ((w_min, w_max), (h_min, h_max), (d_min, d_max)).
                     If provided, skips computing bounds from valid_mask.
        
    Returns:
        (start_w, start_h, start_d) starting coordinates for the crop
    """
    W, H, D = spatial_dims
    
    # Clamp crop_size to not exceed spatial dimensions
    crop_w = min(crop_size, W)
    crop_h = min(crop_size, H)
    crop_d = min(crop_size, D)
    
    if valid_mask is None and valid_bounds is None:
        # Random crop without valid mask constraint
        start_w = np.random.randint(0, max(1, W - crop_w + 1))
        start_h = np.random.randint(0, max(1, H - crop_h + 1))
        start_d = np.random.randint(0, max(1, D - crop_d + 1))
        return start_w, start_h, start_d
    
    # Use precomputed bounds if available, otherwise compute from mask
    if valid_bounds is not None:
        (w_min, w_max), (h_min, h_max), (d_min, d_max) = valid_bounds
    else:
        # Squeeze to get (W, H, D) if needed
        mask = valid_mask.squeeze() if valid_mask.dim() > 3 else valid_mask
        (w_min, w_max), (h_min, h_max), (d_min, d_max) = find_valid_region_bounds(mask)
    valid_w = w_max - w_min + 1
    valid_h = h_max - h_min + 1
    valid_d = d_max - d_min + 1
    
    # Compute optimal crop range to maximize valid region coverage
    # Start position should be chosen such that the crop contains as much valid region as possible
    
    # For each dimension, find the range of valid start positions
    # The crop should ideally contain the entire valid region, or as much as possible
    
    # W dimension
    if valid_w <= crop_w:
        # Crop can contain entire valid region in W - center it or randomize within bounds
        start_w_min = max(0, w_max - crop_w + 1)
        start_w_max = min(W - crop_w, w_min)
    else:
        # Valid region larger than crop - sample within valid region
        start_w_min = w_min
        start_w_max = w_max - crop_w + 1
    start_w_min = max(0, start_w_min)
    start_w_max = max(start_w_min, min(W - crop_w, start_w_max))
    
    # H dimension
    if valid_h <= crop_h:
        start_h_min = max(0, h_max - crop_h + 1)
        start_h_max = min(H - crop_h, h_min)
    else:
        start_h_min = h_min
        start_h_max = h_max - crop_h + 1
    start_h_min = max(0, start_h_min)
    start_h_max = max(start_h_min, min(H - crop_h, start_h_max))
    
    # D dimension
    if valid_d <= crop_d:
        start_d_min = max(0, d_max - crop_d + 1)
        start_d_max = min(D - crop_d, d_min)
    else:
        start_d_min = d_min
        start_d_max = d_max - crop_d + 1
    start_d_min = max(0, start_d_min)
    start_d_max = max(start_d_min, min(D - crop_d, start_d_max))
    
    # Sample randomly within the computed ranges
    start_w = np.random.randint(start_w_min, start_w_max + 1)
    start_h = np.random.randint(start_h_min, start_h_max + 1)
    start_d = np.random.randint(start_d_min, start_d_max + 1)
    
    return start_w, start_h, start_d


def sample_uncertainty_crop(prediction: torch.Tensor,
                            crop_size: int,
                            valid_mask: Optional[torch.Tensor] = None,
                            valid_bounds: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None) -> Tuple[int, int, int]:
    """Sample a crop position centered on the most uncertain region within valid areas.
    
    Uncertainty is measured as proximity to 0.5 (maximum uncertainty for binary prediction).
    Uses efficient strided pooling instead of full convolution for speed.
    
    Args:
        prediction: (B, 1, W, H, D) prediction tensor with values in [0, 1]
        crop_size: Size of the cubic crop
        valid_mask: Optional (B, 1, W, H, D) mask of valid regions
        valid_bounds: Optional precomputed bounds ((w_min, w_max), (h_min, h_max), (d_min, d_max)).
                     If provided, skips computing bounds from valid_mask.
        
    Returns:
        (start_w, start_h, start_d) starting coordinates for the crop
    """
    # Use first sample in batch
    pred = prediction[0, 0]  # (W, H, D)
    W, H, D = pred.shape
    
    crop_w = min(crop_size, W)
    crop_h = min(crop_size, H)
    crop_d = min(crop_size, D)
    
    # Compute uncertainty map: highest at 0.5, lowest at 0 and 1
    uncertainty = 1.0 - 2.0 * torch.abs(pred - 0.5)  # Maps [0,1] -> [0,1] with max at 0.5
    
    # Apply valid mask if provided
    if valid_mask is not None:
        mask = valid_mask.squeeze() if valid_mask.dim() > 3 else valid_mask
        # Zero out uncertainty in invalid regions
        uncertainty = uncertainty * (mask > 0.5).float()
        
        # Get valid region bounds for constraining crop position
        # Use precomputed bounds if available
        (w_min, w_max), (h_min, h_max), (d_min, d_max) = find_valid_region_bounds(mask, valid_bounds)
    else:
        mask = None
        w_min, w_max = 0, W - 1
        h_min, h_max = 0, H - 1
        d_min, d_max = 0, D - 1
    
    # Use average pooling with stride for efficient uncertainty aggregation
    if uncertainty.sum() < 1e-8:
        # No uncertainty (all certain predictions) - fall back to random crop
        return sample_random_crop((W, H, D), crop_size, valid_mask, valid_bounds)
    
    # Use a stride-based approach: divide volume into grid cells and find cell with max uncertainty
    # Grid cell size approximately equals crop_size for efficiency
    stride = max(crop_size // 2, 1)  # Overlap by 50% for better resolution
    
    uncertainty_5d = uncertainty.unsqueeze(0).unsqueeze(0)  # (1, 1, W, H, D)
    
    # Use average pooling with kernel_size=crop_size and stride for efficiency
    # This gives us the average uncertainty in each potential crop region
    pooled = F.avg_pool3d(
        uncertainty_5d, 
        kernel_size=(crop_w, crop_h, crop_d), 
        stride=(stride, stride, stride),
        padding=0
    )
    
    if pooled.numel() == 0:
        # Volume too small for pooling, fall back to random
        return sample_random_crop((W, H, D), crop_size, valid_mask, valid_bounds)
    
    # Find position with maximum average uncertainty
    pooled_squeezed = pooled.squeeze()
    if pooled_squeezed.dim() == 0:
        # Single value
        return 0, 0, 0
    
    flat_idx = pooled_squeezed.argmax().item()
    out_shape = pooled_squeezed.shape
    
    if len(out_shape) == 3:
        grid_w = flat_idx // (out_shape[1] * out_shape[2])
        grid_h = (flat_idx % (out_shape[1] * out_shape[2])) // out_shape[2]
        grid_d = flat_idx % out_shape[2]
    else:
        grid_w, grid_h, grid_d = 0, 0, 0
    
    # Convert grid position to actual crop start position
    start_w = min(grid_w * stride, W - crop_w)
    start_h = min(grid_h * stride, H - crop_h)
    start_d = min(grid_d * stride, D - crop_d)
    
    # Ensure non-negative
    start_w = max(0, start_w)
    start_h = max(0, start_h)
    start_d = max(0, start_d)
    
    return int(start_w), int(start_h), int(start_d)


def translate_coordinates_crop(coords: torch.Tensor, 
                               crop_offset: Tuple[int, int, int]) -> torch.Tensor:
    """Translate coordinates from cropped volume back to original volume coordinates.
    
    Args:
        coords: (N, 3) tensor of coordinates in cropped volume (W, H, D order)
        crop_offset: (offset_w, offset_h, offset_d) crop starting position
        
    Returns:
        (N, 3) tensor of coordinates in original volume
    """
    if coords.shape[0] == 0:
        return coords
    
    offset_tensor = torch.tensor(crop_offset, device=coords.device, dtype=coords.dtype)
    return coords + offset_tensor


def translate_coordinates_downsample(coords: torch.Tensor,
                                     original_shape: Tuple[int, int, int],
                                     downsampled_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Translate coordinates from downsampled volume back to original volume coordinates.
    
    Uses the center of the corresponding region in the original volume.
    
    Args:
        coords: (N, 3) tensor of coordinates in downsampled volume (W, H, D order)
        original_shape: (W, H, D) original volume dimensions
        downsampled_shape: (W', H', D') downsampled volume dimensions
        
    Returns:
        (N, 3) tensor of coordinates in original volume
    """
    if coords.shape[0] == 0:
        return coords
    
    # Compute scale factors
    scale = torch.tensor([
        original_shape[0] / downsampled_shape[0],
        original_shape[1] / downsampled_shape[1],
        original_shape[2] / downsampled_shape[2]
    ], device=coords.device, dtype=torch.float32)
    
    # Map to center of corresponding region: coord_orig = (coord_down + 0.5) * scale - 0.5
    # Then round to nearest integer
    original_coords = ((coords.float() + 0.5) * scale - 0.5).round().long()
    
    # Clamp to valid range
    original_coords[:, 0] = original_coords[:, 0].clamp(0, original_shape[0] - 1)
    original_coords[:, 1] = original_coords[:, 1].clamp(0, original_shape[1] - 1)
    original_coords[:, 2] = original_coords[:, 2].clamp(0, original_shape[2] - 1)
    
    return original_coords


def compute_betti_matching_loss(prediction: Float[Tensor, "batch channels *spatial_dimensions"],
                                target: Float[Tensor, "batch channels *spatial_dimensions"],
                                cpu_batch_size: int,
                                sigmoid=False,
                                relative=False,
                                valid_mask: Float[Tensor, "batch channels *spatial_dimensions"] = None,
                                valid_bounds: typing.Optional[List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]] = None,
                                subsampling_size: typing.Optional[int] = None,
                                subsampling_mode: typing.Optional[str] = None,
                                ) -> List[torch.Tensor]:
    """Compute Betti matching loss for a batch of samples.
    
    Args:
        prediction: (B, 1, *spatial) tensor of prediction probabilities in [0, 1]
        target: (B, 1, *spatial) tensor of target values (0 or 1)
        cpu_batch_size: Batch size for CPU processing
        sigmoid: Whether to apply sigmoid to predictions
        relative: Whether to use relative homology (adds padding)
        valid_mask: Optional (B, 1, *spatial) boolean/float tensor. If provided,
                   only coordinates where valid_mask is True/1 contribute to the loss.
                   This is used to exclude ignore regions.
        valid_bounds: Optional list of precomputed valid region bounds per sample.
                     Each element is ((w_min, w_max), (h_min, h_max), (d_min, d_max)).
                     If provided, skips computing bounds from valid_mask for subsampling.
        subsampling_size: Optional int for subsampling in each spatial dimension.
        subsampling_mode: Subsampling mode, either "downsampling" or "random_crop" or "uncertainty_crop"
    
    Returns:
        List of loss tensors, one per sample in the batch
    """
    batch_size = prediction.shape[0]

    if sigmoid:
        prediction = torch.sigmoid(prediction)

    # Store original shape for coordinate translation
    original_spatial_dims = prediction.shape[2:]
    crop_offsets = None  # List of (w, h, d) offsets for each sample (for cropping)
    downsampled_shape = None  # Shape after downsampling
    
    if subsampling_mode is not None and subsampling_size is not None and subsampling_size < min(original_spatial_dims):
        spatial_dims = prediction.shape[2:]
        
        if subsampling_mode == "downsampling":
            # Downsample to specified size using trilinear interpolation
            target_size = (subsampling_size, subsampling_size, subsampling_size)
            
            # Interpolate prediction and target
            prediction = F.interpolate(prediction, size=target_size, mode='trilinear', align_corners=False)
            target = F.interpolate(target, size=target_size, mode='trilinear', align_corners=False)
            
            # Also downsample valid_mask if provided
            if valid_mask is not None:
                valid_mask = F.interpolate(valid_mask.float(), size=target_size, mode='trilinear', align_corners=False)
            
            downsampled_shape = target_size
            
        elif subsampling_mode in ["random_crop", "uncertainty_crop"]:
            # Process each sample individually with its own crop
            crop_offsets = []
            cropped_predictions = []
            cropped_targets = []
            cropped_masks = [] if valid_mask is not None else None
            valid_ratios = []
            
            for b in range(batch_size):
                sample_pred = prediction[b:b+1]  # Keep batch dim for consistency
                sample_target = target[b:b+1]
                sample_mask = valid_mask[b:b+1] if valid_mask is not None else None
                sample_bounds = valid_bounds[b] if valid_bounds is not None else None
                
                if subsampling_mode == "random_crop":
                    start_w, start_h, start_d = sample_random_crop(spatial_dims, subsampling_size, sample_mask, sample_bounds)
                else:  # uncertainty_crop
                    start_w, start_h, start_d = sample_uncertainty_crop(sample_pred, subsampling_size, sample_mask, sample_bounds)
                
                crop_offsets.append((start_w, start_h, start_d))
                
                # Calculate crop end points
                end_w = min(start_w + subsampling_size, spatial_dims[0])
                end_h = min(start_h + subsampling_size, spatial_dims[1])
                end_d = min(start_d + subsampling_size, spatial_dims[2])
                
                # Apply crop (keeping batch and channel dimensions)
                cropped_pred = sample_pred[:, :, start_w:end_w, start_h:end_h, start_d:end_d]
                cropped_tgt = sample_target[:, :, start_w:end_w, start_h:end_h, start_d:end_d]
                
                cropped_predictions.append(cropped_pred)
                cropped_targets.append(cropped_tgt)
                
                if (valid_mask is not None or valid_bounds is not None):
                    cropped_mask = sample_mask[:, :, start_w:end_w, start_h:end_h, start_d:end_d]
                    cropped_masks.append(cropped_mask)
                    valid_ratios.append(cropped_mask.mean().item())
            
            # Concatenate all cropped samples
            prediction = torch.cat(cropped_predictions, dim=0)
            target = torch.cat(cropped_targets, dim=0)
            if valid_mask is not None:
                valid_mask = torch.cat(cropped_masks, dim=0)

    # Using (1 - ...) to allow binary sorting optimization on the label, which expects values [0, 1]
    prediction = 1 - prediction
    target = 1 - target
    
    if relative:
        pad_value_prediction = prediction.min().item() # make sure to not propagate gradients here!
        pad_value_target = target.min().item()

        prediction = torch.nn.functional.pad(prediction, pad=[1 for _ in range(2 * (len(prediction.shape) - 2))], value=pad_value_prediction)
        target = torch.nn.functional.pad(target, pad=[1 for _ in range(2 * (len(target.shape) - 2))], value=pad_value_target)

    split_indices = np.arange(cpu_batch_size, prediction.shape[0], cpu_batch_size)
    predictions_list_numpy = np.split(prediction.detach().cpu().numpy().astype(np.float64), split_indices)
    targets_list_numpy = np.split(target.detach().cpu().numpy().astype(np.float64), split_indices)

    losses = []

    current_instance_index = 0
    for predictions_cpu_batch, targets_cpu_batch in zip(predictions_list_numpy, targets_list_numpy):
        predictions_cpu_batch = list(predictions_cpu_batch.squeeze(1))
        targets_cpu_batch = list(targets_cpu_batch.squeeze(1))
        
        # Ensure contiguous arrays for the C++ library
        if not (all(a.data.contiguous for a in predictions_cpu_batch) and all(a.data.contiguous for a in targets_cpu_batch)):
            global ENCOUNTERED_NONCONTIGUOUS
            if not ENCOUNTERED_NONCONTIGUOUS:
                print("WARNING! Non-contiguous arrays encountered. Shape:", predictions_cpu_batch[0].shape)
                ENCOUNTERED_NONCONTIGUOUS = True
        predictions_cpu_batch = [np.ascontiguousarray(a) for a in predictions_cpu_batch]
        targets_cpu_batch = [np.ascontiguousarray(a) for a in targets_cpu_batch]

        results = betti_matching.compute_matching(
            predictions_cpu_batch, 
            targets_cpu_batch, 
            include_input2_unmatched_pairs=False, 
            only_dim0=True
        )
        
        for result_arrays in results:
            # Get valid mask for this sample if provided
            sample_valid_mask = None
            if valid_mask is not None:
                sample_valid_mask = valid_mask[current_instance_index].squeeze(0)
            
            loss = _betti_matching_loss(
                prediction[current_instance_index].squeeze(0),
                target[current_instance_index].squeeze(0), 
                result_arrays,
                valid_mask=sample_valid_mask,
            )
            losses.append(loss)
            current_instance_index += 1

    return losses

def _betti_matching_loss_unmatched(unmatched_pairs: Float[Tensor, "M 2"]) -> Float[Tensor, "one_dimension"]:
    """Compute loss for unmatched persistence pairs.
    
    Args:
        unmatched_pairs: (M, 2) tensor of birth/death values for unmatched pairs
        
    Returns:
        Scalar loss tensor
    """
    if unmatched_pairs.shape[0] == 0:
        return torch.tensor(0.0, device=unmatched_pairs.device, dtype=unmatched_pairs.dtype)
    
    return ((unmatched_pairs[:, 0] - unmatched_pairs[:, 1])**2).sum()

def _betti_matching_loss(prediction: Float[Tensor, "*spatial_dimensions"],
                         target: Float[Tensor, "*spatial_dimensions"],
                         betti_matching_result: betti_matching.return_types.BettiMatchingResult,
                         valid_mask: Float[Tensor, "*spatial_dimensions"] = None,
                         ) -> Float[Tensor, "one_dimension"]:
    """Compute Betti matching loss for a single sample.
    
    Args:
        prediction: Spatial tensor of prediction values
        target: Spatial tensor of target values  
        betti_matching_result: Result from betti_matching.compute_matching
        valid_mask: Optional spatial tensor. If provided, only coordinates where
                   valid_mask is True/1 contribute to the loss. Pairs where either
                   birth or death coordinate is in an invalid region are excluded.
        
    Returns:
        Loss tensor of shape (1,)
    """
    ndim = prediction.ndim
    
    def safe_tensor(coord_list):
        """Convert list of coordinate arrays (one per homological dimension) to a single tensor.
        
        Args:
            coord_list: List of numpy arrays, each with shape (N_i, spatial_dims)
                       where N_i is the number of pairs for dimension i
        
        Returns:
            Tensor of shape (total_N, spatial_dims) or empty tensor
        """
        # Handle None or empty list
        if coord_list is None or len(coord_list) == 0:
            return torch.zeros(0, ndim, device=prediction.device, dtype=torch.long)
        
        # Concatenate all dimensions' coordinates
        all_coords = []
        for arr in coord_list:
            if arr is not None and arr.size > 0:
                all_coords.append(arr)
        
        if len(all_coords) == 0:
            return torch.zeros(0, ndim, device=prediction.device, dtype=torch.long)
        
        combined = np.concatenate(all_coords, axis=0)
        return torch.tensor(combined, device=prediction.device, dtype=torch.long)
    
    # Use correct attribute names: input1 = prediction, input2 = target
    prediction_matches_birth_coordinates = safe_tensor(betti_matching_result.input1_matched_birth_coordinates)
    prediction_matches_death_coordinates = safe_tensor(betti_matching_result.input1_matched_death_coordinates)
    target_matches_birth_coordinates = safe_tensor(betti_matching_result.input2_matched_birth_coordinates)
    target_matches_death_coordinates = safe_tensor(betti_matching_result.input2_matched_death_coordinates)
    prediction_unmatched_birth_coordinates = safe_tensor(betti_matching_result.input1_unmatched_birth_coordinates)
    prediction_unmatched_death_coordinates = safe_tensor(betti_matching_result.input1_unmatched_death_coordinates)

    def get_values_at_coords(tensor, coords):
        """Extract values from tensor at given coordinates."""
        if coords.shape[0] == 0:
            return torch.zeros(0, device=tensor.device, dtype=tensor.dtype)
        return tensor[tuple(coords[:, i] for i in range(coords.shape[1]))]
    
    def filter_valid_pairs(birth_coords, death_coords):
        """Filter coordinate pairs to only include those where both birth and death are in valid regions.
        
        Returns:
            Tuple of (filtered_birth_coords, filtered_death_coords)
        """
        if valid_mask is None or birth_coords.shape[0] == 0:
            return birth_coords, death_coords
        
        # Check validity at both birth and death coordinates
        birth_valid = get_values_at_coords(valid_mask, birth_coords) > 0.5
        death_valid = get_values_at_coords(valid_mask, death_coords) > 0.5
        
        # Only keep pairs where both coordinates are valid
        both_valid = birth_valid & death_valid
        
        return birth_coords[both_valid], death_coords[both_valid]
    
    # Filter matched pairs based on valid_mask
    # For matched pairs, we need to filter jointly - if ANY coordinate is invalid, remove the whole pair
    if valid_mask is not None and prediction_matches_birth_coordinates.shape[0] > 0:
        pred_birth_valid = get_values_at_coords(valid_mask, prediction_matches_birth_coordinates) > 0.5
        pred_death_valid = get_values_at_coords(valid_mask, prediction_matches_death_coordinates) > 0.5
        tgt_birth_valid = get_values_at_coords(valid_mask, target_matches_birth_coordinates) > 0.5
        tgt_death_valid = get_values_at_coords(valid_mask, target_matches_death_coordinates) > 0.5
        
        # Only keep matched pairs where ALL four coordinates are valid
        all_valid = pred_birth_valid & pred_death_valid & tgt_birth_valid & tgt_death_valid
        
        prediction_matches_birth_coordinates = prediction_matches_birth_coordinates[all_valid]
        prediction_matches_death_coordinates = prediction_matches_death_coordinates[all_valid]
        target_matches_birth_coordinates = target_matches_birth_coordinates[all_valid]
        target_matches_death_coordinates = target_matches_death_coordinates[all_valid]
    
    # Filter unmatched pairs independently (they don't need to match anything)
    prediction_unmatched_birth_coordinates, prediction_unmatched_death_coordinates = filter_valid_pairs(
        prediction_unmatched_birth_coordinates, prediction_unmatched_death_coordinates
    )

    # (M, 2) tensor of matched persistence pairs for prediction
    prediction_matched_pairs = torch.stack([
        get_values_at_coords(prediction, prediction_matches_birth_coordinates),
        get_values_at_coords(prediction, prediction_matches_death_coordinates)
    ], dim=1) if prediction_matches_birth_coordinates.shape[0] > 0 else torch.zeros(0, 2, device=prediction.device)
    
    # (M, 2) tensor of matched persistence pairs for target
    target_matched_pairs = torch.stack([
        get_values_at_coords(target, target_matches_birth_coordinates),
        get_values_at_coords(target, target_matches_death_coordinates)
    ], dim=1) if target_matches_birth_coordinates.shape[0] > 0 else torch.zeros(0, 2, device=target.device)

    # (M, 2) tensor of unmatched persistence pairs for prediction
    prediction_unmatched_pairs = torch.stack([
        get_values_at_coords(prediction, prediction_unmatched_birth_coordinates),
        get_values_at_coords(prediction, prediction_unmatched_death_coordinates)
    ], dim=1) if prediction_unmatched_birth_coordinates.shape[0] > 0 else torch.zeros(0, 2, device=prediction.device)

    # Compute matched loss: penalize differences between matched pairs
    if prediction_matched_pairs.shape[0] > 0:
        loss_matched = 2 * ((prediction_matched_pairs - target_matched_pairs) ** 2).sum()
    else:
        loss_matched = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)

    # Compute unmatched prediction loss: push unmatched pairs toward diagonal
    loss_unmatched_prediction = _betti_matching_loss_unmatched(prediction_unmatched_pairs)
    
    loss = loss_matched + loss_unmatched_prediction
    return loss.reshape(1)