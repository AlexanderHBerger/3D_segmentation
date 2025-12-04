"""
Metrics for segmentation evaluation
"""
import torch
from typing import Dict, Tuple
import torch.nn.functional as F


def prepare_targets_with_mask(
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Centralized utility to prepare targets and create valid mask.
    
    This function handles the common pattern of:
    1. Creating a mask for valid voxels (excluding ignore_index)
    2. Clamping targets to valid range [0, num_classes)
    
    Args:
        targets: (B, H, W, D) - class indices
        num_classes: Number of classes
        ignore_index: Index to ignore in target
    
    Returns:
        Tuple of (valid_mask, targets_clamped, num_valid_voxels):
        - valid_mask: (B, H, W, D) - boolean mask where targets != ignore_index
        - targets_clamped: (B, H, W, D) - targets clamped to [0, num_classes)
        - num_valid_voxels: scalar - total number of valid voxels
    """
    # Create mask for valid voxels (not ignore_index)
    valid_mask = targets != ignore_index
    
    # Clamp targets to valid range [0, num_classes) for safety
    targets_clamped = torch.clamp(targets, 0, num_classes - 1)
    
    # Count valid voxels
    num_valid_voxels = valid_mask.sum()
    
    return valid_mask, targets_clamped, num_valid_voxels


def dice_coefficient(
    predictions: torch.Tensor,
    targets_clamped: torch.Tensor,
    valid_mask: torch.Tensor,
    include_background: bool = False,
    smooth: float = 1e-5
) -> torch.Tensor:
    """
    Calculate Dice coefficient (soft Dice, sample-wise) for 3D segmentation.
    Computes Dice per sample and averages over the batch.
    
    Args:
        predictions: (B, C, H, W, D) - logits where C is num_classes
        targets_clamped: (B, H, W, D) - class indices clamped to [0, num_classes)
        valid_mask: (B, H, W, D) - boolean mask for valid voxels
        include_background: Whether to include background class in calculation
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient per class (averaged over batch, excluding background if include_background=False)
        
    Raises:
        ValueError: If input dimensions are not 3D or shapes don't match expected format
    """
    # Validate input dimensions - must be 3D
    if predictions.dim() != 5:
        raise ValueError(
            f"Expected 5D predictions tensor (B, C, H, W, D), got shape {predictions.shape}. "
            f"Predictions must be logits with shape (batch, num_classes, height, width, depth)."
        )
    
    if targets_clamped.dim() != 4:
        raise ValueError(
            f"Expected 4D targets tensor (B, H, W, D), got shape {targets_clamped.shape}. "
            f"Targets must be class indices with shape (batch, height, width, depth)."
        )
    
    # Validate spatial dimensions match
    if predictions.shape[0] != targets_clamped.shape[0]:
        raise ValueError(f"Batch size mismatch: predictions {predictions.shape[0]} vs targets {targets_clamped.shape[0]}")
    
    if predictions.shape[2:] != targets_clamped.shape[1:]:
        raise ValueError(f"Spatial dimensions mismatch: predictions {predictions.shape[2:]} vs targets {targets_clamped.shape[1:]}")
    
    # Extract number of classes and convert predictions to probabilities
    num_classes = predictions.shape[1]
    probs = F.softmax(predictions, dim=1)
    
    # Convert targets to one-hot: (B, H, W, D) -> (B, C, H, W, D)
    targets_one_hot = F.one_hot(targets_clamped.long(), num_classes=num_classes)
    targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
    
    # Apply valid mask
    valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(probs)
    probs = probs * valid_mask_expanded
    targets_one_hot = targets_one_hot * valid_mask_expanded
    
    # Calculate Dice per sample
    # Flatten spatial dims: (B, C, H, W, D) -> (B, C, N)
    probs_flat = probs.reshape(probs.shape[0], num_classes, -1)
    target_flat = targets_one_hot.reshape(targets_one_hot.shape[0], num_classes, -1)
    
    intersection = (probs_flat * target_flat).sum(dim=2)  # (B, C)
    union = probs_flat.sum(dim=2) + target_flat.sum(dim=2)  # (B, C)
    
    dice_scores = (2.0 * intersection + smooth) / (union + smooth)  # (B, C)
    
    # Select classes (exclude background if needed)
    start_idx = 0 if include_background else 1
    dice_scores = dice_scores[:, start_idx:]  # (B, C_subset)
    
    # Return mean over batch
    return dice_scores.mean(dim=0)


def hard_dice_coefficient(
    predictions: torch.Tensor,
    targets_clamped: torch.Tensor,
    valid_mask: torch.Tensor,
    include_background: bool = False,
    smooth: float = 1e-5
) -> torch.Tensor:
    """
    Calculate Hard Dice coefficient (sample-wise) for 3D segmentation.
    Computes Hard Dice (using argmax) averaged over the batch.
    
    Args:
        predictions: (B, C, H, W, D) - logits where C is num_classes
        targets_clamped: (B, H, W, D) - class indices clamped to [0, num_classes)
        valid_mask: (B, H, W, D) - boolean mask for valid voxels
        include_background: Whether to include background class in calculation
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient per class (averaged over batch)
    """
    num_classes = predictions.shape[1]
    
    # Get hard predictions (argmax)
    pred_indices = torch.argmax(predictions, dim=1) # (B, H, W, D)
    
    # Convert to one-hot
    pred_one_hot = F.one_hot(pred_indices, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    targets_one_hot = F.one_hot(targets_clamped.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    
    # Apply valid mask
    valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(pred_one_hot)
    pred_one_hot = pred_one_hot * valid_mask_expanded
    targets_one_hot = targets_one_hot * valid_mask_expanded
    
    # Calculate Dice per sample
    # Flatten spatial dims: (B, C, H, W, D) -> (B, C, N)
    pred_flat = pred_one_hot.reshape(pred_one_hot.shape[0], num_classes, -1)
    target_flat = targets_one_hot.reshape(targets_one_hot.shape[0], num_classes, -1)
    
    intersection = (pred_flat * target_flat).sum(dim=2) # (B, C)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) # (B, C)
    
    dice_scores = (2.0 * intersection + smooth) / (union + smooth) # (B, C)
    
    # Select classes
    start_idx = 0 if include_background else 1
    dice_scores = dice_scores[:, start_idx:] # (B, C_subset)
    
    # Return mean over batch
    return dice_scores.mean(dim=0)


class MetricsCalculator:
    """Class to calculate and accumulate metrics during training/validation"""
    
    def __init__(self, num_classes: int = 2, include_background: bool = False, ignore_index: int = -1, compute_calibration: bool = True, num_bins: int = 15):
        self.num_classes = num_classes
        self.include_background = include_background
        self.ignore_index = ignore_index
        self.compute_calibration = compute_calibration
        self.num_bins = num_bins
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.dice_scores = []
        self.hard_dice_scores = []
        self.total_samples = 0
        
        # Calibration metrics accumulators
        if self.compute_calibration:
            self.ece_scores = []
            self.mce_scores = []
            self.nll_scores = []
            self.brier_scores = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Update metrics with new batch
        
        Args:
            predictions: (B, C, H, W, D) - logits
            targets: (B, H, W, D) - class indices
            
        Returns:
            Dictionary of metrics for the current batch
        """
        batch_size = predictions.shape[0]
        self.total_samples += batch_size
        num_classes = predictions.shape[1]
        
        # Perform masking once here (centralized)
        valid_mask, targets_clamped, num_valid_voxels = prepare_targets_with_mask(
            targets, num_classes, self.ignore_index
        )
        
        # Calculate dice metric (pass mask info)
        dice = dice_coefficient(
            predictions, targets_clamped, valid_mask,
            include_background=self.include_background,
            smooth=1e-5
        )
        
        # Accumulate dice scores
        self.dice_scores.append(dice)
        
        # Calculate hard dice
        hard_dice = hard_dice_coefficient(
            predictions, targets_clamped, valid_mask,
            include_background=self.include_background,
            smooth=1e-5
        )
        self.hard_dice_scores.append(hard_dice)
        
        batch_metrics = {
            'dice_mean': dice.mean().item(),
            'dice_hard': hard_dice.mean().item()
        }
        
        # Calculate calibration metrics if enabled (pass mask info)
        if self.compute_calibration:
            cal_metrics = compute_calibration_metrics(
                predictions, targets, targets_clamped, valid_mask, num_valid_voxels,
                num_bins=self.num_bins, ignore_index=self.ignore_index
            )
            self.ece_scores.append(cal_metrics['ece'])
            self.mce_scores.append(cal_metrics['mce'])
            self.nll_scores.append(cal_metrics['nll'])
            self.brier_scores.append(cal_metrics['brier_score'])
            
            batch_metrics.update(cal_metrics)
            
        return batch_metrics
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        if not self.dice_scores:
            return {}
        
        # Stack and average dice
        dice_mean = torch.stack(self.dice_scores).mean(dim=0)
        hard_dice_mean = torch.stack(self.hard_dice_scores).mean(dim=0)
        
        results = {
            'dice_mean': dice_mean.mean().item(),
            'dice_hard': hard_dice_mean.mean().item(),
        }
        
        # Add calibration metrics if computed
        if self.compute_calibration and self.ece_scores:
            results['ece'] = sum(self.ece_scores) / len(self.ece_scores)
            results['mce'] = sum(self.mce_scores) / len(self.mce_scores)
            results['nll'] = sum(self.nll_scores) / len(self.nll_scores)
            results['brier_score'] = sum(self.brier_scores) / len(self.brier_scores)
        
        return results
    
    def get_main_metric(self) -> float:
        """Get main metric for model selection (mean Dice)"""
        if not self.dice_scores:
            return 0.0
        
        dice_mean = torch.stack(self.dice_scores).mean(dim=0)
        return dice_mean.mean().item()

def compute_calibration_metrics(
    logits: torch.Tensor, 
    targets: torch.Tensor,
    targets_clamped: torch.Tensor,
    valid_mask: torch.Tensor,
    num_valid_voxels: torch.Tensor,
    num_bins: int = 15,
    ignore_index: int = -1
):
    """
    Compute calibration metrics for 3D segmentation (sample-wise, then averaged).
    
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Negative Log-Likelihood (NLL)
    - Brier Score

    Args:
        logits (torch.Tensor): Model logits of shape (B, C, H, W, D) where C is num_classes
        targets (torch.Tensor): Original ground truth class indices (may contain ignore_index)
        targets_clamped (torch.Tensor): Clamped targets to [0, num_classes)
        valid_mask (torch.Tensor): Boolean mask for valid voxels
        num_valid_voxels (torch.Tensor): Total count of valid voxels (unused, kept for API compatibility)
        num_bins (int): Number of bins for ECE/MCE computation
        ignore_index (int): Index to ignore in target (used for NLL computation)

    Returns:
        dict: Dictionary containing calibration metrics (averaged over samples)
        
    Raises:
        ValueError: If input dimensions are not 3D or shapes don't match expected format
    """
    
    # Validate input dimensions - must be 3D
    if logits.dim() != 5:
        raise ValueError(f"Expected 5D logits tensor (B, C, H, W, D), got shape {logits.shape}")
    
    if targets.dim() != 4:
        raise ValueError(f"Expected 4D targets tensor (B, H, W, D), got shape {targets.shape}")
    
    # Validate spatial dimensions match
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(f"Batch size mismatch: logits {logits.shape[0]} vs targets {targets.shape[0]}")
    
    if logits.shape[2:] != targets.shape[1:]:
        raise ValueError(f"Spatial dimensions mismatch: logits {logits.shape[2:]} vs targets {targets.shape[1:]}")

    with torch.no_grad():
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Storage for per-sample metrics
        ece_samples = []
        mce_samples = []
        nll_samples = []
        brier_samples = []

        # Compute metrics per sample
        for b in range(batch_size):
            # Get single sample tensors
            probs_b = probs[b]  # (C, H, W, D)
            targets_b = targets[b]  # (H, W, D)
            targets_clamped_b = targets_clamped[b]  # (H, W, D)
            valid_mask_b = valid_mask[b]  # (H, W, D)
            
            # Flatten spatial dimensions for per-voxel metrics
            # (C, H, W, D) -> (H*W*D, C)
            probs_flat = probs_b.permute(1, 2, 3, 0).reshape(-1, num_classes)
            targets_flat = targets_clamped_b.reshape(-1)
            valid_mask_flat = valid_mask_b.reshape(-1)
            
            # Filter to only valid voxels
            probs_valid = probs_flat[valid_mask_flat]
            targets_valid = targets_flat[valid_mask_flat]
            num_valid = valid_mask_flat.sum()
            
            # Skip sample if no valid voxels
            if num_valid == 0:
                continue

            # Get predicted class and confidence
            confidences, predictions = torch.max(probs_valid, dim=1)
            accuracies = predictions.eq(targets_valid).float()

            # 1. Expected Calibration Error (ECE)
            ece = compute_ece(confidences, accuracies, num_bins)
            ece_samples.append(ece.item())

            # 2. Maximum Calibration Error (MCE)
            mce = compute_mce(confidences, accuracies, num_bins)
            mce_samples.append(mce.item())

            # 3. Negative Log-Likelihood (NLL) - per sample
            logits_b = logits[b:b+1]  # Keep batch dim: (1, C, H, W, D)
            targets_b_orig = targets[b:b+1]  # Keep batch dim: (1, H, W, D)
            nll = F.cross_entropy(logits_b, targets_b_orig, reduction='mean', ignore_index=ignore_index)
            nll_samples.append(nll.item())

            # 4. Brier Score per sample
            # Convert targets to one-hot: (H, W, D) -> (C, H, W, D)
            targets_onehot_b = F.one_hot(targets_clamped_b.long(), num_classes=num_classes)
            targets_onehot_b = targets_onehot_b.permute(3, 0, 1, 2).float()  # (C, H, W, D)
            
            # Apply valid mask
            valid_mask_expanded = valid_mask_b.unsqueeze(0).expand_as(probs_b)
            probs_masked = probs_b * valid_mask_expanded
            targets_onehot_masked = targets_onehot_b * valid_mask_expanded
            
            # Compute Brier score only over valid voxels
            brier_score = ((probs_masked - targets_onehot_masked) ** 2).sum() / num_valid
            brier_samples.append(brier_score.item())

        # Average across samples (handle empty case)
        if len(ece_samples) == 0:
            return {
                'ece': 0.0,
                'mce': 0.0,
                'nll': 0.0,
                'brier_score': 0.0
            }
        
        return {
            'ece': sum(ece_samples) / len(ece_samples),
            'mce': sum(mce_samples) / len(mce_samples),
            'nll': sum(nll_samples) / len(nll_samples),
            'brier_score': sum(brier_samples) / len(brier_samples)
        }


def compute_ece(confidences: torch.Tensor, accuracies: torch.Tensor, num_bins: int = 15):
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy,
    averaged across confidence bins weighted by the proportion of samples in each bin.

    Args:
        confidences (torch.Tensor): Predicted confidences (max probabilities) of shape (N,)
        accuracies (torch.Tensor): Binary accuracy (1 if correct, 0 if wrong) of shape (N,)
        num_bins (int): Number of bins to discretize confidence scores

    Returns:
        torch.Tensor: ECE value (scalar)
    """
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=confidences.device)
    ece = torch.zeros(1, device=confidences.device)

    for i in range(num_bins):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def compute_mce(confidences: torch.Tensor, accuracies: torch.Tensor, num_bins: int = 15):
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE measures the worst-case calibration error across all confidence bins,
    representing the maximum deviation between confidence and accuracy.

    Args:
        confidences (torch.Tensor): Predicted confidences (max probabilities) of shape (N,)
        accuracies (torch.Tensor): Binary accuracy (1 if correct, 0 if wrong) of shape (N,)
        num_bins (int): Number of bins to discretize confidence scores

    Returns:
        torch.Tensor: MCE value (scalar)
    """
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=confidences.device)
    mce = torch.zeros(1, device=confidences.device)

    for i in range(num_bins):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            calibration_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            mce = torch.max(mce, calibration_error)

    return mce