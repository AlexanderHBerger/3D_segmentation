"""
Loss functions for segmentation training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd
import numpy as np
from typing import List, Optional

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean', ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W, D) - logits
            targets: (B, H, W, D) - class indices
        """
        predictions = predictions.float()
        
        # Create mask for valid pixels
        valid_mask = targets != self.ignore_index
        
        # Use ignore_index in cross_entropy (PyTorch handles this automatically)
        ce_loss = F.cross_entropy(predictions, targets.long(), reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply valid mask for manual reduction
        if self.reduction == 'mean':
            # Only average over valid pixels
            return (focal_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return (focal_loss * valid_mask).sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss with optional Dice++ support.
    Computes 1 - Dice (or 1 - Dice++)
    
    When plus_plus=True and use_fixed_grad_softmax=True, the Dice++ specific
    gradient correction is automatically applied (requires exponential_correction >= 20).
    """
    def __init__(
        self,
        smooth: float = 1e-5,
        include_background: bool = False,
        ignore_index: int = -1,
        use_fixed_grad_softmax: bool = False,
        exponential_correction: Optional[int] = None,
        plus_plus: bool = False,
        gamma: float = 2.0,
        tversky_beta: float = 0.5
    ):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.ignore_index = ignore_index
        self.use_fixed_grad_softmax = use_fixed_grad_softmax
        self.exponential_correction = exponential_correction
        self.plus_plus = plus_plus
        self.gamma = gamma
        self.tversky_beta = tversky_beta
        
        # Validation for dice_plus_plus with fixed_grad_softmax
        if plus_plus and use_fixed_grad_softmax:
            if exponential_correction is None or exponential_correction < 20:
                raise ValueError(
                    f"dice_plus_plus with use_fixed_grad_softmax requires exponential_correction >= 20, "
                    f"but got {exponential_correction}. Please set exponential_correction >= 20 in config."
                )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W, D) - logits
            targets: (B, H, W, D) - class indices
        """
        # Create mask for valid pixels
        valid_mask = targets != self.ignore_index
        
        # Compute probabilities with selected softmax variant
        num_classes = predictions.shape[1]
        targets_clamped = torch.clamp(targets, 0, num_classes - 1)

        # Convert targets to one-hot for Dice calculation
        targets_one_hot = torch.zeros_like(predictions)
        targets_one_hot.scatter_(1, targets_clamped.unsqueeze(1).long(), 1.0)
        
        if self.use_fixed_grad_softmax:
            # Apply FixedGradSoftmax (requires one-hot targets)
            # When plus_plus=True, use Dice++ specific gradient correction
            probs = FixedGradSoftmax.apply(
                predictions, targets_one_hot, self.exponential_correction, self.plus_plus
            )
        else:
            # Standard softmax
            probs = F.softmax(predictions, dim=1)
        
        # Apply valid mask to both predictions and targets
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(probs)
        probs = probs * valid_mask_expanded
        targets_one_hot = targets_one_hot * valid_mask_expanded
        
        # Vectorized Dice calculation
        start_idx = 0 if self.include_background else 1
        pred_subset = probs[:, start_idx:]
        target_subset = targets_one_hot[:, start_idx:]
        
        # Flatten
        pred_flat = pred_subset.contiguous().view(pred_subset.shape[0], pred_subset.shape[1], -1)
        target_flat = target_subset.contiguous().view(target_subset.shape[0], target_subset.shape[1], -1)
        
        tp = (pred_flat * target_flat).sum(dim=2)
        fp = ((1.0 - target_flat) * pred_flat)
        fn = (target_flat * (1.0 - pred_flat))
        
        if self.plus_plus:
            # Scaling FP = sum((p0 * y1)^gamma)
            fp = fp.pow(self.gamma)
            
            # Scaling FN = sum((p1 * y0)^gamma)
            fn = fn.pow(self.gamma)
        elif self.tversky_beta != 0.5:
            # Scaling FP and FN for Tversky loss
            fp = 2*self.tversky_beta * fp
            fn = 2*(1 - self.tversky_beta) * fn

        fp = fp.sum(dim=2)
        fn = fn.sum(dim=2)
        dice_score = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
            
        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    """Combined Dice (or Dice++/Tversky) + CrossEntropy Loss"""
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
        include_background: bool = False,
        ignore_index: int = -1,
        use_fixed_grad_softmax: bool = False,
        exponential_correction: Optional[int] = None,
        dice_plus_plus: bool = False,
        gamma: float = 2.0,
        tversky_beta: float = 0.5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(
            smooth=smooth,
            include_background=include_background,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=use_fixed_grad_softmax,
            exponential_correction=exponential_correction,
            plus_plus=dice_plus_plus,
            gamma=gamma,
            tversky_beta=tversky_beta
        )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W, D) - logits
            targets: (B, H, W, D) - class indices
        """
        # Only compute CE loss if it has non-zero weight
        if self.ce_weight > 0:
            ce = self.ce_loss(predictions, targets.long())
        else:
            ce = 0.0
        
        # Only compute Dice loss if it has non-zero weight
        if self.dice_weight > 0:
            dice = self.dice_loss(predictions, targets)
        else:
            dice = 0.0
        
        return self.dice_weight * dice + self.ce_weight * ce


class DeepSupervisionLoss(nn.Module):
    """Deep supervision loss for multi-scale outputs"""
    
    def __init__(
        self,
        loss_fn: nn.Module,
        weights: List[float],
        downsampling_scales: Optional[List[int]] = None
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights
        self.downsampling_scales = downsampling_scales or [1, 2, 4, 8, 16]
    
    def forward(self, predictions: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: List of predictions at different scales [(B, C, H, W, D), ...]
            targets: (B, H, W, D) - class indices
        """
        total_loss = 0.0
        
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            if weight == 0:
                continue
            
            # Downsample targets if needed
            if i < len(self.downsampling_scales) and self.downsampling_scales[i] > 1:
                scale = self.downsampling_scales[i]
                target_scaled = F.interpolate(
                    targets.unsqueeze(1).float(),
                    scale_factor=1.0/scale,
                    mode='nearest'
                ).squeeze(1).long()
            else:
                target_scaled = targets
            
            # Resize prediction to match target if needed
            if pred.shape[2:] != target_scaled.shape[1:]:
                pred_resized = F.interpolate(
                    pred,
                    size=target_scaled.shape[1:],
                    mode='trilinear',
                    align_corners=False
                )
            else:
                pred_resized = pred
            
            loss = self.loss_fn(pred_resized, target_scaled)
            total_loss += weight * loss
        
        return total_loss


def get_loss_function(config) -> nn.Module:
    """Get loss function based on configuration"""
    ignore_index = getattr(config.training, 'ignore_index', -1)
    use_fixed_grad_softmax = getattr(config.training, 'use_fixed_grad_softmax', False)
    exponential_correction = getattr(config.training, 'exponential_correction', None)
    
    if config.training.loss_function == "dice":
        return CombinedLoss(
            dice_weight=1.0,
            ce_weight=0.0,
            smooth=1e-5,
            include_background=False,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=use_fixed_grad_softmax,
            exponential_correction=exponential_correction,
            dice_plus_plus=False
        )
    
    elif config.training.loss_function == "ce":
        # throw error if use_fixed_grad_softmax is set
        if use_fixed_grad_softmax:
            raise NotImplementedError("use_fixed_grad_softmax is not implemented for CrossEntropy loss")
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    elif config.training.loss_function == "focal":
        return FocalLoss(alpha=1.0, gamma=2.0, ignore_index=ignore_index)
    
    elif config.training.loss_function == "dice_ce":
        return CombinedLoss(
            dice_weight=config.training.dice_weight,
            ce_weight=config.training.ce_weight,
            smooth=1e-5,
            include_background=False,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=use_fixed_grad_softmax,
            exponential_correction=exponential_correction,
            dice_plus_plus=False
        )
    
    elif config.training.loss_function == "dice_plus_plus":
        gamma = getattr(config.training, 'dice_plus_plus_gamma', 2.0)
        return CombinedLoss(
            dice_weight=1.0,
            ce_weight=0.0,
            smooth=1e-5,
            include_background=False,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=use_fixed_grad_softmax,
            exponential_correction=exponential_correction,
            dice_plus_plus=True,
            gamma=gamma
        )
    
    elif config.training.loss_function == "dice_plus_plus_ce":
        gamma = getattr(config.training, 'dice_plus_plus_gamma', 2.0)
        return CombinedLoss(
            dice_weight=config.training.dice_weight,
            ce_weight=config.training.ce_weight,
            smooth=1e-5,
            include_background=False,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=use_fixed_grad_softmax,
            exponential_correction=exponential_correction,
            dice_plus_plus=True,
            gamma=gamma
        )
    
    elif config.training.loss_function == "tversky":
        tversky_beta = getattr(config.training, 'tversky_beta', 0.5)
        return CombinedLoss(
            dice_weight=1.0,
            ce_weight=0.0,
            smooth=1e-5,
            include_background=False,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=use_fixed_grad_softmax,
            exponential_correction=exponential_correction,
            dice_plus_plus=False,
            tversky_beta=tversky_beta
        )
    
    elif config.training.loss_function == "tversky_ce":
        tversky_beta = getattr(config.training, 'tversky_beta', 0.5)
        return CombinedLoss(
            dice_weight=config.training.dice_weight,
            ce_weight=config.training.ce_weight,
            smooth=1e-5,
            include_background=False,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=use_fixed_grad_softmax,
            exponential_correction=exponential_correction,
            dice_plus_plus=False,
            tversky_beta=tversky_beta
        )
    
    else:
        raise ValueError(f"Unknown loss function: {config.training.loss_function}")


def get_deep_supervision_loss(config, base_loss_fn: nn.Module, n_stages: Optional[int] = None) -> DeepSupervisionLoss:
    """
    Get deep supervision loss wrapper with dynamically calculated weights.
    
    Args:
        config: Configuration object
        base_loss_fn: Base loss function to wrap
        n_stages: Number of network stages (REQUIRED - must be provided by caller)
    
    Returns:
        DeepSupervisionLoss with properly configured weights
    """
    # n_stages must be provided by the caller (from model architecture)
    if n_stages is None:
        raise ValueError(
            "n_stages must be provided to get_deep_supervision_loss. "
            "Calculate it using architectures.calculate_n_stages(patch_size) or "
            "extract it from the model architecture."
        )
    
    # Calculate deep supervision weights following nnUNet pattern
    weights = np.array([1 / (2 ** i) for i in range(n_stages)])
    weights[-1] = 0  # nnUNet sets the last (deepest/lowest resolution) layer to 0
    weights = weights / weights.sum()  # Normalize to sum to 1
    
    # Calculate downsampling scales (cumulative product of strides)
    # First stage has stride 1, subsequent stages have stride 2
    downsampling_scales = [2 ** i if i > 0 else 1 for i in range(n_stages)]
    
    return DeepSupervisionLoss(
        loss_fn=base_loss_fn,
        weights=weights.tolist(),
        downsampling_scales=downsampling_scales
    )


class FixedGradSoftmax(torch.autograd.Function):
    """
    Custom softmax with hybrid gradient in backward pass.
    Forward: Computes softmax and hybrid gradient weights
    Backward: Uses precomputed gradient weights for efficient computation

    This hybrid approach:
    - For small errors (|p_i - y_i| < 0.5): uses standard softmax gradient p*(1-p)
      This preserves normal gradient behavior when predictions are reasonably close to targets
    - For large errors (|p_i - y_i| >= 0.5): uses fixed gradient 0.25
      This avoids vanishing gradients when predictions are very wrong
    - Computes gradient weights in forward for efficiency
    
    Args:
        logits: Input logits tensor
        targets: One-hot encoded targets
        exponential_correction: Optional exponential parameter for tunable gradient weighting.
            If None (default): weight = 0.25 * |probs - targets| (original behavior)
            If positive int: weight = 0.25 * (error - error^exponential_correction)
                - Higher values → stronger suppression of high probability gradients
                - Lower values → gradients closer to linear scaling
        dice_plus_plus: If True, uses Dice++ specific gradient correction.
            Tunable gradient weight: 0.25 * (1 - error^exponential)
            For high errors (near 1), error^exponential ≈ 1, so weight ≈ 0
            For low errors (near 0), error^exponential ≈ 0, so weight ≈ 0.5
            Additionally, for foreground (y=1), gradient is suppressed based on prediction confidence.
    """
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, logits, targets, exponential_correction=None, dice_plus_plus=False):
        probs = torch.softmax(logits, dim=1)

        error = torch.abs(probs - targets)
        
        if dice_plus_plus:
            # Dice++ specific gradient correction
            error_weight = 0.25 * (1 - torch.pow(error, exponential_correction))
            
            # For foreground (y=1), suppress gradient based on prediction confidence
            # This ensures gradient is zero when prediction is already correct
            error_weight[targets == 1] = error_weight[targets == 1] * (1 - torch.pow(probs[targets == 1], 20))
            error_weight[targets == 0] = error_weight[targets == 0] * (1 - torch.pow(1 - probs[targets == 0], 20))
        elif exponential_correction is not None:
            # Tunable gradient weight: 0.25 * (error - error^exponential)
            error_weight = 0.25 * (error - torch.pow(error, exponential_correction))
        else:
            # Original fixed gradient weight
            error_weight = 0.25 * error

        # Only save the scaled error weight, not probs and targets
        ctx.save_for_backward(error_weight)
        return probs

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        error_weight, = ctx.saved_tensors

        # Convert grad_output to the same dtype as error_weight for mixed precision training
        grad_output = grad_output.to(error_weight.dtype)

        weight = error_weight[:, 1:2]  # Single weight
        grad_p_bg = grad_output[:, 0:1]
        grad_p_fg = grad_output[:, 1:2]

        # Compute coupling term once
        coupling = (grad_p_fg - grad_p_bg)

        grad_logits_bg = -weight * coupling
        grad_logits_fg = weight * coupling

        return torch.cat([grad_logits_bg, grad_logits_fg], dim=1), None, None, None