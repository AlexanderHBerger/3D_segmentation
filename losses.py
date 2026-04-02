"""
Loss functions for segmentation training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd
import numpy as np
from typing import List, Optional

try:
    from topograph import TopographLoss
    TOPOGRAPH_AVAILABLE = True
except ImportError:
    TOPOGRAPH_AVAILABLE = False

try:
    from betti_matching_loss import compute_betti_matching_loss
    BETTI_MATCHING_AVAILABLE = True
except ImportError:
    BETTI_MATCHING_AVAILABLE = False


class CrossEntropyLossWrapper(nn.Module):
    """Wrapper around CrossEntropyLoss that returns (loss, components_dict)"""
    
    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Args:
            predictions: (B, C, H, W, D) - logits
            targets: (B, H, W, D) - class indices
            **kwargs: Additional arguments (ignored for this loss)
        
        Returns:
            Tuple of (loss, components_dict) where components_dict contains the loss components
        """
        loss = self.ce_loss(predictions, targets.long())
        return loss, {'ce': loss}


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean', ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Args:
            predictions: (B, C, H, W, D) - logits
            targets: (B, H, W, D) - class indices
            **kwargs: Additional arguments (ignored for this loss)
        
        Returns:
            Tuple of (loss, components_dict) where components_dict contains the loss components
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
            loss = (focal_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            loss = (focal_loss * valid_mask).sum()
        else:
            loss = focal_loss
        
        return loss, {'focal': loss}


class DiceLoss(nn.Module):
    """
    Dice Loss with optional Dice++ and clDice support.
    Computes 1 - Dice (or 1 - Dice++), optionally combined with clDice.
    
    When plus_plus=True and use_fixed_grad_softmax=True, the Dice++ specific
    gradient correction is automatically applied (requires exponential_correction >= 20).
    
    When cldice_alpha > 0, combines standard Dice with clDice (skeleton-based Dice):
        loss = (1 - cldice_alpha) * dice_loss + cldice_alpha * cldice_loss
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
        tversky_beta: float = 0.5,
        cldice_alpha: float = 0.0,
        binary: bool = False,
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
        self.cldice_alpha = cldice_alpha
        self.binary = binary
        
        # Initialize skeletonizer if clDice is enabled
        if self.cldice_alpha > 0:
            self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        
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
            targets: (B, H, W, D) class indices OR (B, N, H, W, D) binary masks when binary=True
        """
        if self.binary:
            # Binary mode: predictions (B, N, H, W, D), targets (B, N, H, W, D)
            probs = torch.sigmoid(predictions)
            # Flatten spatial dims for Dice
            p_flat = probs.reshape(probs.shape[0], probs.shape[1], -1)
            t_flat = targets.float().reshape(targets.shape[0], targets.shape[1], -1)

            tp = (p_flat * t_flat).sum(dim=-1)
            fp = (p_flat * (1.0 - t_flat)).sum(dim=-1)
            fn = ((1.0 - p_flat) * t_flat).sum(dim=-1)
            dice_score = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
            dice_loss = 1.0 - dice_score.mean()
            return dice_loss, {'dice': dice_loss}

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
        dice_loss = 1.0 - dice_score.mean()
        
        # Add clDice component if enabled
        if self.cldice_alpha > 0:
            # For clDice, we need probabilities (not one-hot)
            # Use the same probs we computed earlier
            # clDice expects (B, C, H, W, D) format
            skel_pred = self.soft_skeletonize(probs[:, start_idx:])
            skel_true = self.soft_skeletonize(targets_one_hot[:, start_idx:])
            
            # Compute clDice: precision and sensitivity based on skeletons
            tprec = (torch.sum(skel_pred * targets_one_hot[:, start_idx:]) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
            tsens = (torch.sum(skel_true * probs[:, start_idx:]) + self.smooth) / (torch.sum(skel_true) + self.smooth)
            cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)
            
            # Combine dice and clDice (scaled components)
            scaled_dice = (1.0 - self.cldice_alpha) * dice_loss
            scaled_cldice = self.cldice_alpha * cl_dice
            total_loss = scaled_dice + scaled_cldice
            return total_loss, {'dice': scaled_dice, 'cldice': scaled_cldice}
        
        return dice_loss, {'dice': dice_loss}


class CombinedLoss(nn.Module):
    """Combined Dice (or Dice++/Tversky) + CrossEntropy Loss with optional clDice and weight maps.

    When binary=True, operates on per-channel binary masks using sigmoid + BCE
    instead of softmax + CE. Used for text-prompted segmentation where predictions
    and targets are both (B, N, H, W, D).
    """

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
        tversky_beta: float = 0.5,
        cldice_alpha: float = 0.0,
        use_weight_map: bool = False,
        weight_map_scale: float = 1.0,
        weight_map_bias: float = 1.0,
        binary: bool = False,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.use_weight_map = use_weight_map
        self.weight_map_scale = weight_map_scale
        self.weight_map_bias = weight_map_bias
        self.binary = binary
        if not binary:
            # Use reduction='none' when using weight maps to apply per-pixel weighting
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none' if use_weight_map else 'mean')
        self.dice_loss = DiceLoss(
            smooth=smooth,
            include_background=include_background,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=use_fixed_grad_softmax,
            exponential_correction=exponential_correction,
            plus_plus=dice_plus_plus,
            gamma=gamma,
            tversky_beta=tversky_beta,
            cldice_alpha=cldice_alpha,
            binary=binary,
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, weight_map: Optional[torch.Tensor] = None, **kwargs):
        """
        Args:
            predictions: (B, C, H, W, D) logits
            targets: (B, H, W, D) class indices OR (B, N, H, W, D) binary masks when binary=True
            weight_map: (B, 1, H, W, D) or (B, H, W, D) - per-pixel weights for CE loss (optional)
            **kwargs: Additional arguments (e.g., valid_bounds) passed to subclasses

        Returns:
            Tuple of (loss, components_dict) where components_dict contains scaled loss components
        """
        components = {}
        total_loss = 0.0

        # Only compute CE/BCE loss if it has non-zero weight
        if self.ce_weight > 0:
            if self.binary:
                ce = F.binary_cross_entropy_with_logits(
                    predictions, targets.float(), reduction='mean'
                )
            else:
                ce_per_pixel = self.ce_loss(predictions, targets.long())

                if self.use_weight_map and weight_map is not None:
                    # Apply per-pixel weighting
                    # weight_map shape: (B, 1, H, W, D) or (B, H, W, D)
                    # ce_per_pixel shape: (B, H, W, D)
                    if weight_map.dim() == 5 and weight_map.size(1) == 1:
                        weight_map = weight_map.squeeze(1)

                    # Mask for valid pixels (not ignored)
                    valid_mask = (targets != self.ignore_index).float()

                    # Apply weights and compute mean over valid pixels
                    weighted_ce = ce_per_pixel * (weight_map * self.weight_map_scale + self.weight_map_bias)
                    ce = (weighted_ce * valid_mask).sum() / (valid_mask.sum() + 1e-8)
                else:
                    # Without weight map, ce_per_pixel is already reduced to scalar
                    ce = ce_per_pixel if ce_per_pixel.dim() == 0 else ce_per_pixel.mean()

            scaled_ce = self.ce_weight * ce
            components['ce'] = scaled_ce
            total_loss = total_loss + scaled_ce

        # Only compute Dice loss if it has non-zero weight
        if self.dice_weight > 0:
            dice, dice_components = self.dice_loss(predictions, targets)
            # Add scaled dice components to our components dict
            for key, value in dice_components.items():
                components[key] = self.dice_weight * value
            total_loss = total_loss + self.dice_weight * dice

        return total_loss, components


class DiceTopographLoss(CombinedLoss):
    """Combined Dice (or Dice++/Tversky) + CrossEntropy + Topograph Loss.
    
    Extends CombinedLoss with Topograph loss, which penalizes topologically critical 
    voxels - those that cause changes in connected components (false splits, false 
    merges, etc.).
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        topograph_weight: float = 0.1,
        smooth: float = 1e-5,
        include_background: bool = False,
        ignore_index: int = -1,
        use_fixed_grad_softmax: bool = False,
        exponential_correction: Optional[int] = None,
        dice_plus_plus: bool = False,
        gamma: float = 2.0,
        tversky_beta: float = 0.5,
        cldice_alpha: float = 0.0,
        # Topograph-specific parameters
        topograph_num_processes: int = 4,
        topograph_thres_var: float = 0.0,
        topograph_aggregation: str = "mean",
        topograph_error_type: str = "false_positives",
        topograph_debug: bool = False,
        use_weight_map: bool = False,
    ):
        if not TOPOGRAPH_AVAILABLE:
            raise ImportError("TopographLoss requires the 'topograph' module. Install it to use combined_topograph loss.")
        super().__init__(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            smooth=smooth,
            include_background=include_background,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=use_fixed_grad_softmax,
            exponential_correction=exponential_correction,
            dice_plus_plus=dice_plus_plus,
            gamma=gamma,
            tversky_beta=tversky_beta,
            cldice_alpha=cldice_alpha,
            use_weight_map=use_weight_map,
        )
        self.topograph_weight = topograph_weight
        self.topograph_loss = TopographLoss(
            num_processes=topograph_num_processes,
            aggregation=topograph_aggregation,
            thres_var=topograph_thres_var,
            include_background=include_background,
            sphere=False,
            ignore_index=ignore_index,
            debug=topograph_debug,
            error_type=topograph_error_type,
        )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Args:
            predictions: (B, C, H, W, D) - logits
            targets: (B, H, W, D) - class indices
            **kwargs: Additional arguments (ignored for this loss)
        
        Returns:
            Tuple of (loss, components_dict) where components_dict contains scaled loss components
        """
        total_loss, components = super().forward(predictions, targets, **kwargs)

        # Compute Topograph loss if it has non-zero weight
        if self.topograph_weight > 0:
            # Ignore_index is handled inside TopographLoss
            
            topograph = self.topograph_loss(predictions, targets)
            scaled_topograph = self.topograph_weight * topograph
            components['topograph'] = scaled_topograph
            total_loss = total_loss + scaled_topograph
        
        return total_loss, components


class DiceBettiMatchingLoss(CombinedLoss):
    """Combined Dice (or Dice++/Tversky) + CrossEntropy + Betti Matching Loss.
    
    Extends CombinedLoss with Betti Matching loss, which uses persistent homology 
    to match topological features between prediction and target, penalizing both 
    matched pairs that differ and unmatched topological features.
    
    Note: Betti matching requires binary segmentation (2 classes). For multi-class,
    the foreground class (class 1) probabilities are used.
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        betti_weight: float = 0.1,
        smooth: float = 1e-5,
        include_background: bool = False,
        ignore_index: int = -1,
        use_fixed_grad_softmax: bool = False,
        exponential_correction: Optional[int] = None,
        dice_plus_plus: bool = False,
        gamma: float = 2.0,
        tversky_beta: float = 0.5,
        cldice_alpha: float = 0.0,
        # Betti matching-specific parameters
        betti_cpu_batch_size: int = 16,
        subsampling_size: int = None,
        subsampling_mode: str = "random_crop",
        use_weight_map: bool = False,
    ):
        if not BETTI_MATCHING_AVAILABLE:
            raise ImportError("DiceBettiMatchingLoss requires the 'betti_matching_loss' module. Install it to use combined_betti loss.")
        super().__init__(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            smooth=smooth,
            include_background=include_background,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=use_fixed_grad_softmax,
            exponential_correction=exponential_correction,
            dice_plus_plus=dice_plus_plus,
            gamma=gamma,
            tversky_beta=tversky_beta,
            cldice_alpha=cldice_alpha,
            use_weight_map=use_weight_map,
        )
        self.betti_weight = betti_weight
        self.betti_cpu_batch_size = betti_cpu_batch_size
        self.subsampling_size = subsampling_size
        self.subsampling_mode = subsampling_mode
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, valid_bounds: Optional[List] = None, weight_map: Optional[torch.Tensor] = None, **kwargs):
        """
        Args:
            predictions: (B, C, D, H, W) - logits
            targets: (B, D, H, W) - class indices (0 or 1 for binary segmentation)
            valid_bounds: Optional list of precomputed valid region bounds per sample.
                         Each element is ((w_min, w_max), (h_min, h_max), (d_min, d_max)) or None.
            weight_map: (B, 1, H, W, D) or (B, H, W, D) - per-pixel weights for CE loss (optional)
            **kwargs: Additional arguments passed to parent
        
        Returns:
            Tuple of (loss, components_dict) where components_dict contains scaled loss components
        """
        total_loss, components = super().forward(predictions, targets, valid_bounds=valid_bounds, weight_map=weight_map, **kwargs)
        
        # Compute Betti matching loss if it has non-zero weight
        if self.betti_weight > 0:
            # Convert predictions to probabilities for foreground class
            # Betti matching expects (B, 1, *spatial) tensors with values in [0, 1]
            probs = F.softmax(predictions, dim=1)
            
            # Use foreground probability (class 1) for Betti matching
            # Shape: (B, 1, D, H, W)
            pred_fg = probs[:, 1:2, ...]
            
            # Convert targets to same format: (B, 1, D, H, W) with values 0 or 1
            target_fg = (targets == 1).float().unsqueeze(1)
            
            # Create valid_mask for filtering coordinates in ignore regions
            # valid_mask is True where target is NOT the ignore_index
            valid_mask = (targets != self.ignore_index).float().unsqueeze(1)

            # print ratio of valid voxels in valid_mask
            valid_ratio = valid_mask.mean().item()
            print(f"[Betti Matching] Valid voxel ratio: {valid_ratio:.4f}", flush=True)
            
            betti_losses = compute_betti_matching_loss(
                prediction=pred_fg,
                target=target_fg,
                cpu_batch_size=self.betti_cpu_batch_size,
                sigmoid=False,  # Already applied softmax
                relative=False,
                valid_mask=valid_mask,
                valid_bounds=valid_bounds,
                subsampling_size=self.subsampling_size,
                subsampling_mode=self.subsampling_mode
            )
            
            betti_loss = torch.mean(torch.cat(betti_losses))
            scaled_betti = self.betti_weight * betti_loss
            components['betti_matching'] = scaled_betti
            
            total_loss = total_loss + scaled_betti
        
        return total_loss, components

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
    
    def forward(self, predictions: List[torch.Tensor], targets: torch.Tensor, weight_map: Optional[torch.Tensor] = None, **kwargs):
        """
        Args:
            predictions: List of predictions at different scales [(B, C, H, W, D), ...]
            targets: (B, H, W, D) class indices OR (B, N, H, W, D) binary masks
            weight_map: (B, 1, H, W, D) or (B, H, W, D) - per-pixel weights for CE loss (optional)
            **kwargs: Additional arguments (e.g., valid_bounds) passed to loss_fn

        Returns:
            Tuple of (loss, components_dict) where components_dict contains aggregated scaled loss components
        """
        total_loss = 0.0
        aggregated_components = {}
        binary_targets = targets.dim() == 5

        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            if weight == 0:
                continue

            # Downsample targets if needed
            if i < len(self.downsampling_scales) and self.downsampling_scales[i] > 1:
                scale = self.downsampling_scales[i]
                if binary_targets:
                    # Binary targets (B, N, H, W, D): interpolate directly, keep float
                    target_scaled = F.interpolate(
                        targets.float(),
                        scale_factor=1.0/scale,
                        mode='nearest'
                    )
                else:
                    # Class index targets (B, H, W, D): add/remove channel dim
                    target_scaled = F.interpolate(
                        targets.unsqueeze(1).float(),
                        scale_factor=1.0/scale,
                        mode='nearest'
                    ).squeeze(1).long()

                # Downsample weight_map if provided
                if weight_map is not None:
                    wm = weight_map if weight_map.dim() == 5 else weight_map.unsqueeze(1)
                    weight_map_scaled = F.interpolate(
                        wm.float(),
                        scale_factor=1.0/scale,
                        mode='trilinear',
                        align_corners=False
                    )
                else:
                    weight_map_scaled = None
            else:
                target_scaled = targets
                weight_map_scaled = weight_map

            # Resize prediction to match target if needed
            # For binary targets spatial dims start at index 2, for class indices at index 1
            target_spatial = target_scaled.shape[2:] if binary_targets else target_scaled.shape[1:]
            if pred.shape[2:] != target_spatial:
                pred_resized = F.interpolate(
                    pred,
                    size=target_spatial,
                    mode='trilinear',
                    align_corners=False
                )
            else:
                pred_resized = pred
            
            # Only pass valid_bounds to the main resolution (scale 0)
            # For downsampled scales, bounds would need adjustment
            if i == 0:
                loss, components = self.loss_fn(pred_resized, target_scaled, weight_map=weight_map_scaled, **kwargs)
            else:
                # Pass downsampled weight_map but not valid_bounds
                loss, components = self.loss_fn(pred_resized, target_scaled, weight_map=weight_map_scaled)
            total_loss += weight * loss
            
            # Aggregate components across scales (weighted sum)
            for key, value in components.items():
                if key not in aggregated_components:
                    aggregated_components[key] = 0.0
                aggregated_components[key] = aggregated_components[key] + weight * value
        
        return total_loss, aggregated_components


def get_loss_function(config) -> nn.Module:
    """Get loss function based on configuration"""
    # Text-prompted mode uses binary CombinedLoss (sigmoid + BCE instead of softmax + CE)
    if hasattr(config, 'text_prompted') and config.text_prompted.enabled:
        return CombinedLoss(
            dice_weight=config.training.dice_weight,
            ce_weight=config.training.ce_weight,
            binary=True,
        )

    ignore_index = getattr(config.training, 'ignore_index', -1)
    use_fixed_grad_softmax = getattr(config.training, 'use_fixed_grad_softmax', False)
    exponential_correction = getattr(config.training, 'exponential_correction', None)
    cldice_alpha = getattr(config.training, 'cldice_alpha', 0.0)
    use_weight_map = getattr(config.training, 'use_weight_map', False)
    
    # Topograph-specific config parameters
    topograph_weight = getattr(config.training, 'topograph_weight', 0.1)
    topograph_num_processes = getattr(config.training, 'topograph_num_processes', 4)
    topograph_thres_var = getattr(config.training, 'topograph_thres_var', 0.0)
    topograph_aggregation = getattr(config.training, 'topograph_aggregation', 'mean')
    topograph_debug = getattr(config.training, 'topograph_debug', False)
    topograph_error_type = getattr(config.training, 'topograph_error_type', 'all')

    
    if config.training.loss_function == "dice":
        return CombinedLoss(
            dice_weight=1.0,
            ce_weight=0.0,
            smooth=1e-5,
            include_background=False,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=use_fixed_grad_softmax,
            exponential_correction=exponential_correction,
            dice_plus_plus=False,
            cldice_alpha=cldice_alpha
        )
    
    elif config.training.loss_function == "ce":
        # throw error if use_fixed_grad_softmax is set
        if use_fixed_grad_softmax:
            raise NotImplementedError("use_fixed_grad_softmax is not implemented for CrossEntropy loss")
        return CrossEntropyLossWrapper(ignore_index=ignore_index)
    
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
            dice_plus_plus=False,
            cldice_alpha=cldice_alpha,
            use_weight_map=use_weight_map,
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
            gamma=gamma,
            cldice_alpha=cldice_alpha
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
            gamma=gamma,
            cldice_alpha=cldice_alpha,
            use_weight_map=use_weight_map,
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
            tversky_beta=tversky_beta,
            cldice_alpha=cldice_alpha
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
            tversky_beta=tversky_beta,
            cldice_alpha=cldice_alpha,
            use_weight_map=use_weight_map,
        )
    
    elif config.training.loss_function == "combined_topograph":
        return DiceTopographLoss(
            dice_weight=config.training.dice_weight,
            ce_weight=config.training.ce_weight,
            topograph_weight=topograph_weight,
            smooth=1e-5,
            include_background=False,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=False,
            exponential_correction=None,
            dice_plus_plus=False,
            tversky_beta=0.5, # 0.5 for standard Dice
            cldice_alpha=0.0,
            topograph_num_processes=topograph_num_processes,
            topograph_thres_var=topograph_thres_var,
            topograph_aggregation=topograph_aggregation,
            topograph_debug=topograph_debug,
            topograph_error_type=topograph_error_type,
            use_weight_map=use_weight_map,
        )
    
    elif config.training.loss_function == "combined_betti":
        betti_weight = getattr(config.training, 'betti_weight', 0.1)
        betti_cpu_batch_size = getattr(config.training, 'betti_cpu_batch_size', 16)
        betti_subsampling_size = getattr(config.training, 'betti_subsampling_size', None)
        betti_subsampling_mode = getattr(config.training, 'betti_subsampling_mode', "random_crop")

        return DiceBettiMatchingLoss(
            dice_weight=config.training.dice_weight,
            ce_weight=config.training.ce_weight,
            betti_weight=betti_weight,
            smooth=1e-5,
            include_background=False,
            ignore_index=ignore_index,
            use_fixed_grad_softmax=False,
            exponential_correction=None,
            dice_plus_plus=False,
            tversky_beta=0.5, # 0.5 for standard Dice
            cldice_alpha=0.0,
            betti_cpu_batch_size=betti_cpu_batch_size,
            subsampling_size=betti_subsampling_size,
            subsampling_mode=betti_subsampling_mode,
            use_weight_map=use_weight_map,
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
            exponential_correction = exponential_correction if exponential_correction is not None else 20
            error_weight[targets == 1] = error_weight[targets == 1] * (1 - torch.pow(probs[targets == 1], exponential_correction))
            error_weight[targets == 0] = error_weight[targets == 0] * (1 - torch.pow(1 - probs[targets == 0], exponential_correction))
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

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1., exclude_background=False):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1., exclude_background=False):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        dice = soft_dice(y_true, y_pred)
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice

class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=10):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):

        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):

        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img):
        
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):

        img1 = self.soft_open(img)
        skel = F.relu(img-img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):

        return self.soft_skel(img)