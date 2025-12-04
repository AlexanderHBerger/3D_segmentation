"""
Visualization utilities for debugging training pipeline
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import nibabel as nib


class TrainingVisualizer:
    """Visualizes training data and network outputs for debugging"""
    
    def __init__(self, output_dir: str, max_samples: int = 5):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
            max_samples: Maximum number of samples to visualize per epoch
        """
        self.output_dir = Path(str(output_dir))  # Ensure it's a Path object
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.sample_count = 0
    
    def reset_count(self):
        """Reset sample counter for new epoch"""
        self.sample_count = 0
    
    def visualize_batch(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        outputs: torch.Tensor,
        epoch: int,
        batch_idx: int,
        loss_value: float = None,
        prefix: str = "train"
    ):
        """
        Visualize a batch of images, targets, and predictions
        
        Args:
            images: Input images [B, C, H, W, D]
            targets: Ground truth labels [B, H, W, D]
            outputs: Network predictions [B, C, H, W, D] (logits or probabilities)
            epoch: Current epoch number
            batch_idx: Batch index
            loss_value: Loss value for this batch
            prefix: Prefix for filename (train/val)
        """
        if self.sample_count >= self.max_samples:
            return
        
        # Move to CPU and convert to numpy
        images_np = images.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # Handle outputs (could be logits or list for deep supervision)
        if isinstance(outputs, list):
            outputs_np = outputs[0].detach().cpu().numpy()  # Use main output
        else:
            outputs_np = outputs.detach().cpu().numpy()
        
        # Get predictions (apply softmax if needed)
        if outputs_np.shape[1] > 1:  # Multi-class
            preds_np = np.argmax(outputs_np, axis=1)
            probs_np = torch.softmax(torch.from_numpy(outputs_np), dim=1).numpy()
        else:  # Binary
            probs_np = torch.sigmoid(torch.from_numpy(outputs_np)).numpy()
            preds_np = (probs_np > 0.5).astype(np.int32).squeeze(1)
        
        # Visualize first sample in batch
        batch_size = images_np.shape[0]
        for sample_idx in range(min(batch_size, self.max_samples - self.sample_count)):
            self._visualize_sample(
                image=images_np[sample_idx, 0],  # First channel
                target=targets_np[sample_idx],
                prediction=preds_np[sample_idx],
                probability=probs_np[sample_idx, 1] if probs_np.shape[1] > 1 else probs_np[sample_idx, 0],
                epoch=epoch,
                sample_id=self.sample_count,
                loss_value=loss_value,
                prefix=prefix
            )
            self.sample_count += 1
    
    def _visualize_sample(
        self,
        image: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
        probability: np.ndarray,
        epoch: int,
        sample_id: int,
        loss_value: Optional[float] = None,
        prefix: str = "train"
    ):
        """
        Visualize a single 3D sample with multiple slices
        
        Args:
            image: 3D image array [H, W, D]
            target: 3D target array [H, W, D] (may contain -1 for ignore regions)
            prediction: 3D prediction array [H, W, D]
            probability: 3D probability map [H, W, D]
            epoch: Current epoch
            sample_id: Sample identifier
            loss_value: Loss value
            prefix: Filename prefix
        """
        # Create valid mask (exclude -1 regions)
        valid_mask = target != -1
        
        # Find slices with foreground in each dimension (excluding -1 regions)
        h, w, d = image.shape
        slices = {}
        
        # Create target with -1 masked to 0 for visualization
        target_vis = np.where(valid_mask, target, 0)
        
        # Axial (along depth axis)
        axial_fg = np.sum(target_vis, axis=(0, 1))  # Sum over H, W
        axial_slices = np.where(axial_fg > 0)[0]
        if len(axial_slices) > 0:
            slices['axial'] = axial_slices[len(axial_slices) // 2]  # Middle FG slice
        else:
            slices['axial'] = d // 2  # Fallback to middle
        
        # Coronal (along width axis)
        coronal_fg = np.sum(target_vis, axis=(0, 2))  # Sum over H, D
        coronal_slices = np.where(coronal_fg > 0)[0]
        if len(coronal_slices) > 0:
            slices['coronal'] = coronal_slices[len(coronal_slices) // 2]
        else:
            slices['coronal'] = w // 2
        
        # Sagittal (along height axis)
        sagittal_fg = np.sum(target_vis, axis=(1, 2))  # Sum over W, D
        sagittal_slices = np.where(sagittal_fg > 0)[0]
        if len(sagittal_slices) > 0:
            slices['sagittal'] = sagittal_slices[len(sagittal_slices) // 2]
        else:
            slices['sagittal'] = h // 2
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Set title
        title = f"Epoch {epoch} - Sample {sample_id}"
        if loss_value is not None:
            title += f" - Loss: {loss_value:.4f}"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        views = ['axial', 'coronal', 'sagittal']
        
        for row, view in enumerate(views):
            # Extract slices
            if view == 'axial':
                img_slice = image[:, :, slices[view]]
                tgt_slice = target[:, :, slices[view]]
                pred_slice = prediction[:, :, slices[view]]
                prob_slice = probability[:, :, slices[view]]
                valid_slice = valid_mask[:, :, slices[view]]
            elif view == 'coronal':
                img_slice = image[:, slices[view], :]
                tgt_slice = target[:, slices[view], :]
                pred_slice = prediction[:, slices[view], :]
                prob_slice = probability[:, slices[view], :]
                valid_slice = valid_mask[:, slices[view], :]
            else:  # sagittal
                img_slice = image[slices[view], :, :]
                tgt_slice = target[slices[view], :, :]
                pred_slice = prediction[slices[view], :, :]
                prob_slice = probability[slices[view], :, :]
                valid_slice = valid_mask[slices[view], :, :]
            
            # Create visualization target (0 where invalid, actual value where valid)
            tgt_slice_vis = np.where(valid_slice, tgt_slice, 0)
            
            # Normalize image slice for display using FIXED value range
            # This ensures consistent visualization across all images
            vmin, vmax = -5.0, 16.0
            img_norm = np.clip((img_slice - vmin) / (vmax - vmin), 0, 1)
            
            # Compute statistics for this slice
            img_stats = self._get_statistics(img_slice)
            
            # Count ignore pixels
            n_ignore = np.sum(~valid_slice)
            
            # Plot image with FIXED value range for consistent comparison
            ax = axes[row, 0]
            im = ax.imshow(img_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(f'{view.capitalize()} - Input Image (slice {slices[view]})\n{img_stats}\nFixed range: [{vmin}, {vmax}]')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Plot image with ground truth overlay (excluding -1 regions)
            ax = axes[row, 1]
            ax.imshow(img_norm.T, cmap='gray', origin='lower', alpha=1.0)
            # Overlay ground truth in red (only valid regions)
            gt_mask = np.ma.masked_where((tgt_slice_vis.T == 0) | (~valid_slice.T), tgt_slice_vis.T)
            ax.imshow(gt_mask, cmap='Reds', origin='lower', alpha=0.5, vmin=0, vmax=1)
            # Overlay ignore regions in blue (-1 values)
            ignore_mask = np.ma.masked_where(valid_slice.T, np.ones_like(tgt_slice.T))
            ax.imshow(ignore_mask, cmap='Blues', origin='lower', alpha=0.5, vmin=0, vmax=1)
            fg_count = np.sum((tgt_slice_vis > 0) & valid_slice)
            ax.set_title(f'{view.capitalize()} - GT Overlay\nRed=FG ({fg_count}), Blue=Ignore ({n_ignore})')
            ax.axis('off')
            
            # Plot image with prediction overlay
            ax = axes[row, 2]
            ax.imshow(img_norm.T, cmap='gray', origin='lower', alpha=1.0)
            # Overlay prediction in green
            pred_mask = np.ma.masked_where(pred_slice.T == 0, pred_slice.T)
            im = ax.imshow(pred_mask, cmap='Greens', origin='lower', alpha=0.5, vmin=0, vmax=1)
            ax.set_title(f'{view.capitalize()} - Pred Overlay\nFG pixels: {np.sum(pred_slice > 0)}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Plot image with both overlays (GT=red, Pred=green, overlap=yellow)
            # Only compute metrics on valid pixels
            ax = axes[row, 3]
            ax.imshow(img_norm.T, cmap='gray', origin='lower', alpha=1.0)
            # Create RGB overlay: GT=red, Pred=green (only for valid pixels)
            overlay = np.zeros((*img_norm.T.shape, 3))
            overlay[:, :, 0] = np.where(valid_slice.T, tgt_slice_vis.T, 0)  # Red channel = GT (valid only)
            overlay[:, :, 1] = pred_slice.T  # Green channel = Pred
            # Where both overlap, it will appear yellow
            ax.imshow(overlay, origin='lower', alpha=0.5)
            
            # Compute metrics only on valid pixels
            tp = np.sum((tgt_slice_vis > 0) & (pred_slice > 0) & valid_slice)
            fp = np.sum((tgt_slice_vis == 0) & (pred_slice > 0) & valid_slice)
            fn = np.sum((tgt_slice_vis > 0) & (pred_slice == 0) & valid_slice)
            ax.set_title(f'{view.capitalize()} - Combined\nTP:{tp} FP:{fp} FN:{fn}')
            ax.axis('off')
        
        # Save figure
        output_path = self.output_dir / f"{prefix}_epoch{epoch:03d}_sample{sample_id:03d}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved visualization to {output_path}")
    
    def _get_statistics(self, array: np.ndarray) -> str:
        """Get statistics string for array"""
        return f"min={array.min():.3f}, max={array.max():.3f}, mean={array.mean():.3f}, std={array.std():.3f}"
    
    def visualize_histogram(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        prefix: str = "train"
    ):
        """
        Visualize intensity histograms to check for NaN/Inf values
        
        Args:
            images: Input images [B, C, H, W, D]
            targets: Ground truth labels [B, H, W, D] (may contain -1 for ignore)
            epoch: Current epoch
            prefix: Filename prefix
        """
        images_np = images.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Image histogram
        ax = axes[0]
        img_flat = images_np.flatten()
        
        # Check for invalid values
        n_nan = np.sum(np.isnan(img_flat))
        n_inf = np.sum(np.isinf(img_flat))
        
        if n_nan > 0 or n_inf > 0:
            print(f"WARNING: Found {n_nan} NaN and {n_inf} Inf values in images!")
        
        # Remove invalid values for histogram
        img_flat = img_flat[np.isfinite(img_flat)]
        
        ax.hist(img_flat, bins=100, alpha=0.7, edgecolor='black')
        ax.set_title(f'Input Image Histogram\nNaN: {n_nan}, Inf: {n_inf}\n'
                    f'min={img_flat.min():.3f}, max={img_flat.max():.3f}')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Target histogram (handle -1 separately)
        ax = axes[1]
        tgt_flat = targets_np.flatten()
        unique, counts = np.unique(tgt_flat, return_counts=True)
        
        # Create bar colors: blue for -1 (ignore), other colors for classes
        colors = ['blue' if u == -1 else 'gray' for u in unique]
        
        ax.bar(unique, counts, alpha=0.7, edgecolor='black', color=colors)
        
        # Create legend for -1
        label_dict = {int(u): int(c) for u, c in zip(unique, counts)}
        label_str = ', '.join([f'{k}: {v}' for k, v in label_dict.items()])
        ax.set_title(f'Target Label Distribution\n{label_str}\n(-1=ignore, blue)')
        ax.set_xlabel('Class')
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Save figure
        output_path = self.output_dir / f"{prefix}_histogram_epoch{epoch:03d}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved histogram to {output_path}")
    
    def save_volume_as_nifti(
        self,
        volume: np.ndarray,
        filename: str,
        affine: Optional[np.ndarray] = None
    ):
        """
        Save 3D volume as NIfTI for detailed inspection
        
        Args:
            volume: 3D numpy array
            filename: Output filename
            affine: Affine transformation matrix
        """
        if affine is None:
            affine = np.eye(4)
        
        nifti_img = nib.Nifti1Image(volume, affine)
        output_path = self.output_dir / filename
        nib.save(nifti_img, output_path)
        print(f"Saved NIfTI volume to {output_path}")
    
    def check_batch_validity(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        batch_idx: int = 0
    ) -> Dict[str, bool]:
        """
        Check batch for NaN/Inf values and other issues
        
        Args:
            images: Input images
            targets: Target labels
            outputs: Network outputs (optional)
            batch_idx: Batch index for reporting
        
        Returns:
            Dictionary with validity checks
        """
        checks = {
            'images_valid': True,
            'targets_valid': True,
            'outputs_valid': True,
            'images_finite': True,
            'outputs_finite': True
        }
        
        # Check images
        if torch.isnan(images).any():
            print(f"❌ Batch {batch_idx}: NaN values in images!")
            checks['images_valid'] = False
        
        if torch.isinf(images).any():
            print(f"❌ Batch {batch_idx}: Inf values in images!")
            checks['images_finite'] = False
        
        # Check targets
        if torch.isnan(targets).any():
            print(f"❌ Batch {batch_idx}: NaN values in targets!")
            checks['targets_valid'] = False
        
        # Check outputs
        if outputs is not None:
            if isinstance(outputs, list):
                output_to_check = outputs[0]
            else:
                output_to_check = outputs
            
            if torch.isnan(output_to_check).any():
                print(f"❌ Batch {batch_idx}: NaN values in outputs!")
                checks['outputs_valid'] = False
            
            if torch.isinf(output_to_check).any():
                print(f"❌ Batch {batch_idx}: Inf values in outputs!")
                checks['outputs_finite'] = False
        
        # Print statistics if issues found
        if not all(checks.values()):
            print(f"\nBatch {batch_idx} Statistics:")
            print(f"  Images: min={images.min():.4f}, max={images.max():.4f}, "
                  f"mean={images.mean():.4f}, std={images.std():.4f}")
            print(f"  Targets: unique values={torch.unique(targets).tolist()}")
            if outputs is not None:
                output_to_check = outputs[0] if isinstance(outputs, list) else outputs
                print(f"  Outputs: min={output_to_check.min():.4f}, max={output_to_check.max():.4f}, "
                      f"mean={output_to_check.mean():.4f}, std={output_to_check.std():.4f}")
        
        return checks


def create_summary_plot(metrics_history: Dict[str, List[float]], output_dir: str):
    """
    Create summary plots of training metrics
    
    Args:
        metrics_history: Dictionary of metric lists
        output_dir: Directory to save plots
    """
    output_dir = Path(str(output_dir))  # Ensure it's a Path object
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss plot
    ax = axes[0, 0]
    if 'train_loss' in metrics_history:
        ax.plot(metrics_history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in metrics_history:
        ax.plot(metrics_history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Dice plot
    ax = axes[0, 1]
    if 'val_dice' in metrics_history:
        ax.plot(metrics_history['val_dice'], label='Val Dice', linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Score')
    ax.set_title('Validation Dice Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax = axes[1, 0]
    if 'learning_rate' in metrics_history:
        ax.plot(metrics_history['learning_rate'], linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # NaN count plot (if tracked)
    ax = axes[1, 1]
    if 'nan_count' in metrics_history:
        ax.plot(metrics_history['nan_count'], linewidth=2, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('NaN Count')
        ax.set_title('NaN Values in Batches')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No NaN tracking data', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    output_path = output_dir / 'training_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved training summary to {output_path}")
