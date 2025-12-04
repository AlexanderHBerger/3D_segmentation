"""
Utility functions for training
"""
import os
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 50, mode: str = 'max', min_delta: float = 0.001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
        if mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            self.monitor_op = np.less
            self.min_delta *= -1
    
    def __call__(self, score: float) -> bool:
        """
        Call early stopping check
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool = False,
    checkpoint_dir: Path = Path('./checkpoints'),
    filename: str = 'checkpoint.pth'
):
    """Save model checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = checkpoint_dir / 'best_model.pth'
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    checkpoint: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, Any]:
    """Load model checkpoint"""
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler state
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_bn_momentum(model: torch.nn.Module, momentum: float):
    """Set batch normalization momentum"""
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.momentum = momentum


def freeze_bn(model: torch.nn.Module):
    """Freeze batch normalization layers"""
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()


def unfreeze_bn(model: torch.nn.Module):
    """Unfreeze batch normalization layers"""
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.train()


def get_model_summary(model: torch.nn.Module, input_size: tuple) -> str:
    """Get model summary"""
    try:
        from torchsummary import summary
        summary_str = summary(model, input_size, verbose=0)
        return str(summary_str)
    except ImportError:
        param_info = count_parameters(model)
        return f"Model parameters: {param_info}"


def setup_logging(log_dir: Path, log_level: str = 'INFO'):
    """Setup logging configuration"""
    import logging
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_dir / 'training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def compute_class_weights(data_loader, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced datasets"""
    class_counts = torch.zeros(num_classes)
    
    print("Computing class weights...")
    for batch_idx, (_, targets) in enumerate(data_loader):
        for class_idx in range(num_classes):
            class_counts[class_idx] += (targets == class_idx).sum().item()
        
        if batch_idx % 50 == 0:
            print(f"Processed {batch_idx}/{len(data_loader)} batches")
    
    # Compute weights (inverse frequency)
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    return class_weights


def calculate_dataset_statistics(data_loader) -> Dict[str, float]:
    """Calculate dataset statistics (mean, std, etc.)"""
    
    print("Calculating dataset statistics...")
    
    # Initialize accumulators
    pixel_sum = 0
    pixel_sum_sq = 0
    num_pixels = 0
    min_val = float('inf')
    max_val = float('-inf')
    
    for batch_idx, (images, _) in enumerate(data_loader):
        # Flatten images
        images_flat = images.view(-1)
        
        # Update statistics
        pixel_sum += images_flat.sum().item()
        pixel_sum_sq += (images_flat ** 2).sum().item()
        num_pixels += images_flat.numel()
        
        batch_min = images_flat.min().item()
        batch_max = images_flat.max().item()
        
        min_val = min(min_val, batch_min)
        max_val = max(max_val, batch_max)
        
        if batch_idx % 50 == 0:
            print(f"Processed {batch_idx}/{len(data_loader)} batches")
    
    # Calculate final statistics
    mean = pixel_sum / num_pixels
    variance = (pixel_sum_sq / num_pixels) - (mean ** 2)
    std = np.sqrt(variance)
    
    stats = {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'num_pixels': num_pixels
    }
    
    print("Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return stats


def create_experiment_dir(base_dir: Path, experiment_name: str) -> Path:
    """Create experiment directory with timestamp"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def save_config(config, save_path: Path):
    """Save configuration to file"""
    import json
    
    # Convert config to dictionary
    if hasattr(config, '__dict__'):
        config_dict = {}
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
    else:
        config_dict = config
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


class ProgressTracker:
    """Track training progress"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.learning_rates = []
        self.epoch_times = []
    
    def update(self, train_loss, val_loss, train_metrics, val_metrics, lr, epoch_time):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def save(self, save_path: Path):
        """Save progress to file"""
        import pickle
        
        data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)


def create_validation_samples_table(random_batches: list, worst_batch: dict, wandb):
    """Create wandb table with random batches and worst batch samples"""

    # Define class labels for wandb masks
    class_labels = {
        0: "background",
        1: "metastasis"
    }
    
    # Create wandb table with Sample_ID, Loss, and Image (with masks)
    table = wandb.Table(columns=["Sample_ID", "Loss", "Image"])
    
    # Log all samples from worst batch
    batch_size = worst_batch['images'].shape[0]
    for sample_idx in range(batch_size):
        sample_id = f"worst_batch_sample_{sample_idx}"

        img_slice, target_slice, pred_slice, z_idx = extract_slice_with_lesion(
            worst_batch['images'][sample_idx],
            worst_batch['targets'][sample_idx],
            worst_batch['predictions'][sample_idx]
        )
        
        # Create wandb Image with masks
        mask_img = wandb.Image(
            img_slice,
            masks={
                "ground_truth": {"mask_data": target_slice, "class_labels": class_labels},
                "prediction": {"mask_data": pred_slice, "class_labels": class_labels}
            }
        )
        
        table.add_data(
            sample_id,
            f"{worst_batch['loss']:.4f}",
            mask_img
        )
    
    # Log one random sample from each of 5 random batches
    for batch_idx, batch in enumerate(random_batches):
        # Pick first sample from batch
        sample_idx = 0
        sample_id = f"random_batch_{batch_idx}_sample_{sample_idx}"
        img_slice, target_slice, pred_slice, z_idx = extract_slice_with_lesion(
            batch['images'][sample_idx],
            batch['targets'][sample_idx],
            batch['predictions'][sample_idx]
        )
        
        # Create wandb Image with masks
        mask_img = wandb.Image(
            img_slice,
            masks={
                "ground_truth": {"mask_data": target_slice, "class_labels": class_labels},
                "prediction": {"mask_data": pred_slice, "class_labels": class_labels}
            }
        )
        
        table.add_data(
            sample_id,
            f"{batch['loss']:.4f}",
            mask_img
        )
    
    return table


def extract_slice_with_lesion(image, target, pred_class):
    """Extract slice with most lesion pixels and return numpy arrays"""
    # image: (C, H, W, D) or (H, W, D)
    # target: (H, W, D)
    # prediction: (C, H, W, D)

    pred_class = pred_class[1]  # Get class predictions (H, W, D)
    
    # Remove channel dim from image if present
    if image.dim() == 4:
        image = image.squeeze(0)  # (H, W, D)
    
    # Convert to numpy
    target_np = target.cpu().numpy() if torch.is_tensor(target) else target
    
    # Find slice with lesion (max lesion area in target)
    # Only consider slices with actual lesions (value 1), ignore -1 and 0
    target_lesion_mask = (target_np == 1).astype(np.float32)
    target_sum_per_slice = target_lesion_mask.sum(axis=(0, 1))  # Sum over H, W for each Z
    
    if target_sum_per_slice.max() > 0:
        # Use slice with most lesion pixels
        z_idx = int(np.argmax(target_sum_per_slice))
    else:
        # No lesion, use middle slice
        z_idx = target_np.shape[-1] // 2
    
    # Extract slices as numpy arrays
    img_slice = image[:, :, z_idx].cpu().numpy() if torch.is_tensor(image) else image[:, :, z_idx]
    target_slice = target_np[:, :, z_idx]
    pred_prob_slice = pred_class[:, :, z_idx].cpu().numpy() if torch.is_tensor(pred_class) else pred_class[:, :, z_idx]
    
    # Threshold probabilities to get binary mask for visualization
    pred_slice = (pred_prob_slice > 0.5).astype(np.uint8)
    
    # Replace -1 (ignore index) with 0 (background) for visualization
    target_slice = np.where(target_slice == -1, 0, target_slice).astype(np.uint8)
    
    # Normalize image to 0-255 range for wandb (assuming it's normalized/standardized)
    img_slice = img_slice.astype(np.float32)
    # Clip to reasonable range (e.g., -3 to 3 std for normalized images)
    img_slice = np.clip(img_slice, -3, 3)
    # Scale to 0-255
    img_slice = ((img_slice + 3) / 6.0 * 255).astype(np.uint8)
    
    return img_slice, target_slice, pred_slice, z_idx