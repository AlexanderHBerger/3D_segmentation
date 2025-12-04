"""
Training script for MedNeXt segmentation model
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
from torch.amp import GradScaler, autocast
import wandb
from typing import Dict, Optional, Any
import time
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local imports
from config import get_config
from data_loading_native import DataManager, create_data_loaders
from transforms_torchio import get_training_transforms, get_validation_transforms
from losses import get_loss_function, get_deep_supervision_loss
from metrics import MetricsCalculator
from utils import (AverageMeter, save_checkpoint, load_checkpoint, EarlyStopping, extract_slice_with_lesion)
from visualization import TrainingVisualizer
from architectures import calculate_n_stages

class Trainer:
    """Main training class"""

    def __init__(self, config, fold: int, enable_visualization: bool = False, resume_id: Optional[str] = None):
        self.fold = fold
        self.enable_visualization = enable_visualization
        self.run_id = resume_id
        
        # Determine output path if resuming (directory must exist for checkpoint loading)
        if self.run_id is not None:
            self.output_dir = Path(config.output_dir) / f"fold_{fold}_{self.run_id}"
            if not self.output_dir.exists():
                raise FileNotFoundError(f"Resume requested but directory does not exist: {self.output_dir}")
        
        # Load checkpoint from disk if resuming
        checkpoint = self._load_checkpoint_from_disk() if resume_id is not None else None

        # Load config (either from checkpoint or use provided config)
        self.config = self._load_config(config, checkpoint)
        
        # Set up device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seeds
        self.set_random_seeds(self.config.seed + fold)  # Different seed per fold
        
        # Initialize mixed precision
        self.scaler = GradScaler('cuda' if self.device.type == 'cuda' else 'cpu') if self.config.mixed_precision else None
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion, self.val_loss = self._create_loss_function()
        
        # Initialize metrics
        self.train_metrics = MetricsCalculator(
            num_classes=self.config.data.num_classes,
            include_background=False,
            ignore_index=self.config.training.ignore_index,
            compute_calibration=False  # Disable for training to save time
        )
        self.val_metrics = MetricsCalculator(
            num_classes=self.config.data.num_classes,
            include_background=False,
            ignore_index=self.config.training.ignore_index,
            compute_calibration=True,  # Enable for validation
            num_bins=15
        )
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Initialize tracking variables
        self.epoch = 0
        self.best_metric = 0.0
        self.early_stopping = EarlyStopping(
            patience=self.config.training.patience,
            mode=self.config.training.monitor_mode
        )

        # Track metrics history for debugging
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'learning_rate': [],
            'nan_count': []
        }
        
        # Step 3: Restore state from checkpoint if resuming
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)
        
        # Initialize wandb (with resume if we have a run_id) and create output directory
        if wandb.run is None and not (hasattr(self.config, 'run_id') and self.config.run_id == "offline_run"):
            self._init_wandb() 
            # Set output directory after wandb init to include run_id
        elif wandb.run is not None and self.run_id is None:
            # wandb was already initialized (e.g., by sweep agent)
            # Get run_id from existing wandb run and set up output directory
            self.run_id = wandb.run.id
            print(f"Using existing wandb run with ID: {self.run_id}")
        elif wandb.run is None and hasattr(self.config, 'run_id') and self.config.run_id == "offline_run":
            self.run_id = "offline_run"

        self.output_dir = Path(config.output_dir) / f"fold_{fold}_{self.run_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize visualization
        if self.enable_visualization:
            viz_dir = self.output_dir / "visualizations"
            self.visualizer = TrainingVisualizer(
                output_dir=str(viz_dir),
                max_samples=10  # Visualize up to 10 samples per epoch
            )
            print(f"Visualization enabled - saving to {viz_dir}")
        else:
            self.visualizer = None
        
        self.validation_table = wandb.Table(columns=["Epoch", "Sample_ID", "Loss", "Dice", "Image"], log_mode="INCREMENTAL")
    
    def _load_checkpoint_from_disk(self) -> Optional[Dict[str, Any]]:
        """
        Step 1: Load checkpoint from disk into memory.
        
        Returns:
            Checkpoint dictionary or None if not found
        """
        checkpoint_path = self._find_latest_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found in {self.output_dir}")
        
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint
    
    def _load_config(self, config, checkpoint: Optional[Dict[str, Any]]):
        """
        Determine which config to use (from checkpoint or provided config).
        
        Args:
            config: Config passed from main (could have _use_new_config flag set)
            checkpoint: Loaded checkpoint dictionary or None
        
        Returns:
            Config object to use for training
        """
        # If not resuming, use the provided config
        if checkpoint is None:
            print("\nStarting new training - using config from config.py")
            return config
        
        # If resuming and flag is set, load config from checkpoint
        if not hasattr(config, '_use_new_config') or not config._use_new_config:
            print(f"\n{'='*80}")
            print(f"Loading configuration from checkpoint")
            print(f"{'='*80}\n")
            
            if 'config' not in checkpoint:
                raise ValueError(f"No config found in checkpoint")
            
            loaded_config = checkpoint['config']
            
            print("Configuration loaded successfully from checkpoint!")
            print(f"  Model: {loaded_config.model.architecture} ({loaded_config.model.model_size})")
            print(f"  Batch size: {loaded_config.training.batch_size}")
            print(f"  Learning rate: {loaded_config.training.initial_lr}")
            print(f"  Max epochs: {loaded_config.training.max_epochs}")
            print(f"  W&B Project: {loaded_config.wandb.project}")
            print(f"{'='*80}\n")
            
            return loaded_config
        else:
            # Resuming but using new config from config.py
            print("\nResuming training - using NEW config from config.py (checkpoint config ignored)")
            return config
        
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint in the fold directory"""
        # Check for checkpoint.pth first (most recent)
        checkpoint_file = self.output_dir / "checkpoint.pth"
        if checkpoint_file.exists():
            return checkpoint_file

        return None
    
    def _load_checkpoint(self, checkpoint):
        """Load checkpoint and resume training"""
        
        checkpoint = load_checkpoint(
            checkpoint,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler
        )
        
        # Restore training state
        self.epoch = checkpoint.get('epoch', 0) + 1  # Continue from next epoch
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        # Restore metrics history if available
        if 'metrics_history' in checkpoint:
            self.metrics_history = checkpoint['metrics_history']
        
        print(f"Checkpoint loaded successfully!")
        print(f"  Resuming from epoch: {self.epoch}")
        print(f"  Best metric so far: {self.best_metric:.4f}")
        print(f"  Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Check if fold matches
        checkpoint_fold = checkpoint.get('fold', self.fold)
        if checkpoint_fold != self.fold:
            raise ValueError(f"Checkpoint is from fold {checkpoint_fold}, but training fold {self.fold}")
    
    def set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_model(self) -> nn.Module:
        """Create and initialize model"""
        from model import create_model, get_model_info, initialize_weights
        
        model = create_model(self.config)
        
        # Initialize weights (only for non-transformer architectures)
        architecture = self.config.model.architecture.lower()
        if architecture not in ['primus']:
            initialize_weights(model, init_type='kaiming_normal')
        
        # Move to device
        model = model.to(self.device)
        
        # Print model info
        info = get_model_info(model)
        print(f"Model created - {self.config.model.architecture} {self.config.model.model_size}:")
        print(f"  Total parameters: {info['total_params']:,}")
        print(f"  Trainable parameters: {info['trainable_params']:,}")
        print(f"  Model size: {info['model_size_mb']:.2f} MB")
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.training.initial_lr,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay,
            nesterov=True
        )
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.training.lr_scheduler == 'poly':
            # Create scheduler with full max_epochs
            scheduler = PolynomialLR(
                self.optimizer,
                total_iters=self.config.training.max_epochs,
                power=self.config.training.poly_lr_pow
            )
        else:
            # Default to no scheduler
            scheduler = None
        
        return scheduler
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function"""
        base_loss = get_loss_function(self.config)
        val_loss = get_loss_function(self.config)
        
        if self.config.model.deep_supervision:
            # Calculate number of stages from patch size
            n_stages = calculate_n_stages(self.config.data.patch_size)
            loss_fn = get_deep_supervision_loss(self.config, base_loss, n_stages=n_stages)
        else:
            loss_fn = base_loss
        
        return loss_fn, val_loss
    
    def _create_data_loaders(self):
        """Create data loaders"""
        # Initialize data manager
        data_manager = DataManager(
            data_path=self.config.data.data_path,
            brats_only=self.config.data.brats_only,
            max_samples=self.config.data.max_samples
        )
        
        # Get transforms
        train_transforms = get_training_transforms(self.config)
        val_transforms = get_validation_transforms(self.config)
        
        # Create data loaders with use_preprocessed flag
        train_loader, val_loader = create_data_loaders(
            data_manager=data_manager,
            fold=self.fold,
            config=self.config,
            transforms_train=train_transforms,
            transforms_val=val_transforms,
            use_preprocessed=self.config.data.use_preprocessed  # Use preprocessed numpy arrays
        )
        
        return train_loader, val_loader
    
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        # Determine if we're resuming a run
        resume_mode = "must" if self.run_id is not None else None
        
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=f"fold_{self.fold}",
            tags=self.config.wandb.tags + [f"fold_{self.fold}"],
            notes=self.config.wandb.notes,
            id=self.run_id,  # Use existing run ID if resuming
            resume=resume_mode,  # Resume the run if ID is provided
            config={
                'fold': self.fold,
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__,
                'data': self.config.data.__dict__,
                'augmentation': self.config.augmentation.__dict__
            }
        )
        
        # Store the run ID for checkpointing (important for new runs)
        if self.run_id is None:
            self.run_id = wandb.run.id
            print(f"Created new wandb run with ID: {self.run_id}")
        else:
            print(f"Resumed wandb run with ID: {self.run_id}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch (nnUNet style: fixed 250 iterations)"""
        self.model.train()
        #self.train_metrics.reset()
        
        # Tracking variables
        loss_meter = AverageMeter()
        
        # Reset visualization counter
        if self.visualizer is not None:
            self.visualizer.reset_count()
        
        # Create progress bar for this epoch
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}/{self.config.training.max_epochs}",
            ncols=100,
            leave=True
        )
        
        # Fixed number of iterations per epoch (nnUNet approach)
        for batch_idx, batch in enumerate(pbar):
            # Track iteration time
            iter_start_time = time.time()
            
            # Get next batch
            images, targets = batch['image'], batch['label']

            # print mean and standard deviation of images for debugging
            #print(f"Image stats - mean: {images.mean().item():.4f}, std: {images.std().item():.4f}")
                
            if targets.dim() == 5 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            else:
                raise ValueError(f"Targets should have shape (B, 1, H, W, D), got {targets.shape}")

            # Move to device (normalization now happens in dataset)
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=self.scaler is not None):
                outputs = self.model(images)
                
                if self.config.model.deep_supervision and isinstance(outputs, list):
                    loss = self.criterion(outputs, targets)
                    # Use main output for metrics (highest resolution)
                    main_output = outputs[0]
                else:
                    loss = self.criterion(outputs, targets)
                    main_output = outputs
            
            # Backward pass with gradient clipping
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.optimizer.step()
            
            # Get current loss value
            current_loss = loss.item()
            
            # Update metrics
            loss_meter.update(current_loss, images.size(0))
            #self.train_metrics.update(main_output, targets)

            # Visualize every 50 batches if enabled (not at the very beginning)
            if self.visualizer is not None and batch_idx > 0 and batch_idx % 50 == 0:
                self.visualizer.visualize_batch(
                    images, targets, main_output,
                    self.epoch, batch_idx, current_loss,
                    prefix="train"
                )
            
            # CRITICAL: Clear references to prevent memory leak
            del images, targets, outputs, loss
            if hasattr(self, 'main_output'):
                del main_output
            
            # Calculate iteration time
            iter_time = time.time() - iter_start_time
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar with current loss and iteration time
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{loss_meter.avg:.4f}',
                'iter_time': f'{iter_time:.2f}s'
            })
            
            # Log to wandb every iteration
            global_step = self.epoch * self.config.training.num_iterations_per_epoch + batch_idx
            wandb.log({
                'train/loss_iter': current_loss,
                'train/learning_rate': current_lr,
                'train/iter_time': iter_time,
                'train/iteration': global_step
            }, step=global_step)
        
        # Close progress bar
        pbar.close()
        
        # Compute epoch metrics
        #train_metrics = self.train_metrics.compute()
        train_metrics = {}

        return train_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        loss_meter = AverageMeter()
        
        # Use fixed batch 0 to track same patients across epochs
        display_batch_idx = 0
        
        # Storage for samples to log
        random_samples = []  # Store data for predetermined random batch
        
        # Create progress bar for validation
        pbar = tqdm(
            self.val_loader,
            desc=f"Validation",
            ncols=100,
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images, targets = batch['image'], batch['label']
                
                if targets.dim() == 5 and targets.size(1) == 1:
                    targets = targets.squeeze(1)
                else:
                    raise ValueError(f"Targets should have shape (B, 1, H, W, D), got {targets.shape}")
                
                # Move to device (normalization now happens in dataset)
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                if self.config.model.deep_supervision and isinstance(outputs, list):
                    main_output = outputs[0]
                else:
                    main_output = outputs

                loss = self.val_loss(main_output, targets)
                
                # Update metrics
                loss_meter.update(loss.item(), images.size(0))
                batch_metrics = self.val_metrics.update(main_output, targets)
                
                current_loss = loss.item()
                
                # Save data for predetermined random batch
                if batch_idx == display_batch_idx:
                    # Save all samples in this batch
                    batch_probs = torch.softmax(main_output, dim=1)
                    
                    for i in range(images.shape[0]):
                        img_slice, target_slice, pred_slice, _ = extract_slice_with_lesion(
                            images[i],
                            targets[i],
                            batch_probs[i]
                        )
                        
                        random_samples.append({
                            'img_slice': img_slice,
                            'target_slice': target_slice,
                            'pred_slice': pred_slice,
                            'dice': batch_metrics['dice_mean'],
                            'loss': current_loss,
                            'batch_idx': batch_idx,
                            'sample_idx': i
                        })
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

                # Visualize every 20 batches for validation if enabled (not at the very beginning)
                if self.visualizer is not None and batch_idx > 0 and batch_idx % 20 == 0:
                    self.visualizer.visualize_batch(
                        images, targets, main_output,
                        self.epoch, batch_idx, loss.item(),
                        prefix="val"
                    )
                
                # CRITICAL: Clear references to prevent memory leak
                del images, targets, outputs, loss, main_output
        
        # Close progress bar
        pbar.close()

        # Add samples to the persistent validation table
        self._add_samples_to_table(random_samples)
        
        # Compute epoch metrics - dice, loss, and calibration metrics
        full_metrics = self.val_metrics.compute()
        val_metrics = {
            'loss': loss_meter.avg,
            'dice_mean': full_metrics.get('dice_mean', 0.0),
            'dice_hard': full_metrics.get('dice_hard', 0.0),
            'samples': self.validation_table
        }
        
        # Add calibration metrics if available
        if 'ece' in full_metrics:
            val_metrics['ece'] = full_metrics['ece']
            val_metrics['mce'] = full_metrics['mce']
            val_metrics['nll'] = full_metrics['nll']
            val_metrics['brier_score'] = full_metrics['brier_score']

        # Clean up to free memory
        del random_samples
        
        return val_metrics
    
    def _add_samples_to_table(self, samples: list):
        """Add validation samples to the persistent wandb table"""
        
        # Define class labels for wandb masks
        class_labels = {
            0: "background",
            1: "metastasis"
        }
        
        # Log all samples from the random batch
        for i, sample in enumerate(samples):
            sample_id = f"batch_{sample['batch_idx']}_sample_{sample.get('sample_idx', i)}"
            
            # Create wandb Image with masks
            mask_img = wandb.Image(
                sample['img_slice'],
                masks={
                    "ground_truth": {"mask_data": sample['target_slice'], "class_labels": class_labels},
                    "prediction": {"mask_data": sample['pred_slice'], "class_labels": class_labels}
                }
            )
            
            # Add row with epoch column
            self.validation_table.add_data(
                self.epoch,
                sample_id,
                f"{sample['loss']:.4f}",
                f"{sample['dice']:.4f}",
                mask_img
            )
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for fold {self.fold}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.epoch, self.config.training.max_epochs):
            self.epoch = epoch
            is_best = False
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            self.metrics_history['train_loss'].append(train_metrics.get('loss', 0.0))
            lr = self.optimizer.param_groups[0]['lr']
            self.metrics_history['learning_rate'].append(lr)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Validate
            val_metrics = {}
            if epoch % self.config.training.val_check_interval == 0:
                # Validate
                val_metrics = self.validate_epoch()

                # Record validation metrics
                self.metrics_history['val_loss'].append(val_metrics['loss'])
                self.metrics_history['val_dice'].append(val_metrics.get('dice_mean', 0.0))

                current_metric = val_metrics.get('dice_mean', 0.0)
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
            
            epoch_time = time.time() - epoch_start_time

            # Build log dict without the table
            log_dict = {
                'epoch': epoch,
                'train/learning_rate': lr,
                'train/epoch_time': epoch_time,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items() if k != 'samples'}
            }
            
            # Add table to the same log call if validation happened
            if 'samples' in val_metrics:
                log_dict['val/samples'] = self.validation_table
            
            wandb.log(log_dict)
            
            # Print progress - improved formatting
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{self.config.training.max_epochs} Summary ({epoch_time:.1f}s)")
            print(f"{'='*80}")
            print(f"  Learning Rate: {lr:.6f}")
            if train_metrics:
                print(f"  Training Loss: {train_metrics.get('loss', 0.0):.4f}")

            if len(val_metrics.keys()) > 0:
                print(f"  Validation Loss: {val_metrics['loss']:.4f}")
                print(f"  Validation Dice: {val_metrics['dice_mean']:.4f}")
                print(f"  Validation Hard Dice: {val_metrics['dice_hard']:.4f}")
                
                # Print calibration metrics if available
                if 'ece' in val_metrics:
                    print(f"  Calibration Metrics:")
                    print(f"    ECE: {val_metrics['ece']:.4f}")
                    print(f"    MCE: {val_metrics['mce']:.4f}")
                    print(f"    NLL: {val_metrics['nll']:.4f}")
                    print(f"    Brier Score: {val_metrics['brier_score']:.4f}")
                
                if is_best:
                    print(f"  ★ New Best Dice: {self.best_metric:.4f} ★")
            print(f"{'='*80}\n")
            
            save_checkpoint(
                {
                    'epoch': epoch,
                    'fold': self.fold,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                    'best_metric': self.best_metric,
                    'config': self.config,  # Save the actual Config object directly
                    'metrics_history': self.metrics_history,
                    'run_id': self.run_id  # Save run_id for directory structure
                },
                is_best=is_best,
                checkpoint_dir=self.output_dir
            )
            
            # Early stopping (only check when validation was performed)
            if len(val_metrics.keys()) > 0:
                if self.early_stopping(current_metric):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        print(f"Training completed for fold {self.fold}")
        print(f"Best validation Dice: {self.best_metric:.4f}")
        
        # Create summary plot if visualization enabled
        if self.visualizer is not None:
            from visualization import create_summary_plot
            create_summary_plot(self.metrics_history, str(self.output_dir / "visualizations"))
        
        return self.best_metric


def run_single_fold(fold: int, config, enable_visualization: bool = False, resume_id: Optional[str] = None):
    """Run training for a single fold"""
    print(f"\n{'='*50}")
    print(f"Starting fold {fold}")
    print(f"{'='*50}")
    
    trainer = Trainer(config, fold, enable_visualization=enable_visualization, resume_id=resume_id)
    best_metric = trainer.train()
    
    wandb.finish()
    
    return best_metric


def run_cross_validation(config, enable_visualization: bool = False, resume_id: Optional[str] = None):
    """Run full 5-fold cross validation"""
    fold_results = []
    
    for fold in range(config.data.num_folds):
        best_metric = run_single_fold(fold, config, enable_visualization=enable_visualization, resume_id=resume_id)
        fold_results.append(best_metric)
        
        print(f"Fold {fold} completed with best Dice: {best_metric:.4f}")
    
    # Calculate cross-validation statistics
    mean_metric = np.mean(fold_results)
    std_metric = np.std(fold_results)
    
    print(f"\n{'='*50}")
    print("Cross-validation Results:")
    print(f"{'='*50}")
    for i, result in enumerate(fold_results):
        print(f"Fold {i}: {result:.4f}")
    print(f"Mean ± Std: {mean_metric:.4f} ± {std_metric:.4f}")
    
    # Save results
    results = {
        'fold_results': fold_results,
        'mean': mean_metric,
        'std': std_metric,
        'config': config.__dict__
    }
    
    results_file = Path(config.output_dir) / 'cross_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


if __name__ == "__main__":
    # Load configuration
    config = get_config()
    
    # Run cross-validation
    results = run_cross_validation(config)