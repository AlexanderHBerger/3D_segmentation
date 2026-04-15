"""
Training script for MedNeXt segmentation model
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
from torch.amp import GradScaler, autocast
import wandb
from typing import Dict, Optional, Any, List
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
from utils import (AverageMeter, save_checkpoint, load_checkpoint, EarlyStopping, extract_slice_with_foreground)
from visualization import TrainingVisualizer
from architectures import calculate_n_stages

class Trainer:
    """Main training class"""

    def __init__(self, config, fold: int, enable_visualization: bool = False, resume_id: Optional[str] = None, warm_restart: bool = False, init_checkpoint: Optional[str] = None):
        self.fold = fold
        self.enable_visualization = enable_visualization
        self.run_id = resume_id or getattr(config, 'run_id', None)
        self.warm_restart = warm_restart
        self.init_checkpoint = init_checkpoint
        self.train_on_all = getattr(config.data, 'train_on_all', False)

        # Determine output path if resuming (directory must exist for checkpoint loading)
        if resume_id is not None:
            fold_label = "all" if self.train_on_all else str(fold)
            self.output_dir = Path(config.output_dir) / f"fold_{fold_label}_{self.run_id}"
            if not self.output_dir.exists():
                raise FileNotFoundError(f"Resume requested but directory does not exist: {self.output_dir}")
        
        # Load checkpoint from disk if resuming
        checkpoint = self._load_checkpoint_from_disk() if resume_id is not None else None

        # Load config (either from checkpoint or use provided config)
        self.config = self._load_config(config, checkpoint)
        self._nan_abort = False  # Set by train_epoch if 10 consecutive NaN

        # Set up device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seeds
        self.set_random_seeds(self.config.seed + (0 if self.train_on_all else fold))
        
        # Initialize mixed precision
        if self.config.mixed_precision:
            self.scaler = GradScaler(
                'cuda' if self.device.type == 'cuda' else 'cpu',
                init_scale=self.config.training.initial_grad_scale,
            )
        else:
            self.scaler = None
        
        # Initialize model
        self.model = self._create_model()
        
        # Load model weights from init_checkpoint if provided (for transfer learning / initialization only)
        if self.init_checkpoint is not None:
            self._load_model_weights_only(self.init_checkpoint)

        # Apply finetuning modifications (freeze encoder, LoRA)
        # Order: load pretrained weights → apply LoRA (decomposes MHA and copies weights) → freeze encoder → create optimizer
        # For --resume: LoRA must be applied before _load_checkpoint so state dict keys match
        if self.config.training.lora_enabled:
            self._apply_lora()
        if self.config.training.spectral_norm and self.config.text_prompted.enabled:
            self.model.apply_spectral_norm()
            print("Applied spectral normalization to projection layers")
        if self.config.training.freeze_encoder:
            self._apply_encoder_freezing()

        # Initialize optimizer and scheduler
        # Note: For warm restart, optimizer LR will be adjusted after checkpoint loading
        self.optimizer = self._create_optimizer()
        self.scheduler = None  # Will be created after checkpoint loading if needed
        
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
            compute_calibration=False,
            compute_topological=False,
        )
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Initialize tracking variables
        self.epoch = 0
        self.best_metric = 0.0
        if not self.train_on_all:
            self.early_stopping = EarlyStopping(
                patience=self.config.training.patience,
                mode=self.config.training.monitor_mode
            )
        else:
            self.early_stopping = None

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
        
        # Step 4: Create scheduler after checkpoint loading (for warm restart support)
        if self.scheduler is None:
            self.scheduler = self._create_scheduler()
        
        # Initialize wandb (with resume if we have a run_id) and create output directory
        if wandb.run is None:
            self._init_wandb()
        elif wandb.run is not None and self.run_id is None:
            # wandb was already initialized (e.g., by sweep agent)
            self.run_id = wandb.run.id
            print(f"Using existing wandb run with ID: {self.run_id}")

        # Set output directory if not already set (it's set earlier when resuming from checkpoint)
        if not hasattr(self, 'output_dir'):
            fold_label = "all" if self.train_on_all else str(fold)
            self.output_dir = Path(config.output_dir) / f"fold_{fold_label}_{self.run_id}"
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
    
    def _load_model_weights_only(self, checkpoint_path: str):
        """
        Load only model weights from a checkpoint file for initialization.
        This starts a completely new training run with pretrained weights.
        
        Args:
            checkpoint_path: Path to the checkpoint file to load weights from
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {checkpoint_path}")
        
        print(f"\n{'='*80}")
        print("INITIALIZING MODEL FROM CHECKPOINT (new training run)")
        print(f"{'='*80}")
        print(f"  Loading weights from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' not in checkpoint:
            raise ValueError(f"No model_state_dict found in checkpoint: {checkpoint_path}")
        
        # Load model weights
        model_state = checkpoint['model_state_dict']
        
        # Try to load with strict=True first, fall back to strict=False if architecture differs
        try:
            self.model.load_state_dict(model_state, strict=True)
            print("  Model weights loaded successfully (strict mode)")
        except RuntimeError as e:
            print(f"  Warning: Strict loading failed, trying non-strict: {e}")
            missing, unexpected = self.model.load_state_dict(model_state, strict=False)
            if missing:
                print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        
        # Log checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"  Source checkpoint was at epoch: {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            print(f"  Source checkpoint best metric: {checkpoint['best_metric']:.4f}")
        if 'config' in checkpoint:
            src_config = checkpoint['config']
            if hasattr(src_config, 'model'):
                print(f"  Source model: {src_config.model.architecture} ({src_config.model.model_size})")
        
        print(f"\n  Starting NEW training run with initialized weights")
        print(f"  Epoch: 0, Best metric: 0.0, Fresh optimizer/scheduler")
        print(f"{'='*80}\n")

    def _get_encoder_module(self):
        """Get the encoder submodule (works for both standard and text-prompted models)."""
        if hasattr(self.model, 'encoder'):
            return self.model.encoder
        return None

    def _apply_encoder_freezing(self):
        """Freeze all encoder parameters so they are not updated during training."""
        encoder = self._get_encoder_module()
        if encoder is None:
            print("WARNING: Could not identify encoder module — skipping freeze")
            return

        frozen_count = 0
        for param in encoder.parameters():
            param.requires_grad = False
            frozen_count += param.numel()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n{'='*80}")
        print("ENCODER FREEZING ENABLED")
        print(f"  Frozen encoder parameters: {frozen_count:,}")
        print(f"  Total model parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"{'='*80}\n")

    def _apply_lora(self):
        """Apply LoRA adapters to the transformer decoder attention layers."""
        if not hasattr(self.model, 'transformer_decoder'):
            print("WARNING: Model has no transformer_decoder — skipping LoRA")
            return

        from lora import apply_lora_to_transformer
        frozen, lora_trainable = apply_lora_to_transformer(
            self.model.transformer_decoder,
            rank=self.config.training.lora_rank,
            alpha=self.config.training.lora_alpha,
            dropout=self.config.training.lora_dropout,
        )

        # Move new LoRA parameters to the same device as the model
        self.model.transformer_decoder.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n{'='*80}")
        print("LoRA ENABLED ON TRANSFORMER DECODER")
        print(f"  Rank: {self.config.training.lora_rank}, Alpha: {self.config.training.lora_alpha}")
        print(f"  Frozen transformer parameters: {frozen:,}")
        print(f"  New LoRA parameters: {lora_trainable:,}")
        print(f"  Total model parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"{'='*80}\n")

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
        
        # Load config from checkpoint
        if 'config' not in checkpoint:
            raise ValueError(f"No config found in checkpoint")
        
        loaded_config = checkpoint['config']
        
        # If resuming without use_new_config flag, use checkpoint config entirely
        if not hasattr(config, '_use_new_config') or not config._use_new_config:
            print(f"\n{'='*80}")
            print(f"Loading configuration from checkpoint (NORMAL RESUME)")
            print(f"{'='*80}\n")
            
            print("Configuration loaded successfully from checkpoint!")
            print(f"  Model: {loaded_config.model.architecture} ({loaded_config.model.model_size})")
            print(f"  Batch size: {loaded_config.training.batch_size}")
            print(f"  Learning rate: {loaded_config.training.initial_lr}")
            print(f"  Max epochs: {loaded_config.training.max_epochs}")
            print(f"  W&B Project: {loaded_config.wandb.project}")
            print(f"{'='*80}\n")
            
            return loaded_config
        else:
            # Warm restart: use checkpoint config but override max_epochs from new config
            print(f"\n{'='*80}")
            print("WARM RESTART: Using checkpoint configuration with extended epochs")
            print(f"{'='*80}")
            print(f"Loading config from checkpoint...")
            print(f"  Model: {loaded_config.model.architecture} ({loaded_config.model.model_size})")
            print(f"  Batch size: {loaded_config.training.batch_size}")
            print(f"  Learning rate: {loaded_config.training.initial_lr}")
            print(f"  Original max epochs: {loaded_config.training.max_epochs}")
            print(f"  Extended max epochs: {config.training.max_epochs}")
            print(f"  W&B Project: {loaded_config.wandb.project}")
            #print(f"  Loss function: {loaded_config.training.loss_function}")
            print(f"  Patch size: {loaded_config.data.patch_size}")
            
            # Override warm restart-specific params from the new config
            loaded_config.training.max_epochs = config.training.max_epochs
            loaded_config.training.warm_restart_lr_factor = config.training.warm_restart_lr_factor

            print(f"  Warm restart LR factor: {loaded_config.training.warm_restart_lr_factor}")
            print(f"\nUsing checkpoint config with max_epochs updated to {loaded_config.training.max_epochs}")
            print(f"{'='*80}\n")
            return loaded_config
        
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint in the fold directory"""
        # Check for checkpoint.pth first (most recent)
        checkpoint_file = self.output_dir / "checkpoint.pth"
        if checkpoint_file.exists():
            return checkpoint_file

        return None
    
    def _load_checkpoint(self, checkpoint):
        """Load checkpoint and resume training"""

        # Load model, optimizer state, and scaler (scheduler handled separately below)
        checkpoint = load_checkpoint(
            checkpoint,
            self.model,
            self.optimizer,
            None,  # Scheduler loaded below after creation
            self.scaler
        )

        # Restore training state
        original_epoch = checkpoint.get('epoch', 0)
        self.epoch = original_epoch + 1  # Continue from next epoch
        self.best_metric = checkpoint.get('best_metric', 0.0)

        # Restore metrics history if available
        if 'metrics_history' in checkpoint:
            self.metrics_history = checkpoint['metrics_history']

        # For normal resume (not warm restart), create scheduler and restore its state
        if not self.warm_restart and 'scheduler_state_dict' in checkpoint:
            self.scheduler = self._create_scheduler()
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  Scheduler state restored (last_epoch={checkpoint['scheduler_state_dict'].get('last_epoch', '?')})")
        
        # Apply warm restart if requested
        if self.warm_restart:
            remaining_epochs = self.config.training.max_epochs - self.epoch
            print(f"\n{'='*80}")
            print("WARM RESTART MODE ENABLED")
            print(f"{'='*80}")
            print(f"  Original training completed {original_epoch + 1} epochs (epoch 0-{original_epoch})")
            print(f"  Will continue from epoch {self.epoch} to epoch {self.config.training.max_epochs - 1}")
            print(f"  Remaining epochs: {remaining_epochs}")
            
            # Adjust learning rate with warm restart factor
            # Apply proportionally to each param group (preserves differential LR ratios)
            factor = self.config.training.warm_restart_lr_factor
            print(f"  Warm restart factor: {factor}")
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * factor
                print(f"  LR: {old_lr:.6f} → {param_group['lr']:.6f}")

            print(f"{'='*80}\n")
        else:
            print(f"Checkpoint loaded successfully!")
            print(f"  Resuming from epoch: {self.epoch}")
            print(f"  Best metric so far: {self.best_metric:.4f}")
            print(f"  Current LR: {self.optimizer.param_groups[-1]['lr']:.6f}")
        
        # Check if fold matches
        checkpoint_fold = checkpoint.get('fold', self.fold)
        if checkpoint_fold != self.fold:
            raise ValueError(f"Checkpoint is from fold {checkpoint_fold}, but training fold {self.fold}")
    
    def _convert_valid_bounds(self, valid_bounds_tensor: torch.Tensor):
        """Convert valid_bounds tensor from batch to list of tuples for loss function.
        
        Args:
            valid_bounds_tensor: (B, 6) tensor where each row is [w_min, w_max, h_min, h_max, d_min, d_max]
                                or [-1, -1, -1, -1, -1, -1] if all voxels are valid
        
        Returns:
            List of tuples ((w_min, w_max), (h_min, h_max), (d_min, d_max)) or None for each sample,
            where None indicates all voxels are valid.
        """
        if valid_bounds_tensor is None:
            return None
        
        result = []
        for i in range(valid_bounds_tensor.shape[0]):
            bounds = valid_bounds_tensor[i]
            if bounds[0] == -1:  # Sentinel value for "all valid"
                result.append(None)
            else:
                result.append((
                    (int(bounds[0]), int(bounds[1])),
                    (int(bounds[2]), int(bounds[3])),
                    (int(bounds[4]), int(bounds[5]))
                ))
        return result
    
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
        """Create optimizer with optional parameter groups for differential LR."""
        lr = self.config.training.initial_lr
        common_kwargs = dict(
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay,
            nesterov=True,
        )

        # Differential LR: separate param groups for encoder vs rest
        use_param_groups = (
            not self.config.training.freeze_encoder
            and self.config.training.encoder_lr_factor != 1.0
        )

        if use_param_groups:
            encoder = self._get_encoder_module()
            if encoder is not None:
                encoder_param_ids = set(id(p) for p in encoder.parameters())
                encoder_params = [p for p in self.model.parameters() if p.requires_grad and id(p) in encoder_param_ids]
                other_params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in encoder_param_ids]

                encoder_lr = lr * self.config.training.encoder_lr_factor
                param_groups = [
                    {'params': encoder_params, 'lr': encoder_lr},
                    {'params': other_params, 'lr': lr},
                ]

                print(f"  Differential LR: encoder={encoder_lr:.6f}, rest={lr:.6f} "
                      f"(factor={self.config.training.encoder_lr_factor})")

                optimizer = optim.SGD(param_groups, **common_kwargs)
                return optimizer

        # Default: single param group with only trainable params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.SGD(trainable_params, lr=lr, **common_kwargs)
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.training.lr_scheduler == 'poly':
            # Calculate remaining iterations for scheduler
            # For warm restart: calculate remaining epochs from current position
            # For normal training: use full max_epochs
            if self.warm_restart:
                # Remaining epochs from current epoch to max_epochs
                total_iters = self.config.training.max_epochs - self.epoch
                print(f"  Creating PolynomialLR scheduler for warm restart:")
                print(f"    Current epoch: {self.epoch}")
                print(f"    Max epochs: {self.config.training.max_epochs}")
                print(f"    Remaining iterations: {total_iters}")
            else:
                # Calculate remaining iterations if resuming
                total_iters = self.config.training.max_epochs
                print(f"  Creating PolynomialLR scheduler: total_iters={total_iters}")
            
            scheduler = PolynomialLR(
                self.optimizer,
                total_iters=total_iters,
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
            max_samples=self.config.data.max_samples
        )
        
        # Get transforms
        train_transforms = get_training_transforms(self.config)
        val_transforms = get_validation_transforms(self.config)
        
        # Load text-prompted data if enabled
        precomputed_embeddings = None
        prompts_data = None
        if hasattr(self.config, 'text_prompted') and self.config.text_prompted.enabled:
            tp = self.config.text_prompted
            if tp.precomputed_embeddings_path:
                print(f"Loading precomputed text embeddings from {tp.precomputed_embeddings_path}")
                precomputed_embeddings = torch.load(
                    tp.precomputed_embeddings_path, map_location='cpu', weights_only=True
                )
            if tp.prompts_json_path:
                import json as _json
                from pathlib import Path as _Path
                prompts_path = _Path(tp.prompts_json_path)
                if prompts_path.is_dir():
                    print(f"Loading prompts from directory {prompts_path}")
                    prompts_data = {}
                    for jf in sorted(prompts_path.glob("*.json")):
                        with open(jf) as f:
                            prompts_data[jf.stem] = _json.load(f)
                else:
                    print(f"Loading prompts from {prompts_path}")
                    with open(prompts_path, 'r') as f:
                        prompts_data = _json.load(f)

        # Create data loaders with use_preprocessed flag
        train_loader, val_loader = create_data_loaders(
            data_manager=data_manager,
            fold=self.fold,
            config=self.config,
            transforms_train=train_transforms,
            transforms_val=val_transforms,
            use_preprocessed=self.config.data.use_preprocessed,
            train_on_all=self.train_on_all,
            precomputed_embeddings=precomputed_embeddings,
            prompts_data=prompts_data,
        )

        return train_loader, val_loader
    
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        # Determine if we're resuming a run
        resume_mode = "must" if self.run_id is not None else None
        
        # Always pass the full actual config being used to wandb
        # This ensures wandb displays what's actually running and helps detect errors
        # For resume: config comes from checkpoint
        # For warm restart: config comes from checkpoint with max_epochs updated
        # For new run: config comes from config.py
        fold_label = "all" if self.train_on_all else str(self.fold)
        wandb_config = {
            'fold': self.fold,
            'train_on_all': self.train_on_all,
            'model': self.config.model.__dict__,
            'training': self.config.training.__dict__,
            'data': self.config.data.__dict__,
            'augmentation': self.config.augmentation.__dict__
        }

        tags = self.config.wandb.tags + [f"fold_{fold_label}"]
        if self.train_on_all:
            tags.append("train_on_all")

        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=f"fold_{fold_label}",
            tags=tags,
            notes=self.config.wandb.notes,
            id=self.run_id,  # Use existing run ID if resuming
            resume=resume_mode,  # Resume the run if ID is provided
            config=wandb_config
        )
        
        # Store the run ID for checkpointing (important for new runs)
        if self.run_id is None:
            self.run_id = wandb.run.id
            print(f"Created new wandb run with ID: {self.run_id}")
        else:
            mode = "warm restart" if self.warm_restart else "normal resume"
            print(f"Resumed wandb run with ID: {self.run_id} ({mode})")
            print(f"Wandb config updated to reflect actual running configuration")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch (nnUNet style: fixed 250 iterations)"""
        self.model.train()

        # Tracking variables
        loss_meter = AverageMeter()
        nan_count = 0
        consecutive_nan = 0
        
        # Reset visualization counter
        if self.visualizer is not None:
            self.visualizer.reset_count()
        
        self.train_loader.dataset.set_epoch(self.epoch)
        
        # Create progress bar for this epoch
        pbar = tqdm(
            range(self.config.training.num_iterations_per_epoch),
            desc=f"Epoch {self.epoch}/{self.config.training.max_epochs}",
            ncols=100,
            leave=True,
            smoothing=1
        )
        # Track iteration time
        iter_start_time = time.time()
        self.train_loader_iter = iter(self.train_loader)

        # Fixed number of iterations per epoch (nnUNet approach)
        for batch_idx in pbar:
            try:
                batch = next(self.train_loader_iter)
            except StopIteration:
                new_iter_start_time = time.time()
                self.train_loader_iter = iter(self.train_loader)
                batch = next(self.train_loader_iter)
                print(f"\nReinitialized train loader iterator in {time.time() - new_iter_start_time:.2f} seconds")
            
            images, targets = batch['image'], batch['label']
            valid_bounds = batch.get('valid_bounds', None)
            weight_map = batch.get('weight_map', None)
            text_embedding = batch.get('text_embedding', None)
            distance_field = batch.get('distance_field', None)

            # Standard mode: squeeze (B, 1, H, W, D) -> (B, H, W, D) for multi-class loss
            # Binary mode (num_classes=1): keep (B, 1, H, W, D) for binary CombinedLoss
            if self.config.data.num_classes == 1:
                if targets.dim() == 4:
                    targets = targets.unsqueeze(1)  # (B, H, W, D) -> (B, 1, H, W, D)
            else:
                if targets.dim() == 5 and targets.size(1) == 1:
                    targets = targets.squeeze(1)
                elif targets.dim() != 4:
                    raise ValueError(f"Targets should have shape (B, 1, H, W, D) or (B, H, W, D), got {targets.shape}")

            # Move to device (normalization now happens in dataset)
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if weight_map is not None:
                weight_map = weight_map.to(self.device, non_blocking=True)
            if text_embedding is not None:
                text_embedding = text_embedding.to(self.device, non_blocking=True)
            if distance_field is not None:
                distance_field = distance_field.to(self.device, non_blocking=True)

            # Convert valid_bounds tensor to list of tuples for loss function
            valid_bounds_list = None
            if valid_bounds is not None:
                valid_bounds_list = self._convert_valid_bounds(valid_bounds)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=self.scaler is not None):
                if text_embedding is not None:
                    outputs = self.model(images, text_embedding)
                else:
                    outputs = self.model(images)

                if self.config.model.deep_supervision and isinstance(outputs, list):
                    loss, loss_components = self.criterion(outputs, targets, valid_bounds=valid_bounds_list, weight_map=weight_map, distance_field=distance_field)
                    main_output = outputs[0]
                else:
                    loss, loss_components = self.criterion(outputs, targets, valid_bounds=valid_bounds_list, weight_map=weight_map, distance_field=distance_field)
                    main_output = outputs

            # --- NaN detection: check BEFORE backward to avoid corrupting momentum ---
            current_loss = loss.item()
            if not math.isfinite(current_loss):
                nan_count += 1
                consecutive_nan += 1
                if nan_count <= 5:
                    print(f"\n[NaN] epoch={self.epoch} iter={batch_idx} loss={current_loss}")
                    self._log_nan_diagnostics()
                elif nan_count % 50 == 0:
                    print(f"\n[NaN] epoch={self.epoch} iter={batch_idx} (total NaN count: {nan_count})")
                self.optimizer.zero_grad()
                del images, targets, outputs, loss, loss_components, main_output
                if consecutive_nan >= 10:
                    print(f"\n[FATAL] 10 consecutive NaN iterations — stopping training")
                    self._nan_abort = True
                    break
                continue
            consecutive_nan = 0

            # Backward pass with gradient clipping
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.optimizer.step()

            # Extract loss component values before deletion (for wandb logging)
            loss_component_values = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_components.items()}

            # Track output range for diagnostics
            with torch.no_grad():
                output_min = main_output.min().item()
                output_max = main_output.max().item()

            # Update metrics
            loss_meter.update(current_loss, images.size(0))
            # Visualize every 50 batches if enabled (not at the very beginning)
            if self.visualizer is not None and batch_idx > 0 and batch_idx % 50 == 0:
                self.visualizer.visualize_batch(
                    images, targets, main_output,
                    self.epoch, batch_idx, current_loss,
                    prefix="train"
                )

            # CRITICAL: Clear references to prevent memory leak
            del images, targets, outputs, loss, loss_components, main_output

            # Calculate iteration time
            iter_time = time.time() - iter_start_time

            # Reset iteration timer
            iter_start_time = time.time()

            # Get current learning rate (use last group = primary/non-encoder LR)
            current_lr = self.optimizer.param_groups[-1]['lr']

            # Update progress bar with current loss and iteration time
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{loss_meter.avg:.4f}',
                'iter_time': f'{iter_time:.2f}s'
            })

            # Log to wandb every iteration
            if wandb.run is not None:
                global_step = self.epoch * self.config.training.num_iterations_per_epoch + batch_idx
                log_dict = {
                    'train/loss_iter': current_loss,
                    'train/learning_rate': current_lr,
                    'train/iter_time': iter_time,
                    'train/iteration': global_step,
                    'debug/grad_norm': grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    'debug/output_min': output_min,
                    'debug/output_max': output_max,
                }
                if self.scaler is not None:
                    log_dict['debug/scaler_scale'] = self.scaler.get_scale()
                # Log encoder LR separately when using differential LR
                if len(self.optimizer.param_groups) > 1:
                    log_dict['train/encoder_lr'] = self.optimizer.param_groups[0]['lr']
                # Add loss components to log dict
                for component_name, component_value in loss_component_values.items():
                    log_dict[f'train/loss_{component_name}'] = component_value
                # Log weight norms every 50 iterations
                if batch_idx % 50 == 0:
                    log_dict.update(self._compute_weight_norms())
                wandb.log(log_dict, step=global_step)
        
        # Close progress bar
        pbar.close()

        # Compute epoch metrics
        train_metrics = {'loss': loss_meter.avg}
        self.metrics_history['nan_count'].append(nan_count)
        if nan_count > 0:
            print(f"  NaN iterations this epoch: {nan_count}")

        return train_metrics

    def _compute_weight_norms(self) -> Dict[str, float]:
        """Compute L2 norms of key model layers for diagnostics."""
        norms = {}
        layer_names = {
            'debug/wnorm_project_text': 'project_text_embed',
            'debug/wnorm_project_bottleneck': 'project_bottleneck_embed',
            'debug/wnorm_transformer_norm': 'transformer_decoder.norm',
            'debug/wnorm_decoder_proj_0': 'project_to_decoder_channels.0',
            'debug/wnorm_decoder_stage4': 'decoder.stages.4',
        }
        for log_key, layer_prefix in layer_names.items():
            total_norm = 0.0
            found = False
            for name, param in self.model.named_parameters():
                if name.startswith(layer_prefix) and param.requires_grad:
                    total_norm += param.data.norm(2).item() ** 2
                    found = True
            if found:
                norms[log_key] = total_norm ** 0.5
        return norms

    def _log_nan_diagnostics(self):
        """Log detailed diagnostics when NaN loss is detected."""
        print("  --- NaN Diagnostics ---")
        norms = self._compute_weight_norms()
        for key, val in sorted(norms.items()):
            print(f"  {key}: {val:.4f}")
        if self.scaler is not None:
            print(f"  scaler_scale: {self.scaler.get_scale():.0f}")
        # Log to wandb
        if wandb.run is not None:
            wandb_dict = {**norms, 'debug/nan_event': 1}
            if self.scaler is not None:
                wandb_dict['debug/scaler_scale'] = self.scaler.get_scale()
            wandb.run.log(wandb_dict)

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
        
        is_text_prompted = hasattr(self.config, 'text_prompted') and self.config.text_prompted.enabled

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images, targets = batch['image'], batch['label']
                text_embedding = batch.get('text_embedding', None)

                is_binary = self.config.data.num_classes == 1
                if is_binary:
                    if targets.dim() == 4:
                        targets = targets.unsqueeze(1)
                else:
                    if targets.dim() == 5 and targets.size(1) == 1:
                        targets = targets.squeeze(1)
                    elif targets.dim() != 4:
                        raise ValueError(f"Targets should have shape (B, 1, H, W, D) or (B, H, W, D), got {targets.shape}")

                # Move to device (normalization now happens in dataset)
                images = images.to(self.device)
                targets = targets.to(self.device)
                if text_embedding is not None:
                    text_embedding = text_embedding.to(self.device)

                # Forward pass
                if text_embedding is not None:
                    outputs = self.model(images, text_embedding)
                else:
                    outputs = self.model(images)

                if self.config.model.deep_supervision and isinstance(outputs, list):
                    main_output = outputs[0]
                else:
                    main_output = outputs

                loss, _ = self.val_loss(main_output, targets)

                # Update metrics — squeeze targets to (B, H, W, D) for metric functions
                loss_meter.update(loss.item(), images.size(0))
                targets_for_metrics = targets.squeeze(1) if is_binary else targets
                batch_metrics = self.val_metrics.update(main_output, targets_for_metrics)
                
                current_loss = loss.item()
                
                # Save data for predetermined random batch
                if batch_idx == display_batch_idx:
                    # Save all samples in this batch
                    if is_binary:
                        batch_probs = torch.sigmoid(main_output)
                    else:
                        batch_probs = torch.softmax(main_output, dim=1)

                    targets_for_vis = targets_for_metrics  # (B, H, W, D)
                    for i in range(images.shape[0]):
                        img_slice, target_slice, pred_slice, _ = extract_slice_with_foreground(
                            images[i],
                            targets_for_vis[i],
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

        # Create validation images
        val_images = self._create_validation_images(random_samples)
        
        # Compute epoch metrics - dice, loss, and calibration metrics
        full_metrics = self.val_metrics.compute()

        # Add loss and samples to metrics for logging
        full_metrics['loss'] = loss_meter.avg
        full_metrics['samples'] = val_images

        # Clean up to free memory
        del random_samples
        
        return full_metrics
    
    def _create_validation_images(self, samples: list) -> List[wandb.Image]:
        """Create wandb images for validation samples"""
        
        # Define class labels for wandb masks
        class_labels = {
            0: "background",
            1: "foreground"
        }
        
        images_list = []
        # Log all samples from the random batch
        for i, sample in enumerate(samples):
            # Create wandb Image with masks
            mask_img = wandb.Image(
                sample['img_slice'],
                masks={
                    "ground_truth": {"mask_data": sample['target_slice'], "class_labels": class_labels},
                    "prediction": {"mask_data": sample['pred_slice'], "class_labels": class_labels}
                },
                caption=f"Sample {sample.get('sample_idx', i)} - Loss: {sample['loss']:.4f}, Dice: {sample['dice']:.4f}"
            )
            images_list.append(mask_img)
            
        return images_list
    
    def train(self):
        """Main training loop"""
        fold_label = "all" if self.train_on_all else str(self.fold)
        print(f"Starting training for fold {fold_label}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader is not None:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        else:
            print("Validation: DISABLED (train_on_all mode)")

        # Run initial validation before training when using pretrained weights.
        # Skip when the val dataset is effectively empty (virtual size == 0) —
        # this happens e.g. in text-prompted feasibility runs where the curated
        # per-lesion subset only covers training cases and the val loader
        # cannot produce text_embedding batches. Calling self.model(images)
        # in validate_epoch would then raise TypeError on the missing
        # text_embedding positional arg. PatchDataset exposes __len__
        # (== virtual size; see line 1036 which already relies on it).
        val_dataset_size = (
            len(self.val_loader.dataset) if self.val_loader is not None else 0
        )
        if (
            self.init_checkpoint is not None
            and not self.train_on_all
            and self.val_loader is not None
            and self.epoch == 0
            and val_dataset_size > 0
        ):
            print(f"\n{'='*60}")
            print("Running initial validation (pretrained weight baseline)")
            print(f"{'='*60}")
            init_val_metrics = self.validate_epoch()
            init_dice = init_val_metrics.get('dice_mean', 0.0)
            print(f"Pretrained baseline — val_dice: {init_dice:.4f}, val_loss: {init_val_metrics['loss']:.4f}")
            if wandb.run is not None:
                wandb.run.log({
                    'epoch': -1,
                    **{f'val/{k}': v for k, v in init_val_metrics.items()},
                }, step=0)
        elif (
            self.init_checkpoint is not None
            and not self.train_on_all
            and self.val_loader is not None
            and self.epoch == 0
            and val_dataset_size == 0
        ):
            print(
                "Skipping initial pre-validation — val dataset virtual size is 0 "
                "(e.g. text-prompted val has no prompt-bearing cases)."
            )

        for epoch in range(self.epoch, self.config.training.max_epochs):
            self.epoch = epoch
            is_best = False
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch()
            self.metrics_history['train_loss'].append(train_metrics.get('loss', 0.0))
            lr = self.optimizer.param_groups[-1]['lr']
            self.metrics_history['learning_rate'].append(lr)

            # Check for NaN abort
            if self._nan_abort:
                print(f"\nTraining aborted at epoch {epoch} due to persistent NaN loss.")
                break

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Validate (skip when train_on_all)
            val_metrics = {}
            if not self.train_on_all and (epoch + 1) % self.config.training.val_check_interval == 0:
                val_metrics = self.validate_epoch()

                # Record validation metrics
                self.metrics_history['val_loss'].append(val_metrics['loss'])
                self.metrics_history['val_dice'].append(val_metrics.get('dice_mean', 0.0))

                current_metric = val_metrics.get('dice_mean', 0.0)
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
            elif self.train_on_all:
                # No validation; mark is_best periodically for checkpoint saving
                is_best = (epoch + 1) % self.config.training.val_check_interval == 0
            
            epoch_time = time.time() - epoch_start_time

            # Build log dict without the table
            log_dict = {
                'epoch': epoch,
                'train/learning_rate': lr,
                'train/epoch_time': epoch_time,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()}
            }
            
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
            if not self.train_on_all and len(val_metrics.keys()) > 0:
                if self.early_stopping(current_metric):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        print(f"Training completed for fold {fold_label}")
        if not self.train_on_all:
            print(f"Best validation Dice: {self.best_metric:.4f}")
        else:
            print(f"Completed {self.config.training.max_epochs} epochs (train_on_all, no validation)")
        
        # Create summary plot if visualization enabled
        if self.visualizer is not None:
            from visualization import create_summary_plot
            create_summary_plot(self.metrics_history, str(self.output_dir / "visualizations"))
        
        return self.best_metric


def run_single_fold(fold: int, config, enable_visualization: bool = False, resume_id: Optional[str] = None, warm_restart: bool = False, init_checkpoint: Optional[str] = None):
    """Run training for a single fold"""
    print(f"\n{'='*50}")
    print(f"Starting fold {fold}")
    print(f"{'='*50}")
    
    trainer = Trainer(config, fold, enable_visualization=enable_visualization, resume_id=resume_id, warm_restart=warm_restart, init_checkpoint=init_checkpoint)
    best_metric = trainer.train()
    
    wandb.finish()
    
    return best_metric


def run_cross_validation(config, enable_visualization: bool = False, resume_id: Optional[str] = None, warm_restart: bool = False, init_checkpoint: Optional[str] = None):
    """Run full 5-fold cross validation"""
    fold_results = []
    
    for fold in range(config.data.num_folds):
        best_metric = run_single_fold(fold, config, enable_visualization=enable_visualization, resume_id=resume_id, warm_restart=warm_restart, init_checkpoint=init_checkpoint)
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