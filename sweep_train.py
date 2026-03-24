"""
Training script for W&B sweep hyperparameter optimization
This script is designed to work with wandb sweep on a single fold.
"""
import os
import sys
import argparse
from pathlib import Path
import wandb

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config
from train import Trainer


def load_sweep_config(sweep_id):
    """
    Load configuration for a specific sweep.
    
    Loads the frozen config from JSON snapshot and applies it to the current
    config.py's Config class structure (for picklability).
    
    Args:
        sweep_id: W&B sweep ID (format: username/project/sweep_xxx)
    
    Returns:
        Config object with frozen values from snapshot
    """
    import json
    from copy import deepcopy
    
    # Extract sweep name from sweep ID
    sweep_name = sweep_id.split('/')[-1]
    snapshot_dir = Path(__file__).parent / "sweep_configs" / sweep_name
    json_snapshot_path = snapshot_dir / "config_snapshot.json"
    
    # Get current config structure (for class definitions)
    config = get_config()
    
    if json_snapshot_path.exists():
        print(f"✓ Loading frozen config from: {json_snapshot_path}")
        
        with open(json_snapshot_path, 'r') as f:
            snapshot_dict = json.load(f)
        
        # Apply snapshot values to config object recursively
        def apply_dict_to_config(config_obj, values_dict):
            for key, value in values_dict.items():
                if hasattr(config_obj, key):
                    attr = getattr(config_obj, key)
                    if isinstance(value, dict) and hasattr(attr, '__dict__'):
                        # Recursively apply to nested config objects
                        apply_dict_to_config(attr, value)
                    else:
                        # Set the value
                        setattr(config_obj, key, value)
        
        apply_dict_to_config(config, snapshot_dict)
        print("  ✓ Applied frozen configuration values")
        return config
    else:
        print(f"⚠ WARNING: No config snapshot found at {json_snapshot_path}")
        print("  Sweep may not have been created with ./start_sweep.sh")
        print("  Using current config.py - fixed parameters may be inconsistent!")
        return config


def train_with_sweep():
    """Training function for wandb sweep"""
    
    # Get sweep ID from environment (set by wandb agent)
    sweep_id = os.environ.get('WANDB_SWEEP_ID')
    if not sweep_id:
        print("⚠ WARNING: WANDB_SWEEP_ID not found in environment")
        print("  Loading config from current config.py")
        config = get_config()
    else:
        # Load configuration from sweep snapshot (ensures all agents use same fixed params)
        config = load_sweep_config(sweep_id)
    
    # Get fold and train_on_all from environment variables (set by sweep_agent.sbatch)
    fold = int(os.environ.get('SWEEP_FOLD', '0'))
    train_on_all = os.environ.get('TRAIN_ON_ALL', '0') == '1'
    if train_on_all:
        config.data.train_on_all = True
        fold = -1
        print("Train-on-all mode enabled via TRAIN_ON_ALL env var")
    
    # Parse command line arguments first (before wandb.init)
    # Use parse_known_args() to ignore sweep parameters passed by wandb agent
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (smaller dataset, fewer epochs)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization of training batches')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Initialize model weights from checkpoint path, but start training fresh')
    args, unknown = parser.parse_known_args()
    
    # Store init_checkpoint (CLI arg or env var)
    init_checkpoint = args.init_checkpoint or os.environ.get('INIT_CHECKPOINT', None) or None
    
    # Initialize wandb run manually (required to access wandb.config)
    # Note: The sweep agent will inject sweep parameters into this init
    # We initialize with a minimal config first, then update after applying sweep params
    fold_label = "all" if train_on_all else str(fold)
    sweep_tags = config.wandb.tags + [f"fold_{fold_label}", "sweep"]
    if train_on_all:
        sweep_tags.append("train_on_all")
    wandb.init(
        name=f"fold_{fold_label}",
        tags=sweep_tags,
    )
    
    # Override with sweep parameters (these are the ONLY parameters that vary)
    # Optimizer parameters
    if 'learning_rate' in wandb.config:
        config.training.initial_lr = wandb.config.learning_rate
    if 'weight_decay' in wandb.config:
        config.training.weight_decay = wandb.config.weight_decay
    if 'momentum' in wandb.config:
        config.training.momentum = wandb.config.momentum

    # Model parameters
    if 'deep_supervision' in wandb.config:
        config.model.deep_supervision = wandb.config.deep_supervision

    # Augmentation parameters
    if 'elastic_deform_prob' in wandb.config:
        config.augmentation.elastic_deform_prob = wandb.config.elastic_deform_prob

    # Weight map parameters
    if 'weight_map_scale' in wandb.config:
        config.training.weight_map_scale = wandb.config.weight_map_scale
    if 'weight_map_bias' in wandb.config:
        config.training.weight_map_bias = wandb.config.weight_map_bias

    # Topograph parameters
    if 'topograph_weight' in wandb.config:
        config.training.topograph_weight = wandb.config.topograph_weight
    if 'topograph_aggregation' in wandb.config:
        config.training.topograph_aggregation = wandb.config.topograph_aggregation
    if 'topograph_error_type' in wandb.config:
        config.training.topograph_error_type = wandb.config.topograph_error_type

    # Betti matching parameters
    if 'betti_weight' in wandb.config:
        config.training.betti_weight = wandb.config.betti_weight

    # DSC++ / Tversky parameters
    if 'gamma' in wandb.config:
        config.training.dice_plus_plus_gamma = wandb.config.gamma
    if 'exponential_correction' in wandb.config:
        config.training.exponential_correction = wandb.config.exponential_correction
    if 'tversky_beta' in wandb.config:
        config.training.tversky_beta = wandb.config.tversky_beta
    
    # Now update wandb config with the COMPLETE config (after sweep params are applied)
    # This ensures sweep parameters appear under config.training.<param>
    def config_to_dict(obj):
        """Convert config object to dict recursively"""
        if hasattr(obj, '__dict__'):
            return {k: config_to_dict(v) for k, v in obj.__dict__.items() 
                    if not k.startswith('_')}
        elif isinstance(obj, (list, tuple)):
            return [config_to_dict(item) for item in obj]
        else:
            return obj
    
    wandb.config.update(config_to_dict(config), allow_val_change=True)
    
    # Also log init_checkpoint if provided
    if init_checkpoint:
        wandb.config.update({'init_checkpoint': init_checkpoint}, allow_val_change=True)
    
    # Get wandb run id for directory naming
    run_id = wandb.run.id
    
    # Log ignored arguments (these are sweep parameters handled by wandb.config)
    if unknown:
        print(f"Note: Ignoring wandb sweep arguments (handled via wandb.config): {unknown}")
    
    # Debug mode adjustments
    if args.debug:
        config.training.max_epochs = 10
        config.training.val_check_interval = 2
        config.data.max_samples = 50
        print("Debug mode enabled: reduced epochs and samples")
    
    # Enable visualization if requested
    enable_visualization = args.visualize or args.debug
    
    # Update output directory to use sweeps subdirectory
    config.output_dir = str(Path(config.output_dir) / "sweeps")
    
    # Print configuration for this sweep run
    print("\n" + "="*80)
    print("Sweep Training Configuration:")
    print("="*80)
    print(f"Fold: {fold_label}")
    if train_on_all:
        print(f"Train on ALL data: ENABLED (no validation)")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {config.output_dir}/fold_{fold}_{run_id}")
    if init_checkpoint:
        print(f"Init checkpoint: {init_checkpoint}")
    print("\n" + "-"*80)
    print("SWEEP PARAMETERS (being optimized):")
    print("-"*80)
    sweep_params = wandb.config.keys()
    for param in sorted(sweep_params):
        # Map sweep param names to config locations
        param_map = {
            'learning_rate': ('training.initial_lr', config.training.initial_lr),
            'weight_decay': ('training.weight_decay', config.training.weight_decay),
            'deep_supervision': ('model.deep_supervision', config.model.deep_supervision),
            'weight_map_scale': ('training.weight_map_scale', config.training.weight_map_scale),
            'elastic_deform_prob': ('augmentation.elastic_deform_prob', config.augmentation.elastic_deform_prob),
            'topograph_weight': ('training.topograph_weight', config.training.topograph_weight),
            'topograph_aggregation': ('training.topograph_aggregation', config.training.topograph_aggregation),
            'topograph_error_type': ('training.topograph_error_type', config.training.topograph_error_type),
            'betti_weight': ('training.betti_weight', config.training.betti_weight),
            'momentum': ('training.momentum', config.training.momentum),
        }
        if param in param_map:
            name, value = param_map[param]
            print(f"  {param:<25s} {value}")
        else:
            print(f"  {param:<25s} {getattr(wandb.config, param, '?')}")
    print("\n" + "-"*80)
    print("FIXED PARAMETERS (frozen from config.py):")
    print("-"*80)
    print(f"  Model:            {config.model.architecture} ({config.model.model_size})")
    print(f"  Batch size:       {config.training.batch_size}")
    print(f"  Max epochs:       {config.training.max_epochs}")
    print(f"  LR scheduler:     {config.training.lr_scheduler} (power={config.training.poly_lr_pow})")
    print(f"  Loss function:    {config.training.loss_function}")
    print(f"  Patch size:       {config.data.patch_size}")
    print(f"  Weight map bias:  {config.training.weight_map_bias}")
    print(f"  Seed:             {config.seed + fold}")
    print("="*80 + "\n")
    
    # Create trainer and run training
    # Note: Don't pass resume_id for new sweep runs - the Trainer will automatically
    # pick up the run_id from the already-initialized wandb.run
    trainer = Trainer(
        config, 
        fold=fold,
        enable_visualization=enable_visualization,
        resume_id=None,  # New run, not resuming
        init_checkpoint=init_checkpoint  # Initialize from checkpoint if provided
    )
    
    best_metric = trainer.train()

    if train_on_all:
        # No validation metric; report final training loss for sweep
        final_train_loss = trainer.metrics_history['train_loss'][-1] if trainer.metrics_history['train_loss'] else float('inf')
        wandb.run.summary['final_train_loss'] = final_train_loss
        print(f"\nSweep run completed (train_on_all). Final train loss: {final_train_loss:.4f}")
    else:
        print(f"\nSweep run completed. Best validation Dice: {best_metric:.4f}")
        wandb.run.summary['best_val_dice'] = best_metric

    return best_metric


if __name__ == "__main__":
    train_with_sweep()
