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
    
    # Get fold from environment variable (set by sweep_agent.sbatch)
    fold = int(os.environ.get('SWEEP_FOLD', '0'))
    
    # Initialize wandb run manually (required to access wandb.config)
    # Note: The sweep agent will inject sweep parameters into this init
    # We log the entire config structure for complete transparency
    # Don't specify project - it's inherited from the sweep
    wandb.init(
        name=f"fold_{fold}",
        tags=config.wandb.tags + [f"fold_{fold}", "sweep"],
        config=config.__dict__
    )
    
    # Override with sweep parameters (these are the ONLY parameters that vary)
    # These will overwrite the values in the config that was logged above
    config.training.initial_lr = wandb.config.learning_rate
    config.training.weight_decay = wandb.config.weight_decay
    config.training.momentum = wandb.config.momentum
    
    # Check for gamma parameter (for DSC++ loss)
    if 'gamma' in wandb.config:
        config.training.dice_plus_plus_gamma = wandb.config.gamma
    
    # Check for exponential_correction parameter (for FixedGradSoftmax)
    if 'exponential_correction' in wandb.config:
        config.training.exponential_correction = wandb.config.exponential_correction

    # Check for tversky_beta parameter
    if 'tversky_beta' in wandb.config:
        config.training.tversky_beta = wandb.config.tversky_beta
    
    # Get wandb run id for directory naming
    run_id = wandb.run.id
    
    # Parse remaining command line arguments
    # Use parse_known_args() to ignore sweep parameters passed by wandb agent
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (smaller dataset, fewer epochs)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization of training batches')
    args, unknown = parser.parse_known_args()
    
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
    print(f"Fold: {fold}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {config.output_dir}/fold_{fold}_{run_id}")
    print("\n" + "-"*80)
    print("SWEEP PARAMETERS (being optimized):")
    print("-"*80)
    print(f"  Learning rate:           {config.training.initial_lr}")
    print(f"  Weight decay:            {config.training.weight_decay}")
    print(f"  Momentum:                {config.training.momentum}")
    if hasattr(config.training, 'exponential_correction') and config.training.exponential_correction is not None:
        print(f"  Exponential correction:  {config.training.exponential_correction}")
    if hasattr(config.training, 'dice_plus_plus_gamma'):
        print(f"  DSC++ gamma:             {config.training.dice_plus_plus_gamma}")
    if hasattr(config.training, 'tversky_beta'):
        print(f"  Tversky beta:            {config.training.tversky_beta}")
    print("\n" + "-"*80)
    print("FIXED PARAMETERS (frozen from config.py):")
    print("-"*80)
    print(f"  Model:            {config.model.architecture} ({config.model.model_size})")
    print(f"  Batch size:       {config.training.batch_size}")
    print(f"  Deep supervision: {config.model.deep_supervision}")
    print(f"  Max epochs:       {config.training.max_epochs}")
    print(f"  LR scheduler:     {config.training.lr_scheduler} (power={config.training.poly_lr_pow})")
    print(f"  Loss function:    {config.training.loss_function}")
    print(f"  Patch size:       {config.data.patch_size}")
    print(f"  Normalization:    {config.data.normalization_scheme}")
    print(f"  Mixed precision:  {config.mixed_precision}")
    print(f"  Seed:             {config.seed + fold}")
    print("="*80 + "\n")
    
    # Create trainer and run training
    # Note: Don't pass resume_id for new sweep runs - the Trainer will automatically
    # pick up the run_id from the already-initialized wandb.run
    trainer = Trainer(
        config, 
        fold=fold,
        enable_visualization=enable_visualization,
        resume_id=None  # New run, not resuming
    )
    
    best_metric = trainer.train()
    
    print(f"\nSweep run completed. Best validation Dice: {best_metric:.4f}")
    
    # Log final summary
    wandb.run.summary['best_val_dice'] = best_metric
    
    return best_metric


if __name__ == "__main__":
    train_with_sweep()
