"""
Script to resume a sweep run that was interrupted.

This script:
1. Loads the frozen sweep configuration from the config snapshot
2. Resumes the existing W&B run (same run_id)
3. Loads the sweep parameters from W&B
4. Resumes training from the last checkpoint
5. Continues logging to the same W&B run

This is different from continue_sweep_run.py which uses warm restart.
This script simply resumes interrupted training without any modifications.

Usage:
    python resume_sweep_run.py --run_id <wandb_run_id> --fold <fold_number> --sweep_id <sweep_id>

Example:
    python resume_sweep_run.py --run_id abc123def --fold 0 --sweep_id bergeral/softmax_grad/sweep_xyz
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
        print("  Using current config.py - fixed parameters may be inconsistent!")
        return config


def resume_sweep_run(run_id: str, fold: int, sweep_id: str, visualize: bool = False):
    """
    Resume training a sweep run from checkpoint.
    
    Args:
        run_id: W&B run ID to resume
        fold: Fold number
        sweep_id: W&B sweep ID (format: entity/project/sweep_xxx)
        visualize: Enable visualization
    """
    print(f"\n{'='*80}")
    print(f"RESUMING SWEEP RUN")
    print(f"{'='*80}")
    print(f"Run ID: {run_id}")
    print(f"Fold: {fold}")
    print(f"Sweep ID: {sweep_id}")
    print(f"{'='*80}\n")
    
    # Load configuration from sweep snapshot (ensures same fixed params as original run)
    config = load_sweep_config(sweep_id)
    
    # Update output directory to use sweeps subdirectory (same as sweep_train.py)
    config.output_dir = str(Path(config.output_dir) / "sweeps")
    
    # Verify checkpoint exists
    checkpoint_path = Path(config.output_dir) / f"fold_{fold}_{run_id}" / "checkpoint.pth"
    if not checkpoint_path.exists():
        print(f"✗ Error: Checkpoint not found at {checkpoint_path}")
        print(f"Expected structure: {config.output_dir}/fold_{{fold}}_{{run_id}}/checkpoint.pth")
        sys.exit(1)
    
    print(f"✓ Checkpoint found: {checkpoint_path}")
    
    # Extract entity and project from sweep_id
    parts = sweep_id.split('/')
    if len(parts) == 3:
        entity, project, _ = parts
    else:
        # Fallback to defaults
        entity = "aimt"
        project = "softmax_grad"
        print(f"⚠ Warning: Could not parse sweep_id, using defaults: {entity}/{project}")
    
    # Initialize W&B with resume mode (this will restore the run state)
    print(f"\nInitializing W&B with resume mode...")
    print(f"  Entity: {entity}")
    print(f"  Project: {project}")
    print(f"  Run ID: {run_id}")
    
    wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        name=f"fold_{fold}",
        tags=config.wandb.tags + [f"fold_{fold}", "sweep"],
        config=config.__dict__
    )
    
    print(f"✓ W&B initialized - resuming run {run_id}")
    
    # Apply sweep parameters from wandb.config (these were saved when the run was created)
    config.training.initial_lr = wandb.config.learning_rate
    config.training.weight_decay = wandb.config.weight_decay
    config.training.momentum = wandb.config.momentum
    
    # Check for optional sweep parameters
    if 'gamma' in wandb.config:
        config.training.dice_plus_plus_gamma = wandb.config.gamma
    
    if 'exponential_correction' in wandb.config:
        config.training.exponential_correction = wandb.config.exponential_correction

    if 'tversky_beta' in wandb.config:
        config.training.tversky_beta = wandb.config.tversky_beta
    
    # Print configuration being used
    print(f"\n{'='*80}")
    print("Resume Configuration:")
    print(f"{'='*80}")
    print(f"Output directory: {config.output_dir}/fold_{fold}_{run_id}")
    print(f"\nSweep parameters:")
    print(f"  Learning rate:    {config.training.initial_lr}")
    print(f"  Weight decay:     {config.training.weight_decay}")
    print(f"  Momentum:         {config.training.momentum}")
    if hasattr(config.training, 'exponential_correction') and config.training.exponential_correction is not None:
        print(f"  Exponential correction: {config.training.exponential_correction}")
    if hasattr(config.training, 'dice_plus_plus_gamma'):
        print(f"  DSC++ gamma:      {config.training.dice_plus_plus_gamma}")
    if hasattr(config.training, 'tversky_beta'):
        print(f"  Tversky beta:     {config.training.tversky_beta}")
    print(f"\nFixed parameters:")
    print(f"  Model:            {config.model.architecture} ({config.model.model_size})")
    print(f"  Batch size:       {config.training.batch_size}")
    print(f"  Max epochs:       {config.training.max_epochs}")
    print(f"  Loss function:    {config.training.loss_function}")
    print(f"{'='*80}\n")
    
    # Create trainer with resume_id (no warm_restart flag)
    print(f"Creating trainer and resuming training...")
    trainer = Trainer(
        config,
        fold=fold,
        enable_visualization=visualize,
        resume_id=run_id,
        warm_restart=False  # Standard resume, not warm restart
    )
    
    # Resume training
    print(f"\nResuming training from checkpoint...")
    best_metric = trainer.train()
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best validation Dice: {best_metric:.4f}")
    print(f"{'='*80}\n")
    
    wandb.finish()
    
    return best_metric


def main():
    parser = argparse.ArgumentParser(
        description='Resume an interrupted sweep run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resume a sweep run
  python resume_sweep_run.py --run_id abc123def --fold 0 --sweep_id bergeral/softmax_grad/sweep_xyz
  
  # Resume with visualization enabled
  python resume_sweep_run.py --run_id abc123def --fold 0 --sweep_id bergeral/softmax_grad/sweep_xyz --visualize

Note:
  This script is for resuming interrupted runs. It does NOT use warm restart.
  For extending training with warm restart, use continue_sweep_run.py instead.
        """
    )
    
    parser.add_argument('--run_id', type=str, required=True,
                        help='W&B run ID to resume (e.g., abc123def)')
    parser.add_argument('--fold', type=int, required=True,
                        help='Fold number (0-4)')
    parser.add_argument('--sweep_id', type=str, required=True,
                        help='W&B sweep ID (format: entity/project/sweep_xxx)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization of training batches')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.fold < 0 or args.fold > 4:
        parser.error("Fold must be between 0 and 4")
    
    # Run resume
    try:
        best_metric = resume_sweep_run(
            run_id=args.run_id,
            fold=args.fold,
            sweep_id=args.sweep_id,
            visualize=args.visualize
        )
        print(f"\n✓ Successfully resumed and completed run {args.run_id}")
        print(f"  Best metric: {best_metric:.4f}")
    except Exception as e:
        print(f"\n✗ Error resuming run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
