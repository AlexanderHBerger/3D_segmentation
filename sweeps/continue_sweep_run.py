"""
Script to continue a completed sweep run with warm restart for extended training.

This script:
1. Loads the run configuration from W&B (including sweep parameters)
2. Resumes from the last checkpoint in the fold directory
3. Applies warm restart to the learning rate (continues with reduced LR)
4. Trains for additional epochs as specified in config
5. Continues logging to the same W&B run

Usage:
    python continue_sweep_run.py --run_id <wandb_run_id> --fold <fold_number> [--total_epochs 400]

Example:
    python continue_sweep_run.py --run_id abc123def --fold 0 --total_epochs 400
"""
import os
import sys
import argparse
from pathlib import Path
import wandb
import json

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config
from train import Trainer


def load_config_from_wandb_run(run_path: str):
    """
    Load configuration from a W&B run.
    
    Args:
        run_path: Full W&B run path (format: entity/project/run_id)
    
    Returns:
        Config object with values from the W&B run
    """
    print(f"\n{'='*80}")
    print(f"Loading configuration from W&B run: {run_path}")
    print(f"{'='*80}")
    
    # Initialize W&B API
    api = wandb.Api()
    
    # Get the run
    # Note: There's a known issue with W&B API where loading runs that are part of sweeps
    # can fail with "Object of type Api is not JSON serializable". We work around this by
    # accessing the run through a different method if needed.
    try:
        run = api.run(run_path)
    except Exception as e:
        if "not JSON serializable" in str(e):
            print(f"Warning: Hit W&B serialization bug when loading sweep run")
            print(f"Attempting workaround...")
            # Try to use the internal client to get run data without sweep info
            import requests
            entity, project, run_id_only = run_path.split('/')
            
            # Get run config directly via API
            headers = {"Authorization": f"Bearer {api.api_key}"}
            url = f"https://api.wandb.ai/graphql"
            
            query = """
            query Run($entity: String!, $project: String!, $name: String!) {
                project(entityName: $entity, name: $project) {
                    run(name: $name) {
                        name
                        state
                        createdAt
                        config
                    }
                }
            }
            """
            
            variables = {
                "entity": entity,
                "project": project,
                "name": run_id_only
            }
            
            response = requests.post(
                url,
                json={"query": query, "variables": variables},
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"Error: Failed to fetch run data: {response.text}")
                sys.exit(1)
            
            data = response.json()
            if "errors" in data:
                print(f"Error: GraphQL errors: {data['errors']}")
                sys.exit(1)
            
            run_data = data["data"]["project"]["run"]
            if not run_data:
                print(f"Error: Run not found at {run_path}")
                sys.exit(1)
            
            # Create a simple object to hold the run data
            class SimpleRun:
                def __init__(self, data):
                    self.name = data["name"]
                    self.state = data["state"]
                    self.created_at = data["createdAt"]
                    self.config = json.loads(data["config"]) if isinstance(data["config"], str) else data["config"]
            
            run = SimpleRun(run_data)
            print(f"✓ Successfully loaded run using workaround")
        else:
            raise
    except Exception as e:
        print(f"Error: Could not load run from W&B: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"✓ Run found: {run.name}")
    print(f"  State: {run.state}")
    print(f"  Created: {run.created_at}")
    
    # Get the base config structure
    config = get_config()
    
    # Load config from W&B run and convert to plain dict
    wandb_config = dict(run.config)
    
    def apply_dict_to_config(config_obj, values_dict, prefix=""):
        """Recursively apply dictionary values to config object"""
        for key, value in values_dict.items():
            # W&B sometimes wraps values in {'value': actual_value} dicts
            if isinstance(value, dict) and 'value' in value and len(value) == 1:
                value = value['value']
            
            if hasattr(config_obj, key):
                attr = getattr(config_obj, key)
                if isinstance(value, dict) and hasattr(attr, '__dict__'):
                    # Recursively apply to nested config objects
                    apply_dict_to_config(attr, value, prefix=f"{prefix}{key}.")
                else:
                    # Set the value
                    setattr(config_obj, key, value)
                    if prefix == "":  # Only log top-level for readability
                        print(f"  ✓ Set {key} = {value}")
    
    # Apply W&B config to our config object
    print(f"\nApplying configuration from W&B run...")
    apply_dict_to_config(config, wandb_config)
    
    # Helper to extract value from W&B config (which may be wrapped in {'value': x})
    def get_wandb_value(config_dict, key):
        if key not in config_dict:
            return None
        val = config_dict[key]
        if isinstance(val, dict) and 'value' in val and len(val) == 1:
            return val['value']
        return val
    
    # Extract sweep parameters if this was a sweep run
    print(f"\nSweep parameters from W&B run:")
    if 'learning_rate' in wandb_config:
        lr = get_wandb_value(wandb_config, 'learning_rate')
        print(f"  Learning rate: {lr}")
        config.training.initial_lr = lr
    if 'weight_decay' in wandb_config:
        wd = get_wandb_value(wandb_config, 'weight_decay')
        print(f"  Weight decay: {wd}")
        config.training.weight_decay = wd
    if 'momentum' in wandb_config:
        mom = get_wandb_value(wandb_config, 'momentum')
        print(f"  Momentum: {mom}")
        config.training.momentum = mom
    if 'gamma' in wandb_config:
        gamma = get_wandb_value(wandb_config, 'gamma')
        print(f"  DSC++ gamma: {gamma}")
        config.training.dice_plus_plus_gamma = gamma
    if 'exponential_correction' in wandb_config:
        ec = get_wandb_value(wandb_config, 'exponential_correction')
        print(f"  Exponential correction: {ec}")
        config.training.exponential_correction = ec
    if 'tversky_beta' in wandb_config:
        tb = get_wandb_value(wandb_config, 'tversky_beta')
        print(f"  Tversky beta: {tb}")
        config.training.tversky_beta = tb
    
    # Ensure output_dir includes "sweeps" subdirectory for sweep runs
    # (sweep_train.py adds this, but W&B saves config before that modification)
    if not config.output_dir.endswith('sweeps'):
        config.output_dir = str(Path(config.output_dir) / "sweeps")
        print(f"\n✓ Added 'sweeps' subdirectory to output_dir: {config.output_dir}")
    
    print(f"{'='*80}\n")
    
    return config


def continue_sweep_run(run_id: str, fold: int, total_epochs: int = 400, visualize: bool = False, dry_run: bool = False):
    """
    Continue training a sweep run with warm restart.
    
    Args:
        run_id: W&B run ID to continue
        fold: Fold number
        total_epochs: Total number of epochs to train to (not additional epochs)
        visualize: Enable visualization
        dry_run: If True, only validate setup without running training
    """
    # Construct full run path
    # First try to get from environment, otherwise use default
    entity = os.environ.get('WANDB_ENTITY')
    project = os.environ.get('WANDB_PROJECT', 'softmax_grad')
    
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        # Try to infer from existing run
        run_path = f"{project}/{run_id}"
    
    # Load configuration from W&B
    config = load_config_from_wandb_run(run_path)
    
    # Update max_epochs for extended training
    original_max_epochs = config.training.max_epochs
    config.training.max_epochs = total_epochs
    extend_epochs = total_epochs - original_max_epochs
    
    # Set flag to use new config instead of loading from checkpoint
    # This is critical for warm restart to preserve the updated max_epochs
    config._use_new_config = True
    
    print(f"\n{'='*80}")
    print(f"CONTINUING SWEEP RUN WITH WARM RESTART")
    print(f"{'='*80}")
    print(f"Run ID: {run_id}")
    print(f"Fold: {fold}")
    print(f"Original max epochs: {original_max_epochs}")
    print(f"Target total epochs: {total_epochs}")
    print(f"Extension: {extend_epochs} epochs")
    print(f"Warm restart LR factor: {config.training.warm_restart_lr_factor}")
    print(f"Output directory: {config.output_dir}/fold_{fold}_{run_id}")
    print(f"{'='*80}\n")
    
    # Verify checkpoint exists
    checkpoint_path = Path(config.output_dir) / f"fold_{fold}_{run_id}" / "checkpoint.pth"
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Expected structure: {config.output_dir}/fold_{{fold}}_{{run_id}}/checkpoint.pth")
        sys.exit(1)
    
    print(f"✓ Checkpoint found: {checkpoint_path}")
    
    # If dry run, validate and exit
    if dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN MODE - VALIDATION ONLY")
        print(f"{'='*80}")
        print("✓ Run configuration loaded successfully from W&B")
        print("✓ Checkpoint exists and is accessible")
        print("✓ Output directory structure is correct")
        print("\nConfiguration that would be used:")
        print(f"  Initial LR: {config.training.initial_lr}")
        print(f"  Warm restart LR: {config.training.initial_lr * config.training.warm_restart_lr_factor:.6f}")
        print(f"  Weight decay: {config.training.weight_decay}")
        print(f"  Momentum: {config.training.momentum}")
        print(f"  Original epochs: {original_max_epochs}")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Extension: {extend_epochs} epochs")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Model: {config.model.architecture} ({config.model.model_size})")
        print(f"  Loss function: {config.training.loss_function}")
        
        # Load checkpoint to verify it's valid
        print(f"\nValidating checkpoint contents...")
        import torch
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print(f"  ✓ Checkpoint loaded successfully")
            print(f"  ✓ Last completed epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  ✓ Best metric: {checkpoint.get('best_metric', 'unknown')}")
            print(f"  ✓ Checkpoint fold: {checkpoint.get('fold', 'unknown')}")
            
            if checkpoint.get('fold') != fold:
                print(f"  ✗ WARNING: Checkpoint fold ({checkpoint.get('fold')}) != requested fold ({fold})")
                return None
        except Exception as e:
            print(f"  ✗ ERROR loading checkpoint: {e}")
            return None
        
        print(f"\n{'='*80}")
        print("DRY RUN COMPLETE - All validations passed!")
        print("You can now run without --dry_run to continue training.")
        print(f"{'='*80}\n")
        return None
    
    # Initialize W&B with resume
    print(f"\nInitializing W&B with resume mode...")
    wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="must",
        name=f"fold_{fold}",
        tags=config.wandb.tags + [f"fold_{fold}", "sweep", "warm_restart"],
        config=config.__dict__
    )
    
    print(f"✓ W&B initialized - continuing run {run_id}")
    
    # Log warm restart information
    wandb.run.summary['warm_restart'] = True
    wandb.run.summary['original_max_epochs'] = original_max_epochs
    wandb.run.summary['extended_epochs'] = extend_epochs
    wandb.run.summary['warm_restart_lr_factor'] = config.training.warm_restart_lr_factor
    
    # Create trainer with warm_restart flag
    print(f"\nCreating trainer with warm restart...")
    trainer = Trainer(
        config,
        fold=fold,
        enable_visualization=visualize,
        resume_id=run_id,
        warm_restart=True  # This triggers the warm restart logic
    )
    
    # Train for extended epochs
    print(f"\nStarting extended training...")
    best_metric = trainer.train()
    
    print(f"\n{'='*80}")
    print(f"Extended training completed!")
    print(f"Best validation Dice: {best_metric:.4f}")
    print(f"{'='*80}\n")
    
    # Update final summary
    wandb.run.summary['best_val_dice_after_restart'] = best_metric
    
    wandb.finish()
    
    return best_metric


def main():
    parser = argparse.ArgumentParser(
        description='Continue sweep run with warm restart for extended training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Continue a sweep run to 400 total epochs
  python continue_sweep_run.py --run_id abc123def --fold 0 --total_epochs 400
  
  # Continue with visualization enabled
  python continue_sweep_run.py --run_id abc123def --fold 0 --total_epochs 400 --visualize
  
  # Continue with custom warm restart factor (override config)
  python continue_sweep_run.py --run_id abc123def --fold 0 --total_epochs 400 --warm_restart_factor 0.5

Environment variables:
  WANDB_ENTITY: Your W&B username/team (optional, will try to infer)
  WANDB_PROJECT: W&B project name (default: "Metastases Segmentation")
        """
    )
    
    parser.add_argument('--run_id', type=str, required=True,
                        help='W&B run ID to continue (e.g., abc123def)')
    parser.add_argument('--fold', type=int, required=True,
                        help='Fold number (0-4)')
    parser.add_argument('--total_epochs', type=int, default=400,
                        help='Total number of epochs to train to (default: 400)')
    parser.add_argument('--warm_restart_factor', type=float, default=None,
                        help='Override warm restart LR factor (default: use value from config, typically 0.66)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization of training batches')
    parser.add_argument('--dry_run', action='store_true',
                        help='Validate setup without running training (recommended first)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.fold < 0 or args.fold > 4:
        parser.error("Fold must be between 0 and 4")
    
    if args.total_epochs <= 0:
        parser.error("total_epochs must be positive")
    
    if args.warm_restart_factor is not None:
        if args.warm_restart_factor <= 0 or args.warm_restart_factor > 1:
            parser.error("warm_restart_factor must be between 0 and 1")
    
    # Run continuation
    try:
        best_metric = continue_sweep_run(
            run_id=args.run_id,
            fold=args.fold,
            total_epochs=args.total_epochs,
            visualize=args.visualize,
            dry_run=args.dry_run
        )
        if args.dry_run:
            print(f"\n✓ Dry run completed successfully for run {args.run_id}")
        elif best_metric is not None:
            print(f"\n✓ Successfully continued run {args.run_id}")
            print(f"  Best metric: {best_metric:.4f}")
    except Exception as e:
        print(f"\n✗ Error continuing run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
