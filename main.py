"""
Main entry point for 3D segmentation training
"""
import os
import sys
import argparse
import importlib.util
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


def load_config_from_path(config_path: str):
    """Load config module from a specific file path."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules['config'] = config_module
    spec.loader.exec_module(config_module)
    return config_module.get_config()


def main():
    # Pre-parse to check for config_path before importing config
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_path', type=str, default=None,
                            help='Path to config.py file (for reproducible runs with config snapshots)')
    pre_args, _ = pre_parser.parse_known_args()
    
    # Load config from specified path or default
    if pre_args.config_path:
        print(f"Loading config from snapshot: {pre_args.config_path}")
        config = load_config_from_path(pre_args.config_path)
    else:
        from config import get_config
        config = get_config()
    
    # Now import train (which may depend on config module being loaded)
    from train import run_cross_validation, run_single_fold
    
    parser = argparse.ArgumentParser(description='3D Segmentation Training')
    
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config.py file (for reproducible runs with config snapshots)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to train (0-4). If not specified, runs all folds.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs from config')
    parser.add_argument('--model_size', type=str, default=None, choices=['S', 'B', 'M', 'L'],
                        help='Override Model size: S=Small, B=Base, M=Medium, L=Large')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Override wandb project name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (smaller dataset, fewer epochs)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization of training batches and outputs (useful for debugging NaN issues)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples per split (for debugging)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint by providing the run_id')
    parser.add_argument('--use_new_config', action='store_true',
                        help='Whether to not use the config from checkpoint but the one from config.py (requires --resume)')
    parser.add_argument('--warm_restart', action='store_true',
                        help='Enable warm restart mode: continues training with adjusted learning rate (requires --resume and --use_new_config)')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Initialize model weights from checkpoint path, but start training fresh (new run, epoch 0)')
    parser.add_argument('--train_on_all', action='store_true',
                        help='Train on all data (train + val combined, no validation). Ignores --fold.')

    args = parser.parse_args()
    
    # Validate config loading
    if args.use_new_config and args.resume is None:
        parser.error("--use_new_config requires --resume")
    
    if args.warm_restart and args.resume is None:
        parser.error("--warm_restart requires --resume")
    
    if args.init_checkpoint and args.resume:
        parser.error("--init_checkpoint and --resume are mutually exclusive")

    if args.train_on_all and args.fold is not None:
        print("WARNING: --train_on_all specified, ignoring --fold argument")

    # Config is already loaded at the top (from snapshot or default)
    # Just set the additional flags
    config._use_new_config = args.use_new_config
    config._warm_restart = args.warm_restart
    
    # Store config path for logging/reproducibility
    config._config_path = pre_args.config_path
    
    # For normal resume (without use_new_config), don't apply any overrides
    # The config will be loaded entirely from the checkpoint
    if args.resume and not args.use_new_config:
        print("Normal resume mode: config will be loaded from checkpoint (ignoring command-line overrides)")
    else:
        # For new runs or warm restart, apply command line overrides
        # Note: For warm restart, only --epochs should typically be used
        if args.batch_size is not None:
            config.training.batch_size = args.batch_size

        if args.model_size is not None:
            config.model.model_size = args.model_size
        
        if args.lr is not None:
            config.training.initial_lr = args.lr
        
        if args.epochs is not None:
            config.training.max_epochs = args.epochs
        
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        
        if args.wandb_project is not None:
            config.wandb.project = args.wandb_project
        
        if args.seed is not None:
            config.seed = args.seed
        
        if args.max_samples is not None:
            config.data.max_samples = args.max_samples
            print(f"Max samples mode enabled: limiting to {args.max_samples} samples per split")

    # train_on_all is always applied (even on resume — data split is not stored in checkpoint)
    if args.train_on_all:
        config.data.train_on_all = True

    # Debug mode adjustments
    if args.debug:
        config.training.max_epochs = 10
        config.training.val_check_interval = 2
        print("Debug mode enabled: reduced epochs and val interval")
    
    # Enable visualization if requested (or automatically in debug mode)
    enable_visualization = args.visualize or args.debug
    if enable_visualization:
        print("Visualization enabled: saving batch images and outputs for debugging")
    
    # Determine run_id for organizing experiments
    if args.resume:
        # Resuming an existing run - use provided run_id
        run_id = args.resume
        print(f"Resuming run with ID: {run_id}")
    elif args.no_wandb:
        # Offline mode - use fixed run_id
        run_id = "offline_run"
        print("Wandb logging disabled - using offline_run as run_id")
        import wandb
        wandb.init = lambda **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.finish = lambda: None
    else:
        # New run - will be set after wandb.init
        run_id = None
        # Set up WandB if API key is available
        if 'WANDB_API_KEY' in os.environ:
            print(f"WandB enabled - Project: {config.wandb.project}")
        else:
            print("Warning: WANDB_API_KEY not found in environment")
    
    # Store run_id in config for use in training
    config.run_id = run_id
    
    # Print configuration
    print("\n" + "="*80)
    if args.use_new_config:
        print("Will use configuration from config.py")
    else:
        print("Using configuration from checkpoint")
    print("="*80)
    print("Training Configuration:")
    print(f"  Model size: {config.model.model_size}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.initial_lr}")
    print(f"  Max epochs: {config.training.max_epochs}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Random seed: {config.seed}")
    if config.data.train_on_all:
        print(f"  Train on ALL data: ENABLED (no validation)")
    if config.data.max_samples is not None:
        print(f"  WARNING: Max samples per split: {config.data.max_samples}")
    print("="*80 + "\n")

    # Run training
    if args.train_on_all:
        # Train on all data (no validation)
        print("Training on ALL data (no validation)")
        best_metric = run_single_fold(
            -1,  # sentinel fold value
            config,
            enable_visualization=enable_visualization,
            resume_id=args.resume,
            warm_restart=args.warm_restart,
            init_checkpoint=args.init_checkpoint
        )
        print(f"Training completed.")
    elif args.fold is not None:
        # Single fold training
        print(f"Training single fold: {args.fold}")
        best_metric = run_single_fold(
            args.fold,
            config,
            enable_visualization=enable_visualization,
            resume_id=args.resume,
            warm_restart=args.warm_restart,
            init_checkpoint=args.init_checkpoint
        )
        print(f"Training completed. Best metric: {best_metric:.4f}")
    else:
        # Cross-validation
        print("Running 5-fold cross-validation")
        results = run_cross_validation(
            config,
            enable_visualization=enable_visualization,
            resume_id=args.resume,
            warm_restart=args.warm_restart,
            init_checkpoint=args.init_checkpoint
        )
        print(f"Cross-validation completed. Mean metric: {results['mean']:.4f} ± {results['std']:.4f}")


if __name__ == "__main__":
    main()