"""
Main entry point for MedNeXt segmentation training
"""
import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config
from train import run_cross_validation, run_single_fold


def main():
    parser = argparse.ArgumentParser(description='MedNeXt Brain Metastasis Segmentation Training')
    
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
    parser.add_argument('--brats_only', action='store_true',
                        help='Use only BraTS dataset subjects (for debugging)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples per split (for debugging)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint by providing the run_id')
    parser.add_argument('--use_new_config', action='store_true',
                        help='Whether to not use the config from checkpoint but the one from config.py (requires --resume)')
    
    args = parser.parse_args()
    
    # Validate config loading
    if args.use_new_config and args.resume is None:
        parser.error("--use_new_config requires --resume")

    # Get configuration - will be loaded from checkpoint in Trainer if flag is set
    config = get_config()
    config._use_new_config = args.use_new_config
    
    # Override config with command line arguments
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
    
    # Set debug options for data filtering
    if args.brats_only:
        config.data.brats_only = True
        print("BraTS-only mode enabled: using only BraTS dataset subjects")
    
    if args.max_samples is not None:
        config.data.max_samples = args.max_samples
        print(f"Max samples mode enabled: limiting to {args.max_samples} samples per split")
    
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
    if config.data.brats_only:
        print(f"  WARNING: BraTS-only mode: ENABLED")
    if config.data.max_samples is not None:
        print(f"  WARNING: Max samples per split: {config.data.max_samples}")
    print("="*80 + "\n")
    
    # Run training
    if args.fold is not None:
        # Single fold training
        print(f"Training single fold: {args.fold}")
        best_metric = run_single_fold(
            args.fold, 
            config, 
            enable_visualization=enable_visualization,
            resume_id=args.resume
        )
        print(f"Training completed. Best metric: {best_metric:.4f}")
    else:
        # Cross-validation
        print("Running 5-fold cross-validation")
        results = run_cross_validation(
            config, 
            enable_visualization=enable_visualization,
            resume_id=args.resume
        )
        print(f"Cross-validation completed. Mean metric: {results['mean']:.4f} ± {results['std']:.4f}")


if __name__ == "__main__":
    main()