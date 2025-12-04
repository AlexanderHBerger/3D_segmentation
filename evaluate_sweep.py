#!/usr/bin/env python3
"""
Evaluate sweep script.

Finds the best run from a W&B sweep based on val/dice_hard metric,
runs inference with save_logits and save_probabilities flags,
and evaluates predictions against ground truth.

Usage:
    python evaluate_sweep.py --sweep_id <sweep_id>
    
    # Example with full sweep ID:
    python evaluate_sweep.py --sweep_id username/Metastases%20Segmentation/abc123xyz
    
    # Or just the sweep name (assumes current project):
    python evaluate_sweep.py --sweep_id abc123xyz
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import wandb

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config, get_config
from inference import (
    Predictor,
    preprocess_case_for_inference,
    postprocess_prediction,
    create_predictor_wrapper,
    InferenceDataset,
    collate_fn
)
from evaluate import (
    find_matching_files,
    evaluate_sample_wrapper,
    evaluate_sample,
    aggregate_metrics
)
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference
from threading import Lock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_sweep_id(sweep_id: str) -> Tuple[str, str, str]:
    """
    Parse sweep ID into entity, project, and sweep name.
    
    Args:
        sweep_id: Can be in formats:
            - "entity/project/sweep_name"
            - "sweep_name" (uses default entity/project)
            
    Returns:
        Tuple of (entity, project, sweep_name)
    """
    parts = sweep_id.split('/')
    
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 1:
        # Use default entity/project from config
        config = get_config()
        entity = config.wandb.entity or wandb.api.default_entity
        project = config.wandb.project
        return entity, project, parts[0]
    else:
        raise ValueError(f"Invalid sweep_id format: {sweep_id}. "
                        f"Expected 'entity/project/sweep_name' or 'sweep_name'")


def get_best_run(sweep_id: str, metric: str = "val/dice_hard") -> Tuple[wandb.apis.public.Run, Dict[str, Any]]:
    """
    Get the best run from a sweep based on a metric.
    
    Args:
        sweep_id: W&B sweep ID
        metric: Metric to use for selecting best run (default: val/dice_hard)
        
    Returns:
        Tuple of (best_run, run_info_dict)
    """
    entity, project, sweep_name = parse_sweep_id(sweep_id)
    
    logger.info(f"Fetching sweep: {entity}/{project}/{sweep_name}")
    
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_name}")
    
    # Get all runs from the sweep
    runs = list(sweep.runs)
    
    if not runs:
        raise ValueError(f"No runs found in sweep {sweep_id}")
    
    logger.info(f"Found {len(runs)} runs in sweep")
    
    # Find the best run based on the metric
    best_run = None
    best_metric_value = float('-inf')
    
    for run in runs:
        # Try to get metric from summary (handle both dict and JSON string formats)
        try:
            summary_dict = run.summary._json_dict
            # If summary is a string, parse it as JSON
            if isinstance(summary_dict, str):
                summary_dict = json.loads(summary_dict)
            
            if metric in summary_dict:
                metric_value = summary_dict[metric]
                if metric_value is not None and metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_run = run
        except Exception as e:
            logger.debug(f"Could not get metric from summary for run {run.id}: {e}")
            continue
    
    if best_run is None:
        # Try to find in history if not in summary
        logger.warning(f"Metric '{metric}' not found in summary, searching history...")
        for run in runs:
            try:
                history = run.history(keys=[metric])
                if not history.empty and metric in history.columns:
                    max_val = history[metric].max()
                    if max_val > best_metric_value:
                        best_metric_value = max_val
                        best_run = run
            except Exception as e:
                logger.debug(f"Could not get metric from history for run {run.id}: {e}")
                continue
    
    if best_run is None:
        raise ValueError(f"Could not find metric '{metric}' in any run")
    
    # Parse config (may be a JSON string)
    run_config = best_run.config
    if isinstance(run_config, str):
        run_config = json.loads(run_config)
    
    # Collect run info
    run_info = {
        'run_id': best_run.id,
        'run_name': best_run.name,
        'sweep_id': sweep_name,
        'entity': entity,
        'project': project,
        f'best_{metric}': best_metric_value,
        'config': run_config
    }
    
    logger.info(f"Best run: {best_run.name} ({best_run.id})")
    logger.info(f"  {metric}: {best_metric_value:.4f}")
    
    return best_run, run_info


def flatten_wandb_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten W&B config that may have nested 'value' keys.
    
    W&B sweep configs often have format: {'key': {'value': actual_value}}
    This function extracts the actual values.
    
    Args:
        config_dict: Raw config dict from W&B
        
    Returns:
        Flattened config dict
    """
    flattened = {}
    for key, value in config_dict.items():
        # Skip internal W&B keys
        if key.startswith('_'):
            continue
        
        if isinstance(value, dict) and 'value' in value:
            # Extract the actual value
            flattened[key] = value['value']
        else:
            flattened[key] = value
    
    return flattened


def load_config_from_run(run: wandb.apis.public.Run) -> Config:
    """
    Load configuration from a W&B run.
    
    Args:
        run: W&B run object
        
    Returns:
        Config object with values from the run
    """
    config = get_config()
    
    # Parse config (may be a JSON string)
    run_config = run.config
    if isinstance(run_config, str):
        run_config = json.loads(run_config)
    
    # Flatten the nested 'value' structure
    run_config = flatten_wandb_config(run_config)
    
    # Apply run config values recursively
    def apply_dict_to_config(config_obj, values_dict):
        for key, value in values_dict.items():
            if hasattr(config_obj, key):
                attr = getattr(config_obj, key)
                if isinstance(value, dict) and hasattr(attr, '__dict__'):
                    apply_dict_to_config(attr, value)
                else:
                    setattr(config_obj, key, value)
    
    apply_dict_to_config(config, run_config)
    
    return config


def find_checkpoint(run_dir: Path) -> Path:
    """
    Find the best checkpoint in a run directory.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Path to the best checkpoint
    """
    # # Try best_model.pth first
    # best_model = run_dir / "best_model.pth"
    # if best_model.exists():
    #     return best_model
    
    # Fall back to checkpoint.pth
    checkpoint = run_dir / "checkpoint.pth"
    if checkpoint.exists():
        return checkpoint
    
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def run_inference(
    checkpoint_path: Path,
    input_folder: Path,
    output_folder: Path,
    config: Config,
    device: str = 'cuda',
    num_workers: int = 6,
    num_postprocess_workers: int = 6
) -> None:
    """
    Run inference using the Predictor class.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_folder: Path to input images
        output_folder: Path to save predictions
        config: Configuration object
        device: Device to run on
        num_workers: Number of data loading workers
        num_postprocess_workers: Number of post-processing workers
    """
    logger.info(f"Running inference...")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Input: {input_folder}")
    logger.info(f"  Output: {output_folder}")
    
    predictor = Predictor(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
        tile_step_size=0.5,
        use_gaussian=True,
        verbose=True,
        save_preprocessed=False,
        sw_batch_size=2,
        filter_to_brain=False,
        num_workers=num_workers,
        num_postprocess_workers=num_postprocess_workers,
        use_preprocessed=False,  # Always use raw NIfTI for test data
        save_logits=True,  # Always save logits
        save_probabilities=True  # Always save probabilities
    )
    
    predictor.predict_from_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        file_pattern="*.nii.gz"
    )


def run_evaluation(
    predictions_folder: Path,
    labels_folder: Path,
    output_folder: Path,
    num_bins: int = 15,
    num_workers: int = 12
) -> Dict[str, Any]:
    """
    Run evaluation on predictions.
    
    Args:
        predictions_folder: Path to predictions
        labels_folder: Path to ground truth labels
        output_folder: Path to save evaluation results
        num_bins: Number of bins for calibration metrics
        num_workers: Number of parallel workers
        
    Returns:
        Aggregated metrics dictionary
    """
    logger.info(f"Running evaluation...")
    logger.info(f"  Predictions: {predictions_folder}")
    logger.info(f"  Labels: {labels_folder}")
    logger.info(f"  Output: {output_folder}")
    
    # Find matching files
    matches = find_matching_files(
        predictions_folder, labels_folder,
        pred_suffix=".nii.gz",
        label_suffix=".nii.gz"
    )
    
    if len(matches) == 0:
        raise ValueError("No matching files found!")
    
    logger.info(f"Found {len(matches)} matching samples")
    
    # Prepare arguments for parallel processing
    eval_args = [
        (case_name, pred_path, label_path, predictions_folder, num_bins)
        for case_name, pred_path, label_path in matches
    ]
    
    # Evaluate samples in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(evaluate_sample_wrapper, arg): arg[0] for arg in eval_args}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            case_name = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {case_name}: {e}")
                results.append({'case_name': case_name, 'error': str(e)})
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save per-sample CSV
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_path = output_folder / 'per_sample_metrics.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved per-sample metrics to: {csv_path}")
    
    # Aggregate and save JSON
    aggregated = aggregate_metrics(df)
    json_path = output_folder / 'aggregated_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"Saved aggregated metrics to: {json_path}")
    
    return aggregated


def run_inference_and_evaluation(
    checkpoint_path: Path,
    input_folder: Path,
    output_folder: Path,
    labels_folder: Path,
    evaluation_folder: Path,
    config: Config,
    device: str = 'cuda',
    num_workers: int = 6,
    num_postprocess_workers: int = 6,
    num_bins: int = 15,
    sw_batch_size: int = 4
) -> Dict[str, Any]:
    """
    Run inference and evaluation in a combined pipeline with proper parallelization.
    
    Pipeline structure:
    - DataLoader prefetches and preprocesses samples in background (num_workers threads)
    - GPU runs sliding window inference 
    - ThreadPool handles post-processing + evaluation in background (num_postprocess_workers threads)
    
    The key is that while post-processing/evaluation runs on CPU for sample N,
    the GPU is already processing sample N+1. The DataLoader also prefetches
    sample N+2 in parallel.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_folder: Path to input images (imagesTs)
        output_folder: Path to save predictions
        labels_folder: Path to ground truth labels (labelsTs)
        evaluation_folder: Path to save evaluation results
        config: Configuration object
        device: Device to run on
        num_workers: Number of data loading workers
        num_postprocess_workers: Number of post-processing/evaluation workers
        num_bins: Number of bins for calibration metrics
        sw_batch_size: Batch size for sliding window inference (windows per batch)
        
    Returns:
        Aggregated metrics dictionary
    """
    from inference import postprocess_prediction
    import queue
    import time
    
    logger.info(f"Running combined inference and evaluation...")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Input: {input_folder}")
    logger.info(f"  Output: {output_folder}")
    logger.info(f"  Labels: {labels_folder}")
    logger.info(f"  Evaluation: {evaluation_folder}")
    
    # Create output folders
    output_folder.mkdir(parents=True, exist_ok=True)
    evaluation_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Optimize for inference
    if device_obj.type == 'cuda':
        torch.set_num_threads(1)  # Prevent CPU oversubscription
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    from model import create_model
    logger.info("Creating model")
    model = create_model(config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device_obj)
    model.eval()
    
    # Use torch.compile for faster inference if available (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device_obj.type == 'cuda':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("Model compiled with torch.compile for faster inference")
        except Exception as e:
            logger.warning(f"torch.compile failed, using eager mode: {e}")
    
    epoch = checkpoint.get('epoch', 'unknown')
    logger.info(f"Checkpoint loaded (epoch {epoch})")
    logger.info(f"Model ready on {device_obj}")
    
    # Find all input files
    input_files = sorted(input_folder.glob("*.nii.gz"))
    
    if len(input_files) == 0:
        raise ValueError(f"No files found in {input_folder}")
    
    logger.info(f"\nFound {len(input_files)} files total")
    logger.info(f"Processing with:")
    logger.info(f"  - Preprocessing workers: {num_workers}")
    logger.info(f"  - Post-processing/evaluation workers: {num_postprocess_workers}")
    logger.info(f"  - Sliding window batch size: {sw_batch_size}")
    logger.info(f"  - Patch size: {config.data.patch_size}")
    
    # Create dataset and dataloader for parallel preprocessing
    # prefetch_factor controls how many batches are prefetched per worker
    dataset = InferenceDataset(
        image_paths=input_files,
        target_spacing=config.data.target_spacing,
        verbose=False,
        save_preprocessed=False,
        output_dir=None,
        use_preprocessed=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # One 3D volume at a time (they have different sizes)
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device_obj.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
    )
    
    # Create predictor wrapper for deep supervision handling
    predictor = create_predictor_wrapper(model)
    
    # Results storage (thread-safe)
    results = []
    results_lock = Lock()
    
    # Timing stats
    inference_times = []
    postprocess_times = []
    
    # Progress tracking
    completed_count = [0]  # Use list for mutable closure
    pbar = tqdm(total=len(input_files), desc="Inference + Evaluation", 
                dynamic_ncols=True, smoothing=0.1)
    pbar_lock = Lock()
    
    def postprocess_evaluate_and_save(
        predicted_logits: torch.Tensor,
        properties: Dict[str, Any],
        output_path: Path,
        case_name: str,
        label_path: Path,
        start_time: float
    ) -> None:
        """Post-process, save, and evaluate a single sample."""
        postprocess_start = time.time()
        try:
            # Postprocess and save prediction
            seg_nib = postprocess_prediction(
                predicted_logits,
                properties,
                save_preprocessed=False,
                output_dir=output_path.parent,
                case_name=case_name,
                filter_to_brain=False,
                save_logits=True,
                save_probabilities=True
            )
            
            # Save segmentation
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(seg_nib, output_path)
            
            # Evaluate if label exists
            if label_path.exists():
                metrics = evaluate_sample(
                    case_name=case_name,
                    pred_path=output_path,
                    label_path=label_path,
                    pred_folder=output_path.parent,
                    num_bins=num_bins
                )
                
                with results_lock:
                    results.append(metrics)
            else:
                logger.warning(f"No label found for {case_name}, skipping evaluation")
            
            postprocess_time = time.time() - postprocess_start
            with results_lock:
                postprocess_times.append(postprocess_time)
                
        except Exception as e:
            logger.error(f"Error processing {case_name}: {e}")
            import traceback
            traceback.print_exc()
            with results_lock:
                results.append({'case_name': case_name, 'error': str(e)})
        
        finally:
            with pbar_lock:
                completed_count[0] += 1
                pbar.update(1)
                # Update description with timing info
                if len(inference_times) > 0 and len(postprocess_times) > 0:
                    avg_inf = np.mean(inference_times[-10:])  # Last 10 samples
                    avg_pp = np.mean(postprocess_times[-10:])
                    pbar.set_postfix({
                        'inf': f'{avg_inf:.1f}s',
                        'pp': f'{avg_pp:.1f}s',
                        'pending': len(postprocess_futures) - completed_count[0]
                    })
    
    # Create thread pool for background post-processing + evaluation
    # Using more workers here since post-processing is CPU-bound and can run in parallel
    postprocess_executor = ThreadPoolExecutor(max_workers=num_postprocess_workers)
    postprocess_futures = []
    
    try:
        # Create CUDA stream for async operations
        if device_obj.type == 'cuda':
            inference_stream = torch.cuda.Stream()
        else:
            inference_stream = None
        
        # Process each batch - DataLoader handles prefetching automatically
        for tensors_batch, properties_batch, paths_batch in dataloader:
            for image_tensor, properties, input_path in zip(tensors_batch, properties_batch, paths_batch):
                # Get case name
                case_name = input_path.name.replace('.nii.gz', '').replace('_0000', '')
                output_path = output_folder / f"{case_name}.nii.gz"
                label_path = labels_folder / f"{case_name}.nii.gz"
                
                inference_start = time.time()
                
                try:
                    # Add batch dimension and transfer to GPU
                    # Use non_blocking=True for async transfer while previous postprocess runs
                    image_tensor = image_tensor.unsqueeze(0).to(device_obj, non_blocking=True)
                    
                    # Run inference (optionally in separate CUDA stream)
                    with torch.no_grad():
                        if inference_stream is not None:
                            with torch.cuda.stream(inference_stream):
                                predicted_logits = sliding_window_inference(
                                    inputs=image_tensor,
                                    roi_size=config.data.patch_size,
                                    sw_batch_size=sw_batch_size,
                                    predictor=predictor,
                                    overlap=0.5,
                                    mode="gaussian",
                                    sigma_scale=0.125,
                                    padding_mode="constant",
                                    cval=0.0,
                                    sw_device=device_obj,
                                    device=device_obj,
                                    progress=False
                                )
                            # Sync to ensure inference is complete before moving to CPU
                            inference_stream.synchronize()
                        else:
                            predicted_logits = sliding_window_inference(
                                inputs=image_tensor,
                                roi_size=config.data.patch_size,
                                sw_batch_size=sw_batch_size,
                                predictor=predictor,
                                overlap=0.5,
                                mode="gaussian",
                                sigma_scale=0.125,
                                padding_mode="constant",
                                cval=0.0,
                                sw_device=device_obj,
                                device=device_obj,
                                progress=False
                            )
                    
                    inference_time = time.time() - inference_start
                    inference_times.append(inference_time)
                    
                    # Remove batch dimension and move to CPU (non-blocking)
                    # This frees GPU memory while the tensor transfers
                    predicted_logits = predicted_logits[0].cpu()
                    
                    # Clear GPU cache periodically to prevent memory fragmentation
                    if len(inference_times) % 50 == 0 and device_obj.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Submit post-processing + evaluation to background thread
                    # This returns immediately, allowing the next inference to start
                    future = postprocess_executor.submit(
                        postprocess_evaluate_and_save,
                        predicted_logits=predicted_logits,
                        properties=properties,
                        output_path=output_path,
                        case_name=case_name,
                        label_path=label_path,
                        start_time=inference_start
                    )
                    postprocess_futures.append(future)
                    
                except Exception as e:
                    logger.error(f"Error during inference for {input_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    with pbar_lock:
                        pbar.update(1)
        
        # Wait for all post-processing to complete
        logger.info(f"\nInference complete. Waiting for {len(postprocess_futures) - completed_count[0]} remaining post-processing tasks...")
        for future in as_completed(postprocess_futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Post-processing error: {e}")
                
    finally:
        pbar.close()
        postprocess_executor.shutdown(wait=True)
    
    # Print timing statistics
    if inference_times:
        logger.info(f"\nTiming Statistics:")
        logger.info(f"  Inference: {np.mean(inference_times):.2f}s ± {np.std(inference_times):.2f}s per sample")
        logger.info(f"  Post-processing: {np.mean(postprocess_times):.2f}s ± {np.std(postprocess_times):.2f}s per sample")
        logger.info(f"  Total time: {sum(inference_times):.1f}s inference, {sum(postprocess_times):.1f}s post-processing")
    
    logger.info(f"\n✓ All predictions and evaluations complete!")
    
    # Save evaluation results
    if results:
        df = pd.DataFrame(results)
        
        # Save per-sample CSV
        csv_path = evaluation_folder / 'per_sample_metrics.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved per-sample metrics to: {csv_path}")
        
        # Aggregate and save JSON
        aggregated = aggregate_metrics(df)
        json_path = evaluation_folder / 'aggregated_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        logger.info(f"Saved aggregated metrics to: {json_path}")
        
        return aggregated
    else:
        logger.warning("No evaluation results collected!")
        return {'num_samples': 0, 'metrics': {}}


def print_summary(run_info: Dict[str, Any], aggregated: Dict[str, Any]) -> None:
    """Print a summary of the evaluation results."""
    print("\n" + "="*80)
    print("SWEEP EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nBest Run Information:")
    print(f"  Run ID: {run_info['run_id']}")
    print(f"  Run Name: {run_info['run_name']}")
    print(f"  Sweep ID: {run_info['sweep_id']}")
    
    if 'best_val/dice_hard' in run_info:
        print(f"  Validation Dice (hard): {run_info['best_val/dice_hard']:.4f}")
    
    print(f"\nTest Set Results:")
    print(f"  Number of samples: {aggregated['num_samples']}")
    print(f"  Samples with lesions: {aggregated['num_samples_with_lesion']}")
    
    if 'summary' in aggregated:
        print("\nKey Metrics (mean ± std, median):")
        print("-"*60)
        for metric, values in aggregated['summary'].items():
            print(f"  {metric:20s}: {values['mean']:.4f} ± {values['std']:.4f} (median: {values['median']:.4f})")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate best run from a W&B sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--sweep_id', '-s',
        type=str,
        required=True,
        help='W&B sweep ID (format: entity/project/sweep_name or just sweep_name)'
    )

    parser.add_argument(
        '--dataset_path', '-d',
        type=str,
        required=True,
        help='Path to dataset containing imagesTs and labelsTs folders'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='val/dice_hard',
        help='Metric to use for selecting best run'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=6,
        help='Number of workers for data loading'
    )
    
    parser.add_argument(
        '--num_postprocess_workers',
        type=int,
        default=6,
        help='Number of workers for post-processing'
    )
    
    parser.add_argument(
        '--num_eval_workers',
        type=int,
        default=12,
        help='Number of workers for evaluation'
    )
    
    parser.add_argument(
        '--sw_batch_size',
        type=int,
        default=4,
        help='Sliding window batch size (number of windows processed in parallel per sample)'
    )
    
    parser.add_argument(
        '--skip_inference',
        action='store_true',
        help='Skip inference and only run evaluation (predictions must exist)'
    )
    
    parser.add_argument(
        '--skip_evaluation',
        action='store_true',
        help='Skip evaluation and only run inference'
    )
    
    parser.add_argument(
        '--separate_passes',
        action='store_true',
        help='Run inference and evaluation separately (old behavior) instead of combined'
    )
    
    parser.add_argument(
        '--output_subdir',
        type=str,
        default='test_predictions',
        help='Subdirectory name for predictions within the run folder'
    )
    
    args = parser.parse_args()
    
    # Get the best run from the sweep
    best_run, run_info = get_best_run(args.sweep_id, args.metric)
    
    # Load configuration from the run
    config = load_config_from_run(best_run)
    
    # Determine the run directory
    # Sweep runs are stored in experiments/sweeps/fold_X_<run_id>
    sweeps_dir = Path(config.output_dir) / "sweeps"
    
    # Find the run directory by matching the run ID
    run_dir = None
    run_id = best_run.id
    
    for subdir in sweeps_dir.iterdir():
        if subdir.is_dir() and run_id in subdir.name:
            run_dir = subdir
            break
    
    if run_dir is None:
        raise FileNotFoundError(
            f"Could not find run directory for {run_id} in {sweeps_dir}"
        )
    
    logger.info(f"Run directory: {run_dir}")
    
    # Find the checkpoint
    checkpoint_path = find_checkpoint(run_dir)
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Get test data paths
    images_folder, labels_folder = Path(args.dataset_path) / "imagesTs", Path(args.dataset_path) / "labelsTs"
    
    # Set output paths within the run directory
    predictions_folder = run_dir / args.output_subdir
    evaluation_folder = run_dir / "evaluation"
    
    # Decide which mode to use
    if args.skip_inference:
        # Only run evaluation on existing predictions
        logger.info("Skipping inference (--skip_inference flag set)")
        if not predictions_folder.exists():
            raise FileNotFoundError(
                f"Predictions folder not found: {predictions_folder}. "
                f"Run without --skip_inference first."
            )
        
        if not args.skip_evaluation:
            aggregated = run_evaluation(
                predictions_folder=predictions_folder,
                labels_folder=labels_folder,
                output_folder=evaluation_folder,
                num_bins=15,
                num_workers=args.num_eval_workers
            )
            
            # Save run info
            run_info_path = evaluation_folder / 'run_info.json'
            with open(run_info_path, 'w') as f:
                json.dump(run_info, f, indent=2)
            
            # Print summary
            print_summary(run_info, aggregated)
    
    elif args.skip_evaluation:
        # Only run inference
        run_inference(
            checkpoint_path=checkpoint_path,
            input_folder=images_folder,
            output_folder=predictions_folder,
            config=config,
            device=args.device,
            num_workers=args.num_workers,
            num_postprocess_workers=args.num_postprocess_workers
        )
    
    elif args.separate_passes:
        # Run inference and evaluation separately (old behavior)
        run_inference(
            checkpoint_path=checkpoint_path,
            input_folder=images_folder,
            output_folder=predictions_folder,
            config=config,
            device=args.device,
            num_workers=args.num_workers,
            num_postprocess_workers=args.num_postprocess_workers
        )
        
        aggregated = run_evaluation(
            predictions_folder=predictions_folder,
            labels_folder=labels_folder,
            output_folder=evaluation_folder,
            num_bins=15,
            num_workers=args.num_eval_workers
        )
        
        # Save run info
        run_info_path = evaluation_folder / 'run_info.json'
        with open(run_info_path, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        # Print summary
        print_summary(run_info, aggregated)
    
    else:
        # Default: combined inference + evaluation pipeline
        aggregated = run_inference_and_evaluation(
            checkpoint_path=checkpoint_path,
            input_folder=images_folder,
            output_folder=predictions_folder,
            labels_folder=labels_folder,
            evaluation_folder=evaluation_folder,
            config=config,
            device=args.device,
            num_workers=args.num_workers,
            num_postprocess_workers=args.num_postprocess_workers,
            num_bins=15,
            sw_batch_size=args.sw_batch_size
        )
        
        # Save run info
        run_info_path = evaluation_folder / 'run_info.json'
        with open(run_info_path, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        # Print summary
        print_summary(run_info, aggregated)
    
    logger.info(f"\n✓ Evaluation complete!")
    logger.info(f"  Predictions: {predictions_folder}")
    logger.info(f"  Evaluation: {evaluation_folder}")


if __name__ == "__main__":
    main()
