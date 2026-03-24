"""
3D Topograph Loss - Standalone topology-aware loss for 3D segmentation.

Based on: Lux et al (2024) Topograph: An efficient Graph-Based Framework for
Strictly Topology Preserving Image Segmentation (https://arxiv.org/pdf/2411.03228)

Note: Theoretical guarantees do not hold in 3D, but the general framework is applied.
Uses 26-connectivity for true foreground, 6-connectivity for other classes.

This is a standalone loss - combine with Dice/CE via DiceTopographLoss in losses.py.
"""

from enum import Enum
from typing import Optional
import time
import os
import uuid

import cc3d
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


# Debug output directory for saving paired images
TOPOGRAPH_DEBUG_DIR = "/vol/miltank/users/bergeral/vesuvius/segmentation_model/topograph_analysis"


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    CE = "ce"
    RMS = "rms"


class TopographLoss(nn.Module):
    """
    Standalone 3D Topograph Loss for topology-aware image segmentation.
    
    Uses 26-connectivity for true foreground and 6-connectivity for other classes.
    Identifies topologically critical regions (false splits, merges, holes) and
    penalizes the probability of incorrect predictions in those regions.
    
    This is a standalone loss - for combined Dice+Topograph, use DiceTopographLoss.
    """

    def __init__(
        self,
        num_processes: int = 4,
        aggregation: AggregationType | str = AggregationType.MEAN,
        thres_var: float = 0.0,
        include_background: bool = False,
        ignore_index: int = -1,
        sphere: bool = False,
        debug: bool = False,
        error_type: str = "all",
    ) -> None:
        """
        Args:
            num_processes: Number of parallel processes for computation.
            aggregation: Aggregation method (mean, sum, max, min, ce, rms).
            thres_var: Variance of Gaussian threshold noise (0 = no noise).
            include_background: If True, includes background class in computation.
            ignore_index: Label index to ignore in computation (-1 = none).
            sphere: If True, use spherical structuring element for topology.
            debug: If True, prints timing and statistics information.
            error_type: Type of topological errors to consider ("all", "false_positives", "false_negatives").
        """
        super().__init__()

        self.num_processes = num_processes
        self.pool = None
        if self.num_processes > 1:
            self.pool = mp.Pool(num_processes)
        
        self.aggregation = (
            AggregationType(aggregation)
            if not isinstance(aggregation, AggregationType)
            else aggregation
        )
        self.thres_var = thres_var
        self.include_background = include_background
        self.ignore_index = ignore_index
        self.sphere = sphere
        self.debug = debug
        self.error_type = error_type

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the topograph loss.

        Args:
            predictions: Tensor of shape (B, C, D, H, W) - logits.
            targets: Tensor of shape (B, D, H, W) - class indices.

        Returns:
            The computed topograph loss.
        """
        if self.debug:
            total_start_time = time.perf_counter()
            preprocess_start = time.perf_counter()
        
        num_classes = predictions.shape[1]
        starting_class = 0 if self.include_background else 1
        batch_size = predictions.shape[0]
        spatial_shape = predictions.shape[2:]
        
        # Apply softmax to get probabilities (needed for loss computation)
        probs = F.softmax(predictions, dim=1)
        
        # Create argmax predictions directly from logits (more efficient than argmax(softmax(x)))
        argmax_preds = torch.argmax(predictions, dim=1)
        argmax_gts = targets.clone()
        
        # Handle ignore_index: set both pred and gt to 0 (background) in ignored regions
        if self.ignore_index != 0:
            ignore_mask = (targets == self.ignore_index)
            argmax_preds = argmax_preds.clone()
            argmax_preds[ignore_mask] = 0
            argmax_gts[ignore_mask] = 0
        
        # Apply Gaussian threshold noise if configured
        if self.thres_var > 0:
            probs_detached = probs.detach().clone()
            
            # Add noise to a randomly selected class per sample
            for b in range(predictions.shape[0]):
                noise_class = torch.randint(0, num_classes, (1,), device=predictions.device).item()
                thres_noise = torch.randn(1, device=predictions.device) * self.thres_var
                neg_noise = thres_noise / max(1, num_classes - 1)
                
                # Add noise to selected class, subtract from all to maintain normalization
                probs_detached[b, noise_class] += thres_noise + neg_noise
                probs_detached[b] -= neg_noise
            
            # Recompute argmax with noisy probabilities
            argmax_preds_noisy = torch.argmax(probs_detached, dim=1)
            # Re-apply ignore mask
            if self.ignore_index != 0:
                argmax_preds_noisy[ignore_mask] = 0
        else:
            argmax_preds_noisy = argmax_preds
        
        # Apply sphere padding (periodic boundary conditions)
        if self.sphere:
            argmax_preds_padded = F.pad(argmax_preds_noisy, (1, 1, 1, 1, 1, 1), value=0)
            argmax_gts_padded = F.pad(argmax_gts, (1, 1, 1, 1, 1, 1), value=0)
        else:
            argmax_preds_padded = argmax_preds_noisy
            argmax_gts_padded = argmax_gts
        
        if self.debug:
            preprocess_time = time.perf_counter() - preprocess_start
            prepare_inputs_start = time.perf_counter()
            # Generate unique ID for this batch
            batch_debug_id = str(uuid.uuid4())[:8]
        
        # Prepare inputs for each class and batch
        single_calc_inputs = []
        
        # Prepare ignore_mask for debug saving if needed
        if self.debug and self.ignore_index != 0:
            ignore_mask_np = (targets == self.ignore_index).cpu().numpy()
            if self.sphere:
                # Pad ignore mask similarly to predictions/gt
                ignore_mask_padded = np.pad(ignore_mask_np, ((0, 0), (1, 1), (1, 1), (1, 1)), constant_values=False)
            else:
                ignore_mask_padded = ignore_mask_np
        else:
            ignore_mask_padded = None
        
        for class_index in range(starting_class, num_classes):
            # Binarize for current class
            bin_preds = (argmax_preds_padded == class_index).cpu().numpy().astype(np.uint8)
            bin_gts = (argmax_gts_padded == class_index).cpu().numpy().astype(np.uint8)

            for i in range(predictions.shape[0]):
                single_calc_inputs.append({
                    "bin_pred": bin_preds[i],
                    "bin_gt": bin_gts[i],
                    "sample_no": i,
                    "class_index": class_index,
                    "debug": self.debug,
                    "debug_id": batch_debug_id if self.debug else None,
                    "ignore_mask": ignore_mask_padded[i] if ignore_mask_padded is not None else None,
                    "error_type": self.error_type,
                })

        if self.debug:
            prepare_inputs_time = time.perf_counter() - prepare_inputs_start
            cc_start = time.perf_counter()
        
        # Process samples with multiprocessing
        if self.num_processes > 1 and self.pool is not None:
            chunksize = max(1, len(single_calc_inputs) // self.num_processes)
            results = list(self.pool.imap_unordered(
                single_sample_class_loss, single_calc_inputs, chunksize=chunksize
            ))
        else:
            results = list(map(single_sample_class_loss, single_calc_inputs))
        
        if self.debug:
            cc_time = time.perf_counter() - cc_start
            loss_agg_start = time.perf_counter()
            total_nodes = 0
            total_critical_nodes = 0
            total_critical_voxels = 0
            total_regions_processed = 0
            total_tp_nodes = 0
            total_tn_nodes = 0
            total_fp_nodes = 0
            total_fn_nodes = 0
            total_fp_voxels = 0
            total_fn_voxels = 0

        # =====================================================================
        # OPTIMIZED LOSS AGGREGATION
        # Collect all indices first, then do single batched GPU operations
        # =====================================================================
        
        # Collect all critical voxel data across all samples/classes
        all_sample_indices = []  # Which sample each voxel belongs to
        all_region_labels = []   # Which region each voxel belongs to (for scatter ops)
        all_z_indices = []
        all_y_indices = []
        all_x_indices = []
        
        region_counter = 0
        d, h, w = predictions.shape[2:]
        
        for result in results:
            if self.debug:
                region_indices_list, sample_no, class_index, debug_info = result
                total_nodes += debug_info.get('total_nodes', 0)
                total_critical_nodes += debug_info.get('num_critical_nodes', 0)
                total_tp_nodes += debug_info.get('tp_nodes', 0)
                total_tn_nodes += debug_info.get('tn_nodes', 0)
                total_fp_nodes += debug_info.get('fp_nodes', 0)
                total_fn_nodes += debug_info.get('fn_nodes', 0)
                total_fp_voxels += debug_info.get('fp_voxels', 0)
                total_fn_voxels += debug_info.get('fn_voxels', 0)
            else:
                region_indices_list, sample_no, class_index = result[:3]
            
            for region_indices in region_indices_list:
                if len(region_indices[0]) == 0:
                    continue

                # Adjust indices for sphere padding (subtract 1) if sphere mode
                if self.sphere:
                    region_indices = tuple(idx - 1 for idx in region_indices)
                    
                    # Clip indices to valid range (in case padding artifacts)
                    valid_mask = (
                        (region_indices[0] >= 0) & (region_indices[0] < d) &
                        (region_indices[1] >= 0) & (region_indices[1] < h) &
                        (region_indices[2] >= 0) & (region_indices[2] < w)
                    )
                    region_indices = tuple(idx[valid_mask] for idx in region_indices)
                
                if len(region_indices[0]) == 0:
                    continue
                
                n_voxels = len(region_indices[0])
                
                if self.debug:
                    total_critical_voxels += n_voxels
                    total_regions_processed += 1
                
                # Collect indices
                all_sample_indices.append(np.full(n_voxels, sample_no, dtype=np.int64))
                all_region_labels.append(np.full(n_voxels, region_counter, dtype=np.int64))
                all_z_indices.append(region_indices[0])
                all_y_indices.append(region_indices[1])
                all_x_indices.append(region_indices[2])
                
                region_counter += 1
        
        # Compute loss
        if region_counter == 0:
            # No critical regions found
            g_loss = torch.tensor(0.0, device=predictions.device)
        else:
            # Concatenate all indices
            all_sample_indices = np.concatenate(all_sample_indices)
            all_region_labels = np.concatenate(all_region_labels)
            all_z_indices = np.concatenate(all_z_indices)
            all_y_indices = np.concatenate(all_y_indices)
            all_x_indices = np.concatenate(all_x_indices)
            
            # Convert to torch tensors on GPU
            sample_idx = torch.from_numpy(all_sample_indices).to(predictions.device)
            region_labels = torch.from_numpy(all_region_labels).to(predictions.device)
            z_idx = torch.from_numpy(all_z_indices).to(predictions.device)
            y_idx = torch.from_numpy(all_y_indices).to(predictions.device)
            x_idx = torch.from_numpy(all_x_indices).to(predictions.device)
            
            # Get predicted class at each critical voxel (single indexing operation)
            class_at_voxel = argmax_preds[sample_idx, z_idx, y_idx, x_idx]
            
            # Get probability of predicted class at each voxel (single indexing operation)
            voxel_probs = probs[sample_idx, class_at_voxel, z_idx, y_idx, x_idx]
            
            # Aggregate per region using scatter_reduce
            num_regions = region_counter
            
            match self.aggregation:
                case AggregationType.MAX:
                    # scatter_reduce with 'amax' for max per region
                    region_max = torch.zeros(num_regions, device=predictions.device)
                    region_max.scatter_reduce_(0, region_labels, voxel_probs, reduce='amax', include_self=False)
                    g_loss = region_max.sum()
                    
                case AggregationType.MIN:
                    # scatter_reduce with 'amin' for min per region
                    region_min = torch.full((num_regions,), float('inf'), device=predictions.device)
                    region_min.scatter_reduce_(0, region_labels, voxel_probs, reduce='amin', include_self=False)
                    g_loss = region_min.sum()
                    
                case AggregationType.SUM:
                    # scatter_reduce with 'sum'
                    region_sum = torch.zeros(num_regions, device=predictions.device)
                    region_sum.scatter_reduce_(0, region_labels, voxel_probs, reduce='sum', include_self=False)
                    g_loss = region_sum.sum()
                    
                case AggregationType.MEAN:
                    # For mean: sum / count per region
                    region_sum = torch.zeros(num_regions, device=predictions.device)
                    region_count = torch.zeros(num_regions, device=predictions.device)
                    region_sum.scatter_reduce_(0, region_labels, voxel_probs, reduce='sum', include_self=False)
                    region_count.scatter_reduce_(0, region_labels, torch.ones_like(voxel_probs), reduce='sum', include_self=False)
                    region_mean = region_sum / region_count.clamp(min=1)
                    g_loss = region_mean.sum()
                    
                case AggregationType.RMS:
                    # For RMS: sqrt(sum(x^2) / count) per region
                    region_sum_sq = torch.zeros(num_regions, device=predictions.device)
                    region_count = torch.zeros(num_regions, device=predictions.device)
                    region_sum_sq.scatter_reduce_(0, region_labels, voxel_probs ** 2, reduce='sum', include_self=False)
                    region_count.scatter_reduce_(0, region_labels, torch.ones_like(voxel_probs), reduce='sum', include_self=False)
                    region_rms = torch.sqrt(region_sum_sq / region_count.clamp(min=1))
                    g_loss = region_rms.sum()
                    
                case AggregationType.CE:
                    # For CE: need logits and targets per voxel, then mean CE per region
                    voxel_logits = predictions[sample_idx, :, z_idx, y_idx, x_idx]  # (N, C)
                    voxel_targets = targets[sample_idx, z_idx, y_idx, x_idx]  # (N,)
                    
                    # Compute per-voxel CE loss
                    voxel_ce = F.cross_entropy(voxel_logits, voxel_targets.long(), reduction='none')
                    
                    # Mean CE per region
                    region_ce_sum = torch.zeros(num_regions, device=predictions.device)
                    region_count = torch.zeros(num_regions, device=predictions.device)
                    region_ce_sum.scatter_reduce_(0, region_labels, voxel_ce, reduce='sum', include_self=False)
                    region_count.scatter_reduce_(0, region_labels, torch.ones_like(voxel_ce), reduce='sum', include_self=False)
                    region_ce_mean = region_ce_sum / region_count.clamp(min=1)
                    g_loss = region_ce_mean.sum()
                    
                case _:
                    # Default to mean
                    region_sum = torch.zeros(num_regions, device=predictions.device)
                    region_count = torch.zeros(num_regions, device=predictions.device)
                    region_sum.scatter_reduce_(0, region_labels, voxel_probs, reduce='sum', include_self=False)
                    region_count.scatter_reduce_(0, region_labels, torch.ones_like(voxel_probs), reduce='sum', include_self=False)
                    region_mean = region_sum / region_count.clamp(min=1)
                    g_loss = region_mean.sum()

        # Normalize by batch size and number of classes
        num_class_range = max(1, num_classes - starting_class)
        g_loss = g_loss / (predictions.shape[0] * num_class_range)
        
        if self.debug:
            loss_agg_time = time.perf_counter() - loss_agg_start
            total_time = time.perf_counter() - total_start_time
            
            print("\n" + "="*60)
            print("TOPOGRAPH LOSS DEBUG INFO")
            print("="*60)
            print(f"Input shape: predictions={tuple(predictions.shape)}, targets={tuple(targets.shape)}")
            print(f"Batch size: {batch_size}, Num classes: {num_classes}, Spatial shape: {spatial_shape}")
            print(f"Aggregation: {self.aggregation.value}, Num processes: {self.num_processes}")
            print("-"*60)
            print("TIMING BREAKDOWN:")
            print(f"  Preprocessing (softmax, argmax, padding): {preprocess_time*1000:.2f} ms")
            print(f"  Prepare inputs (binarization, to numpy):  {prepare_inputs_time*1000:.2f} ms")
            print(f"  CC labeling & graph building:             {cc_time*1000:.2f} ms")
            print(f"  Loss aggregation:                         {loss_agg_time*1000:.2f} ms")
            print(f"  TOTAL:                                    {total_time*1000:.2f} ms")
            print("-"*60)
            print("STATISTICS:")
            print(f"  Total nodes (all samples/classes):        {total_nodes}")
            print(f"    ├─ TP nodes (pred=1, gt=1):             {total_tp_nodes}")
            print(f"    ├─ TN nodes (pred=0, gt=0):             {total_tn_nodes}")
            print(f"    ├─ FP nodes (pred=1, gt=0):             {total_fp_nodes}")
            print(f"    └─ FN nodes (pred=0, gt=1):             {total_fn_nodes}")
            print(f"  Total critical nodes:                     {total_critical_nodes}")
            total_error_voxels = total_fp_voxels + total_fn_voxels
            non_critical_voxels = total_error_voxels - total_critical_voxels
            critical_ratio = total_critical_voxels / total_error_voxels * 100 if total_error_voxels > 0 else 0
            print(f"  Total error voxels (FP + FN):             {total_error_voxels}")
            print(f"    ├─ Critical voxels:                     {total_critical_voxels} ({critical_ratio:.1f}%)")
            print(f"    └─ Non-critical voxels:                 {non_critical_voxels} ({100-critical_ratio:.1f}%)")
            print(f"  Regions processed:                        {total_regions_processed}")
            print(f"  Samples x Classes processed:              {len(single_calc_inputs)}")
            print("-"*60)
            print(f"RESULT: Topograph Loss = {g_loss.item():.6f}")
            print("="*60 + "\n", flush=True)
        
        return g_loss


def single_sample_class_loss(args: dict):
    """Wrapper for single sample class loss computation."""
    return _single_sample_class_loss(**args)


def _single_sample_class_loss(bin_pred, bin_gt, sample_no, class_index, debug=False, debug_id=None, ignore_mask=None, error_type="all"):
    """
    Compute critical regions for a single sample and class.
    
    Uses cc3d for connected component labeling and region graph construction.
    True foreground (pred=1, gt=1) uses 26-connectivity.
    Other classes use 6-connectivity.
    
    Args:
        bin_pred: Binary prediction array (D, H, W)
        bin_gt: Binary ground truth array (D, H, W)
        sample_no: Sample index in the batch
        class_index: Class index being processed
        debug: If True, collect debug statistics
        debug_id: Unique identifier for debug file saving
        ignore_mask: Boolean mask of ignored regions (D, H, W), prediction foreground is masked out here
        error_type: Type of topological errors to consider ("all", "false_positives", "false_negatives")
    
    Returns:
        Tuple of (region_indices_list, sample_no, class_index)
    """
    # Apply ignore_mask: mask out prediction foreground in ignored regions
    # This ensures predictions in ignored regions don't contribute to topology analysis
    if ignore_mask is not None:
        bin_pred_masked = bin_pred.copy()
        bin_pred_masked[ignore_mask] = 0
    else:
        bin_pred_masked = bin_pred
    
    # Create paired image: encodes pred and gt state
    # 0 = TN (pred=0, gt=0), 1 = FP (pred=1, gt=0)
    # 2 = FN (pred=0, gt=1), 3 = TP (pred=1, gt=1)
    paired_img = bin_pred_masked.astype(np.uint8) + 2 * bin_gt.astype(np.uint8)
    
    # Set up debug subfolder for this sample/class pair
    debug_subfolder = None
    if debug and debug_id is not None:
        debug_subfolder = os.path.join(TOPOGRAPH_DEBUG_DIR, f"{debug_id}_sample{sample_no}_class{class_index}")
        os.makedirs(debug_subfolder, exist_ok=True)
        
        # Save paired image in subfolder
        paired_filepath = os.path.join(debug_subfolder, "paired_img.tif")
        tifffile.imwrite(paired_filepath, paired_img)
        
        # Also save ignore_mask if present for reference
        if ignore_mask is not None:
            ignore_filepath = os.path.join(debug_subfolder, "ignore_mask.tif")
            tifffile.imwrite(ignore_filepath, ignore_mask.astype(np.uint8))
    
    # Label each intersection class separately with appropriate connectivity
    # cc3d works efficiently with boolean/uint8 arrays
    # True foreground (TP: paired=3) with 26-connectivity
    tp_labels, tp_n = cc3d.connected_components(paired_img == 3, connectivity=26, return_N=True)
    
    # Other classes with 6-connectivity (TN) or 26-connectivity (FP, FN)
    tn_labels, tn_n = cc3d.connected_components(paired_img == 0, connectivity=6, return_N=True)
    fp_labels, fp_n = cc3d.connected_components(paired_img == 1, connectivity=26, return_N=True)
    fn_labels, fn_n = cc3d.connected_components(paired_img == 2, connectivity=26, return_N=True)
    
    total_n = tp_n + tn_n + fp_n + fn_n
    
    if total_n == 0:
        if debug:
            return [], sample_no, class_index, {
                'total_nodes': 0, 'num_critical_nodes': 0,
                'tp_nodes': 0, 'tn_nodes': 0, 'fp_nodes': 0, 'fn_nodes': 0,
                'fp_voxels': 0, 'fn_voxels': 0
            }
        return [], sample_no, class_index
    
    # Combine labels into a single label array with unique IDs using vectorized operations
    # Pre-compute offsets
    offset_tn = tp_n
    offset_fp = tp_n + tn_n
    offset_fn = tp_n + tn_n + fp_n
    
    # Build all_labels efficiently: only write non-zero regions
    all_labels = np.zeros(paired_img.shape, dtype=np.int32)
    
    # Use where to avoid double boolean indexing
    if tp_n > 0:
        mask = tp_labels > 0
        all_labels[mask] = tp_labels[mask]
    if tn_n > 0:
        mask = tn_labels > 0
        all_labels[mask] = tn_labels[mask] + offset_tn
    if fp_n > 0:
        mask = fp_labels > 0
        all_labels[mask] = fp_labels[mask] + offset_fp
    if fn_n > 0:
        mask = fn_labels > 0
        all_labels[mask] = fn_labels[mask] + offset_fn
    
    # Build region graph using 26-connectivity (captures all adjacencies)
    edges = cc3d.region_graph(all_labels, connectivity=26)
    
    # Build node info: for each region, store predicted class and gt class
    node_info = {}  # node_id -> (pred_class, gt_class)
    
    # TP nodes: pred=1, gt=1
    for i in range(1, tp_n + 1):
        node_info[i] = (1, 1)
    
    # TN nodes: pred=0, gt=0
    for i in range(1, tn_n + 1):
        node_info[tp_n + i] = (0, 0)
    
    # FP nodes: pred=1, gt=0
    for i in range(1, fp_n + 1):
        node_info[tp_n + tn_n + i] = (1, 0)
    
    # FN nodes: pred=0, gt=1
    for i in range(1, fn_n + 1):
        node_info[tp_n + tn_n + fp_n + i] = (0, 1)
    
    # Build adjacency list
    adjacency = {i: set() for i in range(1, total_n + 1)}
    for n1, n2 in edges:
        if n1 == 0 or n2 == 0:  # Skip background (0 label)
            continue
        if n1 in adjacency and n2 in adjacency:
            adjacency[n1].add(n2)
            adjacency[n2].add(n1)
    
    # Find critical nodes
    critical_nodes = get_critical_nodes(node_info, adjacency, error_type=error_type)
    
    # Create relabel masks for critical nodes
    region_indices_list = create_relabel_masks(critical_nodes, all_labels)
    
    # Save critical regions mask if debug mode is enabled
    if debug and debug_subfolder is not None:
        # Create a mask showing only critical regions
        critical_mask = np.zeros_like(all_labels, dtype=np.uint8)
        for node_id in critical_nodes:
            critical_mask[all_labels == node_id] = node_id % 255 + 1  # Assign unique label (1-255)
        
        critical_filepath = os.path.join(debug_subfolder, "critical_regions.tif")
        tifffile.imwrite(critical_filepath, critical_mask)
    
    if debug:
        # Count total voxels in error regions (FP and FN)
        fp_voxels = int(np.sum(paired_img == 1))
        fn_voxels = int(np.sum(paired_img == 2))
        
        debug_info = {
            'total_nodes': total_n,
            'tp_nodes': tp_n,
            'tn_nodes': tn_n,
            'fp_nodes': fp_n,
            'fn_nodes': fn_n,
            'num_critical_nodes': len(critical_nodes),
            'num_edges': len(edges),
            'fp_voxels': fp_voxels,
            'fn_voxels': fn_voxels,
        }
        return region_indices_list, sample_no, class_index, debug_info
    
    return region_indices_list, sample_no, class_index


def get_critical_nodes(node_info, adjacency, error_type="all"):
    """
    Identify critical nodes that cause topological errors.
    
    A node is critical if it is incorrectly predicted (pred != gt) and
    does not have exactly one correct foreground neighbor and one correct
    background neighbor.
    
    Args:
        node_info: Dict mapping node_id -> (pred_class, gt_class)
        adjacency: Dict mapping node_id -> set of neighbor node_ids
    
    Returns:
        List of critical node IDs
    """
    critical_nodes = []
    
    for node_id, (pred_class, gt_class) in node_info.items():
        # Skip correctly predicted nodes
        if pred_class == gt_class:
            continue
        
        neighbors = adjacency.get(node_id, set())
        
        # Count correct foreground neighbors (TP: pred=1, gt=1)
        correct_fg_count = 0
        # Count correct background neighbors (TN: pred=0, gt=0)
        correct_bg_count = 0
        
        for nbr in neighbors:
            if nbr not in node_info:
                continue
            nbr_pred, nbr_gt = node_info[nbr]
            
            # Correct foreground: pred=1 and gt=1
            if nbr_pred == 1 and nbr_gt == 1:
                correct_fg_count += 1
            # Correct background: pred=0 and gt=0
            elif nbr_pred == 0 and nbr_gt == 0:
                correct_bg_count += 1
        
        # Critical if not exactly one correct FG and one correct BG neighbor
        if (correct_fg_count != 1 or correct_bg_count != 1) and (error_type == "all" or (error_type == "false_positives" and pred_class == 1) or (error_type == "false_negatives" and pred_class == 0)):
                    critical_nodes.append(node_id)
    
    return critical_nodes


def create_relabel_masks(critical_nodes, all_labels):
    """
    Create index masks for each critical node region.
    
    Optimized version: single pass through the array using vectorized operations.
    
    Args:
        critical_nodes: List of critical node IDs
        all_labels: Label array (D, H, W)
    
    Returns:
        List of index tuples, each containing (z_indices, y_indices, x_indices)
    """
    if len(critical_nodes) == 0:
        return []
    
    # Convert to set for O(1) lookup
    critical_set = set(critical_nodes)
    
    # Create mask of all critical voxels in single pass
    # Use np.isin for vectorized membership test
    critical_node_array = np.array(list(critical_set), dtype=np.int32)
    is_critical = np.isin(all_labels, critical_node_array)
    
    # Get all critical voxel coordinates at once
    critical_coords = np.nonzero(is_critical)
    
    if len(critical_coords[0]) == 0:
        return []
    
    # Get the label at each critical voxel
    labels_at_coords = all_labels[critical_coords]
    
    # Sort by label to group voxels belonging to same region
    sort_idx = np.argsort(labels_at_coords)
    sorted_labels = labels_at_coords[sort_idx]
    sorted_z = critical_coords[0][sort_idx]
    sorted_y = critical_coords[1][sort_idx]
    sorted_x = critical_coords[2][sort_idx]
    
    # Find split points where label changes
    label_changes = np.nonzero(np.diff(sorted_labels))[0] + 1
    split_points = np.concatenate([[0], label_changes, [len(sorted_labels)]])
    
    # Build result list
    region_indices_list = []
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        region_indices_list.append((
            sorted_z[start:end],
            sorted_y[start:end],
            sorted_x[start:end]
        ))
    
    return region_indices_list
