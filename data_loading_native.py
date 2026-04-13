"""
Native PyTorch data loading implementation for medical image segmentation.

This implementation uses standard PyTorch Dataset and DataLoader without
TorchIO's Queue and Sampler systems, providing:
- Full control over patch sampling logic
- Standard PyTorch multiprocessing for data loading
- Foreground oversampling (nnUNet style)
- Efficient patch extraction with caching
- Compatible with TorchIO transforms

Key differences from TorchIO version:
- Uses torch.utils.data.Dataset for patch sampling
- Standard DataLoader with configurable num_workers
- Manual implementation of foreground oversampling
- Pre-caching of volume locations for faster sampling
"""
import os
import json
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset, IterableDataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Iterator
from pathlib import Path
import nibabel as nib
import copy
from concurrent.futures import ThreadPoolExecutor
import time
from transforms_torchio import OversizedCrop


class DataManager:
    """
    Data management using nnUNet structure.
    Handles file paths, split loading, and metadata.
    """
    
    def __init__(
        self,
        data_path: str,
        splits_file: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        self.data_path = Path(data_path)
        self.splits_file = splits_file or self.data_path / "splits_final.json"
        self.max_samples = max_samples

        # Load metadata
        self.dataset_json = self._load_dataset_json()
        self.splits = self._load_splits()

        # Apply filtering if requested
        if self.max_samples is not None:
            self._apply_filters()
    
    def _load_dataset_json(self) -> Dict:
        """Load dataset.json"""
        dataset_json_path = self.data_path / "dataset.json"
        if not dataset_json_path.exists():
            # If dataset.json doesn't exist, create a minimal one
            return {"name": "Unknown", "numTraining": 0}
        with open(dataset_json_path, 'r') as f:
            return json.load(f)
    
    def _load_splits(self) -> List[Dict]:
        """Load cross-validation splits"""
        with open(self.splits_file, 'r') as f:
            return json.load(f)
    
    def _apply_filters(self):
        """Apply max_samples limit to splits"""
        for fold_idx, fold_data in enumerate(self.splits):
            train_ids = fold_data['train'][:self.max_samples]
            val_ids = fold_data['val'][:self.max_samples]

            self.splits[fold_idx]['train'] = train_ids
            self.splits[fold_idx]['val'] = val_ids

            print(f"Fold {fold_idx} after filtering: {len(train_ids)} train, {len(val_ids)} val")

    def get_fold_case_ids(self, fold: int) -> Tuple[List[str], List[str]]:
        """
        Get case IDs for training and validation.
        
        Args:
            fold: Cross-validation fold number
        
        Returns:
            Tuple of (train_ids, val_ids)
        """
        if fold >= len(self.splits):
            raise ValueError(f"Fold {fold} not available. Only {len(self.splits)} folds found.")
        
        fold_data = self.splits[fold]
        return fold_data['train'], fold_data['val']

    def get_all_case_ids(self) -> List[str]:
        """
        Get all unique case IDs across all folds (train + val combined).
        Used for train_on_all mode.

        Returns:
            Sorted list of all unique case IDs
        """
        all_ids = set()
        for fold_data in self.splits:
            all_ids.update(fold_data['train'])
            all_ids.update(fold_data['val'])
        return sorted(all_ids)


class PatchDataset(IterableDataset):
    """
    Unified PyTorch IterableDataset for efficient patch-based training and validation.
    
    This dataset addresses the I/O bottleneck by loading each volume once and
    extracting multiple patches from it before moving to the next volume.
    
    Key Features:
    -------------
    1. **Streaming/Iterable Approach**: Loads a volume into RAM, extracts N patches
       (patches_per_volume), then moves to the next volume. This dramatically reduces
       I/O overhead compared to loading a fresh volume for each patch.
    
    2. **Worker Partitioning**: When using multiple DataLoader workers, each worker
       processes a mutually exclusive subset of volumes. Worker i takes files
       [i, i+num_workers, i+2*num_workers, ...]. This prevents data duplication.
    
    3. **Internal Shuffling**: Since IterableDataset doesn't support DataLoader's
       shuffle=True, shuffling is done internally at the start of each epoch.
       Each worker shuffles its assigned subset of case IDs.
       (Only for training mode when is_training=True)
    
    4. **Training/Validation Modes**: Controlled by is_training flag:
       - Training: Shuffling enabled, random patch variation, oversized crop + augmentation
       - Validation: Deterministic order, fixed patches per volume, direct crop to target size
    
    Batch Diversity & Multi-Worker Interleaving:
    --------------------------------------------
    With this design, patches from a single volume are yielded consecutively by one
    worker. However, when num_workers > 1, the DataLoader interleaves batches from
    different workers. Since each worker processes different patients, the resulting
    batches naturally contain patches from multiple patients.
    
    Example with batch_size=4, num_workers=4, patches_per_volume=16:
    - Worker 0 yields patches from Patient A (16 patches)
    - Worker 1 yields patches from Patient B (16 patches)
    - Worker 2 yields patches from Patient C (16 patches)
    - Worker 3 yields patches from Patient D (16 patches)
    - DataLoader collects: [A_patch1, B_patch1, C_patch1, D_patch1] -> Batch 1
    - DataLoader collects: [A_patch2, B_patch2, C_patch2, D_patch2] -> Batch 2
    - ... and so on
    
    Result: Each batch contains patches from 4 different patients, providing
    excellent batch diversity without needing a shuffle buffer.
    
    Recommendation: Set num_workers >= batch_size for optimal batch diversity.
    If num_workers < batch_size, some batches may contain multiple patches from
    the same patient.
    
    Transform Compatibility:
    -----------------------
    The existing TorchIO transform pipeline expects a full volume and handles
    cropping internally (via RandomCrop or similar). This dataset:
    1. Creates a TorchIO Subject from the loaded volume
    2. Passes the SAME volume to the transform pipeline N times
    3. Since transforms include random cropping, each pass produces a different patch
    4. The transform pipeline is applied to a deep copy of the subject to ensure
       independent random augmentations for each patch
    """
    
    def __init__(
        self,
        case_ids: List[str],
        data_path: Path,
        patch_size: Tuple[int, int, int],
        patches_per_volume: int = 16,
        is_training: bool = True,
        use_preprocessed: bool = False,
        use_compressed: bool = False,
        transforms: Optional[tio.Compose] = None,
        cache_volumes: bool = False,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        seed: Optional[int] = None,
        foreground_oversample_percent: Optional[float] = 0.33,
        oversize_factor: Optional[float] = 1.2,
        compute_valid_bounds: bool = False,
        verbose: bool = False,
        # Text-prompted mode parameters
        text_prompted: bool = False,
        precomputed_embeddings: Optional[Dict[str, 'torch.Tensor']] = None,
        prompts_data: Optional[Dict[str, list]] = None,
        distance_field_weight: float = 0.0,
        distance_field_sigma: float = 20.0,
    ):
        """
        Args:
            case_ids: List of case identifiers
            data_path: Path to data directory
            patch_size: Final target patch size (z, y, x)
            patches_per_volume: Number of patches to extract from each loaded volume.
                               Higher values = less I/O overhead but less variety per epoch.
            is_training: If True, enables shuffling, random patch variation, and oversized crop.
                        If False, deterministic order, fixed patches, direct crop.
            use_preprocessed: Whether to load preprocessed numpy arrays
            use_compressed: Whether to use compressed .npz files
            transforms: TorchIO transforms to apply (augmentation for training)
            cache_volumes: Whether to cache loaded volumes in memory (high memory usage)
            target_spacing: Target spacing (z, y, x) for preprocessed data affine matrix
            seed: Random seed for reproducibility. If None, uses a random seed.
            foreground_oversample_percent: Probability of sampling foreground-centered patches
            oversize_factor: Factor for oversized crop (training only)
            verbose: If True, print debugging information like foreground ratios
            compute_valid_bounds: Whether to compute valid bounds for labels
        """
        super().__init__()
        self.case_ids = case_ids
        self.data_path = data_path
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.is_training = is_training
        self.use_preprocessed = use_preprocessed
        self.use_compressed = use_compressed
        self.transforms = transforms
        self.cache_volumes = cache_volumes
        self.target_spacing = target_spacing
        self.seed = seed
        self.verbose = verbose
        self.compute_valid_bounds = compute_valid_bounds
        
        # Volume cache (optional)
        self._volume_cache = {} if cache_volumes else None
        
        # Create affine matrix for preprocessed data (diagonal with spacing)
        if use_preprocessed and target_spacing is not None:
            self.preprocessed_affine = np.diag([target_spacing[0], target_spacing[1], target_spacing[2], 1.0])
        else:
            self.preprocessed_affine = np.eye(4)

        # Training mode: Use oversized crop with foreground oversampling
        # Validation mode: Direct crop to target size with high foreground bias
        if is_training:
            self.oversized_crop = OversizedCrop(
                target_size=tuple(int(s * oversize_factor) for s in patch_size),
                foreground_oversample_percent=foreground_oversample_percent,
                padding_mode='minimum',
                center=False
            )
        else:
            # For validation, no oversized crop needed
            self.oversized_crop = None
        
        # Final crop transform (applied after all augmentations)
        self.final_crop = OversizedCrop(
            target_size=patch_size,
            foreground_oversample_percent=1.0 if not is_training else 0.,
            padding_mode='minimum',
            center=True if is_training else False
        )

        self.executor = ThreadPoolExecutor(max_workers=1)

        # Text-prompted mode
        self.text_prompted = text_prompted
        self.precomputed_embeddings = precomputed_embeddings
        self.prompts_data = prompts_data
        self.distance_field_weight = distance_field_weight
        self.distance_field_sigma = distance_field_sigma
        if text_prompted:
            assert precomputed_embeddings is not None, "precomputed_embeddings required for text-prompted mode"
            assert prompts_data is not None, "prompts_data required for text-prompted mode"

        # For validation text-prompted: precompute total prompt count for accurate __len__
        self._val_tp_total_prompts = None
        if not is_training and text_prompted and prompts_data:
            self._val_tp_total_prompts = sum(
                len(self._select_validation_prompts(cid)) for cid in case_ids
            )

        # Epoch counter for proper shuffling across epochs
        # IMPORTANT: Call set_epoch(epoch) before each epoch to ensure different shuffling
        self._epoch = 0

        mode_str = "training" if is_training else "validation"
        tp_str = " [text-prompted]" if text_prompted else ""
        print(f"PatchDataset ({mode_str}{tp_str}) initialized: {len(case_ids)} cases, "
              f"{patches_per_volume} patches/volume, "
              f"virtual size = {len(self)}")
    
    def _compute_valid_bounds(self, label: torch.Tensor) -> torch.Tensor:
        """Compute bounding box of valid (non-ignore) region from label tensor.
        
        Args:
            label: (C, W, H, D) label tensor where -1 indicates ignore regions
            
        Returns:
            Tensor of shape (6,) containing [w_min, w_max, h_min, h_max, d_min, d_max],
            or tensor of [-1, -1, -1, -1, -1, -1] if all voxels are valid (no ignore regions).
        """
        # Check if there are any ignore regions
        valid_mask = label[0] != -1
        if valid_mask.all():
            return torch.tensor([-1, -1, -1, -1, -1, -1], dtype=torch.long)  # Sentinel for "all valid"
        
        if not valid_mask.any():
            return torch.tensor([-1, -1, -1, -1, -1, -1], dtype=torch.long)  # No valid voxels (edge case)
        
        # Find valid voxel coordinates efficiently using nonzero
        valid_coords = torch.nonzero(valid_mask, as_tuple=False)
        
        # Compute min/max along each axis
        mins = valid_coords.min(dim=0).values
        maxs = valid_coords.max(dim=0).values
        
        return torch.tensor([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]], dtype=torch.long)
    
    def _select_validation_prompts(self, case_id: str) -> list:
        """
        Select a structured set of prompts for validation:
        1 global + up to 3 regional (different regions) + up to 2 lesion (different groups).

        Returns list of prompt entries (max 6).
        """
        prompts = self.prompts_data.get(case_id, [])
        if not prompts:
            return []

        by_type = {'global': [], 'region': [], 'lesion': []}
        for p in prompts:
            pt = p.get('prompt_type', 'global')
            if pt in by_type:
                by_type[pt].append(p)

        selected = []

        # 1 global
        if by_type['global']:
            selected.append(by_type['global'][np.random.randint(len(by_type['global']))])

        # Up to 3 regional — one per unique region (keyed by lesion_numbers)
        if by_type['region']:
            groups = {}
            for p in by_type['region']:
                key = tuple(sorted(p.get('lesion_numbers', [])))
                groups.setdefault(key, []).append(p)
            keys = list(groups.keys())
            np.random.shuffle(keys)
            for key in keys[:3]:
                g = groups[key]
                selected.append(g[np.random.randint(len(g))])

        # Up to 2 lesion — one per unique lesion group
        if by_type['lesion']:
            groups = {}
            for p in by_type['lesion']:
                key = tuple(sorted(p.get('lesion_numbers', [])))
                groups.setdefault(key, []).append(p)
            keys = list(groups.keys())
            np.random.shuffle(keys)
            for key in keys[:2]:
                g = groups[key]
                selected.append(g[np.random.randint(len(g))])

        return selected

    def _get_distance_field(self, subject, case_id: str, prompt_entry: dict):
        """Compute distance field from the cropped atlas for the prompt's target region(s).

        Computes EDT on-the-fly from the already-cropped seg_atlas in the subject.
        This ensures the distance field always matches the patch spatial dimensions.

        Returns a (1, H, W, D) float tensor or None.
        """
        if self.distance_field_weight <= 0:
            return None
        if prompt_entry.get('prompt_type') == 'global':
            return None
        if 'seg_atlas' not in subject or 'seg_cc' not in subject:
            return None

        atlas = subject['seg_atlas'].data.long()[0].numpy()
        seg_cc = subject['seg_cc'].data.long()[0].numpy()
        lesion_numbers = prompt_entry.get('lesion_numbers', [])

        # Find atlas regions containing the target lesions
        target_regions = set()
        for ln in lesion_numbers:
            mask = seg_cc == ln
            if mask.any():
                region_labels = atlas[mask]
                region_labels = region_labels[region_labels > 0]
                target_regions.update(region_labels.tolist())

        if not target_regions:
            return None

        # Build region mask and compute EDT on the cropped patch
        region_mask = np.isin(atlas, list(target_regions))
        if region_mask.all() or not region_mask.any():
            return None

        from scipy.ndimage import distance_transform_edt
        distance = distance_transform_edt(~region_mask).astype(np.float32)

        # Normalize: sigmoid -> 0 inside region, ~1 far away
        sigma = self.distance_field_sigma
        normalized = 1.0 / (1.0 + np.exp(-(distance - 3 * sigma) / sigma))

        return torch.from_numpy(normalized).unsqueeze(0).float()

    def _build_text_prompted_batch(
        self,
        subject,
        label_data: torch.Tensor,
        case_id: str,
        prompt_entry: Optional[dict] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Build a batch dict for text-prompted mode.

        If prompt_entry is None, selects a random prompt (training).
        If provided, uses that prompt directly (validation).

        Args:
            subject: TorchIO Subject with 'image' and optionally 'seg_cc'.
            label_data: (C, H, W, D) label tensor (standard multi-class or cc labels).
            case_id: Case identifier.
            prompt_entry: Pre-selected prompt dict (optional, for validation).

        Returns:
            Batch dict with 'image', 'label' (binary), 'text_embedding', 'case_id',
            or None if no valid prompt exists for this case.
        """
        if prompt_entry is None:
            prompts = self.prompts_data.get(case_id, [])
            if not prompts:
                return None
            prompt_entry = prompts[np.random.randint(len(prompts))]

        prompt_text = prompt_entry['prompt']
        lesion_numbers = prompt_entry.get('lesion_numbers', None)
        label_value = prompt_entry.get('label_value', None)

        # Look up precomputed embedding
        embedding = self.precomputed_embeddings.get(prompt_text, None)
        if embedding is None:
            return None

        # Extract binary mask
        if 'seg_cc' in subject:
            # Instance labels available: extract mask for specific lesion(s)
            cc_data = subject['seg_cc'].data.long()
            if lesion_numbers is not None:
                binary_mask = torch.zeros_like(cc_data[0:1], dtype=torch.bool)
                # Filter to CC labels that actually exist in the preprocessed
                # volume — tiny components (1-few voxels) can vanish during
                # nearest-neighbor resampling to isotropic spacing
                available_labels = set(cc_data.unique().tolist())
                for ln in lesion_numbers:
                    if ln in available_labels:
                        binary_mask = binary_mask | (cc_data[0:1] == ln)
                binary_mask = binary_mask.float()
            elif label_value is not None:
                binary_mask = (cc_data == label_value).float()
            else:
                # Fallback: use all foreground
                binary_mask = (label_data > 0).float()
        else:
            # No instance labels, use standard label
            if label_value is not None:
                binary_mask = (label_data == label_value).float()
            else:
                binary_mask = (label_data > 0).float()

        # Shape: (1, H, W, D) -> (N=1, H, W, D) for consistency with model output
        if binary_mask.dim() == 4:
            binary_mask = binary_mask[0:1]  # Keep first channel only

        # Compute distance field for spatial prior loss
        distance_field = self._get_distance_field(subject, case_id, prompt_entry)

        batch = {
            'image': subject['image'].data,
            'label': binary_mask,  # (1, H, W, D) binary mask
            'text_embedding': embedding,  # (embedding_dim,) tensor
            'case_id': case_id,
        }
        if self.distance_field_weight > 0:
            # Always include distance_field when enabled so collation works;
            # zeros = no spatial penalty (e.g., global prompts or missing atlas)
            if distance_field is not None:
                batch['distance_field'] = distance_field
            else:
                batch['distance_field'] = torch.zeros_like(binary_mask)
        return batch

    def __len__(self) -> int:
        """
        Return the 'virtual' size of the dataset.
        
        Total Length = (Number of Cases) * (patches_per_volume)
        
        Note: For IterableDataset, __len__ is informational and used by progress
        bars and learning rate schedulers. The actual number of samples yielded
        depends on the iteration logic.
        """
        if self._val_tp_total_prompts is not None:
            return self._val_tp_total_prompts
        return len(self.case_ids) * self.patches_per_volume
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch number to ensure different shuffling each epoch.
        
        IMPORTANT: This must be called before each epoch when using this dataset
        for training. Without calling set_epoch(), the same shuffle order will be
        used every epoch, causing the same samples to appear at the same iterations.
        
        This is similar to torch.utils.data.distributed.DistributedSampler.set_epoch().
        
        Args:
            epoch: The current epoch number (0-indexed)
        
        Example:
            for epoch in range(num_epochs):
                train_loader.dataset.set_epoch(epoch)
                for batch in train_loader:
                    ...
        """
        self._epoch = epoch
    
    def _get_worker_info(self) -> Tuple[int, int]:
        """
        Get worker ID and total number of workers.
        
        Returns:
            Tuple of (worker_id, num_workers).
            If running in main process (no workers), returns (0, 1).
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Running in main process (num_workers=0)
            return 0, 1
        return worker_info.id, worker_info.num_workers
    
    def _partition_case_ids(self, worker_id: int, num_workers: int) -> List[str]:
        """
        Partition case IDs so each worker gets a mutually exclusive subset.
        
        Worker i takes files [i, i+num_workers, i+2*num_workers, ...].
        This ensures no overlap between workers.
        
        Args:
            worker_id: Current worker's ID (0-indexed)
            num_workers: Total number of workers
        
        Returns:
            List of case IDs assigned to this worker
        """
        # Interleaved partitioning: worker i gets indices [i, i+n, i+2n, ...]
        return self.case_ids[worker_id::num_workers]
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over volumes and yield multiple patches per volume.
        
        This method:
        1. Detects worker info and partitions case IDs
        2. Shuffles the assigned case IDs (for epoch randomness)
        3. For each case: loads volume, extracts patches_per_volume patches
        4. Each patch is produced by applying transforms (with random crop) to the volume
        
        Yields:
            Dictionary with 'image', 'label', and 'case_id' for each patch
        """
        worker_id, num_workers = self._get_worker_info()
        
        # Partition case IDs for this worker
        worker_case_ids = self._partition_case_ids(worker_id, num_workers)
        
        if len(worker_case_ids) == 0:
            # This worker has no data to process
            return
        
        # Shuffle case IDs for this epoch (training only)
        # Use worker-specific seed derived from base seed, worker_id, AND epoch for reproducibility
        # while ensuring different workers have different shuffles AND different epochs have different shuffles
        if self.is_training:
            if self.seed is not None:
                # Combine base seed with worker_id AND epoch for reproducibility
                # IMPORTANT: self._epoch must be updated via set_epoch() before each epoch
                # to ensure different shuffling. Without this, the same order is used every epoch.
                epoch_seed = self.seed + worker_id + self._epoch * 1000
                rng = np.random.default_rng(epoch_seed)
            else:
                # Use a random seed (different each iteration through the dataset)
                # Note: This is non-reproducible but ensures variety
                rng = np.random.default_rng()
            
            # Shuffle the case IDs in-place for this epoch
            shuffled_case_ids = list(worker_case_ids)
            rng.shuffle(shuffled_case_ids)
        else:
            # Validation: Keep deterministic order
            shuffled_case_ids = list(worker_case_ids)

        current_case = shuffled_case_ids[0]
        image, label, affine, foreground_coords, valid_coords, weight_map, seg_cc, seg_atlas = self._load_volume(current_case)
        
        # Iterate over volumes
        for i, case_id in enumerate(shuffled_case_ids):
            # Determine next case for prefetching
            next_case_id = shuffled_case_ids[i+1] if i + 1 < len(shuffled_case_ids) else None
            
            if next_case_id:
                # Start loading the next volume in a separate thread
                future = self.executor.submit(self._load_volume, next_case_id)

            subject_dict = {
                'image': tio.ScalarImage(tensor=image, affine=affine),
                'label': tio.LabelMap(tensor=label, affine=affine),
                'case_id': case_id
            }
            
            # Add weight map as ScalarImage so it gets spatial transforms
            if weight_map is not None:
                subject_dict['weight_map'] = tio.ScalarImage(tensor=weight_map, affine=affine)

            # Add connected component labels for text-prompted mode
            if seg_cc is not None:
                subject_dict['seg_cc'] = tio.LabelMap(tensor=seg_cc, affine=affine)

            # Add atlas labels for distance field loss
            if seg_atlas is not None:
                subject_dict['seg_atlas'] = tio.LabelMap(tensor=seg_atlas, affine=affine)

            if not self.is_training:
                case_hash = int(hash(case_id) % 1e8) 
                torch.manual_seed(self.seed + case_hash)
                np.random.seed(self.seed + case_hash)
            
            if foreground_coords is not None:
                subject_dict['foreground_coords'] = foreground_coords

            if valid_coords is not None:
                subject_dict['valid_coords'] = valid_coords
                
            base_subject = tio.Subject(**subject_dict)

            # Training: Random variation in patches per volume
            # Validation: Fixed number of patches
            if self.is_training:
                patches_per_volume = np.random.randint(self.patches_per_volume // 2, self.patches_per_volume + 1)
            else:
                patches_per_volume = self.patches_per_volume

            # Validation text-prompted: select prompts first, then crop toward
            # each prompt's specific lesions so the patch always contains them.
            if not self.is_training and self.text_prompted and case_id in self.prompts_data:
                val_prompts = self._select_validation_prompts(case_id)
                original_fg_coords = base_subject.get('foreground_coords', None)

                for prompt_entry in val_prompts:
                    # Compute prompt-specific foreground coords from full-volume seg_cc
                    lesion_numbers = prompt_entry.get('lesion_numbers', None)
                    if lesion_numbers and 'seg_cc' in base_subject:
                        cc_full = base_subject['seg_cc'].data.long()[0]  # (H, W, D)
                        prompt_fg = torch.zeros_like(cc_full, dtype=torch.bool)
                        for ln in lesion_numbers:
                            prompt_fg = prompt_fg | (cc_full == ln)
                        coords = torch.nonzero(prompt_fg, as_tuple=False).numpy()
                        if len(coords) > 0:
                            base_subject['foreground_coords'] = coords

                    subject = self.final_crop(base_subject)
                    label_data = subject['label'].data.long()

                    batch = self._build_text_prompted_batch(
                        subject, label_data, case_id, prompt_entry
                    )
                    if batch is not None:
                        yield batch

                # Restore original foreground coords for safety
                if original_fg_coords is not None:
                    base_subject['foreground_coords'] = original_fg_coords
            else:
                # Training path (all modes) and validation non-text-prompted
                for patch_idx in range(patches_per_volume):
                    # Training: Apply oversized crop, then transforms, then final crop
                    # Validation: Apply transforms (if any), then final crop
                    if self.is_training and self.oversized_crop is not None:
                        subject = self.oversized_crop(base_subject)
                    else:
                        subject = base_subject

                    if self.transforms is not None and len(self.transforms.transforms) > 0:
                        subject = self.transforms(subject)

                    # Apply final crop to target patch size
                    subject = self.final_crop(subject)

                    # Ensure label is long type
                    label_data = subject['label'].data.long()

                    # Compute valid bounds from label (where label != -1)
                    # This is much faster than computing from full mask tensor in loss
                    if self.compute_valid_bounds:
                        valid_bounds = self._compute_valid_bounds(label_data)
                    else:
                        valid_bounds = None

                    # Optional: print ratio of foreground voxels in the label (if verbose)
                    if self.verbose:
                        foreground_ratio = (label_data > 0).float().mean().item()
                        print(f"Foreground voxel ratio: {foreground_ratio:.4f} for patch {patch_idx} of case {case_id}", flush=True)

                    # Yield the patch
                    if self.text_prompted and case_id in self.prompts_data:
                        # Text-prompted mode: select prompt, extract binary mask, look up embedding
                        batch = self._build_text_prompted_batch(
                            subject, label_data, case_id
                        )
                        if batch is not None:
                            yield batch
                    else:
                        batch = {
                            'image': subject['image'].data,
                            'label': label_data,
                            'case_id': case_id,
                        }

                        # Include weight map if available
                        if 'weight_map' in subject:
                            batch['weight_map'] = subject['weight_map'].data

                        # Only include valid_bounds if computed (to avoid collation issues with None)
                        if self.compute_valid_bounds:
                            batch['valid_bounds'] = valid_bounds

                        yield batch

            # Wait for the next volume to finish loading (if prefetch was started)
            if next_case_id:
                image, label, affine, foreground_coords, valid_coords, weight_map, seg_cc, seg_atlas = future.result()
    
    def _load_volume(self, case_id: str):
        """
        Load volume and coordinate arrays.

        Returns:
            Tuple of (image, label, affine, foreground_coords, valid_coords, weight_map, seg_cc, seg_atlas)
            seg_cc/seg_atlas are None unless text_prompted mode and the key exists in data.
        """
        # Check cache first
        if self._volume_cache is not None and case_id in self._volume_cache:
            print("Using cached volume for case:", case_id)
            return self._volume_cache[case_id]
        
        foreground_coords = None
        valid_coords = None
        weight_map = None
        seg_cc = None
        
        if self.use_preprocessed:
            # Load preprocessed numpy arrays (.npy or .npz)
            if self.use_compressed:
                data_file = self.data_path / f"{case_id}.npz"
                weight_map_file = self.data_path / f"{case_id}_weight.npz"
                
                if not data_file.exists():
                    raise FileNotFoundError(f"Missing compressed file for case {case_id}: {data_file}")
                
                data = np.load(data_file)

                image = torch.from_numpy(data['data']).float()
                label = torch.from_numpy(data['seg']).long()

                # Load instance labels (connected components) for text-prompted mode
                if self.text_prompted and 'seg_cc' in data:
                    seg_cc = torch.from_numpy(data['seg_cc']).long()
                else:
                    seg_cc = None

                # Load atlas labels for distance field loss
                if self.text_prompted and 'seg_atlas' in data:
                    seg_atlas = torch.from_numpy(data['seg_atlas']).long()
                else:
                    seg_atlas = None

                # Load weight map if available
                if weight_map_file.exists():
                    weight_data = np.load(weight_map_file)
                    weight_map = torch.from_numpy(weight_data['weights']).float().unsqueeze(0)
            else:
                raise NotImplementedError("Uncompressed preprocessed loading not implemented.")
            
            # Load foreground coordinates if available (works for both compressed and uncompressed modes)
            foreground_coords_file = self.data_path / f"{case_id}_foreground_coords.npy"
            if foreground_coords_file.exists():
                foreground_coords = np.load(foreground_coords_file)

            # Load valid coordinates if available
            valid_coords_file = self.data_path / f"{case_id}_valid_coords.npy"
            if valid_coords_file.exists():
                valid_coords = np.load(valid_coords_file)
            
            # For preprocessed data, use the affine matrix created in __init__
            # This is a diagonal matrix with the target spacing values
            affine = self.preprocessed_affine
        else:
            raise NotImplementedError("Native loading of raw data not implemented.")
        
        # Cache if enabled
        if self._volume_cache is not None:
            self._volume_cache[case_id] = (image, label, affine, foreground_coords, valid_coords, weight_map, seg_cc, seg_atlas)

        return image, label, affine, foreground_coords, valid_coords, weight_map, seg_cc, seg_atlas


def create_data_loaders(
    data_manager: DataManager,
    fold: int,
    config,
    transforms_train=None,
    transforms_val=None,
    use_preprocessed: bool = False,
    train_on_all: bool = False,
    precomputed_embeddings: Optional[Dict[str, 'torch.Tensor']] = None,
    prompts_data: Optional[Dict[str, list]] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create native PyTorch data loaders using IterableDataset.
    
    This implementation uses IterableDataset for efficient I/O:
    - Each worker loads a volume once and extracts multiple patches
    - Workers process mutually exclusive subsets of volumes
    - DataLoader interleaving provides batch diversity across patients
    
    IMPORTANT: For training, you MUST call train_loader.dataset.set_epoch(epoch)
    before each epoch to ensure different shuffling. Without this, the same
    samples will appear in the same order every epoch, causing periodic loss curves.
    
    Args:
        data_manager: DataManager instance
        fold: Cross-validation fold
        config: Configuration object
        transforms_train: Training transforms (TorchIO Compose, should include random crop)
        transforms_val: Validation transforms (TorchIO Compose, no random crop)
        use_preprocessed: If True, skip preprocessing (resampling, normalization)
    
    Returns:
        Tuple of (train_loader, val_loader)
    
    Important Notes on Batch Diversity:
    ------------------------------------
    With IterableDataset, patches from one volume are yielded consecutively by a 
    single worker. However, when num_workers > 1, the DataLoader interleaves 
    samples from different workers (each processing different patients).
    
    For optimal batch diversity, ensure: num_workers >= batch_size
    
    Example: batch_size=4, num_workers=4, patches_per_volume=16
    - Each batch contains patches from 4 different patients (one per worker)
    - This provides excellent diversity without needing a shuffle buffer
    
    If num_workers < batch_size, some batches may contain multiple patches from
    the same patient. Consider using a shuffle buffer in that case (not implemented
    here but could be added using a buffer that collects patches before yielding).
    """
    
    # Get case IDs
    if train_on_all:
        train_ids = data_manager.get_all_case_ids()
        val_ids = []
        print(f"Train on ALL data: {len(train_ids)} total cases (no validation)")
    else:
        train_ids, val_ids = data_manager.get_fold_case_ids(fold)
        print(f"Fold {fold}: {len(train_ids)} training cases, {len(val_ids)} validation cases")
    
    # Get patches_per_volume from config, default to 1
    patches_per_volume_train = getattr(config.training, 'patches_per_volume', 1)
    patches_per_volume_val = getattr(config.training, 'patches_per_volume_val', 8)
    
    # Create preprocessing pipeline (only if not using preprocessed data)
    if not use_preprocessed:
        preprocessing = tio.Compose([
            tio.CopyAffine('image', copy=False),
            tio.Resample(
                target=config.data.target_spacing,
                image_interpolation='linear',
                label_interpolation='nearest',
                copy=False
            ),
        ], copy=False)
        normalization = tio.ZNormalization(masking_method=None, copy=False)

        train_transforms_list = [normalization, preprocessing]
        val_transforms_list = [normalization, preprocessing]
        
        # Combine preprocessing with transforms
        if transforms_train is not None:
            train_transforms_list.append(transforms_train)

        train_transforms_full = tio.Compose(train_transforms_list, copy=False)
        
        if transforms_val is not None:
            val_transforms_list.append(transforms_val)

        val_transforms_full = tio.Compose(val_transforms_list, copy=False)
    else:
        # Data already preprocessed
        train_transforms_full = transforms_train
        val_transforms_full = transforms_val
    
    # Text-prompted mode detection
    text_prompted = hasattr(config, 'text_prompted') and config.text_prompted.enabled

    # Create training dataset (IterableDataset with multiple patches per volume)
    # Transforms handle: resampling, normalization, oversized cropping, augmentation
    # Final center crop to patch_size is done in the dataset after transforms
    train_dataset = PatchDataset(
        case_ids=train_ids,
        data_path=data_manager.data_path,
        patch_size=config.data.patch_size,
        patches_per_volume=patches_per_volume_train,
        is_training=True,
        use_preprocessed=use_preprocessed,
        use_compressed=getattr(config.data, 'use_compressed', False),
        transforms=train_transforms_full,
        cache_volumes=False,
        target_spacing=config.data.target_spacing if use_preprocessed else None,
        seed=getattr(config, 'seed', None),
        oversize_factor=1.25,
        foreground_oversample_percent=config.training.oversample_foreground_percent,
        verbose=False,
        compute_valid_bounds=getattr(config.training, 'betti_weight', 0.0) > 0,
        text_prompted=text_prompted,
        precomputed_embeddings=precomputed_embeddings,
        prompts_data=prompts_data,
        distance_field_weight=getattr(config.text_prompted, 'distance_field_weight', 0.0) if hasattr(config, 'text_prompted') else 0.0,
        distance_field_sigma=getattr(config.text_prompted, 'distance_field_sigma', 20.0) if hasattr(config, 'text_prompted') else 20.0,
    )
    
    # Create data loaders
    # Note: For IterableDataset, shuffle=False is required (shuffling is internal)
    # Note: sampler is not compatible with IterableDataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,  # IterableDataset handles shuffling internally
        # sampler is not used with IterableDataset
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        #prefetch_factor=4 if config.num_workers > 0 else None,
    )

    if val_ids:
        # Create validation dataset (unified PatchDataset in validation mode)
        # Extracts multiple patches per volume for better validation coverage
        val_dataset = PatchDataset(
            case_ids=val_ids,
            data_path=data_manager.data_path,
            patch_size=config.data.patch_size,
            patches_per_volume=patches_per_volume_val,
            is_training=False,
            use_preprocessed=use_preprocessed,
            use_compressed=getattr(config.data, 'use_compressed', False),
            transforms=val_transforms_full,
            cache_volumes=False,
            target_spacing=config.data.target_spacing if use_preprocessed else None,
            seed=getattr(config, 'seed', None),
            foreground_oversample_percent=1.0,  # High foreground bias for validation
            verbose=False,
            compute_valid_bounds=getattr(config.training, 'betti_weight', 0.0) > 0,
            text_prompted=text_prompted,
            precomputed_embeddings=precomputed_embeddings,
            prompts_data=prompts_data,
            distance_field_weight=getattr(config.text_prompted, 'distance_field_weight', 0.0) if hasattr(config, 'text_prompted') else 0.0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=max(1, config.num_workers // 2),
            pin_memory=config.pin_memory,
            drop_last=False,
        )
    else:
        val_loader = None

    return train_loader, val_loader