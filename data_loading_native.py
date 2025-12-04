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
from torch.utils.data import Dataset, DataLoader, RandomSampler
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import nibabel as nib


class DataManager:
    """
    Data management using nnUNet structure.
    Handles file paths, split loading, and metadata.
    """
    
    def __init__(
        self,
        data_path: str,
        splits_file: Optional[str] = None,
        brats_only: bool = False,
        max_samples: Optional[int] = None
    ):
        self.data_path = Path(data_path)
        self.splits_file = splits_file or self.data_path / "splits_final.json"
        self.brats_only = brats_only
        self.max_samples = max_samples
        
        # Load metadata
        self.dataset_json = self._load_dataset_json()
        self.splits = self._load_splits()
        
        # Apply filtering if requested
        if self.brats_only or self.max_samples is not None:
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
        """Apply BraTS-only filter and max_samples limit"""
        for fold_idx, fold_data in enumerate(self.splits):
            # Filter training IDs
            train_ids = fold_data['train']
            if self.brats_only:
                train_ids = [case_id for case_id in train_ids if self._is_brats_case(case_id)]
            if self.max_samples is not None:
                train_ids = train_ids[:self.max_samples]
            
            # Filter validation IDs
            val_ids = fold_data['val']
            if self.brats_only:
                val_ids = [case_id for case_id in val_ids if self._is_brats_case(case_id)]
            if self.max_samples is not None:
                val_ids = val_ids[:self.max_samples]
            
            # Update splits
            self.splits[fold_idx]['train'] = train_ids
            self.splits[fold_idx]['val'] = val_ids
            
            print(f"Fold {fold_idx} after filtering: {len(train_ids)} train, {len(val_ids)} val")
    
    def _is_brats_case(self, case_id: str) -> bool:
        """Check if case ID is from BraTS dataset"""
        return case_id.startswith('BraTS') or 'BraTS' in case_id
    
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
    
    def get_fold_subjects(
        self, 
        fold: int,
        use_preprocessed: bool = False
    ) -> Tuple[List[tio.Subject], List[tio.Subject]]:
        """
        Get TorchIO subjects for training and validation.
        
        Compatibility method for TorchIO data loader interface.
        Creates TorchIO subjects from case IDs.
        
        Args:
            fold: Cross-validation fold number
            use_preprocessed: Whether to use preprocessed data (numpy arrays)
        
        Returns:
            Tuple of (train_subjects, val_subjects)
        """
        if fold >= len(self.splits):
            raise ValueError(f"Fold {fold} not available. Only {len(self.splits)} folds found.")
        
        fold_data = self.splits[fold]
        train_ids = fold_data['train']
        val_ids = fold_data['val']
        
        # Create subjects for training
        train_subjects = []
        for case_id in train_ids:
            subject = self._create_subject(case_id, use_preprocessed)
            if subject is not None:
                train_subjects.append(subject)
        
        # Create subjects for validation
        val_subjects = []
        for case_id in val_ids:
            subject = self._create_subject(case_id, use_preprocessed)
            if subject is not None:
                val_subjects.append(subject)
        
        return train_subjects, val_subjects
    
    def _create_subject(self, case_id: str, use_preprocessed: bool) -> Optional[tio.Subject]:
        """
        Create a TorchIO Subject from case ID.
        
        TorchIO automatically handles:
        - Lazy loading (files loaded only when needed)
        - Format detection (NIfTI, DICOM, numpy arrays, etc.)
        - Proper affine transformation handling
        """
        if use_preprocessed:
            # Preprocessed numpy data - data_path points to preprocessed directory
            data_file = self.data_path / f"{case_id}_data.npy"
            seg_file = self.data_path / f"{case_id}_seg.npy"
            
            if not data_file.exists() or not seg_file.exists():
                raise FileNotFoundError(f"Missing files for case {case_id}: {data_file}, {seg_file}")
            
            # Load numpy arrays
            data = np.load(data_file)
            seg = np.load(seg_file)
            
            # TorchIO can create images from numpy arrays directly
            # Just need to provide the array and an affine matrix
            # Since data is already preprocessed, use identity affine
            affine = np.eye(4)
            
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=data, affine=affine),
                label=tio.LabelMap(tensor=seg, affine=affine),
                case_id=case_id
            )
            
            return subject
        
        else:
            # Raw nnUNet data structure - data_path points to raw directory
            image_path = self.data_path / "imagesTr" / f"{case_id}_0000.nii.gz"
            label_path = self.data_path / "labelsTr" / f"{case_id}.nii.gz"
            
            if not image_path.exists():
                return None
            
            # Create TorchIO subject
            # ScalarImage: for continuous-valued images (MRI, CT)
            # LabelMap: for discrete labels (segmentation masks)
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
                case_id=case_id  # Store metadata
            )
            
            return subject


class PatchDataset(Dataset):
    """
    PyTorch Dataset for patch-based training and validation.
    
    This dataset:
    1. Loads full volumes on-the-fly (lazy loading)
    2. Applies Augmentations
    3. Applies final center crop to target patch size
    """
    
    def __init__(
        self,
        case_ids: List[str],
        data_path: Path,
        patch_size: Tuple[int, int, int],
        use_preprocessed: bool = False,
        transforms: Optional[tio.Compose] = None,
        cache_volumes: bool = False,
        target_spacing: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Args:
            case_ids: List of case identifiers
            data_path: Path to data directory
            patch_size: Final target patch size (z, y, x)
            use_preprocessed: Whether to load preprocessed numpy arrays
            transforms: TorchIO transforms to apply
            cache_volumes: Whether to cache loaded volumes in memory (high memory usage)
            target_spacing: Target spacing (z, y, x) for preprocessed data affine matrix
        """
        self.case_ids = case_ids
        self.data_path = data_path
        self.patch_size = patch_size
        self.use_preprocessed = use_preprocessed
        self.transforms = transforms
        self.cache_volumes = cache_volumes
        self.target_spacing = target_spacing
        
        # Volume cache (optional)
        self._volume_cache = {} if cache_volumes else None
        
        # Create affine matrix for preprocessed data (diagonal with spacing)
        if use_preprocessed and target_spacing is not None:
            self.preprocessed_affine = np.diag([target_spacing[0], target_spacing[1], target_spacing[2], 1.0])
        else:
            self.preprocessed_affine = np.eye(4)
        
        # Final crop transform (applied after all augmentations)
        self.final_crop = tio.CropOrPad(
            target_shape=patch_size,
            padding_mode='edge'
        )
        
        print(f"PatchDataset initialized: {len(case_ids)} cases")
    
    def __len__(self) -> int:
        return len(self.case_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a volume and apply transforms (which handle patch extraction).
        
        Args:
            idx: Index (corresponds to case_id index)
        
        Returns:
            Dictionary with 'image' and 'label' tensors
        """
        # Get case ID directly from index
        case_id = self.case_ids[idx]
        
        # Load full volume (from cache or disk) with affine matrix
        image, label, affine = self._load_volume(case_id)
        
        # Create TorchIO subject with full volume
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image, affine=affine),
            label=tio.LabelMap(tensor=label, affine=affine),
            case_id=case_id
        )
        
        # Apply transforms if provided
        # Transforms handle: resampling, normalization, oversized cropping, augmentation
        if self.transforms is not None:
            subject = self.transforms(subject)
        
        # Apply final center crop to target patch size
        # This is done for both training (after augmentation) and validation
        # For training: removes border artifacts from augmentation
        # For validation: ensures consistent evaluation size
        subject = self.final_crop(subject)
        
        # Ensure label is long type (TorchIO might convert it to float)
        label_data = subject['label'].data.long()
        
        # Return as dictionary
        return {
            'image': subject['image'].data,
            'label': label_data,
            'case_id': case_id
        }
    
    def _load_volume(self, case_id: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Load full volume (image and label) with affine matrix.
        
        Returns:
            Tuple of (image, label, affine) where:
            - image: torch tensor with shape (C, H, W, D)
            - label: torch tensor with shape (C, H, W, D)
            - affine: numpy array with shape (4, 4)
        """
        # Check cache first
        if self._volume_cache is not None and case_id in self._volume_cache:
            return self._volume_cache[case_id]
        
        if self.use_preprocessed:
            # Load preprocessed numpy arrays
            data_file = self.data_path / f"{case_id}_data.npy"
            seg_file = self.data_path / f"{case_id}_seg.npy"
            
            if not data_file.exists() or not seg_file.exists():
                raise FileNotFoundError(f"Missing files for case {case_id}: {data_file}, {seg_file}")
            
            image = torch.from_numpy(np.load(data_file)).float()
            label = torch.from_numpy(np.load(seg_file)).long()
            
            # For preprocessed data, use the affine matrix created in __init__
            # This is a diagonal matrix with the target spacing values
            affine = self.preprocessed_affine
        else:
            # Load NIfTI files
            image_path = self.data_path / "imagesTr" / f"{case_id}_0000.nii.gz"
            label_path = self.data_path / "labelsTr" / f"{case_id}.nii.gz"
            
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image file: {image_path}")
            
            # Load using nibabel
            image_nii = nib.load(str(image_path))
            label_nii = nib.load(str(label_path))
            
            # Get affine from image (both should have the same affine)
            affine = image_nii.affine
            
            # Convert to torch tensors
            # NIfTI shape is (H, W, D), we need (C, H, W, D)
            image_data = image_nii.get_fdata()
            label_data = label_nii.get_fdata()
            
            image = torch.from_numpy(image_data).float().unsqueeze(0)  # Add channel dim
            label = torch.from_numpy(label_data).long().unsqueeze(0)
        
        # Cache if enabled
        if self._volume_cache is not None:
            self._volume_cache[case_id] = (image, label, affine)
        
        return image, label, affine


def create_data_loaders(
    data_manager: DataManager,
    fold: int,
    config,
    transforms_train=None,
    transforms_val=None,
    use_preprocessed: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create native PyTorch data loaders.
    
    This implementation uses standard PyTorch Dataset and DataLoader
    without TorchIO's Queue system.
    
    Args:
        data_manager: DataManager instance
        fold: Cross-validation fold
        config: Configuration object
        transforms_train: Training transforms (TorchIO Compose)
        transforms_val: Validation transforms (TorchIO Compose)
        use_preprocessed: If True, skip preprocessing (resampling, normalization)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Get case IDs for this fold
    train_ids, val_ids = data_manager.get_fold_case_ids(fold)
    
    print(f"Fold {fold}: {len(train_ids)} training cases, {len(val_ids)} validation cases")
    
    # Create preprocessing pipeline (only if not using preprocessed data)
    if not use_preprocessed:
        preprocessing = tio.Compose([
            tio.CopyAffine('image'),
            tio.Resample(
                target=config.data.target_spacing,
                image_interpolation='linear',
                label_interpolation='nearest'
            ),
        ])
        normalization = tio.ZNormalization(masking_method=None)

        train_transforms_list = [normalization,preprocessing]
        val_transforms_list = [normalization,preprocessing]
        
        # Combine preprocessing with transforms
        if transforms_train is not None:
            train_transforms_list.append(transforms_train)

        train_transforms_full = tio.Compose(train_transforms_list)
        
        if transforms_val is not None:
            val_transforms_list.append(transforms_val)

        val_transforms_full = tio.Compose(val_transforms_list)
    else:
        # Data already preprocessed
        train_transforms_full = transforms_train
        val_transforms_full = transforms_val
    
    # Create training dataset
    # Transforms handle: resampling, normalization, oversized cropping, augmentation
    # Final center crop to patch_size is done in the dataset after transforms
    train_dataset = PatchDataset(
        case_ids=train_ids,
        data_path=data_manager.data_path,
        patch_size=config.data.patch_size,
        use_preprocessed=use_preprocessed,
        transforms=train_transforms_full,
        cache_volumes=False,  # Set to True if you have enough memory
        target_spacing=config.data.target_spacing if use_preprocessed else None,
    )
    
    # Create validation dataset
    # Transforms handle: resampling, normalization
    # Final center crop to patch_size is done in the dataset after transforms
    val_dataset = PatchDataset(
        case_ids=val_ids,
        data_path=data_manager.data_path,
        patch_size=config.data.patch_size,
        use_preprocessed=use_preprocessed,
        transforms=val_transforms_full,
        cache_volumes=False,
        target_spacing=config.data.target_spacing if use_preprocessed else None,
    )

    train_sampler = RandomSampler(
        train_dataset, 
        replacement=True, 
        num_samples=config.training.num_iterations_per_epoch * config.training.batch_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=False, # Shuffling handled by RandomSampler
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0,  # Keep workers alive between epochs
        #prefetch_factor=4 if config.num_workers > 0 else 0,  # Prefetch batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,  # Same batch size as training
        shuffle=False,
        num_workers=config.num_workers // 2,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    
    return train_loader, val_loader


def create_inference_loader(
    image_paths: List[str],
    config,
    transforms=None
) -> DataLoader:
    """
    Create data loader for inference.
    
    Args:
        image_paths: List of paths to input images
        config: Configuration object
        transforms: Optional transforms for inference
    
    Returns:
        DataLoader for inference
    """
    # Create case IDs from paths
    case_ids = [Path(p).stem.replace('_0000', '') for p in image_paths]
    
    # Create temporary data manager (inference mode)
    class InferenceDataset(Dataset):
        def __init__(self, image_paths, transforms):
            self.image_paths = image_paths
            self.transforms = transforms
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            case_id = Path(image_path).stem.replace('_0000', '')
            
            # Load image using TorchIO
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                case_id=case_id
            )
            
            # Apply transforms
            if self.transforms is not None:
                subject = self.transforms(subject)
            
            return {
                'image': subject['image'].data,
                'case_id': case_id,
                'affine': subject['image'].affine
            }
    
    # Preprocessing pipeline
    preprocessing = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(
            target=config.data.target_spacing,
            image_interpolation='linear'
        ),
    ])
    
    normalization = tio.ZNormalization(masking_method=None)
    
    if transforms is not None:
        full_transforms = tio.Compose([normalization, preprocessing, transforms])
    else:
        full_transforms = tio.Compose([normalization, preprocessing])
    
    dataset = InferenceDataset(image_paths, full_transforms)
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return loader


# Helper function for sliding window inference
def sliding_window_inference(
    model,
    image: torch.Tensor,
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
    device: torch.device = torch.device('cuda')
) -> torch.Tensor:
    """
    Perform sliding window inference on full volume.
    
    Args:
        model: Trained model
        image: Input image tensor [C, H, W, D]
        patch_size: Size of patches to extract
        overlap: Overlap between patches (0.0 to 1.0)
        device: Device to run inference on
    
    Returns:
        Predicted segmentation [H, W, D]
    """
    from monai.inferers import sliding_window_inference as monai_sliding_window
    
    # Add batch dimension if needed
    if image.ndim == 3:
        image = image.unsqueeze(0)  # Add channel dimension
    if image.ndim == 4:
        image = image.unsqueeze(0)  # Add batch dimension
    
    # Use MONAI's optimized sliding window
    with torch.no_grad():
        output = monai_sliding_window(
            inputs=image.to(device),
            roi_size=patch_size,
            sw_batch_size=4,  # Process 4 patches at a time
            predictor=model,
            overlap=overlap,
            mode='gaussian',  # Gaussian weighting for smooth blending
            device=device
        )
    
    # Remove batch dimension and get class prediction
    output = output.squeeze(0)  # Remove batch
    if output.shape[0] > 1:  # Multi-class
        output = torch.argmax(output, dim=0)
    else:  # Binary
        output = (output.squeeze(0) > 0.5).long()
    
    return output
