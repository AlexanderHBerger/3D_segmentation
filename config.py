"""
Configuration file for MedNeXt training based on nnUNet learnings
"""
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class DataConfig:
    """Data configuration based on nnUNet preprocessing"""
    # Dataset path - interpretation depends on use_preprocessed flag:
    #   - If use_preprocessed=False: path to raw nnUNet data (imagesTr/labelsTr structure)
    #   - If use_preprocessed=True: path to preprocessed numpy arrays (.npy files)
    data_path: str = "/ministorage/ahb/data/nnUNet_preprocessed/Dataset017_SmallBrats_Fast"
    
    # Preprocessing mode
    use_preprocessed: bool = True  # If True: load .npy files, skip resampling/normalization
                                   # If False: load NIfTI from data_path, apply full preprocessing
    
    # Dataset properties from nnUNet analysis
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Patch configuration from nnUNet 3d_fullres
    #patch_size: Tuple[int, int, int] = (160, 192, 160)
    patch_size: Tuple[int, int, int] = (80, 96, 80)
    #patch_size : Tuple[int, int, int] = (60, 72, 60)  # Reduced for memory constraints

    # Normalization (nnUNet uses ZScore with mask)
    normalization_scheme: str = "zscore"
    use_mask_for_norm: bool = True
    
    # Channel configuration
    num_input_channels: int = 1  # T1c only
    num_classes: int = 2  # background + metastasis
    
    # Cross-validation
    num_folds: int = 1
    
    # Debug options (only used when use_preprocessed=False)
    brats_only: bool = False
    max_samples: Optional[int] = None


@dataclass 
class ModelConfig:
    """Model configuration"""
    # Architecture selection
    # Options: "PlainUNet", "ResUNet", "Primus", "MedNeXt"
    architecture: str = "ResUNet"
    
    # Input/Output configuration
    in_channels: int = 1
    n_classes: int = 2
    
    # Model size - Options: "S", "B", "M", "L"
    # S: Small - Faster training, lower memory, good for quick experiments
    # B: Base - Balanced performance, similar to nnUNet defaults
    # M: Medium - More capacity, better performance on complex tasks
    # L: Large - Maximum capacity, highest memory requirements
    model_size: str = "B"
    
    # Kernel size - for CNN architectures (PlainUNet, ResUNet, MedNeXt)
    # Common values: 3 (default), 5 (larger receptive field)
    # Not used for Primus (transformer-based)
    kernel_size: int = 3
    
    # Deep supervision - improves gradient flow and convergence
    # Recommended: True for ResUNet/PlainUNet, False for Primus
    deep_supervision: bool = False
    
    # Primus-specific: Patch embedding size for vision transformer
    # Typical values: (8, 8, 8), (16, 16, 16), (32, 32, 32)
    # Smaller patches = more tokens = higher memory but better detail
    # Only used when architecture="Primus"
    # Set to None for adaptive patch size based on input dimensions
    primus_patch_embed_size: Optional[Tuple[int, int, int]] = (4,4,4)


@dataclass
class TrainingConfig:
    """Training configuration based on nnUNet setup"""
    # Conservative batch size for memory constraints (can increase if more GPU memory available)
    batch_size: int = 16
    
    # Training parameters
    max_epochs: int = 400
    num_iterations_per_epoch: int = 250  # nnUNet uses fixed 250 iterations per epoch
    initial_lr: float = 0.008
    weight_decay: float = 2e-5
    momentum: float = 0.98
    
    # Foreground oversampling (nnUNet uses 0.33)
    oversample_foreground_percent: float = 0.33
    
    # Learning rate scheduling
    lr_scheduler: str = "poly"  # nnUNet uses polynomial decay
    poly_lr_pow: float = 0.9
    
    # Loss configuration
    loss_function: str = "tversky"  # Dice + CrossEntropy like nnUNet
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    dice_plus_plus_gamma: float = 2.0  # Gamma for DSC++ loss
    tversky_beta: float = 0.5  # Beta for Tversky loss (only used when loss_function is "tversky" or "tversky_ce")
    ignore_index: int = -1  # Ignore label value in loss calculation
    
    # Softmax gradient configuration
    # If True: uses FixedGradSoftmax (hybrid gradient for Dice loss computation)
    # If False: uses standard softmax (PyTorch default)
    use_fixed_grad_softmax: bool = True
    exponential_correction: Optional[int] = 50
    
    # Deep supervision (if using MedNeXt with deep supervision)
    deep_supervision_weights: List[float] = None
    
    # Validation
    val_check_interval: int = 20  # Check validation every N epochs
    patience: int = 200  # Early stopping patience
    
    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "val_dice"
    monitor_mode: str = "max"


@dataclass
class AugmentationConfig:
    """Data augmentation configuration based on nnUNet"""
    # Spatial transformations - nnUNet uses p_rotation=0.2, p_scaling=0.2
    rotation_prob: float = 0.2
    rotation_x: Tuple[float, float] = (-30, 30)
    rotation_y: Tuple[float, float] = (-30, 30)
    rotation_z: Tuple[float, float] = (-30, 30)

    scaling_prob: float = 0.2
    scale_range: Tuple[float, float] = (0.7, 1.4)
    
    # nnUNet sets p_elastic_deform=0 (disabled in SpatialTransform)
    elastic_deform_prob: float = 0.0
    elastic_deform_alpha: Tuple[float, float] = (0., 900.)
    elastic_deform_sigma: Tuple[float, float] = (9., 13.)
    
    # Intensity transformations - nnUNet exact specification
    # Gamma with invert: p=0.1
    gamma_prob: float = 0.1
    gamma_range: Tuple[float, float] = (0.7, 1.5)
    gamma_retain_stats: bool = True
    
    # Gamma without invert: p=0.3
    gamma_no_invert_prob: float = 0.3

    # Gaussian Noise: p=0.1
    gaussian_noise_prob: float = 0.1
    gaussian_noise_variance: Tuple[float, float] = (0., 0.1)
    
    # Gaussian Blur: p=0.2
    gaussian_blur_prob: float = 0.2
    gaussian_blur_sigma: Tuple[float, float] = (0.5, 1.0)
    
    # Brightness (Multiplicative): p=0.15
    brightness_prob: float = 0.15
    brightness_range: Tuple[float, float] = (0.75, 1.25)
    
    # Contrast: p=0.15
    contrast_prob: float = 0.15
    contrast_range: Tuple[float, float] = (0.75, 1.25)
    contrast_preserve_range: bool = True
    
    # Simulate Low Resolution: p=0.25
    simulate_low_res_prob: float = 0.25
    low_res_scale_range: Tuple[float, float] = (0.5, 1.0)
    
    # Mirror augmentation - nnUNet uses p=0.5 per axis
    mirror_prob: float = 0.5
    mirror_axes: Tuple[int, ...] = (0, 1, 2)


@dataclass
class WandbConfig:
    """Weights & Biases configuration"""
    project: str = "Metastases Segmentation"
    entity: str = None  # Set to your wandb username/team
    tags: List[str] = None
    notes: str = "Training for brain metastasis segmentation based on nnUNet learnings"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["mednext", "brain", "metastasis", "segmentation", "medical"]


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    wandb: WandbConfig = WandbConfig()
    
    # General settings
    seed: int = 42
    num_workers: int = 8
    pin_memory: bool = True
    
    # GPU settings
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Output directory
    output_dir: str = "/ministorage/ahb/scratch/segmentation_model/experiments"
    
    def __post_init__(self):
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate deep supervision for Primus architecture
        if self.model.architecture.lower() == "primus" and self.model.deep_supervision:
            print("\n" + "="*80)
            print("WARNING: Primus architecture does not support deep supervision.")
            print("Automatically setting deep_supervision to False.")
            print("="*80 + "\n")
            self.model.deep_supervision = False

        # If fixed grad softmax is not used, ensure exponential correction is None
        if not self.training.use_fixed_grad_softmax:
            if self.training.exponential_correction is not None:
                print("\n" + "="*80)
                print("WARNING: FixedGradSoftmax is disabled, setting exponential_correction to None.")
                print("="*80 + "\n")
                self.training.exponential_correction = None
        
        # Validate exponential_correction for dice_plus_plus with fixed_grad_softmax
        # dice_plus_plus can work without fixed_grad_softmax (normal behavior)
        # But when both are combined, exponential_correction must be >= 20
        if (self.training.loss_function in ["dice_plus_plus", "dice_plus_plus_ce"] 
            and self.training.use_fixed_grad_softmax):
            if self.training.exponential_correction is None:
                print("\n" + "="*80)
                print("WARNING: dice_plus_plus with use_fixed_grad_softmax requires exponential_correction >= 20.")
                print("Setting exponential_correction to 50 (default for dice_plus_plus_correction).")
                print("="*80 + "\n")
                self.training.exponential_correction = 50
            elif self.training.exponential_correction < 20:
                print("\n" + "="*80)
                print(f"WARNING: dice_plus_plus with use_fixed_grad_softmax requires exponential_correction >= 20.")
                print(f"Current value: {self.training.exponential_correction}. Setting to 50.")
                print("="*80 + "\n")
                self.training.exponential_correction = 50
        
        # Warn if tversky_beta is set to non-default but loss is not tversky
        if self.training.tversky_beta != 0.5:
            if self.training.loss_function not in ["tversky", "tversky_ce"]:
                print("\n" + "="*80)
                print(f"WARNING: tversky_beta={self.training.tversky_beta} is set but loss_function is '{self.training.loss_function}'.")
                print("tversky_beta is only used when loss_function is 'tversky' or 'tversky_ce'. Ignoring tversky_beta.")
                print("="*80 + "\n")


def get_config() -> Config:
    """Get default configuration"""
    return Config()
