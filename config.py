"""
Configuration for 3D segmentation training based on nnUNet methodology
"""
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class DataConfig:
    """Data configuration based on nnUNet preprocessing"""
    # Dataset path - interpretation depends on use_preprocessed flag:
    #   - If use_preprocessed=False: path to raw nnUNet data (imagesTr/labelsTr structure)
    #   - If use_preprocessed=True: path to preprocessed numpy arrays (.npy or .npz files)
    data_path: str = "./data"
    # Preprocessing mode
    use_preprocessed: bool = True  # If True: load .npy/.npz files, skip resampling/normalization
                                   # If False: load NIfTI from data_path, apply full preprocessing
    
    # Compression mode (only used when use_preprocessed=True)
    use_compressed: bool = True  # If True: load .npz files, If False: load .npy files
    
    # Dataset properties from nnUNet analysis
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Patch configuration from nnUNet 3d_fullres
    patch_size: Tuple[int, int, int] = (192, 192, 192)

    # Normalization (nnUNet uses ZScore with mask)
    normalization_scheme: str = "zscore"
    use_mask_for_norm: bool = True
    
    # Channel configuration
    num_input_channels: int = 1
    num_classes: int = 2  # background + foreground
    
    # Cross-validation
    num_folds: int = 3
    
    # Train on all data (no validation split)
    train_on_all: bool = False

    # Debug options
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
    batch_size: int = 6
    patches_per_volume: int = 10   # nnUNet uses 4 patches per volume for 3d_fullres
    
    # Training parameters
    max_epochs: int = 1000
    num_iterations_per_epoch: int = 250  # nnUNet uses fixed 250 iterations per epoch
    initial_lr: float = 0.001
    weight_decay: float = 5e-5
    momentum: float = 0.99
    
    # Warm restart parameters (for continuing training with extended epochs)
    warm_restart_lr_factor: float = 0.4  # Multiplier for initial LR on warm restart

    # Finetuning / transfer learning
    freeze_encoder: bool = False        # Freeze all encoder parameters (requires_grad=False)
    encoder_lr_factor: float = 1.0      # Encoder LR = initial_lr * this factor; ignored when freeze_encoder=True
    lora_enabled: bool = False           # Apply LoRA to transformer decoder attention layers
    lora_rank: int = 16                  # LoRA rank (lower = fewer params, higher = more expressive)
    lora_alpha: float = 32.0             # LoRA scaling factor (alpha/rank is the effective scale)
    lora_dropout: float = 0.0            # Dropout on LoRA adapter layers

    # Numerical stability / regularization
    logit_clamp: float = 0.0             # Clamp model output logits to [-val, val] before loss; 0=disabled
    spectral_norm: bool = False          # Apply spectral normalization to projection layers
    initial_grad_scale: float = 65536.0  # Initial GradScaler scale factor

    # Foreground oversampling (nnUNet uses 0.33)
    oversample_foreground_percent: float = 0.33  # Always sample foreground patches
    
    # Learning rate scheduling
    lr_scheduler: str = "poly"  # nnUNet uses polynomial decay
    poly_lr_pow: float = 0.9
    
    # Loss configuration
    loss_function: str = "combined"
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    cldice_alpha: float = 0.0  # Weight for clDice component (0 = disabled, 0.5 = equal weight with Dice)
    
    weight_map_scale: float = 6.0 # Scale factor for the weight maps (0 = no weighting)
    weight_map_bias: float = 0.2  # Bias added to the weights
    use_weight_map: bool = True  # If True, use per-pixel weight maps for CE loss
    
    topograph_weight: float = 0.0  # Weight for topograph loss component
    topograph_num_processes: int = 16
    topograph_thres_var: float = 0.0
    topograph_aggregation: str = "CE" 
    topograph_debug: bool = False
    topograph_error_type: str = "false_positives"

    # Betti matching loss configuration (used when loss_function contains "betti")
    betti_weight: float = 0.0  # Weight for Betti matching loss component
    betti_relative: bool = False  # Use relative homology (pads input with boundary)
    betti_cpu_batch_size: int = 16  # CPU batch size for matching computation (lower = less memory)
    betti_subsampling_size: Optional[int] = 64  # If set, subsample to this many points for matching (reduces memory)
    betti_subsampling_mode: str = "random_crop"  # Method for subsampling

    dice_plus_plus_gamma: float = 2.0  # Gamma for DSC++ loss
    tversky_beta: float = 0.5  # Beta for Tversky loss (only used when loss_function is "tversky" or "tversky_ce")

    ignore_index: int = -1  # Ignore label value in loss calculation
    
    # Softmax gradient configuration
    # If True: uses FixedGradSoftmax (hybrid gradient for Dice loss computation)
    # If False: uses standard softmax (PyTorch default)
    use_fixed_grad_softmax: bool = False
    exponential_correction: Optional[int] = 50
    
    # Deep supervision (if using MedNeXt with deep supervision)
    deep_supervision_weights: List[float] = None
    
    # Validation
    val_check_interval: int = 10  # Check validation every N epochs
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
    elastic_deform_prob: float = 0.2
    elastic_deform_max_displacement: float = 5.0  # Max displacement in voxels (TorchIO parameterization)
    elastic_deform_num_control_points: int = 7    # B-spline control grid resolution per axis
    
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
class TextPromptedConfig:
    """Configuration for text-prompted segmentation mode (VoxTell-style)"""
    enabled: bool = False

    # Text encoder
    text_encoder_model: str = "Qwen/Qwen3-Embedding-4B"
    text_embedding_dim: int = 2560  # Output dim of Qwen3-Embedding-4B
    precomputed_embeddings_path: Optional[str] = None  # Path to .pt file with precomputed embeddings

    # Transformer decoder (text-image fusion)
    query_dim: int = 2048
    transformer_num_heads: int = 8
    transformer_num_layers: int = 6
    decoder_layer: int = 4  # Which encoder stage to use as spatial context for transformer

    # VoxTell-style decoder
    num_maskformer_stages: int = 5
    num_heads: int = 32  # Fusion channels per decoder stage
    project_to_decoder_hidden_dim: int = 2048

    # Data pipeline
    prompt_csv_dir: Optional[str] = None  # Directory with per-sample CSV prompt files
    prompts_json_path: Optional[str] = None  # Path to generated prompts JSON
    instance_labels_suffix: str = "_cc"  # Suffix for connected component instance labels
    atlas_labels_suffix: str = "_atlas"  # Suffix for atlas region labels
    max_prompts_per_sample: int = 1  # Number of text prompts per training sample

    # Distance field auxiliary loss (spatial prior from atlas regions)
    distance_field_weight: float = 0.0   # Weight for distance-based spatial penalty; 0=disabled
    distance_field_sigma: float = 20.0   # Controls sigmoid falloff (voxels): 0 inside region, ~1 far away


@dataclass
class WandbConfig:
    """Weights & Biases configuration"""
    project: str = "3D-Segmentation"
    entity: Optional[str] = None  # Set to your wandb username/team
    tags: List[str] = None
    notes: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["segmentation", "3d"]


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    text_prompted: TextPromptedConfig = field(default_factory=TextPromptedConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    # General settings
    seed: int = 42
    num_workers: int = 14
    pin_memory: bool = True
    
    # GPU settings
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Output directory
    output_dir: str = "./experiments"
    
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
        
        # Validate text-prompted configuration
        if self.text_prompted.enabled:
            arch = self.model.architecture.lower()
            if arch not in ['resunet', 'residual_unet', 'residual', 'plainunet', 'plain_unet', 'plain']:
                raise ValueError(
                    f"Text-prompted mode only supports ResUNet or PlainUNet, "
                    f"got architecture='{self.model.architecture}'"
                )
            from architectures import calculate_n_stages
            n_stages = calculate_n_stages(self.data.patch_size)
            if self.text_prompted.decoder_layer >= n_stages:
                print(f"\nWARNING: decoder_layer ({self.text_prompted.decoder_layer}) >= "
                      f"n_stages ({n_stages}). Clamping to {n_stages - 1}.")
                self.text_prompted.decoder_layer = n_stages - 1

            # Text-prompted uses single-channel sigmoid output (binary per-prompt)
            if self.data.num_classes != 1:
                print(f"NOTE: Setting num_classes=1 for text-prompted mode (was {self.data.num_classes})")
                self.data.num_classes = 1

            # Disable mirror augmentation: flipping breaks location-specific prompts
            # (e.g., "left frontal" becomes right frontal after L-R flip)
            if self.augmentation.mirror_prob > 0:
                print("NOTE: Disabling mirror augmentation (incompatible with location-specific text prompts)")
                self.augmentation.mirror_prob = 0.0

        # Validate finetuning configuration
        if self.training.encoder_lr_factor < 0:
            raise ValueError(f"encoder_lr_factor must be >= 0, got {self.training.encoder_lr_factor}")
        if self.training.encoder_lr_factor == 0.0 and not self.training.freeze_encoder:
            print("NOTE: encoder_lr_factor=0.0 is equivalent to freezing — setting freeze_encoder=True")
            self.training.freeze_encoder = True
        if self.training.freeze_encoder and self.training.encoder_lr_factor != 1.0:
            print("NOTE: encoder_lr_factor is ignored when freeze_encoder=True")
        if self.training.lora_enabled and not self.text_prompted.enabled:
            raise ValueError("LoRA requires text-prompted mode (--text_prompted) since it targets the transformer decoder")

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
