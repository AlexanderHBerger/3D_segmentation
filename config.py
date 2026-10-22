"""Training configuration.

A dataset config is a Python file with a `get_config()` that starts from `Config()` and
changes what it needs. Values can be overridden on the command line with
`--set section.field=value`.
"""
import ast
import dataclasses
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class DataConfig:
    data_path: str = ""
    patch_size: Tuple[int, ...] = (128, 128, 128)
    # Only used to build the affine of the 3D spatial augmentations.
    target_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
    num_input_channels: int = 1
    num_classes: int = 2
    channel_indices: Optional[Tuple[int, ...]] = None   # subset of the stored channels
    cache_volumes: bool = False
    # Training crops are oversized by this factor before augmentation and cropped back
    # afterwards, so that rotation/scaling do not leave empty borders.
    oversize_factor: float = 1.25
    crop_align: int = 1                    # crop origins are multiples of this
    val_crop_center: str = "foreground"    # foreground | volume | random_fixed
    val_subsample_stride: int = 1
    max_samples: Optional[int] = None


@dataclass
class ModelConfig:
    architecture: str = "ResUNet"          # PlainUNet | ResUNet | Primus
    model_size: str = "B"                  # S | B | M | L
    kernel_size: int = 3
    # Physical spacing used to plan per-axis strides; None gives isotropic strides.
    arch_spacing: Optional[Tuple[float, ...]] = None
    primus_patch_embed_size: Tuple[int, int, int] = (8, 8, 8)


@dataclass
class TrainingConfig:
    batch_size: int = 2
    num_iterations_per_epoch: int = 250
    max_epochs: int = 1000
    # Stop after this many epochs without changing the learning-rate schedule.
    stop_at_epoch: Optional[int] = None
    # Keep a copy of the checkpoint after these epochs (fork points).
    milestone_epochs: Tuple[int, ...] = ()
    initial_lr: float = 0.01
    weight_decay: float = 5e-5
    momentum: float = 0.99
    poly_lr_pow: float = 0.9
    grad_accum: int = 1
    # Overlap the CPU part of the loss with the GPU work of the next micro-batch. Two
    # micro-batches are accumulated per optimizer step. Only for losses that support it.
    interleaved: bool = False
    patches_per_volume: int = 8
    patches_per_volume_val: int = 4
    oversample_foreground_percent: float = 0.33
    ignore_index: int = -1

    loss_function: str = "dice_ce"         # see losses.LOSSES
    loss_kwargs: dict = field(default_factory=dict)   # arguments of that loss
    dice_weight: float = 1.0
    ce_weight: float = 1.0

    val_check_interval: int = 10
    compute_betti_matching_metric: bool = False
    betti_metric_num_workers: Optional[int] = None
    betti_metric_max_samples: Optional[int] = None
    betti_metric_sample_stride: int = 1
    compute_calibration_metrics: bool = False


@dataclass
class AugmentationConfig:
    rotation_prob: float = 0.2
    rotation_range: Tuple[float, float] = (-30.0, 30.0)
    scaling_prob: float = 0.2
    scale_range: Tuple[float, float] = (0.7, 1.4)
    rot90_prob: float = 0.0                # exact 90 degree rotations in rot90_axes
    rot90_axes: Tuple[int, int] = (0, 1)
    gaussian_noise_prob: float = 0.1
    gaussian_noise_variance: Tuple[float, float] = (0.0, 0.1)
    gaussian_blur_prob: float = 0.2
    gaussian_blur_sigma: Tuple[float, float] = (0.5, 1.0)
    brightness_prob: float = 0.15
    brightness_range: Tuple[float, float] = (0.75, 1.25)
    contrast_prob: float = 0.15
    contrast_range: Tuple[float, float] = (0.75, 1.25)
    simulate_low_res_prob: float = 0.25
    low_res_scale_range: Tuple[float, float] = (0.5, 1.0)
    gamma_prob: float = 0.1                # with inversion
    gamma_no_invert_prob: float = 0.3
    gamma_range: Tuple[float, float] = (0.7, 1.5)
    mirror_prob: float = 0.5
    mirror_axes: Tuple[int, ...] = (0, 1, 2)


@dataclass
class WandbConfig:
    project: str = "3D-Segmentation"
    entity: Optional[str] = None
    tags: Tuple[str, ...] = ()
    run_name: Optional[str] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 42
    num_workers: int = 8
    pin_memory: bool = True
    mixed_precision: bool = True
    output_dir: str = "./experiments"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Build a config from a (possibly older) dict, ignoring unknown keys."""
        cfg = cls()
        for section in ("data", "model", "training", "augmentation", "wandb"):
            for k, v in (d.get(section) or {}).items():
                _set(getattr(cfg, section), k, v, strict=False)
        for k in ("seed", "num_workers", "pin_memory", "mixed_precision", "output_dir"):
            if k in d:
                setattr(cfg, k, d[k])
        return cfg


def _set(obj, name, value, strict=True):
    if isinstance(obj, dict):
        obj[name] = value
        return
    fields = {f.name: f for f in dataclasses.fields(obj)}
    if name not in fields:
        if strict:
            raise KeyError(f"{type(obj).__name__} has no field {name!r}")
        return
    if isinstance(value, list):
        value = tuple(value)
    setattr(obj, name, value)


def apply_overrides(config: Config, overrides) -> Config:
    """Apply `section.field=value` strings (`training.loss_kwargs.weight=0.1` sets a key of
    a dict field); values are parsed as Python literals."""
    for item in overrides or ():
        key, _, raw = item.partition("=")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        *path, name = key.strip().split(".")
        obj = config
        for p in path:
            obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
        _set(obj, name, value)
    return config


def load_config(path: str) -> Config:
    """Import a config file and call its get_config(). Its directory is put on the path,
    so configs can import each other."""
    folder = str(Path(path).resolve().parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(f"_config_{Path(path).stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()
