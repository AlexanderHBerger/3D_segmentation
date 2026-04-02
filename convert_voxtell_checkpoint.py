#!/usr/bin/env python
"""
Convert a VoxTell checkpoint to the native checkpoint format.

Reads VoxTell's plans.json to reconstruct the architecture, loads the weights,
and saves a checkpoint with the proper Config embedded — usable for inference
and fine-tuning with our pipeline.

Usage:
    python convert_voxtell_checkpoint.py \
        --voxtell_dir /path/to/voxtell_v1.1 \
        --fold 0 \
        --output /path/to/converted_checkpoint.pth
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from config import Config
from text_prompted_model import TextPromptedModel


def _resolve_class(class_name: str):
    """Resolve a dotted class name string to the actual class object."""
    parts = class_name.rsplit(".", 1)
    module_path, cls_name = parts[0], parts[1]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def build_config_from_plans(plans: dict) -> Config:
    """Build a Config from VoxTell's plans.json."""
    cfg_3d = plans["configurations"]["3d_fullres"]
    arch_kwargs = cfg_3d["architecture"]["arch_kwargs"]
    patch_size = tuple(cfg_3d["patch_size"])

    config = Config()
    config.data.patch_size = patch_size
    config.data.target_spacing = (1.0, 1.0, 1.0)
    config.data.num_input_channels = 1
    config.data.num_classes = 2

    config.model.architecture = "ResUNet"
    config.model.in_channels = 1
    config.model.n_classes = 2
    config.model.kernel_size = 3
    config.model.deep_supervision = False

    config.text_prompted.enabled = True

    return config


def build_arch_params_from_plans(plans: dict) -> dict:
    """Build the arch_params dict that TextPromptedModel expects.

    This mimics what architectures.get_network_parameters returns, but uses
    the exact values from VoxTell's plans.json instead of our size presets.
    """
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

    cfg_3d = plans["configurations"]["3d_fullres"]
    raw_kwargs = cfg_3d["architecture"]["arch_kwargs"]
    requires_import = cfg_3d["architecture"].get("_kw_requires_import", [])

    # Resolve string class references to actual classes
    arch_kwargs = {}
    for key, value in raw_kwargs.items():
        if key in requires_import and value is not None:
            arch_kwargs[key] = _resolve_class(value)
        elif key == "norm_op_kwargs" and value is not None:
            arch_kwargs[key] = value
        elif key == "nonlin_kwargs" and value is not None:
            arch_kwargs[key] = value
        elif key == "dropout_op_kwargs":
            arch_kwargs[key] = value
        else:
            arch_kwargs[key] = value

    # Add fields expected by our pipeline
    arch_kwargs["input_channels"] = 1
    arch_kwargs["num_classes"] = 2
    arch_kwargs["deep_supervision"] = False

    return {
        "architecture_class": ResidualEncoderUNet,
        "architecture_kwargs": arch_kwargs,
    }


def convert(voxtell_dir: Path, fold: int, output_path: Path):
    plans_path = voxtell_dir / "plans.json"
    ckpt_path = voxtell_dir / f"fold_{fold}" / "checkpoint_final.pth"

    if not plans_path.exists():
        raise FileNotFoundError(f"plans.json not found at {plans_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # 1. Load plans and build config + arch params
    with open(plans_path) as f:
        plans = json.load(f)

    config = build_config_from_plans(plans)
    arch_params = build_arch_params_from_plans(plans)

    print(f"Architecture: ResidualEncoderUNet")
    print(f"  n_stages: {arch_params['architecture_kwargs']['n_stages']}")
    print(f"  features_per_stage: {arch_params['architecture_kwargs']['features_per_stage']}")
    print(f"  n_blocks_per_stage: {arch_params['architecture_kwargs']['n_blocks_per_stage']}")
    print(f"  patch_size: {config.data.patch_size}")

    # 2. Create TextPromptedModel with VoxTell's architecture
    tp = config.text_prompted
    model = TextPromptedModel(
        arch_params=arch_params,
        text_embedding_dim=tp.text_embedding_dim,
        query_dim=tp.query_dim,
        transformer_num_heads=tp.transformer_num_heads,
        transformer_num_layers=tp.transformer_num_layers,
        decoder_layer=tp.decoder_layer,
        num_maskformer_stages=tp.num_maskformer_stages,
        num_heads=tp.num_heads,
        project_to_decoder_hidden_dim=tp.project_to_decoder_hidden_dim,
        patch_size=config.data.patch_size,
        deep_supervision=config.model.deep_supervision,
    )

    # 3. Load VoxTell weights
    print(f"\nLoading VoxTell weights from {ckpt_path}")
    voxtell_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(
        voxtell_ckpt["network_weights"], strict=False
    )
    print(f"  Missing keys:    {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  First 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  First 5 unexpected: {unexpected[:5]}")

    # 4. Verify total parameter counts match
    our_params = sum(p.numel() for p in model.parameters())
    voxtell_params = sum(v.numel() for v in voxtell_ckpt["network_weights"].values())
    print(f"\n  Our model params:     {our_params:,}")
    print(f"  VoxTell weight params: {voxtell_params:,}")

    # 5. Save as native checkpoint
    native_checkpoint = {
        "epoch": 0,
        "fold": fold,
        "model_state_dict": model.state_dict(),
        "config": config,
        "best_metric": 0.0,
        "source": f"converted from VoxTell: {ckpt_path}",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(native_checkpoint, output_path)
    print(f"\nSaved native checkpoint to {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VoxTell checkpoint to native format"
    )
    parser.add_argument(
        "--voxtell_dir", type=str, required=True,
        help="Path to VoxTell model directory (containing plans.json and fold_X/)"
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Fold number (default: 0)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the converted checkpoint .pth file"
    )
    args = parser.parse_args()
    convert(Path(args.voxtell_dir), args.fold, Path(args.output))


if __name__ == "__main__":
    main()
