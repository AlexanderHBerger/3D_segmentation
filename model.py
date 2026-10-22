"""Network factory: nnU-Net style PlainUNet / ResUNet and Primus from dynamic_network_architectures."""
from typing import List

import torch.nn as nn
from dynamic_network_architectures.architectures.primus import PrimusB, PrimusL, PrimusM, PrimusS
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import (convert_dim_to_conv_op,
                                                                  get_matching_instancenorm)

# size -> (base features, max features, blocks per stage)
PLAIN_UNET = {
    "S": (16, 256, 2),
    "B": (32, 320, 2),
    "M": (32, 512, 3),
    "L": (48, 512, 3),
}
RES_UNET = {
    "S": (16, 256, (1, 2, 3, 4, 4, 4, 4)),
    "B": (32, 320, (1, 3, 4, 6, 6, 6, 6)),
    "M": (32, 512, (1, 3, 4, 6, 8, 8, 8)),
    "L": (48, 512, (2, 4, 6, 8, 10, 10, 10)),
}
PRIMUS = {"S": PrimusS, "B": PrimusB, "M": PrimusM, "L": PrimusL}


def calculate_n_stages(patch_size, min_feature_map_size: int = 4) -> int:
    """Halve every axis until the smallest one would drop below the minimum size."""
    n_pool = []
    for size in patch_size:
        n, s = 0, size
        while s >= 2 * min_feature_map_size:
            s /= 2
            n += 1
        n_pool.append(n)
    return max(3, min(7, min(n_pool) + 1))


def plan_strides(patch_size, spacing=None, min_feature_map_size: int = 4,
                 max_n_stages: int = 7) -> List[List[int]]:
    """Per-stage strides. Without a spacing every axis is downsampled at every stage. With
    a spacing an axis is only downsampled while it is within a factor 2 of the finest axis
    (as in nnU-Net), so coarse axes wait until the fine ones have caught up."""
    n_dim = len(patch_size)
    if spacing is None:
        n_stages = calculate_n_stages(patch_size, min_feature_map_size)
        return [[1] * n_dim] + [[2] * n_dim] * (n_stages - 1)

    spacing = [float(s) for s in spacing]
    size = [int(s) for s in patch_size]
    strides = [[1] * n_dim]
    while len(strides) < max_n_stages:
        finest = min(spacing)
        axes = [i for i in range(n_dim)
                if spacing[i] / finest < 2.0 and size[i] >= 2 * min_feature_map_size]
        if not axes:
            break
        strides.append([2 if i in axes else 1 for i in range(n_dim)])
        for i in axes:
            spacing[i] *= 2.0
            size[i] //= 2
    if len(strides) < 3:
        raise ValueError(f"patch {tuple(patch_size)} at spacing {tuple(spacing)} allows only "
                         f"{len(strides)} stages")
    for axis in range(n_dim):
        total = 1
        for s in strides:
            total *= s[axis]
        if patch_size[axis] % total:
            raise ValueError(f"patch axis {axis} ({patch_size[axis]}) is not divisible by its "
                             f"total stride {total}")
    return strides


def create_model(config) -> nn.Module:
    arch = config.model.architecture.lower()
    size = config.model.model_size.upper()
    patch_size = tuple(config.data.patch_size)
    in_ch, n_cls = config.data.num_input_channels, config.data.num_classes

    if arch == "primus":
        if len(patch_size) != 3:
            raise ValueError("Primus is 3D only")
        model = PRIMUS[size](input_channels=in_ch, output_channels=n_cls, input_shape=patch_size,
                             patch_embed_size=tuple(config.model.primus_patch_embed_size))
    elif arch in ("resunet", "plainunet"):
        strides = plan_strides(patch_size, config.model.arch_spacing)
        n_stages, n_dim = len(strides), len(patch_size)
        conv_op = convert_dim_to_conv_op(n_dim)
        common = dict(
            input_channels=in_ch, n_stages=n_stages, conv_op=conv_op,
            kernel_sizes=[[config.model.kernel_size] * n_dim] * n_stages, strides=strides,
            num_classes=n_cls, conv_bias=True, norm_op=get_matching_instancenorm(conv_op),
            norm_op_kwargs={"eps": 1e-5, "affine": True}, dropout_op=None, dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=False)
        if arch == "resunet":
            base, top, blocks = RES_UNET[size]
            model = ResidualEncoderUNet(
                features_per_stage=[min(base * 2 ** i, top) for i in range(n_stages)],
                n_blocks_per_stage=list(blocks[:n_stages]),
                n_conv_per_stage_decoder=[1] * (n_stages - 1), **common)
        else:
            base, top, n_conv = PLAIN_UNET[size]
            model = PlainConvUNet(
                features_per_stage=[min(base * 2 ** i, top) for i in range(n_stages)],
                n_conv_per_stage=[n_conv] * n_stages,
                n_conv_per_stage_decoder=[n_conv] * (n_stages - 1), **common)
    else:
        raise ValueError(f"unknown architecture {config.model.architecture!r}")

    if hasattr(model, "initialize"):
        model.apply(model.initialize)
    if arch != "primus":
        initialize_weights(model)
    return model


def initialize_weights(model: nn.Module):
    """Kaiming-normal convolutions, unit/zero norm layers."""
    def init(m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm,
                            nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(init)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
