"""
Text-prompted 3D segmentation model.

Adapts VoxTell's architecture (Rokuss et al., CVPR 2026) into the existing
training framework. Combines a CNN encoder (ResidualEncoder or PlainConvEncoder)
with a transformer-based text-image fusion module and a multi-scale decoder
that produces per-prompt binary segmentation masks.

Architecture:
    Image → Encoder → skip features
    Text embedding → project → transformer cross-attention with spatial features
    → mask embeddings → multi-scale einsum fusion in decoder → (B, N, H, W, D)
"""

import torch
from torch import nn
from typing import List, Tuple, Type, Union

from dynamic_network_architectures.building_blocks.helper import (
    get_matching_convtransp,
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He,
    init_last_bn_before_add_to_0,
)
from einops import rearrange, repeat
from positional_encodings.torch_encodings import PositionalEncoding3D

from transformer import TransformerDecoder, TransformerDecoderLayer


class TextPromptedModel(nn.Module):
    """
    Text-prompted 3D segmentation model with VoxTell-style architecture.

    Combines a standard encoder backbone (ResidualEncoder or PlainConvEncoder)
    with a transformer decoder for text-image fusion, and a multi-scale decoder
    that produces per-prompt binary segmentation masks via einsum operations.

    Args:
        arch_params: Architecture parameters dict from get_network_parameters().
            Must contain 'architecture_kwargs' with encoder configuration.
        text_embedding_dim: Dimension of precomputed text embeddings.
        query_dim: Internal dimension for text-image fusion.
        transformer_num_heads: Number of attention heads in transformer decoder.
        transformer_num_layers: Number of transformer decoder layers.
        decoder_layer: Which encoder stage to use as spatial context (0-indexed).
        num_maskformer_stages: Number of decoder stages with text fusion.
        num_heads: Number of fusion channels per decoder stage.
        project_to_decoder_hidden_dim: Hidden dim for projection MLPs.
        patch_size: Input patch size (D, H, W), used to compute spatial shapes.
        deep_supervision: Whether to output multi-scale predictions.
    """

    TRANSFORMER_NUM_HEADS = 8
    TRANSFORMER_NUM_LAYERS = 6

    def __init__(
        self,
        arch_params: dict,
        text_embedding_dim: int = 2560,
        query_dim: int = 2048,
        transformer_num_heads: int = 8,
        transformer_num_layers: int = 6,
        decoder_layer: int = 4,
        num_maskformer_stages: int = 5,
        num_heads: int = 32,
        project_to_decoder_hidden_dim: int = 2048,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        self.deep_supervision = deep_supervision
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.text_embedding_dim = text_embedding_dim
        self.project_to_decoder_hidden_dim = project_to_decoder_hidden_dim

        # ---- Build encoder from arch_params ----
        arch_kwargs = arch_params['architecture_kwargs']
        arch_class = arch_params['architecture_class']

        # Extract encoder-relevant params
        self.n_stages = arch_kwargs['n_stages']
        self.features_per_stage = arch_kwargs['features_per_stage']
        conv_op = arch_kwargs['conv_op']
        kernel_sizes = arch_kwargs['kernel_sizes']
        strides = arch_kwargs['strides']
        conv_bias = arch_kwargs.get('conv_bias', True)
        norm_op = arch_kwargs.get('norm_op', None)
        norm_op_kwargs = arch_kwargs.get('norm_op_kwargs', None)
        dropout_op = arch_kwargs.get('dropout_op', None)
        dropout_op_kwargs = arch_kwargs.get('dropout_op_kwargs', None)
        nonlin = arch_kwargs.get('nonlin', None)
        nonlin_kwargs = arch_kwargs.get('nonlin_kwargs', None)
        input_channels = arch_kwargs['input_channels']

        # Determine encoder type from arch_class
        from dynamic_network_architectures.architectures.unet import (
            ResidualEncoderUNet, PlainConvUNet,
        )
        if arch_class is ResidualEncoderUNet:
            n_blocks_per_stage = arch_kwargs['n_blocks_per_stage']
            self.encoder = ResidualEncoder(
                input_channels=input_channels,
                n_stages=self.n_stages,
                features_per_stage=self.features_per_stage,
                conv_op=conv_op,
                kernel_sizes=kernel_sizes,
                strides=strides,
                n_blocks_per_stage=n_blocks_per_stage,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                block=BasicBlockD,
                return_skips=True,
                disable_default_stem=False,
                stem_channels=None,
            )
        elif arch_class is PlainConvUNet:
            n_conv_per_stage = arch_kwargs['n_conv_per_stage']
            self.encoder = PlainConvEncoder(
                input_channels=input_channels,
                n_stages=self.n_stages,
                features_per_stage=self.features_per_stage,
                conv_op=conv_op,
                kernel_sizes=kernel_sizes,
                strides=strides,
                n_conv_per_stage=n_conv_per_stage,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                return_skips=True,
            )
        else:
            raise ValueError(
                f"Text-prompted mode only supports ResUNet or PlainUNet, "
                f"got {arch_class}"
            )

        # ---- Compute dynamic decoder configs from patch_size ----
        self.decoder_configs = self._compute_decoder_configs(
            patch_size, strides, self.features_per_stage
        )

        # ---- Decoder ----
        n_conv_per_stage_decoder = arch_kwargs.get(
            'n_conv_per_stage_decoder',
            [1] * (self.n_stages - 1)
        )
        self.decoder = TextPromptedDecoder(
            encoder=self.encoder,
            num_classes=1,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            num_maskformer_stages=num_maskformer_stages,
            num_heads=num_heads,
        )

        # ---- Text-image fusion components ----
        self.selected_decoder_layer = decoder_layer
        if decoder_layer >= self.n_stages:
            raise ValueError(
                f"decoder_layer ({decoder_layer}) must be < n_stages ({self.n_stages})"
            )

        selected_channels = self.features_per_stage[decoder_layer]
        selected_shape = self.decoder_configs[decoder_layer]['shape']
        h, w, d = selected_shape

        # Project bottleneck image features to query dim
        self.project_bottleneck_embed = nn.Sequential(
            nn.Linear(selected_channels, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )

        # Project text embeddings to query dim
        text_hidden_dim = 2048
        self.project_text_embed = nn.Sequential(
            nn.Linear(text_embedding_dim, text_hidden_dim),
            nn.GELU(),
            nn.Linear(text_hidden_dim, query_dim),
        )

        # Project mask embeddings to each decoder stage's channel dim
        self.project_to_decoder_channels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(query_dim, project_to_decoder_hidden_dim),
                nn.GELU(),
                nn.Linear(
                    project_to_decoder_hidden_dim,
                    cfg['channels'] * num_heads if stage_idx != 0 else cfg['channels']
                ),
            )
            for stage_idx, cfg in enumerate(
                list(self.decoder_configs.values())[:num_maskformer_stages]
            )
        ])

        # 3D positional encoding for spatial tokens
        pos_embed = PositionalEncoding3D(query_dim)(
            torch.zeros(1, h, w, d, query_dim)
        )
        pos_embed = rearrange(pos_embed, 'b h w d c -> (h w d) b c')
        self.register_buffer('pos_embed', pos_embed)

        # Transformer decoder for text-image cross-attention
        transformer_layer = TransformerDecoderLayer(
            d_model=query_dim,
            nhead=transformer_num_heads,
            normalize_before=True,
        )
        decoder_norm = nn.LayerNorm(query_dim)
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=transformer_layer,
            num_layers=transformer_num_layers,
            norm=decoder_norm,
        )

    @staticmethod
    def _compute_decoder_configs(
        patch_size: Tuple[int, ...],
        strides: list,
        features_per_stage: list,
    ) -> dict:
        """
        Compute decoder stage configurations dynamically from patch size.

        Unlike VoxTell's hardcoded DECODER_CONFIGS for 192^3, this computes
        spatial shapes for any patch size and stride configuration.

        The spatial shape at each stage is the encoder OUTPUT shape, i.e.,
        after applying that stage's stride (downsampling).

        Returns:
            Dict mapping stage_idx -> {"channels": int, "shape": tuple}
        """
        configs = {}
        spatial = list(patch_size)
        for stage_idx in range(len(features_per_stage)):
            # Apply downsampling for this stage first
            if stage_idx < len(strides):
                stride = strides[stage_idx]
                if isinstance(stride, int):
                    spatial = [s // stride for s in spatial]
                else:
                    spatial = [s // st for s, st in zip(spatial, stride)]
            # Record shape after downsampling (= encoder output at this stage)
            configs[stage_idx] = {
                'channels': features_per_stage[stage_idx],
                'shape': tuple(spatial),
            }
        return configs

    def forward(
        self,
        img: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            img: Input image (B, C, D, H, W).
            text_embedding: Precomputed text embeddings (B, N, embedding_dim)
                or (B, N, 1, embedding_dim).

        Returns:
            (B, N, D, H, W) per-prompt logits, or list for deep supervision.
        """
        # Encoder
        skips = self.encoder(img)

        # Select spatial features for transformer
        selected_feature = skips[self.selected_decoder_layer]

        # Project image features: (B, C, D, H, W) -> (H*W*D, B, query_dim)
        bottleneck_embed = rearrange(selected_feature, 'b c d h w -> b h w d c')
        bottleneck_embed = self.project_bottleneck_embed(bottleneck_embed)
        bottleneck_embed = rearrange(bottleneck_embed, 'b h w d c -> (h w d) b c')

        # Normalize text_embedding to (B, N, embedding_dim)
        if text_embedding.dim() == 2:
            text_embedding = text_embedding.unsqueeze(1)  # (B, dim) -> (B, 1, dim)
        elif text_embedding.dim() == 4:
            text_embedding = text_embedding.squeeze(2)

        # Project text: (B, N, D) -> (N, B, query_dim)
        text_embed = repeat(text_embedding, 'b n dim -> n b dim')
        text_embed = self.project_text_embed(text_embed)

        # Transformer cross-attention: text queries attend to spatial features
        mask_embedding, _ = self.transformer_decoder(
            tgt=text_embed,
            memory=bottleneck_embed,
            pos=self.pos_embed,
            memory_key_padding_mask=None,
        )
        # (N, B, query_dim) -> (B, N, query_dim)
        mask_embedding = repeat(mask_embedding, 'n b dim -> b n dim')

        # Project to decoder channel dims for each stage
        mask_embeddings = [
            projection(mask_embedding)
            for projection in self.project_to_decoder_channels
        ]

        # Generate segmentation per prompt
        num_prompts = text_embedding.shape[1]
        outs = []
        for prompt_idx in range(num_prompts):
            prompt_embeds = [m[:, prompt_idx:prompt_idx + 1] for m in mask_embeddings]
            outs.append(self.decoder(skips, prompt_embeds))

        # Concatenate across prompts for each scale
        outs = [torch.cat(scale_outs, dim=1) for scale_outs in zip(*outs)]

        if not self.deep_supervision:
            return outs[0]
        return outs

    @staticmethod
    def initialize(module):
        """He initialization with last BN set to zero."""
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class TextPromptedDecoder(nn.Module):
    """
    Decoder with multi-scale mask-embedding fusion.

    Upsamples encoder features and fuses them with text-informed mask
    embeddings at multiple scales via einsum operations.

    Adapted from VoxTell's VoxTellDecoder.
    """

    def __init__(
        self,
        encoder: Union[PlainConvEncoder, ResidualEncoder],
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        num_maskformer_stages: int = 5,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.num_heads = num_heads

        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        assert len(n_conv_per_stage) == n_stages_encoder - 1

        # Inherit hyperparameters from encoder
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias
        norm_op = encoder.norm_op
        norm_op_kwargs = encoder.norm_op_kwargs
        dropout_op = encoder.dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs
        nonlin = encoder.nonlin
        nonlin_kwargs = encoder.nonlin_kwargs

        # Build decoder stages from bottleneck to highest resolution
        stages = []
        transpconvs = []
        seg_layers = []
        for stage_idx in range(1, n_stages_encoder):
            # Add num_heads channels for stages with mask embedding fusion
            if stage_idx <= n_stages_encoder - num_maskformer_stages:
                input_features_below = encoder.output_channels[-stage_idx]
            else:
                input_features_below = encoder.output_channels[-stage_idx] + num_heads

            input_features_skip = encoder.output_channels[-(stage_idx + 1)]
            stride_for_transpconv = encoder.strides[-stage_idx]

            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip,
                stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))

            stages.append(StackedConvBlocks(
                n_conv_per_stage[stage_idx - 1], encoder.conv_op,
                2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(stage_idx + 1)], 1,
                conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, False  # nonlin_first
            ))

            seg_layers.append(encoder.conv_op(
                input_features_skip + num_heads, num_classes,
                1, 1, 0, bias=True
            ))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(
        self,
        skips: List[torch.Tensor],
        mask_embeddings: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Forward pass through decoder with mask embedding fusion.

        Args:
            skips: Encoder skip connections (computation order, last = bottleneck).
            mask_embeddings: Per-stage mask embeddings, low-res to high-res order.

        Returns:
            List of segmentation predictions, highest resolution first.
        """
        lres_input = skips[-1]
        seg_outputs = []

        # Reverse to match decoder order (bottleneck first)
        mask_embeddings = mask_embeddings[::-1]

        for stage_idx in range(len(self.stages)):
            x = self.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = self.stages[stage_idx](x)

            if stage_idx == (len(self.stages) - 1):
                # Final stage: generate segmentation via einsum
                seg_pred = torch.einsum(
                    'b c h w d, b n c -> b n h w d', x, mask_embeddings[-1]
                )
                seg_outputs.append(seg_pred)
            elif stage_idx >= len(self.stages) - len(mask_embeddings):
                # Intermediate stages: multi-head fusion
                mask_embedding = mask_embeddings.pop(0)
                batch_size, _, channels = mask_embedding.shape
                mask_embedding_reshaped = mask_embedding.view(
                    batch_size, self.num_heads, -1
                )
                fusion_features = torch.einsum(
                    'b c h w d, b n c -> b n h w d',
                    x, mask_embedding_reshaped
                )
                x = torch.cat((x, fusion_features), dim=1)
                seg_outputs.append(self.seg_layers[stage_idx](x))

            lres_input = x

        # Reverse so highest resolution is first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            return seg_outputs[:1]
        else:
            return seg_outputs
