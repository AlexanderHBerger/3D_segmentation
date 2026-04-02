"""
LoRA (Low-Rank Adaptation) for transformer decoder attention layers.

Lightweight implementation without external dependencies (no peft/loralib).
Decomposes nn.MultiheadAttention's fused Q/K/V projection into separate
linear layers wrapped with low-rank adapters.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen nn.Linear layer.

    Freezes the original linear weights and adds trainable low-rank matrices
    A (in_features → rank) and B (rank → out_features). Output is:
        linear(x) + B(A(dropout(x))) * (alpha / rank)

    B is zero-initialized so the adapter starts as identity (no change to output).
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_A = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear.out_features, bias=False)
        # Zero-init B so adapter is identity at start
        nn.init.zeros_(self.lora_B.weight)
        # Kaiming uniform for A (default nn.Linear init)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base + lora


class LoRAMultiheadAttention(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention with LoRA adapters.

    Decomposes the fused in_proj_weight into separate Q, K, V linear layers,
    wraps each (plus out_proj) with LoRALinear, and implements multi-head
    attention manually. LoRA corrections are applied BEFORE attention computation,
    which is necessary because attention involves a non-linear softmax.

    Expects input in (seq_len, batch, embed_dim) format (not batch_first).
    Returns (output, attn_weights) matching nn.MultiheadAttention's interface.
    """

    def __init__(self, mha: nn.MultiheadAttention, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        d_model = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = d_model // mha.num_heads
        self.d_model = d_model
        has_bias = mha.in_proj_bias is not None

        # Decompose fused in_proj_weight (3*d_model, d_model) into separate Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=has_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=has_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=has_bias)

        with torch.no_grad():
            self.q_proj.weight.copy_(mha.in_proj_weight[:d_model])
            self.k_proj.weight.copy_(mha.in_proj_weight[d_model:2 * d_model])
            self.v_proj.weight.copy_(mha.in_proj_weight[2 * d_model:])
            if has_bias:
                self.q_proj.bias.copy_(mha.in_proj_bias[:d_model])
                self.k_proj.bias.copy_(mha.in_proj_bias[d_model:2 * d_model])
                self.v_proj.bias.copy_(mha.in_proj_bias[2 * d_model:])

        # Copy out_proj
        self.out_proj = nn.Linear(d_model, d_model, bias=mha.out_proj.bias is not None)
        with torch.no_grad():
            self.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(mha.out_proj.bias)

        # Wrap each projection with LoRA (freezes original weights, adds trainable A/B)
        self.q_proj = LoRALinear(self.q_proj, rank, alpha, dropout)
        self.k_proj = LoRALinear(self.k_proj, rank, alpha, dropout)
        self.v_proj = LoRALinear(self.v_proj, rank, alpha, dropout)
        self.out_proj = LoRALinear(self.out_proj, rank, alpha, dropout)

        self.attn_dropout_p = mha.dropout

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query: (seq_q, batch, d_model)
            key:   (seq_k, batch, d_model)
            value: (seq_k, batch, d_model)
            attn_mask: (seq_q, seq_k) or (batch*num_heads, seq_q, seq_k)
            key_padding_mask: (batch, seq_k)

        Returns:
            output: (seq_q, batch, d_model)
            attn_weights: (batch, seq_q, seq_k) or None
        """
        seq_q, batch, _ = query.shape
        seq_k = key.shape[0]

        # Project with LoRA-augmented Q, K, V
        q = self.q_proj(query)  # (seq_q, batch, d_model)
        k = self.k_proj(key)    # (seq_k, batch, d_model)
        v = self.v_proj(value)  # (seq_k, batch, d_model)

        # Reshape for multi-head attention: (seq, batch, d) -> (batch, num_heads, seq, head_dim)
        q = q.view(seq_q, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(seq_k, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(seq_k, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Compute attention using PyTorch's efficient SDPA
        dropout_p = self.attn_dropout_p if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
        # attn_output: (batch, num_heads, seq_q, head_dim)

        # Reshape back: (batch, num_heads, seq_q, head_dim) -> (seq_q, batch, d_model)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(seq_q, batch, self.d_model)

        # Project output with LoRA-augmented out_proj
        output = self.out_proj(attn_output)

        # Compute attention weights for return if needed (used by transformer for logging)
        # Note: scaled_dot_product_attention doesn't return weights, compute separately
        if need_weights:
            with torch.no_grad():
                scale = 1.0 / math.sqrt(self.head_dim)
                # Use first head's weights as representative (average over heads)
                attn_weights = torch.bmm(
                    q.mean(dim=1),  # (batch, seq_q, head_dim)
                    k.mean(dim=1).transpose(1, 2)  # (batch, head_dim, seq_k)
                ) * scale
                if key_padding_mask is not None:
                    attn_weights = attn_weights.masked_fill(
                        key_padding_mask.unsqueeze(1), float('-inf')
                    )
                attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            attn_weights = None

        return output, attn_weights


def apply_lora_to_transformer(
    transformer_decoder: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> Tuple[int, int]:
    """Apply LoRA to all attention layers in a TransformerDecoder.

    Replaces nn.MultiheadAttention modules with LoRAMultiheadAttention,
    and freezes all non-LoRA transformer parameters (FFN, LayerNorm).

    Args:
        transformer_decoder: TransformerDecoder module containing decoder layers
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout

    Returns:
        (frozen_params, lora_params): Count of frozen and new LoRA trainable parameters
    """
    frozen_params = 0
    lora_params = 0

    for layer in transformer_decoder.layers:
        # Replace self_attn with LoRA version
        if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, nn.MultiheadAttention):
            layer.self_attn = LoRAMultiheadAttention(layer.self_attn, rank, alpha, dropout)

        # Replace multihead_attn (cross-attention) with LoRA version
        if hasattr(layer, 'multihead_attn') and isinstance(layer.multihead_attn, nn.MultiheadAttention):
            layer.multihead_attn = LoRAMultiheadAttention(layer.multihead_attn, rank, alpha, dropout)

        # Freeze non-LoRA parameters in this layer (FFN linears, LayerNorms)
        for name, param in layer.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
                lora_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()

    # Freeze the final norm layer if it exists
    if hasattr(transformer_decoder, 'norm') and transformer_decoder.norm is not None:
        for param in transformer_decoder.norm.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

    return frozen_params, lora_params
