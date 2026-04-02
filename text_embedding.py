"""
Text embedding utilities for text-prompted segmentation.

Adapted from VoxTell (Rokuss et al., CVPR 2026).
Supports encoding free-text prompts into vector representations
using large language models (default: Qwen3-Embedding-4B).
"""

from typing import List, Union

import torch
from torch import Tensor


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Extract the last non-padding token's hidden state from each sequence.

    Args:
        last_hidden_states: (B, seq_len, hidden_dim) from transformer output.
        attention_mask: (B, seq_len) binary mask (1 = real token, 0 = padding).

    Returns:
        (B, hidden_dim) embeddings from the last real token per sequence.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


def wrap_with_instruction(text_prompts: List[str]) -> List[str]:
    """
    Wrap text prompts with VoxTell's anatomical instruction template.

    This template helps the text encoder understand the domain context,
    improving embedding quality for anatomical terms.

    Args:
        text_prompts: List of raw text prompts (e.g., ["liver", "left kidney"]).

    Returns:
        List of instruction-wrapped prompts.
    """
    instruct = (
        'Given an anatomical term query, retrieve the precise '
        'anatomical entity and location it represents'
    )
    return [f'Instruct: {instruct}\nQuery: {text}' for text in text_prompts]


class TextEncoder:
    """
    Text encoder that converts free-text prompts to embedding vectors.

    Uses a frozen pretrained language model (default: Qwen3-Embedding-4B)
    with last-token pooling and instruction wrapping.

    Args:
        model_name: HuggingFace model identifier for the text encoder.
        max_length: Maximum token length for text inputs.
        device: Device to use for encoding.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        max_length: int = 8192,
        device: torch.device = torch.device('cuda')
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.max_length = max_length
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side='left'
        )
        self.model = AutoModel.from_pretrained(model_name).eval()

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension of the text encoder."""
        return self.model.config.hidden_size

    @torch.inference_mode()
    def encode_prompts(
        self,
        text_prompts: Union[str, List[str]],
        wrap_instruction: bool = True,
    ) -> Tensor:
        """
        Encode text prompts into embedding vectors.

        Args:
            text_prompts: Single prompt string or list of prompts.
            wrap_instruction: Whether to wrap prompts with the anatomical
                instruction template (recommended for VoxTell compatibility).

        Returns:
            Tensor of shape (1, N, embedding_dim) where N is number of prompts.
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        n_prompts = len(text_prompts)

        if wrap_instruction:
            text_prompts = wrap_with_instruction(text_prompts)

        # Move model to device for encoding
        self.model = self.model.to(self.device)

        tokens = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        outputs = self.model(**tokens)
        embeddings = last_token_pool(
            outputs.last_hidden_state, tokens['attention_mask']
        )
        embeddings = embeddings.view(1, n_prompts, -1)

        # Move model back to CPU to free GPU memory
        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()

        return embeddings

    @torch.inference_mode()
    def encode_prompts_batched(
        self,
        text_prompts: List[str],
        batch_size: int = 32,
        wrap_instruction: bool = True,
    ) -> Tensor:
        """
        Encode a large number of prompts in batches.

        Args:
            text_prompts: List of text prompts.
            batch_size: Number of prompts per batch.
            wrap_instruction: Whether to wrap with instruction template.

        Returns:
            Tensor of shape (1, N, embedding_dim).
        """
        if wrap_instruction:
            text_prompts = wrap_with_instruction(text_prompts)

        self.model = self.model.to(self.device)
        all_embeddings = []

        for i in range(0, len(text_prompts), batch_size):
            batch = text_prompts[i:i + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            outputs = self.model(**tokens)
            embeddings = last_token_pool(
                outputs.last_hidden_state, tokens['attention_mask']
            )
            all_embeddings.append(embeddings.cpu())

        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0).unsqueeze(0)  # (1, N, dim)
