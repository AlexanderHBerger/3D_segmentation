"""
Precompute text embeddings for all prompts in a dataset.

Loads per-case prompt JSON files from a directory, collects all unique
prompt texts, encodes them with the text encoder model, and saves as a .pt file.
The output is used during training to avoid loading the large text model.

Usage:
    python precompute_text_embeddings.py \
        --prompts_dir /path/to/prompts/ \
        --output /path/to/embeddings.pt \
        --text_encoder Qwen/Qwen3-Embedding-4B \
        --batch_size 32
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from text_embedding import TextEncoder


def load_prompts_from_dir(prompts_dir: Path) -> dict:
    """Load all per-case prompt JSON files into a single dict."""
    prompts_data = {}
    for jf in sorted(prompts_dir.glob("*.json")):
        with open(jf) as f:
            prompts_data[jf.stem] = json.load(f)
    return prompts_data


def main():
    parser = argparse.ArgumentParser(
        description="Precompute text embeddings for text-prompted segmentation"
    )
    parser.add_argument(
        '--prompts_dir', type=str, required=True,
        help='Directory containing per-case prompt JSON files (output of generate_prompts.py)'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output path for precomputed embeddings (.pt file)'
    )
    parser.add_argument(
        '--text_encoder', type=str, default='Qwen/Qwen3-Embedding-4B',
        help='HuggingFace model name for text encoding'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for encoding'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for text encoding'
    )
    args = parser.parse_args()

    # Load prompts from per-case files
    prompts_dir = Path(args.prompts_dir)
    print(f"Loading prompts from {prompts_dir}")
    prompts_data = load_prompts_from_dir(prompts_dir)

    # Collect all unique prompt texts
    unique_prompts = set()
    for case_id, prompt_list in prompts_data.items():
        for entry in prompt_list:
            unique_prompts.add(entry['prompt'])

    unique_prompts = sorted(unique_prompts)
    print(f"Found {len(unique_prompts)} unique prompts across {len(prompts_data)} cases")

    # Initialize text encoder
    print(f"Loading text encoder: {args.text_encoder}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    encoder = TextEncoder(
        model_name=args.text_encoder,
        device=device,
    )

    # Encode all prompts in batches
    print(f"Encoding prompts (batch_size={args.batch_size})...")
    all_embeddings = encoder.encode_prompts_batched(
        unique_prompts,
        batch_size=args.batch_size,
        wrap_instruction=True,
    )
    # all_embeddings: (1, N, embedding_dim) -> individual tensors
    all_embeddings = all_embeddings.squeeze(0)  # (N, embedding_dim)

    # Build prompt -> embedding dict
    embeddings_dict = {}
    for i, prompt in enumerate(unique_prompts):
        embeddings_dict[prompt] = all_embeddings[i]

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings_dict, output_path)
    print(f"Saved {len(embeddings_dict)} embeddings to {output_path}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")


if __name__ == '__main__':
    main()
