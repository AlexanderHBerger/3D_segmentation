---
name: architecture
description: Deep dive into model architectures, data flow, loss functions, key modules, and design decisions. Invoke when asked about internals, how a component works, or before modifying model/training code.
---

# Architecture

## Standard pipeline

**Data flow:** Raw NIfTI → preprocessing (resample to isotropic, ZScore with brain mask) → patch-based DataLoader with foreground oversampling (33%/67%) → TorchIO augmentation → model → loss → metrics → checkpointing.

**Model output:** `(B, num_classes, H, W, D)` multi-class logits → softmax + argmax.

## Text-prompted pipeline

**Data flow:** Raw NIfTI → preprocessing (resample isotropic, crop to nonzero, **global** ZScore — VoxTell-compatible) → DataLoader loads instance labels (`seg_cc`), selects random prompt, extracts binary mask via fuzzy ±1 size category matching, looks up precomputed embedding → model takes `(image, text_embedding)` → binary per-prompt output.

**Model output:** `(B, N, H, W, D)` per-prompt logits with sigmoid > 0.5. N = number of text prompts.

**Architecture (adapted from VoxTell, Rokuss et al. CVPR 2026):**
1. Encoder (ResidualEncoder or PlainConvEncoder) extracts multi-scale skip features.
2. Selected encoder stage features projected to `query_dim` (2048).
3. Text embeddings (Qwen3-Embedding-4B, dim=2560) projected to `query_dim`.
4. Transformer decoder (6 layers, 8 heads, pre-norm): text queries cross-attend to spatial image features with 3D positional encoding.
5. Mask embeddings projected to each decoder stage's channel dimensions.
6. `TextPromptedDecoder`: multi-scale einsum fusion `(spatial_features, mask_embeddings) → segmentation`.

**Backbone support:** ResUNet and PlainUNet only (compatible encoder interfaces from `dynamic_network_architectures`).

## Key modules

- `config.py` — all configuration via dataclasses: `DataConfig`, `ModelConfig`, `TrainingConfig`, `AugmentationConfig`, `TextPromptedConfig`, `WandbConfig`. `get_config()` returns the global config.
- `main.py` — CLI entry point, parses args, dispatches to `train.py`.
- `train.py` — `Trainer` class with full training loop. Detects `text_prompted` mode and passes `text_embedding` to model forward.
- `model.py` — `create_model(config)` routes to standard or `TextPromptedModel` based on `config.text_prompted.enabled`.
- `architectures.py` — `create_architecture()` factory: PlainUNet, ResUNet, Primus, MedNeXt with size variants S/B/M/L. `get_network_parameters()` returns encoder params reused by text-prompted model.
- `text_prompted_model.py` — `TextPromptedModel` (encoder + transformer + decoder) and `TextPromptedDecoder` (einsum mask-embedding fusion). Dynamically computes decoder spatial configs from patch_size.
- `transformer.py` — DETR-based `TransformerDecoder` and `TransformerDecoderLayer` for text-image cross-attention. Adapted from VoxTell.
- `lora.py` — LoRA for transformer attention. `LoRALinear` wraps frozen `nn.Linear`; `LoRAMultiheadAttention` decomposes fused Q/K/V into separate LoRA-wrapped projections. `apply_lora_to_transformer()` applies LoRA to all attention layers and freezes non-LoRA transformer params.
- `text_embedding.py` — `TextEncoder` wraps HuggingFace models (default Qwen3-Embedding-4B). `last_token_pool()`, `wrap_with_instruction()` utilities.
- `data_loading_native.py` — `PatchDataset` (IterableDataset) with optional text-prompted mode: loads CC instance labels, selects prompts, looks up precomputed embeddings. `DataManager` handles nnUNet directory structure.
- `losses.py` — Standard (Dice, CE, combined, Focal, Tversky, clDice, DSC++) plus `TextPromptedLoss` (BCE + Dice per prompt) and `TextPromptedDeepSupervisionLoss`.
- `transforms_torchio.py` — nnUNet-standard augmentations via TorchIO.
- `topograph.py` — topology-preserving loss (graph-based, 26-connectivity). Enabled via `training.topograph_weight > 0`.
- `betti_matching_loss.py` — Betti number matching loss (requires external C++ lib). Enabled via `training.betti_weight > 0`.
- `metrics.py` — Dice, IoU, Sensitivity, Specificity, Hausdorff, ASD, clDice, calibration.
- `inference.py` — sliding window (MONAI). `create_text_prompted_predictor()` wraps text embeddings in closure for MONAI compatibility.
- `evaluate.py` — per-sample CSV + aggregated JSON metrics, lesion size stratification.
- `generate_prompts.py` — prompts from Dataset018 CSV metadata. Template (fast) or LLM (Qwen2.5-7B). Groups by (region, size_category), fuzzy ±1 matching, three levels: lesion / region / global. Lesion-level skipped when all core lesions < 0.002 ml. `clean_location()` maps FreeSurfer labels via lookup. Meta-prompts in `meta-prompts/`.
- `precompute_text_embeddings.py` — encodes unique prompts, saves as `.pt` dict.
- `preprocessing/` — dataset-specific. `preprocess_text_prompted.py` handles Dataset018 with instance labels (`_cc`) and atlas labels (`_atlas`).

## External dependencies

- MedNeXt optionally imported from `/ministorage/ahb/scratch/MedNeXt`.
- VoxTell reference at `/ministorage/ahb/scratch/VoxTell` (read-only, not imported).
- PlainUNet/ResUNet/Primus from `dynamic-network-architectures` package.

## Key design decisions

- Standard preprocessing follows nnUNet: ZScore uses brain mask only, resample isotropic before normalize.
- Text-prompted preprocessing matches VoxTell: resample isotropic (scipy.ndimage.zoom order=3 images, order=0 labels), crop to nonzero (with `binary_fill_holes`), then **global** ZScore on the cropped volume (not masked). Ensures compatibility with pretrained VoxTell weights.
- Text-prompted preprocessing includes anti-aliasing (Gaussian low-pass) before downsampling axes and `np.clip(image, 0, None)` after resampling to eliminate cubic-interp undershoot at brain/background boundaries. Without the clamp, negative undershoot values become the volume minimum after ZScore.
- Deep supervision supported for PlainUNet/ResUNet, not Primus.
- Checkpoint serialization includes full config, optimizer, scheduler state for exact resumability.
- `--init_checkpoint` loads only model weights (via `_load_model_weights_only`) for transfer learning: tries strict loading first, falls back to non-strict if architecture differs. Starts a fresh training run (epoch 0, new optimizer). Converted VoxTell weights at `experiments/voxtell_converted/checkpoint.pth`.
- Warm restart reduces initial LR by 0.4x.
- Fold -1 = "train on all data" sentinel.
- Text-prompted uses **precomputed embeddings**, not on-the-fly — Qwen3-Embedding-4B (4B params) is too large to co-load with the segmentation model.
- Text-prompted dynamically computes decoder spatial configs from `patch_size` and encoder strides (unlike VoxTell's hardcoded 192³).
- `TextPromptedConfig.enabled` defaults to `False` — all existing code paths unchanged when disabled.
- Text-prompted output is per-prompt binary `(B, N, H, W, D)` with sigmoid, not multi-class softmax.
- Prompt labels use **fuzzy ±1 size category matching**: a "small" prompt includes lesions in {tiny, small, medium} in the same region, reducing false negatives. 4 categories: tiny (<0.03ml), small (0.03–0.18ml), medium (0.18–1.5ml), large (>1.5ml).
- Locations `Unknown` and `CSF` excluded from lesion/region prompts.
- **Text-prompted validation**: structured prompt sampling — 1 global + up to 3 regional (different regions) + up to 2 lesion (different groups) = max 6 patches/volume. Prompts selected *before* cropping; crop centered on the prompt's specific lesion foreground (from `seg_cc`), guaranteeing targets are in patch. Training uses random selection.
- **Betti metrics** (`compute_topological`) disabled during validation by default for performance. Re-enable by setting `compute_topological=True` on the val `MetricsCalculator`.

See the `finetuning` skill for freeze/LoRA/differential-LR specifics.
