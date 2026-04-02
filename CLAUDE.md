# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Ideas & Future Work

See [IDEAS.md](IDEAS.md) for collected ideas, open questions, and future directions.

## Project Overview

3D medical image segmentation framework implementing nnUNet-style training. Supports multiple architectures (PlainUNet, ResUNet, Primus/ViT, MedNeXt) with topology-aware loss functions and **text-prompted segmentation** (VoxTell-style). Built for CT/MRI volumetric segmentation with cross-validation, W&B experiment tracking, and sliding-window inference.

## Environment

Always use the `nnunet` conda environment for running Python code:
```bash
conda run -n nnunet python <script.py>
```

**SLURM partitions:**
- `minilab-gpu` — dedicated GPU partition (L40S on ai-gpu12/13, H100 on ai-gpu14)
- `minilab-cpu` — dedicated CPU partition
- `preempt_gpu` — preemptible GPU partition with access to more GPUs across the cluster (jobs may be killed and requeued). Use `--qos=low`. **Cannot access `/ministorage/`** — only use for jobs that don't need minilab storage.
  ```bash
  sbatch --partition=preempt_gpu --qos=low --gres=gpu:1 --mem=64G ...
  # Interactive:
  srun --partition=preempt_gpu --qos=low --nodes=1 --tasks-per-node=1 --mem=64G --gres=gpu:1 --pty /usr/bin/bash -i
  ```

**Using `preempt_gpu` for training/debugging:**

Since preempt nodes cannot access `/ministorage/`, code and data must be copied to `/midtier/paetzollab/scratch/ahb4007/` first. The minilab copy is always the source of truth.

1. Make all code changes on minilab (`/ministorage/ahb/3D_segmentation/`)
2. Sync code (excluding experiments/wandb): `rsync -a --exclude='experiments/' --exclude='wandb/' --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='sbatch_debug_preempt.sbatch' --exclude='sbatch_train_preempt.sbatch' --exclude='sbatch_train_voxtell_preempt.sbatch' /ministorage/ahb/3D_segmentation/ /midtier/paetzollab/scratch/ahb4007/3D_segmentation/`
3. Copy/sync preprocessed data if changed: `rsync -a /ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/ /midtier/paetzollab/scratch/ahb4007/data/nnUNet_preprocessed/Dataset018_TextPrompted/`
4. Copy checkpoints as needed (e.g., VoxTell weights for init): `cp /ministorage/ahb/3D_segmentation/experiments/voxtell_converted/checkpoint.pth /midtier/paetzollab/scratch/ahb4007/3D_segmentation/experiments/voxtell_converted/`
5. **Always submit via sbatch script file** — `--wrap` does not work on preempt nodes (shell init issues). Preempt sbatch scripts live at `/midtier/paetzollab/scratch/ahb4007/3D_segmentation/sbatch_*_preempt.sbatch` and use `eval "$(conda shell.bash hook)"` instead of `module load`.

## Commands

```bash
# Install dependencies
conda run -n nnunet pip install -r requirements.txt

# --- Standard Segmentation ---

# Train all folds (5-fold cross-validation)
python main.py

# Train single fold
python main.py --fold 0

# Train with overrides
python main.py --fold 0 --batch_size 4 --lr 0.01 --epochs 500 --model_size M

# Debug mode (10 epochs, reduced data, visualization enabled)
python main.py --debug --max_samples 10 --visualize

# Resume training
python main.py --resume <run_id>

# Warm restart (extended training with reduced LR)
python main.py --resume <run_id> --use_new_config --warm_restart --epochs 500

# Transfer learning (load weights, train from epoch 0)
python main.py --init_checkpoint /path/to/checkpoint.pth

# --- Finetuning (transfer learning with frozen/adapted components) ---

# Freeze encoder (train only decoder/transformer/projections)
python main.py --text_prompted --init_checkpoint /path/to/voxtell.pth --freeze_encoder --fold 0

# Differential learning rate (encoder trains 10x slower)
python main.py --text_prompted --init_checkpoint /path/to/voxtell.pth --encoder_lr_factor 0.1 --fold 0

# LoRA on transformer decoder (freeze encoder + LoRA = minimal trainable params)
python main.py --text_prompted --init_checkpoint /path/to/voxtell.pth --freeze_encoder --lora --fold 0

# LoRA with custom rank
python main.py --text_prompted --init_checkpoint /path/to/voxtell.pth --lora --lora_rank 32 --fold 0

# Two-phase finetuning:
# Phase 1: Freeze encoder, train rest
python main.py --text_prompted --init_checkpoint /path/to/voxtell.pth --freeze_encoder --fold 0
# Phase 2: Unfreeze with low encoder LR (warm restart from phase 1)
python main.py --resume <phase1_run_id> --use_new_config --warm_restart --encoder_lr_factor 0.1 --epochs 200

# Train on all data (no validation split)
python main.py --train_on_all

# Hyperparameter sweep
python sweep_train.py --sweep_id <wandb_sweep_id>

# Inference
python inference.py --input_folder /path/to/images --output_folder /path/to/preds --checkpoint /path/to/checkpoint.pth --fold 0

# Evaluation
python evaluate.py --predictions /path/to/preds --labels /path/to/labels --output /path/to/output

# --- Text-Prompted Segmentation ---

# 1. Preprocess dataset (images + instance labels + atlas)
python preprocessing/preprocess_text_prompted.py \
    --raw_data_dir /path/to/Dataset018_MetastasisCollectionPrompts \
    --output_dir /path/to/preprocessed --num_workers 8

# 2. Generate prompts from CSV metadata (one JSON per case in output dir)
python generate_prompts.py \
    --csv_dir /path/to/Dataset018_MetastasisCollectionPrompts/imagesTr \
    --output /path/to/prompts/ --mode template

# 2b. Generate diverse LLM-based prompts (requires GPU, ~35h)
python generate_prompts.py \
    --csv_dir /path/to/Dataset018_MetastasisCollectionPrompts/imagesTr \
    --output /path/to/prompts/ --mode llm \
    --llm_model Qwen/Qwen2.5-7B-Instruct --meta_prompt_dir ./meta-prompts

# 3. Precompute text embeddings (requires Qwen3-Embedding-4B, ~8GB VRAM)
python precompute_text_embeddings.py \
    --prompts_dir /path/to/prompts/ \
    --output /path/to/embeddings.pt

# 4. Train text-prompted model
python main.py --text_prompted \
    --precomputed_embeddings /path/to/embeddings.pt \
    --prompts_json /path/to/prompts/ --fold 0

# Text-prompted training with VoxTell pretrained weights (transfer learning)
python main.py --text_prompted \
    --precomputed_embeddings /path/to/embeddings.pt \
    --prompts_json /path/to/prompts/ --fold 0 \
    --init_checkpoint /path/to/checkpoint.pth

# Text-prompted inference (with text encoder at inference time)
python inference.py --checkpoint /path/to/checkpoint.pth \
    --input_folder /path/to/images --output_folder /path/to/preds \
    --text_prompts "brain metastasis" "lesion in left frontal cortex"
```

## Testing

**Always run tests after making changes** to verify nothing is broken:
```bash
conda run -n nnunet python -m pytest tests/ -v
```

Tests cover the text-prompted dataloader (mask correctness, prompt selection, seg_cc preservation through transforms, output shapes, edge cases) and backward compatibility with the standard pipeline.

Tests also cover the preprocessing pipeline (orientation handling for RAS/RPS/LPS/LAS, resampling with anti-aliasing and clamping, crop-to-nonzero, ZScore normalization, label alignment, end-to-end preprocessing), localization consistency (atlas/CSV/seg_cc alignment), and prompt generation (CSV loading, FreeSurfer location name cleaning, tiny lesion filtering, fuzzy size matching).

## Architecture

### Standard Pipeline

**Data flow:** Raw NIfTI → Preprocessing (resample to isotropic, ZScore normalize with brain mask) → Patch-based DataLoader with foreground oversampling (33%/67%) → TorchIO augmentation → Model → Loss → Metrics → Checkpointing

**Model output:** `(B, num_classes, H, W, D)` multi-class logits with softmax + argmax

### Text-Prompted Pipeline

**Data flow:** Raw NIfTI → Preprocessing (resample to isotropic, crop to nonzero, global ZScore — VoxTell-compatible) → DataLoader loads instance labels (`seg_cc`), selects random prompt, extracts binary mask using fuzzy ±1 size category matching, looks up precomputed embedding → Model takes `(image, text_embedding)` → Binary per-prompt output

**Model output:** `(B, N, H, W, D)` per-prompt logits with sigmoid > 0.5, where N = number of text prompts

**Architecture (adapted from VoxTell, Rokuss et al. CVPR 2026):**
1. Encoder (ResidualEncoder or PlainConvEncoder) extracts multi-scale skip features
2. Selected encoder stage features projected to query_dim (2048)
3. Text embeddings (from Qwen3-Embedding-4B, dim=2560) projected to query_dim
4. Transformer decoder (6 layers, 8 heads, pre-norm): text queries cross-attend to spatial image features with 3D positional encoding
5. Mask embeddings projected to each decoder stage's channel dimensions
6. TextPromptedDecoder: multi-scale einsum fusion `(spatial_features, mask_embeddings) → segmentation`

**Backbone support:** ResUNet and PlainUNet only (both use compatible encoder interfaces from `dynamic_network_architectures`)

## Key Modules

- `config.py` — All configuration via dataclasses: DataConfig, ModelConfig, TrainingConfig, AugmentationConfig, TextPromptedConfig, WandbConfig. `get_config()` returns the global config.
- `main.py` — CLI entry point, parses args and dispatches to `train.py`
- `train.py` — `Trainer` class with full training loop. Detects `text_prompted` mode and passes `text_embedding` to model forward.
- `model.py` — `create_model(config)` routes to standard architecture or `TextPromptedModel` based on `config.text_prompted.enabled`
- `architectures.py` — Factory function `create_architecture()` supporting PlainUNet, ResUNet, Primus, MedNeXt with size variants S/B/M/L. `get_network_parameters()` returns encoder params reused by text-prompted model.
- `text_prompted_model.py` — `TextPromptedModel` (encoder + transformer fusion + decoder) and `TextPromptedDecoder` (einsum-based mask embedding fusion). Dynamically computes decoder spatial configs from patch_size.
- `transformer.py` — DETR-based `TransformerDecoder` and `TransformerDecoderLayer` for text-image cross-attention. Adapted from VoxTell.
- `lora.py` — LoRA (Low-Rank Adaptation) for transformer attention layers. `LoRALinear` wraps frozen `nn.Linear` with trainable low-rank A/B matrices. `LoRAMultiheadAttention` decomposes fused Q/K/V into separate LoRA-wrapped projections. `apply_lora_to_transformer()` applies LoRA to all attention layers and freezes non-LoRA transformer params.
- `text_embedding.py` — `TextEncoder` class wrapping HuggingFace models (default: Qwen3-Embedding-4B). `last_token_pool()` and `wrap_with_instruction()` utilities.
- `data_loading_native.py` — `PatchDataset` (IterableDataset) with optional text-prompted mode: loads CC instance labels, selects prompts, looks up precomputed embeddings. `DataManager` handles nnUNet directory structure.
- `losses.py` — Standard losses (Dice, CE, combined, Focal, Tversky, clDice, DSC++) plus `TextPromptedLoss` (binary BCE + Dice per prompt) and `TextPromptedDeepSupervisionLoss`.
- `transforms_torchio.py` — nnUNet-standard augmentations via TorchIO
- `topograph.py` — Topology-preserving loss (graph-based, 26-connectivity). Enabled via `training.topograph_weight > 0`
- `betti_matching_loss.py` — Betti number matching loss (requires external C++ library). Enabled via `training.betti_weight > 0`
- `metrics.py` — Dice, IoU, Sensitivity, Specificity, Hausdorff, ASD, clDice, calibration metrics
- `inference.py` — Sliding window inference (MONAI). `create_text_prompted_predictor()` wraps text embeddings in closure for MONAI compatibility.
- `evaluate.py` — Per-sample CSV + aggregated JSON metrics, with lesion size stratification
- `generate_prompts.py` — Generates prompts from Dataset018 CSV metadata. Template mode (fast) or LLM mode (diverse, uses local Qwen2.5-7B-Instruct). Groups lesions by (region, size_category), applies fuzzy ±1 category label matching, three levels: lesion (size+location), region (location only), global (all). Lesion-level prompts are skipped when all core lesions are < 0.002ml (noisy fragments). `clean_location()` converts FreeSurfer labels to readable text using a lookup table for concatenated cortical names. Meta-prompts in `meta-prompts/`.
- `precompute_text_embeddings.py` — Encodes all unique prompts with text encoder, saves as .pt dict
- `preprocessing/` — Dataset-specific preprocessing. `preprocess_text_prompted.py` handles Dataset018 with instance labels (`_cc`) and atlas labels (`_atlas`).

**External dependencies:**
- MedNeXt architecture optionally imported from `/ministorage/ahb/scratch/MedNeXt`
- VoxTell reference implementation at `/ministorage/ahb/scratch/VoxTell` (read-only reference, not imported)
- PlainUNet/ResUNet/Primus from `dynamic-network-architectures` package

## Key Design Decisions

- Standard preprocessing follows nnUNet methodology: ZScore normalization uses brain mask only, resampling to isotropic spacing before normalization
- Text-prompted preprocessing matches VoxTell: resample to isotropic (scipy.ndimage.zoom order=3 for images, order=0 for labels), crop to nonzero (with binary_fill_holes), then **global ZScore** on the cropped volume (not masked). This ensures compatibility with pretrained VoxTell weights.
- Text-prompted preprocessing includes anti-aliasing (Gaussian low-pass filter) before downsampling axes and `np.clip(image, 0, None)` after resampling to eliminate cubic interpolation undershoot at brain/background boundaries. Without the clamp, negative undershoot values would become the volume minimum after ZScore instead of the background.
- Deep supervision is supported for PlainUNet/ResUNet but not Primus
- Checkpoint serialization includes full config, optimizer, and scheduler state for exact resumability
- `--init_checkpoint` loads only model weights (via `_load_model_weights_only`) for transfer learning: tries strict loading first, falls back to non-strict if architecture differs (e.g., loading VoxTell pretrained weights). Starts a fresh training run (epoch 0, new optimizer). Converted VoxTell weights are at `experiments/voxtell_converted/checkpoint.pth`.
- Warm restart reduces initial LR by 0.4x factor
- Fold -1 is used as sentinel value for "train on all data" mode
- Text-prompted mode uses precomputed embeddings (not on-the-fly) for training speed — the Qwen3-Embedding-4B model is 4B params and too large to co-load with segmentation model
- Text-prompted model dynamically computes decoder spatial configs from `patch_size` and encoder strides (unlike VoxTell's hardcoded 192^3 shapes)
- `TextPromptedConfig.enabled` defaults to `False` — all existing code paths are unchanged when disabled
- Text-prompted output is per-prompt binary masks `(B, N, H, W, D)` with sigmoid, not multi-class softmax
- Prompt labels use **fuzzy ±1 size category matching**: a "small" prompt includes lesions in {tiny, small, medium} within the same region to reduce false negatives. 4 categories: tiny (<0.03ml), small (0.03-0.18ml), medium (0.18-1.5ml), large (>1.5ml)
- Locations "Unknown" and "CSF" are excluded from lesion-level and region-level prompts
- **Text-prompted validation** uses structured prompt sampling per volume: 1 global + up to 3 regional (different regions) + up to 2 lesion (different lesion groups) = max 6 patches per volume. Prompts are selected *before* cropping, and the crop is centered on the prompt's specific lesion foreground (from `seg_cc`), guaranteeing the target lesions are in the patch. Training still uses random prompt selection.
- **Betti number metrics** (`compute_topological`) are disabled during validation by default for performance. Can be re-enabled by setting `compute_topological=True` on the val `MetricsCalculator`.
- **Finetuning options:** `freeze_encoder` sets `requires_grad=False` on encoder params and excludes them from optimizer (saves memory). `encoder_lr_factor` creates separate optimizer param groups for differential LR. `lora_enabled` applies LoRA adapters to transformer decoder attention layers — decomposes fused Q/K/V, wraps with low-rank adapters, freezes all original transformer weights. Custom implementation in `lora.py` (no peft dependency). For `--init_checkpoint` + LoRA: weights load into original MHA first, then LoRA decomposes and copies them.

## Localization Pipeline (Dataset018 creation)

Source script: `/ministorage/ahb/scratch/create_localization_dataset.py` with helpers from `/ministorage/ahb/scratch/lesion_analysis.py`.

**Data flow:** Raw image → skull-strip → ANTs SyN registration (template→patient, inverse transform warps atlas to patient space) → `scipy.ndimage.label()` on GT segmentation → CC labels → per-lesion analysis (overlay CC with atlas, `expand_labels(distance=8)` for fuzzy region assignment) → per-case CSV metadata.

**Key details:**
- Atlas registration uses ANTs with `nearestNeighbor` interpolation for label warping
- `expand_labels(distance=8)` compensates for lesions at atlas region boundaries — ~21% of lesions use "near" modifier (no direct atlas overlap, recovered via expansion), only 0.17% remain "Unknown"
- Location determination: if lesion overlaps atlas region directly → "in", if only after expansion → "near"
- `SKIP_LOCATIONS = {"Unknown", "CSF"}` excludes 0.23% of lesions from lesion/region prompts (but they remain in global prompts)

## Dataset: Dataset018_MetastasisCollectionPrompts

Located at `/ministorage/ahb/data/nnUNet_raw/Dataset018_MetastasisCollectionPrompts`. Derived from Dataset015_MetastasisCollection with added text-prompt annotations.

**Structure:**
- `imagesTr/{case_id}_0000.nii.gz` — T1c MRI images (symlinks to Dataset015)
- `imagesTr/{case_id}.csv` — Per-lesion metadata: lesion_number, size_ml, location, location_modifier, axial_slice, bbox dims
- `labelsTr/{case_id}.nii.gz` — Binary segmentation masks (symlinks to Dataset015)
- `labelsTr/{case_id}_cc.nii.gz` — Connected component instance labels (each metastasis uniquely labeled). Missing for 34 empty-foreground cases (no lesions → no CC).
- `labelsTr/{case_id}_atlas.nii.gz` — FreeSurfer atlas-based anatomical region labels
- `splits_final.json` — Cross-validation splits (symlink to Dataset015)

**Raw data orientations:** RAS (2995 cases, 96.1%), RPS (84 Stanford, 2.7%), LPS (31 BRATS, 1.0%), LAS (6 BRATS thick-slice, 0.2%). All reoriented to RAS during preprocessing via `nib.as_closest_canonical()`. 66 unique spacings; most common: 1mm isotropic (60.4%).

**Stats:** 3,116 training cases, 614 test cases, 85 unique brain region locations, ~32,316 individual lesion annotations across 3 source datasets (BRATS, Stanford, NYU).
