---
name: commands-reference
description: Complete CLI command reference for training, inference, evaluation, and the text-prompted pipeline in this repo. Invoke when the user asks how to run something.
---

# Commands Reference

All commands assume the `nnunet` conda environment. **Never run any of these directly on the login node.** Wrap in the appropriate SLURM submission:

- Training / inference / sanity-check / anything using GPU → `submit-slurm` skill (`minilab-gpu` or `preempt_gpu`).
- Tests / preprocessing / `summarize_run.py` / quick Python → `srun --partition=minilab-cpu conda run -n nnunet <cmd>` (or `sbatch` for long jobs).

The bare `python main.py …` forms below show the *payload* — always prefix with the right `srun`/`sbatch` wrapper, or invoke the `submit-slurm` skill to do it for you.

## Setup

```bash
conda run -n nnunet pip install -r requirements.txt
```

## Standard segmentation

```bash
# Train all folds (5-fold CV)
python main.py

# Single fold
python main.py --fold 0

# With overrides
python main.py --fold 0 --batch_size 4 --lr 0.01 --epochs 500 --model_size M

# Debug (10 epochs, reduced data, visualization)
python main.py --debug --max_samples 10 --visualize

# Resume
python main.py --resume <run_id>

# Warm restart (extended training, reduced LR)
python main.py --resume <run_id> --use_new_config --warm_restart --epochs 500

# Transfer learning (load weights, start from epoch 0)
python main.py --init_checkpoint /path/to/checkpoint.pth

# Train on all data (no val split)
python main.py --train_on_all

# Hyperparameter sweep
python sweep_train.py --sweep_id <wandb_sweep_id>
```

## Finetuning modes

See the `finetuning` skill for the full matrix (freeze/unfreeze/LoRA/differential-LR).

## Inference and evaluation

```bash
python inference.py --input_folder /path/to/images \
    --output_folder /path/to/preds \
    --checkpoint /path/to/checkpoint.pth --fold 0

python evaluate.py --predictions /path/to/preds \
    --labels /path/to/labels --output /path/to/output
```

## Text-prompted pipeline

Four steps, run in order:

```bash
# 1. Preprocess (images + instance labels + atlas)
python preprocessing/preprocess_text_prompted.py \
    --raw_data_dir /path/to/Dataset018_MetastasisCollectionPrompts \
    --output_dir /path/to/preprocessed --num_workers 8

# 2. Generate prompts from CSV metadata (one JSON per case)
python generate_prompts.py \
    --csv_dir /path/to/Dataset018_MetastasisCollectionPrompts/imagesTr \
    --output /path/to/prompts/ --mode template

# 2b. Diverse LLM prompts (~35h on GPU)
python generate_prompts.py \
    --csv_dir /path/to/Dataset018_MetastasisCollectionPrompts/imagesTr \
    --output /path/to/prompts/ --mode llm \
    --llm_model Qwen/Qwen2.5-7B-Instruct --meta_prompt_dir ./meta-prompts

# 3. Precompute embeddings (Qwen3-Embedding-4B, ~8 GB VRAM)
python precompute_text_embeddings.py \
    --prompts_dir /path/to/prompts/ \
    --output /path/to/embeddings.pt

# 4. Train
python main.py --text_prompted \
    --precomputed_embeddings /path/to/embeddings.pt \
    --prompts_json /path/to/prompts/ --fold 0

# 4b. Text-prompted + VoxTell init
python main.py --text_prompted \
    --precomputed_embeddings /path/to/embeddings.pt \
    --prompts_json /path/to/prompts/ --fold 0 \
    --init_checkpoint experiments/voxtell_converted/checkpoint.pth

# Inference with text prompts
python inference.py --checkpoint /path/to/checkpoint.pth \
    --input_folder /path/to/images --output_folder /path/to/preds \
    --text_prompts "brain metastasis" "lesion in left frontal cortex"
```

## Testing

Dispatch to `minilab-cpu`:

```bash
srun --partition=minilab-cpu --qos=normal --mem=8G --cpus-per-task=4 \
     --time=00:30:00 \
     conda run -n nnunet python -m pytest tests/ -v
```

Covers the text-prompted dataloader (mask correctness, prompt selection, seg_cc preservation), preprocessing (orientations, resampling, label alignment), localization consistency (atlas/CSV/seg_cc), prompt generation (size matching, FreeSurfer cleaning), and backward compatibility with the standard pipeline.
