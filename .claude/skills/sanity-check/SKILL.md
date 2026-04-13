---
name: sanity-check
description: Runs pre-training correctness checks (imports, overfit-one-batch, label-shuffle control, zero-input, finite grads, dataloader determinism) and writes a SANITY_OK marker keyed by current git sha. Required before any sbatch training submission. Invoke when the experimenter asks for it or before a user submits a job.
---

# Sanity Check

Non-negotiable gate before submitting a training run. SGD converges in the presence of bugs — this skill's job is to catch those bugs cheaply (~30 s on GPU) before wasting SLURM hours.

## What it checks

Runs `scripts/sanity_check.py`. All checks live in one script so they stay discoverable and editable. The six checks:

1. **Imports** — `config`, `model`, `architectures`, `losses`, `data_loading_native` import without error.
2. **Overfit one batch** — train a single batch for ~80 steps; final loss must be < 30 % of initial. Catches broken gradient flow, wrong loss sign, accidentally-frozen params.
3. **Label-shuffle control** — repeat (2) with voxel-shuffled labels. Must *fail* to converge to the same level (≥ 2× real-batch final loss). Catches image/label misalignment, batchnorm-stats leakage, trivial constant output.
4. **Zero-input** — forward pass on all-zero input produces finite output.
5. **Finite gradients** — standard batch backward produces finite gradients with non-zero norm.
6. **Dataloader determinism** — two fresh dataloaders with the same seed produce the same first sample (soft-skip if preprocessed data unavailable).

## When to run

- Before every new training submission on a changed git sha.
- After any edit to `train.py`, `model.py`, `architectures.py`, `losses.py`, `data_loading_native.py`, `text_prompted_model.py`, or `transformer.py`.
- Before submitting a preempt-partition job (cannot afford to discover a bug after 2 hours on preempt).

## How to invoke — always via SLURM

The sanity-check script loads a model and runs a brief training loop → needs a GPU. **Never run it on the login node.** Dispatch via `srun` on `minilab-gpu` (interactive, blocks, streams output):

```bash
# Standard mode
srun --partition=minilab-gpu --qos=normal --gres=gpu:1 --mem=16G \
     --cpus-per-task=4 --time=00:15:00 --job-name=sanity \
     conda run -n nnunet python scripts/sanity_check.py

# Text-prompted mode
srun --partition=minilab-gpu --qos=normal --gres=gpu:1 --mem=16G \
     --cpus-per-task=4 --time=00:15:00 --job-name=sanity \
     conda run -n nnunet python scripts/sanity_check.py --text-prompted

# Single check (debugging)
srun --partition=minilab-gpu --qos=normal --gres=gpu:1 --mem=16G \
     --cpus-per-task=4 --time=00:10:00 \
     conda run -n nnunet python scripts/sanity_check.py --only overfit_one_batch
```

The script itself takes ~30 s on a L40S once it starts; srun queue time is usually seconds on minilab. If `minilab-gpu` is saturated, fall back to `preempt_gpu` (preemption grace covers the run):

```bash
srun --partition=preempt_gpu --qos=low --gres=gpu:1 --mem=16G \
     --cpus-per-task=4 --time=00:15:00 \
     /midtier/paetzollab/scratch/ahb4007/conda/envs/nnunet/bin/python \
     /midtier/paetzollab/scratch/ahb4007/3D_segmentation/scripts/sanity_check.py
```
(The preempt path requires code staged to `/midtier/…` — see the `submit-slurm` skill.)

## What the skill does

1. Identify the mode from the pending experiment (`--text-prompted` flag on the target `main.py` command).
2. Run the script. On failure, **stop** — do not continue to sbatch. Report the failing check's message verbatim.
3. On success, the script writes `.claude/SANITY_OK_<git_sha>`. The `submit-slurm` skill and the sbatch pre-hook both check for this marker.
4. If the user edits any training-affecting file after a successful run, the Stop hook invalidates the marker automatically.

## Red flags to surface even if a check technically passes

- Overfit final loss > 0.15 after 80 steps — model may not have enough capacity for the task or LR is wrong.
- Label-shuffle final loss only 2.1× real-batch loss (just above threshold) — borderline, suspect weak label-image correspondence in the batch.
- Any check took > 60 s — the data path may be the bottleneck, flag to user.

## Red lines

- Do **not** bypass the marker check to save time. The 30 seconds save multi-hour failed runs.
- Do **not** mark a run as "sanity OK" via any mechanism other than `scripts/sanity_check.py`.
- Do **not** edit the script's thresholds to make a run pass — fix the bug.
