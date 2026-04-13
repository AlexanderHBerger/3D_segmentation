---
name: submit-slurm
description: Cluster-aware SLURM submission. Decides partition (minilab-gpu vs preempt_gpu) based on job classification and live cluster load, handles data staging for preempt nodes, configures requeue, reuses existing sbatch scripts where possible. Invoke whenever a training/sanity/sweep job needs to go to SLURM — do not hand-write sbatch.
---

# Submit SLURM

All sbatch submissions flow through this skill. The experimenter invokes it with a structured request; it returns a job id + bookkeeping.

## Critical: nothing runs on the login node

Claude runs in tmux on the login node, which has **no compute**. Every command that imports torch, runs tests, preprocesses data, or anything non-trivial **must** be submitted via SLURM. This skill owns both GPU and CPU submissions.

## What this user can actually submit (verified)

| Partition | Hardware | QoS / Account | Preemption | Storage | Walltime |
|-----------|----------|---------------|------------|---------|----------|
| `minilab-gpu` | L40S (ai-gpu12/13), H100 (ai-gpu14) | `paetzollab`, `high`/`normal` | **none** | `/ministorage/` ok | 3 d |
| `preempt_gpu` | shared cluster nodes (many GPUs) | `ALL`, `--qos=low` required | **yes, after 1 h grace** | **no `/ministorage/`** — stage to `/midtier/paetzollab/scratch/ahb4007/` | 3 d |
| `minilab-cpu` | ai-cpu07–09 (3 nodes, ~128 CPUs, ~500 GB mem each) | `paetzollab`, `high`/`normal` | none | `/ministorage/` ok | 3 d |

**Not accessible to `paetzollab` account** (do not propose):
- `sablab-gpu` (requires `sablab`/`scu` account)
- `sablab-gpu-low` (requires `radgenlab`/`lilab`/`cocolab`/`ideallab`/`sablab`/`scu`)
- `scu-gpu`, `mosadeghlab-gpu`, `cocolab-cpu`, `grosenicklab-gpu`, `yiwanglab-gpu` (wrong account)

## CPU submissions (tests, preprocessing, analysis)

Default to **`srun`** for blocking interactive CPU work (fast, returns output inline). Use `sbatch` only when the job is long and you want detachment.

**Recipes** — canonical CPU-job invocations for this repo:

```bash
# Pytest (Stop hook uses this pattern; you can invoke manually too)
srun --partition=minilab-cpu --qos=normal --mem=8G --cpus-per-task=4 \
     --time=00:30:00 \
     conda run -n nnunet python -m pytest tests/ -q -x --no-header

# Run summarize_run.py for an in-flight analysis (fast, cpu-only, reads W&B)
srun --partition=minilab-cpu --qos=normal --mem=4G --cpus-per-task=2 \
     --time=00:15:00 \
     conda run -n nnunet python scripts/summarize_run.py <run_id>

# Preprocessing (longer; use sbatch)
sbatch --partition=minilab-cpu --qos=normal --mem=64G --cpus-per-task=8 \
       --time=12:00:00 --job-name=preprocess \
       --output=/ministorage/ahb/3D_segmentation/slurm_logs/%x_%j.out \
       --wrap 'conda run -n nnunet \
               python preprocessing/preprocess_text_prompted.py --num_workers 8 ...'

# Quick one-off python check (imports, dict inspections, etc.)
srun --partition=minilab-cpu --qos=normal --mem=4G --cpus-per-task=1 \
     --time=00:05:00 \
     conda run -n nnunet python -c 'import torch; print(torch.__version__)'
```

Sizing guidance:
- Tests: 4 CPUs, 8 GB, 30 min.
- `summarize_run.py`: 2 CPUs, 4 GB, 15 min.
- Preprocessing: 8–16 CPUs, 64 GB, hours.
- Quick introspection: 1 CPU, 4 GB, 5 min.

Never run `conda run -n nnunet python …` directly on the login node. If you catch yourself about to, wrap in `srun --partition=minilab-cpu` instead.

## Request interface

The experimenter passes a dict:
```
{
  "python_command": "python main.py --text_prompted --fold 0 --init_checkpoint ...",
  "partition_hint": "auto" | "minilab-gpu" | "preempt_gpu",
  "gpu_count": 1,
  "mem_gb": 64,
  "walltime": "24:00:00",
  "requires_ministorage": true | false,
  "job_class": "sanity" | "debug" | "train" | "sweep"
}
```

Returns:
```
{
  "job_id": "...",
  "partition_used": "...",
  "script_path": "path/to/sbatch_used_or_generated.sbatch",
  "requeue_plan": "slurm_native" | "watchdog" | "none"
}
```

## Decision logic

1. **Classify the job** (if `partition_hint == "auto"`):
   - `sanity` or `debug` < 1 h → `preempt_gpu` (preemption grace covers it; frees minilab).
   - `debug` interactive → `minilab-gpu` (no staging, no interruption risk).
   - `train` (single config, many hours) → **default `minilab-gpu`**. Check live load; if saturated, propose `preempt_gpu` + auto-requeue to the user as a choice (do not silently switch).
   - `sweep` (many parallel configs) → `preempt_gpu` preferred (volume beats reliability).
2. **If `requires_ministorage=True` and target is `preempt_gpu`** → data-staging gate (see below). Either stage, or redirect to `minilab-gpu` with a note to the user.
3. **Load heuristic** (log results in the proposal):
   ```bash
   sinfo -p minilab-gpu -o "%t %D"
   squeue -p minilab-gpu -h -t RUNNING,PENDING | wc -l
   sinfo -p preempt_gpu -o "%t %D"
   ```

## Data-staging gate (preempt only)

`preempt_gpu` nodes cannot see `/ministorage/`. Before submitting:

1. Verify `/midtier/paetzollab/scratch/ahb4007/3D_segmentation/` has the current code — rsync if stale:
   ```bash
   rsync -a \
     --exclude='experiments/' --exclude='wandb/' \
     --exclude='__pycache__/' --exclude='.pytest_cache/' \
     --exclude='sbatch_debug_preempt.sbatch' \
     --exclude='sbatch_train_preempt.sbatch' \
     --exclude='sbatch_train_voxtell_preempt.sbatch' \
     /ministorage/ahb/3D_segmentation/ \
     /midtier/paetzollab/scratch/ahb4007/3D_segmentation/
   ```
2. Verify `/midtier/paetzollab/scratch/ahb4007/data/nnUNet_preprocessed/Dataset018_TextPrompted/` exists (or whichever preprocessed dataset the run needs). Rsync if stale:
   ```bash
   rsync -a /ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/ \
            /midtier/paetzollab/scratch/ahb4007/data/nnUNet_preprocessed/Dataset018_TextPrompted/
   ```
3. Copy any required init checkpoints (e.g., VoxTell weights):
   ```bash
   cp /ministorage/ahb/3D_segmentation/experiments/voxtell_converted/checkpoint.pth \
      /midtier/paetzollab/scratch/ahb4007/3D_segmentation/experiments/voxtell_converted/
   ```

**Gotcha — `--wrap` doesn't work on preempt nodes** (shell init issues). Preempt jobs must use a script file. Preempt sbatch scripts live at `/midtier/paetzollab/scratch/ahb4007/3D_segmentation/sbatch_*_preempt.sbatch` and use `eval "$(conda shell.bash hook)"` instead of `module load`.

## Reuse vs generate

1. **Inspect existing sbatch scripts** in the repo root (`sbatch_*.sbatch`). Known set (as of refactor):
   - `sbatch_1_preprocess.sbatch`, `sbatch_2_generate_prompts.sbatch`, `sbatch_2b_generate_prompts_reverse.sbatch`
   - `sbatch_3_precompute_embeddings.sbatch`
   - `sbatch_4_debug_val.sbatch`, `sbatch_4_train_text_prompted.sbatch`
   - `sbatch_debug_nan.sbatch`
   - Preempt variants under `/midtier/paetzollab/scratch/ahb4007/3D_segmentation/sbatch_*_preempt.sbatch`
2. If the experimenter's request matches an existing script's shape (same partition, same Python command structure, same resource profile) → reuse. On `minilab-gpu` you can pass the Python command via `--wrap`.
3. Generate a fresh sbatch script only when no existing one fits. Template for preempt:
   ```bash
   #!/bin/bash
   #SBATCH --partition=preempt_gpu
   #SBATCH --qos=low
   #SBATCH --gres=gpu:1
   #SBATCH --mem=64G
   #SBATCH --time=24:00:00
   #SBATCH --requeue
   #SBATCH --job-name=<name>
   #SBATCH --output=/midtier/paetzollab/scratch/ahb4007/slurm_logs/%x_%j.out

   eval "$(conda shell.bash hook)"
   conda activate nnunet
   cd /midtier/paetzollab/scratch/ahb4007/3D_segmentation
   <python_command>
   ```
4. **Pre-submit validation**: `sbatch --test-only <script>`. Block on syntax / partition / account errors.

## Preemption and requeue

- Always include `#SBATCH --requeue` on `preempt_gpu`.
- Rely on **per-epoch async `last_checkpoint.pth`** + **validation-time `best_checkpoint.pth`** for continuation. No signal trapping — training script must checkpoint every epoch asynchronously.
- On preemption: if SLURM auto-requeues, the restarted job must use `main.py --resume <run_id>` pathway. Wire this into the sbatch script via `RESUME_RUN` env var (set in the generated script only if a resume is detected, e.g., on a re-run of a known `job-name`).
- If auto-requeue is unreliable in practice (to be verified during rollout), fall back to the watchdog: a short bash script on `preempt_cpu` polling `sacct` every 5 min and re-submitting on `PREEMPTED` / `TIMEOUT` / `NODE_FAIL`. Keep orchestration out of the training job itself so it survives.

## Sanity gate

Before calling `sbatch`, verify the marker exists:
```bash
test -f .claude/SANITY_OK_$(git rev-parse HEAD)
```
If missing, invoke the `sanity-check` skill first (unless `job_class=sanity`, in which case the sanity script itself is the payload).

## After submission

1. Return `{job_id, partition_used, script_path, requeue_plan}` to the caller.
2. The experimenter is responsible for logging to `LAB_NOTEBOOK.md` via the `lab-notebook` skill and for the mid-run analyst polling loop.

## Red lines

- Never submit to a partition the user's account cannot access.
- Never submit to `preempt_gpu` without data-staging verified.
- Never silently switch partitions — surface the load/staging situation and ask.
- Never omit `--requeue` on `preempt_gpu`.
- Never hand-write sbatch outside this skill.
