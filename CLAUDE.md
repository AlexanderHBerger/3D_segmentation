# CLAUDE.md

Guidance for Claude Code when working in this repo. **Short by design** — loaded every session.

## Project

3D medical image segmentation (nnUNet-style) for brain metastasis MRI. Two modes:
- **Standard**: multi-class softmax segmentation. Backbones: PlainUNet, ResUNet, Primus, MedNeXt.
- **Text-prompted** (VoxTell-style): per-prompt binary segmentation. Text embeddings from Qwen3-Embedding-4B cross-attend to encoder features in a transformer decoder. Backbones: ResUNet, PlainUNet only.

Experiment tracking: W&B. Cross-validation: 5-fold (fold `-1` = train on all).

## Architecture in 3 bullets

1. Preprocessing (resample isotropic, ZScore) → patch-based `IterableDataset` with foreground oversampling → TorchIO augmentation → model → loss → metrics → checkpointing.
2. Text-prompted path adds a transformer decoder where text queries cross-attend to image features; `TextPromptedDecoder` does multi-scale einsum fusion (spatial features × mask embeddings).
3. Configuration lives in `config.py` (dataclasses); `main.py` is the CLI entry point; `train.py` owns the training loop.

## Environment (non-negotiable)

Always use the `nnunet` conda env: `conda run -n nnunet python ...`.

## Nothing runs on the login node

The user drives Claude from a tmux session on the login node — there are **no CPU/GPU resources there**. Every non-trivial command (tests, sanity checks, preprocessing, training, evaluation, anything that imports torch, any script that takes more than a couple of seconds) **must** go through SLURM:

- **CPU work** (pytest, preprocessing, analysis scripts, `summarize_run.py`, quick python imports of heavy libs) → `minilab-cpu` partition. 3 nodes (ai-cpu07–09), ~128 CPUs each, reliably available. Use `srun --partition=minilab-cpu --mem=8G --time=00:30:00 conda run -n nnunet <cmd>` for blocking interactive work, or `sbatch` for longer/batch work.
- **GPU work** (training, sanity-check with model, inference) → `minilab-gpu` or `preempt_gpu` via the `submit-slurm` skill.
- **Allowed on login node**: reading/writing/editing files (Read, Write, Edit, Glob, Grep — pure filesystem I/O, no compute), `git`, `ls`, `cat`, `sinfo`/`squeue`/`sacct`/`scontrol`/`sbatch`/`srun`/`scancel`, `rsync` to/from local disks, invoking skills and subagents. The rule is about not running *workloads* (python/torch/pytest/preprocessing/anything that consumes CPU-seconds or RAM), not about blocking file or SLURM-control operations.

Hooks and skills enforce this for workload commands. If you're tempted to run something that will do real work on the login node, wrap it in `srun --partition=minilab-cpu`.

## Scientific-method directives

**The three-bucket rule.** An ML experiment measures three things at once: the **research idea**, the **implementation**, and the **experimental setup**. Any "this didn't work" or "this worked" statement **must** be tagged to one of these buckets with a one-sentence justification. SGD converges in the presence of bugs — do not conflate signal with correctness.

**Default to suspicion.** When analyzing results, try to falsify first. Interpret graphs and numbers adversarially. If the result confirms a hypothesis on a single seed, treat it as preliminary and say so loudly.

**Sit with ambiguity.** Do not conclude prematurely. If the data doesn't discriminate between two explanations, say the data doesn't discriminate — propose the minimal experiment that would.

**Separate experimenter from analyst.** Never analyze results in the same subagent that proposed the hypothesis. Use the `analyst` subagent, and pass it only the *measurable question* + artifact paths + the `summarize_run.py` digest — never the researcher's narrative.

**Token pressure.** When main-context usage exceeds ~60%, stop reading new large files yourself. Delegate deep reads to a subagent, or ask the user to start a fresh session.

## Lab-notebook protocol

- `@LAB_NOTEBOOK.md` — append-only structured log of in-flight / findings / failed runs. Write via the `lab-notebook` skill only, which gates writes against aggregation rules.
- `@IDEAS.md` — standing hypotheses and parked questions. Rarely updated.

## SLURM

For any `sbatch` submission, invoke the `submit-slurm` skill. **Do not hand-write sbatch scripts.** The skill owns partition selection (`minilab-gpu` vs `preempt_gpu`), data staging to `/midtier/paetzollab/scratch/ahb4007/` when using preempt nodes, requeue configuration, and existing-script reuse.

## Skill index

On-demand references. Invoke by name when the task matches:

- `commands-reference` — full CLI catalog for training, inference, evaluation, text-prompted pipeline.
- `architecture` — deep-dive into modules, data flow, loss functions, design decisions.
- `dataset018` — Dataset018 structure, orientations, stats.
- `finetuning` — freeze/differential-LR/LoRA/two-phase modes.
- `localization` — atlas registration pipeline used to build Dataset018.
- `submit-slurm` — cluster-aware SLURM submission (see above).
- `sanity-check` — pre-training correctness checks. Required before any training submission.
- `run-experiment` — full experiment cadence: propose → confirm → sanity-check → submit → intermediate analysis → analyze → log.
- `analyze-run` — post-hoc run analysis via `summarize_run.py` + `analyst` subagent.
- `triage-failure` — three-bucket classifier when results surprise you.
- `lab-notebook` — gated writes to `LAB_NOTEBOOK.md`.

## Subagents

- `researcher` — propose falsifiable hypothesis + experiment design.
- `implementer` — make code changes, run tests.
- `experimenter` — run & monitor jobs; cancels non-promising runs via mid-run analyst check-ins (with user confirmation).
- `analyst` — adversarial review, receives only digest + artifact paths.

All four default to opus. Delegate reads and deep work to them to keep the main context clean.

## Testing

Run tests after code changes: `conda run -n nnunet python -m pytest tests/ -v`.
