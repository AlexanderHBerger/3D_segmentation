# CLAUDE.md

Guidance for Claude Code when working in this repo. **Short by design** — loaded every session.

## Project

3D medical image segmentation (nnUNet-style) for brain metastasis MRI. Two modes:
- **Standard**: multi-class softmax segmentation. Backbones: PlainUNet, ResUNet, Primus, MedNeXt. Default dataset: `Dataset015_MetastasisCollection`.
- **Text-prompted** (VoxTell-style): per-prompt binary segmentation. Text embeddings from Qwen3-Embedding-4B cross-attend to encoder features in a transformer decoder. Backbones: ResUNet, PlainUNet only. Default dataset: `Dataset018_MetastasisCollectionPrompts` (derived from 015 via symlinks; adds per-lesion CSVs and atlas labels). **Always use 018 for text-prompted experiments unless the proposal explicitly justifies otherwise.**

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

## Orchestrator scope (hard boundary)

The main Claude session is the **orchestrator** — it routes, not researches. To keep the main context lean, the orchestrator reads only a closed list of files directly; every other read **MUST** be delegated via the `Task` tool to an `Explore` subagent or a specialist (researcher / implementer / experimenter / analyst).

**Orchestrator MAY read/grep/glob directly:**
- `CLAUDE.md`, `@LAB_NOTEBOOK.md`, `@IDEAS.md`, `README_CLAUDE.md`, plan files under `~/.claude/plans/`.
- `.claude/**` — skill `SKILL.md` files, agent role cards, hooks, settings (needed for routing + config work).
- Digest files (`/tmp/digest_*.md`, `.claude/digest_*.md`) — already size-bounded by `summarize_run.py`.
- SLURM status command output (`squeue`, `sinfo`, `sacct`, `scontrol`) — naturally small.

**Orchestrator MUST delegate (no direct read/grep/glob):**
- Source code (`*.py`, `*.sbatch`, anything in the repo outside `.claude/`).
- W&B logs or raw training output beyond the digest.
- Multi-file investigations ("how is X wired?", "where is Y defined?").
- Anything in `slurm_logs/`.
- Any file the orchestrator did not already open earlier in this session.

Brief the subagent with a specific question + artifact paths, and work from the returned summary — raw files do not enter the orchestrator's context. The orchestrator's job is routing, user-gate decisions, and writing lab-notebook entries; everything else is delegated.

**Why**: every direct file read accretes context irreversibly. Specialist subagents have their own disposable contexts for research work, so defer reading until a specialist owns the question.

## Lab-notebook protocol

- `@LAB_NOTEBOOK.md` — append-only structured log of in-flight / findings / failed runs. Write via the `lab-notebook` skill only, which gates writes against aggregation rules.
- `@IDEAS.md` — standing hypotheses and parked questions. Rarely updated.

## Git-flow (applies to every experiment / feature)

- New idea or feature → new branch off `main`, named `experiment/<slug>` or `feature/<slug>`. Do all experimentation on that branch.
- Every code change (`.py`, `.sbatch`, config defaults) on the branch → a commit with a descriptive message. Autonomous-fix commits are prefixed `autonomous fix:`.
- Config-only changes (CLI flags in the proposal's `main.py` invocation, no file edits) → log under the active `## In-flight` entry in `LAB_NOTEBOOK.md`, **do not commit**. The `implementer` agent enforces this policy.
- On analyst `accept` → `gh pr create` from the branch, with the digest + W&B link in the PR body. User reviews and merges.
- On `reject` or `inconclusive` → branch stays local, Failed/inconclusive LAB_NOTEBOOK entry is the audit trail. No push, no PR.
- `main` stays clean — no experimental code lands until the analyst has accepted and the user has merged.

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

### Reuse an existing agent before spawning a fresh one

Agent teams is enabled (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`). When a prior subagent is still contextually relevant, prefer `SendMessage` to its `agentId` (returned by the initial `Task` call) over spawning a fresh `Task`. The prior instance retains in-context memory of the proposal / diff / digest — briefing from scratch wastes tokens and drops nuance, including any user feedback already conveyed through the orchestrator.

**Reuse (`SendMessage`) when:**
- Following up on the same work item with the same specialist (e.g., implementer applying a small revision to the diff it just made; researcher refining the proposal after user feedback).
- The context it built last turn is still load-bearing and would have to be re-briefed verbatim.

**Fresh `Task` when:**
- Topic changes — new experiment, new branch, new feature.
- **Adversarial independence is the whole point, especially `analyst`. Every analyst invocation is a fresh `Task`, never a `SendMessage`.** Continuing an analyst across runs or across mid-run checks contaminates its stance — this is the non-negotiable separation from `## Scientific-method directives`.
- Long gap since the agent last ran (> ~30 min; state probably drifted).
- You want a specialist role but the original agent hit an error state or produced a malformed output — start clean.

**`TeamCreate` when:**
- Parallel multi-agent work where 3–5 workers need to coordinate / debate / claim tasks from a shared list (competing hypotheses, parallel code review, cross-layer refactor). See the agent-teams doc; teammates are full Claude sessions, not one-shot summaries.

Rule of thumb: reuse is the default for continuity; fresh spawn is for independence and topic switches; team is for real parallelism.

## Testing

Run tests after code changes: `conda run -n nnunet python -m pytest tests/ -v`.
