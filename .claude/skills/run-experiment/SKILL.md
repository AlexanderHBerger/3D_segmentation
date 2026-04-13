---
name: run-experiment
description: Orchestrates the full experiment cadence — propose → user-confirm → implement-if-needed → sanity-check → submit → mid-run analyst check-ins → final analyze → log. Invoke when the user says "run an experiment" or "test whether X".
---

# Run Experiment

The canonical workflow. Enforces the propose→confirm→execute→report cadence and the scientific-method directives from `CLAUDE.md`.

## Inputs expected

User invokes with a one-line research goal (e.g., "run experiment: does LoRA rank 16 match full finetune on fold 0?"). No more is required — the skill drives the rest.

## Workflow

### Phase 1 — Propose
1. Invoke the **`researcher`** subagent with the user's goal. It returns a written proposal (hypothesis, discriminating measurement, setup, baselines, sanity checks, pre-mortem, cost, seeds).
2. Present the proposal to the user. **Wait for explicit approval** before any code or submission action. Do not assume "sure" on ambiguity — ask.

### Phase 2 — Implement (only if required)
3. If the proposal requires code changes (new ablation flag, new loss component, new config option), invoke the **`implementer`** subagent with the approved proposal.
4. Implementer returns a diff + test results. All tests must pass. If a test fails, surface to the user — do not proceed.

### Phase 3 — Sanity gate
5. Invoke the **`sanity-check`** skill on the current git sha. If it fails, stop and report — do not advance.
6. On success, the marker `.claude/SANITY_OK_<sha>` exists.

### Phase 4 — Submit
7. Invoke the **`experimenter`** subagent with the approved proposal + the confirmed `main.py` invocation.
8. Experimenter calls the `submit-slurm` skill, which picks the partition (minilab-gpu vs preempt_gpu), stages data if needed, and submits with `--requeue` where applicable.
9. Experimenter logs the in-flight entry via the `lab-notebook` skill.

### Phase 5 — Monitor + mid-run analyst
10. Experimenter polls job state on a cadence (15 min, 1 h, 3 h). At each check, runs `scripts/summarize_run.py --partial <run_id>` and invokes a **fresh** `analyst` subagent with only the digest + measurable question + decision rule. No researcher narrative.
11. If the analyst recommends `kill`, surface the digest and rationale to the **user**. Wait for confirmation before `scancel`.

### Phase 6 — Final analyze
12. On job completion, invoke the **`analyze-run`** skill, which runs the final `summarize_run.py` and invokes a fresh `analyst` for the final verdict.
13. The orchestrator writes a Findings entry (or Failed entry) via the `lab-notebook` skill.

## Red lines

- **Never skip Phase 1's user approval.** The researcher proposes, the user confirms. No silent execution.
- **Never skip the sanity gate.** It exists for exactly the bugs humans miss during enthusiasm.
- **Never feed the researcher's hypothesis narrative to the analyst.** Digest + question + decision rule only.
- **Never write a Findings entry without an analyst verdict.** A conclusion requires adversarial review.
- **Do not silently change partition, config, or resources** mid-flow. Any deviation requires a new propose→confirm round.

## When something goes wrong mid-flow

- Surprise result → invoke the `triage-failure` skill. Do not jump to a fix. Classify idea / implementation / setup first.
- Preemption loop → surface after 3 requeues with no progress.
- Analyst says "inconclusive" → do not force a finding; log under `## Failed / inconclusive` with a "would revisit if" clause, and propose next step to user.
