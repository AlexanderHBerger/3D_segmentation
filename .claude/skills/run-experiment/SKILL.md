---
name: run-experiment
description: Orchestrates the full experiment cadence — propose → user-confirm → implement-if-needed → sanity-check → submit → mid-run analyst check-ins → final analyze → log. Invoke when the user says "run an experiment" or "test whether X".
---

# Run Experiment

The canonical workflow. Enforces the propose→confirm→execute→report cadence and the scientific-method directives from `CLAUDE.md`.

## Inputs expected

User invokes with a one-line research goal (e.g., "run experiment: does LoRA rank 16 match full finetune on fold 0?").

Optional: `--autonomous` flag. With autonomy enabled, the pipeline's mid-run loop will kill and re-fix-and-rerun without waiting for user confirmation, as long as the failure signals are unambiguous (see `experimenter` agent for the four signals). Used for overnight runs. Without it, every kill requires your OK.

## Workflow

### Phase 0 — Git branch

Before anything else, create a new branch from `main` named after the hypothesis, e.g. `experiment/lora-rank-16-vs-full` or `feature/distance-field-loss`. All commits for this experiment (implementer fixes, autonomous fixes, final changes) land on this branch. `main` stays untouched until the final PR merge (Phase 7).

```bash
git fetch origin main
git checkout -b experiment/<short-name> origin/main
```

If the user's goal naturally extends a branch that already exists, use that branch instead of creating a new one — but confirm with the user in the proposal.

### Phase 1 — Propose
1. Invoke the **`researcher`** subagent with the user's goal. It returns a written proposal (hypothesis, discriminating measurement, setup, baselines, sanity checks, pre-mortem, cost, seeds, **branch name**).
2. Present the proposal to the user. **Wait for explicit approval** before any code or submission action. Do not assume "sure" on ambiguity — ask. Autonomous mode does **not** bypass this initial approval; you opt into autonomy for the monitoring/fix loop, not for the initial hypothesis.

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

**The orchestrator (you) owns the wait-and-re-invoke loop.** The experimenter performs one poll per invocation and returns with the current state + a suggested next-check interval. After each experimenter return, you `ScheduleWakeup(suggested_minutes * 60)` (clamped to `[60, 3600]`; chain wakes for longer intervals) and then re-invoke the experimenter — default via `SendMessage` to the prior instance so it keeps the job context, fresh `Task` only if that fails. `ScheduleWakeup` requires `/loop` dynamic mode for overnight autonomy; start autonomous runs with `/loop /run-experiment --autonomous <goal>`.

10. Experimenter polls job state on a cadence (15 min, 1 h, 3 h). At each check, runs `scripts/summarize_run.py --partial <run_id>` and invokes a **fresh** `analyst` subagent with only the digest + measurable question + decision rule. No researcher narrative.
11. If the analyst recommends `kill`, surface the digest and rationale to the **user**. Wait for confirmation before `scancel`.

### Phase 6 — Final analyze
12. On job completion, invoke the **`analyze-run`** skill, which runs the final `summarize_run.py` and invokes a fresh `analyst` for the final verdict.
13. The orchestrator writes a Findings entry (or Failed entry) via the `lab-notebook` skill.

### Phase 7 — Git: commit state, PR (only on accept)

The branch has accumulated commits (implementer fixes, autonomous fixes) throughout Phases 2–5. At the end:

- **Analyst verdict `accept`** →
  - Confirm any uncommitted changes on the branch are either committed (code) or captured as LAB_NOTEBOOK notes (config-only).
  - `git push -u origin <branch>`.
  - `gh pr create` with:
    - Title: the hypothesis.
    - Body: the researcher's proposal + the analyst's accept verdict + link to the final digest + link to the W&B run.
    - Pipeline does **not** merge — it creates the PR and hands to the user for review.
- **Analyst verdict `reject` or `inconclusive`** →
  - Do not push, do not open a PR. The branch stays local for the user to inspect or delete.
  - The Failed/inconclusive LAB_NOTEBOOK entry is the audit trail.

Branch hygiene: if the experiment ran for multiple autonomous retries, the branch may have many `autonomous fix: ...` commits. At PR creation, the pipeline offers the user the option of a squash-merge with a clean message. User decides.

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

## Autonomous-mode lifecycle (summary)

When invoked with `--autonomous`:
- Phase 0–3 are identical (branch, propose, confirm, implement, sanity).
- Phase 4 (submit) sets `CLAUDE_AUTONOMOUS=1` in the experimenter's context.
- Phase 5 (monitor) is an **orchestrator-scheduled** `ScheduleWakeup` loop: each wake, you re-invoke the experimenter for one poll. On kill with an unambiguous signal, the experimenter runs triage → implementer (non-fundamental only) → commit (code) or notebook-note (config-only) → sanity-check → resubmit in that single invocation, then hands back; you resume scheduling. Infinite until a natural exit (accept / reject+idea / reject+fundamental-impl / user intervention / no-progress guard).
- Phase 6–7 only run once: on final accept of the last rerun, PR is created; on any reject exit, branch stays local and Failed entry is the audit trail.
