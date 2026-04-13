---
name: analyze-run
description: Post-hoc analysis of a completed (or partial) run. Builds a compact digest via scripts/summarize_run.py, then invokes a fresh analyst subagent with the digest + measurable question + decision rule. Never passes the researcher's narrative to the analyst. Invoke for final analysis and for mid-run kill-or-continue decisions.
---

# Analyze Run

LLMs are poor at reading raw 10k-line loss logs or interpreting PNG plots. This skill solves that by running `scripts/summarize_run.py` **first**, so the analyst sees a structured digest, not raw data.

## Inputs

- `run_id` — W&B run id (e.g., `pirxiw5h`) or `entity/project/id`.
- `--baseline <id>` (optional) — baseline run for side-by-side comparison.
- `--partial` (optional) — mid-run digest, tolerant of unfinished runs.
- `measurable_question` — the single primary comparison from the proposal (e.g., "does val/dice_hard on fold 0 for run X exceed baseline run Y by ≥ 0.01?").
- `decision_rule` — the accept/reject threshold from the proposal.

## Workflow

1. **Build the digest — via SLURM on `minilab-cpu`.** The login node has no compute; never invoke `python …` directly there.
   ```bash
   srun --partition=minilab-cpu --qos=normal --mem=4G --cpus-per-task=2 \
        --time=00:15:00 \
        conda run -n nnunet python scripts/summarize_run.py <run_id> \
            [--partial] [--baseline <id>] --output /tmp/digest_<run_id>.md
   ```
   This pulls: metadata, convergence verdict, primary metrics (`val/dice_hard`, `val/loss`, `train/loss_iter` aggregated), loss components (`train/loss_*` per-run), perf metrics (`train/epoch_time`, `train/iter_time`, `train/learning_rate`), system metrics (GPU util + memory). Flags low GPU util (< 70 %) as a setup/impl issue, not a research signal.

2. **Invoke the `analyst` subagent** via Task with **only**:
   - The digest content (paste from the file or pass its path).
   - The measurable question.
   - The decision rule.
   - (If mid-run): instruction to output `continue | continue (watch) | kill`.
   - (If final): instruction to output `accept | reject | inconclusive`.

   **Do not** pass:
   - The researcher's hypothesis framing.
   - Prior orchestrator conversation context.
   - Speculation about "what we hoped to see."

3. **Receive the verdict.** Format:
   ```
   VERDICT: ...
   CONFIDENCE: low | medium | high
   BUCKET: idea | implementation | setup   (only if not accept)
   REASONING: ≤ 5 sentences, grounded in digest numbers.
   WHAT WOULD CHANGE MY MIND: one concrete experiment or measurement.
   ```

4. **Act on the verdict.**
   - `accept` or `reject` (final) → invoke `lab-notebook` skill to write a Findings entry with the verdict's confidence + bucket.
   - `inconclusive` (final) → invoke `lab-notebook` to write a Failed/inconclusive entry with the "would change my mind" field as the `Would revisit if` clause.
   - `kill` (mid-run) → surface to user with digest + rationale, wait for confirmation, then experimenter `scancel`s.
   - `continue` / `continue (watch)` → experimenter proceeds with normal polling cadence.

## Red lines

- Never skip the digest step. Raw W&B scraping + manual interpretation fills context and invites pattern-locking.
- Never write a Findings entry without a verdict.
- Never summarize the verdict in your own words before passing it to the user — paste it verbatim.
- If the digest is missing data (e.g., W&B API throttled), do not guess — rerun or surface the gap.
