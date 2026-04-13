---
name: lab-notebook
description: Gated append to LAB_NOTEBOOK.md. Invoke before adding any experiment entry — enforces aggregation, compactness, and the three-bucket rule. Other skills call this; users rarely invoke directly.
---

# Lab Notebook Write Protocol

`LAB_NOTEBOOK.md` is the authoritative, append-only research log. Write discipline matters more than completeness — a bloated notebook poisons future context.

## Before appending, answer these questions

1. **Is this aggregable with an existing entry?** Same hypothesis, new seed → extend the existing entry by adding a line. Same ablation axis → add a row. **Do not create a new section for a variation of an existing one.**
2. **Would the user lose information if we didn't write it?** If every number is derivable from the W&B run + git sha + the `summarize_run.py` digest, **link, don't paraphrase.** Paste only the headline finding (≤ 1 sentence).
3. **Does this belong in `IDEAS.md` instead?** Standing hypotheses, open questions, future directions → `IDEAS.md`. Bound to a specific run/result → `LAB_NOTEBOOK.md`.

If any answer disqualifies the append, do not append.

## Section routing

- `## In-flight` — while a job is running or pending. Format:
  ```
  - job_<id> | <one-line hypothesis> | expected: <signal we're looking for> | sanity: <sha> | <W&B URL> | started: <date>
  ```
  Move to `## Findings` or `## Failed` when the run concludes.

- `## Findings` — concluded, positive/negative but interpretable. Template:
  ```
  ### <short title>  [YYYY-MM-DD]
  - **Hypothesis**: one line.
  - **Result**: headline number (mean ± std), n=<N>, bucket: [idea|implementation|setup] if negative.
  - **Confidence**: high/medium/low + why.
  - **Digest**: scripts/summarize_run.py output path.
  - **W&B**: run URL.
  - **Confirmation plan** (only if n=1): what would replicate this.
  ```

- `## Failed / inconclusive` — negative or uninterpretable results. Critical for preventing re-proposal. Template:
  ```
  ### <short title>  [YYYY-MM-DD]
  - **What we tried**: one line.
  - **What happened**: one line with the killing signal.
  - **Bucket**: [idea|implementation|setup] + one-sentence justification.
  - **Would revisit if**: condition that would make it worth retrying.
  ```

## Hard rules

- Max ~15 lines per entry.
- Every Finding carries an explicit `n=` field. `n=1` is called out loudly with a confirmation plan.
- Every "didn't work" is tagged `[idea|implementation|setup]` with one-sentence justification.
- Never paste raw loss curves, full metric dumps, or configuration tables. Link to the digest instead.
- When editing: prefer extending an existing entry over adding a new one.

## When invoked

1. Read the current `LAB_NOTEBOOK.md` and check whether the content you're about to add aggregates with any existing entry.
2. Apply the three questions above.
3. If the write is justified, use `Edit` to add a minimal entry into the correct section.
4. Report back: (a) which section was touched, (b) whether you extended an existing entry or created a new one, (c) the entry's content in ≤ 5 lines.
