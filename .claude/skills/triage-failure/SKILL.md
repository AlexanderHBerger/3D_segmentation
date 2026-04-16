---
name: triage-failure
description: Three-bucket classifier for unexpected results. Forces the classification into idea / implementation / setup before any fix is attempted, plus the minimal experiment that would discriminate between buckets. Invoke when a run surprised you, when an analyst returned `reject` or `inconclusive`, or before a user jumps to debugging.
---

# Triage Failure

Do not fix what you have not classified. The three-bucket rule is the single most effective guard against wasted debugging cycles.

## When to invoke

- An experiment surprised you (positively or negatively).
- The analyst returned `reject` or `inconclusive`.
- A sanity check passed but training diverges.
- Intermediate metrics look "off" and you're tempted to tweak.
- **Before** any patch attempt. If you've already started a fix, stop and triage first.

## The three buckets

| Bucket | Meaning | Typical signals | Typical fix |
|--------|---------|-----------------|-------------|
| **idea** | The research hypothesis is likely wrong. The setup was sound, the implementation was sound, the number just didn't move the right way. | Sanity-check passed. GPU util healthy. Baseline behaves as expected. New-condition curve is smooth, converged, just lands in the wrong place. | Propose a new hypothesis. Update `IDEAS.md`. Log under `## Failed / inconclusive` with bucket=idea. |
| **implementation** | A code bug in the change made the measurement invalid. | Loss not decreasing or NaN, shuffled-label check (in retrospect) would have failed, gradients zero on a component, parameters not registered in optimizer, wrong mode (train vs eval), wrong data path being loaded. | Implementer finds and fixes the bug. Re-run. |
| **setup** | The experiment configuration doesn't measure what you claimed. | Baseline trained 2× longer, different seeds, different dataset split, different patch size, wrong partition, GPU-util bottleneck distorting optimizer behavior, sanity-check skipped, wrong init checkpoint. | Fix the setup. Re-run baseline + new condition on matched configuration. |

## Workflow

1. **State the observation in one sentence.** What number moved in what direction on which metric.
2. **List the three possible bucket assignments.** For each bucket, write one sentence: *if this bucket is correct, what specifically explains the observation?*
3. **Identify the discriminating experiment.** The minimal, cheapest thing that would distinguish the buckets. Examples:
   - To distinguish idea vs. implementation: re-run sanity-check on the current sha; if a check that used to pass now fails → implementation.
   - To distinguish idea vs. setup: re-run with identical config to the baseline except the single research-relevant change. If the new run now matches baseline → setup (config drift).
   - To distinguish implementation vs. setup: rerun the exact same code and config on a different seed / different fold. If behavior is stable but wrong → implementation. If it's different every time → stochastic setup issue.
4. **Pick the cheapest discriminator.** Execute it. Report the result.
5. **Commit to a bucket** with one-sentence justification, and only then decide on the fix.
6. **Log the triage** via the `lab-notebook` skill under `## Failed / inconclusive`, with the bucket, justification, and "would revisit if" clause.

## Anti-patterns (do not)

- "Let's just re-run and see." → Re-running without a discriminator is burning compute.
- "I think it's the LR." → That's pattern-locking; triage before tweaking.
- "The sanity check passed, so it must be the idea." → Sanity checks are not exhaustive; some bugs only surface at scale. Still required to run the discriminator.
- "We'll figure it out once we see more runs." → No. One bucket, now, or do not proceed.

## Red lines

- Do not fix the code before you have a bucket.
- Do not declare "idea is wrong" on n=1 without the discriminating experiment.
- If you cannot design a discriminator, the observation is not yet well-enough characterized — gather more specific data before claiming anything.

## Autonomous-mode use

When invoked with `CLAUDE_AUTONOMOUS=1` by the experimenter during an overnight kill-and-fix cycle, this skill:

- Reads the kill digest and the analyst's verdict. The analyst has already given a bucket; your job is to confirm it and propose the *next action*, not to rerun the full discriminator workflow.
- **Proposes a fix yourself** (no fixed whitelist — use judgment grounded in the digest's signals). Example: NaN-hit at step 30k → propose grad_clip + AMP off. Flatline at zero → propose higher foreground oversampling or altered loss-component weights. Sustained loss increase → propose LR reduction or AMP inspection.
- The fix must be **non-fundamental**; the implementer enforces this. If your proposed fix requires a fundamental change (architecture, loss design, dataloader logic, preprocessing), say so — the loop will stop and surface to the user, no fix is applied.
- Record the proposed fix + the signal that triggered it + the bucket rationale in the triage output. The experimenter passes this to the implementer, which either applies it (commit on branch) or reports back that only a fundamental change would work.
- Do not apply the fix yourself — you are the classifier, the implementer is the applicator.
