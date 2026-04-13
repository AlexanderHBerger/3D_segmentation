# Claude Code Research Pipeline — User Playbook

How to drive this repo's Claude Code research pipeline. Not loaded by Claude automatically; read this as a human when you need a refresher.

## The core loop

```
  researcher → you-confirm → implementer (if needed) → sanity-check
    → experimenter → submit-slurm → (mid-run analyst check-ins)
    → analyze-run → lab-notebook
```

Each arrow is a gate. You are the decision-maker at every step where the pipeline explicitly pauses.

## Starting an experiment

Enter plan mode and invoke the skill:

```
/run-experiment does LoRA rank 16 match full finetune on fold 0?
```

The `run-experiment` skill orchestrates the full loop. It will pause after the researcher writes its proposal and wait for your approval before touching code or the cluster.

## Analyzing a run

Always analyze in a **fresh session** to avoid carrying the experiment's framing into the analysis:

```
$ claude --rename analyze-<run_id>
> /analyze-run <run_id>
```

Never ask "did it work?" in the same session where you designed the experiment — that's exactly the confirmation-bias failure mode the pipeline is built to prevent.

## When results surprise you

```
/triage-failure
```

This forces classification into `idea | implementation | setup` **before** any fix attempt. If you skip this, you'll fix a bug that doesn't exist or miss the real cause.

## Session persistence (your SSH-drop problem)

Your laptop sleeps, SSH dies, the Claude session is gone mid-run. Four options, in order of preference:

1. **Hybrid (recommended daily driver)** — VSCode for editing, tmux-wrapped CLI Claude for long-running autonomous work. Run `tmux new -s claude` on the server, start `claude` inside it. Detach with `Ctrl-b d`, reattach later with `tmux attach -t claude`. VSCode shows the same files but you drive Claude from the terminal.
2. **VSCode extension + `claude --continue`** — simpler. If the laptop sleeps, reconnect, run `claude --continue` (or `--resume` to pick a session). In-flight *tool calls* are lost on disconnect, but conversation state is preserved.
3. **tmux alone** — no VSCode extension. Purest persistence. Same as (1) minus the editor.
4. **SLURM-dispatched autonomous run** — for truly long autonomous work, submit Claude itself as a SLURM job:
   ```bash
   sbatch --partition=minilab-gpu --time=24:00:00 --wrap \
     'claude -p "<your research prompt>" --permission-mode auto'
   ```
   The job writes findings directly to `LAB_NOTEBOOK.md`. Your laptop becomes irrelevant.

Name sessions by hypothesis so you can find them:

```
/rename lora-rank-ablation
```

## The scientific-method contract (read this once, internalize)

- **The three-bucket rule.** Any result (positive or negative) is tagged `idea | implementation | setup` with one-sentence justification. SGD keeps descending through bugs; you must distinguish signal from convergence.
- **Experimenter ≠ analyst.** The researcher proposes, the analyst judges. They run in different subagents with no shared narrative. Never cross-contaminate.
- **Default to suspicion.** The `analyst` subagent is explicitly adversarial. If its verdict sounds too confident, question the digest.
- **Sit with ambiguity.** If n=1, the finding is preliminary. `LAB_NOTEBOOK.md` entries enforce this.
- **Token-pressure awareness.** When main-context usage > 60 %, the main Claude delegates deep reads to subagents or asks you to start a fresh session. You should honor this, not override it.

## Files you'll touch

| File | Who writes | When |
|------|-----------|------|
| `LAB_NOTEBOOK.md` | only the `lab-notebook` skill | after every concluded experiment |
| `IDEAS.md` | researcher or you | when a standing hypothesis or parked question emerges |
| `CLAUDE.md` | rarely, only after weeks of observed behavior drift | when Claude consistently misses something important |
| `.claude/skills/*` | you, when a new reusable workflow emerges | additively — prune rather than bloat |
| `scripts/summarize_run.py`, `scripts/sanity_check.py` | the `implementer`, when new checks/metrics are worth enforcing | when a failure mode repeats across runs |

## Hooks that will interrupt you

Configured in `.claude/settings.json`:

1. **`sbatch` without `SANITY_OK`** → blocked. Run `/sanity-check` first (or `python scripts/sanity_check.py`).
2. **Touching `/ministorage/` in a preempt context** → warned. Stage to `/midtier/paetzollab/scratch/ahb4007/` instead.
3. **End of turn, any `*.py` changed** → pytest fast-fail runs. If a training-affecting file changed, the `SANITY_OK` marker is invalidated automatically.
4. **End of turn, in-flight SLURM jobs** → reminds Claude to schedule a poll or hand back to you.

## Subagent model assignments

All four start on `opus`. If you hit usage limits, downgrade in this order (lowest risk first):

1. `experimenter` → sonnet (most mechanical)
2. `implementer` → sonnet (code generation is robust on sonnet)
3. `analyst` → keep on opus if possible (adversarial reasoning is the hardest to get right)
4. `researcher` → keep on opus (hypothesis design is the highest-leverage step)

Edit the `model:` field in `.claude/agents/<role>.md` to change.

## Anti-patterns (what breaks the pipeline)

- Letting the main Claude session accumulate history across multiple unrelated experiments. Use `/clear` or start a fresh session.
- Asking Claude to "just debug quickly" without a triage-failure classification. You'll end up fixing imagined bugs.
- Feeding the researcher's hypothesis narrative into the analyst prompt "so it has context." That defeats the separation.
- Hand-writing sbatch scripts. Always go through `submit-slurm`.
- Editing `CLAUDE.md` reactively every time Claude does something slightly wrong. Pruning is more valuable than appending.

## When something is genuinely broken

Open an issue on this repo with:
- the command/prompt that misbehaved,
- what you expected vs. what happened,
- the digest path if a run is involved,
- whether the `SANITY_OK` marker was valid at the time.

Don't tweak the pipeline skills in-place until you've observed the failure twice. Most one-off oddities are not worth a rule change.
