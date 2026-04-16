#!/usr/bin/env bash
# PreToolUse hook on Bash.
# If the command submits to SLURM (srun / sbatch) AND the submit-slurm skill
# has NOT been invoked in this session within the freshness window, emit a
# stderr warning. Non-blocking — exit 0 always. Purpose is diagnosis, not
# enforcement. See Option 2 in the pipeline design discussion.
#
# Only fires on actual submissions (command starts with srun/sbatch, possibly
# after leading env assignments, split across ; | && || chains). Queries like
# sinfo/squeue/sacct/scontrol/scancel are ignored. Substring matches in
# arguments (e.g., `grep srun foo.sh`) are ignored.

set -euo pipefail

FRESHNESS_SECONDS=600   # 10 min window.

payload="$(cat)"

# Python does the command parsing: extract the command field, walk each
# pipeline/chain segment, strip leading env assignments, and check the first
# real token. Prints the original command only when it is a submission; else
# prints nothing so bash short-circuits.
cmd="$(printf '%s' "$payload" | python3 -c '
import json, re, sys
try:
    d = json.load(sys.stdin)
    cmd = d.get("tool_input", {}).get("command", "")
except Exception:
    sys.exit(0)

is_submit = False
for seg in re.split(r"[;|&]+", cmd):
    seg = seg.strip()
    # Strip leading env assignments: VAR=val VAR2=val2 actual_cmd ...
    while re.match(r"^[A-Za-z_]\w*=\S+\s+", seg):
        seg = re.sub(r"^[A-Za-z_]\w*=\S+\s+", "", seg, count=1)
    first = seg.split(None, 1)[0] if seg else ""
    if first in ("srun", "sbatch"):
        is_submit = True
        break

if is_submit:
    print(cmd, end="")
')"

# Non-submission commands produce empty output → exit silently.
[[ -z "$cmd" ]] && exit 0

marker_dir="/tmp/claude_skill_markers_${USER:-unknown}"
session_key="${CLAUDE_SESSION_ID:-${TMUX_PANE:-${PPID:-default}}}"
session_key="${session_key//[^A-Za-z0-9_-]/_}"
marker="$marker_dir/${session_key}_submit-slurm"

fresh=0
if [[ -f "$marker" ]]; then
    now="$(date +%s)"
    # stat flavor differs between GNU coreutils (-c) and BSD (-f). Try both.
    mtime="$(stat -c %Y "$marker" 2>/dev/null || stat -f %m "$marker" 2>/dev/null || echo 0)"
    age=$((now - mtime))
    [[ $age -le $FRESHNESS_SECONDS ]] && fresh=1
fi

if [[ $fresh -eq 0 ]]; then
    short_cmd="${cmd:0:140}"
    echo "[submit-slurm bypass] Direct srun/sbatch without a recent submit-slurm skill invocation in this session." >&2
    echo "    cmd: $short_cmd" >&2
    echo "    Expected flow: Skill(submit-slurm) -> follow its recipes. See .claude/skills/submit-slurm/SKILL.md." >&2
fi

exit 0
