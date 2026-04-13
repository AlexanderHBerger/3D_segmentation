#!/usr/bin/env bash
# PreToolUse hook for Bash. If the command is rsync/cp/sbatch touching
# /ministorage/ AND the preempt context is active (CLAUDE_PREEMPT=1 env var,
# set by the submit-slurm skill when preparing a preempt job), warn loudly.
# preempt_gpu nodes cannot see /ministorage/.
#
# Does NOT block — warning only. Exit 0 always.

set -euo pipefail

payload="$(cat)"
cmd="$(printf '%s' "$payload" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get("tool_input", {}).get("command", ""))
except Exception:
    pass
')"

if [[ "${CLAUDE_PREEMPT:-0}" != "1" ]]; then
  exit 0
fi

# Only warn for rsync/cp/sbatch that reference /ministorage/
case "$cmd" in
  *rsync*|*cp*|*sbatch*)
    if printf '%s' "$cmd" | grep -q "/ministorage/"; then
      echo "[hook: ministorage-preempt WARNING] Command references /ministorage/ while a preempt-GPU context is active." >&2
      echo "    preempt_gpu nodes cannot access /ministorage/. Stage to /midtier/paetzollab/scratch/ahb4007/ instead." >&2
    fi
    ;;
esac

exit 0
