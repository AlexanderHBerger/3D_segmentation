#!/usr/bin/env bash
# PreToolUse hook for Bash(sbatch ...). Blocks submission unless a
# SANITY_OK_<git-sha> marker exists for the current HEAD.
#
# Input: JSON on stdin with tool_input.command.
# Exit 2 + stderr → block the tool call.

set -euo pipefail

# Read stdin into a variable (hooks receive JSON)
payload="$(cat)"

# Extract the command; tolerate missing jq
cmd="$(printf '%s' "$payload" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get("tool_input", {}).get("command", ""))
except Exception:
    pass
')"

# Act when the command invokes sbatch OR an srun that runs main.py (training).
# Pytest, preprocessing, summarize_run, sanity_check itself are exempt.
is_training=0
case "$cmd" in
  *sbatch*) is_training=1 ;;
  *srun*main.py*) is_training=1 ;;
esac
[[ $is_training -eq 0 ]] && exit 0

# Sanity-check itself is allowed through
case "$cmd" in
  *sanity_check.py*) exit 0 ;;
  *pytest*) exit 0 ;;
  *summarize_run.py*) exit 0 ;;
  *preprocess_*) exit 0 ;;
esac

repo_root="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then
  exit 0
fi
sha="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
if [[ -z "$sha" ]]; then
  exit 0
fi

marker="$repo_root/.claude/SANITY_OK_$sha"
if [[ ! -f "$marker" ]]; then
  echo "[hook: check_sanity_marker] BLOCKED: no SANITY_OK marker for current HEAD ($sha)." >&2
  echo "[hook: check_sanity_marker] Run the sanity-check skill first:" >&2
  echo "    conda run -n nnunet python scripts/sanity_check.py" >&2
  exit 2
fi

exit 0
