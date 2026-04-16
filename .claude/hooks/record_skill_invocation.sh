#!/usr/bin/env bash
# PostToolUse hook on the Skill tool.
# Writes a per-session marker file whenever a skill is invoked, so other hooks
# can answer "was skill X consulted recently?". The warn_slurm_bypass hook is
# the current consumer.
#
# Marker layout: /tmp/claude_skill_markers_<user>/<session_key>_<skill>
# Freshness is encoded as the file's mtime; the consumer defines the window.
#
# Exit 0 always — this hook is informational, never blocks.

set -euo pipefail

payload="$(cat)"
skill="$(printf '%s' "$payload" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get("tool_input", {}).get("skill", ""))
except Exception:
    pass
')"

# Nothing to record if this wasn't a Skill invocation or the payload was malformed.
[[ -z "$skill" ]] && exit 0

marker_dir="/tmp/claude_skill_markers_${USER:-unknown}"
mkdir -p "$marker_dir"

# Session scope, in order of preference:
#   CLAUDE_SESSION_ID (set by Claude Code when available)
#   TMUX_PANE         (stable across tool calls inside one tmux pane)
#   PPID              (last-resort fallback; not guaranteed stable across spawns)
session_key="${CLAUDE_SESSION_ID:-${TMUX_PANE:-${PPID:-default}}}"
session_key="${session_key//[^A-Za-z0-9_-]/_}"

# Sanitize skill name similarly (skills can be namespaced like `plugin:skill`).
skill_safe="${skill//[^A-Za-z0-9_.-]/_}"

touch "$marker_dir/${session_key}_${skill_safe}"
exit 0
