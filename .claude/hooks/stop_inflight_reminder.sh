#!/usr/bin/env bash
# Stop hook. If LAB_NOTEBOOK.md has entries in ## In-flight section AND
# those jobs are still RUNNING or PENDING, remind the orchestrator to
# either schedule a wake-up to poll, or hand control back to the user.
#
# Advisory only. Exit 0.

set -euo pipefail

repo_root="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then exit 0; fi
notebook="$repo_root/LAB_NOTEBOOK.md"
if [[ ! -f "$notebook" ]]; then exit 0; fi

# Extract job IDs listed under "## In-flight"
job_ids="$(python3 - "$notebook" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text()
# Section between "## In-flight" and the next "## " header
m = re.search(r'^## In-flight\s*\n(.*?)(?=^## |\Z)', text, re.DOTALL | re.MULTILINE)
if not m:
    sys.exit(0)
section = m.group(1)
# Match `job_12345` or `| job | 12345 |` patterns
ids = re.findall(r'\bjob_(\d+)\b|\b(\d{5,})\b', section)
flat = [a or b for a, b in ids]
print('\n'.join(sorted(set(flat))))
PY
)"

if [[ -z "$job_ids" ]]; then exit 0; fi

live_count=0
for jid in $job_ids; do
  state="$(sacct -j "$jid" --format=State --noheader --parsable2 2>/dev/null | head -n 1 | tr -d ' ')"
  case "$state" in
    RUNNING|PENDING|REQUEUED)
      live_count=$((live_count + 1))
      echo "[hook: stop_inflight] job $jid is still $state" >&2
      ;;
  esac
done

if [[ $live_count -gt 0 ]]; then
  echo "[hook: stop_inflight] $live_count in-flight job(s). Either schedule a wake-up to poll (ScheduleWakeup) or explicitly hand back to the user before ending the turn." >&2
fi

exit 0
