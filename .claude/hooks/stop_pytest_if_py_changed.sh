#!/usr/bin/env bash
# Stop hook. Runs once at end of turn.
# If any tracked *.py file has been modified in the working tree this turn,
# run the test suite fast-fail. Also invalidates SANITY_OK marker if any
# training-affecting file was touched.
#
# Output goes to stderr as a notification (non-blocking). Exit 0 always.

set -euo pipefail

repo_root="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then exit 0; fi
cd "$repo_root"

# Were any .py files modified in the working tree? (unstaged changes)
changed_py="$(git diff --name-only -- '*.py' 2>/dev/null || true)"
changed_py_staged="$(git diff --cached --name-only -- '*.py' 2>/dev/null || true)"
all_changed_py="$(printf '%s\n%s' "$changed_py" "$changed_py_staged" | sort -u | sed '/^$/d')"

if [[ -z "$all_changed_py" ]]; then
  exit 0
fi

echo "[hook: stop_pytest] *.py files changed this turn:" >&2
printf '  %s\n' $all_changed_py >&2

# Invalidate SANITY_OK marker if training-affecting files were touched
training_files_regex='^(train|model|architectures|losses|data_loading_native|text_prompted_model|transformer|config)\.py$|^preprocessing/'
if printf '%s\n' $all_changed_py | grep -Eq "$training_files_regex"; then
  echo "[hook: stop_pytest] training-affecting file changed — invalidating SANITY_OK markers" >&2
  rm -f .claude/SANITY_OK_* 2>/dev/null || true
fi

# Nothing runs on the login node — dispatch pytest to minilab-cpu via srun.
# srun blocks until the job finishes and streams output, which is what we want
# for a Stop hook. 30-minute walltime, 8 GB, 4 CPUs.
echo "[hook: stop_pytest] dispatching pytest to minilab-cpu via srun..." >&2
if ! srun --partition=minilab-cpu --qos=normal --mem=32G --cpus-per-task=4 \
          --time=00:30:00 --job-name=claude-pytest \
          conda run -n nnunet python -m pytest tests/ -q -x --no-header \
          2>&1 | tail -n 40 >&2; then
  echo "[hook: stop_pytest] ⚠ tests failed or srun errored. Address before next submission." >&2
fi

exit 0
