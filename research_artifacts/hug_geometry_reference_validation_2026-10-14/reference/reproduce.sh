#!/bin/sh
set -eu
source_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
audit_dir=/private/tmp/hug_reference_compare_reproduction_2026-10-14
mkdir -p "$audit_dir"
for file in protocol.json compare_reference.py check_clipping.py compact_comparison.py plot_reference.py; do cp "$source_dir/$file" "$audit_dir/$file"; done
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages'
python_exec=/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python
"$python_exec" "$audit_dir/compare_reference.py" > "$audit_dir/run.log" 2>&1
"$python_exec" "$audit_dir/check_clipping.py"
"$python_exec" "$audit_dir/compact_comparison.py"
"$python_exec" "$audit_dir/plot_reference.py"
