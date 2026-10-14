#!/bin/sh
set -eu
bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
task_output_dir=${1:-$(mktemp -d /private/tmp/hug-sampling-reproduce.XXXXXX)}
mkdir -p "$task_output_dir"
for name in check_sampling.py cell_intervals.py protocol.json protected_hashes.json test_cells.py independent_review.py write_witnesses.py witness_protocol.json acceptance_rule.json summarize_sampling.py plot_sampling.py; do
  cp "$bundle_dir/$name" "$task_output_dir/$name"
done
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages'
export MPLCONFIGDIR="$task_output_dir/matplotlib-cache"
export XDG_CACHE_HOME="$task_output_dir/font-cache"
task_python='/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python'
"$task_python" "$task_output_dir/test_cells.py"
"$task_python" "$task_output_dir/independent_review.py"
"$task_python" "$task_output_dir/check_sampling.py" --out "$task_output_dir"
"$task_python" "$task_output_dir/write_witnesses.py"
"$task_python" "$task_output_dir/summarize_sampling.py"
"$task_python" "$task_output_dir/plot_sampling.py"
printf '%s\n' "$task_output_dir"
