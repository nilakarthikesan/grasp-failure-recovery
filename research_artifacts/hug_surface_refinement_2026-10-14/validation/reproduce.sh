#!/bin/sh
set -eu
task_python=/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python
task_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
task_candidate=/private/tmp/hug_surface_refinement_2026-10-14/guarded_candidate_targets.json
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_NUM_THREADS=4
export PYTHONPATH=/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages
"$task_python" "$task_directory/validate_candidate.py" --out "$task_directory" --candidate "$task_candidate" --case-index 8 12 16
"$task_python" "$task_directory/validate_candidate.py" --out "$task_directory" --candidate "$task_candidate" --case-index 20 24 28
"$task_python" "$task_directory/summarize_candidate.py"
"$task_python" "$task_directory/diagnose_existing_witnesses.py"
export MPLCONFIGDIR=/private/tmp/hug_refinement_validation_plot_cache
"$task_python" "$task_directory/plot_candidate.py"
