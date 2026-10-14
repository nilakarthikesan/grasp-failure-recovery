#!/bin/sh
set -eu
# Run the published sibling sampling script first to recover nine ephemeral
# exact current-distance exports. Original saved source assets and old v4 query
# caches are required locally; no assets/caches are committed in this bundle.
TASK_OUT=${1:-/private/tmp/hug_orientation_reproduction_2026-10-14}
TASK_SHARED=${2:-/private/tmp/hug_sampling_checks_2026-10-14/ephemeral}
TASK_HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$TASK_OUT"
cp "$TASK_HERE/protocol.json" "$TASK_OUT/protocol.json"
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages \
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python "$TASK_HERE/check_orientation.py" --out "$TASK_OUT" --sampling-ephemeral "$TASK_SHARED"
