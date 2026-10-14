#!/bin/sh
set -eu
BUNDLE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
STATIC_REPRO_DIR=/private/tmp/hug_surface_reference_reproduction_2026-10-14
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python "$BUNDLE_DIR/prepare_reproduction.py" "$BUNDLE_DIR" "$STATIC_REPRO_DIR"
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python "$STATIC_REPRO_DIR/compare_surfaces.py" --reference-report "$STATIC_REPRO_DIR/reference/hug_reference_comparison.json" --out "$STATIC_REPRO_DIR"
