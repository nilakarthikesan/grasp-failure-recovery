#!/bin/sh
set -eu
cd /private/tmp/hug-palm-anchor-2026-10-14
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages' /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python diagnose_palm_anchor.py
