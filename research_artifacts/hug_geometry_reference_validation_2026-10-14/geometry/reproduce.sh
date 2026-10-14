#!/bin/sh
set -eu
# Reuse the existing installed runtime. Supply a new output directory; never
# replace prior results. No installation, download, inference or physics.
if [ "$#" -ne 1 ]; then
  printf '%s\n' 'Usage: sh reproduce.sh /private/tmp/new-hug-geometry-output' >&2
  exit 2
fi
test ! -e "$1"
mkdir -p "$1"
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cp "$script_dir/protocol.json" "$script_dir/investigate_geometry.py" "$script_dir/verify_download_provenance.py" "$script_dir/plot_geometry_causes.py" "$script_dir/check_ray_sanity.py" "$1/"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages
runtime_python=/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python
"$runtime_python" "$1/investigate_geometry.py" --out "$1"
"$runtime_python" "$1/verify_download_provenance.py"
"$runtime_python" "$1/check_ray_sanity.py"
"$runtime_python" "$1/plot_geometry_causes.py"
