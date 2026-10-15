"""Pin NumPy/MuJoCo ahead of dependency overlays, then execute one named script."""
from pathlib import Path
import numpy,mujoco,sys,runpy
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src')
sys.path.extend(['/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps','/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps','/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages'])
sys.path.insert(0,'/private/tmp/hug_current_execution_bridge_2026-10-15_v1');sys.path.insert(0,str(HERE))
script=sys.argv[1];assert script in ('test_scene_bridge.py','prepare_protocol.py','run_scene_bridge.py');sys.argv=[str(HERE/script),*sys.argv[2:]]
runpy.run_path(str(HERE/script),run_name='__main__')
