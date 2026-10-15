"""Use declared current NumPy/MuJoCo before appending legacy dependencies."""
from pathlib import Path
import os,sys,runpy
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='4'
import numpy,mujoco
assert numpy.__version__=='2.2.6' and mujoco.__version__=='3.3.7'
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
base=Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check')
sys.path.insert(0,str(ROOT/'src'))
sys.path.extend([str(base/'hug-deps'),str(base/'deps'),str(base/'deps/cmeel.prefix/lib/python3.12/site-packages')])
sys.path.insert(0,'/private/tmp/hug_feasibility_restoration_2026-10-15_v1')
sys.path.insert(0,'/private/tmp/hug_current_scene_bridge_2026-10-15_v1')
sys.path.insert(0,'/private/tmp/hug_current_execution_bridge_2026-10-15_v1')
sys.path.insert(0,str(Path(__file__).resolve().parent))
runpy.run_path(str(Path(__file__).with_name('run_witness.py')),run_name='__main__')
