"""Import-only guard: no native scenes/models, FK or geometry queries."""
import sys,json,hashlib,ast
from pathlib import Path
import numpy,mujoco
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
b=Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check');here=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'));sys.path.extend([str(b/'hug-deps'),str(b/'deps'),str(b/'deps/cmeel.prefix/lib/python3.12/site-packages')])
for path in ('/private/tmp/hug_feasibility_restoration_2026-10-15_v1','/private/tmp/hug_current_scene_bridge_2026-10-15_v1','/private/tmp/hug_current_execution_bridge_2026-10-15_v1',str(here)):sys.path.insert(0,path)
import exact_direction,scene_runtime,native_reference,xml_reference,numba,run_witness
from grasp_failure_prediction.evaluation import pose_validation
numba.set_num_threads(2)
assert numpy.__version__=='2.2.6' and mujoco.__version__=='3.3.7'
assert Path(pose_validation.__file__).resolve()==ROOT/'src/grasp_failure_prediction/evaluation/pose_validation.py'
assert exact_direction.a._link_positions is pose_validation._link_positions
for path in here.glob('*.py'):ast.parse(path.read_text())
report={'passed':True,'numpy':numpy.__version__,'mujoco':mujoco.__version__,'numba_threads':numba.get_num_threads(),'models':0,'geometry_queries':0,'FK':0,
        'imported_sources':{m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (pose_validation,scene_runtime,native_reference,xml_reference,run_witness,exact_direction)}}
(here/'import_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
