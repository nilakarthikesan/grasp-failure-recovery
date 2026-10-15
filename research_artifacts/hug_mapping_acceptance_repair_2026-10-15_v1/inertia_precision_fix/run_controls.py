from pathlib import Path
import numpy,mujoco,sys,importlib.util,importlib.machinery,json
H=Path(__file__).resolve().parent;W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sys.path.insert(0,str(W/'src'));sys.path.extend(['/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps','/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps','/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages'])
counts={'steps':0,'forwards':0,'compiles':0}
def no_step(*a,**kw):counts['steps']+=1;raise RuntimeError('No physics in object precision controls')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,no_step)
original=mujoco.mj_forward
def forward(*a,**kw):counts['forwards']+=1;return original(*a,**kw)
mujoco.mj_forward=forward
name='grasp_failure_prediction.evaluation.object_assets';loader=importlib.machinery.SourceFileLoader(name,str(H/'object_assets.py.proposed'));spec=importlib.util.spec_from_loader(name,loader);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;loader.exec_module(module)
# Pytest requires a Python filename; content-identical temp copy, no repo edit.
(H/'test_object_assets_proposed.py').write_bytes((H/'test_object_assets.py.proposed').read_bytes())
import pytest
status=pytest.main(['-q',str(H/'test_object_assets_proposed.py'),'-k','inertia_precision or native_compound_geometry_and_inertial_frames_survive_import or invalid_physical_frames_and_asset_closures_are_rejected'])
assert counts['steps']==0
receipt={'exit_code':int(status),'numpy_version':numpy.__version__,'mujoco_version':mujoco.__version__,'physics_steps':counts['steps'],'static_forward_calls':counts['forwards'],'scope':'Manufactured objects plus existing static hand/object import regressions; no saved HUG grasp placement, new proposals, or evals.'}
(H/'controls_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(status)
