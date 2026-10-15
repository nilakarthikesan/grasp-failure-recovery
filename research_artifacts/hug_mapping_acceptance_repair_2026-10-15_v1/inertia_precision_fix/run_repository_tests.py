from pathlib import Path
import sys,json,numpy,mujoco
root=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sys.path.insert(0,str(root/'src'))
sys.path.extend(['/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps','/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps','/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages'])
counts={'physics_steps':0,'static_forwards':0}
def no_step(*a,**kw):counts['physics_steps']+=1;raise RuntimeError('No physics authorized during static mapping tests')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,no_step)
original=mujoco.mj_forward
def forward(*a,**kw):counts['static_forwards']+=1;return original(*a,**kw)
mujoco.mj_forward=forward
import pytest
result=pytest.main(['-q','-p','no:cacheprovider',str(root/'tests/test_object_assets.py'),'-k','not free_fall'])
assert counts['physics_steps']==0
out={'return_code':result,'numpy_version':numpy.__version__,'mujoco_version':mujoco.__version__,**counts,'tests':'tests/test_object_assets.py -k not free_fall','uses_applied_repository_source':True,'new_eval_data':False}
Path(__file__).with_name('repository_test_receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));raise SystemExit(result)
