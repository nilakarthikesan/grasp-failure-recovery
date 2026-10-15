"""Manufactured build-only transaction admission and forbidden-operation checks."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('shape_transaction_controls',Path(__file__).with_name('run_shape_build.py'))
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

def report():
    return dict(status='COMPLETE',failure=None,build_protocol_sha256='a'*64,plan_sha256='b'*64,
        production_promotion=False,new_collection_or_training_allowed=False,native_fidelity_accepted=False,
        licensed_asset_copied=False,counts=dict(neutral_mano_forwards=1,neutral_mano_forward_attempts=1,geometry_builder_returns=1,
            MuJoCo_forwards=0,physics_steps=0,HUG_generator_calls=0,retarget_calls=0))

def test_build_complete_does_not_admit_any_native_pose_or_task_data():
    driver.check_completion(report(),dict(physics=0), 'a'*64,'b'*64)
    assert driver.BUDGET['maximum_neutral_mano_forwards']==1
    assert all(v==0 for k,v in driver.BUDGET.items() if not k.startswith('maximum_'))

@pytest.mark.parametrize('kind',['wrong_protocol','wrong_plan','failure','promotion','extra_forward','no_forward','physics','boolean_attempt','nonzero_attempt'])
def test_forged_incomplete_or_wrong_scope_report_refuses(kind):
    value=report();counts=dict(physics=0)
    if kind=='wrong_protocol':value['build_protocol_sha256']='c'*64
    if kind=='wrong_plan':value['plan_sha256']='c'*64
    if kind=='failure':value['failure']={'type':'RuntimeError'}
    if kind=='promotion':value['production_promotion']=True
    if kind=='extra_forward':value['counts']['neutral_mano_forwards']=2
    if kind=='no_forward':value['counts']['neutral_mano_forwards']=0
    if kind=='physics':value['counts']['physics_steps']=1
    if kind=='boolean_attempt':counts['physics']=False
    if kind=='nonzero_attempt':counts['physics']=1
    with pytest.raises(ValueError):driver.check_completion(value,counts,'a'*64,'b'*64)

@pytest.mark.parametrize('api,kind',[('from_xml_path','MuJoCo_model_attempts'),('from_xml_string','MuJoCo_model_attempts'),
    ('from_binary_path','MuJoCo_model_attempts'),('mj_forward','MuJoCo_forward_attempts'),('mj_step','physics_step_attempts'),
    ('mj_step1','physics_step_attempts'),('mj_step2','physics_step_attempts')])
def test_guards_refuse_before_model_or_integration_call(monkeypatch,api,kind):
    # Restore the process-wide extension module after this control.
    original={key:getattr(driver.mujoco,key) for key in ('MjModel','mj_forward','mj_step','mj_step1','mj_step2')}
    counts=dict(MuJoCo_model_attempts=0,MuJoCo_forward_attempts=0,physics_step_attempts=0)
    try:
        driver.install_guards(counts)
        target=getattr(driver.mujoco.MjModel,api) if api.startswith('from_') else getattr(driver.mujoco,api)
        with pytest.raises(RuntimeError,match='forbids'):target('manufactured')
        assert counts[kind]==1 and sum(counts.values())==1
    finally:
        for key,value in original.items():setattr(driver.mujoco,key,value)
