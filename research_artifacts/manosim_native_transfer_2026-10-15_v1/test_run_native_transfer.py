"""Manufactured receipt/admission tests; no native model, geometry or solve."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest
import numpy as np

spec=importlib.util.spec_from_file_location('native_transfer_harness_controls',Path(__file__).with_name('run_native_transfer.py'))
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

def verdict():
    return dict(kinematic_pose_transfer_status='PASS',surface_fidelity_status='FAIL',
        native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False)
def counts():return dict(static_forwards=20,physics_step_attempts=0,HUG_generator_attempts=0,official_retarget_attempts=0)
def case_counts():return dict(parametric_mano_forwards=2,full_field_variant_scans=2,collider_point_queries=1)

def test_pose_pass_is_separate_from_surface_fidelity():
    driver.check_result(verdict(),counts(),case_counts())
    v=verdict();v['kinematic_pose_transfer_status']='FAIL';v['surface_fidelity_status']='INCOMPLETE'
    driver.check_result(v,counts(),case_counts())

@pytest.mark.parametrize('change',['surface_pass','missing_kinematic','promotion','nested_pass','task_label'])
def test_no_pose_only_or_nested_physical_admission(change):
    v=verdict()
    if change=='surface_pass':v['surface_fidelity_status']='PASS'
    if change=='missing_kinematic':v.pop('kinematic_pose_transfer_status')
    if change=='promotion':v['production_promotion']=True
    if change=='nested_pass':v['variant']=[dict(accepted_mapping=True)]
    if change=='task_label':v['variant']=dict(task_outcome='success')
    with pytest.raises(ValueError):driver.check_result(v,counts(),case_counts())

@pytest.mark.parametrize('key,value',[('static_forwards',201),('static_forwards',True),('static_forwards',0),
    ('physics_step_attempts',1),('HUG_generator_attempts',1),('official_retarget_attempts',1)])
def test_operation_budget_not_an_optimizer_success_flag(key,value):
    c=counts();c[key]=value
    with pytest.raises(ValueError):driver.check_result(verdict(),c,case_counts())

@pytest.mark.parametrize('key,value',[('parametric_mano_forwards',3),('parametric_mano_forwards',0),
    ('full_field_variant_scans',1),('full_field_variant_scans',3),('parametric_mano_forwards',True)])
def test_both_variants_and_fixed_parametric_forward_count_required(key,value):
    c=case_counts();c[key]=value
    with pytest.raises(ValueError):driver.check_result(verdict(),counts(),c)

def test_full_array_index_is_write_time_bound(tmp_path):
    f=tmp_path/'mesh.npz';np.savez_compressed(f,x=np.array([1.,2.]))
    row=dict(path=str(f),sha256=driver.sha(f),bytes=f.stat().st_size,label='mesh')
    assert driver.artifact_index(tmp_path,[row])==[row]
    f.write_bytes(b'changed')
    with pytest.raises(ValueError):driver.artifact_index(tmp_path,[row])

@pytest.mark.parametrize('change',['unindexed','missing','duplicate','outside','symlink','bad_byte_type'])
def test_artifact_index_rejects_incomplete_or_forged_receipts(tmp_path,change):
    out=tmp_path/'out';out.mkdir();f=out/'mesh.npz';np.savez_compressed(f,x=[1])
    row=dict(path=str(f),sha256=driver.sha(f),bytes=f.stat().st_size,label='mesh');rows=[row]
    if change=='unindexed':np.savez_compressed(out/'extra.npz',x=[1])
    if change=='missing':f.unlink()
    if change=='duplicate':rows.append(dict(row))
    if change=='outside':row['path']=str(tmp_path/'other.npz');Path(row['path']).write_bytes(f.read_bytes())
    if change=='symlink':f.rename(out/'real.npz');f.symlink_to(out/'real.npz')
    if change=='bad_byte_type':row['bytes']=True
    with pytest.raises(ValueError):driver.artifact_index(out,rows)

def test_only_existing_hash_bound_native_inputs_are_admissible(tmp_path):
    f=tmp_path/'source';f.write_bytes(b'actual source');bindings={str(f):driver.sha(f)}
    driver.verify(bindings);f.write_bytes(b'changed source')
    with pytest.raises(ValueError):driver.verify(bindings)

def test_native_limits_are_numerical_not_contact_allowances():
    assert driver.NUMERICAL==dict(joint_anchor_comparison_limit_m=5e-6,rotation_element_comparison_limit=1e-6,
        saved_native_reconstruction_limit_m=5e-6,scientific_surface_acceptance_allowance=False)
    assert driver.BUDGET['maximum_parametric_mano_forwards_per_case']==2
    assert driver.BUDGET['maximum_full_field_variant_scans_per_case']==2
    assert driver.INDICES==(20,28)

@pytest.fixture
def prior(tmp_path,monkeypatch):
    root=tmp_path/'project';v3=root/'prior';(v3/'run_v1').mkdir(parents=True)
    monkeypatch.setattr(driver,'ROOT',root);monkeypatch.setattr(driver,'V3',v3)
    protocol=v3/'protocol.json';protocol.write_text(json.dumps(dict(bindings={})))
    digest=driver.sha(protocol);freeze='a'*40;rows=[];inventory={};checked=[]
    for index in (20,28):
        d=v3/'run_v1'/str(index);d.mkdir();p=d/'result.json'
        p.write_text(json.dumps(dict(protocol_sha256=digest,freeze_commit=freeze)))
        rows.append(dict(index=index,status='COMPLETED',result=str(p),result_sha256=driver.sha(p)))
        inventory[str(p.relative_to(root))]=dict(sha256=driver.sha(p),bytes=p.stat().st_size)
        checked.append(dict(index=index,result_sha256=driver.sha(p)))
    report=v3/'run_v1/report.json';report.write_text(json.dumps(dict(status='COMPLETED_BOUNDED_PILOT',protocol_sha256=digest,freeze_commit=freeze,cases=rows)))
    inventory[str(report.relative_to(root))]=dict(sha256=driver.sha(report),bytes=report.stat().st_size)
    receipt=v3/'run_v1/independent_readback.json'
    receipt.write_text(json.dumps(dict(active_protocol_sha256=digest,freeze_commit=freeze,
        status='VERIFIED_SAVED_ARRAYS_AND_REPORT_INTERPRETATION',input_source_bindings_verified_before_and_after=0,
        HEAD_remained_fixed=True,source_git_blobs_checked=True,artifact_inventory=inventory,cases=checked)))
    return v3,protocol,report,receipt

def test_prior_evidence_requires_exact_report_and_case_bytes(prior):
    driver.completed_v3()
    _,_,report,_=prior
    r=json.loads(report.read_text());r['different_output']='same protocol and freeze';report.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='independently verified'):driver.completed_v3()

@pytest.mark.parametrize('failure',['active_protocol','freeze','case_digest','inventory_digest','inventory_bytes'])
def test_unused_preparation_or_other_result_cannot_replace_active_evidence(prior,failure):
    _,_,_,receipt=prior;a=json.loads(receipt.read_text())
    if failure=='active_protocol':a['active_protocol_sha256']='0'*64
    if failure=='freeze':a['freeze_commit']='b'*40
    if failure=='case_digest':a['cases'][0]['result_sha256']='0'*64
    key=next(k for k in a['artifact_inventory'] if '/20/' in k)
    if failure=='inventory_digest':a['artifact_inventory'][key]['sha256']='0'*64
    if failure=='inventory_bytes':a['artifact_inventory'][key]['bytes']+=1
    receipt.write_text(json.dumps(a))
    with pytest.raises(ValueError):driver.completed_v3()

@pytest.mark.parametrize('failure',[None,'wrong_revision','dirty_tracked_file'])
def test_upstream_pin_requires_clean_tracked_measured_bytes(tmp_path,monkeypatch,failure):
    file=tmp_path/'asset.xml';file.write_text('manufactured');calls=[]
    monkeypatch.setattr(driver.subprocess,'check_output',lambda *a,**k:('0'*40 if failure=='wrong_revision' else driver.UPSTREAM)+'\n')
    def execute(command,**kwargs):
        calls.append(command)
        if failure=='dirty_tracked_file' and 'diff' in command:raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(driver.subprocess,'run',execute)
    if failure:
        with pytest.raises((ValueError,subprocess.CalledProcessError)):driver.upstream_identity(tmp_path,[file])
    else:
        driver.upstream_identity(tmp_path,[file])
        assert any('ls-files' in c and '--error-unmatch' in c for c in calls)
        assert any('diff' in c and 'HEAD' in c for c in calls)
