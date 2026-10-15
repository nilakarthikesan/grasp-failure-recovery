"""Manufactured receipt/admission tests; no native model, geometry or solve."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
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


@pytest.fixture
def prior_build(tmp_path,monkeypatch):
    root=tmp_path/'project';here=root/'v2';directory=here/'build_run_v1';raw=directory/'raw';raw.mkdir(parents=True)
    monkeypatch.setattr(driver,'ROOT',root);monkeypatch.setattr(driver,'HERE',here)
    protocol=here/driver.BUILD_PROTOCOL_NAME;plan=dict(fixed='literalplan')
    protocol.write_text(json.dumps(dict(plan=plan,bindings={},tracked_sources=[])))
    import hashlib
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    artifact=raw/'producer.npz';np.savez_compressed(artifact,x=[1.,2.])
    row=dict(path=str(artifact),sha256=driver.sha(artifact),bytes=artifact.stat().st_size,label='producer')
    builder=dict(build_protocol_sha256=driver.sha(protocol),plan_sha256=digest(plan),artifacts=[row])
    builder_path=directory/'builder_report.json';builder_path.write_text(json.dumps(builder));(raw/'build_report.json').write_bytes(builder_path.read_bytes())
    freeze='a'*40;worker=directory/'worker_result.json';worker.write_text(json.dumps(dict(status='COMPLETE',
        protocol_sha256=driver.sha(protocol),freeze_commit=freeze,builder_report_sha256=driver.sha(builder_path))))
    report=directory/'report.json';report.write_text(json.dumps(dict(status='COMPLETE',protocol_sha256=driver.sha(protocol),
        freeze_commit=freeze,worker_result_sha256=driver.sha(worker))))
    receipt=directory/'independent_readback.json';receipt.write_text(json.dumps(dict(
        schema_version='manosim_exact_shape_build_independent_saved_readback_v1',
        status='VERIFIED_LITERAL_SHAPE_BUILD_ARTIFACTS_AND_PRODUCER',build_protocol_sha256=driver.sha(protocol),freeze_commit=freeze,
        transaction_report_sha256=driver.sha(report),worker_result_sha256=driver.sha(worker),builder_report_sha256=driver.sha(builder_path),
        all_bindings_verified_before=True,all_bindings_verified_after=True,expected_bindings=0,
        artifact_inventory={str(artifact):{k:row[k] for k in ('sha256','bytes')}})))
    generated=dict(source_root=str(raw),expected_shape_sha256='b'*64)
    monkeypatch.setattr(driver,'load',lambda *a:SimpleNamespace(digest=digest,generated_asset_selection=lambda path:generated,verify_plan=lambda p:None))
    return SimpleNamespace(protocol=protocol,builder=builder_path,raw=raw,report=report,worker=worker,receipt=receipt,generated=generated)


def test_complete_build_receipt_cannot_substitute_earlier_preparation(prior_build):
    _,selection,evidence=driver.completed_build()
    assert selection==prior_build.generated and prior_build.receipt in evidence
    # A different unused declaration keeps the same plan but changes actual bytes.
    p=json.loads(prior_build.protocol.read_text());p['unused_declaration']=True;prior_build.protocol.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='frozen shape build'):driver.completed_build()


@pytest.mark.parametrize('kind',['protocol','freeze','worker','builder','unverified','count','missing_array','array_size','schema','raw_report'])
def test_build_requires_exact_independent_saved_artifact_and_freeze_identity(prior_build,kind):
    receipt=json.loads(prior_build.receipt.read_text())
    if kind=='protocol':receipt['build_protocol_sha256']='0'*64
    if kind=='freeze':receipt['freeze_commit']='0'*40
    if kind=='worker':receipt['worker_result_sha256']='0'*64
    if kind=='builder':receipt['builder_report_sha256']='0'*64
    if kind=='unverified':receipt['all_bindings_verified_after']=False
    if kind=='count':receipt['expected_bindings']=True
    if kind=='missing_array':receipt['artifact_inventory']={}
    if kind=='array_size':next(iter(receipt['artifact_inventory'].values()))['bytes']+=1
    if kind=='schema':receipt['schema_version']='other'
    if kind=='raw_report':(prior_build.raw/'build_report.json').write_text('not exact-copy producer bytes')
    prior_build.receipt.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):driver.completed_build()


def test_git_sources_partition_does_not_reclassify_untracked_preserved_inputs(tmp_path,monkeypatch):
    root=tmp_path/'project';raw=root/'v2/build_run_v1/raw';monkeypatch.setattr(driver,'ROOT',root)
    own=root/'new_audit.py';old=root/'old_bound_source.py';untracked=root/'contactgen_quality.py';ignored=root/'runtime/hand.xml';asset=raw/'assets/hand.xml';external=tmp_path/'dependency.py'
    bindings={str(p):'a'*64 for p in (own,old,untracked,ignored,asset,external)}
    tracked,generated=driver.source_partitions(bindings,raw,[str(own),str(asset),str(external)],[str(old)])
    assert tracked==sorted([str(own),str(old)]) and generated=={str(asset):'a'*64}
    assert str(untracked) in bindings and str(untracked) not in tracked
    assert str(ignored) in bindings and str(ignored) not in tracked
    with pytest.raises(ValueError,match='missing byte binding'):
        driver.source_partitions(bindings,raw,[str(root/'missing.py')],[])


def test_failure_after_saved_arrays_retains_only_actual_creation_time_receipts(tmp_path):
    file=tmp_path/'decoded_native_reference.npz';np.savez_compressed(file,x=[1.,2.])
    original=dict(path=str(file),sha256=driver.sha(file),bytes=file.stat().st_size,label='saved_before_failure')
    # A later interrupted write has no creation-time binding and cannot be
    # retroactively relabelled as verified output.
    (tmp_path/'partial_variant.npz').write_bytes(b'interrupted NPZ')
    case=SimpleNamespace(checkpoint_artifacts=[],auxiliary_artifacts=[copy.deepcopy(original)])
    receipt=driver.partial_artifact_receipt(tmp_path,case)
    assert receipt['status']=='VERIFIED_RECORDED_ROWS_ONLY' and receipt['complete_array_inventory'] is False
    assert receipt['recorded_rows']==receipt['verified_rows']==[original]
    assert receipt['unindexed_array_paths']==[str(tmp_path/'partial_variant.npz')]
    file.write_bytes(b'changed after original write')
    bad=driver.partial_artifact_receipt(tmp_path,case)
    assert bad['status']=='RECORDED_ROW_VERIFICATION_ERRORS' and bad['verified_rows']==[]
    assert bad['recorded_rows']==[original] and bad['verification_errors']
    assert case.auxiliary_artifacts==[original]
