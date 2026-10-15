"""Manufactured transaction/guard controls; no native data or geometry queries."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('saved_geometry_ablation_controls',Path(__file__).with_name('run_ablation.py'))
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def complete_counts():
    return dict(full_field_scans=9,triangle_point_tests=9,model_calls=0,FK_engine_calls=0,
        physics_steps=0,HUG_generator_calls=0,retarget_calls=0,predictor_fits=0)
def result():
    return dict(schema_version='manosim_saved_geometry_construction_ablation_case_v1',index=20,counts=complete_counts(),
        reference_control=dict(passed=True,exact_masks=True,max_distance_error_m=1e-13),
        stages=[dict(stage_id=stage,surface_fidelity_status='FAIL') for stage in driver.VARIANTS],
        accepted_mapping=False,native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False)
def budget():
    b=driver.QueryBudget(100)
    for stage in driver.VARIANTS:b('full_field_scan',stage);b('triangle_point_tests',1)
    return b


def test_exact_fixed_nine_stage_contract():
    assert len(driver.VARIANTS)==9 and driver.BUDGET['maximum_seconds_per_case']==600
    assert driver.BUDGET['maximum_triangle_point_tests_per_query_cell']==18620
    driver.check_result(result(),{key:0 for key in driver.FORBIDDEN_KEYS},complete_counts(),budget())


@pytest.mark.parametrize('kind,value',[('full_field_scan','unexpected'),('triangle_point_tests',1),('other',1)])
def test_before_work_guards_reject_undeclared_operations(kind,value):
    b=driver.QueryBudget(10)
    with pytest.raises((ValueError,RuntimeError)):b(kind,value)
    assert b.stages==[] and b.triangle_point_tests==0


@pytest.mark.parametrize('value',[True,0,-1,1.5,11])
def test_overcap_or_noninteger_query_is_refused_before_count_or_work(value):
    b=driver.QueryBudget(10);b('full_field_scan',driver.VARIANTS[0])
    with pytest.raises((ValueError,RuntimeError)):b('triangle_point_tests',value)
    assert b.triangle_point_tests==0


def test_exact_stage_order_single_occurrence_and_exhaustion():
    b=budget()
    with pytest.raises(RuntimeError):b('full_field_scan',driver.VARIANTS[0])
    assert b.stages==list(driver.VARIANTS)
    b=driver.QueryBudget(10);b('full_field_scan',driver.VARIANTS[0])
    with pytest.raises(RuntimeError):b('full_field_scan',driver.VARIANTS[0])


@pytest.mark.parametrize('kind',['surface_pass','promote','task_label','reference_error','reference_masks','reference_missing','missing_stage','reordered_stage','counter','forbidden'])
def test_complete_case_cannot_loosen_identity_or_surface_admission(kind):
    v=result();c=complete_counts();f={key:0 for key in driver.FORBIDDEN_KEYS}
    if kind=='surface_pass':v['stages'][0]['surface_fidelity_status']='PASS'
    elif kind=='promote':v['stages'][3]['accepted_mapping']=True
    elif kind=='task_label':v['physical_outcome']='success'
    elif kind=='reference_error':v['reference_control']['max_distance_error_m']=1.01e-12
    elif kind=='reference_masks':v['reference_control']['exact_masks']=False
    elif kind=='reference_missing':v.pop('reference_control')
    elif kind=='missing_stage':v['stages'].pop()
    elif kind=='reordered_stage':v['stages'].reverse()
    elif kind=='counter':c['triangle_point_tests']+=1
    else:f['MuJoCo_FK_attempts']=1
    with pytest.raises(ValueError):driver.check_result(v,f,c,budget())


def test_exact_two_output_roots_and_no_alias(tmp_path,monkeypatch):
    root=tmp_path/'project';here=root/'research_artifacts/manosim_geometry_ablation_2026-10-15_v1'
    monkeypatch.setattr(driver,'ROOT',root);monkeypatch.setattr(driver,'HERE',here)
    p=dict(output_root=str(here/'run_v1'),raw_root=str(root/'runs/manosim_geometry_ablation_2026-10-15_v1/run_v1/raw'))
    assert driver.permitted_roots(p,here/'run_v1')==(here/'run_v1',Path(p['raw_root']))
    with pytest.raises(ValueError):driver.permitted_roots(p,here/'other')
    p['raw_root']=str(tmp_path/'outside')
    with pytest.raises(ValueError):driver.permitted_roots(p,here/'run_v1')


def test_git_source_selection_keeps_input_data_hash_only(tmp_path,monkeypatch):
    root=tmp_path/'project';monkeypatch.setattr(driver,'ROOT',root)
    raw=root/'runs/manosim_geometry_ablation_2026-10-15_v1/run_v1/raw'
    own=root/'new.py';inherited=root/'old.py';untracked=root/'preserved_untracked.py'
    decoded=root/'prior/run_v1/20/decoded_native_reference.npz';producer=root/'prior/build_run_v1/raw/source.py'
    bindings={str(path):'a'*64 for path in (own,inherited,untracked,decoded,producer)}
    selected=driver.source_partitions(bindings,[str(own),str(decoded),str(producer)],[str(inherited)],raw)
    assert selected==sorted([str(own),str(inherited)]) and str(untracked) not in selected
    with pytest.raises(ValueError):driver.source_partitions(bindings,[str(root/'missing.py')],[],raw)


def artifact(path,label='array'):
    return dict(path=str(path),sha256=driver.sha(path),bytes=path.stat().st_size,label=label)


@pytest.mark.parametrize('kind',['valid','unindexed','changed','duplicate','outside','symlink','bytes_bool'])
def test_two_root_creation_time_inventory_is_complete(tmp_path,kind):
    tracked=tmp_path/'tracked';raw=tmp_path/'raw';tracked.mkdir();raw.mkdir()
    f=raw/'field.npz';np.savez_compressed(f,x=[1.,2.]);rows=[artifact(f)]
    if kind=='unindexed':np.savez_compressed(raw/'extra.npz',x=[1])
    elif kind=='changed':f.write_bytes(b'changed')
    elif kind=='duplicate':rows.append(dict(rows[0]))
    elif kind=='outside':g=tmp_path/'outside.npz';g.write_bytes(f.read_bytes());rows[0]=artifact(g)
    elif kind=='symlink':g=raw/'actual.npz';f.rename(g);f.symlink_to(g)
    elif kind=='bytes_bool':rows[0]['bytes']=True
    if kind=='valid':assert driver.artifact_index((tracked,raw),rows,complete=True)==rows
    else:
        with pytest.raises(ValueError):driver.artifact_index((tracked,raw),rows,complete=True)


def test_failed_partial_rows_never_get_rebased_to_mutated_bytes(tmp_path):
    f=tmp_path/'field.npz';np.savez_compressed(f,x=[1.,2.]);row=artifact(f)
    case=SimpleNamespace(artifact_rows=[copy.deepcopy(row)])
    (tmp_path/'interrupted.npz').write_bytes(b'incomplete')
    good=driver.partial_artifact_receipt((tmp_path,),case)
    assert good['recorded_rows']==good['verified_rows']==[row] and not good['complete_array_inventory']
    f.write_bytes(b'changed after write')
    bad=driver.partial_artifact_receipt((tmp_path,),case)
    assert bad['recorded_rows']==[row] and bad['verified_rows']==[] and bad['verification_errors']
    assert case.artifact_rows==[row]


@pytest.fixture
def native(tmp_path,monkeypatch):
    root=tmp_path/'project';native=root/'native';run=native/'run_v1';run.mkdir(parents=True)
    monkeypatch.setattr(driver,'ROOT',root);monkeypatch.setattr(driver,'NATIVE',native)
    source=root/'fixed_input';source.write_bytes(b'unchanged native/geometry input')
    protocol=native/'native_protocol_v2.json';protocol.write_text(json.dumps(dict(schema_version='hug_literal_shape_manosim_static_transfer_v2',
        bindings={str(source):driver.sha(source)},indices=[20,28])))
    monkeypatch.setattr(driver,'NATIVE_PROTOCOL_SHA256',driver.sha(protocol));monkeypatch.setattr(driver,'NATIVE_BINDING_COUNT',1)
    freeze='a'*40;rows=[];checked=[];inventory={}
    for index in (20,28):
        folder=run/str(index);folder.mkdir();f=folder/'measured.npz';np.savez_compressed(f,x=[index])
        a=artifact(f);r=folder/'result.json';r.write_text(json.dumps(dict(status='COMPLETED_NATIVE_STATIC_DIAGNOSTIC',
            protocol_sha256=driver.sha(protocol),freeze_commit=freeze,artifacts=[a],
            measurement=dict(kinematic_pose_transfer_status='PASS',surface_fidelity_status='FAIL'))))
        rows.append(dict(index=index,status='COMPLETED',returncode=0,result=str(r),result_sha256=driver.sha(r)))
        checked.append(dict(index=index,status='VERIFIED_COMPLETED_STATIC_DIAGNOSTIC',result_sha256=driver.sha(r),kinematic_gate='PASS',surface_gate='FAIL'))
        for file in (f,r):inventory[str(file)]={key:artifact(file)[key] for key in ('sha256','bytes')}
    report=run/'report.json';report.write_text(json.dumps(dict(status='COMPLETED_NATIVE_STATIC_PILOT',protocol_sha256=driver.sha(protocol),freeze_commit=freeze,cases=rows)))
    inventory[str(report)]={key:artifact(report)[key] for key in ('sha256','bytes')}
    receipt=run/'independent_readback_v2.json';receipt.write_text(json.dumps(dict(schema_version='manosim_literal_shape_native_static_independent_saved_readback_v2',
        status='VERIFIED_COMPLETED_NATIVE_STATIC_SAVED_ARRAYS',active_protocol_sha256=driver.sha(protocol),freeze_commit=freeze,
        expected_bindings=1,all_bindings_verified_before=True,all_bindings_verified_after=True,
        actual_execution_status='COMPLETED_NATIVE_STATIC_PILOT',errors=[],artifact_inventory=inventory,cases=checked)))
    monkeypatch.setattr(driver,'NATIVE_READBACK_SHA256',driver.sha(receipt))
    return SimpleNamespace(root=root,source=source,protocol=protocol,report=report,receipt=receipt)


def test_admission_uses_exact_final_receipt_and_every_saved_byte(native):
    p,evidence=driver.completed_native();assert len(p['bindings'])==1 and native.receipt in evidence
    native.source.write_bytes(b'changed inherited source')
    with pytest.raises(ValueError):driver.completed_native()


@pytest.mark.parametrize('kind',['protocol','receipt','report','array','result'])
def test_no_replacement_of_prior_output_or_unused_declaration(native,kind):
    file={'protocol':native.protocol,'receipt':native.receipt,'report':native.report,
        'array':native.root/'native/run_v1/20/measured.npz','result':native.root/'native/run_v1/28/result.json'}[kind]
    file.write_bytes(file.read_bytes()+b'\n')
    with pytest.raises(ValueError):driver.completed_native()


@pytest.mark.parametrize('kind',['status','freeze','count','before','case_digest','case_pose'])
def test_semantically_incomplete_or_forged_prior_verdict_refused(native,monkeypatch,kind):
    r=json.loads(native.receipt.read_text())
    if kind=='status':r['status']='earlier failed verifier'
    elif kind=='freeze':r['freeze_commit']='b'*40
    elif kind=='count':r['expected_bindings']=True
    elif kind=='before':r['all_bindings_verified_before']=False
    elif kind=='case_digest':r['cases'][0]['result_sha256']='0'*64
    else:r['cases'][0]['kinematic_gate']='FAIL'
    native.receipt.write_text(json.dumps(r));monkeypatch.setattr(driver,'NATIVE_READBACK_SHA256',driver.sha(native.receipt))
    with pytest.raises(ValueError):driver.completed_native()


@pytest.mark.parametrize('name,key',[('MjModel','MuJoCo_model_attempts'),('MjData','MuJoCo_data_attempts'),
    ('mj_forward','MuJoCo_forward_attempts'),('mj_kinematics','MuJoCo_FK_attempts'),('mj_step','physics_step_attempts')])
def test_model_engine_and_physics_APIs_are_blocked_without_running_them(monkeypatch,name,key):
    fake=SimpleNamespace(MjModel=object,MjData=object,mj_forward=lambda:None,mj_kinematics=lambda:None,mj_step=lambda:None)
    monkeypatch.setitem(sys.modules,'mujoco',fake)
    monkeypatch.setitem(sys.modules,'grasp_failure_prediction.evaluation.retargeting',None)
    counts={k:0 for k in driver.FORBIDDEN_KEYS};driver.install_guards(counts)
    with pytest.raises(RuntimeError):getattr(fake,name)()
    assert counts[key]==1


@pytest.mark.parametrize('module,key',[('torch','MANO_import_attempts'),('manotorch.manolayer','MANO_import_attempts'),
    ('hug.models','HUG_generator_attempts'),('sklearn.linear_model','training_attempts')])
def test_model_or_training_import_refused_before_import(module,key):
    counts={k:0 for k in driver.FORBIDDEN_KEYS};guard=driver.ForbiddenImports(counts)
    with pytest.raises(RuntimeError):guard.find_spec(module)
    assert counts[key]==1


def test_worker_failure_preserves_original_rows_and_identical_error_results(tmp_path,monkeypatch):
    tracked=tmp_path/'tracked';raw=tmp_path/'raw';tracked.mkdir();raw.mkdir()
    p=dict(case_limits={'20':dict(maximum_triangle_point_tests=100)})
    monkeypatch.setattr(driver,'validate',lambda *a:(p,'a'*40))
    monkeypatch.setattr(driver,'permitted_roots',lambda *a:(tracked,raw))
    monkeypatch.setattr(driver,'install_guards',lambda *a:None)
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    class Case:
        def __init__(self,index,p,out,**kwargs):
            self.out=out;self.artifact_rows=[];self.counts=complete_counts()
        def run(self):
            f=self.out/'field.npz';np.savez_compressed(f,x=[1.,2.]);self.artifact_rows.append(artifact(f))
            (self.out/'interrupted.npz').write_bytes(b'incomplete')
            raise RuntimeError('manufactured failure after creation-time publication')
    monkeypatch.setattr(driver,'load',lambda *a:SimpleNamespace(AblationCase=Case))
    code=driver.worker(protocol,'a'*40,tracked,20)
    assert code==1
    result=json.loads((tracked/'20/result.json').read_text())
    assert result['status']=='HARD_STOP' and result['measurement'] is None
    assert result['failure']['type']=='RuntimeError'
    partial=result['partial_artifacts']
    assert partial['recorded_rows']==partial['verified_rows'] and len(partial['verified_rows'])==1
    assert partial['unindexed_array_paths']==[str(raw/'20/interrupted.npz')]
    assert (tracked/'20/result.json').read_bytes()==(raw/'20/result.json').read_bytes()
