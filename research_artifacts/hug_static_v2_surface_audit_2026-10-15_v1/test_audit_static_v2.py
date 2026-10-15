"""Manufactured audit controls. Never run a model, solver, FK or physics."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('manufactured_static_v2_audit',HERE/'audit_static_v2.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


@pytest.fixture(autouse=True)
def no_actual_work(monkeypatch):
    def forbidden(*a,**k):pytest.fail('No native solve, forward or physics in manufactured audit tests')
    for name in ('mj_step','mj_step1','mj_step2','mj_forward'):monkeypatch.setattr(mujoco,name,forbidden)


def production():
    configuration=dict(profile_id=audit.PROFILE,config_path='/manufactured/config.yml',config_sha256='a'*64,
        vector_scaling=1.,low_pass_filter_enabled=False,temporal_norm_delta=0.,reset_before_proposal=True,
        refinement_calls=20,supported_optimizer_status_codes=[1,2,3,4,5,6])
    names=tuple(f'j{i}' for i in range(24));q=np.zeros(24);statuses=[6]*20
    pose=SimpleNamespace(joint_names=names,qpos=q,landmarks_wrist_m=np.zeros((21,3)),reference_vectors_m=np.zeros((10,3)))
    record=dict(schema_version='hug_mapping_record_v1',translation_strategy_id=audit.STRATEGY,outcome=None,
        fresh_instance_per_proposal=True,status='returned',mapping_status='ok',optimizer_status_gate_passed=True,
        raw_refinement_qpos_rad=np.zeros((20,24)).tolist(),optimizer_status_codes=statuses,
        joint_names=list(names),qpos_rad=q.tolist(),joint_limits_rad=np.tile([-1.,1.],(24,1)).tolist(),
        max_final_clipping_rad=0.,provenance={**{k:v for k,v in configuration.items() if k!='supported_optimizer_status_codes'},
            'optimizer_status_codes':statuses})
    return pose,record,configuration


def test_returned_budget_termination_is_recorded_without_claiming_convergence():
    pose,record,config=production();gate=audit.validate_mapping_record(pose,record,config,20)
    assert gate['output_available'] and gate['optimizer_convergence_not_implied'] and gate['physical_outcome'] is None


@pytest.mark.parametrize('kind',['bool_status','bad_status','missing_status','wrong_profile','changed_q','clipping'])
def test_forged_status_profile_and_command_distinctions_refuse(kind):
    pose,record,config=production()
    if kind=='bool_status':record['optimizer_status_codes'][0]=True
    if kind=='bad_status':record['optimizer_status_codes'][0]=99
    if kind=='missing_status':record['optimizer_status_codes'].pop()
    if kind=='wrong_profile':record['provenance']['profile_id']='legacy_shadow_vector_v1'
    if kind=='changed_q':record['qpos_rad'][0]=1e-12
    if kind=='clipping':record['max_final_clipping_rad']=1e-12
    with pytest.raises(ValueError):audit.validate_mapping_record(pose,record,config,20)


def test_actual_clipping_is_explicit_and_not_confused_with_raw_command():
    pose,record,config=production();raw=np.array(record['raw_refinement_qpos_rad']);raw[-1,0]=1.0001
    pose.qpos[0]=1.;record.update(qpos_rad=pose.qpos.tolist(),raw_refinement_qpos_rad=raw.tolist(),
                               max_final_clipping_rad=float(raw[-1,0]-1.))
    assert audit.validate_mapping_record(pose,record,config,20)['max_final_clipping_rad']>0


def test_nonfinite_and_ragged_refused_returns_are_preserved_before_validation(tmp_path):
    _,record,config=production();record.update(status='refused',mapping_status='refused_invalid_retarget',
        raw_refinement_qpos_rad=[np.full(24,np.nan).tolist(),np.zeros(23).tolist()],optimizer_status_codes=[1,1])
    binding=audit.save_production_evidence(tmp_path,record,{},None)
    saved=audit.read(binding['path']);assert saved['physical_outcome'] is None
    assert saved['mapping_record']['raw_refinement_qpos_rad'][0][0]=={'nonfinite_float':'nan'}
    with np.load(binding['trace_npz']['path'],allow_pickle=False) as z:
        assert np.isnan(z['raw_refinement_qpos_rad_00']).all() and z['raw_refinement_qpos_rad_01'].shape==(23,)
    gate=audit.validate_mapping_record(None,record,config,2)
    assert not gate['output_available'] and not gate['raw_trace_finite'] and gate['physical_outcome'] is None
    assert len(audit.production_evidence_index(binding,tmp_path))==2


def test_production_numeric_trace_tampering_cannot_be_rebound_at_completion(tmp_path):
    pose,record,_=production();binding=audit.save_production_evidence(tmp_path,record,{},pose)
    Path(binding['trace_npz']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='observation changed'):audit.production_evidence_index(binding,tmp_path)


def preprocessing():
    grasp=SimpleNamespace(landmarks_3d=np.zeros((21,3)),T_camera_wrist=np.eye(4))
    indices=np.array([np.zeros(10,int),np.arange(10)])
    trace=dict(operator_landmarks_m=[np.zeros((21,3))],reference_vectors_m=[np.zeros((10,3))],
        target_link_human_indices=[indices],actual_ref_value=[np.zeros((10,3),np.float32) for _ in range(20)])
    return grasp,trace


def test_actual_float32_vectors_indices_and_returned_preprocessor_fields_match():
    grasp,trace=preprocessing();pose,_,_=production()
    out=audit.preprocessing_receipt(grasp,pose,trace,20)
    assert out['passed'] and out['actual_solver_input_dtype']=='float32' and out['scalar_operator_landmark_max_difference_m']==0
    trace['actual_ref_value'][3]=np.zeros((10,3),np.float64)
    with pytest.raises(ValueError,match='preprocessing/reference'):audit.preprocessing_receipt(grasp,pose,trace,20)


def test_partial_refused_preprocessor_trace_is_not_called_pass():
    grasp,_=preprocessing();receipt=audit.preprocessing_receipt(grasp,None,{},0)
    assert not receipt['passed'] and receipt['status']=='PARTIAL_REFUSED_PREPROCESSING'


class FakeAdapter:
    def __init__(self):
        self.source_urdf_sha256='source';self.model_geometry_sha256='model';self.palm_body_id=1
        self.palm_position0=np.zeros(3);self.root_quaternion0=np.array([1.,0.,0.,0.]);self.calls=0
        self.data=SimpleNamespace(xpos=np.zeros((2,3)))
    def _named_q(self,value):return np.array([value[f'j{i}'] for i in range(24)])
    def _restore(self):pass
    def _targets(self,pos,quat,q):self.pos=pos.copy();self.quat=quat.copy();self.q=q.copy()
    def _forward(self):self.calls+=1
    def _domains(self,q,shift,omega):return True,False
    def _readback(self,q,physical,engineering):
        return SimpleNamespace(joint_qpos_rad=q.copy(),root_position_m=self.pos.copy(),root_quaternion_wxyz=self.quat.copy(),
            engineering_domain_pass=engineering,physical_joint_limits_pass=physical)


def payload():
    return dict(schema_version='shadow_current_static_named_root_pose_v1',joint_targets_rad={f'j{i}':1. for i in range(24)},
        root_position_m=[.1,.2,.3],root_quaternion_wxyz=[1.,0.,0.,0.],source_urdf_sha256='source',model_geometry_sha256='model')


def test_diagnostic_literal_measures_outside_domain_without_clamp_or_gate_change():
    from grasp_failure_prediction.evaluation import surface_pose
    adapter=FakeAdapter();out=audit.diagnostic_literal(adapter,payload(),surface_pose)
    assert not out.engineering_domain_pass and out.physical_joint_limits_pass and adapter.calls==1
    np.testing.assert_array_equal(out.joint_qpos_rad,np.ones(24))
    np.testing.assert_array_equal(out.root_position_m,[.1,.2,.3])


def test_diagnostic_literal_refuses_changed_root_serialization_and_source():
    from grasp_failure_prediction.evaluation import surface_pose
    adapter=FakeAdapter();original=adapter._readback
    def changed(*a):
        value=original(*a);value.root_position_m[0]=np.nextafter(value.root_position_m[0],np.inf);return value
    adapter._readback=changed
    with pytest.raises(ValueError,match='literal command changed'):audit.diagnostic_literal(adapter,payload(),surface_pose)
    p=payload();p['source_urdf_sha256']='wrong'
    with pytest.raises(ValueError,match='source/schema mismatch'):audit.diagnostic_literal(FakeAdapter(),p,surface_pose)


def full():
    return dict(fields={'required':np.ones((1,6,3),bool)},distance_m=np.zeros((1,6)),margin_m=np.zeros((1,6,3)),
        orientation=dict(valid=np.array([True]),passed=np.array([False]),score=np.array([-.5]),distance_m=np.array([.001])),
        state=SimpleNamespace(joint_qpos_rad=np.zeros(24),root_position_m=np.zeros(3),root_quaternion_wxyz=np.array([1.,0.,0.,0.]),tip_error_norms_m=np.zeros(5)))


@pytest.mark.parametrize('what',['field','orientation','root','distance'])
def test_fresh_literal_readback_checks_full_arrays_and_root_not_only_scalar_rank(what):
    a=full();b=copy.deepcopy(a)
    if what=='field':b['fields']['required'][0,0,0]=False
    if what=='orientation':b['orientation']['score'][0]=-.6
    if what=='root':b['state'].root_position_m[0]=1e-12
    if what=='distance':b['distance_m'][0,0]=1e-12
    with pytest.raises(ValueError,match='Fresh literal readback'):audit.compare_readback(a,b)


def test_missing_worker_result_makes_attempt_total_unknown_not_zero():
    rows=[dict(index=20,status='TIME_BUDGET_EXCEEDED',result=None),dict(index=28,status='NOT_RUN')]
    x=audit.attempt_denominator(rows)
    assert x['actual_dex_refinement_attempts'] is None and not x['attempt_total_known']
    assert x['reported_dex_refinement_attempts_subtotal']==0 and x['conservative_dex_refinement_attempts_upper_bound']==20


def test_exact_completed_and_partial_worker_attempt_counts_remain_visible(tmp_path):
    rows=[]
    for i,value in ((20,20),(28,7)):
        path=tmp_path/f'{i}.json';audit.write(path,dict(counts=dict(dex_refinement_attempts=value)))
        rows.append(dict(index=i,status='COMPLETED',result=str(path),result_sha256=audit.sha(path)))
    x=audit.attempt_denominator(rows)
    assert x['actual_dex_refinement_attempts']==27 and x['attempt_total_known']


def test_unused_prior_protocol_cannot_be_labeled_as_executed(tmp_path):
    protocol=tmp_path/'protocol.json';audit.write(protocol,dict(schema_version='manufactured'))
    report=tmp_path/'report.json';audit.write(report,dict(status='COMPLETED_BOUNDED_PILOT',native_fidelity_accepted=False,
        new_collection_or_training_allowed=False,protocol_sha256='wrong',freeze_commit='a'*40,cases=[dict(index=20),dict(index=28)]))
    readback=tmp_path/'readback.json';audit.write(readback,dict(status='VERIFIED_SAVED_ARRAYS_AND_REPORT_INTERPRETATION',
        native_fidelity_accepted=False,report_sha256=audit.sha(report),protocol_sha256='wrong',freeze_commit='a'*40,
        cases=[dict(index=20),dict(index=28)]))
    with pytest.raises(ValueError,match='Complete frozen v3 evidence'):audit.completed_v3(report,readback,protocol)


def test_diagnostic_report_cannot_manufacture_mapping_pass():
    with pytest.raises(ValueError,match='cannot issue mapping acceptance'):
        audit.util.no_mapping_acceptance(dict(accepted_mapping=True))


def test_case_caps_refuse_before_fourth_scan_or_151st_raw_execution():
    calls=[]
    case=SimpleNamespace(counts={'raw_callbacks':149},checkpoint_artifacts=[])
    def fake_full(*a,**k):calls.append(True);case.checkpoint_artifacts.append({})
    case._full_current=fake_full;audit.install_case_budgets(case)
    case._raw_budget()
    with pytest.raises(RuntimeError,match='150 raw'):case._raw_budget()
    assert case.counts['raw_callbacks']==150
    for _ in range(3):case._full_current()
    with pytest.raises(RuntimeError,match='before query'):case._full_current()
    assert len(calls)==3


def test_all_checkpoint_write_time_hashes_are_checked_before_publication(tmp_path,monkeypatch):
    monkeypatch.setattr(audit.util,'visual_artifact_index',lambda *a:[])
    monkeypatch.setattr(audit,'production_evidence_index',lambda *a:[])
    path=tmp_path/'checkpoint_01_original.npz'
    with path.open('xb') as f:np.savez(f,manufactured=np.zeros(1))
    receipts=[dict(path=str(path),sha256=audit.sha(path),bytes=path.stat().st_size,label='original')]
    assert len(audit.checkpoint_index(tmp_path,receipts,{}, {},20)[0])==1
    with path.open('wb') as f:np.savez(f,manufactured=np.ones(1))
    with pytest.raises(ValueError,match='Write-time full-field checkpoint changed'):
        audit.checkpoint_index(tmp_path,receipts,{}, {},20)
