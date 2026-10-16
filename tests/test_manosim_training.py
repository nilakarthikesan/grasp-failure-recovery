"""Manufactured native records and SOFTWARE-ONLY CPU fitting, never native data."""
from pathlib import Path
import json

import numpy as np
import pytest
from pydantic import ValidationError

from grasp_failure_prediction.evaluation.manosim_recording import (
    NativeRecordingContract, NativeRecorder, NativeAttemptMetadata, QPOS_NAMES, QVEL_NAMES, CTRL_NAMES,
)
from grasp_failure_prediction.prediction import manosim_training as m


def H(i): return 'sha256:' + format(i, '064x')


def put(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return path


@pytest.fixture(autouse=True)
def no_native_work(monkeypatch):
    import mujoco
    def blocked(*a, **k): raise AssertionError('manufactured training controls forbid native model/FK/physics')
    for name in ('MjModel', 'MjData', 'mj_forward', 'mj_kinematics', 'mj_step', 'mj_step1', 'mj_step2'):
        monkeypatch.setattr(mujoco, name, blocked)


def contract():
    return NativeRecordingContract(assets_sha256=H(1), shape_sha256=H(2), collider_sha256=H(3), controller_sha256=H(4), compiled_model_sha256=H(5),
        controller_protocol_id='manufactured_native_controller_v1', sampling_protocol_id='manufactured_sampling_v1', control_interval_s=.04)


def recorder(c, sign=1.):
    r = NativeRecorder(c)
    for i in range(1, 24):
        q = np.r_[np.zeros(3), np.tile([sign, 0., 0., 0.], 16)]
        r.append(time_s=round(i*.04, 10), action_interval_start_s=round((i-1)*.04, 10), phase='close_fingers' if i<=21 else 'lift' if i==22 else 'hold',
            qpos_by_name=dict(zip(QPOS_NAMES,q)), qvel_by_name=dict(zip(QVEL_NAMES, np.full(51, i*.01))),
            requested_ctrl_by_name=dict.fromkeys(CTRL_NAMES, .5), effective_ctrl_by_name=dict.fromkeys(CTRL_NAMES,.4))
    return r


def collection(tmp_path, changes=None, categories=3):
    c=contract(); rows=[]; paths={}; changes=changes or {}
    for category in range(categories):
        for replicate in range(2):
            i=2*category+replicate; identity=f'a{i}'; seeds=dict(generator=3000+i,physics=0,sensor=i)
            row=dict(attempt_id=identity, category_id=f'c{category}',object_id=f'o{category}',geometry_group_id=f'g{category}',observation_group_id=f'capture{i}',
                proposal_group_id=f'proposal{i}',scenario_id=f'scenario{i}',lift_start_s=.84,contract_sha256=c.contract_sha256,capture_sha256=H(100+i),scenario_sha256=H(200+i),native_pose_sha256=H(300+i),seeds=seeds)
            rows.append(row); status='completed'; mapping={'status':'ok'}; outcome=dict(success=replicate==0,label_rule_id='native_task_label_v1')
            if replicate: outcome.update(failure_type='drop',first_failure_time_s=.9)
            change=changes.get(identity,{})
            if change.get('mapping'):
                mapping=dict(status=change['mapping'],code='manufactured_refusal',message='manufactured refusal');status='not_executed';outcome=None
            if change.get('status'):status=change['status'];outcome=None
            if 'outcome' in change:outcome=change['outcome']
            hashes=dict(code=H(6),configuration=H(7),inputs=H(8),source_profile=H(9),hand_version=H(10),label_rule=H(11),capture=row['capture_sha256'],scenario=row['scenario_sha256'],native_pose=row['native_pose_sha256'],seeds=m.canonical_sha256(seeds))
            meta=NativeAttemptMetadata(attempt_id=identity,contract_sha256=c.contract_sha256,object_id=row['object_id'],geometry_group_id=row['geometry_group_id'],observation_group_id=row['observation_group_id'],proposal_group_id=row['proposal_group_id'],mapping=mapping,execution_status=status,outcome=outcome,provenance_hashes=hashes)
            r=recorder(c) if status!='not_executed' else NativeRecorder(c)
            if status!='not_executed':r.append_diagnostics(time_s=.001, values={'contact_force':np.array([999.]),'object_height':np.array([4.])})
            r.write(tmp_path/identity,meta);paths[identity]=m.binding(tmp_path/identity/'attempt.json')
    schedule=dict(schema_version='manosim_native_task_schedule_v1',domain=m.NATIVE_DOMAIN,outcome_selection=False,source_profile_sha256=H(9),hand_version_sha256=H(10),label_rule_id='native_task_label_v1',label_rule_sha256=H(11),contracts=[c.model_dump(mode='json')],attempts=rows)
    sp=put(tmp_path/'schedule.json',schedule);cp=put(tmp_path/'collection.json',dict(schema_version='manosim_native_collection_v1',schedule=m.binding(sp),attempts=paths,qualifications={}))
    return cp,sp,c


def qualify(cp, sp, c, monkeypatch):
    root=cp.parent;schedule=m.Schedule.model_validate(m.read(sp));artifact=put(root/'native_task_evidence.json',{'software_only':True});artifacts=[m.binding(artifact)]
    identity=dict(domain=m.NATIVE_DOMAIN,profile_sha256=schedule.source_profile_sha256,hand_version_sha256=schedule.hand_version_sha256,recording_contract_sha256=c.contract_sha256,
                  controller_sha256=c.controller_sha256,collider_sha256=c.collider_sha256,label_rule_sha256=schedule.label_rule_sha256)
    protocol=put(root/'task_protocol.json',dict(scope='native_task_execution',**identity));freeze='a'*40
    independent=put(root/'task_independent.json',dict(schema_version='manosim_native_task_independent_readback_v1',status='VERIFIED_NATIVE_TASK_QUALIFICATION',protocol_sha256=m.file_sha256(protocol),freeze_commit=freeze,
        task_execution_qualified=True,artifact_bindings_sha256=m.canonical_sha256(artifacts),**identity))
    evidence=put(root/'task_qualification.json',dict(schema_version='manosim_native_task_qualification_v1',status='VERIFIED_COMPLETED_NATIVE_TASK_QUALIFICATION',scope='native_task_execution',
        task_execution_qualified=True,scientific_training_allowed=True,fixture_only=False,freeze_commit=freeze,frozen_protocol=m.binding(protocol),independent_receipt=m.binding(independent),artifacts=artifacts,**identity))
    value=m.read(cp);value['qualifications']={c.contract_sha256:m.binding(evidence)};put(cp,value)
    monkeypatch.setattr(m,'_verify_frozen_task_protocol',lambda *a:None) # manufactured evidence only; no actual admission
    return evidence


def test_native_feature_cutoff_pose_sign_and_postcutoff_invariance():
    c=contract();r=recorder(c);a=r.arrays();X,t,times=m.prefix(a,c,m.PrefixSpec(),lift_start_s=.84)
    assert X.shape==(20,169) and X.dtype==np.float32 and abs(t-.8)<1e-12 and max(times)<.84
    np.testing.assert_array_equal(X[:,67:118],a['hand_qvel'][:20].astype(np.float32));np.testing.assert_array_equal(X[:,118:],a['effective_ctrl'][:20].astype(np.float32))
    negative=recorder(c,-1).arrays();np.testing.assert_array_equal(m.prefix(negative,c,m.PrefixSpec(),lift_start_s=.84)[0],X)
    a['hand_qvel'][20:]=1e8;a['effective_ctrl'][20:]=-1e8
    np.testing.assert_array_equal(m.prefix(a,c,m.PrefixSpec(),lift_start_s=.84)[0],X)
    assert 'diagnostics' not in m.FIELDS and 'requested_ctrl' in m.feature_contract(m.PrefixSpec())['excluded_features']


@pytest.mark.parametrize('fault',['shadow','diagnostic','phase_return','history','invalid_names'])
def test_wrong_domain_layout_diagnostic_and_time_leakage_refused(fault):
    c=contract();a=recorder(c).arrays()
    if fault=='shadow':a['domain']=np.asarray('shadow_hand_right')
    elif fault=='diagnostic':a['contact_force']=np.ones(23)
    elif fault=='invalid_names':a['ctrl_names']=np.asarray(QVEL_NAMES)
    elif fault=='history':a={k:v[:19] if getattr(v,'shape',()) and len(v)==23 else v for k,v in a.items()}
    else:a['phase'][22]='close_fingers';a['phase_start_s'][22]=.88;a['phase_elapsed_s'][22]=.04
    with pytest.raises(ValueError):m.prefix(a,c,m.PrefixSpec(),lift_start_s=.84)


def test_refusal_runtime_unknown_and_early_labels_keep_denominators(tmp_path):
    cp,sp,c=collection(tmp_path,dict(a0={'mapping':'setup_error'},a1={'status':'runtime_error'},a2={'outcome':dict(success=False,failure_type='drop',label_rule_id='native_task_label_v1')},a3={'outcome':dict(success=False,failure_type='drop',label_rule_id='native_task_label_v1',first_failure_time_s=.8)},a4={'outcome':dict(success=False,failure_type='approach_collision',label_rule_id='native_task_label_v1',first_failure_time_s=.1)}))
    d=m.load_data(cp);assert len(d.rows)==1 and len(d.excluded)==5 and d.readiness['scheduled']==6
    assert set(d.readiness['exclusion_counts'])=={'mapping_or_setup_refusal','runtime_or_unexecuted','unknown_failure_time','already_failed_at_or_before_cutoff','already_failed_approach_collision'}
    assert not d.readiness['scientific_training_allowed'] and not d.readiness['adequacy_guaranteed']


def test_exact_scheduled_capture_scenario_seed_and_refusal_accounting(tmp_path):
    cp,sp,c=collection(tmp_path);d=m.load_data(cp);assert d.X.shape==(6,20,169) and d.readiness['class_counts']=={'success':3,'future_failure':3}
    value=m.read(cp);value['attempts'].pop('a0');put(cp,value)
    with pytest.raises(ValueError,match='including refusals'):m.load_data(cp)


@pytest.mark.parametrize('field',['scenario_sha256','capture_sha256','seeds','object_id','contract_sha256'])
def test_manufactured_provenance_rebinding_cannot_admit_old_or_other_records(tmp_path,field):
    cp,sp,c=collection(tmp_path);s=m.read(sp);s['attempts'][0][field]={'generator':9,'physics':0,'sensor':0} if field=='seeds' else H(999) if field.endswith('sha256') else 'different_object';put(sp,s)
    value=m.read(cp);value['schedule']=m.binding(sp);put(cp,value)
    with pytest.raises(ValueError):m.load_data(cp)


def test_splits_use_only_schedule_and_keep_all_parent_groups_disjoint(tmp_path):
    cp,sp,c=collection(tmp_path);folds=m.freeze_splits(sp);fp=put(tmp_path/'splits.json',folds);d=m.load_data(cp);indices=m.split_indices(d,fp)
    assert len(indices)==3 and all(f['status']=='READY' for f in indices)
    for fold in indices:
        for field in m.IDENTITIES:
            groups=[{d.rows[i][field] for i in fold['indices'][role]} for role in m.ROLES]
            assert not any(groups[i]&groups[j] for i in range(3) for j in range(i))
    # Pre-run assignment does not inspect outcome files at all.
    for row in cp.parent.glob('a*/attempt.json'):row.unlink()
    assert m.freeze_splits(sp)==folds


def test_insufficient_categories_and_singleclass_roles_are_explicit(tmp_path):
    cp,sp,c=collection(tmp_path,categories=2)
    with pytest.raises(ValueError,match='three disjoint'):m.freeze_splits(sp)
    other=tmp_path/'three';other.mkdir();cp,sp,c=collection(other,{'a1':{'status':'runtime_error'}});fp=put(other/'split.json',m.freeze_splits(sp))
    folds=m.split_indices(m.load_data(cp),fp);assert all(f['status']=='SKIPPED_EMPTY_OR_SINGLE_CLASS_ROLE' for f in folds)


def test_fixture_pass_or_caller_pass_cannot_qualify_scientific_training(tmp_path,monkeypatch):
    cp,sp,c=collection(tmp_path);fake=put(tmp_path/'fixture.json',{'schema_version':'manosim_authored_capsule_execution_v3_report','status':'COMPLETED','qualification':'PASS','scientific_training_allowed':True})
    value=m.read(cp);value['qualifications']={c.contract_sha256:m.binding(fake)};put(cp,value);d=m.load_data(cp)
    assert not d.readiness['qualified_native_task_evidence'] and not d.readiness['scientific_training_allowed']
    with pytest.raises(ValueError,match='not admitted'):m.prepare_training(cp,put(tmp_path/'splits.json',m.freeze_splits(sp)),tmp_path/'training.json')


def test_exact_external_qualification_requires_independent_inventory_and_hashes(tmp_path,monkeypatch):
    cp,sp,c=collection(tmp_path);q=qualify(cp,sp,c,monkeypatch);assert m.load_data(cp).readiness['scientific_training_allowed']
    artifact=tmp_path/'native_task_evidence.json';artifact.write_text('tampered')
    assert not m.load_data(cp).readiness['scientific_training_allowed']


def test_training_preparation_binds_inputs_models_and_native_shape(tmp_path,monkeypatch):
    cp,sp,c=collection(tmp_path);qualify(cp,sp,c,monkeypatch);fp=put(tmp_path/'splits.json',m.freeze_splits(sp));protocol=tmp_path/'training.json'
    monkeypatch.setattr(m,'_runtime',lambda:dict(device='cpu',software_only=True));p=m.prepare_training(cp,fp,protocol)
    assert p['models']==list(m.MODELS) and p['maximum_candidate_fits_per_fold']==24 and p['input_contract']['channels']==169
    assert p['scientific_domain']==m.NATIVE_DOMAIN and p['freeze_before_fitting'] and str(cp) in p['input_bindings']
    with pytest.raises(FileExistsError):m.prepare_training(cp,fp,protocol)


def test_fit_refuses_role_or_shape_leakage_before_any_fitter(monkeypatch):
    monkeypatch.setattr(m,'fit_logistic',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must not fit')))
    X=np.zeros((6,20,169),np.float32);y=np.array([0,1]*3);indices=dict(train=np.array([0,1]),validation=np.array([2,3]),test=np.array([4,5]))
    with pytest.raises(ValueError):m.fit_fold(X[:,:,:72],y,indices,m.TrainingSpec(),0)
    indices['test']=indices['train']
    with pytest.raises(ValueError,match='disjoint'):m.fit_fold(X,y,indices,m.TrainingSpec(),0)


def test_software_only_cpu_models_have_finite_predictions_and_train_only_normalization():
    rng=np.random.default_rng(7);X=rng.normal(0,.1,(12,20,169)).astype(np.float32);y=np.array([0,1]*6);X[y==1,:,67]+=.4
    indices=dict(train=np.arange(4),validation=np.arange(4,8),test=np.arange(8,12));X[8:,:,0]+=100
    fits=m.fit_fold(X,y,indices,m.TrainingSpec(epochs=2),0)
    assert tuple(f['model'] for f in fits)==m.MODELS
    for f in fits:
        assert abs(f['state']['channel_mean'][0]-X[:4,:,0].astype(float).mean())<1e-12
        assert np.isfinite(f['probabilities']['test']).all() and np.all((f['probabilities']['test']>=0)&(f['probabilities']['test']<=1))
    # These are generated arrays: no native/task inference, scientific result or artifact.
    software_smoke=dict(scope='software_only',native_task_data_used=0,scientific_training_performed=False)
    assert not software_smoke['scientific_training_performed']


def test_declared_pre_execution_lift_boundary_prevents_outcome_dependent_window_selection():
 c=contract();a=recorder(c).arrays()
 with pytest.raises(ValueError,match='prospectively'):m.prefix(a,c,m.PrefixSpec(),lift_start_s=.88)


def test_future_failure_outside_recorded_horizon_is_unknown_not_training_label(tmp_path):
 cp,sp,c=collection(tmp_path,{'a1':{'outcome':dict(success=False,failure_type='drop',label_rule_id='native_task_label_v1',first_failure_time_s=99.)}})
 with pytest.raises(ValueError,match='outside recorded'):m.load_data(cp)


def test_train_command_never_fits_or_creates_output_before_qualification_and_freeze(tmp_path,monkeypatch):
 cp,sp,c=collection(tmp_path);q=qualify(cp,sp,c,monkeypatch);fp=put(tmp_path/'splits.json',m.freeze_splits(sp));protocol=tmp_path/'training.json'
 monkeypatch.setattr(m,'_runtime',lambda:dict(device='cpu',software_only=True));m.prepare_training(cp,fp,protocol)
 monkeypatch.setattr(m,'fit_fold',lambda *a:(_ for _ in ()).throw(AssertionError('must not fit')))
 # The prepared protocol is deliberately not committed: scientific fitting is blocked.
 with pytest.raises((ValueError,__import__('subprocess').CalledProcessError)):m.train(protocol,tmp_path/'models')
 assert not (tmp_path/'models').exists()


def test_final_binding_failure_quarantines_model_markers_and_keeps_hard_stop(tmp_path,monkeypatch):
 cp,sp,c=collection(tmp_path);d=m.load_data(cp);fp=put(tmp_path/'split.json',m.freeze_splits(sp));fold=m.split_indices(d,fp)[0];protocol=put(tmp_path/'protocol.json',{'manufactured_transaction':True})
 p=dict(input_contract=m.feature_contract(m.PrefixSpec()),collection=m.binding(cp));calls=[]
 def validate(*a):
  calls.append(True)
  if len(calls)>1:raise ValueError('manufactured final input mutation')
  return p,d,[fold],m.PrefixSpec(),m.TrainingSpec(epochs=1),'a'*40
 def fit(*a):
  probabilities={role:np.full(len(fold['indices'][role]),.5) for role in ('validation','test')}
  return [dict(model='training_prevalence',state={'prevalence':np.asarray(.5)},selection={},metrics={},probabilities=probabilities)]
 monkeypatch.setattr(m,'_validate_protocol',validate);monkeypatch.setattr(m,'fit_fold',fit);out=tmp_path/'models';r=m.train(protocol,out)
 assert r['status']=='HARD_STOP' and not list(out.rglob('model.json')) and len(list(out.rglob('model.failed_partial.json')))==3
 with np.load(next(out.rglob('predictions.npz')),allow_pickle=False) as a:
  assert {'test_attempt_ids','test_proposal_group_ids','test_y','test_p_success'}<=set(a.files)


@pytest.mark.parametrize('change',['head','protocol_file','protocol_value','data_bindings'])
def test_valid_but_different_final_declaration_is_quarantined(tmp_path,monkeypatch,change):
 cp,sp,c=collection(tmp_path);d=m.load_data(cp);fp=put(tmp_path/'split.json',m.freeze_splits(sp));fold=m.split_indices(d,fp)[0]
 protocol=put(tmp_path/'protocol.json',{'manufactured_transaction':True});p=dict(input_contract=m.feature_contract(m.PrefixSpec()),collection=m.binding(cp));calls=[]
 def validate(*a):
  calls.append(True);final=len(calls)>1;value=dict(p);head='a'*40;dataset=d
  if final:
   if change=='head':head='b'*40
   elif change=='protocol_file':put(protocol,{'manufactured_transaction':'newly_committed_but_different'})
   elif change=='protocol_value':value['new_valid_declaration']=True
   else:dataset=__import__('dataclasses').replace(d,bindings={**d.bindings,'new_input':H(999)})
  return value,dataset,[fold],m.PrefixSpec(),m.TrainingSpec(epochs=1),head
 def fit(*a):
  return [dict(model='training_prevalence',state={'prevalence':np.asarray(.5)},selection={},metrics={},
               probabilities={role:np.full(len(fold['indices'][role]),.5) for role in ('validation','test')})]
 monkeypatch.setattr(m,'_validate_protocol',validate);monkeypatch.setattr(m,'fit_fold',fit)
 out=tmp_path/'models';report=m.train(protocol,out)
 assert report['status']=='HARD_STOP' and report['failure']['message']=='native training transaction identity changed during fitting'
 assert not list(out.rglob('model.json')) and len(list(out.rglob('model.failed_partial.json')))==3
