"""Manufactured static validation controls; integration APIs are prohibited."""
from copy import deepcopy
import gzip
import hashlib
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation import mapping_validation as mv
from grasp_failure_prediction.evaluation.retargeting import RetargetedHandPose


@pytest.fixture(autouse=True)
def no_physics(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('no integration in static validation controls')
    for name in ('mj_step','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,name,forbidden)


@pytest.fixture
def case(tmp_path,monkeypatch):
    hand=tmp_path/'shadow_hand';hand.mkdir();urdf=hand/'shadow_hand_right.urdf'
    links=''.join(f'<link name="{name}"/>' for name in mv.pv.TASK_LINK_NAMES)
    joints=''.join(f'<joint name="fixed_{i}" type="fixed"><parent link="palm"/><child link="{name}"/><origin xyz="{i*.001} 0 .03"/></joint>' for i,name in enumerate(mv.pv.TASK_LINK_NAMES))
    urdf.write_text('<robot name="control"><link name="base"/><link name="palm"/>'+links+'<joint name="joint" type="revolute"><parent link="base"/><child link="palm"/><axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint>'+joints+'</robot>')
    frames,_=mv.source_fk(urdf,['joint'],[0.]);positions=np.array([frames[n][:3,3] for n in mv.pv.TASK_LINK_NAMES])
    local=np.zeros((21,3))
    local[mv.HUMAN_TASK_INDICES[1]]=positions@mv.MANO_TO_OPERATOR_RIGHT
    wrist=np.eye(4);wrist[:3,3]=[0,0,.8]
    points=local+wrist[:3,3]
    k=np.array([[112.,0,111.5],[0,112.,111.5],[0,0,1.]])
    uv,_=mv.project_camera(points,k,denominator_epsilon=1e-8,include_nonpositive=True)
    grasp=SimpleNamespace(landmarks_3d=points,T_camera_wrist=wrist,t=wrist[:3,3].copy(),R_6d=np.eye(3)[:,:2].reshape(6),landmarks_2d=uv,mesh_vertices=np.tile([0,0,.8],(778,1)),mesh_faces=np.array([[0,1,2]],int))
    pose=RetargetedHandPose(('joint',),np.array([0.]),local@mv.MANO_TO_OPERATOR_RIGHT.T,positions.copy(),wrist)
    class Validator:
        def __init__(self,*args):self.model=SimpleNamespace();self.data=SimpleNamespace();self.pose=None
        def validate(self,pose):
            self.pose=pose
            self.model.pose=pose
            return SimpleNamespace(mean_fingertip_error_m=.006,maximum_fingertip_error_m=.012,fingertip_errors_m=np.array([.003,.004,.005,.006,.012]),contact_pairs=(('a','b'),))
    monkeypatch.setattr(mv.pv,'ShadowPoseValidator',Validator)
    def positions_for(model,data,path):
        frames,_=mv.source_fk(path,model.pose.joint_names,model.pose.qpos)
        return np.array([frames[n][:3,3] for n in mv.pv.TASK_LINK_NAMES])
    monkeypatch.setattr(mv.pv,'_link_positions',positions_for)
    cm=mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="front" pos="0 0 0" quat="0 1 0 0" fovy="90"/></worldbody></mujoco>')
    cd=mujoco.MjData(cm);mujoco.mj_forward(cm,cd)
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,mv.MANO_TO_OPERATOR_RIGHT.T.reshape(9))
    reference=lambda value:dict(mesh_vertices=grasp.mesh_vertices.copy(),landmarks_3d=grasp.landmarks_3d.copy(),mesh_faces=grasp.mesh_faces.copy(),bindings={'manufactured':'control'})
    kwargs=dict(urdf_root=tmp_path,T_world_camera=np.eye(4),camera_intrinsics=k,image_size=(224,224),world_palm=(np.array([0,0,.8]),quat),camera_model=cm,camera_data=cd,native_reconstructor=reference,mapping_kind='saved_command')
    return grasp,pose,kwargs


def test_complete_recomputed_static_gate_and_measured_contacts(case):
    g,p,kw=case;report=mv.validate_hug_mapping(g,p,**kw)
    assert report['static_input_gate']=='PASS'
    assert report['checks']['static_contacts']['status']=='MEASURED'
    assert report['checks']['fresh_mapping_provenance']['status']=='NOT_APPLICABLE'
    assert report['counts']['physics_step_attempts']==0
    assert report['new_outcome'] is None and not report['native_surface_fidelity_accepted']


@pytest.mark.parametrize('missing,gate',[('native_reconstructor','native_reconstruction'),('world_palm','mapped_world_palm'),('camera_model','compiled_camera_calibration')])
def test_missing_evidence_stays_incomplete(case,missing,gate):
    g,p,kw=case;kw[missing]=None
    report=mv.validate_hug_mapping(g,p,**kw)
    assert report['static_input_gate']=='INCOMPLETE'
    assert report['checks'][gate]['status']=='NOT_RUN'


def test_bad_native_reconstruction_refused(case):
    g,p,kw=case;reference=kw['native_reconstructor'](g);reference['mesh_vertices'][0,0]+=.001
    kw['native_reconstructor']=lambda value:reference
    report=mv.validate_hug_mapping(g,p,**kw)
    assert report['static_input_gate']=='FAIL'
    assert report['checks']['native_reconstruction']['status']=='FAIL'


def test_changed_camera_and_projection_are_not_accepted(case):
    g,p,kw=case;g.landmarks_2d[0,0]+=.001
    report=mv.validate_hug_mapping(g,p,**kw)
    assert report['checks']['camera_export_consistency']['status']=='FAIL'
    kw['T_world_camera']=np.eye(4);kw['T_world_camera'][0,3]=.01
    report=mv.validate_hug_mapping(g,p,**kw)
    assert report['checks']['compiled_camera_calibration']['status']=='FAIL'


def test_literal_limit_has_zero_tolerance(case):
    g,p,kw=case;p.qpos[0]=1.+1e-12
    report=mv.validate_hug_mapping(g,p,**kw)
    assert report['checks']['literal_joint_limits']['status']=='FAIL'


def test_bad_frame_and_missing_names_fail_without_physical_label(case):
    g,p,kw=case;p.reference_vectors_m[0,0]+=.001
    assert mv.validate_hug_mapping(g,p,**kw)['checks']['native_frame_input']['status']=='FAIL'
    bad=RetargetedHandPose(('missing',),p.qpos,p.landmarks_wrist_m,p.reference_vectors_m,p.T_camera_wrist)
    report=mv.validate_hug_mapping(g,bad,**kw)
    assert report['static_input_gate']=='FAIL' and report['new_outcome'] is None


def test_integration_attempt_in_callback_is_blocked_and_reported(case):
    g,p,kw=case
    kw['native_reconstructor']=lambda value:mujoco.mj_step(None,None)
    report=mv.validate_hug_mapping(g,p,**kw)
    assert report['static_input_gate']=='FAIL'
    assert report['counts']['physics_step_attempts']==1


def record(tmp_path):
    config=tmp_path/'config.yml';config.write_text('manufactured')
    pose=SimpleNamespace(joint_names=('joint',),qpos=np.array([.5]))
    value=dict(schema_version='hug_mapping_record_v1',status='returned',mapping_status='ok',outcome=None,
               fresh_instance_per_proposal=True,translation_strategy_id='declared',joint_names=['joint'],qpos_rad=[.5],
               joint_limits_rad=[[-1,1]],raw_refinement_qpos_rad=[[.5]]*20,optimizer_status_codes=[3]*20,
               provenance=dict(profile_id=mv.LEGACY_RETARGETING_PROFILE,config_path=str(config),config_sha256=mv.sha(config),
                               vector_scaling=1.,low_pass_filter_enabled=False,temporal_norm_delta=.004,reset_before_proposal=False,refinement_calls=20))
    return pose,value


def test_complete_raw_status_record_is_verified_not_trusted(tmp_path):
    pose,value=record(tmp_path)
    assert mv.mapping_record_gate(value,pose)['status']=='PASS'
    value['optimizer_status_gate_passed']=True;value['optimizer_status_codes'][-1]=0
    assert mv.mapping_record_gate(value,pose)['status']=='FAIL'


@pytest.mark.parametrize('change',[
    {'fresh_instance_per_proposal':False},{'raw_refinement_qpos_rad':[[.4]]*20},
    {'optimizer_status_codes':[3]*19},{'outcome':{'success':False}},
])
def test_provenance_refusals_are_not_trainable(tmp_path,change):
    pose,value=record(tmp_path);value.update(change)
    assert mv.mapping_record_gate(value,pose)['status']=='FAIL'
    assert mv.mapping_record_gate(None,pose)['status']=='NOT_RUN'


def test_configuration_hash_is_checked(tmp_path):
    pose,value=record(tmp_path);Path=__import__('pathlib').Path
    Path(value['provenance']['config_path']).write_text('changed')
    assert mv.mapping_record_gate(value,pose)['status']=='FAIL'


def test_manifest_verification_handles_compressed_payload_and_detects_changes(tmp_path):
    payload=json.dumps({'value':1}).encode();(tmp_path/'case.json.gz').write_bytes(gzip.compress(payload))
    (tmp_path/'manifest.json').write_text(json.dumps({'files':{'case.json':hashlib.sha256(payload).hexdigest()}}))
    store=mv.BoundEvidence(tmp_path)
    assert store.read('case.json')=={'value':1}
    assert store.references[0]['payload_sha256']==hashlib.sha256(payload).hexdigest()
    (tmp_path/'case.json.gz').write_bytes(gzip.compress(b'{}'))
    with pytest.raises(ValueError,match='hash mismatch'):store.read('case.json')


def test_absent_historical_evidence_does_not_fabricate_pass(tmp_path):
    report=mv.historical_evidence(tmp_path,0,'a'*64)
    assert all(row['status']=='NOT_RUN' for row in report['checks'].values())
    assert not report['qualifies_current_output'] and report['new_physics_steps']==0


def test_global_integration_guard_is_restored_after_error():
    original=mujoco.mj_step
    with pytest.raises(RuntimeError,match='forbids physics'):
        with mv.static_only():mujoco.mj_step(None,None)
    assert mujoco.mj_step is original


def test_nested_guards_count_attempt_once_at_each_scope():
    with mv.static_only() as outer:
        with mv.static_only() as inner:
            with pytest.raises(RuntimeError):mujoco.mj_step(None,None)
        assert inner['physics_step_attempts']==outer['physics_step_attempts']==1
