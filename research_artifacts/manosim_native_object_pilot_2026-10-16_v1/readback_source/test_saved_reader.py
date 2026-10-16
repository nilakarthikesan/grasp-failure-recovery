"""Manufactured arrays/files only; no native/object dataset or runtime calls."""
import importlib.util,json
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('saved',Path(__file__).with_name('verify_saved_pilot.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_quaternion_raw_sign_and_endpoint_object_offset():
    q=np.r_[np.array([1.,2,3]),np.tile([1.,0,0,0],16),[4,5,6,1,0,0,0]]
    rest=np.tile([1.,0,0,0],(16,1));p,R,obj=m.endpoint(q,rest,np.array([.2,0,0]))
    assert np.array_equal(p,[1,2,3]) and np.array_equal(obj,[4.2,5,6]) and np.array_equal(R,np.eye(3))
    q[70]*=2
    with pytest.raises(ValueError):m.endpoint(q,rest,np.zeros(3))

def test_journal_preserves_creation_hash_through_quarantine(tmp_path):
    row=dict(path='/raw/a',sha256='a'*64,bytes=5,label='x');moved=dict(row,path='/raw/b',original_publication_path='/raw/a')
    p=tmp_path/'j';p.write_text(json.dumps(row)+'\n'+json.dumps(dict(event='artifact_relocation',original_path='/raw/a',path='/raw/b',sha256='a'*64,bytes=5))+'\n')
    m.check_journal(p,[moved]);moved['sha256']='b'*64
    with pytest.raises(ValueError):m.check_journal(p,[moved])

def trace():
    dt=.002;M=2750;obj=np.zeros((M+1,3));obj[1250:,2]=.15
    st=dict(qpos=np.zeros((M+1,74)),object_com_world_m=obj,wrist_position_world_m=obj.copy(),wrist_rotation_world=np.tile(np.eye(3),(M+1,1,1)))
    ho=np.zeros(M);ho[500:]=1
    ft=dict(force_time_s=np.arange(M)*dt,hand_object_normal_force_N=ho,hand_table_normal_force_N=np.zeros(M),object_table_normal_force_N=np.zeros(M))
    return st,ft,dt

def test_label_requires_actual_absence_of_table_support_in_hold():
    st,ft,dt=trace();assert m.independent_label(st,ft,dt)['success']
    ft['object_table_normal_force_N'][1250]=.002
    assert m.independent_label(st,ft,dt)==dict(success=False,failure_type='failed_acquisition',first_failure_time_s=2.5)

def test_label_earliest_clocked_event_beats_later_failure_priority():
    st,ft,dt=trace();ft['hand_object_normal_force_N'][499]=.002;ft['hand_table_normal_force_N'][1600]=2
    assert m.independent_label(st,ft,dt)['first_failure_time_s']==.998

def test_saved_contact_J57_uses_opposite_object_and_hand_reactions():
    p=np.array([.1,.2,.3]);rot=np.tile(np.eye(3),(16,1,1));jac=np.zeros((3,57));jac[:,:3]=-np.eye(3);jac[:,3:6]=-np.cross(np.eye(3),p).T;jac[:,51:54]=np.eye(3);jac[:,54:57]=np.cross(np.eye(3),p).T
    a=dict(start_hand_anchors=np.zeros((16,3)),start_hand_rotations=rot,contact_position=p,start_object_body_rotation=np.eye(3),start_object_body_origin=np.zeros(3),body_ids=np.array([1,17]),contact_frame=np.eye(3),contact_dim=np.array(3),efc_address=np.array(0),efc_J=jac,contact_force6=np.array([2.,.1,.2,0,0,0]),efc_force=np.array([2.,.1,.2]),start_qvel=np.linspace(-1,1,57))
    c=dict(scene=dict(compiled_hand=dict(body_ids=list(range(1,17)),joint_ids=list(range(19))),object_body_id=17))
    assert m.contact_witness(a,{},c)['jacobian_max_error']==0
    a['efc_J'][0,51]*=-1
    with pytest.raises(ValueError):m.contact_witness(a,{},c)

def alignment():
    N=32;tri=np.tile([[-1.,-1,1],[1,-1,1],[0,1,1]],(N,1,1));p=np.tile([0.,0,1],(N,1));K=np.diag([100.,100.,1.])
    return dict(passed_support=np.array(True),pixel_uv=np.zeros((N,2)),nearest_triangles=tri,K=K,depth_m=np.ones(N),camera_rays=p,points_camera=p,points_world=p,nearest_points=p,distances_m=np.zeros(N),cameraWorld64=np.eye(4),allowances_m=np.full(N,.001+.5*np.sqrt(2)/100))

def test_alignment_checks_recorded_point_membership_and_allowance_without_search():
    a=alignment();assert m.alignment_readback(a)
    a['nearest_points']=a['nearest_points'].copy();a['nearest_points'][0]=[4,4,1];a['distances_m'][0]=np.sqrt(32)
    with pytest.raises(ValueError):m.alignment_readback(a)

def test_raw_camera_metadata_equality_distinguishes_float32_cast():
    capture=np.array([[173.46493931,0,111.92222222],[0,173.46493931,74.58888889],[0,0,1.]])
    source=capture.astype(np.float32);assert not np.array_equal(capture,source) and np.array_equal(source,capture.astype(np.float32))

def test_no_native_or_geometry_runtime_can_be_imported():
    for module in ('mujoco','torch','trimesh','grasp_failure_prediction'):
        with pytest.raises(RuntimeError):__import__(module)
