"""Manufactured coordinate tests only: no saved native/model/FK/skin work."""
import importlib.util,math
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('source_coordinate_controls',Path(__file__).with_name('source_coordinates.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def fixture(rootnorm=3.7):
    aa=np.linspace(-.4,.5,48,dtype=np.float32);aa[:3]=np.asarray([rootnorm,.13,-.25],dtype=np.float32)
    t=np.asarray([.03125,-.0625,.375],dtype=np.float32)
    camera=np.eye(4);camera[:3,:3]=c.rotation_matrices(c.exp_quaternion(np.array([.45,-.2,.13])));camera[:3,3]=[.33,-.29,.4]
    rest=c.exp_quaternion(np.linspace(-.15,.18,48).reshape(16,3));rest[0]=[1.,0.,0.,0.]
    return aa,t,camera,rest


@pytest.mark.parametrize('rootnorm',[.1,2.7,3.7])
def test_postAA_source_initialized_state_recovers_exactfloat32_payload(rootnorm):
    aa,t,camera,rest=fixture(rootnorm);q,receipt=c.source_initialized_qpos(aa,t,camera,rest)
    r=c.recover_source_inputs(q,rest,camera,aa);guard=c.require_original_bytes(r,aa,t)
    assert guard['original_axisangle_bytes_exact'] and guard['original_translation_bytes_exact']
    assert r['axis_angles_float32'].tobytes()==aa.tobytes() and r['translation_camera_float32'].tobytes()==t.tobytes()
    assert max(r['float32_axisangle_rotation_roundtrip_errors'])<1e-12
    assert receipt['old_GS64_state_domain_reused'] is False and receipt['geometry_fitting_or_offset'] is False


@pytest.mark.parametrize('angle',[0.,.01,math.pi,3.7,5.9])
def test_equivalent_branch_uses_reference_only_for_coordinate_choice(angle):
    original=np.asarray([0.,angle,0.],dtype=np.float32)
    q=c.exp_quaternion(original.astype(float))
    first,k1=c.nearest_equivalent_axis_angle(q,original);second,k2=c.nearest_equivalent_axis_angle(-q,original)
    np.testing.assert_allclose(first,original.astype(float),rtol=0,atol=4e-15)
    np.testing.assert_allclose(second,original.astype(float),rtol=0,atol=4e-15)
    assert k1 in c.BRANCHES and k2 in c.BRANCHES
    np.testing.assert_allclose(c.rotation_matrices(c.exp_quaternion(first)),c.rotation_matrices(q),rtol=0,atol=2e-15)


def test_byte_guard_refuses_signedzero_and_oneULP_without_epsilon_snap():
    aa,t,cam,rest=fixture();q,_=c.source_initialized_qpos(aa,t,cam,rest);record=c.recover_source_inputs(q,rest,cam,aa)
    changed=aa.copy();changed[4]=np.nextafter(changed[4],np.float32(np.inf))
    with pytest.raises(ValueError,match='byte identity'):c.require_original_bytes(record,changed,t)
    zeros=np.zeros(48,dtype=np.float32);negative=zeros.copy();negative[4]=-0.
    rec=dict(axis_angles_float32=zeros,translation_camera_float32=t)
    with pytest.raises(ValueError,match='byte identity'):c.require_original_bytes(rec,negative,t)
    with pytest.raises(ValueError):c.require_original_bytes(rec,zeros.astype(float),t)


def test_small_crossing_of_2pi_keeps_globally_nearest_equivalent_branch():
    reference=np.array([6.25,0.,0.],dtype=np.float32)
    q=c.exp_quaternion(np.array([6.30,0.,0.],dtype=float))
    for quaternion in (q,-q):
        aa,_=c.nearest_equivalent_axis_angle(quaternion,reference)
        np.testing.assert_allclose(aa,np.array([6.30,0.,0.]),rtol=0,atol=4*np.finfo(float).eps)


def test_wrong_root_state_fails_before_any_source_or_model_operation():
    aa,t,camera,rest=fixture();q,_=c.source_initialized_qpos(aa,t,camera,rest)
    q[3:7]=c.multiply(q[3:7],c.exp_quaternion(np.asarray([0.,1e-5,0.])))
    record=c.recover_source_inputs(q,rest,camera,aa)
    with pytest.raises(ValueError,match='byte identity'):c.require_original_bytes(record,aa,t)


@pytest.mark.parametrize('defect',['camera_shear','camera_nan','nonunit','wrongAAshape','wrongAAdtype','outbranchdomain'])
def test_invalid_coordinate_inputs_refuse(defect):
    aa,t,camera,rest=fixture()
    if defect=='camera_shear':camera[0,0]+=.1
    elif defect=='camera_nan':camera[0,0]=np.nan
    elif defect=='nonunit':rest[2,0]*=2
    elif defect=='wrongAAshape':aa=aa.reshape(16,3)
    elif defect=='wrongAAdtype':aa=aa.astype(float)
    else:
        with pytest.raises(ValueError):c.nearest_equivalent_axis_angle(np.asarray([1.,0.,0.,0.]),np.array([7.,0.,0.],dtype=np.float32))
        return
    with pytest.raises(ValueError):c.source_initialized_qpos(aa,t,camera,rest)


def test_finite109grid_is_regenerated_from_newq0_and_never_mutates_it():
    aa,t,camera,rest=fixture();q0,_=c.source_initialized_qpos(aa,t,camera,rest);saved=q0.copy()
    grid=[dict(delta51=[0.]*51)]
    for i in range(108):
        delta=np.zeros(51);delta[i%51]=.001*(1+i//51);grid.append(dict(delta51=delta.tolist()))
    states=c.motion_states(q0,grid)
    np.testing.assert_array_equal(q0,saved);np.testing.assert_array_equal(states[0],q0)
    assert states.shape==(109,67) and np.max(np.abs(states[1:]-states[0]))>0
    with pytest.raises(ValueError):c.motion_states(q0,grid[:108])


def test_sourcefloat32_translation_must_precede_camera64_transform():
    camera=np.eye(4);camera[:3,3]=[.33,-.29,.4]
    vertices=np.array([[2**24,0.,0.]],dtype=np.float32);translation=np.array([1.,0.,0.],dtype=np.float32)
    translated32=vertices+translation
    world=c.world_from_camera32(translated32,camera)
    late=vertices.astype(float)+translation.astype(float)+camera[:3,3]
    assert world[0,0]!=late[0,0] and abs(world[0,0]-late[0,0])==1.
    np.testing.assert_array_equal(world,translated32.astype(float)+camera[:3,3])
    with pytest.raises(ValueError):c.world_from_camera32(translated32.astype(float),camera)
