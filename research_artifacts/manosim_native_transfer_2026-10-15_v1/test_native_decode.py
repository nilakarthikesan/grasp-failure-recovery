"""Manufactured numeric/observation controls; no licensed model or FK engine."""
from contextlib import nullcontext
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value);return value


decode=module('manufactured_native_decoder',HERE/'native_decode.py')
case=module('manufactured_native_case',HERE/'audit_case.py')


@pytest.fixture(autouse=True)
def no_native_work(monkeypatch):
    def forbidden(*a,**k):pytest.fail('No native asset/model/FK/physics in manufactured controls')
    for key in ('mj_forward','mj_step','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,key,forbidden)


def artificial():
    poses=np.zeros(48,np.float32);poses[:3]=[.2,-.3,.1];poses[3:6]=[.1,.2,-.1];poses[39:42]=[-.2,.1,.3]
    rots=decode.axis_angle_rotations(poses).astype(np.float32)
    rest=np.zeros((16,3))
    for j,p in enumerate(decode.PARENTS[1:],1):rest[j]=rest[p]+[.01,.001*(j%3),.002*j]
    base=np.eye(4);base[:3,:3]=rots[0]
    anchors,world=decode.independent_fk(rest,rots,base)
    transforms=np.tile(np.eye(4),(16,1,1));transforms[:,:3,:3]=world;transforms[:,:3,3]=anchors
    landmarks=np.zeros((21,3));landmarks[list(decode.SNAP_TO_NATIVE)]=anchors
    for tip,bone in zip((4,8,12,16,20),(15,3,6,12,9)):landmarks[tip]=anchors[bone]+[.01,0,0]
    vertices=np.arange(778*3).reshape(778,3)*1e-6
    t=np.array([.1,.2,-.1],np.float32);wrist=np.eye(4);wrist[:3,:3]=rots[0];wrist[:3,3]=t
    raw=dict(t=t[None],R_6d=rots[0,:,:2].reshape(1,6),pose_6d=np.zeros((1,15,6)),shape=np.zeros((1,10),np.float32),
        T_camera_wrist=wrist,mesh_vertices=vertices+t,landmarks_3d=landmarks+t,mesh_faces=np.array([[0,1,2]]))
    data=dict(vertices=vertices,landmarks=landmarks,raw_6d_global_rotation=rots[0],raw_6d_local_rotations=rots[1:],
        actual_full_rotations=rots,actual_full_poses=poses,posed_transforms_abs=transforms,neutral_anchors=rest,
        faces=raw['mesh_faces'].copy(),actual_hands_mean=np.zeros((1,45)))
    camera=np.eye(4);camera[:3,:3]=decode.axis_angle_rotations(np.r_[np.array([0.,0.,.4]),np.zeros(45)])[0]
    camera[:3,3]=[1.,-2.,.3]
    return raw,data,camera


def test_source_global_local_and_camera_translation_applied_exactly_once():
    raw,data,camera=artificial();arrays,receipt=decode.verify_decoded(raw,data,camera,np.zeros(10,np.float32))
    assert receipt['native_reconstruction_pass'] and receipt['independent_fk_pass']
    assert receipt['max_saved_21_to_16_anchor_error_m']<1e-12
    expected=decode.transform(np.asarray(raw['t']).reshape(1,3),camera[:3,:3],camera[:3,3])[0]
    np.testing.assert_array_equal(arrays['independent_anchors_world_m'][0],expected)
    np.testing.assert_allclose(arrays['independent_rotations_world'][0],camera[:3,:3]@data['actual_full_rotations'][0],atol=1e-15,rtol=0)
    assert not receipt['limits_are_contact_allowances'] and receipt['no_shadow_operator_basis']


@pytest.mark.parametrize('kind',['shape','shear_camera','conflicting_t','homogeneous_wrist'])
def test_malformed_native_frame_and_shape_provenance_refuse(kind):
    raw,data,camera=artificial();shape=np.zeros(10,np.float32)
    if kind=='shape':shape[0]=1.
    if kind=='shear_camera':camera[0,1]+=.00001
    if kind=='conflicting_t':raw['T_camera_wrist'][0,3]+=.00001
    if kind=='homogeneous_wrist':raw['T_camera_wrist'][3,0]=1.
    with pytest.raises(ValueError):decode.verify_decoded(raw,data,camera,shape)


def test_parent_recursion_uses_anchors_not_centroid_origins():
    raw,data,camera=artificial();arrays,_=decode.verify_decoded(raw,data,camera,np.zeros(10))
    state=SimpleNamespace(joint_anchor_world_m=arrays['independent_anchors_world_m'].copy(),
        world_body_rotations=arrays['independent_rotations_world'].copy(),body_origin_world_m=np.ones((16,3)))
    assert case.compare_pose(state,arrays)['numerical_pose_pass']
    state.joint_anchor_world_m=state.body_origin_world_m
    assert not case.compare_pose(state,arrays)['numerical_pose_pass']


def test_float64_same_source_column_reshape_and_exp_are_explicit_proper_rotations():
    raw,data,camera=artificial();arrays,receipt=decode.verify_decoded(raw,data,camera,np.zeros(10))
    staged=arrays['staging_full_rotations']
    np.testing.assert_allclose(np.einsum('nji,njk->nik',staged,staged),np.tile(np.eye(3),(16,1,1)),atol=1e-12,rtol=0)
    np.testing.assert_array_equal(staged[0],decode.source_six_d_float64(np.asarray(raw['R_6d'],np.float32)))
    assert receipt['explicit_float64_axisangle_representation_max_change']<=1e-6
    assert 'no SVD or fit' in receipt['staging_rotation_representation']
    with pytest.raises(ValueError,match='Degenerate'):decode.source_six_d_float64(np.zeros(6))


def test_observer_retains_actual_postmean_outputs_and_restores_source_methods():
    raw,data,_=artificial();seen=[]
    class Layer:
        use_pca=False;center_idx=0;rot_mode='axisang';side='right';th_hands_mean=np.zeros((1,45))
        def rotation_by_axisang(self,*a,**k):return dict(full_rots=data['actual_full_rotations'][None],full_poses=data['actual_full_poses'][None])
        def forward(self,*a,**k):
            value=self.rotation_by_axisang(*a,**k);seen.append(value)
            return SimpleNamespace(transforms_abs=data['posed_transforms_abs'][None])
        def __call__(self,*a,**k):return self.forward(*a,**k)
        def get_mano_closed_faces(self):return data['faces']
    layer=Layer();original_rot=layer.rotation_by_axisang;original_forward=layer.forward
    class Mano:
        mano_layer=layer
        def __call__(self,*a,**k):
            self.mano_layer(np.zeros((1,48)),None)
            return dict(vertices=data['vertices'][None],landmarks_3d=data['landmarks'][None],
                R_3x3=data['raw_6d_global_rotation'][None],pose_3x3=data['raw_6d_local_rotations'][None])
    fake=SimpleNamespace(no_grad=nullcontext,from_numpy=lambda x:x,zeros=lambda shape,**k:np.zeros(shape),float32=np.float32)
    calls=[];out=decode.observed_parametric_forward(Mano(),np.zeros(99,np.float32),np.zeros((1,10),np.float32),fake,lambda:calls.append(True))
    assert len(calls)==2 and len(seen)==2
    assert layer.rotation_by_axisang==original_rot and layer.forward==original_forward
    np.testing.assert_array_equal(out['actual_full_rotations'],data['actual_full_rotations'])
    layer.th_hands_mean=np.ones((1,45))
    with pytest.raises(ValueError,match='flat/no-PCA'):decode.observed_parametric_forward(Mano(),np.zeros(99),np.zeros((1,10)),fake,lambda:None)


def test_analytic_capsule_axis_side_hemisphere_and_inside_boundary_distances():
    points=np.array([[0,0,0],[0,0,2],[0,0,3],[1,0,0],[2,0,0],[1,0,2]],float)
    result=case.capsule_signed_distance(points,np.zeros(3),np.eye(3),1.,2.)
    np.testing.assert_array_equal(result,[-1.,-1.,0.,0.,1.,0.])
    sphere=case.capsule_signed_distance(points,np.zeros(3),np.eye(3),1.,0.)
    np.testing.assert_array_equal(sphere,np.linalg.norm(points,axis=1)-1.)


def test_capsule_rigid_frame_invariance_and_malformed_frame_refusal():
    p=np.array([[1.,0.,0.],[0.,0.,3.],[-1.,1.,-3.]])
    rotation=decode.axis_angle_rotations(np.r_[[.2,.3,-.4],np.zeros(45)])[0];position=np.array([.1,-.2,.4])
    world=decode.transform(p,rotation,position)
    np.testing.assert_allclose(case.capsule_signed_distance(world,position,rotation,1.,2.),
        case.capsule_signed_distance(p,np.zeros(3),np.eye(3),1.,2.),atol=1e-14,rtol=0)
    with pytest.raises(ValueError,match='Proper rigid'):case.capsule_signed_distance(p,np.zeros(3),2*np.eye(3),1.,2.)


def test_same_part_grouping_keeps_ring_little_and_thumb_distinct():
    groups={case.PARTS[p]:[] for p in range(6)}
    for bone,part in enumerate(case.BONE_PART):groups[case.PARTS[part]].append(bone)
    assert groups==dict(palm=[0],index=[1,2,3],middle=[4,5,6],ring=[10,11,12],little=[7,8,9],thumb=[13,14,15])


def test_parametric_and_field_caps_refuse_before_extra_execution():
    value=case.AuditCase.__new__(case.AuditCase);value.counts=dict(parametric_mano_forwards=2,full_field_variant_scans=2)
    with pytest.raises(RuntimeError,match='Two CPU'):value._parametric_budget()
    with pytest.raises(RuntimeError,match='before query'):value._full_field(None,None,'mesh',None,None)
    assert value.counts==dict(parametric_mano_forwards=2,full_field_variant_scans=2)


def test_literal_replay_refuses_one_float_changed_anchor_even_with_equal_scalar_error():
    first=SimpleNamespace(named_qpos={'wrist_tx':1.},joint_anchor_world_m=np.zeros((16,3)),world_body_rotations=np.tile(np.eye(3),(16,1,1)),
        local_body_rotations=np.tile(np.eye(3),(16,1,1)),body_origin_world_m=np.zeros((16,3)),wrist_anchor_world_transform=np.eye(4))
    second=copy.deepcopy(first);case.exact_replay(first,second);second.joint_anchor_world_m[0,0]=1e-17
    with pytest.raises(ValueError,match='readback changed'):case.exact_replay(first,second)


def test_complete_native_fields_and_write_time_receipts_have_no_surface_pass(tmp_path,monkeypatch):
    root=HERE.parents[1]
    fields=module('manufactured_inherited_native_fields',root/'research_artifacts/hug_full_field_mapping_repair_2026-10-15_v1/audit_full_field.py')
    value=case.AuditCase.__new__(case.AuditCase);value.output=tmp_path;value.v1=fields
    value.counts=dict(full_field_variant_scans=0,collider_point_queries=0);value.checkpoint_artifacts=[];value.auxiliary_artifacts=[]
    value.query=np.zeros((3,3));value.radius=np.zeros(3);value.native_distance=np.zeros((3,6));value.grid=dict(required_relationships=54)
    model=SimpleNamespace(ngeom=16,geom_contype=np.ones(16,int),geom_conaffinity=np.ones(16,int),
        geom_bodyid=np.arange(16),geom_margin=np.zeros(16),geom_gap=np.zeros(16),geom_type=np.full(16,int(mujoco.mjtGeom.mjGEOM_SPHERE)))
    monkeypatch.setattr(mujoco,'mj_id2name',lambda model,kind,body:f'mano_body_{body}')
    value.a=SimpleNamespace(geom_rep=lambda *a:dict(kind=int(mujoco.mjtGeom.mjGEOM_SPHERE)),
        geom_query=lambda g,p:(np.full(len(p),.010),np.zeros((len(p),3)),np.zeros((len(p),3))))
    result=value._full_field(model,None,'mesh',np.zeros(3),np.eye(3))
    assert result['summary']['required']==54 and result['summary']['definite_lost']==54
    assert result['surface_fidelity_status']=='FAIL' and result['full_orientation_status']=='NOT_RUN'
    row=value.checkpoint_artifacts[0];assert row['sha256']==case.sha(row['path'])
    with np.load(row['path'],allow_pickle=False) as z:
        assert z['required'].shape==(3,6,3) and z['required'].sum()==54
        assert z['mapped_distance_by_part_m'].shape==(3,6)
