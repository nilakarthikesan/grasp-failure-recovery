"""Manufactured arrays only, no engine or producer imports."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('independent_profile',Path(__file__).with_name('verify_saved.py'));v=importlib.util.module_from_spec(s);s.loader.exec_module(v)

def test_gravity_total_force_and_nonzero_COM_moment():
 mass=np.full(16,.025);COM=np.zeros((1,16,3));COM[0,1,0]=.1;anchors=np.zeros((1,16,3));R=np.tile(np.eye(3),(1,16,1,1));tau=v.gravity(mass,COM,anchors,R)
 np.testing.assert_allclose(tau[0,:3],[0,0,3.924],rtol=0,atol=1e-14);assert tau[0,4]==pytest.approx(-.024525);assert tau[0,7]==pytest.approx(-.024525)

def test_nonzero_joint_anchor_compensates_body_geometry():
 a=dict(anchors_start=np.tile([1.,2.,3.],(2,16,1)),body_rotations_start=np.tile(np.eye(3),(2,16,1,1)));m=dict(geom_bodyid=np.arange(1,17),jnt_pos=np.tile([.1,.2,.3],(19,1)),geom_pos=np.tile([.4,.5,.6],(16,1)),geom_quat=np.tile([1.,0,0,0],(16,1)));c=dict(body_ids=list(range(1,17)),joint_ids=list(range(19)));C,R,ids=v.geom_frames(a,m,c);np.testing.assert_allclose(C,np.tile([1.3,2.3,3.3],(2,16,1)),rtol=0,atol=1e-15)

def test_analytic_capsule_table_and_sphere_bound():
 C=np.zeros((1,16,3));C[:,:,2]=1;R=np.tile(np.eye(3),(1,16,1,1));size=np.tile([.01,.02,0],(16,1));palm=np.array([[-.01,0,0],[.01,0,0],[0,.01,0],[0,0,.01]]);d=v.clearance(C,R,size,palm,np.array([0.,0.,2.]),.1,0)
 assert d.shape==(1,16,2);assert d[0,1,0]==pytest.approx(.97);assert d[0,1,1]==pytest.approx(.87)

def test_declaration_has12fixtures9750steps():
 p=v.plan();assert len(p)==12;assert sum(f['steps'] for f in p)==9750

def test_reference_approach_timing_and_multifinger_indent():
 goal=np.r_[np.zeros(3),np.tile([1.,0,0,0],16)];start=goal.copy();start[0]=-.06;indent=np.array([-.0005,0,-.0005])
 f=next(f for f in v.plan() if f['kind']=='guarded_sphere_table_approach');q=v.refs(f,start,goal,indent);np.testing.assert_allclose(q[-1],goal,atol=1e-15);assert q[f['steps']//2,0]==pytest.approx(-.03)
 f=next(f for f in v.plan() if f['kind']=='two_finger_planes');q=v.refs(f,goal,goal,indent);np.testing.assert_allclose(q[-1,:3],indent,atol=1e-15)

def test_body_J_ancestor_support_and_total_translation():
 p=np.array([[1.,0,0]]);a=np.zeros((1,16,3));R=np.tile(np.eye(3),(1,16,1,1));J=v.body_J(p,a,R,6);np.testing.assert_array_equal(J[0,:,:3],np.eye(3));assert np.any(J[:,:,18:21]);assert not np.any(J[:,:,6:15]);assert np.any(J[:,:,3:6])
def test_receipt_only_scalar_conversion_keeps_failures():
 assert v.numpy_scalar(np.bool_(False)) is False;assert v.numpy_scalar(np.float64(.002))==.002
 with pytest.raises(TypeError):v.numpy_scalar(np.array([1.]))
