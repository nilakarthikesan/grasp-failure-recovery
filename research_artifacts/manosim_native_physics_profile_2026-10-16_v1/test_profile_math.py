"""Manufactured arrays/XML only; no engine, MANO, FK or contact queries."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('new_profile_math',Path(__file__).with_name('profile_math.py'));p=importlib.util.module_from_spec(s);s.loader.exec_module(p)


def test_explicit_xml_profile_changes_only_compensation_routing():
 root=ET.Element('mujoco');w=ET.SubElement(root,'worldbody');act=ET.SubElement(root,'actuator')
 for i,name in enumerate(p.BODY_NAMES):
  b=ET.SubElement(w,'body',name=name,pos='0 0 0');ET.SubElement(b,'inertial',mass='.025',diaginertia='.1 .1 .1');ET.SubElement(b,'joint',type='ball')
  if i==0:
   for k in range(3):ET.SubElement(b,'joint',type='slide')
 ET.SubElement(act,'position',kp='30',dampratio='1');text=ET.tostring(root,encoding='unicode');out=ET.fromstring(p.gravity_profile_xml(text))
 assert all(b.get('gravcomp')=='1' for b in out.findall('./worldbody//body'))
 assert all(j.get('actuatorgravcomp')=='false' for j in out.findall('.//joint'))
 assert ET.tostring(root.find('actuator'))==ET.tostring(out.find('actuator'))
 assert 'gravcomp' not in root.find('./worldbody/body').attrib


def test_hand_only_compensation_model_refuses_routes_and_object_compensation():
 model=SimpleNamespace(body_gravcomp=np.r_[0.,np.ones(16),0.],jnt_actgravcomp=np.zeros(19),ngravcomp=16,opt=SimpleNamespace(gravity=np.array([0.,0.,-9.81])))
 c=SimpleNamespace(body_ids=tuple(range(1,17)),joint_ids=tuple(range(19)))
 assert p.compensated_model_receipt(model,c)['coriolis_compensated'] is False
 model.body_gravcomp[-1]=1
 with pytest.raises(ValueError):p.compensated_model_receipt(model,c)
 model.body_gravcomp[-1]=0;model.jnt_actgravcomp[5]=1
 with pytest.raises(ValueError):p.compensated_model_receipt(model,c)


def test_gravity_force_all_bodies_and_right_ball_virtual_work():
 mass=np.arange(1,17,dtype=float)/1000;anchors=np.zeros((16,3));R=np.tile(np.eye(3),(16,1,1));centers=np.tile([.1,.2,.3],(16,1));g=np.array([0.,0.,-9.81]);v=np.sin(np.arange(51))
 tau=p.gravity_compensation_force(mass,centers,anchors,R,g)
 np.testing.assert_allclose(tau[:3],-mass.sum()*g,rtol=0,atol=1e-15)
 force=-mass.sum()*g;np.testing.assert_allclose(tau[3:6],np.cross(centers[0],force),rtol=0,atol=1e-15)
 power=sum(float((-mass[i]*g)@(p.point_jacobian(centers[i],i,anchors,R)@v)) for i in range(16))
 assert float(tau@v)==pytest.approx(power,abs=1e-15)
 receipt=p.compensation_checks(tau,tau,tau,tau,np.zeros(51));assert all(receipt['checks'].values())
 moving=p.compensation_checks(tau,tau,tau+1,tau,np.ones(51));assert all(moving['checks'].values()) and moving['moving_residual_required_zero'] is False


def test_two_plane_move_is_fixed_minimum_norm_and_refuses_noisy_replacements():
 N=np.array([[0.,0.,1.],[1.,0.,0.]]);delta,r=p.two_plane_indentation(N)
 np.testing.assert_allclose(delta,[-.0005,0,-.0005],rtol=0,atol=1e-18)
 assert r['outcome_selection_used'] is False
 with pytest.raises(ValueError,match='ill-conditioned'):p.two_plane_indentation(np.array([[0.,0.,1.],[0.,0.,1.]]))
 with pytest.raises(ValueError,match='no rescaling'):p.two_plane_indentation(N,max_displacement_m=.0001)


def test_guard_covers_between_knots_and_refuses_unqualified_mesh_or_missing_domain():
 x=np.array([[0.,0.,0.],[.004,0,0]])
 d=np.full((2,16,1),.003);r=p.rigid_translation_guard(d,x,required_clearance_m=.002,oracle_error_m=0,oracle_id='analytic_plane_sphere_hand_bounds_v1')
 assert not r['passed'] and r['minimum_segment_clearance_lower_m']==pytest.approx(.001)
 d[:]=.005;r=p.rigid_translation_guard(d,x,required_clearance_m=.002,oracle_error_m=1e-9,oracle_id='analytic_plane_sphere_hand_bounds_v1');assert r['passed'] and not r['rotation_and_finger_motion_allowed']
 with pytest.raises(ValueError,match='analytic'):p.rigid_translation_guard(d,x,required_clearance_m=0,oracle_error_m=0,oracle_id='unqualified_mesh_GJK')
 with pytest.raises(ValueError,match='Complete'):p.rigid_translation_guard(d[:,:0],x,required_clearance_m=0,oracle_error_m=0,oracle_id='analytic_plane_sphere_hand_bounds_v1')


def test_all16_capsule_and_convex_palm_bounds_cover_table_and_sphere():
    c=np.zeros((16,3));c[:,2]=.2;R=np.tile(np.eye(3),(16,1,1));size=np.tile([.01,.02,0],(16,1));verts=np.array([[-.02,-.02,-.02],[.02,.02,-.02],[.02,-.02,.02],[-.02,.02,.02]])
    d,r=p.analytic_hand_separations(c,R,size,verts,np.array([.3,0,.2]),.02,0.)
    assert d.shape==(16,2) and r['capsule_count']==15
    np.testing.assert_allclose(d[1:,0],.17);np.testing.assert_allclose(d[1:,1],.27)
    assert d[0,0]==pytest.approx(.18) and d[0,1]<=.28
    # A sphere centered on a capsule and a table cutting the hand must refuse clearance.
    blocked,_=p.analytic_hand_separations(c,R,size,verts,c[1],.02,.2)
    assert (blocked[1:]<0).all()
