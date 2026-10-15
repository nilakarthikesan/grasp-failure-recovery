"""Manufactured scene controls; no HUG or Shadow model construction."""
from pathlib import Path
import sys,tempfile,unittest,copy,xml.etree.ElementTree as ET
sys.path.insert(0,'/private/tmp/hug_current_execution_bridge_2026-10-15_v1')
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from object_reference import read_object,triangle_connectivity
from scene_checks import object_check,controller_and_table_check
from scene_runtime import stage_named_palm
from grasp_failure_prediction.evaluation.object_assets import load_object_geometry,append_object_geometry
from grasp_failure_prediction.evaluation.runner import AdroitShadowRunner
from xml_reference import transform,quaternion_matrix_wxyz
physics=[0]
def forbidden(*args,**kwargs):physics[0]+=1;raise AssertionError('Synthetic controls forbid steps')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbidden)
TOL={'current_position_m':1e-10,'current_rotation_matrix':1e-10,'mesh_vertex_set_m':1e-6}
OBJ='v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
XML='''<mujoco model="manufactured"><compiler angle="radian"/><asset><mesh name="mesh" file="tetra.obj" scale=".2 .1 .3"/></asset><worldbody><body name="source" pos="1 2 3"><freejoint/><inertial pos=".01 -.02 .03" mass="1.2" diaginertia=".01 .02 .025"/><geom type="mesh" mesh="mesh" pos=".01 -.04 .02" quat=".9238795325112867 0 0 .3826834323650898" contype="0" conaffinity="0" group="2"/><geom type="mesh" mesh="mesh" pos=".01 -.04 .02" quat=".9238795325112867 0 0 .3826834323650898" group="3" friction="1 .005 .0001"/></body></worldbody></mujoco>'''
def fixture(folder,extra=False):
 p=Path(folder);(p/'tetra.obj').write_text(OBJ);(p/'object.xml').write_text(XML);ref=read_object(p/'object.xml');g=load_object_geometry(p/'object.xml');root=ET.Element('mujoco');ET.SubElement(root,'compiler',angle='radian');w=ET.SubElement(root,'worldbody');body=append_object_geometry(root,w,g)
 if extra:ET.SubElement(body,'geom',name='unexpected',type='sphere',size='.001',contype='0',conaffinity='0')
 model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model);p=np.array([.4,-.2,.3]);q=Rotation.from_euler('xyz',[.3,-.2,.4]).as_quat()[[3,0,1,2]];data.qpos[:3]=p;data.qpos[3:7]=q;mujoco.mj_forward(model,data);return model,data,ref,p,q
def controller_fixture():
 from types import SimpleNamespace
 from grasp_failure_prediction.evaluation.runner import _add_shadow_actuation_spec
 names=['WRJ1','WRJ2']+['F'+str(i) for i in range(22)]
 joints=''.join('<body name="link'+str(i)+'" pos="0 0 '+str(.02*i)+'"><joint name="'+n+'" range="-1 1" actuatorfrcrange="-2 2"/><geom type="sphere" size=".01"/></body>' for i,n in enumerate(names))
 xml='<mujoco><compiler angle="radian"/><worldbody><geom name="table" type="box" size=".4 .4 .025" pos="0 0 -.025" friction="1 .005 .0001"/><body name="forearm"><freejoint name="world_joint"/><geom type="sphere" size=".01"/>'+joints+'</body><body name="object" pos="1 0 0"><freejoint name="object_free"/><geom name="object_geom" type="sphere" size=".01"/></body></worldbody></mujoco>'
 m=_add_shadow_actuation_spec(mujoco.MjSpec.from_string(xml)).compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d);ob=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object');f=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'forearm');hand={f}|set(int(m.jnt_bodyid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in names)
 r=SimpleNamespace(model=m,data=d,_hand_joint_names=names,_hand_body_ids=hand,object_geom_ids=set(np.flatnonzero(m.geom_bodyid==ob)));src={'joints':[{'name':n,'kind':'revolute','limits':np.array([-1,1])} for n in names]}
 return r,src,names,np.zeros(24),np.zeros(3),np.array([1.,0,0,0]),{n:2 for n in names}
class Tests(unittest.TestCase):
 def test_scaled_rotated_raw_object_to_imported_scene(self):
  with tempfile.TemporaryDirectory() as d:
   m,a,r,p,q=fixture(d);v=object_check(m,a,r,p,q,1.2,TOL);self.assertTrue(v['passed'],v)
 def test_default_body_pose_not_composed_after_reset(self):
  with tempfile.TemporaryDirectory() as d:
   m,a,r,p,q=fixture(d);np.testing.assert_allclose(a.xpos[1],p,atol=0,rtol=0);self.assertTrue(object_check(m,a,r,p,q,1.2,TOL)['passed'])
 def test_wrong_object_transform_fails(self):
  with tempfile.TemporaryDirectory() as d:
   m,a,r,p,q=fixture(d);self.assertFalse(object_check(m,a,r,p+[.001,0,0],q,1.2,TOL)['passed'])
 def test_visual_connectivity_change_fails(self):
  with tempfile.TemporaryDirectory() as d:
   m,a,r,p,q=fixture(d);m.mesh_face[0]=m.mesh_face[0][[0,2,1]];self.assertFalse(object_check(m,a,r,p,q,1.2,TOL)['passed'])
 def test_vertex_distortion_fails(self):
  with tempfile.TemporaryDirectory() as d:
   m,a,r,p,q=fixture(d);m.mesh_vert[0,0]+=.001;self.assertFalse(object_check(m,a,r,p,q,1.2,TOL)['passed'])
 def test_collision_mask_change_fails(self):
  with tempfile.TemporaryDirectory() as d:
   m,a,r,p,q=fixture(d);g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'object_geom');m.geom_contype[g]=0;self.assertFalse(object_check(m,a,r,p,q,1.2,TOL)['passed'])
 def test_extra_object_geom_fails_complete_inventory(self):
  with tempfile.TemporaryDirectory() as d:
   m,a,r,p,q=fixture(d,True);v=object_check(m,a,r,p,q,1.2,TOL);self.assertFalse(v['complete_object_geom_inventory']);self.assertFalse(v['passed'])
 def test_mass_and_inertia_scale_consistently(self):
  with tempfile.TemporaryDirectory() as d:
   m,a,r,p,q=fixture(d);m.body_mass[1]*=2;m.body_inertia[1]*=2;self.assertTrue(object_check(m,a,r,p,q,2.4,TOL)['passed']);m.body_inertia[1]*=1.01;self.assertFalse(object_check(m,a,r,p,q,2.4,TOL)['passed'])
 def test_actual_static_palm_method_with_nontrivial_wrist(self):
  m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="forearm"><freejoint name="world_joint"/><geom type="sphere" size=".01"/><body name="wrist" pos=".1 .02 .03"><joint name="w" axis="0 1 0"/><geom type="sphere" size=".01"/><body name="palm" pos=".02 .03 .04"><joint name="p" axis="1 0 0"/><geom type="sphere" size=".01"/></body></body></body></worldbody></mujoco>')
  r=AdroitShadowRunner.__new__(AdroitShadowRunner);r.model=m;r.data=mujoco.MjData(m);r._root_qpos_address=0;r._hand_joint_names=('w','p');r._hand_qpos_addresses=np.array([7,8]);r._palm_body_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'palm');T=transform([.3,-.4,.5],Rotation.from_euler('xyz',[.5,.3,-.7]).as_matrix());position,quaternion,target=stage_named_palm(r,('p','w'),[-.2,.4],T)
  np.testing.assert_array_equal(r.data.qpos[7:],[.4,-.2]);np.testing.assert_allclose(r.data.xpos[r._palm_body_id],T[:3,3],atol=1e-14,rtol=0);np.testing.assert_allclose(r.data.xmat[r._palm_body_id].reshape(3,3),T[:3,:3],atol=1e-14,rtol=0);self.assertEqual(r.data.time,0.)
 def test_unsupported_object_frame_fails_closed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'tetra.obj').write_text(OBJ);(p/'object.xml').write_text(XML.replace('pos=".01 -.04 .02" quat=".9238795325112867 0 0 .3826834323650898"','euler="0 0 .2"'))
   with self.assertRaises(ValueError):read_object(p/'object.xml')
 def test_native_world_palm_helper_matches_independent_composition(self):
  from types import SimpleNamespace
  from native_reference import expected_native_palm,quaternion_from_matrix
  from grasp_failure_prediction.evaluation.case_runner import _hug_world_palm_pose
  m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="palm"><geom type="sphere" size=".01"/></body></worldbody></mujoco>');a=mujoco.MjData(m)
  for k in range(8):
   R=Rotation.from_rotvec(np.array([.21,-.37,.53])*(k+1)).as_matrix();a.xmat[1]=R.ravel();validator=SimpleNamespace(model=m,data=a,set_pose=lambda pose:None)
   C=transform([.2,-.1,.5],Rotation.from_rotvec([.7,-.3,.2]).as_matrix());W=transform([.03,.04,.17],Rotation.from_rotvec([-.2,.8,.6]).as_matrix());W[0,0]+=1e-8
   with tempfile.TemporaryDirectory() as folder:
    np.save(Path(folder)/'T_world_camera.npy',C);pos,quat=_hug_world_palm_pose(SimpleNamespace(T_camera_wrist=W),None,validator,Path(folder));expected=expected_native_palm(C,W,transform([0,0,0],R))
    np.testing.assert_allclose(pos,expected['palm_world'][:3,3],atol=1e-15,rtol=0);np.testing.assert_allclose(quaternion_matrix_wxyz(quat),expected['palm_world'][:3,:3],atol=1e-14,rtol=0)
 def test_missing_world_camera_refuses_native_helper(self):
  from grasp_failure_prediction.evaluation.case_runner import _hug_world_palm_pose
  with tempfile.TemporaryDirectory() as folder:
   with self.assertRaises(FileNotFoundError):_hug_world_palm_pose(None,None,None,Path(folder))
 def test_current_actuation_configuration_positive(self):
  args=controller_fixture();self.assertTrue(controller_and_table_check(*args)['passed'])
 def test_unhandled_defaults_refuse_independent_object_reference(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder);(p/'tetra.obj').write_text(OBJ);(p/'object.xml').write_text(XML.replace('<asset>','<default><geom margin=".01"/></default><asset>'))
   with self.assertRaises(ValueError):read_object(p/'object.xml')
 def test_actuator_semantic_corruptions_refused(self):
  for field,value in [('actuator_gear',np.array([2.,0,0,0,0,0])),('actuator_gaintype',int(mujoco.mjtGain.mjGAIN_AFFINE)),('actuator_biastype',int(mujoco.mjtBias.mjBIAS_NONE)),('actuator_dyntype',int(mujoco.mjtDyn.mjDYN_FILTER))]:
   args=controller_fixture();getattr(args[0].model,field)[0]=value;self.assertFalse(controller_and_table_check(*args)['passed'],field)
 def test_weld_and_table_corruptions_refused(self):
  args=controller_fixture();args[0].data.eq_active[0]=0;self.assertFalse(controller_and_table_check(*args)['passed'])
  args=controller_fixture();args[0].model.eq_data[0,3]=.001;self.assertFalse(controller_and_table_check(*args)['passed'])
  for field,value in [('geom_margin',.01),('geom_gap',.01),('geom_friction',np.array([.4,.005,.0001]))]:
   args=controller_fixture();m=args[0].model;g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'table');getattr(m,field)[g]=value;self.assertFalse(controller_and_table_check(*args)['passed'],field)
 def tearDown(self):self.assertEqual(physics[0],0)
if __name__=='__main__':unittest.main(verbosity=2)
