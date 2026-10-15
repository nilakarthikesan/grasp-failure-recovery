"""Manufactured XML/compiled controls; no saved HUG data or native hand assets."""
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import unittest
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from xml_reference import read_urdf,forward_xml,source_limit_rows,obj_vertices,transform,quaternion_matrix_wxyz
from bridge_model import assign_named,assign_carrier,compare_body_frames,compare_collisions,joint_metadata_against_source
from grasp_failure_prediction.evaluation.pose_validation import load_shadow_model,load_shadow_spec

TOL={'current_position_m':1e-10,'current_rotation_matrix':1e-10,'primitive_size_m':1e-12,'mesh_vertex_set_m':1e-6}


def cube(path,half):
    pts=np.array([[x,y,z] for x in [-half,half] for y in [-half,half] for z in [-half,half]])
    faces=[[1,5,7],[1,7,3],[2,4,8],[2,8,6],[1,2,6],[1,6,5],[3,7,8],[3,8,4],[1,3,4],[1,4,2],[5,6,8],[5,8,7]]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(['v '+' '.join(map(str,v)) for v in pts]+['f '+' '.join(map(str,f)) for f in faces])+'\n')


def fixture(root):
    hand=root/'shadow_hand';hand.mkdir();cube(hand/'one/shared.obj',.01);cube(hand/'two/shared.obj',.02)
    xml='''<robot name="bridge_control">
<link name="world"/>
<link name="forearm"><inertial><mass value="1"/><inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".001"/></inertial><collision><geometry><mesh filename="one/shared.obj"/></geometry></collision></link>
<joint name="world_joint" type="fixed"><parent link="world"/><child link="forearm"/><origin xyz=".1 -.2 .3" rpy=".2 -.1 .3"/></joint>
<link name="linkA"><inertial><mass value="1"/><inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".001"/></inertial><collision><origin xyz=".01 0 .02" rpy=".1 .2 -.3"/><geometry><box size=".02 .03 .04"/></geometry></collision></link>
<joint name="A" type="revolute"><parent link="forearm"/><child link="linkA"/><origin xyz=".04 0 0" rpy=".1 .2 0"/><axis xyz="0 0 1"/><limit lower="-.7" upper=".8" effort="1" velocity="1"/></joint>
<link name="linkB"><inertial><mass value="1"/><inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".001"/></inertial><collision><geometry><mesh filename="two/shared.obj"/></geometry></collision></link>
<joint name="B" type="revolute"><parent link="linkA"/><child link="linkB"/><origin xyz="0 .03 0" rpy="0 .2 .1"/><axis xyz="1 0 0"/><limit lower="-.6" upper=".9" effort="1" velocity="1"/></joint>
<link name="tip"/><joint name="fixed_tip" type="fixed"><parent link="linkB"/><child link="tip"/><origin xyz="0 0 .05" rpy="0 0 .2"/></joint>
</robot>'''
    path=hand/'shadow_hand_right.urdf';path.write_text(xml);return read_urdf(path)


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.root=Path(self.temp.name);self.source=fixture(self.root)
        self.names=['A','B'];self.q=np.array([.31,-.22])
    def tearDown(self):self.temp.cleanup()
    def realize(self,floating=False,carrier=None,names=None,q=None):
        m=load_shadow_model(self.root,floating_root=floating);d=mujoco.MjData(m)
        assign_named(m,d,self.names if names is None else names,self.q if q is None else q)
        if floating:
            p,quat=carrier;assign_carrier(m,d,p,quat)
        mujoco.mj_forward(m,d);self.assertEqual(d.time,0.);return m,d
    def test_fixed_xml_all_frames_match_current_compiled(self):
        m,d=self.realize();f=forward_xml(self.source,self.names,self.q)
        self.assertTrue(compare_body_frames(self.source,f,m,d,1e-10,1e-10)['passed'])
    def test_floating_carrier_replaces_fixed_origin(self):
        pos=np.array([.3,.4,-.2]);xyzw=Rotation.from_euler('xyz',[.7,-.3,.2]).as_quat();q=xyzw[[3,0,1,2]]
        m,d=self.realize(True,(pos,q));T=transform(pos,quaternion_matrix_wxyz(q));f=forward_xml(self.source,self.names,self.q,T)
        self.assertTrue(compare_body_frames(self.source,f,m,d,1e-10,1e-10)['passed'])
        wrong=forward_xml(self.source,self.names,self.q,T@self.source['joints'][0]['origin'])
        self.assertFalse(compare_body_frames(self.source,wrong,m,d,1e-10,1e-10)['passed'])
    def test_named_order_permutation_is_invariant(self):
        m,d=self.realize(names=self.names[::-1],q=self.q[::-1]);f=forward_xml(self.source,self.names,self.q)
        self.assertTrue(compare_body_frames(self.source,f,m,d,1e-10,1e-10)['passed'])
    def test_wrong_values_or_namespace_detected(self):
        m,d=self.realize(names=self.names,q=self.q[::-1]);f=forward_xml(self.source,self.names,self.q)
        self.assertFalse(compare_body_frames(self.source,f,m,d,1e-10,1e-10)['passed'])
        with self.assertRaises(ValueError):assign_named(m,d,['A','missing'],self.q)
        with self.assertRaises(ValueError):forward_xml(self.source,['A','A'],self.q)
    def test_mesh_basename_collision_and_shapes(self):
        m,d=self.realize();f=forward_xml(self.source,self.names,self.q)
        meshes={(c['path'],tuple(c['scale'])):obj_vertices(c['path'],c['scale']) for c in self.source['collisions'] if c['kind']=='mesh'}
        result=compare_collisions(self.source,f,m,d,meshes,TOL)
        self.assertTrue(result['passed'],result)
        wrong={k:next(iter(meshes.values())) for k in meshes}
        self.assertFalse(compare_collisions(self.source,f,m,d,wrong,TOL)['passed'])
    def test_contact_mask_and_margin_faults_detected(self):
        m,d=self.realize();f=forward_xml(self.source,self.names,self.q)
        meshes={(c['path'],tuple(c['scale'])):obj_vertices(c['path'],c['scale']) for c in self.source['collisions'] if c['kind']=='mesh'}
        m.geom_contype[0]=0
        self.assertFalse(compare_collisions(self.source,f,m,d,meshes,TOL)['passed'])
        m.geom_contype[0]=1;m.geom_margin[0]=1e-5
        self.assertFalse(compare_collisions(self.source,f,m,d,meshes,TOL)['passed'])
    def test_extra_compiled_geom_cannot_escape_inventory(self):
        spec=load_shadow_spec(self.root)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.007,0,0],pos=[.5,.5,.5])
        m=spec.compile();d=mujoco.MjData(m);assign_named(m,d,self.names,self.q);mujoco.mj_forward(m,d)
        f=forward_xml(self.source,self.names,self.q)
        meshes={(c['path'],tuple(c['scale'])):obj_vertices(c['path'],c['scale']) for c in self.source['collisions'] if c['kind']=='mesh'}
        self.assertFalse(compare_collisions(self.source,f,m,d,meshes,TOL)['passed'])
    def test_disabled_joint_limit_flag_cannot_pass_metadata(self):
        m,d=self.realize();rows=assign_named(m,d,self.names,self.q)
        self.assertTrue(all(r['source_compiled_limits_match'] for r in joint_metadata_against_source(self.source,rows,1e-12)))
        joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'A');m.jnt_limited[joint]=False
        rows=assign_named(m,d,self.names,self.q)
        self.assertFalse(all(r['source_compiled_limits_match'] for r in joint_metadata_against_source(self.source,rows,1e-12)))
    def test_literal_source_limit_one_ulp_is_not_accepted(self):
        self.assertTrue(all(r['literal_pass'] for r in source_limit_rows(self.source,self.names,[.8,-.6])))
        rows=source_limit_rows(self.source,self.names,[np.nextafter(.8,np.inf),-.6])
        self.assertFalse(rows[0]['literal_pass']);self.assertGreater(rows[0]['violation_rad'],0.)
    def test_quaternion_order_fault_detected(self):
        pos=np.array([.1,.2,.3]);xyzw=Rotation.from_euler('xyz',[.2,.4,-.3]).as_quat();correct=xyzw[[3,0,1,2]]
        m,d=self.realize(True,(pos,correct));wrong=forward_xml(self.source,self.names,self.q,transform(pos,quaternion_matrix_wxyz(xyzw)))
        self.assertFalse(compare_body_frames(self.source,wrong,m,d,1e-10,1e-10)['passed'])
    def test_collision_order_or_local_frame_fault_detected(self):
        m,d=self.realize();f=forward_xml(self.source,self.names,self.q)
        meshes={(c['path'],tuple(c['scale'])):obj_vertices(c['path'],c['scale']) for c in self.source['collisions'] if c['kind']=='mesh'}
        changed=copy.deepcopy(self.source);changed['collisions'][1]['origin'][:3,3]+=[.01,0,0]
        self.assertFalse(compare_collisions(changed,f,m,d,meshes,TOL)['passed'])

if __name__=='__main__':unittest.main(verbosity=2)
