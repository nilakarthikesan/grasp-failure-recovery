"""Pure synthetic buffer tests; no licensed data, native model or integration."""
import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
k=load('manufactured_camera_graph',HERE/'contact_kinematics.py')
core=load('manufactured_frozen_fullskin',ROOT/'research_artifacts/manosim_articulated_skin_2026-10-15_v1/skin_kinematics.py')
c=load('manufactured_coupling_case',HERE/'coupling_case.py')

@pytest.fixture(autouse=True)
def no_native(monkeypatch):
 import mujoco
 def no(*a,**kw):raise AssertionError('Synthetic controls forbid real native/model/FK/physics')
 for name in ('MjModel','MjData','mj_forward','mj_step','mj_integratePos','mj_fullM','mj_contactForce'):monkeypatch.setattr(mujoco,name,no)


def graph_fixture():
 rng=np.random.default_rng(2151);n=4
 a=dict(template=rng.normal(size=(n,3)).astype('float32')*.01,shapedirs=rng.normal(size=(n,3,10)).astype('float32')*.001,posedirs=rng.normal(size=(n,3,135)).astype('float32')*.0001,joint_regressor=rng.uniform(size=(16,n)).astype('float32'),weights=rng.uniform(size=(n,16)).astype('float32'))
 a['joint_regressor']/=a['joint_regressor'].sum(1,keepdims=True);a['weights']/=a['weights'].sum(1,keepdims=True);a['weights']*=1.0002 # deliberately nonunit literal rows: never normalize in candidate
 rest=np.tile([1.,0,0,0],(16,1));rest[1]=[np.cos(.13),np.sin(.13),0,0]
 q=np.r_[np.array([.12,-.15,.43]),np.tile([1.,0,0,0],16)];C=np.eye(4);C[:3,:3]=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);C[:3,3]=[.02,-.03,.07]
 counts={};count=lambda name:counts.__setitem__(name,counts.get(name,0)+1)
 graph=k.CameraGraph(core=core,qpos=q,rest_quaternions=rest,camera_world=C,beta32=np.zeros((1,10),np.float32),arrays32=a,source_hashes={'frozen':'manufactured'},count=count)
 return graph,counts,a,C,rest


def test_new_camera_graph_jvp_fd_vjp_and_registered_profile_buffers():
 g,counts,a,C,rest=graph_fixture();V=g.baseline().detach().numpy();J=g.jacobian().detach().numpy();FD=g.finite_difference().detach().numpy()
 assert np.all(np.abs(J-FD)<=1e-8+1e-6*np.abs(FD))
 np.testing.assert_allclose(J[:,:,:3],np.broadcast_to(np.eye(3),(4,3,3)),rtol=0,atol=1e-14)
 F=np.sin(np.arange(12)).reshape(4,3);tau=g.force_vjp(F).detach().numpy();explicit=np.einsum('vcd,vc->d',J,F,optimize=False)
 np.testing.assert_allclose(tau,explicit,rtol=0,atol=1e-13)
 assert counts==dict(baseline_skin_evaluations=1,jvp_skin_evaluations=51,fd_skin_evaluations=102,vjp_skin_evaluations=1)
 selected=dict(face_id=0,vertex_ids=[0,1,2],barycentric=[1/3]*3);p=k.material_point(g,V,J,selected)
 t,pv,error=p.load(np.array([.1,.2,.3]),np.linspace(-.1,.2,51),profile_hash=g.profile_hash,state_hash=g.state_hash)
 assert abs(error)<1e-13 and t.shape==(51,) and pv.shape==(3,)
 with pytest.raises(ValueError,match='Mixed'):p.load(np.ones(3),np.zeros(51),profile_hash='source32',state_hash=g.state_hash)
 with pytest.raises(ValueError,match='Mixed'):p.verify(g.profile_hash,'other q state')
 with pytest.raises(ValueError,match='Unregistered'):k.material_point(g,V.astype('float32').astype('float64'),J,selected)
 p.point_J.setflags(write=True);p.point_J[0,0]+=1
 with pytest.raises(ValueError,match='mutated'):p.verify(g.profile_hash,g.state_hash)


def test_camera_graph_does_not_commute_root_into_unnormalized_LBS():
 g,_,a,C,rest=graph_fixture();V=g.baseline().detach().numpy();qbody=core.quaternion_product(torch.from_numpy(rest),g.base[3:].reshape(16,4));worldR=core.quaternion_matrices(qbody)
 old=core.skin_state(worldR,g.base[:3],g.beta,g.arrays)['vertices'].detach().numpy()
 assert np.max(np.abs(V-old))>1e-9
 assert all(np.array_equal(g.arrays[name].numpy(),a[name].astype('float64')) for name in a)


@pytest.mark.parametrize('fault',['camera','shape','dtype','quaternion'])
def test_explicit_profile_inputs_refuse_hidden_conversion(fault):
 g,_,a,C,rest=graph_fixture();q=g.q.copy();beta=np.zeros((1,10),np.float32)
 if fault=='camera':C[0,0]+=.1
 elif fault=='shape':beta=beta.reshape(10)
 elif fault=='dtype':a['weights']=a['weights'].astype('float64')
 else:q[3]*=.9
 with pytest.raises(ValueError):k.CameraGraph(core=core,qpos=q,rest_quaternions=rest,camera_world=C,beta32=beta,arrays32=a,source_hashes={})


def test_fixed_points_are_label_minima_with_arbitrary_not_facing_normals():
 faces=np.tile(np.array([[0,1,2]],np.int64),(1552,1));labels=np.arange(1552)%6
 p=k.fixed_material_points(faces,labels)
 assert [x['face_id'] for x in p]==list(range(6)) and [x['part'] for x in p]==list(k.PARTS)
 assert all(x['barycentric']==[1/3]*3 and x['plane_penetration_m']==1e-4 for x in p)
 assert np.all(np.einsum('ij,ij->i',k.NORMALS,k.TANGENTS,optimize=False)==0)
 assert np.all(np.linalg.norm(k.NORMALS,axis=1)==1) # named planes, not source facet normals
 with pytest.raises(ValueError):k.fixed_material_points(faces,np.zeros(1552,dtype=int))


def test_native_mass_response_mixed_units_gate_and_bad_inertia():
 M=np.diag(np.r_[np.full(3,.2),np.full(48,.01)]);a0=np.linspace(-.1,.1,51);tau=np.sin(np.arange(51))*.03;aF=a0+np.linalg.solve(M,tau)
 result,r=c.mass_response(M,a0,aF,tau);assert result['passed'] and np.max(np.abs(r))<1e-14
 wrong=aF.copy();wrong[10]+=.01;assert not c.mass_response(M,a0,wrong,tau)[0]['passed']
 M[20,20]=0
 with pytest.raises(ValueError,match='positive'):c.mass_response(M,a0,aF,tau)
 M[20,20]=np.nan
 with pytest.raises(ValueError,match='Finite'):c.mass_response(M,a0,aF,tau)


def test_scene_collision_masks_exclude_default_template_and_preserve_native_parameters(tmp_path):
 from xml.etree import ElementTree as ET
 assets=''.join(f'<mesh name="m{i}" file="m{i}.obj"/>' for i in range(16))
 bodies=''.join(f'<body name="b{i}" pos=".1 .2 .3"><inertial mass=".02" diaginertia=".001 .002 .003"/><joint name="j{i}" type="ball"/><geom mesh="m{i}" contype="1" conaffinity="1"/></body>' for i in range(16))
 xml=f'<mujoco><default><geom friction=".7 .1 .01"/></default><asset>{assets}</asset><worldbody>{bodies}</worldbody><actuator><position name="p0" joint="j0" kp="1"/></actuator></mujoco>'
 p=tmp_path/'original.xml';p.write_text(xml)
 for i in range(16):(tmp_path/f'm{i}.obj').write_text('manufactured asset; not compiled')
 result,skeleton=c.scene_xml(p);before=ET.fromstring(xml);after=ET.fromstring(result)
 assert len(after.findall('.//geom'))==17 and len(after.findall('./worldbody//geom'))==16
 assert ET.tostring(before.find('./default/geom'))==ET.tostring(after.find('./default/geom'))
 assert all(g.get('contype')=='0' and g.get('conaffinity')=='0' for g in after.findall('./worldbody//geom'))
 for tag in ('body','joint','inertial','position'):
  assert [x.attrib for x in before.iter(tag)]==[x.attrib for x in after.iter(tag)]
 assert after.find('./option/flag').attrib==dict(contact='disable',actuation='disable') and len(skeleton)==64
 assert all(Path(m.get('file')).is_absolute() for m in after.findall('./asset/mesh'))
 p.write_text(xml.replace('<geom friction=".7 .1 .01"/>',''))
 with pytest.raises(ValueError,match='default'):c.scene_xml(p)


def test_named51_layout_checks_actual_names_addresses_bodies_and_not_just_width():
 parents=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
 names=['wrist_tx','wrist_ty','wrist_tz','wrist_rot']+[f'mano_joint_{j}' for j in range(1,16)]
 rows=[dict(name=n,qpos_address=i if i<3 else 3+4*(i-3),dof_address=i if i<3 else 3+3*(i-3),type=2 if i<3 else 1) for i,n in enumerate(names)]
 model=SimpleNamespace(nq=67,nv=51,nu=51,njnt=19,body_parentid=np.r_[0,[0 if i==0 else parents[i]+1 for i in range(16)]],jnt_qposadr=np.array([x['qpos_address'] for x in rows]),jnt_dofadr=np.array([x['dof_address'] for x in rows]),jnt_type=np.array([x['type'] for x in rows]),jnt_bodyid=np.array([1 if i<4 else i-2 for i in range(19)]),jnt_axis=np.vstack((np.eye(3),np.zeros((16,3)))))
 mj=SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY=1,mjOBJ_JOINT=2),mjtJoint=SimpleNamespace(mjJNT_SLIDE=2,mjJNT_BALL=1),mj_name2id=lambda m,t,n:int(n.rsplit('_',1)[1])+1 if t==1 else names.index(n))
 assert c.named_layout(model,mj,rows)==rows
 model.jnt_dofadr[5]+=1
 with pytest.raises(ValueError,match='layout'):c.named_layout(model,mj,rows)
 model.jnt_dofadr[5]-=1;model.jnt_bodyid[5]=1
 with pytest.raises(ValueError,match='body'):c.named_layout(model,mj,rows)
