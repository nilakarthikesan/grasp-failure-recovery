"""Manufactured fixtures only. No native asset/model/FK/step/query execution."""
from pathlib import Path
from types import SimpleNamespace as NS
from xml.etree import ElementTree as ET
import importlib.util,sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
c=load('manufactured_compensated_case',HERE/'physics_case.py');u=c.u

def q0():return np.r_[np.zeros(3),np.tile([1.,0,0,0],16)]

def manufactured_xml(tmp_path):
 root=ET.Element('mujoco');d=ET.SubElement(root,'default');ET.SubElement(d,'geom',solref='.04 1',solimp='.5 .95 .001');a=ET.SubElement(root,'asset');w=ET.SubElement(root,'worldbody');act=ET.SubElement(root,'actuator')
 for i in range(16):
  ET.SubElement(a,'mesh',name=f'mano_bone_{i}',file=f'm{i}.obj');b=ET.SubElement(w,'body',name=f'mano_body_{i}',pos='0 0 0');ET.SubElement(b,'inertial',mass='.025',diaginertia='.1 .2 .3');ET.SubElement(b,'joint',name=f'j{i}',type='ball');ET.SubElement(b,'geom',type='mesh' if i==0 else 'capsule',size='.01 .02')
  if i==0:
   for j in range(3):ET.SubElement(b,'joint',name=f's{j}',type='slide')
  ET.SubElement(act,'position',joint=f'j{i}',kp='30')
 path=tmp_path/'source.xml';path.write_text(ET.tostring(root,encoding='unicode'));return path,root


def test_distinct_xml_profile_preserves_all_source_inertias_gains_and_static_obstacles(tmp_path):
 path,original=manufactured_xml(tmp_path)
 for kind in c.KINDS:
  planes=[dict(bone=bone,position_m=[0.,j*.1,0.],quaternion_wxyz=[1.,0.,0.,0.]) for j,bone in enumerate((3,6))]
  root=ET.fromstring(c.scene_xml(path,kind,.002,planes));world=root.findall('./worldbody//geom');assert len(world)==(16 if kind=='gravity_motion_hold' else 18)
  assert [ET.tostring(x) for x in original.findall('.//inertial')+original.findall('./actuator/*')]==[ET.tostring(x) for x in root.findall('.//inertial')+root.findall('./actuator/*')]
  assert ET.tostring(original.find('./default/geom'))==ET.tostring(root.find('./default/geom'))
  assert root.find('option').attrib['gravity']=='0 0 -9.81' and root.find('option/flag').attrib['energy']=='enable'
  assert all(x.get('gravcomp')=='1' for x in root.findall('./worldbody//body'))
  if kind=='two_finger_planes':
   assert [root.find(f'./worldbody//body[@name="mano_body_{i}"]/geom').get('contype') for i in (3,6)]==['1','2']
   assert [x.get('pos') for x in root.findall('./worldbody/geom')]==['0.0 0.0 0.0','0.0 0.1 0.0']
  elif kind=='guarded_sphere_table_approach':
   assert root.find('./worldbody/geom[@name="approach_sphere"]').get('pos')=='0.3 0.0 0.1'
   assert all(g.get('contype')=='2' and g.get('conaffinity')=='1' for g in root.findall('./worldbody/body/geom'))
   assert all(g.get('contype')=='1' and g.get('conaffinity')=='2' for g in root.findall('./worldbody/geom'))


def test_fixed_source_goals_are_not_offsets_or_gravity_sag():
 goal=q0();goal[:3]=[.01,.02,.03];start=q0();start[:3]=[-.05,0,0];indent=np.array([-.0005,0,-.0005])
 for f in c.fixture_plan():
  if f['kind']!='two_finger_planes':np.testing.assert_allclose(c.reference_qpos(f,start,goal,indent,f['duration_s']),goal,atol=1e-16)
  else:np.testing.assert_allclose(c.reference_qpos(f,start,goal,indent,.75)[:3],goal[:3]+indent,atol=1e-16)


def two_contacts(reverse=False,active=True):
 anchors=np.zeros((16,3));rotations=np.tile(np.eye(3),(16,1,1));centers=np.array([[0,0,.03],[0,0,0],[.1,0,.03],[.1,0,0.]])
 basis=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
 pairs=[(0,1,3),(2,3,6)];contacts=[];J=[];force=[]
 for j,(hg,pg,bone) in enumerate(pairs):
  B=basis.copy();sign=1
  if reverse:B=np.diag([-1,1,-1])@B;sign=-1
  point=np.array([.1*j,0,0]);row=sign*B@u.body_point_J(point,bone,anchors,rotations);f=np.array([.2+j*.1,.01,-.02])
  contacts.append(NS(geom=np.array([hg,pg] if reverse else [pg,hg]),dim=3,frame=B.ravel(),pos=point,dist=0.,efc_address=3*j if active else -1,solref=np.array([.04,1]),solimp=np.array([.5,.95,.001,.5,2]),friction=np.array([1.,1.,.005,.005,.0001])))
  J.append(row);force.append(f)
 J=np.concatenate(J) if active else np.zeros((0,51));force=np.concatenate(force) if active else np.zeros(0);nefc=6 if active else 0
 data=NS(ncon=2,nefc=nefc,contact=contacts,geom_xpos=centers,geom_xmat=np.tile(np.eye(3).ravel(),(4,1)),efc_J=J,efc_force=force,efc_R=np.ones(nefc),efc_aref=force.copy(),qacc=np.zeros(51),qfrc_constraint=J.T@force)
 model=NS(geom_size=np.tile([.01,.02,0],(4,1)),geom_pos=centers.copy(),geom_quat=np.tile([1.,0,0,0],(4,1)),geom_sameframe=np.zeros(4,int))
 engine=NS(force=lambda m,d,k:np.r_[force[3*k:3*k+3],np.zeros(3)])
 return model,data,anchors,rotations,pairs,engine


@pytest.mark.parametrize('reverse',[False,True])
def test_two_independent_finger_Jacobians_sum_forces_and_virtual_work(monkeypatch,reverse):
 model,data,a,R,pairs,engine=two_contacts(reverse);monkeypatch.setattr(c,'frames',lambda *args:(a,R));v=np.sin(np.arange(51))
 out=c.multi_contact(model,data,None,engine,v,pairs)
 assert out['simultaneous'] and out['contact_active'].tolist()==[True,True]
 for key in ('contact_J_error','contact_gap_error_m','contact_normal_error','generalized_translation_force_error_N','generalized_angular_force_error_Nm','virtual_power_error_W'):assert np.max(out[key])<1e-14
 np.testing.assert_allclose(out['independent_contact_J'].reshape(6,51).T@out['contact_force'].ravel(),data.qfrc_constraint,atol=1e-15)


@pytest.mark.parametrize('fault',['duplicate','extra_constraint','5mm','wrong_force','wrong_compliance'])
def test_two_finger_contact_refuses_bad_inventory_and_force_semantics(monkeypatch,fault):
 model,data,a,R,pairs,engine=two_contacts();monkeypatch.setattr(c,'frames',lambda *args:(a,R))
 if fault=='duplicate':data.contact[1].geom[:]=data.contact[0].geom
 elif fault=='extra_constraint':data.nefc=7;data.efc_J=np.vstack((data.efc_J,np.zeros((1,51))))
 elif fault=='5mm':data.contact[0].dist=-.006
 elif fault=='wrong_force':engine.force=lambda *a:np.ones(6)
 else:data.contact[0].solref[0]=.01
 with pytest.raises(RuntimeError):c.multi_contact(model,data,None,engine,np.zeros(51),pairs)


def test_two_plane_tangency_allows_absent_constraints_but_not_wrong_frames_or_duplicate_contacts():
 model,data,a,R,pairs,engine=two_contacts(active=False)
 for n in (0,1,2):
  data.ncon=n;checks=c.two_plane_preflight(model,data,pairs);assert all(checks.values()) and all(type(x) is bool for x in checks.values())
 data.ncon=2;data.contact[1].geom[:]=data.contact[0].geom
 assert not all(c.two_plane_preflight(model,data,pairs).values())
 data.ncon=0;data.geom_xpos[1,0]+=.01
 assert not c.two_plane_preflight(model,data,pairs)['pair0_plane_position']


@pytest.mark.parametrize('fault',['bone','wrist','gravmap','contact_missing','single_finger','one_zero_reaction','clearance'])
def test_physical_assessment_keeps_all16bones_both_contacts_and_fixed_clearance(fault):
 kind='two_finger_planes' if fault in ('contact_missing','single_finger','one_zero_reaction') else 'guarded_sphere_table_approach' if fault=='clearance' else 'gravity_motion_hold';f=next(x for x in c.fixture_plan() if x['kind']==kind);n=f['steps']
 a=dict(endpoint_time_s=np.arange(1,n+1)*f['dt_s'],bone_angle_error_rad=np.zeros((n,16)),wrist_goal_error_m=np.zeros(n),gravity_translation_error_N=np.zeros(n),gravity_angular_error_Nm=np.zeros(n),actual_clearance_lower_m=np.full(n,.01),contact_active=np.ones((n,2),bool),normal_force_N=np.ones((n,2)))
 for key in ('contact_J_error','contact_gap_error_m','contact_normal_error','cone_primal','cone_dual','cone_complementarity'):a[key]=np.zeros((n,2))
 for key in ('generalized_translation_force_error_N','generalized_angular_force_error_Nm','virtual_power_error_W','virtual_power_scale_W'):a[key]=np.zeros(n)
 a['cone_force_scale']=a['cone_gradient_scale']=np.ones((n,2))
 assert c.assess(f,a,{}, {},{},dict(passed=True))['status']=='PASS'
 if fault=='bone':a['bone_angle_error_rad'][-1,15]=np.deg2rad(2.1)
 elif fault=='wrist':a['wrist_goal_error_m'][-1]=.0021
 elif fault=='gravmap':a['gravity_angular_error_Nm'][0]=1.1e-10
 elif fault=='contact_missing':a['contact_active'][-1,0]=False
 elif fault=='single_finger':a['normal_force_N'][:,1]=0
 elif fault=='one_zero_reaction':a['normal_force_N'][-1,1]=0
 else:a['actual_clearance_lower_m'][0]=.0019
 assert c.assess(f,a,{}, {},{},dict(passed=True))['status']=='FAIL'


def test_gravity_diagnostics_uses_pre_step_velocity_for_force_clock(monkeypatch):
 anchors=np.zeros((16,3));R=np.tile(np.eye(3),(16,1,1));mass=np.full(16,.025);COM=np.zeros((17,3));model=NS(body_mass=np.r_[0,mass],opt=NS(gravity=np.array([0,0,-9.81])))
 expected=c.p.gravity_compensation_force(mass,COM[1:],anchors,R,model.opt.gravity);data=NS(xipos=COM,qfrc_gravcomp=expected,qfrc_bias=expected+1,qfrc_passive=expected,qvel=np.zeros(51));contract=NS(body_ids=tuple(range(1,17)));monkeypatch.setattr(c,'frames',lambda *a:(anchors,R))
 receipt,_=c.gravity_diagnostics(model,data,contract,np.ones(51))
 assert receipt['stationary'] is False and all(receipt['checks'].values())
 assert not all(c.gravity_diagnostics(model,data,contract)[0]['checks'].values())


def test_planes_derive_from_exact_saved_q0_before_compile_and_never_mutate_world_geometry(tmp_path):
 goal=q0();a={};expected=[]
 for j,bone in enumerate((3,6)):
  R=u.matrix(u.quat_exp(np.array([.1*j,.2,-.1])));center=np.array([.1*j,.02,.1]);size=np.array([.01,.02,0]);normal=u.plane_normal(R);position=center-normal*(size[0]+size[1]/np.sqrt(2));quat=u.plane_quaternion(normal);prefix=f'pair{j}_'
  a.update({prefix+'pre_qpos':goal,prefix+'ncon':np.asarray(0),prefix+'actual_geom_xpos':np.stack((center,position)),prefix+'actual_geom_xmat':np.stack((R,u.matrix(quat))),prefix+'actual_geom_pos':np.stack((np.zeros(3),position)),prefix+'actual_geom_size':np.stack((size,[1,1,.01]))});expected.append(position)
 path=tmp_path/'sealed.npz';np.savez_compressed(path,**a);row=dict(path=str(path),sha256=c.sha(path),bytes=path.stat().st_size);planes=c.planes_from_saved_witness(row,goal)
 for j in (0,1):np.testing.assert_array_equal(planes[j]['position_m'],expected[j])
 source,root=manufactured_xml(tmp_path);xml=ET.fromstring(c.scene_xml(source,'two_finger_planes',.002,planes))
 for j,g in enumerate(xml.findall('./worldbody/geom')):np.testing.assert_array_equal([float(x) for x in g.get('pos').split()],expected[j])
 assert 'place_world_plane(' not in Path(c.__file__).read_text()
 changed=goal.copy();changed[0]=1e-12
 with pytest.raises(ValueError,match='Exact original q0'):c.planes_from_saved_witness(row,changed)
 path.write_bytes(b'tampered')
 with pytest.raises(ValueError,match='changed'):c.planes_from_saved_witness(row,goal)


def test_compiled_world_cache_contains_geometry_and_rejects_stale_original_plane_bounds():
 model,data,a,R,pairs,engine=two_contacts(active=False)
 model.body_bvhadr=np.array([0]);model.body_bvhnum=np.array([3]);model.body_geomadr=np.array([0]);model.body_geomnum=np.array([2]);model.body_contype=np.array([3]);model.body_conaffinity=np.array([3]);model.bvh_child=np.array([[1,2],[-1,-1],[-1,-1]]);model.bvh_nodeid=np.array([-1,1,3]);model.bvh_depth=np.array([0,1,1]);model.bvh_aabb=np.array([[0.,0,0,100,100,100]]*3);model.geom_aabb=np.zeros((4,6));data.xipos=np.zeros((1,3));data.ximat=np.eye(3).ravel()[None]
 r=c.world_cache_receipt(model,data,pairs);assert all(r['checks'].values()) and r['runtime_world_geometry_mutation'] is False
 before=c.cache_signature(model);sealed=c.final_model_witness(model);model.bvh_aabb[:,2]=-100;model.bvh_aabb[:,5]=.001
 r=c.world_cache_receipt(model,data,pairs);assert not all(r['checks'].values()) and c.cache_signature(model)!=before
 assert all(x['passed'] is False for row in r['planes'] for x in row['cache_checks'])
 np.testing.assert_array_equal(sealed['final_model_bvh_aabb'][:,2],np.zeros(3));np.testing.assert_array_equal(sealed['final_model_geom_quat'],model.geom_quat)


def test_approach_final_tracking_window_is_inside_fixed_half_second_dwell():
 f=next(x for x in c.fixture_plan() if x['kind']=='guarded_sphere_table_approach');goal=q0();start=goal.copy();start[:3]=c.PATH_SHIFT
 assert f['duration_s']==1.5
 for t in (1.,1.1,1.4,1.5):np.testing.assert_array_equal(c.reference_qpos(f,start,goal,np.zeros(3),t),goal)
 assert c.LIMITS['tracking_wrist_m']==.002 and c.LIMITS['positive_contact_force_N']==1e-8
