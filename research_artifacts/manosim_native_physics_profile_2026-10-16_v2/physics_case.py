"""Prospective compensated native capsule fixtures; no native work on import."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,hashlib,json,sys
from xml.etree import ElementTree as ET
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
API=ROOT/'src/grasp_failure_prediction/evaluation/manosim_capsule.py'
SCHEMA='manosim_native_compensated_physics_case_v2'
BUDGET=dict(models=6,data=6,static_forwards=120,steps=5625,contact_force_reads=2250,maximum_seconds=300,outer_seconds=310)
FORBIDDEN=dict(source_MANO_calls=0,HUG_calls=0,retarget_calls=0,training_calls=0,unregistered_engine_calls=0)
KINDS=('gravity_motion_hold','two_finger_planes','guarded_sphere_table_approach')
DURATIONS=(1.5,.75,1.5)
SPHERE_CENTER=np.array([.3,0.,.1]);SPHERE_RADIUS=.02;TABLE_Z=-.1
PATH_KNOTS=51;PATH_SHIFT=np.array([-.06,0.,0.]);PATH_CLEARANCE=.002;ORACLE_ERROR=1e-9


def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
p=load('native_physics_profile_math',HERE/'profile_math.py');hand_api=load('native_physics_hand_api',HERE/'native_hand.py')
old=load('native_physics_frozen_capsule_case',ROOT/'research_artifacts/manosim_capsule_execution_2026-10-15_v3/capsule_case.py');u=old.math
LIMITS=dict(u.LIMITS)

def fixture_plan():return [dict(index=i,kind=k,dt_s=dt,duration_s=T,steps=int(round(T/dt))) for i in (20,28) for dt in (.002,.001) for k,T in zip(KINDS,DURATIONS,strict=True)]
math=SimpleNamespace(FLAGS=p.FLAGS,LIMITS=LIMITS,fixture_plan=fixture_plan)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CACHE_ARRAYS=('body_bvhadr','body_bvhnum','body_geomadr','body_geomnum','body_contype','body_conaffinity','bvh_aabb','bvh_child','bvh_nodeid','bvh_depth','geom_aabb')

def cache_signature(model):
 h=hashlib.sha256()
 for key in CACHE_ARRAYS:
  a=np.ascontiguousarray(getattr(model,key));h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()


def final_model_witness(model):
 return {f'final_model_{key}':np.asarray(getattr(model,key)).copy() for key in (*CACHE_ARRAYS,'geom_pos','geom_quat','geom_sameframe')}


def planes_from_saved_witness(witness,goal):
 """Pure saved-array plan; same radial+axial normals/depth, no fresh FK/queries."""
 path=Path(witness['path'])
 if sha(path)!=witness['sha256'] or path.stat().st_size!=witness['bytes']:raise ValueError('Bound pre-step plane witness changed')
 with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 rows=[]
 for j,bone in enumerate((3,6)):
  prefix=f'pair{j}_';q=a[prefix+'pre_qpos']
  if q.shape!=(67,) or not np.array_equal(q,goal):raise ValueError('Exact original q0 plane witness required')
  if int(a[prefix+'ncon'])!=0:raise ValueError('Original pre-step tangent witness must have zero contacts')
  center=p.finite(a[prefix+'actual_geom_xpos'][0],(3,));R=p.finite(a[prefix+'actual_geom_xmat'][0],(3,3));size=p.finite(a[prefix+'actual_geom_size'][0],(3,));normal=u.plane_normal(R)
  if size[0]<=0 or size[1]<=0:raise ValueError('Positive original capsule dimensions required')
  position=center-normal*(size[0]+size[1]/np.sqrt(2));quaternion=u.plane_quaternion(normal)
  if np.max(np.abs(position-a[prefix+'actual_geom_pos'][1]))>LIMITS['contact_gap_m'] or np.max(np.abs(u.matrix(quaternion)-a[prefix+'actual_geom_xmat'][1]))>LIMITS['contact_normal']:raise ValueError('Source plane construction differs from original witness')
  rows.append(dict(bone=bone,position_m=position.tolist(),quaternion_wxyz=quaternion.tolist(),normal_world=normal.tolist(),capsule_center_m=center.tolist(),capsule_rotation=R.tolist(),capsule_size_m=size.tolist()))
 return rows


def world_cache_receipt(model,data,pairs):
 """Validate saved cache contains each compile-time plane origin/tangent point.

 Arithmetic allowance scales only cache representation (~mjMAXVAL). It never
 enters physical plane-gap, contact, normal or tracking thresholds.
 """
 adr=int(model.body_bvhadr[0]);num=int(model.body_bvhnum[0]);checks=dict(world_has_two_compiled_planes=bool(model.body_geomnum[0]==2),world_has_static_tree=bool(adr>=0 and num==3));receipts=[]
 if not all(checks.values()):raise ValueError('Exact compiled two-plane world BVH required')
 bounds=np.asarray(model.bvh_aabb,dtype=float);nodes=np.asarray(model.bvh_nodeid);children=np.asarray(model.bvh_child)
 if bounds.ndim!=2 or bounds.shape[1]!=6 or not np.isfinite(bounds).all():raise ValueError('Finite actual compiled BVH bounds required')
 world_position=p.finite(data.xipos[0],(3,));world_rotation=p.finite(np.asarray(data.ximat[0]).reshape(3,3),(3,3))
 for j,(hg,pg,bone) in enumerate(pairs):
  leaf=np.flatnonzero(nodes[adr:adr+num]==pg)+adr
  if len(leaf)!=1 or np.any(children[int(leaf[0])]!=-1):raise ValueError('Exactly one compiled BVH leaf per declared plane required')
  plane_position=p.finite(data.geom_xpos[pg],(3,));normal=p.finite(np.asarray(data.geom_xmat[pg]).reshape(3,3)[:,2],(3,));axis=np.asarray(data.geom_xmat[hg]).reshape(3,3)[:,2];size=model.geom_size[hg]
  point=data.geom_xpos[hg]-np.sign(float(normal@axis))*size[1]*axis-size[0]*normal
  targets=np.stack((plane_position,point));local=np.einsum('ji,nj->ni',world_rotation,targets-world_position,optimize=False)
  errors=[]
  for node in (adr,int(leaf[0])):
   b=bounds[node];scale=max(1.,float(np.abs(b).max()),float(np.abs(local).max()));allowance=4096*np.finfo(float).eps*scale
   excess=float(np.max(np.abs(local-b[:3])-b[3:]));ok=bool(np.all(b[3:]>=0) and excess<=allowance);checks[f'plane{j}_cache_node{node}_contains_geometry']=ok;errors.append(dict(node=node,maximum_outside_m=excess,representation_arithmetic_allowance_m=allowance,passed=ok))
  receipts.append(dict(plane_geom_id=int(pg),capsule_geom_id=int(hg),leaf_node=int(leaf[0]),world_plane_origin_m=plane_position.tolist(),tangent_point_m=point.tolist(),cache_checks=errors))
 return dict(schema_version='manosim_compile_time_world_plane_cache_v1',checks=checks,planes=receipts,cache_signature=cache_signature(model),runtime_world_geometry_mutation=False,physical_acceptance_allowance_changed=False,old_v1_cache_arrays_measured=False)


def scene_xml(base,kind,dt,plane_spec=None):
 if kind not in KINDS or dt not in (.002,.001):raise ValueError('Declared physics fixture required')
 base=Path(base).resolve();root=ET.fromstring(p.gravity_profile_xml(base.read_text()));geoms=root.findall('./worldbody//geom')
 if len(geoms)!=16 or len(root.findall('./default/geom'))!=1:raise ValueError('Original16 collider inventory required')
 for mesh in root.findall('./asset/mesh'):mesh.set('file',str((base.parent/mesh.get('file')).resolve()))
 opt=root.find('option')
 if opt is None:opt=ET.SubElement(root,'option')
 for k,v in dict(timestep=str(dt),integrator='implicitfast',cone='elliptic',jacobian='dense',gravity='0 0 -9.81').items():opt.set(k,v)
 flag=opt.find('flag')
 if flag is None:flag=ET.SubElement(opt,'flag')
 flag.set('contact','enable' if kind!='gravity_motion_hold' else 'disable');flag.set('energy','enable')
 for g in geoms:g.set('contype','0');g.set('conaffinity','0')
 world=root.find('worldbody')
 if kind=='two_finger_planes':
  if not isinstance(plane_spec,list) or [x.get('bone') for x in plane_spec]!=[3,6]:raise ValueError('Precompiled literal world-plane specification required')
  for j,bone in enumerate((3,6)):
   g=root.find(f'./worldbody//body[@name="mano_body_{bone}"]/geom');g.set('name',f'qualification_capsule_{bone}');g.set('contype',str(1<<j));g.set('conaffinity',str(1<<j))
   ET.SubElement(world,'geom',name=f'qualification_plane_{bone}',type='plane',size='1 1 .01',pos=' '.join(map(str,p.finite(plane_spec[j]['position_m'],(3,)))),quat=' '.join(map(str,p.finite(plane_spec[j]['quaternion_wxyz'],(4,)))),contype=str(1<<j),conaffinity=str(1<<j),condim='3',friction='1 .005 .0001')
 elif kind=='guarded_sphere_table_approach':
  for g in geoms:g.set('contype','2');g.set('conaffinity','1')
  ET.SubElement(world,'geom',name='approach_table',type='plane',size='1 1 .01',pos=f'0 0 {TABLE_Z}',contype='1',conaffinity='2')
  ET.SubElement(world,'geom',name='approach_sphere',type='sphere',size=str(SPHERE_RADIUS),pos=' '.join(map(str,SPHERE_CENTER)),contype='1',conaffinity='2')
 return ET.tostring(root,encoding='unicode')+'\n'


def frames(data,c):return old.frames(data,c)
def collider_geometry(model,data,c):
 gids=[]
 for bid in c.body_ids:
  g=np.flatnonzero(np.asarray(model.geom_bodyid)==bid)
  if len(g)!=1:raise ValueError('Exactly one collider per named hand body required')
  gids.append(int(g[0]))
 if list(np.asarray(model.geom_type)[gids])!=[7,*([3]*15)]:raise ValueError('Original convex palm plus15capsules required')
 mesh=int(model.geom_dataid[gids[0]]);start=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh]);vertices=np.asarray(model.mesh_vert[start:start+n]).astype(np.float64)
 return dict(ids=np.asarray(gids),centers=np.asarray(data.geom_xpos)[gids].copy(),rotations=np.asarray(data.geom_xmat)[gids].reshape(16,3,3).copy(),sizes=np.asarray(model.geom_size)[gids].copy(),palm_vertices=vertices.copy())


def gravity_diagnostics(model,data,c,qvel_at_force_clock=None):
 anchors,R=frames(data,c);ids=list(c.body_ids);expected=p.gravity_compensation_force(model.body_mass[ids],np.asarray(data.xipos)[ids],anchors,R,model.opt.gravity)
 receipt=p.compensation_checks(expected,data.qfrc_gravcomp[:51],data.qfrc_bias[:51],data.qfrc_passive[:51],data.qvel[:51] if qvel_at_force_clock is None else qvel_at_force_clock)
 arrays=dict(expected_gravity_tangent=expected,actual_gravity_tangent=np.asarray(data.qfrc_gravcomp[:51]).copy(),actual_bias_tangent=np.asarray(data.qfrc_bias[:51]).copy(),actual_passive_tangent=np.asarray(data.qfrc_passive[:51]).copy(),body_COM_world=np.asarray(data.xipos)[ids].copy(),anchors=anchors,rotations=R)
 return receipt,arrays


def reference_qpos(f,start,goal,indent,t):
 if f['kind']=='gravity_motion_hold':return u.trajectory(start,goal,u.smooth_fraction(t,.5))
 if f['kind']=='guarded_sphere_table_approach':return u.trajectory(start,goal,u.smooth_fraction(t,1.))
 q=goal.copy();q[:3]+=indent*u.smooth_fraction(t,.15);return q


def multi_contact(model,data,c,engine,pre_v,pairs):
 """Independent sum over up to2 declared rigid contacts; no skin Jacobian."""
 if data.ncon>2:raise RuntimeError('Declared two-contact inventory exceeded')
 out=dict(contact_active=np.zeros(2,bool),normal_force_N=np.zeros(2),engine_gap_m=np.zeros(2),contact_J_error=np.zeros(2),contact_gap_error_m=np.zeros(2),contact_normal_error=np.zeros(2),cone_primal=np.zeros(2),cone_dual=np.zeros(2),cone_complementarity=np.zeros(2),cone_force_scale=np.ones(2),cone_gradient_scale=np.ones(2),contact_force=np.zeros((2,3)),contact_J=np.zeros((2,3,51)),independent_contact_J=np.zeros((2,3,51)),contact_frame=np.zeros((2,3,3)),contact_point=np.zeros((2,3)),contact_gradient=np.zeros((2,3)),contact_geom_ids=np.full((2,2),-1,int),capsule_center=np.zeros((2,3)),capsule_R=np.zeros((2,3,3)),plane_position=np.zeros((2,3)),plane_R=np.zeros((2,3,3)),analytic_capsule_gap_m=np.zeros(2),efc_R=np.zeros((2,3)),efc_aref=np.zeros((2,3)),simultaneous=False)
 for j,(hg,pg,bone) in enumerate(pairs):
  normal=np.asarray(data.geom_xmat[pg]).reshape(3,3)[:,2];gap=u.capsule_gap(data.geom_xpos[hg],data.geom_xmat[hg].reshape(3,3),model.geom_size[hg],data.geom_xpos[pg],normal);out['analytic_capsule_gap_m'][j]=gap
  if gap<-LIMITS['maximum_penetration_m']:raise RuntimeError('Declared5mm maximum contact penetration escaped')
 anchors,R=frames(data,c);Jall=np.asarray(data.efc_J).reshape(int(data.nefc),51);tau=np.zeros(51);seen=set();active_rows=set()
 for k in range(int(data.ncon)):
  contact=data.contact[k];key=frozenset(map(int,contact.geom));matches=[j for j,x in enumerate(pairs) if key==frozenset(x[:2])]
  if len(matches)!=1 or matches[0] in seen or int(contact.dim)!=3:raise RuntimeError('Unexpected or duplicate multifinger contact')
  j=matches[0];seen.add(j);hg,pg,bone=pairs[j];basis=u.finite(np.asarray(contact.frame).reshape(3,3),(3,3));point=u.finite(contact.pos,(3,));sign=1 if int(contact.geom[1])==hg else -1
  capR=np.asarray(data.geom_xmat[hg]).reshape(3,3);planeR=np.asarray(data.geom_xmat[pg]).reshape(3,3);n=planeR[:,2];gap=u.capsule_gap(data.geom_xpos[hg],capR,model.geom_size[hg],data.geom_xpos[pg],n)
  if min(float(contact.dist),gap)<-LIMITS['maximum_penetration_m']:raise RuntimeError('Declared5mm maximum contact penetration escaped')
  out['engine_gap_m'][j]=contact.dist;out['contact_gap_error_m'][j]=abs(gap-contact.dist);out['contact_normal_error'][j]=np.max(np.abs(basis[0]-sign*n));out['contact_frame'][j]=basis;out['contact_point'][j]=point;out['contact_geom_ids'][j]=contact.geom;out['capsule_center'][j]=data.geom_xpos[hg];out['capsule_R'][j]=capR;out['plane_position'][j]=data.geom_xpos[pg];out['plane_R'][j]=planeR
  adr=int(contact.efc_address)
  if adr<0:continue
  if adr+3>data.nefc or active_rows.intersection(range(adr,adr+3)):raise RuntimeError('Unexpected active contact row partition')
  active_rows.update(range(adr,adr+3));independent=sign*basis@u.body_point_J(point,bone,anchors,R);J=Jall[adr:adr+3];ef=np.asarray(data.efc_force[adr:adr+3]);force=engine.force(model,data,k)[:3]
  if not np.allclose(force,ef,rtol=0,atol=1e-10):raise RuntimeError('Elliptic contact coordinates differ')
  if not np.array_equal(contact.solref,[.04,1.]) or not np.array_equal(contact.solimp,[.5,.95,.001,.5,2.]) or not np.array_equal(contact.friction[:2],[1.,1.]):raise RuntimeError('Authored contact compliance/friction changed')
  reg=np.asarray(data.efc_R[adr:adr+3]);ref=np.asarray(data.efc_aref[adr:adr+3]);gradient=J@data.qacc+reg*ef-ref;cone=u.cone_residual(ef,gradient,float(contact.friction[0]));tau+=independent.T@force
  out['contact_active'][j]=True;out['normal_force_N'][j]=force[0];out['contact_force'][j]=force;out['contact_J'][j]=J;out['independent_contact_J'][j]=independent;out['contact_J_error'][j]=np.max(np.abs(J-independent));out['contact_gradient'][j]=gradient;out['efc_R'][j]=reg;out['efc_aref'][j]=ref
  for key in ('primal','dual','complementarity','force_scale','gradient_scale'):out['cone_'+key][j]=cone[key]
 if active_rows!=set(range(int(data.nefc))):raise RuntimeError('Undeclared constraint rows in multifinger fixture')
 residual=np.asarray(data.qfrc_constraint)-tau;probe=np.sin(np.arange(51));actual_power=float(np.asarray(data.qfrc_constraint)@pre_v);expected_power=float(tau@pre_v)
 out.update(simultaneous=bool(out['contact_active'].all()),generalized_translation_force_error_N=float(np.abs(residual[:3]).max()),generalized_angular_force_error_Nm=float(np.abs(residual[3:]).max()),virtual_power_error_W=max(abs(actual_power-expected_power),abs(float(residual@probe))),virtual_power_scale_W=max(abs(actual_power),abs(expected_power)))
 return out


def two_plane_preflight(model,data,pairs):
 """Touching may produce0..2 contacts; any present contact must match exactly."""
 checks={};seen=set();inventory=0<=int(data.ncon)<=2
 for j,(hg,pg,bone) in enumerate(pairs):
  R=p.finite(np.asarray(data.geom_xmat[pg]).reshape(3,3),(3,3));position=p.finite(data.geom_xpos[pg],(3,));n=R[:,2]
  gap=u.capsule_gap(data.geom_xpos[hg],data.geom_xmat[hg].reshape(3,3),model.geom_size[hg],position,n)
  checks[f'pair{j}_sameframe_NONE']=bool(model.geom_sameframe[pg]==0)
  checks[f'pair{j}_plane_position']=bool(np.max(np.abs(position-model.geom_pos[pg]))<=LIMITS['contact_gap_m'])
  checks[f'pair{j}_plane_rotation']=bool(np.max(np.abs(R-u.matrix(model.geom_quat[pg])))<=LIMITS['contact_normal'])
  checks[f'pair{j}_analytic_tangent_gap']=bool(abs(gap)<=LIMITS['contact_gap_m'])
 for k in range(int(data.ncon)):
  contact=data.contact[k];matches=[j for j,x in enumerate(pairs) if set(map(int,contact.geom))==set(x[:2])]
  if len(matches)!=1 or matches[0] in seen:inventory=False;continue
  j=matches[0];seen.add(j);hg,pg,bone=pairs[j];sign=1 if int(contact.geom[1])==hg else -1
  checks[f'contact{j}_dimension']=bool(int(contact.dim)==3)
  checks[f'contact{j}_engine_tangent_gap']=bool(np.isfinite(contact.dist) and abs(float(contact.dist))<=LIMITS['contact_gap_m'])
  basis=p.finite(np.asarray(contact.frame).reshape(3,3),(3,3));n=np.asarray(data.geom_xmat[pg]).reshape(3,3)[:,2]
  checks[f'contact{j}_normal']=bool(np.max(np.abs(basis[0]-sign*n))<=LIMITS['contact_normal'])
 checks['at_most_two_unique_declared_pairs']=bool(inventory)
 checks['zero_contact_no_constraint_rows']=bool(data.ncon!=0 or (data.nefc==0 and not np.any(data.qfrc_constraint)))
 return checks


def prepare_case(index,spec,protocol,output,engine,notify):
 import mujoco
 api=load('physics_frozen_mapping_api',API);output=Path(output);artifacts=[];prepared=[]
 def seal(path,label):
  row=dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,label=label);artifacts.append(row);notify(dict(event='artifact',artifact=row));return row
 def save(path,label,**arrays):
  with Path(path).open('xb') as f:np.savez_compressed(f,**arrays)
  return seal(Path(path),label)
 with np.load(spec['literal_buffers']['path'],allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
 with np.load(spec['source_readbacks']['path'],allow_pickle=False) as z:src={k:z[k].copy() for k in z.files}
 target=api.target_from_source(b['original_postAA32'],b['original_camera_translation32'],b['original_cameraWorld64'],b['rest_quaternions'],shape_float32=b['shape'])
 if not np.array_equal(target.qpos,b['source_initialized_q0']) or not np.array_equal(target.qpos,src['actual_named_qpos'][0]):raise ValueError('Unchanged source target required')
 assets=api.inspect_authored_assets(spec['capsule_xml'],spec['shape_asset'],spec['builder_report'],expected_shape_sha256=spec['expected_shape_sha256'],native_shape_float32=b['shape'])
 for f in [x for x in fixture_plan() if x['index']==index]:
  directory=output/(f['kind']+'_'+str(round(1/f['dt_s'])));directory.mkdir();xp=directory/'fixture.xml';xp.write_text(scene_xml(spec['capsule_xml'],f['kind'],f['dt_s'],spec['plane_spec']));seal(xp,'explicit_compensated_new_fixture_XML');model=engine.model(xp)
  resolve=lambda kind,name:int(mujoco.mj_name2id(model,{'body':mujoco.mjtObj.mjOBJ_BODY,'joint':mujoco.mjtObj.mjOBJ_JOINT,'actuator':mujoco.mjtObj.mjOBJ_ACTUATOR,'mesh':mujoco.mjtObj.mjOBJ_MESH,'geom':mujoco.mjtObj.mjOBJ_GEOM}[kind],name))
  c=old.validate_with_snapshot(model,resolve,assets,b['rest_quaternions'],api,save,directory,xp);profile=p.compensated_model_receipt(model,c);data=engine.data(model);goal=target.qpos.copy();data.qpos[:]=goal;data.qvel[:]=0;data.ctrl[:]=api.command_from_qpos(goal,c).ctrl;engine.forward(model,data)
  anchors,R=frames(data,c);cmd=api.command_from_qpos(goal,c);static=u.static_checks(data.qpos,goal,anchors,R,src['source_joint_anchors_world'][0],src['source_body_rotations_world'][0],data.actuator_length,cmd.ctrl,data.actuator_force);grav,ga=gravity_diagnostics(model,data,c)
  if not all(static.values()) or not all(grav['checks'].values()):raise ValueError('Static compensated source/transmission/gravity gate failed')
  save(directory/'static_command.npz','source_mapping_and_independent_gravity_cancellation',qpos=data.qpos.copy(),ctrl=data.ctrl.copy(),actuator_length=data.actuator_length.copy(),actuator_force=data.actuator_force.copy(),source_anchors=src['source_joint_anchors_world'][0],source_rotations=src['source_body_rotations_world'][0],**ga)
  start=goal.copy();indent=np.zeros(3);pairs=[];preflight=None;guard=None
  if f['kind']=='gravity_motion_hold':start[:3]+=[.01,-.01,.01];start[3:7]=u.quat_product(goal[3:7],u.quat_exp(np.array([.1,-.05,.08])));start[7:]=np.tile([1.,0,0,0],15)
  elif f['kind']=='two_finger_planes':
   normals=[]
   for j,bone in enumerate((3,6)):
    hg=resolve('geom',f'qualification_capsule_{bone}');pg=resolve('geom',f'qualification_plane_{bone}');plane=spec['plane_spec'][j];normal=u.plane_normal(data.geom_xmat[hg].reshape(3,3))
    if np.max(np.abs(normal-np.asarray(plane['normal_world'])))>LIMITS['contact_normal'] or np.max(np.abs(data.geom_xpos[hg]-np.asarray(plane['capsule_center_m'])))>LIMITS['contact_gap_m']:raise ValueError('New compiled q0 differs from bound plane witness')
    normals.append(normal);pairs.append((hg,pg,bone))
   indent,indent_receipt=p.two_plane_indentation(np.array(normals))
  else:
   start[:3]+=PATH_SHIFT;distances=[];geometry=[]
   for fraction in np.linspace(0,1,PATH_KNOTS):
    q=goal.copy();q[:3]+=(1-fraction)*PATH_SHIFT;data.qpos[:]=q;data.qvel[:]=0;data.ctrl[:]=api.command_from_qpos(q,c).ctrl;engine.forward(model,data);g=collider_geometry(model,data,c);d,receipt=p.analytic_hand_separations(g['centers'],g['rotations'],g['sizes'],g['palm_vertices'],SPHERE_CENTER,SPHERE_RADIUS,TABLE_Z);distances.append(d);geometry.append(g)
   offsets=(1-np.linspace(0,1,PATH_KNOTS))[:,None]*PATH_SHIFT
   guard=p.rigid_translation_guard(np.array(distances),offsets,required_clearance_m=PATH_CLEARANCE,oracle_error_m=ORACLE_ERROR,oracle_id='analytic_plane_sphere_hand_bounds_v1')
   save(directory/'approach_path.npz','all51knots_all16colliders_analytic_sphere_table_lower_bounds',distances_m=np.asarray(distances),root_offsets_m=offsets,collider_centers_world=np.asarray([x['centers'] for x in geometry]),collider_rotations_world=np.asarray([x['rotations'] for x in geometry]),collider_sizes_m=geometry[0]['sizes'],palm_vertices_local=geometry[0]['palm_vertices'],sphere_center=SPHERE_CENTER,sphere_radius=np.asarray(SPHERE_RADIUS),table_z=np.asarray(TABLE_Z))
   if not guard['passed']:raise ValueError('Fixed sphere/table approach guard refused; no replacement path')
  data.qpos[:]=start;data.qvel[:]=0;data.ctrl[:]=api.command_from_qpos(start,c).ctrl;data.qacc_warmstart[:]=0;engine.forward(model,data)
  if pairs:
   operands={}
   for j,(hg,pg,bone) in enumerate(pairs):
    a=old.contact_operands(model,data,hg,pg,pre_time_s=data.time,pre_qpos=data.qpos,pre_qvel=data.qvel,reason='SIMULTANEOUS_SECOND_FORWARD_PREFLIGHT');operands.update({f'pair{j}_{k}':v for k,v in a.items()})
   preflight=two_plane_preflight(model,data,pairs);cache=world_cache_receipt(model,data,pairs)
   preflight.update(cache['checks'])
   operands.update({f'cache_{key}':np.asarray(getattr(model,key)).copy() for key in CACHE_ARRAYS});operands.update(cache_body_inertial_position=np.asarray(data.xipos).copy(),cache_body_inertial_rotation=np.asarray(data.ximat).reshape(-1,3,3).copy())
   save(directory/'plane_preflight.npz','both_realized_plane_frames_and_tangent_pairs',**operands,indentation_world_m=indent,planned_normals=np.asarray(normals))
   if not all(preflight.values()):raise ValueError('Two-plane preflight refused before dynamics')
  save(directory/'fixture_model.npz','unchanged_native_inertials_gains_plus_named_gravity_profile',**{key:np.asarray(getattr(model,key)).copy() for key in (*api._COMPILED_ARRAYS,*CACHE_ARRAYS)},jnt_actgravcomp=model.jnt_actgravcomp.copy(),gravity=model.opt.gravity.copy(),native_target=goal,initial_state=start,indentation_world_m=indent)
  samples=np.array([reference_qpos(f,start,goal,indent,k*f['dt_s']) for k in range(f['steps']+1)]);branches=api.require_continuous_principal_targets(samples,c)
  meta=directory/'command_contract.json';meta.write_text(json.dumps(dict(target=target.metadata(),compiled=c.metadata(),gravity_profile=profile,static_checks=static,gravity_checks=grav,plane_preflight=preflight,world_cache=cache if pairs else None,world_geometry_runtime_mutations=False,approach_guard=guard,planned_branch_minimum_margin_rad=float(min(x.pi_margin_rad.min() for x in branches))),sort_keys=True,indent=2,allow_nan=False)+'\n');seal(meta,'pre_dynamics_source_gravity_path_contact_and_branch_gates')
  prepared.append(dict(fixture=f,directory=directory,model=model,data=data,contract=c,goal=goal,start=start,indent=indent,pairs=pairs,static=static,gravity_checks=grav,plane_preflight=preflight,cache_signature=cache_signature(model),approach_guard=guard));notify(dict(event='static_fixture_preflight_completed',fixture=f,steps=0))
 return dict(index=index,api=api,source=src,artifacts=artifacts,prepared=prepared,save=save)


def assess(f,a,static,gravity,planes,guard):
 if len(a['endpoint_time_s'])!=f['steps'] or not np.isfinite(a['endpoint_time_s']).all():raise ValueError('Exact declared finite trajectory required')
 tail=a['endpoint_time_s']>=f['duration_s']-.1-1e-12
 if not tail.any():raise ValueError('Final assessment window missing')
 checks={};metrics={}
 checks['all16_bone_tracking']=bool(a['bone_angle_error_rad'][tail].max()<=LIMITS['tracking_bone_rad']);checks['wrist_to_unchanged_goal']=bool(a['wrist_goal_error_m'][tail].max()<=LIMITS['tracking_wrist_m'])
 checks['independent_gravity_force_map']=bool(np.all(a['gravity_translation_error_N']<=p.FORCE_LIMIT_N) and np.all(a['gravity_angular_error_Nm']<=p.TORQUE_LIMIT_NM))
 metrics.update(wrist_tail_max_m=float(a['wrist_goal_error_m'][tail].max()),bone_tail_max_rad=float(a['bone_angle_error_rad'][tail].max()))
 if f['kind']=='two_finger_planes':
  checks['two_simultaneous_positive_tail_reactions']=bool(np.all(a['contact_active'][tail]) and np.all(a['normal_force_N'][tail]>LIMITS['positive_contact_force_N']))
  for field,key in [('contact_J_error','contact_J'),('contact_gap_error_m','contact_gap_m'),('contact_normal_error','contact_normal'),('generalized_translation_force_error_N','contact_translation_force_N'),('generalized_angular_force_error_Nm','contact_angular_force_Nm')]:checks[field]=bool(a[field].max()<=LIMITS[key])
  checks['summed_virtual_power']=bool(np.all(a['virtual_power_error_W']<=LIMITS['virtual_power_absolute_W']+LIMITS['virtual_power_relative']*a['virtual_power_scale_W']))
  for kind,limit in [('primal','cone_primal_relative'),('dual','cone_dual_relative'),('complementarity','cone_complementarity_relative')]:
   scale=a['cone_force_scale'] if kind=='primal' else a['cone_gradient_scale'] if kind=='dual' else a['cone_force_scale']*a['cone_gradient_scale'];checks['cone_'+kind]=bool(np.all(a['cone_'+kind]<=LIMITS[limit]*scale))
 if f['kind']=='guarded_sphere_table_approach':checks['predeclared_swept_analytic_guard']=bool(guard and guard['passed']);checks['actual_trajectory_minimum_clearance']=bool(np.all(a['actual_clearance_lower_m']>=PATH_CLEARANCE))
 return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,metrics=metrics,static_source_and_transmission_checks=static,gravity_preflight=gravity,plane_preflight=planes,approach_guard=guard,physical_outcome=None,**p.FLAGS)


def execute_case(context,engine,notify):
 api=context['api'];src=context['source'];save=context['save'];measurements=[];context['measurements']=measurements
 for s in context['prepared']:
  context['current_session']=s;f=s['fixture'];directory=s['directory'];model=s['model'];data=s['data'];c=s['contract'];goal=s['goal'];rows=[];context['current_fixture']=f;context['partial_rows']=rows
  for k in range(f['steps']):
   t=k*f['dt_s'];reference=reference_qpos(f,s['start'],goal,s['indent'],t);cmd=api.command_from_qpos(reference,c);data.ctrl[:]=cmd.ctrl;pre_q=data.qpos.copy();pre_v=data.qvel.copy();clock=float(data.time);engine.step(model,data);anchors,R=frames(data,c);grav,ga=gravity_diagnostics(model,data,c,pre_v)
   # All geometry/forces returned by mj_step share pre_time_s, endpoints separate.
   extra={}
   if s['pairs']:
    try:extra=multi_contact(model,data,c,engine,pre_v,s['pairs'])
    except BaseException:
     operands={}
     for j,(hg,pg,bone) in enumerate(s['pairs']):operands.update({f'pair{j}_{name}':value for name,value in old.contact_operands(model,data,hg,pg,pre_time_s=clock,pre_qpos=pre_q,pre_qvel=pre_v,reason='MULTIFINGER_GUARD_FAILURE').items()})
     save(directory/'failed_contact_operands.npz','FAILED_actual_multicontact_operands_before_refusal',**operands);raise
   else:
    if data.ncon or data.nefc or np.any(data.qfrc_constraint):raise RuntimeError('Guarded free-motion fixture generated unintended contact')
   clearance=np.nan
   if f['kind']=='guarded_sphere_table_approach':
    g=collider_geometry(model,data,c);d,_=p.analytic_hand_separations(g['centers'],g['rotations'],g['sizes'],g['palm_vertices'],SPHERE_CENTER,SPHERE_RADIUS,TABLE_Z);clearance=float(d.min()-ORACLE_ERROR)
    if clearance<PATH_CLEARANCE:raise RuntimeError('Actual approach geometry escaped declared clearance; no replanning')
   delta=np.abs(ga['actual_gravity_tangent']-ga['expected_gravity_tangent'])
   row=dict(pre_time_s=clock,endpoint_time_s=float(data.time),pre_qpos=pre_q,pre_qvel=pre_v,endpoint_qpos=data.qpos.copy(),endpoint_qvel=data.qvel.copy(),reference_qpos=reference,requested_ctrl=cmd.ctrl.copy(),effective_ctrl=data.ctrl.copy(),anchors_start=anchors,body_rotations_start=R,bone_angle_error_rad=u.angle_errors(R,src['source_body_rotations_world'][0]),wrist_goal_error_m=float(np.linalg.norm(data.qpos[:3]-goal[:3])),start_qacc=data.qacc.copy(),start_qfrc_gravcomp=data.qfrc_gravcomp.copy(),start_expected_gravcomp=ga['expected_gravity_tangent'],start_body_COM_world=ga['body_COM_world'],start_qfrc_bias=data.qfrc_bias.copy(),start_qfrc_passive=data.qfrc_passive.copy(),start_qfrc_actuator=data.qfrc_actuator.copy(),start_qfrc_constraint=data.qfrc_constraint.copy(),gravity_translation_error_N=float(delta[:3].max()),gravity_angular_error_Nm=float(delta[3:].max()),warning_counts=data.warning.number.copy(),actual_clearance_lower_m=clearance if np.isfinite(clearance) else 0.,**extra)
   if not all(np.isfinite(np.asarray(v)).all() for v in row.values()):raise RuntimeError('Nonfinite compensated fixture channel')
   rows.append(row)
  engine.forward(model,data);api.verify_compiled_contract(model,c);p.compensated_model_receipt(model,c)
  if cache_signature(model)!=s['cache_signature']:raise ValueError('Compiled collision-cache arrays changed')
  arrays={key:np.array([r[key] for r in rows]) for key in rows[0]};save(directory/'trajectory.npz','actual_gravity_compensated_force_clock_and_state_endpoints',**arrays,**final_model_witness(model));assessment=assess(f,arrays,s['static'],s['gravity_checks'],s['plane_preflight'],s['approach_guard'])
  measurements.append(dict(fixture=f,assessment=assessment,final_qpos=data.qpos.tolist(),final_rotations=frames(data,c)[1].tolist(),command_clipping=False,physical_outcome=None,**p.FLAGS));notify(dict(event='fixture_completed',fixture=f,status=assessment['status']));context['current_fixture']=None;context['partial_rows']=[]
 convergence=[]
 for kind in KINDS:
  pair=[m for m in measurements if m['fixture']['kind']==kind];dx=float(np.linalg.norm(np.array(pair[0]['final_qpos'])[:3]-np.array(pair[1]['final_qpos'])[:3]));dr=float(u.angle_errors(pair[0]['final_rotations'],pair[1]['final_rotations']).max());convergence.append(dict(kind=kind,wrist_difference_m=dx,bone_difference_rad=dr,passed=dx<=LIMITS['convergence_wrist_m'] and dr<=LIMITS['convergence_bone_rad']))
 return dict(schema_version=SCHEMA,index=context['index'],status='COMPLETED',measurements=measurements,convergence=convergence,qualification='PASS' if all(m['assessment']['status']=='PASS' for m in measurements) and all(x['passed'] for x in convergence) else 'NOT_QUALIFIED',artifacts=context['artifacts'],forbidden_counts=FORBIDDEN,physical_outcome=None,**p.FLAGS)
