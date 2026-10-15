"""Fixed material-point generalized-load proof. No work occurs at import.

Synthetic planes may load backfaces. This is not collision search, object
orientation, rigid-root-wrench equivalence, integration or native admission.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys
from xml.etree import ElementTree as ET
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SKIN=ROOT/'research_artifacts/manosim_articulated_skin_2026-10-15_v1'
SOURCE=ROOT/'research_artifacts/manosim_native_source_pipeline_2026-10-15_v1'
BUDGET=dict(maximum_seconds_per_case=600,model_compilations=1,data_constructions=1,
 baseline_skin_evaluations=1,jvp_skin_evaluations=51,fd_skin_evaluations=102,vjp_skin_evaluations=1,
 material_points=6,law_evaluations=12,static_load_conditions=12,static_forward_calls=24,full_mass_matrix_calls=12,
 physics_steps=0,object_geometry_queries=0,source_MANO_calls=0)
GATES=dict(derivative_absolute=1e-8,derivative_relative=1e-6,finite_difference_step=1e-6,
 virtual_work_absolute_W=1e-10,virtual_work_relative=1e-10,translation_mass_residual_N=1e-8,angular_mass_residual_Nm=1e-10,
 linear_reaction_N=1e-9,common_point_angular_reaction_Nm=1e-10,source_alignment_vertex_m=5e-6,source_alignment_normal_component=1e-6,
 numerical_limits_are_contact_allowances=False)
PARAMETERS=dict(stiffness_n_per_m=20000.,damping_ns_per_m=2*np.sqrt(20000.*.2),friction=.5,friction_speed_m_per_s=.001)
FLAGS=dict(accepted_mapping=False,native_contact_qualified=False,physical_dynamics_qualified=False,new_collection_or_training_allowed=False)
FORBIDDEN=dict(physics_steps=0,object_geometry_queries=0,source_MANO_calls=0,HUG_calls=0,retarget_calls=0,training_calls=0,stock_contact_force_calls=0)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def scene_xml(base):
 base=Path(base).resolve();root=ET.parse(base).getroot()
 if root.tag!='mujoco' or root.findall('.//include') or root.findall('.//deformable'):raise ValueError('Standalone original hand XML required')
 geoms=root.findall('./worldbody//geom');defaults=root.findall('./default/geom')
 if len(root.findall('./asset/mesh'))!=16 or len(geoms)!=16 or len(defaults)!=1:raise ValueError('Exact16worldgeometries/one unchanged default template required')
 default_before=ET.tostring(defaults[0],encoding='unicode')
 # Compare protected attributes/individual inertial/joint/actuator elements,
 # rather than serialized body children whose collision masks intentionally change.
 before=[(x.tag,dict(x.attrib)) for x in root.iter() if x.tag in ('body','joint','inertial','motor','position','velocity','general')]
 for mesh in root.findall('./asset/mesh'):
  p=(base.parent/mesh.attrib['file']).resolve()
  if not p.is_file():raise ValueError('Original mesh asset missing')
  mesh.set('file',str(p))
 for g in geoms:g.set('contype','0');g.set('conaffinity','0')
 opt=root.find('option')
 if opt is None:opt=ET.SubElement(root,'option')
 flag=opt.find('flag')
 if flag is None:flag=ET.SubElement(opt,'flag')
 flag.set('contact','disable');flag.set('actuation','disable')
 after=[(x.tag,dict(x.attrib)) for x in root.iter() if x.tag in ('body','joint','inertial','motor','position','velocity','general')]
 if before!=after or ET.tostring(defaults[0],encoding='unicode')!=default_before:raise ValueError('Original skeleton/inertials/actuators/default template changed')
 return ET.tostring(root,encoding='unicode')+'\n',hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()


def mass_response(M,baseline_acceleration,loaded_acceleration,tau):
 M=np.asarray(M);a0=np.asarray(baseline_acceleration);aF=np.asarray(loaded_acceleration);t=np.asarray(tau)
 if M.shape!=(51,51) or any(x.shape!=(51,) for x in (a0,aF,t)) or not all(np.isfinite(x).all() for x in (M,a0,aF,t)):raise ValueError('Finite actual native51 mass operands required')
 if np.max(np.abs(M-M.T))>1e-12:raise ValueError('Actual mass matrix not symmetric')
 eigen=np.linalg.eigvalsh(M)
 if eigen[0]<=0:raise ValueError('Actual native inertia must be positive definite; no pseudoinverse')
 residual=np.einsum('ij,j->i',M,aF-a0,optimize=False)-t
 passed=bool(np.max(np.abs(residual[:3]))<=GATES['translation_mass_residual_N'] and np.max(np.abs(residual[3:]))<=GATES['angular_mass_residual_Nm'])
 return dict(passed=passed,translation_max_residual_N=float(np.max(np.abs(residual[:3]))),angular_max_residual_Nm=float(np.max(np.abs(residual[3:]))),minimum_mass_eigenvalue=float(eigen[0]),maximum_mass_eigenvalue=float(eigen[-1])),residual


def named_layout(model,mj,expected):
 names=['wrist_tx','wrist_ty','wrist_tz','wrist_rot']+[f'mano_joint_{j}' for j in range(1,16)]
 if (model.nq,model.nv,model.nu,model.njnt)!=(67,51,51,19):raise ValueError('Exact native67/51/51/19 required')
 bodies=[int(mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,f'mano_body_{i}')) for i in range(16)]
 if min(bodies)<1 or len(set(bodies))!=16:raise ValueError('All16 actual anatomical body names required')
 parents=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
 for i,bid in enumerate(bodies):
  if int(model.body_parentid[bid])!=(0 if i==0 else bodies[parents[i]]):raise ValueError('Original anatomical parent tree changed')
 rows=[]
 for k,name in enumerate(names):
  jid=int(mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,name))
  if jid<0:raise ValueError('Missing named native joint')
  row=dict(name=name,qpos_address=int(model.jnt_qposadr[jid]),dof_address=int(model.jnt_dofadr[jid]),type=int(model.jnt_type[jid]))
  wanted=dict(name=name,qpos_address=k if k<3 else 3+4*(k-3),dof_address=k if k<3 else 3+3*(k-3),type=int(mj.mjtJoint.mjJNT_SLIDE) if k<3 else int(mj.mjtJoint.mjJNT_BALL))
  if row!=wanted:raise ValueError('Named native tangent layout changed')
  if int(model.jnt_bodyid[jid])!=bodies[0 if k<4 else k-3]:raise ValueError('Named joint/anatomical body mismatch')
  if k<3 and not np.array_equal(model.jnt_axis[jid],np.eye(3)[k]):raise ValueError('Named world slide basis changed')
  rows.append(row)
 if rows!=expected:raise ValueError('Compiled layout differs from frozen source domain')
 return rows


def run_case(spec,output,protocol,checkpoint):
 import torch,mujoco
 if (np.__version__,torch.__version__,mujoco.__version__)!=('2.2.6','2.11.0','3.4.0'):raise ValueError('Exact isolated runtime required')
 if protocol['budget']!=BUDGET or protocol['gates']!=GATES or protocol['parameters']!=PARAMETERS:raise ValueError('Frozen point-coupling method changed')
 kin=load('coupling_camera_graph',HERE/'contact_kinematics.py');core=load('frozen_fullskin_core',SKIN/'skin_kinematics.py');law=load('frozen_compliant_law',SKIN/'contact_law.py')
 torch.set_num_threads(1);output=Path(output);artifacts=[];counts={k:0 for k in BUDGET if k!='maximum_seconds_per_case'}
 def count(k):
  if k not in counts or counts[k]>=BUDGET[k]:raise RuntimeError('Before-work static coupling budget exhausted '+k)
  counts[k]+=1;checkpoint(dict(event='operation',name=k,amount=1,counts=dict(counts)))
 def seal(p,label):
  row=dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size,label=label);artifacts.append(row);checkpoint(dict(event='artifact',artifact=row));return row
 def save(name,label,**a):
  p=output/name
  with p.open('xb') as stream:np.savez_compressed(stream,**a)
  return seal(p,label)
 def arrays(row):
  if sha(row['path'])!=row['sha256'] or Path(row['path']).stat().st_size!=row['bytes']:raise ValueError('Sealed input mutated')
  with np.load(row['path'],allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
 b=arrays(spec['literal_buffers']);source=arrays(spec['source_readbacks']);old=arrays(spec['original_compiled_readback'])
 q0=b['source_initialized_q0'];source_q=source['actual_named_qpos'][0];source_V=source['source_vertices_world'][0]
 if q0.shape!=(67,) or source_V.shape!=(778,3) or not np.isfinite(source_V).all() or not np.array_equal(q0,source_q) or b['shape'].dtype!=np.float32 or b['shape'].shape!=(1,10):raise ValueError('Exact new source-initialized q0/shape required')
 selections=kin.fixed_material_points(b['faces'],b['native_face_labels'])
 if selections!=spec['material_points']:raise ValueError('Prospective original anatomical face choices changed')
 graph=kin.CameraGraph(core=core,qpos=q0,rest_quaternions=b['rest_quaternions'],camera_world=b['original_cameraWorld64'],beta32=b['shape'],arrays32={k:b[k] for k in core.ARRAY_KEYS},source_hashes=protocol['profile_source_hashes'],count=count)
 V=graph.baseline().detach().numpy();J=graph.jacobian().detach().numpy();FD=graph.finite_difference().detach().numpy()
 derivative_error=np.abs(J-FD);derivative_pass=bool(np.all(derivative_error<=GATES['derivative_absolute']+GATES['derivative_relative']*np.abs(FD)))
 n64=kin.face_normals(V,b['faces']);n32=kin.face_normals(source_V,b['faces']);vertex_error=float(np.max(np.abs(V-source_V)));normal_error=float(np.max(np.abs(n64-n32)))
 alignment_pass=vertex_error<=GATES['source_alignment_vertex_m'] and normal_error<=GATES['source_alignment_normal_component']
 profile_row=output/'physical_profile.json';profile_row.write_text(json.dumps(dict(profile_payload=graph.profile_payload,profile_hash=graph.profile_hash,state_hash=graph.state_hash,source32_derivative_qualified=False),indent=2,sort_keys=True)+'\n');seal(profile_row,'distinct_camera_graph64_geometry_and_tangent_profile')
 points=[];conditions=[];vertex_forces=np.zeros_like(V);explicit_aggregate=np.zeros(51);gap_checks=[]
 root_body=core.quaternion_product(torch.from_numpy(b['rest_quaternions']),torch.from_numpy(q0[3:].reshape(16,4)));root_R=core.quaternion_matrices(root_body)[0].numpy()
 for selected in selections:
  count('material_points');point=kin.material_point(graph,V,J,selected);points.append(point);n=np.asarray(selected['normal_world']);t=np.asarray(selected['sliding_tangent_world']);offset=float(n@point.point_world_m+kin.PENETRATION_M)
  ids=np.asarray(selected['vertex_ids']);bar=np.asarray(selected['barycentric']);Jfd=np.einsum('i,ijk->jk',bar,FD[ids],optimize=False);gap_ad=n@point.point_J;gap_fd=n@Jfd
  gap_checks.append(bool(np.all(np.abs(gap_ad-gap_fd)<=GATES['derivative_absolute']+GATES['derivative_relative']*np.abs(gap_fd))))
  for mode in ('normal','slide'):
   velocity=np.zeros(51)
   if mode=='slide':velocity[:3]=PARAMETERS['friction_speed_m_per_s']*t
   count('law_evaluations');r=law.evaluate_contact(signed_gap_m=float(n@point.point_world_m-offset),object_outward_normal_world=n.copy(),common_point_world_m=point.point_world_m.copy(),object_wrench_origin_world_m=np.zeros(3),hand_point_jacobian=point.point_J.copy(),object_point_jacobian=np.zeros((3,1)),hand_generalized_velocity=velocity,object_generalized_velocity=np.zeros(1),parameters=law.ContactParameters(**PARAMETERS))
   tau,pv,work=point.load(r['hand_force_world_n'],velocity,profile_hash=graph.profile_hash,state_hash=graph.state_hash)
   if not np.allclose(tau,r['hand_generalized_load'],rtol=0,atol=GATES['virtual_work_absolute_W']):raise ValueError('Kernel applied another point tangent')
   for vi,weight in zip(ids,bar):vertex_forces[vi]+=weight*r['hand_force_world_n']
   explicit_aggregate+=tau
   conditions.append(dict(part=selected['part'],mode=mode,point_index=len(points)-1,velocity=velocity,tau=tau,point_velocity=pv,force=r['hand_force_world_n'],object_force=r['object_force_world_n'],object_wrench=r['object_wrench_force_then_torque'],gap_ad=gap_ad,gap_fd=gap_fd,gap_source32=float(n@np.einsum('i,ij->j',bar,source_V[ids],optimize=False)-offset),gap=float(r['signed_gap_m']),point=point.point_world_m.copy(),work_error=abs(work),law_power_error=abs(r['power_identity_residual_w']),root_wrench_residual_world_Nm=root_R@tau[3:6]-np.cross(point.point_world_m-q0[:3],r['hand_force_world_n'])))
 vjp=graph.force_vjp(vertex_forces).detach().numpy();vjp_error=float(np.max(np.abs(vjp-explicit_aggregate)))
 save('surface_tangent_and_alignment.npz','same_camera_graph_full51AD_FD_VJP_and_source32_alignment',vertices_physical64=V,vertices_source32=source_V,faces=b['faces'],native_face_labels=b['native_face_labels'],jacobian=J,central_difference=FD,aggregate_vertex_forces=vertex_forces,aggregate_explicit_tau=explicit_aggregate,aggregate_vjp_tau=vjp,q0=q0,shape=b['shape'],camera=b['original_cameraWorld64'],rest_quaternions=b['rest_quaternions'],normal_physical64=n64,normal_source32=n32)
 xml,skeleton_hash=scene_xml(spec['base_xml']);xp=output/'native_contact_disabled.xml';xp.write_text(xml);seal(xp,'original_inertials_skeleton_contacts_actuation_off')
 count('model_compilations');model=mujoco.MjModel.from_xml_path(str(xp));layout=named_layout(model,mujoco,spec['saved_named_joint_addresses']);count('data_constructions');data=mujoco.MjData(model)
 if not np.array_equal(model.body_mass,old['body_mass']) or not np.array_equal(model.body_inertia,old['body_inertia']) or np.any(model.geom_contype) or np.any(model.geom_conaffinity):raise ValueError('Original inertials/collision-off model required')
 if int(model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_CONTACT)==0 or int(model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)==0:raise ValueError('Contacts and actuation must be disabled')
 baseline_ctrl=old['retained_ctrl'].copy();state_rows=[];condition_rows=[];checks=[]
 for i,c in enumerate(conditions):
  count('static_load_conditions');data.qpos[:]=q0;data.qvel[:]=c['velocity'];data.ctrl[:]=baseline_ctrl;data.time=0.;data.qfrc_applied[:]=0.;data.xfrc_applied[:]=0.;data.qacc_warmstart[:]=0.
  protected={k:np.asarray(getattr(data,k)).copy() for k in ('qpos','qvel','ctrl','act','mocap_pos','mocap_quat','userdata')};warn=data.warning.number.copy()
  if np.any(warn):raise ValueError('Zero numerical warning baseline required')
  operands=[]
  for label,load_value in (('baseline',np.zeros(51)),('loaded',c['tau'])):
   data.qfrc_applied[:]=load_value;data.qacc_warmstart[:]=0.;count('static_forward_calls');mujoco.mj_forward(model,data)
   if data.time!=0 or any(not np.array_equal(getattr(data,k),v) for k,v in protected.items()) or not np.array_equal(data.qfrc_applied,load_value) or np.any(data.xfrc_applied) or data.ncon or data.nefc or np.any(data.qfrc_constraint) or np.any(data.qfrc_actuator) or not np.array_equal(data.warning.number,warn):raise ValueError('Static paired state/force/warning integrity failed')
   row={k:np.asarray(getattr(data,k)).copy() for k in ('qpos','qvel','ctrl','qacc','qM','qfrc_applied','qfrc_bias','qfrc_passive','qfrc_constraint','qfrc_actuator')}
   if not all(np.isfinite(x).all() for x in row.values()):raise ValueError('Nonfinite native static mass operands')
   operands.append(row)
  a0,aF=operands
  for key in ('qM','qfrc_bias','qfrc_passive','qfrc_constraint','qfrc_actuator'):
   if not np.array_equal(a0[key],aF[key]):raise ValueError('Paired model/bias/passive force changed')
  dense=np.zeros((51,51));count('full_mass_matrix_calls');mujoco.mj_fullM(model,dense,a0['qM'])
  result,residual=mass_response(dense,a0['qacc'],aF['qacc'],c['tau']);power_limit=GATES['virtual_work_absolute_W']+GATES['virtual_work_relative']*abs(float(c['force']@c['point_velocity']))
  reaction=float(np.linalg.norm(c['force']+c['object_force']));moment=float(np.linalg.norm(np.cross(c['point'],c['force'])+c['object_wrench'][3:]))
  passed=result['passed'] and c['work_error']<=power_limit and c['law_power_error']<=power_limit and reaction<=GATES['linear_reaction_N'] and moment<=GATES['common_point_angular_reaction_Nm'];checks.append(passed)
  numeric={k:np.asarray(v) for k,v in c.items() if k not in ('part','mode')}
  numeric.update({prefix+'_'+key:value for prefix,row in (('baseline',a0),('loaded',aF)) for key,value in row.items()});numeric.update(mass_matrix=dense,mass_response_residual=residual,time_s=np.asarray(data.time),warning_counts=data.warning.number.copy())
  save(f'condition_{i:02d}.npz','fixed_arbitrary_knownplane_materialpoint_'+c['part']+'_'+c['mode'],**numeric)
  condition_rows.append(dict(condition=i,part=c['part'],mode=c['mode'],passed=passed,mass_response_passed=result['passed'],work_error_W=c['work_error'],law_power_error_W=c['law_power_error'],linear_reaction_error_N=reaction,common_point_moment_error_Nm=moment,root_wrench_residual_world_Nm=c['root_wrench_residual_world_Nm'].tolist(),root_rigid_wrench_equivalence_claimed=False,**{k:v for k,v in result.items() if k!='passed'}))
 save('original_model_and_named_layout.npz','unaltered_native_mass_inertial_named51_mapping',body_mass=model.body_mass.copy(),body_inertia=model.body_inertia.copy(),body_ipos=model.body_ipos.copy(),body_iquat=model.body_iquat.copy(),body_parentid=model.body_parentid.copy(),joint_type=model.jnt_type.copy(),joint_qpos_address=model.jnt_qposadr.copy(),joint_dof_address=model.jnt_dofadr.copy(),joint_axis=model.jnt_axis.copy(),dof_parentid=model.dof_parentid.copy(),dof_Madr=model.dof_Madr.copy(),geom_contype=model.geom_contype.copy(),geom_conaffinity=model.geom_conaffinity.copy(),disableflags=np.asarray(model.opt.disableflags))
 if counts!={k:v for k,v in BUDGET.items() if k!='maximum_seconds_per_case'}:raise ValueError('Exact static operation inventory incomplete')
 coupling_pass=derivative_pass and all(gap_checks) and vjp_error<=GATES['virtual_work_absolute_W'] and all(checks)
 return dict(status='COMPLETED',counts=counts,forbidden_counts=FORBIDDEN,artifacts=artifacts,static_point_coupling_status='PASS' if coupling_pass else 'FAIL',source32_alignment_status='PASS' if alignment_pass else 'FAIL',profile_hash=graph.profile_hash,state_hash=graph.state_hash,material_points=selections,named_layout=layout,skeleton_inertial_parameter_hash=skeleton_hash,conditions=condition_rows,
  metrics=dict(max_derivative_absolute_error=float(derivative_error.max()),derivative_all_columns_passed=derivative_pass,all_gap_derivatives_passed=all(gap_checks),aggregate_VJP_max_error=vjp_error,source32_vertex_max_component_error_m=vertex_error,source32_normal_max_component_error=normal_error),
  force_fixture_scope='arbitrary known-plane materialpoint loads; can be backface; not original-object orientation/search',source32_derivative_qualified=False,root_rigid_wrench_equivalence_claimed=False,physical_outcome=None,**FLAGS)
