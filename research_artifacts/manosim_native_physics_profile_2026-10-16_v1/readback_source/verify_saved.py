"""Independent stdlib/NumPy saved-only compensated-native profile readback.
No producer/helper, engine, MANO, native FK, collision or physics calls.
Reference and actual clearance are direct arithmetic on sealed operands.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
import argparse,hashlib,importlib.util,json,math,sys,traceback
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_native_profile_arrays',HERE/'array_helpers.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
SCHEMA='manosim_native_compensated_physics_independent_saved_readback_v1'
KINDS=('gravity_motion_hold','two_finger_planes','guarded_sphere_table_approach')
BUDGET=dict(models=12,data=12,static_forwards=240,steps=9750,contact_force_reads=4500,maximum_seconds=600,outer_seconds=610)
GATES=h.LIMITS
FLAGS=h.FLAGS

def plan():return [dict(index=i,kind=k,dt_s=dt,duration_s=T,steps=int(round(T/dt))) for i in (20,28) for dt in (.002,.001) for k,T in zip(KINDS,(1.5,.75,1.))]
def ordinary(a):return np.asarray(a,dtype=np.float64)
def scalar_dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))
def dot(A,b):return np.array([scalar_dot(row,b) for row in A])
def matrix(A,B):return np.array([[scalar_dot(A[i],B[:,j]) for j in range(B.shape[1])] for i in range(A.shape[0])])
def body_J(points,anchors,R,body):
 n=len(points);J=np.zeros((n,3,51));J[:,:,:3]=np.eye(3)
 while body>=0:
  axes=R[:,body].transpose(0,2,1);J[:,:,3+3*body:6+3*body]=np.cross(axes,points[:,None,:]-anchors[:,body,None,:]).transpose(0,2,1);body=h.PARENTS[body]
 return J

def gravity(mass,COM,anchors,R):
 result=np.zeros((len(COM),51))
 for b in range(16):
  J=body_J(COM[:,b],anchors,R,b);result+=9.81*float(mass[b])*J[:,2,:]
 return result

def geom_frames(a,m,contract):
 bodyids=np.asarray(contract['body_ids']);jids=np.asarray(contract['joint_ids'])[[3,*range(4,19)]];gids=[]
 for b in bodyids:
  ids=np.flatnonzero(m['geom_bodyid']==b);h.require(len(ids)==1,'One original collider per named body');gids.append(int(ids[0]))
 positions=[];rotations=[]
 for i,g in enumerate(gids):
  # xanchor = body_origin + Rbody*jnt_pos, including nonzero authored anchors.
  local=m['geom_pos'][g]-m['jnt_pos'][jids[i]];positions.append(a['anchors_start'][:,i]+np.einsum('nij,j->ni',a['body_rotations_start'][:,i],local,optimize=False));rotations.append(np.einsum('nij,jk->nik',a['body_rotations_start'][:,i],h.rotation(m['geom_quat'][g]),optimize=False))
 return np.stack(positions,axis=1),np.stack(rotations,axis=1),np.asarray(gids)

def clearance(centers,R,size,palm,sphere,radius,table):
 # Vectorized over time, all16 anatomical colliders. Convex-palm vertex
 # support and enclosing sphere are conservative;15capsules are analytic.
 n=len(centers);out=np.empty((n,16,2));pc=(palm.min(axis=0)+palm.max(axis=0))/2;pr=float(np.linalg.norm(palm-pc,axis=1).max());pcw=centers[:,0]+np.einsum('nij,j->ni',R[:,0],pc,optimize=False);world=np.einsum('nij,vj->nvi',R[:,0],palm,optimize=False)+centers[:,0,None,:]
 out[:,0,0]=world[:,:,2].min(axis=1)-table;out[:,0,1]=np.linalg.norm(pcw-sphere,axis=1)-pr-radius
 for i in range(1,16):
  axis=R[:,i,:,2];q=np.einsum('ni,ni->n',sphere-centers[:,i],axis,optimize=False);q=np.clip(q,-size[i,1],size[i,1]);closest=centers[:,i]+q[:,None]*axis
  out[:,i,0]=centers[:,i,2]-np.abs(axis[:,2])*size[i,1]-size[i,0]-table;out[:,i,1]=np.linalg.norm(closest-sphere,axis=1)-size[i,0]-radius
 h.require(np.isfinite(out).all(),'Nonfinite analytic separation replay');return out

def refs(f,start,goal,indent):
 N=f['steps'];times=np.arange(N+1)*f['dt_s'];out=np.broadcast_to(goal,(N+1,67)).copy()
 if f['kind']=='two_finger_planes':out[:,:3]+=h.ramp(times,.15)[:,None]*indent
 else:
  scale=h.ramp(times,.5 if f['kind']=='gravity_motion_hold' else 1.);out[:,:3]=(1-scale[:,None])*start[:3]+scale[:,None]*goal[:3]
  for i in range(16):
   sl=slice(3+4*i,7+4*i);a=start[sl];b=goal[sl];relative=h.qmul(a*np.array([1,-1,-1,-1]),b);relative=relative if relative[0]>=0 else -relative;length=np.linalg.norm(relative[1:]);axis=relative[1:]*(2 if length<1e-15 else 2*np.arctan2(length,relative[0])/length)
   for j,t in enumerate(scale):
    angle=t*axis;theta=np.linalg.norm(angle);q=np.r_[np.cos(theta/2),angle*(.5 if theta<1e-15 else np.sin(theta/2)/theta)];out[j,sl]=h.qmul(a,q)
 return out

def snapshot_checks(directory,meta,source,goal):
 st=h.npz(directory/'static_command.npz');early=h.npz(directory/'prevalidation_compiled_model.npz');m=h.npz(directory/'fixture_model.npz');c=meta['compiled'];h.exact(st['qpos'],goal,'Static unchanged source q0');h.near(st['ctrl'],h.controls(goal,c),1e-14,'Static principal command');h.near(st['ctrl'],st['actuator_length'],GATES['transmission_length'],'Static actual transmission');h.near(st['actuator_force'],np.zeros(51),GATES['zero_servo_force'],'Static zero servo force')
 ae=h.near(st['anchors'],source['source_joint_anchors_world'][0],GATES['static_anchor_m'],'Static source anchors');re=h.near(st['rotations'],source['source_body_rotations_world'][0],GATES['static_rotation_component'],'Static sourceR');expected=gravity(m['body_mass'][c['body_ids']],st['body_COM_world'][None],st['anchors'][None],st['rotations'][None])[0]
 h.near(st['actual_gravity_tangent'][:3],expected[:3],1e-8,'Static gravity translation');h.near(st['actual_gravity_tangent'][3:],expected[3:],1e-10,'Static gravity torque');h.near(st['actual_passive_tangent'][:3],st['actual_bias_tangent'][:3],1e-8,'Static passive/bias cancellation');h.near(st['actual_passive_tangent'][3:],st['actual_bias_tangent'][3:],1e-10,'Static angular passive/bias cancellation')
 expected_gc=np.zeros(len(m['body_mass']));expected_gc[c['body_ids']]=1;h.exact(m['body_gravcomp'],expected_gc,'Hand-only gravcomp profile');h.require(not m['jnt_actgravcomp'][c['joint_ids']].any(),'Passive not actuator gravity routing');h.near(m['gravity'],[0,0,-9.81],0,'Fixed gravity')
 h.near(early['model_dimensions'][:4],[67,51,51,19],0,'Standalone51D layout');h.require(int(early['model_dimensions'][4])==17,'No added moving body');h.require(str(early['fixture_xml_sha256'].item())==h.sha(directory/'fixture.xml'),'Early XML identity')
 inertia=h.compiled_snapshot_check(directory/'prevalidation_compiled_model.npz',directory/'fixture.xml',c['inertia_representations'],None)
 root=ET.parse(directory/'fixture.xml').getroot();h.require(all(b.get('gravcomp')=='1' for b in root.findall('./worldbody//body')) and all(j.get('actuatorgravcomp')=='false' for j in root.findall('./worldbody//joint')),'Explicit XML compensation profile')
 return dict(anchor_error_m=ae,rotation_error=re,gravity_translation_error_N=float(np.max(np.abs(expected[:3]-st['actual_gravity_tangent'][:3]))),gravity_torque_error_Nm=float(np.max(np.abs(expected[3:]-st['actual_gravity_tangent'][3:]))),compiled_inertia=inertia),m,c

def plane_preflight(directory,m,c):
 a=h.npz(directory/'plane_preflight.npz');checks={};rows=[];ncon=int(a['pair0_ncon']);nefc=int(a['pair0_nefc']);h.require(0<=ncon<=2,'Tangency contact count');pairs=[]
 for j,b in enumerate((3,6)):
  prefix=f'pair{j}_';g=a[prefix+'actual_geom_ids'];h.require(g.shape==(2,) and int(a[prefix+'actual_geom_bodyid'][0])==c['body_ids'][b] and int(a[prefix+'actual_geom_bodyid'][1])==0,'Plane/body identity');pairs.append(set(map(int,g)));P=a[prefix+'actual_geom_xpos'];R=a[prefix+'actual_geom_xmat'];size=a[prefix+'actual_geom_size'][0];normal=R[1,:,2];gap=float(scalar_dot(normal,P[0]-P[1])-size[0]-size[1]*abs(scalar_dot(normal,R[0,:,2])));pg=int(g[1]);h.near(a[prefix+'actual_geom_pos'],m['geom_pos'][g],0,'Plane preflight model pos');h.near(a[prefix+'actual_geom_quat'],m['geom_quat'][g],0,'Plane preflight model quat')
  checks[f'pair{j}_sameframe_NONE']=bool(a[prefix+'actual_geom_sameframe'][1]==0);checks[f'pair{j}_plane_position']=bool(np.max(np.abs(P[1]-m['geom_pos'][pg]))<=GATES['contact_gap_m']);checks[f'pair{j}_plane_rotation']=bool(np.max(np.abs(R[1]-h.rotation(m['geom_quat'][pg])))<=GATES['contact_normal']);checks[f'pair{j}_analytic_tangent_gap']=abs(gap)<=GATES['contact_gap_m'];rows.append(dict(bone=b,plane_geom_id=pg,gap_m=gap,sameframe=int(a[prefix+'actual_geom_sameframe'][1])))
 seen=set();inventory=True
 for k in range(ncon):
  pair=set(map(int,a['pair0_contact_geom'][k]));matches=[j for j,pair0 in enumerate(pairs) if pair==pair0]
  if len(matches)!=1 or matches[0] in seen:inventory=False;continue
  j=matches[0];seen.add(j);hg=int(a[f'pair{j}_actual_geom_ids'][0]);normal=a[f'pair{j}_actual_geom_xmat'][1,:,2];sign=1 if int(a['pair0_contact_geom'][k,1])==hg else -1;checks[f'contact{j}_dimension']=bool(a['pair0_contact_dim'][k]==3);checks[f'contact{j}_engine_tangent_gap']=abs(float(a['pair0_contact_dist'][k]))<=GATES['contact_gap_m'];checks[f'contact{j}_normal']=bool(np.max(np.abs(a['pair0_contact_frame'][k].reshape(3,3)[0]-sign*normal))<=GATES['contact_normal'])
 checks['at_most_two_unique_declared_pairs']=inventory;checks['zero_contact_no_constraint_rows']=bool(ncon!=0 or (nefc==0 and not a['pair0_qfrc_constraint'].any()));h.require(all(checks.values()),'Static2plane preflight failed');return checks,rows

def contact_check(a,m,c):
 N=len(a['pre_time_s']);centers,R,gids=geom_frames(a,m,c);Jsum=np.zeros((N,51));Jerrors=np.zeros((N,2));gaperrors=np.zeros((N,2));nerrors=np.zeros((N,2));active=a['contact_active'].astype(bool);primal=np.zeros((N,2));dual=primal.copy();complement=primal.copy();fs=np.ones((N,2));gs=fs.copy();penetration=[]
 for j,b in enumerate((3,6)):
  hg=int(gids[b]);possible=np.flatnonzero((m['geom_type']==0)&(m['geom_bodyid']==0)&(m['geom_contype']==(1<<j)));h.require(len(possible)==1,'Declared unique paired world plane');pg=int(possible[0]);planeR=h.rotation(m['geom_quat'][pg]);n=planeR[:,2];delta=centers[:,b]-m['geom_pos'][pg];gap=np.einsum('ni,i->n',delta,n,optimize=False)-m['geom_size'][hg,0]-m['geom_size'][hg,1]*np.abs(np.einsum('ni,i->n',R[:,b,:,2],n,optimize=False));h.require(gap.min()>=-.005,'All-pair analytic5mm guard escaped');penetration.append(float(gap.min()));present=a['contact_geom_ids'][:,j,0]>=0
  for k in np.flatnonzero(present):h.require(set(map(int,a['contact_geom_ids'][k,j]))=={hg,pg},'Contact pair differs')
  h.require(np.all(~active[:,j]|present),'Active without present contact');idx=np.flatnonzero(present);sign=np.where(a['contact_geom_ids'][idx,j,1]==hg,1.,-1.);h.near(a['capsule_center'][idx,j],centers[idx,b],1e-12,'Contact capsule center from saved actual skeleton');h.near(a['capsule_R'][idx,j],R[idx,b],1e-12,'Contact capsuleR from saved skeleton');h.near(a['plane_position'][idx,j],np.broadcast_to(m['geom_pos'][pg],(len(idx),3)),1e-14,'Actual plane pos');h.near(a['plane_R'][idx,j],np.broadcast_to(planeR,(len(idx),3,3)),1e-14,'Actual planeR');gaperrors[idx,j]=np.abs(gap[idx]-a['engine_gap_m'][idx,j]);nerrors[idx,j]=np.max(np.abs(a['contact_frame'][idx,j,0]-sign[:,None]*n),axis=1);h.require(np.min(a['engine_gap_m'][idx,j],initial=0)>=-.005,'Engine5mm guard escaped')
  idx=np.flatnonzero(active[:,j]);F=a['contact_force'][idx,j];basis=a['contact_frame'][idx,j];sign=np.where(a['contact_geom_ids'][idx,j,1]==hg,1.,-1.);J=sign[:,None,None]*np.einsum('nki,nij->nkj',basis,body_J(a['contact_point'][idx,j],a['anchors_start'][idx],a['body_rotations_start'][idx],b),optimize=False);h.near(a['independent_contact_J'][idx,j],J,1e-12,'Saved independent J');Jerrors[idx,j]=np.max(np.abs(a['contact_J'][idx,j]-J),axis=(1,2));Jsum[idx]+=np.einsum('nki,nk->ni',J,F,optimize=False)
  gradient=np.einsum('nki,ni->nk',a['contact_J'][idx,j],a['start_qacc'][idx],optimize=False)+a['efc_R'][idx,j]*F-a['efc_aref'][idx,j];h.near(a['contact_gradient'][idx,j],gradient,1e-12,'KKT gradient');primal[idx,j]=np.maximum.reduce([np.zeros(len(idx)),-F[:,0],np.linalg.norm(F[:,1:],axis=1)-F[:,0]]);dual[idx,j]=np.maximum(0,np.linalg.norm(gradient[:,1:],axis=1)-gradient[:,0]);complement[idx,j]=np.abs(np.sum(F*gradient,axis=1));fs[idx,j]=1+np.linalg.norm(F,axis=1);gs[idx,j]=1+np.linalg.norm(gradient,axis=1)
 residual=a['start_qfrc_constraint']-Jsum;translation=np.max(np.abs(residual[:,:3]),axis=1);angular=np.max(np.abs(residual[:,3:]),axis=1);actual=np.einsum('ni,ni->n',a['start_qfrc_constraint'],a['pre_qvel'],optimize=False);expected=np.einsum('ni,ni->n',Jsum,a['pre_qvel'],optimize=False);probe=np.sin(np.arange(51));power=np.maximum(np.abs(actual-expected),np.abs(np.einsum('ni,i->n',residual,probe,optimize=False)));scale=np.maximum(np.abs(actual),np.abs(expected))
 values=dict(contact_J_error=Jerrors,contact_gap_error_m=gaperrors,contact_normal_error=nerrors,generalized_translation_force_error_N=translation,generalized_angular_force_error_Nm=angular,virtual_power_error_W=power,virtual_power_scale_W=scale,cone_primal=primal,cone_dual=dual,cone_complementarity=complement,cone_force_scale=fs,cone_gradient_scale=gs)
 for k,value in values.items():h.near(a[k],value,1e-12,'Contact scalar replay '+k)
 h.near(a['normal_force_N'],a['contact_force'][:,:,0],1e-12,'Normal force channel');return values,dict(active_contact_reads=int(active.sum()),minimum_analytic_gap_m=min(penetration),max_J_error=float(Jerrors.max()),max_translation_reaction_error_N=float(translation.max()),max_angular_reaction_error_Nm=float(angular.max()),max_virtual_power_error_W=float(power.max()),all_tail_force_scope='Every tail sample on both fingers must be active and >fixed1e-8N, not an average.',J_force_checks_applicability='ESTABLISHED_ON_RECORDED_ACTIVE_ROWS' if active.any() else 'NOT_ESTABLISHED_ZERO_ACTIVE_CONTACTS',normal_gap_comparison_applicability='ESTABLISHED_ON_RECORDED_PRESENT_ROWS' if np.any(a['contact_geom_ids'][:,:,0]>=0) else 'NOT_ESTABLISHED_ZERO_PRESENT_CONTACTS')

def approach_check(directory,a,m,c):
 path=h.npz(directory/'approach_path.npz');d=clearance(path['collider_centers_world'],path['collider_rotations_world'],path['collider_sizes_m'],path['palm_vertices_local'],path['sphere_center'],float(path['sphere_radius']),float(path['table_z']));h.near(path['distances_m'],d,1e-12,'All51knots16colliders2obstacles distance lower bounds');h.require(d.shape==(51,16,2),'Exact reference path coverage');off=path['root_offsets_m'];h.near(off,(1-np.linspace(0,1,51))[:,None]*np.array([-.06,0,0]),1e-14,'Fixed rigid root offsets');travel=np.linalg.norm(np.diff(off,axis=0),axis=1)/2;lower=np.minimum(d[:-1],d[1:])-travel[:,None,None]-1e-9;passed=bool(np.all(lower>=.002))
 C,R,gids=geom_frames(a,m,c);sizes=m['geom_size'][gids];h.near(sizes,path['collider_sizes_m'],0,'Actual/reference collider sizes');h.require(int(m['geom_dataid'][gids[0]])==0,'Original first-asset palm mesh identity');h.near(path['palm_vertices_local'],m['mesh_vert'][:len(path['palm_vertices_local'])],0,'Saved compiled first-palm vertices');actual=clearance(C,R,sizes,path['palm_vertices_local'],path['sphere_center'],float(path['sphere_radius']),float(path['table_z'])).min(axis=(1,2))-1e-9;h.near(a['actual_clearance_lower_m'],actual,1e-12,'Actual per-step analytic clearance');h.require(not a['start_qfrc_constraint'].any(),'Approach no-contact constraint forces')
 return dict(passed=passed,minimum_segment_clearance_lower_m=float(lower.min()),required_clearance_m=.002,oracle_error_m=1e-9,oracle_id='analytic_plane_sphere_hand_bounds_v1',knot_count=51,collider_count=16,obstacle_count=2,rotation_and_finger_motion_allowed=False,object_query_fitting=False,sampled_only=False,proof_scope='1-Lipschitz signed separation under rigid root translation and fixed obstacles; only stated analytic oracle'),actual,dict(reference_minimum_lower_m=float(lower.min()),actual_sampled_minimum_lower_m=float(actual.min()),reference_continuous_translation_certificate=True,actual_servo_continuous_certificate=False,palm_geometry_scope='Sealed actual first-mesh vertex subset and compiler source identity; no new engine mesh query.')

def completed_fixture(f,directory,measurement,source,goal):
 meta=h.read(directory/'command_contract.json');static,m,c=snapshot_checks(directory,meta,source,goal);a=h.npz(directory/'trajectory.npz');N=f['steps'];tail=a['endpoint_time_s']>=f['duration_s']-.1-1e-12
 h.require(a['pre_qpos'].shape==a['endpoint_qpos'].shape==(N,67) and a['pre_qvel'].shape==a['endpoint_qvel'].shape==(N,51),'Recorded trajectory shape');h.near(a['pre_time_s'],np.arange(N)*f['dt_s'],1e-12,'Pre clocks');h.near(a['endpoint_time_s'],np.arange(1,N+1)*f['dt_s'],1e-12,'Endpoint clocks');h.exact(a['pre_qpos'][1:],a['endpoint_qpos'][:-1],'qpos temporal alignment');h.exact(a['pre_qvel'][1:],a['endpoint_qvel'][:-1],'qvel temporal alignment');h.exact(a['pre_qpos'][0],m['initial_state'],'Exact initial state');h.exact(m['native_target'],goal,'Unchanged target');h.principal(a['pre_qpos']);h.principal(a['endpoint_qpos']);h.require(not a['warning_counts'].any(),'Nonzero warning counts');reference=refs(f,m['initial_state'],goal,m['indentation_world_m']);h.near(a['reference_qpos'],reference[:-1],1e-14,'Exact declared reference trajectory');h.near(a['requested_ctrl'],h.controls(a['reference_qpos'],c),1e-14,'All actual named principal controls');h.exact(a['requested_ctrl'],a['effective_ctrl'],'Command clipping/mutation');h.near(a['bone_angle_error_rad'],h.angles(a['body_rotations_start'],source['source_body_rotations_world'][0]),1e-14,'All16 tracking angles');wrist=np.linalg.norm(a['endpoint_qpos'][:,:3]-goal[:3],axis=1);h.near(a['wrist_goal_error_m'],wrist,1e-14,'Wrist to unchanged goal')
 expected=gravity(m['body_mass'][c['body_ids']],a['start_body_COM_world'],a['anchors_start'],a['body_rotations_start']);h.near(a['start_expected_gravcomp'],expected,1e-12,'Sealed expected gravity mapping');delta=np.abs(a['start_qfrc_gravcomp']-expected);h.near(a['gravity_translation_error_N'],delta[:,:3].max(axis=1),1e-12,'Gravity force scalar');h.near(a['gravity_angular_error_Nm'],delta[:,3:].max(axis=1),1e-12,'Gravity torque scalar')
 metrics=dict(wrist_tail_max_m=float(wrist[tail].max()),bone_tail_max_rad=float(a['bone_angle_error_rad'][tail].max()));checks=dict(all16_bone_tracking=metrics['bone_tail_max_rad']<=GATES['tracking_bone_rad'],wrist_to_unchanged_goal=metrics['wrist_tail_max_m']<=GATES['tracking_wrist_m'],independent_gravity_force_map=bool(np.all(delta[:,:3]<=1e-8) and np.all(delta[:,3:]<=1e-10)));contact=None;approach=None;reads=0
 if f['kind']=='two_finger_planes':
  planechecks,planes=plane_preflight(directory,m,c);h.require(meta['plane_preflight']==planechecks,'Plane preflight booleans differ');v,contact=contact_check(a,m,c);reads=contact['active_contact_reads'];checks['two_simultaneous_positive_tail_reactions']=bool(np.all(a['contact_active'][tail]) and np.all(a['normal_force_N'][tail]>GATES['positive_contact_force_N']))
  for field,key in [('contact_J_error','contact_J'),('contact_gap_error_m','contact_gap_m'),('contact_normal_error','contact_normal'),('generalized_translation_force_error_N','contact_translation_force_N'),('generalized_angular_force_error_Nm','contact_angular_force_Nm')]:checks[field]=bool(np.max(v[field])<=GATES[key])
  checks['summed_virtual_power']=bool(np.all(v['virtual_power_error_W']<=1e-10+1e-10*v['virtual_power_scale_W']))
  for field,key,scale in [('cone_primal','cone_primal_relative',v['cone_force_scale']),('cone_dual','cone_dual_relative',v['cone_gradient_scale']),('cone_complementarity','cone_complementarity_relative',v['cone_force_scale']*v['cone_gradient_scale'])]:checks[field]=bool(np.all(v[field]<=GATES[key]*scale))
  contact.update(plane_preflight=planes,tail_minimum_force_each_N=a['normal_force_N'][tail].min(axis=0).tolist())
 elif f['kind']=='guarded_sphere_table_approach':
  guard,actual,approach=approach_check(directory,a,m,c);h.scalar_equal(meta['approach_guard'],guard,'Reference path guard');checks['predeclared_swept_analytic_guard']=guard['passed'];checks['actual_trajectory_minimum_clearance']=bool(np.all(actual>=.002))
 else:h.require(not a['start_qfrc_constraint'].any(),'Free-motion unexpected constraint force')
 assessment=measurement['assessment'];h.require(assessment['checks']==checks and assessment['status']==('PASS' if all(checks.values()) else 'FAIL'),'Independent assessment differs');h.scalar_equal(assessment['metrics'],metrics,'Tail metrics');h.closed(assessment);h.closed(measurement);h.require(measurement['command_clipping'] is False,'Command clipping');h.exact(np.asarray(measurement['final_qpos']),a['endpoint_qpos'][-1],'Final endpoint qpos');h.require(meta['static_checks']==dict(qpos=True,anchors=True,rotations=True,transmission=True,zero_servo_force=True),'Static mapping status differs')
 residual=a['start_qfrc_bias']-a['start_qfrc_gravcomp'];return dict(fixture=f,status=assessment['status'],checks=checks,metrics=metrics,static=static,contact=contact,approach=approach,trajectory_rows=N,contact_reads=reads,moving_bias_minus_gravity_max_abs=float(np.abs(residual[np.any(a['pre_qvel']!=0,axis=1)]).max(initial=0.)),moving_coriolis_required_cancelled=False,maximum_gravity_translation_error_N=float(delta[:,:3].max()),maximum_gravity_torque_error_Nm=float(delta[:,3:].max()),final_qpos=measurement['final_qpos'],final_rotations=measurement['final_rotations'])

def partial_description(path):
 with np.load(path,allow_pickle=False) as f:
  arrays={k:dict(shape=list(f[k].shape),dtype=str(f[k].dtype),nonfinite_values=int(np.count_nonzero(~np.isfinite(f[k]))) if f[k].dtype.kind in 'fci' else None) for k in f.files}
 return dict(artifact=h.bind(path),arrays=arrays,not_physical_qualification=True)

def verify(protocol,freeze,report_sha):
 protocol=Path(protocol).resolve();root=protocol.parents[2];p=h.read(protocol);h.require(p['schema_version']=='manosim_native_compensated_physics_v1','Unexpected profile');h.closed(p);h.require(p['method']['budget']==BUDGET and p['method']['plan']==plan() and p['method']['limits']==GATES,'Prospective plan/gates differ');before=h.closure(p,protocol,root,freeze);out=Path(p['output_root']);raw=Path(p['raw_root']);rp=out/'report.json';h.require(h.sha(rp)==report_sha,'Explicit expected report differs');initial_report=h.bind(rp);r=h.read(rp);h.closed(r);h.require(r['freeze_commit']==freeze and r['protocol_sha256']==h.sha(protocol),'Report freeze/identity');started=h.read(out/'STARTED.json');h.require(started['freeze_commit']==freeze and started['protocol_sha256']==h.sha(protocol) and started['budget']==BUDGET,'Started identity');worker=None;details=[];snapshots=[];static_rows=[];artifacts=[];partial=[]
 for key in ('worker_result','raw_result'):
  if r.get(key):h.rowcheck(r[key])
 if r.get('worker_result'):
  h.require(Path(r['worker_result']['path']).read_bytes()==Path(r['raw_result']['path']).read_bytes(),'Worker raw/tracked mismatch');worker=h.read(r['worker_result']['path']);h.closed(worker);h.require(worker['schema_version']=='manosim_native_compensated_physics_v1_worker' and worker['freeze_commit']==freeze and worker['protocol_sha256']==h.sha(protocol),'Worker identity');h.require(worker['cases']==r['cases'] and worker['fixture_schedule']==r['fixture_schedule'] and worker['actual_counts']==r['actual_counts'],'Worker/report content mismatch');h.require([x['fixture'] for x in worker['fixture_schedule']]==plan(),'Exact12fixture schedule');counts=worker['actual_counts'];h.require(all(isinstance(v,int) and 0<=v<=BUDGET[k] for k,v in counts.items()),'Operation budget escaped');h.require(all(sum(row[k] for row in worker['per_case_counts'].values())==v for k,v in counts.items()),'Per-case/global count mismatch');artifacts=worker['artifacts'];paths={}
  for row in artifacts:
   h.rowcheck(row);path=Path(row['path']).resolve();h.require(path.is_relative_to(raw) and path not in paths and not path.is_symlink(),'Artifact containment/uniqueness');paths[path]=row
   if path.name=='fixture.xml':h.require(p['fixture_xml_sha256'][str(path.parent.relative_to(raw))]==h.sha(path),'Unplanned fixture XML')
  extras={x.resolve() for x in raw.rglob('*') if x.is_file()}-set(paths)-{raw/'progress.json',raw/'result.json'};h.require(not any(x.is_symlink() for x in raw.rglob('*')),'Output symlink');partial=[h.bind(q) for q in sorted(extras)]
  for path in paths:
   if path.name=='prevalidation_compiled_model.npz':
    try:snapshots.append(h.compiled_snapshot_check(path,path.with_name('fixture.xml'),None,None))
    except ValueError as exc:
     if r['status']=='COMPLETED':raise
     snapshots.append(dict(snapshot=h.bind(path),all_output_postconditions_passed=False,error=str(exc),scope='Preserved failed static validation; no physical qualification'))
   if path.name=='command_contract.json':
    index=int(path.relative_to(raw).parts[0]);spec=p['cases'][str(index)];source=h.npz(spec['source_readbacks']['path']);b=h.npz(spec['literal_buffers']['path']);meta=h.read(path);static,_,_=snapshot_checks(path.parent,meta,source,b['source_initialized_q0']);static_rows.append(dict(fixture_path=str(path.parent.relative_to(raw)),readback=static))
   if path.name.startswith('partial_') or path.name=='failed_contact_operands.npz':partial.append(partial_description(path))
  for case in worker['cases']:
   h.closed(case);index=case['index'];spec=p['cases'][str(index)];source=h.npz(spec['source_readbacks']['path']);b=h.npz(spec['literal_buffers']['path']);goal=b['source_initialized_q0'];h.exact(goal,source['actual_named_qpos'][0],'Unchanged baseline q0');rows=[]
   for measurement in case.get('measurements',[]):
    f=measurement['fixture'];h.require(f in plan() and f['index']==index,'Unplanned completed fixture');directory=raw/str(index)/(f['kind']+'_'+str(round(1/f['dt_s'])));rows.append(completed_fixture(f,directory,measurement,source,goal))
   convergence=[]
   if case['status']=='COMPLETED':
    h.require(len(rows)==6,'Completed6fixture denominator')
    for kind in KINDS:
     pair=[x for x in rows if x['fixture']['kind']==kind];dx=float(np.linalg.norm(np.asarray(pair[0]['final_qpos'])[:3]-np.asarray(pair[1]['final_qpos'])[:3]));dr=float(h.angles(pair[0]['final_rotations'],pair[1]['final_rotations']).max());convergence.append(dict(kind=kind,wrist_difference_m=dx,bone_difference_rad=dr,passed=dx<=.001 and dr<=np.deg2rad(1)))
    for a,b0 in zip(case['convergence'],convergence):h.scalar_equal(a,b0,'Convergence')
    qualification='PASS' if all(x['status']=='PASS' for x in rows) and all(x['passed'] for x in convergence) else 'NOT_QUALIFIED';h.require(case['qualification']==qualification,'Case qualification differs')
   else:qualification='NOT_QUALIFIED'
   details.append(dict(index=index,producer_status=case['status'],qualification=qualification,fixtures=rows,convergence=convergence,operation_counts=worker['per_case_counts'][str(index)]))
  if r['status']=='COMPLETED':
   h.require(len(artifacts)==80 and len(snapshots)==12 and len(static_rows)==12 and len(details)==2 and not extras,'Exact completed80artifact/12static inventory');h.require(worker['all12_static_checks_before_dynamics'] is True and not worker['forbidden_attempts'] and worker['status']=='COMPLETED' and worker['failure'] is None and r['failure'] is None and r['returncode']==0,'Completed worker transaction');h.require(all(counts[k]==BUDGET[k] for k in ('models','data','static_forwards','steps')),'Complete operation counts');h.require(sum(x['contact_reads'] for c in details for x in c['fixtures'])==counts['contact_force_reads'],'Actual contact-force read denominator');h.require(sum(x['trajectory_rows'] for c in details for x in c['fixtures'])==9750,'All9750recorded intervals');static_positions=[i for i,a in enumerate(artifacts) if Path(a['path']).name=='command_contract.json'];trajectory_positions=[i for i,a in enumerate(artifacts) if Path(a['path']).name=='trajectory.npz'];h.require(max(static_positions)<min(trajectory_positions),'All static artifacts must be published before any trajectories');h.require(all(x['status']=='COMPLETED_ASSESSED' for x in worker['fixture_schedule']),'Completed schedule');h.require(r['qualification']==('PASS' if all(c['qualification']=='PASS' for c in details) else 'NOT_QUALIFIED'),'Top qualification')
  else:h.require(r['status']=='HARD_STOP' and r['qualification']=='NOT_QUALIFIED','Hard stop remains unqualified')
 for a in artifacts:h.rowcheck(a)
 for a in r.get('partial_files',[]):h.rowcheck(a)
 for k in ('worker_result','raw_result'):
  if r.get(k):h.rowcheck(r[k])
 h.require(h.bind(rp)==initial_report,'Report changed during replay');after=h.closure(p,protocol,root,freeze);h.require(before==after,'Closure changed')
 return dict(schema_version=SCHEMA,status='VERIFIED_COMPLETED_BOUNDED_NATIVE_PROFILE_SAVED_READBACK' if r['status']=='COMPLETED' else 'VERIFIED_HARD_STOP_NATIVE_PROFILE_SAVED_READBACK',experiment_status=r['status'],reported_bounded_qualification=r['qualification'],protocol=h.bind(protocol),report=h.bind(rp),freeze_commit=freeze,closure_before=before,closure_after=after,actual_operation_counts=r['actual_counts'],fixture_schedule=r['fixture_schedule'],cases=details,static_preflight_readbacks=static_rows,compiled_snapshots=snapshots,artifact_count=len(artifacts),partial_artifacts=partial,worker_failure=worker.get('failure') if worker else None,parent_failure=r['failure'],physical_outcome=None,**{k:False for k in FLAGS},operations=dict(engine_calls=0,MANO_or_skin_calls=0,native_FK_or_geometry_API_calls=0,physics_steps=0),limits=['Saved-array arithmetic verifies the bounded record, not physical reality or task-grasp success.','Static-before-first-step228forwards is a frozen Engine gate assertion cross-checked against12preflights, ordered artifacts and recorded clocks; no engine rerun.','Reference rigid-translation certificate is continuous; servo-deviating actual trajectory clearance is only sampled once per step.','Contact friction/compliance checks are frozen producer assertions and compiled-model parameters; per-contact parameter vectors are not present in every trajectory row.','No full-skin articulation/forces parity, human-oracle quality, new task collection or training admission.'])

def numpy_scalar(value):
 if isinstance(value,np.generic):return value.item()
 raise TypeError('Unsupported receipt object '+type(value).__name__)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--freeze-commit',required=True);ap.add_argument('--expected-report-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();h.require(not a.output.exists(),'Write-once independent receipt');error=None
 try:r=verify(a.protocol,a.freeze_commit,a.expected_report_sha256)
 except BaseException as exc:error=exc;r=dict(schema_version=SCHEMA,status='VERIFICATION_HARD_STOP',failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),physical_outcome=None,**{k:False for k in FLAGS})
 r['verifier_sources']=[dict(**h.bind(path),source=path.read_text()) for path in (Path(__file__),HERE/'array_helpers.py')];a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(r,f,sort_keys=True,indent=2,allow_nan=False,default=numpy_scalar);f.write('\n')
 print(json.dumps(dict(status=r['status'],output=str(a.output),sha256=h.sha(a.output))));return 1 if error else 0
if __name__=='__main__':raise SystemExit(main())
