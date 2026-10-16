"""Independent saved-only authored-capsule execution readback.

stdlib + NumPy only. Never imports producer modules, engine, MANO, FK or geometry
kernels. Reads sealed outputs and replays coordinate/algebra comparisons. A PASS
here means consistency of the recorded bounded experiment, not physical/task or
mapping admission. Missing operands are explicitly reported, never fabricated.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,traceback
from xml.etree import ElementTree as ET
import numpy as np
SCHEMA='manosim_capsule_execution_independent_saved_readback_v2'
STUDY_SCHEMA='manosim_authored_capsule_execution_v2'
PRIOR_FREEZE='a4b447e1c0467bf877798dd64b090e8ad11af301'
PRIOR_PROTOCOL_SHA='0b5c48a7a9e72e66e2075651b19553a49ca2ec8824a28b1049ed74d16d8ef027'
PRIOR_REPORT_SHA='62fd0a4fc41e5430b5e5e981a72cb856ae8f9b57b7f66e36f325a4720b18de9b'
PRIOR_RECEIPT_SHA='da0dc34c722d960adbcdbdbdad60fdb12b58cd8cc88780ac98580df59b05c51a'
PARENTS=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
KINDS=('zero_gravity_tracking','authored_gravity_equilibrium','isolated_capsule_plane')
FLAGS=('accepted_mapping','native_task_contact_qualified','full_skin_parity_claimed','new_collection_or_training_allowed')
LIMITS=dict(static_anchor_m=5e-6,static_rotation_component=1e-6,transmission_length=1e-10,zero_servo_force=1e-8,tracking_wrist_m=.002,tracking_bone_rad=float(np.deg2rad(2)),gravity_balance_fraction=.01,convergence_wrist_m=.001,convergence_bone_rad=float(np.deg2rad(1)),contact_gap_m=1e-9,contact_normal=1e-10,contact_J=1e-10,virtual_power_absolute_W=1e-10,virtual_power_relative=1e-10,contact_translation_force_N=1e-8,contact_angular_force_Nm=1e-10,cone_primal_relative=1e-8,cone_dual_relative=1e-7,cone_complementarity_relative=1e-7,positive_contact_force_N=1e-8,maximum_penetration_m=.005,clock_s=1e-12,raw_ball_norm=1e-8)

def require(x,message):
 if not bool(x):raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
 def pairs(rows):
  out={}
  for k,v in rows:
   require(k not in out,'Duplicate JSON key');out[k]=v
  return out
 def reject(value):raise ValueError('Nonfinite JSON constant '+value)
 def check(x):
  if isinstance(x,float):require(np.isfinite(x),'Nonfinite JSON exponent')
  elif isinstance(x,dict):
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
 out=json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=reject);check(out);return out
def bind(path):
 p=Path(path).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def rowcheck(row):
 p=Path(row['path']);require(p.is_absolute() and not p.is_symlink(),'Absolute nonsymlink artifact required');require(bind(p)=={k:row[k] for k in ('path','sha256','bytes')},'Artifact hash/bytes changed: '+str(p))
def npz(path):
 with np.load(path,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
 for k,v in a.items():require(v.dtype.kind in 'biufSU','Object/unsupported sealed operand '+k);require(v.dtype.kind in 'SU' or np.isfinite(v).all(),'Nonfinite sealed operand '+k)
 return a
def near(a,b,tol,message):
 a=np.asarray(a);b=np.asarray(b);require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),message+' shape/finiteness');d=float(np.max(np.abs(a-b))) if a.size else 0.;require(d<=tol,message+': '+str(d));return d
def exact(a,b,message):
 a=np.asarray(a);b=np.asarray(b);require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),message)
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])
def closed(value):
 require(value.get('physical_outcome') is None,'Physical outcome must remain null')
 for key in FLAGS:require(value.get(key) is False,'Admission flag changed: '+key)
def closure(p,protocol,root,freeze):
 require(git(root,'rev-parse','HEAD').decode().strip()==freeze,'Exact HEAD hold required')
 for path,digest in p['active_bindings'].items():require(sha(path)==digest,'Active binding changed '+path)
 for relative,digest in p['selected_git_sources'].items():require(hashlib.sha256(git(root,'show',freeze+':'+relative)).hexdigest()==digest,'Frozen source blob changed '+relative)
 require(git(root,'show',freeze+':'+str(protocol.relative_to(root)))==protocol.read_bytes(),'Protocol Git blob differs')
 return dict(active_bindings=len(p['active_bindings']),selected_git_sources=len(p['selected_git_sources']),protocol_blob_verified=True,exact_HEAD=freeze)
def plan():return [dict(index=i,kind=k,dt_s=dt,duration_s=T,steps=round(T/dt)) for i in (20,28) for dt in (.002,.001) for k,T in zip(KINDS,(1.5,1.5,.5))]
def principal(q):
 q=np.asarray(q,float);require(q.shape[-1]==67,'Named67 qpos required');balls=q[...,3:].reshape(q.shape[:-1]+(16,4));norm=np.sqrt(np.einsum('...ij,...ij->...i',balls,balls,optimize=False));require(np.max(np.abs(norm-1))<=1e-8,'Raw ball norm escaped');balls=balls/norm[...,None];s=np.sqrt(np.einsum('...ij,...ij->...i',balls[...,1:],balls[...,1:],optimize=False));nz=s>=1e-15;axis=np.zeros_like(balls[...,1:]);np.divide(balls[...,1:],s[...,None],out=axis,where=nz[...,None]);angle=2*np.arctan2(np.where(nz,s,0),balls[...,0]);angle=np.where(angle>np.pi,angle-2*np.pi,angle);require(np.all(np.pi-np.abs(angle)>1e-6),'Principal command chart crossed exclusion');return axis*angle[...,None]
def controls(q,contract):
 q=np.asarray(q,float);rv=principal(q);gear=np.asarray(contract['gears']);ids=np.asarray(contract['actuator_ids']);require(gear.shape==(51,6) and set(ids)==set(range(51)),'Named actuator contract malformed');canonical=np.empty(q.shape[:-1]+(51,));canonical[...,:3]=q[...,:3]*gear[:3,0];canonical[...,3:]=np.sum(np.repeat(rv,3,axis=-2)*gear[3:,:3],axis=-1);out=np.empty_like(canonical);out[...,ids]=canonical;return out

def angles(actual,target):
 relative=np.einsum('...bji,...bjk->...bik',np.asarray(target),np.asarray(actual),optimize=False);s=np.linalg.norm(np.stack((relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],relative[...,1,0]-relative[...,0,1]),axis=-1),axis=-1)/2;c=(np.trace(relative,axis1=-2,axis2=-1)-1)/2;return np.arctan2(s,np.clip(c,-1,1))
def qmul(a,b):return np.r_[a[0]*b[0]-a[1:]@b[1:],a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]
def ramp(t,T):
 x=np.clip(np.asarray(t)/T,0,1);return x*x*x*(10+x*(-15+6*x))
def reference(fixture,start,goal,normal):
 times=np.arange(fixture['steps']+1)*fixture['dt_s'];out=np.broadcast_to(goal,(len(times),67)).copy()
 if fixture['kind']=='isolated_capsule_plane':out[:,:3]-=.0005*ramp(times,.1)[:,None]*normal
 elif fixture['kind']=='zero_gravity_tracking':
  s=ramp(times,.5);out[:,:3]=(1-s[:,None])*start[:3]+s[:,None]*goal[:3]
  for j in range(16):
   sl=slice(3+4*j,7+4*j);a=start[sl];b=goal[sl];rel=qmul(a*np.array([1,-1,-1,-1]),b);rel=rel if rel[0]>=0 else -rel;v=np.linalg.norm(rel[1:]);log=rel[1:]*(2 if v<1e-15 else 2*np.arctan2(v,rel[0])/v)
   for n,f in enumerate(s):
    r=f*log;theta=np.linalg.norm(r);e=np.r_[np.cos(theta/2),r*(.5 if theta<1e-15 else np.sin(theta/2)/theta)];out[n,sl]=qmul(a,e)
 return out

def point_jac(points,anchors,R):
 n=len(points);J=np.zeros((n,3,51));J[:,:,:3]=np.eye(3)
 for bid in (3,2,1,0):
  # local body angular velocities; columns are world-space owner axes.
  axis=R[:,bid].transpose(0,2,1);J[:,:,3+3*bid:6+3*bid]=np.cross(axis,points[:,None,:]-anchors[:,bid,None,:]).transpose(0,2,1)
 return J

def scalar_equal(actual,expected,label):
 require(set(actual)==set(expected),label+' keys differ')
 for k,v in expected.items():
  if isinstance(v,bool):require(actual[k] is v,label+' decision differs '+k)
  elif isinstance(v,list):near(actual[k],v,1e-12,label+' '+k)
  elif isinstance(v,(float,int)):require(abs(actual[k]-v)<=1e-12*max(1,abs(v)),label+' scalar differs '+k)
  else:require(actual[k]==v,label+' differs '+k)

def fixture_check(f,base,measurement,source,goal,artifact_paths,snapshot_context=None):
 key=f['kind']+'_'+str(round(1/f['dt_s']));directory=base/key
 for name in ('fixture.xml','prevalidation_compiled_model.npz','static_command.npz','fixture_model.npz','command_contract.json','trajectory.npz'):require(directory/name in artifact_paths,'Missing indexed fixture artifact '+name)
 x=ET.parse(directory/'fixture.xml').getroot();require(sha(directory/'fixture.xml')==artifact_paths[directory/'fixture.xml']['sha256'],'Fixture XML bytes changed');require(x.find('option').get('timestep')==str(f['dt_s']),'XML timestep differs');require(x.find('option').get('integrator')=='implicitfast','Unexpected integrator')
 st=npz(directory/'static_command.npz');m=npz(directory/'fixture_model.npz');a=npz(directory/'trajectory.npz');meta=read(directory/'command_contract.json');contract=meta['compiled'];early=npz(directory/'prevalidation_compiled_model.npz');require(contract['schema_version']=='manosim_authored_capsule_compiled_contract_v2','Compiled v2 metadata required');
 for field,array in [('body_ids','named_body_ids'),('joint_ids','named_joint_ids'),('actuator_ids','named_actuator_ids')]:near(contract[field],early[array],0,'Early/contract named IDs '+field)
 near(contract['qpos_addresses'],early['jnt_qposadr'][early['named_joint_ids']],0,'Compiled qpos addresses');near(contract['dof_addresses'],early['jnt_dofadr'][early['named_joint_ids']],0,'Compiled dof addresses');inertia=compiled_snapshot_check(directory/'prevalidation_compiled_model.npz',directory/'fixture.xml',contract['inertia_representations'],snapshot_context);require(inertia['all_output_postconditions_passed'],'Independent compiled inertia postcondition fails');N=f['steps'];T=f['duration_s'];dt=f['dt_s'];tail=a['endpoint_time_s']>=T-.1-1e-12
 exact(m['native_target'],goal,'Native target changed');exact(st['qpos'],goal,'Static qpos changed');near(st['ctrl'],controls(goal,contract),1e-14,'Static named control');near(st['actuator_length'],st['ctrl'],LIMITS['transmission_length'],'Static transmission');near(st['actuator_force'],np.zeros(51),LIMITS['zero_servo_force'],'Static servo force');near(st['anchors'],source['source_joint_anchors_world'][0],LIMITS['static_anchor_m'],'Static source anchors');near(st['rotations'],source['source_body_rotations_world'][0],LIMITS['static_rotation_component'],'Static source rotations')
 checks=dict(qpos=True,anchors=True,rotations=True,transmission=True,zero_servo_force=True);require(meta['planned_branch_check']['samples']==N+1 and meta['planned_branch_check']['finite_sampled_check_only'] is True,'Sampled branch denominator');require(meta['static_checks']==checks and measurement['assessment']['static_source_and_transmission_checks']==checks,'Static gates differ')
 require(a['pre_qpos'].shape==(N,67) and a['endpoint_qvel'].shape==(N,51),'Exact trajectory dimensions');near(a['pre_time_s'],np.arange(N)*dt,1e-12,'Pre-step clock');near(a['endpoint_time_s'],(np.arange(N)+1)*dt,1e-12,'Endpoint clock');exact(a['pre_qpos'][1:],a['endpoint_qpos'][:-1],'Recorded qpos continuity');exact(a['pre_qvel'][1:],a['endpoint_qvel'][:-1],'Recorded qvel continuity');exact(a['pre_qpos'][0],m['initial_state'],'Initial fixture state');exact(a['effective_ctrl'],a['requested_ctrl'],'Requested/effective control differs');require(not a['warning_counts'].any(),'Warning counts nonzero');principal(a['pre_qpos']);principal(a['endpoint_qpos'])
 refs=reference(f,m['initial_state'],goal,m['plane_normal']);near(a['reference_qpos'],refs[:-1],1e-14,'Reference ramp');near(a['requested_ctrl'],controls(a['reference_qpos'],contract),1e-14,'Principal named control');near(a['phase_code'],(np.arange(N)*dt >= (.1 if f['kind']=='isolated_capsule_plane' else .5)).astype(a['phase_code'].dtype),0,'Phase clock');near(a['bone_angle_error_rad'],angles(a['body_rotations_start'],source['source_body_rotations_world'][0]),1e-14,'All16 bone angles')
 planned=controls(refs,contract);planned_hash=hashlib.sha256(planned.tobytes()).hexdigest();require(planned_hash==meta['planned_branch_check']['ctrl_sequence_sha256'],'Sampled planned commands hash differs')
 near(a['start_actuator_length'],controls(a['pre_qpos'],contract),1e-12,'Actual principal actuator lengths');
 force_expected=a['requested_ctrl']*m['actuator_gainprm'][:,0]+m['actuator_biasprm'][:,0]+a['start_actuator_length']*m['actuator_biasprm'][:,1]+a['start_actuator_velocity']*m['actuator_biasprm'][:,2];near(a['start_actuator_force'],force_expected,1e-10,'Authored scalar position actuator force');canonical_force=a['start_actuator_force'][:,np.asarray(contract['actuator_ids'])];gears=np.asarray(contract['gears']);qtau=np.empty((N,51));qtau[:,:3]=canonical_force[:,:3]*gears[:3,0];qtau[:,3:]=(canonical_force[:,3:,None]*gears[3:,:3]).reshape(N,16,3,3).sum(axis=2).reshape(N,48);near(a['start_qfrc_actuator'],qtau,1e-10,'Gear-transposed actuator force');
 near(a['actuator_power_W'],np.sum(a['start_qfrc_actuator']*a['pre_qvel'],axis=1),1e-12,'Actuator power');near(a['passive_power_W'],np.sum(a['start_qfrc_passive']*a['pre_qvel'],axis=1),1e-12,'Passive power');near(a['contact_power_W'],np.sum(a['start_qfrc_constraint']*a['pre_qvel'],axis=1),1e-12,'Constraint power')
 metrics={};outchecks={};contact_details=None
 if f['kind']!='isolated_capsule_plane':
  require(not np.any(a['start_qfrc_constraint']),'Contact-off constraint force');expected=goal[:3].copy();mass=float(m['body_mass'].sum())
  if f['kind']=='authored_gravity_equilibrium':expected[2]-=mass*9.81/300
  wrist=np.linalg.norm(a['endpoint_qpos'][tail,:3]-expected,axis=1);metrics.update(wrist_tail_max_m=float(wrist.max()),bone_tail_max_rad=float(a['bone_angle_error_rad'][tail].max()),expected_wrist_world_m=expected.tolist());outchecks['wrist_tracking_or_declared_sag']=metrics['wrist_tail_max_m']<=LIMITS['tracking_wrist_m'];outchecks['all16_bone_tracking']=metrics['bone_tail_max_rad']<=LIMITS['tracking_bone_rad']
  if f['kind']=='authored_gravity_equilibrium':
   bal=np.mean(a['start_qfrc_actuator'][tail,:3]+a['start_qfrc_passive'][tail,:3]-a['start_qfrc_bias'][tail,:3],axis=0);metrics['gravity_tail_force_balance_N']=float(np.linalg.norm(bal));outchecks['authored_gravity_balance']=metrics['gravity_tail_force_balance_N']<=LIMITS['gravity_balance_fraction']*mass*9.81
 else:
  active=a['contact_active'].astype(bool);outchecks['positive_tail_capsule_reaction']=bool(np.any(active[tail]) and np.mean(a['normal_force_N'][tail])>LIMITS['positive_contact_force_N']);require(np.all(a['contact_force'][active,0]>=-1e-12),'Negative normal force');near(a['normal_force_N'],a['contact_force'][:,0],1e-12,'Normal force channel')
  idx=np.flatnonzero(active);point=a['contact_point_world_m'][idx];basis=a['contact_frame'][idx];require(np.max(np.abs(basis@basis.transpose(0,2,1)-np.eye(3)),initial=0.)<=1e-10,'Contact basis not orthonormal');normal=m['plane_normal'];hand=np.flatnonzero(m['geom_bodyid']==contract['body_ids'][3]);plane=np.flatnonzero((m['geom_type']==0)&(m['geom_bodyid']==0));require(len(hand)==len(plane)==1,'Unique capsule/plane model inventory');pairs=a['actual_contact_geom_ids'][idx];require(all(set(map(int,pair))=={int(hand[0]),int(plane[0])} for pair in pairs),'Actual contact geom identities differ');sign=np.where(pairs[:,1]==hand[0],1.,-1.);J=sign[:,None,None]*(basis@point_jac(point,a['anchors_start'][idx],a['body_rotations_start'][idx]));near(a['independent_contact_J'][idx],J,1e-12,'Independent point Jacobian');tau=np.einsum('nki,nk->ni',J,a['contact_force'][idx],optimize=False);residual=a['start_qfrc_constraint'][idx]-tau
  ej=np.max(np.abs(a['engine_contact_J'][idx]-J),axis=(1,2));trans=np.max(np.abs(residual[:,:3]),axis=1);angular=np.max(np.abs(residual[:,3:]),axis=1);near(a['contact_J_error'][idx],ej,1e-12,'J error scalar');near(a['generalized_translation_force_error_N'][idx],trans,1e-12,'Translation force scalar');near(a['generalized_angular_force_error_Nm'][idx],angular,1e-12,'Angular force scalar');near(a['contact_normal_error'][idx],np.max(np.abs(basis[:,0]-sign[:,None]*normal),axis=1),1e-14,'Contact normal scalar')
  fce=a['contact_force'][idx];g=a['engine_contact_J'][idx]@a['start_qacc'][idx,:,None];g=g[:,:,0]+a['contact_regularizer'][idx]*fce-a['contact_reference_acceleration'][idx];near(a['contact_gradient'][idx],g,1e-12,'Contact KKT gradient');mu=1.;primal=np.maximum.reduce([np.zeros(len(idx)),-fce[:,0],np.linalg.norm(fce[:,1:],axis=1)-mu*fce[:,0]]);dual=np.maximum(0,mu*np.linalg.norm(g[:,1:],axis=1)-g[:,0]);comp=np.abs(np.sum(fce*g,axis=1));fs=1+np.linalg.norm(fce,axis=1);gs=1+np.linalg.norm(g,axis=1)
  for name,values in [('cone_primal',primal),('cone_dual',dual),('cone_complementarity',comp),('cone_force_scale',fs),('cone_gradient_scale',gs)]:near(a[name][idx],values,1e-12,name)
  v=a['pre_qvel'][idx];probe=np.sin(np.arange(51));engineJ=a['engine_contact_J'][idx];p1=np.sum(fce*np.einsum('nki,ni->nk',engineJ,v,optimize=False),axis=1);p2=np.sum(tau*v,axis=1);p3=np.sum(fce*np.einsum('nki,i->nk',engineJ,probe,optimize=False),axis=1);p4=tau@probe;power=np.maximum(np.abs(p1-p2),np.abs(p3-p4));scale=np.maximum(np.abs(p1),np.abs(p3));near(a['virtual_power_error_W'][idx],power,1e-12,'Virtual power scalar');near(a['virtual_power_scale_W'][idx],scale,1e-12,'Virtual power scale')
  near(a['contact_gap_error_m'],np.abs(a['analytic_gap_m']-a['engine_gap_m']),1e-14,'Gap comparison scalar');require(np.min(a['analytic_gap_m'])>=-.005 and np.min(a['engine_gap_m'])>=-.005,'Penetration hard bound');gap_replay='SEALED_PRODUCER_ANALYTIC_GAP_ONLY'
  if all(k in a for k in ('capsule_center_world_m','capsule_rotation_world','capsule_size_m')):
   gap=(a['capsule_center_world_m']-m['plane_position'])@normal-a['capsule_size_m'][:,0]-a['capsule_size_m'][:,1]*np.abs(a['capsule_rotation_world'][:,:,2]@normal);present=a['contact_present'].astype(bool);near(a['plane_position_world_m'],np.broadcast_to(m['plane_position'],(N,3)),1e-14,'Plane position');near(a['plane_rotation_world'][:,:,2],np.broadcast_to(normal,(N,3)),1e-14,'Plane normal');near(a['analytic_gap_m'][present],gap[present],1e-14,'Analytic capsule-plane gap');require(np.all(~active|present),'Active contact without present contact');near(a['actual_contact_solref'][active],np.broadcast_to([.04,1.],(int(active.sum()),2)),0,'Authored contact solref');near(a['actual_contact_solimp'][active],np.broadcast_to([.5,.95,.001,.5,2.],(int(active.sum()),5)),0,'Authored contact solimp');near(a['actual_contact_friction'][active,:2],np.ones((int(active.sum()),2)),0,'Authored contact friction');gap_replay='INDEPENDENT_FROM_SAVED_COMPILED_CAPSULE_POSE'
  for field,limit in [('contact_J_error',LIMITS['contact_J']),('contact_gap_error_m',LIMITS['contact_gap_m']),('contact_normal_error',LIMITS['contact_normal']),('generalized_translation_force_error_N',LIMITS['contact_translation_force_N']),('generalized_angular_force_error_Nm',LIMITS['contact_angular_force_Nm'])]:metrics[field]=float(a[field].max());outchecks[field]=metrics[field]<=limit
  outchecks['virtual_power']=bool(np.all(a['virtual_power_error_W']<=1e-10+1e-10*a['virtual_power_scale_W']));outchecks['cone_primal']=bool(np.all(a['cone_primal']<=1e-8*a['cone_force_scale']));outchecks['cone_dual']=bool(np.all(a['cone_dual']<=1e-7*a['cone_gradient_scale']));outchecks['cone_complementarity']=bool(np.all(a['cone_complementarity']<=1e-7*a['cone_force_scale']*a['cone_gradient_scale']));contact_details=dict(active_intervals=int(active.sum()),gap_replay=gap_replay,maximum_independent_J_error=float(ej.max()) if len(idx) else None,actual_contact_parameter_values='Verified saved active-row solref/solimp/friction')
 status='PASS' if all(outchecks.values()) else 'FAIL';assessment=measurement['assessment'];scalar_equal(assessment['checks'],outchecks,'Fixture checks');scalar_equal(assessment['metrics'],metrics,'Fixture metrics');require(assessment['status']==status,'Fixture status differs');closed(assessment);closed(measurement);require(measurement['command_clipping'] is False,'Command clipping claimed');exact(np.asarray(measurement['final_qpos']),a['endpoint_qpos'][-1],'Final qpos differs')
 return dict(fixture=f,status=status,checks=outchecks,metrics=metrics,contact=contact_details,compiled_inertia=inertia,final_qpos=measurement['final_qpos'],final_rotations=measurement['final_rotations'],trajectory_rows=N)

def verify(protocol,freeze,expected_report_sha=None):
 protocol=Path(protocol).resolve();root=protocol.parents[2];p=read(protocol);require(p['schema_version']==STUDY_SCHEMA,'Unexpected protocol');closed(p);require(p['method']['plan']==plan() and p['method']['limits']==LIMITS and p['method']['all12_static_checks_before_first_step'] is True,'Declared fixed method differs');before=closure(p,protocol,root,freeze);prior=prior_failure(root,p);out=Path(p['output_root']);raw=Path(p['raw_root']);rp=out/'report.json';report=read(rp)
 if expected_report_sha:require(sha(rp)==expected_report_sha,'Explicit report hash differs')
 require(report['protocol_sha256']==sha(protocol) and report['freeze_commit']==freeze,'Report freeze identity differs');closed(report);started=read(out/'STARTED.json');require(started['protocol_sha256']==sha(protocol) and started['freeze_commit']==freeze and started['budget']==p['method']['budget'],'STARTED identity differs')
 for key in ('worker_result','raw_result'):
  if report.get(key):rowcheck(report[key])
 worker=None;artifacts=[];details=[];schedule=[];partial_inventory=[];compiled_snapshots=[];static_receipts=[]
 if report.get('worker_result'):
  require(Path(report['worker_result']['path']).read_bytes()==Path(report['raw_result']['path']).read_bytes(),'Raw/tracked worker bytes differ');worker=read(report['worker_result']['path']);require(worker['protocol_sha256']==sha(protocol) and worker['freeze_commit']==freeze,'Worker freeze identity differs');closed(worker);require(worker['cases']==report['cases'],'Worker/report case rows differ');require(worker.get('schema_version')==STUDY_SCHEMA+'_worker','Worker schema differs');require(worker.get('remaining_case_indices_NOT_RUN')==report.get('remaining_case_indices_NOT_RUN'),'Remaining-case identities differ');artifacts=worker['artifacts'];paths={}
  require(report.get('actual_counts')==worker.get('actual_counts'),'Report/worker counts differ');schedule=worker.get('fixture_schedule',[]);require(report.get('fixture_schedule')==schedule and [x['fixture'] for x in schedule]==plan(),'All12 schedule identity differs');
  allowed={'COMPLETED_ASSESSED','HARD_STOPPED_PARTIAL','STATIC_PREFLIGHT_PASSED_DYNAMICS_NOT_RUN','HARD_STOPPED_STATIC_PREFLIGHT','NOT_RUN'};require(all(x['status'] in allowed for x in schedule),'Unknown schedule status');
  for key,value in worker.get('actual_counts',{}).items():require(isinstance(value,int) and 0<=value<=p['method']['budget'][key],'Actual quota bound escaped')
  require(all(sum(worker['per_case_counts'][str(i)].get(k,0) for i in (20,28))==value for k,value in worker.get('actual_counts',{}).items()),'Partial per-case/global counts differ')
  for case in worker['cases']:
   closed(case)
   for measurement in case.get('measurements',[]):closed(measurement);closed(measurement['assessment'])
  for row in artifacts:
   rowcheck(row);path=Path(row['path']).resolve();require(path.is_relative_to(raw.resolve()) and path not in paths,'Artifact escaped/duplicate');paths[path]=row
   if path.name=='fixture.xml':
    fixture_key=str(path.parent.relative_to(raw));require(p['fixture_xml_sha256'].get(fixture_key)==sha(path),'Partial/complete XML not declared')
  extras={q.resolve() for q in raw.rglob('*') if q.is_file()}-set(paths)-{raw.resolve()/'result.json',raw.resolve()/'progress.json'};require(not any(q.is_symlink() for q in raw.rglob('*')),'Output symlink')
  for path in paths:
   if path.name=='prevalidation_compiled_model.npz':
    context=dict(protocol=p,index=int(path.relative_to(raw).parts[0]),root=root);metadata_path=path.with_name('command_contract.json');metadata=read(metadata_path)['compiled']['inertia_representations'] if metadata_path in paths else None;compiled_snapshots.append(compiled_snapshot_check(path,path.with_name('fixture.xml'),metadata,context))
   if path.name=='command_contract.json':
    index=int(path.relative_to(raw).parts[0]);spec=p['cases'][str(index)];literal=npz(spec['literal_buffers']['path']);source=npz(spec['source_readbacks']['path']);goal=literal['source_initialized_q0'];st=npz(path.with_name('static_command.npz'));meta=read(path);checks=meta['static_checks'];require(checks==dict(qpos=True,anchors=True,rotations=True,transmission=True,zero_servo_force=True),'Static receipt not passed');exact(st['qpos'],goal,'Static native q0');near(st['ctrl'],controls(goal,meta['compiled']),1e-14,'Static named transmission control');near(st['ctrl'],st['actuator_length'],LIMITS['transmission_length'],'Static engine transmission');near(st['actuator_force'],np.zeros(51),LIMITS['zero_servo_force'],'Static zero servo');anchor_error=near(st['anchors'],source['source_joint_anchors_world'][0],LIMITS['static_anchor_m'],'Static native anchors');rotation_error=near(st['rotations'],source['source_body_rotations_world'][0],LIMITS['static_rotation_component'],'Static native rotations');static_receipts.append(dict(fixture_path=str(path.parent.relative_to(raw)),all_five_static_checks_independently_passed=True,maximum_anchor_component_error_m=anchor_error,maximum_rotation_component_error=rotation_error))
  require(len(compiled_snapshots)<=worker.get('actual_counts',{}).get('models',0),'More early snapshots than compile attempts')
  if report['status']=='COMPLETED':require(len(artifacts)==72 and len(compiled_snapshots)==12,'Exact72 completed artifacts/12 early compiled snapshots');require(not extras and all(x['status']=='COMPLETED_ASSESSED' for x in schedule),'Unindexed/incomplete completed artifacts')
  partial_inventory=[bind(x) for x in sorted(extras)]
  for cell in schedule:
   f=cell['fixture'];stem=str(f['index'])+'/'+f['kind']+'_'+str(round(1/f['dt_s']));names={str(path.relative_to(raw)) for path in paths};case=next((c for c in worker['cases'] if c['index']==f['index']),{});done=any(m['fixture']==f for m in case.get('measurements',[]));expected='COMPLETED_ASSESSED' if done else ('HARD_STOPPED_PARTIAL' if stem+'/partial_live_state.npz' in names else ('STATIC_PREFLIGHT_PASSED_DYNAMICS_NOT_RUN' if stem+'/command_contract.json' in names else ('HARD_STOPPED_STATIC_PREFLIGHT' if stem+'/fixture.xml' in names else 'NOT_RUN')));require(cell['status']==expected,'Saved schedule differs from artifact/assessment accounting')
  if report['status']=='COMPLETED':
   require(worker['status']=='COMPLETED' and report['returncode']==0 and report['failure'] is None and worker['failure'] is None,'Completed failure mismatch');require(worker['all12_static_checks_before_dynamics'] is True and not worker['forbidden_attempts'],'Static gate/forbidden operations');require([c['index'] for c in worker['cases']]==[20,28] and worker['remaining_case_indices_NOT_RUN']==[],'Case denominator changed')
   counts=worker['actual_counts'];require(all(counts[k]==p['method']['budget'][k] for k in ('models','data','static_forwards','steps')) and counts['contact_force_reads']<=p['method']['budget']['contact_force_reads'],'Actual global quota inventory');require(all(sum(worker['per_case_counts'][str(i)][k] for i in (20,28))==counts[k] for k in counts),'Per-case/global counts differ')
   # All static receipts precede every first trajectory publication. True step
   # execution order is the frozen Engine gate assertion, not file timestamps.
   trajectory_positions=[i for i,a in enumerate(artifacts) if Path(a['path']).name=='trajectory.npz'];static_positions=[i for i,a in enumerate(artifacts) if Path(a['path']).name=='command_contract.json'];require(len(trajectory_positions)==12 and len(static_positions)==12 and max(static_positions)<min(trajectory_positions),'All12 static artifacts must be sealed before trajectories')
   for case in worker['cases']:
    i=case['index'];spec=p['cases'][str(i)];literal=npz(spec['literal_buffers']['path']);source=npz(spec['source_readbacks']['path']);goal=literal['source_initialized_q0'];exact(goal,source['actual_named_qpos'][0],'Original qualified source q0 differs');require(case['status']=='COMPLETED' and len(case['measurements'])==6,'Case fixture denominator');closed(case);require(all(v==0 for v in case['forbidden_counts'].values()),'Forbidden case operations');rows=[]
    for fixture,measurement in zip([f for f in plan() if f['index']==i],case['measurements']):require(measurement['fixture']==fixture,'Fixture ordering differs');require(sha(raw/str(i)/(fixture['kind']+'_'+str(round(1/fixture['dt_s'])))/'fixture.xml')==p['fixture_xml_sha256'][str(i)+'/'+fixture['kind']+'_'+str(round(1/fixture['dt_s']))],'Declared XML hash differs');rows.append(fixture_check(fixture,raw/str(i),measurement,source,goal,paths,dict(protocol=p,index=i,root=root)))
    convergence=[]
    for kind in KINDS:
     pair=[v for v in rows if v['fixture']['kind']==kind];dx=float(np.linalg.norm(np.asarray(pair[0]['final_qpos'])[:3]-np.asarray(pair[1]['final_qpos'])[:3]));dr=float(angles(pair[0]['final_rotations'],pair[1]['final_rotations']).max());convergence.append(dict(kind=kind,wrist_difference_m=dx,bone_difference_rad=dr,passed=dx<=.001 and dr<=float(np.deg2rad(1))))
    require(all(worker['per_case_counts'][str(i)][k]==p['method']['per_case_budget'][k] for k in ('models','data','static_forwards','steps')),'Per-case quotas differ');require(len(case['convergence'])==3,'Convergence count');
    for actual,expected in zip(case['convergence'],convergence):scalar_equal(actual,expected,'Convergence')
    qualification='PASS' if all(v['status']=='PASS' for v in rows) and all(v['passed'] for v in convergence) else 'NOT_QUALIFIED';require(case['qualification']==qualification,'Case qualification differs');details.append(dict(index=i,qualification=qualification,fixtures=rows,convergence=convergence,operation_counts=worker['per_case_counts'][str(i)]))
   require(report['qualification']==('PASS' if all(c['qualification']=='PASS' for c in details) else 'NOT_QUALIFIED'),'Top qualification differs')
  else:
   require(report['status']=='HARD_STOP' and report['qualification']=='NOT_QUALIFIED','Failure must remain unqualified')
   for case in worker['cases']:
    i=case['index'];spec=p['cases'][str(i)];literal=npz(spec['literal_buffers']['path']);source=npz(spec['source_readbacks']['path']);goal=literal['source_initialized_q0'];exact(goal,source['actual_named_qpos'][0],'Partial original source q0');rows=[]
    for measurement in case.get('measurements',[]):
     fixture=measurement['fixture'];require(fixture in plan() and fixture['index']==i,'Partial completed-fixture identity');rows.append(fixture_check(fixture,raw/str(i),measurement,source,goal,paths,dict(protocol=p,index=i,root=root)))
    details.append(dict(index=i,status=case['status'],qualification='NOT_QUALIFIED',fixtures=rows,convergence=None,failed_fixture=case.get('failed_fixture'),operation_counts=worker['per_case_counts'][str(i)]))
 for row in report.get('partial_files',[]):rowcheck(row)
 for row in artifacts:rowcheck(row)
 for case in details:
  reads=sum(r['contact']['active_intervals'] for r in case['fixtures'] if r['contact'] is not None);require(reads==case['operation_counts']['contact_force_reads'] or report['status']=='HARD_STOP','Saved active-contact/read count mismatch')
 after=closure(p,protocol,root,freeze);require(before==after,'Closure count changed');return dict(schema_version=SCHEMA,status='VERIFIED_COMPLETED_BOUNDED_CAPSULE_SAVED_READBACK' if report['status']=='COMPLETED' else 'VERIFIED_HARD_STOP_CAPSULE_SAVED_READBACK',experiment_status=report['status'],reported_bounded_qualification=report['qualification'],protocol=bind(protocol),report=bind(rp),freeze_commit=freeze,closure_before=before,closure_after=after,artifact_count=len(artifacts),failed_plane_frame_evidence=failed_plane_frame_evidence(p,worker),static_preflight_receipts=static_receipts,compiled_snapshot_count=len(compiled_snapshots),compiled_snapshots=compiled_snapshots,prior_failure=prior,actual_operation_counts=worker.get('actual_counts',{}) if worker else {},fixture_schedule=schedule,remaining_case_indices_NOT_RUN=report.get('remaining_case_indices_NOT_RUN'),experiment_failure=report.get('failure'),worker_failure=worker.get('failure') if worker else None,verified_unindexed_partial_inventory=partial_inventory,cases=details,physical_outcome=None,**{k:False for k in FLAGS},operations=dict(engine_calls=0,model_constructions=0,MANO_or_skin_calls=0,FK_or_geometry_queries=0,physics_steps=0),limits=['A saved-only algebra audit cannot recreate native simulation or validate physical reality.','Actual integration operation counts and static-before-first-step enforcement are frozen implementation assertions, cross-checked against sealed inventory and recorded clocks.','Final body rotations are a sealed post-forward measurement; only start-of-step rotations are stored throughout trajectories.','No full-skin parity, task grasp success, collection or training admission is established.'])

def prior_failure(root,active):
 old=root/'research_artifacts/manosim_capsule_execution_2026-10-15_v1';pp=old/'protocol.json';rp=old/'run_v1/report.json';ip=old/'run_v1/independent_readback.json'
 for path,digest in ((pp,PRIOR_PROTOCOL_SHA),(rp,PRIOR_REPORT_SHA),(ip,PRIOR_RECEIPT_SHA)):
  require(sha(path)==digest,'Preserved v1 failure lineage changed');require(active['active_bindings'].get(str(path))==digest,'Prior failure not bound prospectively')
 p=read(pp);r=read(rp);i=read(ip);require(r['status']=='HARD_STOP' and r['qualification']=='NOT_QUALIFIED' and i['status']=='VERIFIED_HARD_STOP_CAPSULE_SAVED_READBACK','Prior failure status changed');require(i['report']['sha256']==PRIOR_REPORT_SHA and i['protocol']['sha256']==PRIOR_PROTOCOL_SHA,'Prior independent identity changed');closed(r);require(r['actual_counts']==dict(models=1,data=0,static_forwards=0,steps=0,contact_force_reads=0),'Prior before-data failure counts changed');historical=0;live=0
 for path,digest in p['active_bindings'].items():
  path=Path(path);relative=str(path.relative_to(root)) if path.is_relative_to(root) else None
  if relative in p['selected_git_sources']:
   require(hashlib.sha256(git(root,'show',PRIOR_FREEZE+':'+relative)).hexdigest()==digest,'Prior frozen source changed');historical+=1
  else:require(sha(path)==digest,'Prior unchanged non-source binding changed');live+=1
 require(git(root,'show',PRIOR_FREEZE+':'+str(pp.relative_to(root)))==pp.read_bytes(),'Prior frozen protocol changed')
 for row in r['partial_files']:rowcheck(row)
 return dict(protocol=bind(pp),report=bind(rp),independent_receipt=bind(ip),historical_source_blobs=historical,unchanged_non_source_bindings=live,original_actual_counts=r['actual_counts'],status='PRESERVED_HARD_STOP_BEFORE_DATA_NO_POSE_OR_DYNAMICS_PASS')

def rotation(q):
 q=np.asarray(q,dtype=float);require(q.shape==(4,) and np.isfinite(q).all() and abs(float(q@q)-1)<=1e-12,'Compiled quaternion invalid');w,x,y,z=q;k=2/float(q@q);return np.array([[1-k*(y*y+z*z),k*(x*y-z*w),k*(x*z+y*w)],[k*(x*y+z*w),1-k*(x*x+z*z),k*(y*z-x*w)],[k*(x*z-y*w),k*(y*z+x*w),1-k*(x*x+y*y)]])

def inertia_arithmetic(full,principal,q):
 # Independent arithmetic only, no eigensolver/compiler invocation. Both
 # termination tests of pinned user_util.cc611-700 are retained prospectively.
 import math
 f=np.asarray(full,dtype=float);eig=np.asarray(principal,dtype=float);require(f.shape==(6,) and eig.shape==(3,) and np.isfinite(f).all() and np.isfinite(eig).all() and np.all(eig>0),'Finite positive inertia operands required');I=np.array([[f[0],f[3],f[4]],[f[3],f[1],f[5]],[f[4],f[5],f[2]]]);R=rotation(q)
 D=np.array([[math.fsum(float(R[k,i])*float(I[k,l])*float(R[l,j]) for k in range(3) for l in range(3)) for j in range(3)] for i in range(3)]);diag=np.diag(D);off=D-np.diag(diag);unit=np.finfo(float).eps/2;g32=32*unit/(1-32*unit);g128=128*unit/(1-128*unit);low=np.nextafter((1-1e-12)/(1+g32),0.);sq=np.nextafter(low*low,0.);t2=np.nextafter(np.nextafter(1/sq,np.inf)-1,np.inf);t=np.nextafter(np.sqrt(t2),np.inf);coefficient=np.nextafter(t/(1-t2)*(1+g32),np.inf);scale=float(np.max(np.sum(np.abs(I),axis=1)));arithmetic=g128*scale;span=float(diag.max()-diag.min());bound=max(1e-12,float(coefficient)*span)+arithmetic;diag_error=float(np.max(np.abs(diag-eig)));off_error=float(np.max(np.abs(off)));reconstructed=np.array([[math.fsum(float(R[i,k])*float(eig[k])*float(R[j,k]) for k in range(3)) for j in range(3)] for i in range(3)]);require(np.isfinite([arithmetic,span,bound,diag_error,off_error]).all() and scale>0,'Invalid inertia arithmetic')
 return dict(source_in_compiled_frame_kg_m2=D.tolist(),diagonal_error_kg_m2=diag_error,diagonal_arithmetic_bound_kg_m2=arithmetic,maximum_offdiagonal_kg_m2=off_error,jacobi_offdiagonal_bound_kg_m2=bound,diagonal_span_kg_m2=span,cosine_stop_coefficient=float(coefficient),gamma32=g32,gamma128=g128,maximum_reconstructed_tensor_error_kg_m2=float(np.max(np.abs(reconstructed-I))),output_eigenframe_postconditions_passed=bool(diag_error<=arithmetic and off_error<=bound))

def compiled_snapshot_check(snapshot_path,xml_path,declared_metadata=None,context=None):
 a=npz(snapshot_path);root=ET.parse(xml_path).getroot();bodies={b.get('name'):b for b in root.findall('./worldbody//body')};names=[f'mano_body_{i}' for i in range(16)];require(a['source_body_names'].tolist()==names,'Exact named source-body order required');ids=np.asarray(a['named_body_ids']);require(ids.shape==(16,) and len(set(map(int,ids)))==16 and np.all(ids>0),'Named compiled body identities');source=[];rows=[]
 if context is not None:
  pp=context['protocol'];ii=context['index'];spec=pp['cases'][str(ii)];literal=npz(spec['literal_buffers']['path']);require(a['validation_performed'].shape==() and a['validation_performed'].item() is False,'Snapshot must precede mapping validation');near(a['model_dimensions'],np.array([67,51,51,19,17,17]),0,'Exact early compiled dimensions');require(str(a['fixture_xml_sha256'].item())==sha(xml_path),'Early XML identity');require(str(a['authored_capsule_xml_sha256'].item())==sha(spec['capsule_xml']),'Early authored XML identity');require(str(a['asset_closure_sha256'].item())==spec['asset_closure_sha256'],'Early asset closure identity');require(str(a['mapping_controller_source_sha256'].item())==pp['active_bindings'][str(context['root']/'src/grasp_failure_prediction/evaluation/manosim_capsule.py')],'Early controller-source identity');exact(a['rest_quaternions'],literal['rest_quaternions'],'Early literal rest quaternions');jn=['wrist_t'+x for x in 'xyz']+['wrist_rot']+[f'mano_joint_{j}' for j in range(1,16)];an=['wrist_motor_t'+x for x in 'xyz']+['wrist_motor_r'+x for x in 'xyz']+[f'mano_motor_{j}_{x}' for j in range(1,16) for x in 'xyz'];require(a['source_joint_names'].tolist()==jn and a['source_actuator_names'].tolist()==an,'Early named state/controller order');require(set(map(int,a['named_joint_ids']))==set(range(19)) and set(map(int,a['named_actuator_ids']))==set(range(51)),'Early named-ID permutation')
 for i,name in enumerate(names):
  body=bodies[name];inertial=body.find('inertial');require(inertial is not None,'Explicit source inertia required');full=np.asarray([float(x) for x in inertial.get('fullinertia').split()]);require(full.shape==(6,),'Six source fullinertia values');source.append(full);bid=int(ids[i]);require(float(a['body_mass'][bid])==float(inertial.get('mass')),'Source mass changed');near(a['body_ipos'][bid],[float(x) for x in inertial.get('pos','0 0 0').split()],0,'Source inertial origin');near(a['body_pos'][bid],[float(x) for x in body.get('pos','0 0 0').split()],0,'Source body origin');source_q=np.asarray([float(x) for x in body.get('quat','1 0 0 0').split()]);near(rotation(a['body_quat'][bid]),rotation(source_q),128*np.finfo(float).eps,'Source body rest rotation');require(a['body_parentid'][bid]==(0 if i==0 else ids[PARENTS[i]]),'Source anatomical tree changed');row=inertia_arithmetic(full,a['body_inertia'][bid],a['body_iquat'][bid]);row.update(body_name=name,body_id=bid);rows.append(row)
  if declared_metadata is not None:
   m=declared_metadata[i];require(m['body_name']==name and m['body_id']==bid and m['rule']=='mujoco_3_4_0_mjuu_eig3_postcondition_v1' and m['primary_source_sha256']=='f9d5ef77317707039f12658e620ff3393f1eb8a0e7087111e4740207f2ee2522','Inertia compiler-source identity changed');near(m['source_fullinertia_kg_m2'],full,0,'Metadata source inertia');near(m['compiled_principal_kg_m2'],a['body_inertia'][bid],0,'Metadata principal inertia');near(m['compiled_iquat'],a['body_iquat'][bid],0,'Metadata compiled inertia quaternion');near(m['source_in_compiled_frame_kg_m2'],row['source_in_compiled_frame_kg_m2'],row['diagonal_arithmetic_bound_kg_m2'],'Independent inertia-frame arithmetic');require(m['actual_compiler_iteration_count'] is None and m['compiler_maximum_iterations']==500 and m['iteration_limit_is_not_assumed_converged'] is True and m['source_inertia_changed'] is False,'No inferred compiler convergence/source mutation');require(m['output_eigenframe_postconditions_passed']==row['output_eigenframe_postconditions_passed'],'Independent inertia decision differs')
   for key in ('diagonal_error_kg_m2','maximum_offdiagonal_kg_m2','maximum_reconstructed_tensor_error_kg_m2','diagonal_span_kg_m2'):require(abs(m[key]-row[key])<=row['diagonal_arithmetic_bound_kg_m2']*4,'Independent inertia residual differs '+key)
   for key in ('diagonal_arithmetic_bound_kg_m2','jacobi_offdiagonal_bound_kg_m2','cosine_stop_coefficient','gamma32','gamma128'):require(abs(m[key]-row[key])<=max(abs(row[key])*1e-13,np.finfo(float).tiny),'Independent inertia bound differs '+key)
 near(a['source_fullinertia6'],np.asarray(source),0,'Snapshot source fullinertia differs from literal XML')
 if 'source_body_mass' in a:near(a['source_body_mass'],[float(bodies[name].find('inertial').get('mass')) for name in names],0,'Snapshot source masses differ from XML')
 return dict(snapshot=bind(snapshot_path),body_count=16,all_output_postconditions_passed=all(r['output_eigenframe_postconditions_passed'] for r in rows),bodies=rows,actual_compiler_iteration_count=None,scope='Source-equivalent approximate compiler eigenframe, not physical parameter calibration')

def failed_plane_frame_evidence(p,worker):
 if not worker or worker.get('status')!='HARD_STOP':return None
 raw=Path(p['raw_root']);failure=next((r for r in worker.get('fixture_schedule',[]) if r['status']=='HARD_STOPPED_PARTIAL' and r['fixture']['kind']=='isolated_capsule_plane'),None)
 if not failure:return None
 f=failure['fixture'];d=raw/str(f['index'])/(f['kind']+'_'+str(round(1/f['dt_s'])));early=npz(d/'prevalidation_compiled_model.npz');model=npz(d/'fixture_model.npz');live=npz(d/'partial_live_state.npz');plane=np.flatnonzero((early['geom_type']==0)&(early['geom_bodyid']==0));require(len(plane)==1,'Unique saved plane');pid=int(plane[0]);initial=early['geom_quat'][pid];changed=model['geom_quat'][pid];live_norm=float(np.max(np.abs(np.linalg.norm(live['qpos'][3:].reshape(16,4),axis=1)-1)));require(live_norm<=LIMITS['raw_ball_norm'],'Failed live state quaternion invalid');require(not live['warning_counts'].any() and not live['qfrc_applied'].any() and not live['xfrc_applied'].any(),'Failure warning/unplanned-force evidence');xml=ET.parse(d/'fixture.xml').getroot();literal=xml.find('./worldbody/geom[@name="qualification_plane"]');require(literal is not None,'Literal plane absent')
 return dict(fixture=f,partial_live_state=bind(d/'partial_live_state.npz'),failed_live_time_s=float(live['time_s']),live_warning_count=int(live['warning_counts'].sum()),live_ball_norm_max_error=live_norm,compiled_plane_quaternion_before_runtime_edit=initial.tolist(),saved_model_plane_quaternion_after_runtime_edit=changed.tolist(),declared_plane_normal_world=model['plane_normal'].tolist(),model_quaternion_rotates_z_to_declared_normal_max_error=near(rotation(changed)[:,2],model['plane_normal'],1e-14,'Declared model plane normal'),literal_XML_plane_quaternion=literal.get('quat','1 0 0 0'),compiled_geom_sameframe_array_saved='geom_sameframe' in early,actual_failure_geom_frames_saved=any(k in live for k in ('geom_xpos','geom_xmat')),actual_failure_contact_distance_or_frame_saved=any(k in live for k in ('contact_frame','engine_gap_m','contact_dist')),source_based_hypothesis_only='A runtime identity-to-nonidentity plane quaternion edit can conflict with an unchanged compiled same-frame shortcut. The actual failure geom frames/cache/contact distance were not saved, so this is not a measured explanation of the penetration refusal.',measured_contact_or_force_qualification=False)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--freeze-commit',required=True);parser.add_argument('--expected-report-sha256');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();require(not args.output.exists(),'Independent receipt is write-once');failure=None
 try:r=verify(args.protocol,args.freeze_commit,args.expected_report_sha256)
 except BaseException as exc:r=dict(schema_version=SCHEMA,status='VERIFICATION_HARD_STOP',failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),physical_outcome=None,**{k:False for k in FLAGS});failure=exc
 r['verifier_source']=dict(**bind(__file__),source=Path(__file__).read_text());args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('x') as f:json.dump(r,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(status=r['status'],receipt=str(args.output),sha256=sha(args.output))))
 return 1 if failure else 0
if __name__=='__main__':raise SystemExit(main())
