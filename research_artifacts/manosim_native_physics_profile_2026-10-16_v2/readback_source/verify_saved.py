"""Independent v2 saved-only reader; exact v1 independent math retained.
No producer imports, engine/native/FK/skin/contact API calls or physics replay.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,traceback
from xml.etree import ElementTree as ET
import numpy as np
HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path,expected=None):
 if expected is not None and digest(path)!=expected:raise ValueError('Own independent math preimage changed')
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v1=load('exact_independent_profile_v1_math',HERE/'v1_saved_math.py','6ad02f6f671f728b7192daf23d9dc0838783b11cccdf0efcde01ad027a734688')
if digest(HERE/'array_helpers.py')!='2d6ff3859aff047b873b583145bc8a8011e8ad8b4551f65ce9b685f808004eb7':raise ValueError('Own array math preimage changed')
h=v1.h;cache=load('independent_compile_time_cache',HERE/'cache_math.py')
SCHEMA='manosim_native_compensated_physics_independent_saved_readback_v2'
KINDS=v1.KINDS;GATES=v1.GATES;FLAGS=v1.FLAGS
BUDGET=dict(models=12,data=12,static_forwards=240,steps=11250,contact_force_reads=4500,maximum_seconds=600,outer_seconds=610)
def plan():return [dict(index=i,kind=k,dt_s=dt,duration_s=T,steps=int(round(T/dt))) for i in (20,28) for dt in (.002,.001) for k,T in zip(KINDS,(1.5,.75,1.5))]
snapshot_checks=v1.snapshot_checks;partial_description=v1.partial_description



refs=v1.refs;gravity=v1.gravity;contact_check=v1.contact_check;approach_check=v1.approach_check

def plane_preflight_v2(directory,m,c):
 checks,rows=v1.plane_preflight(directory,m,c);a=h.npz(directory/'plane_preflight.npz');adr=int(m['body_bvhadr'][0]);num=int(m['body_bvhnum'][0]);checks.update(world_has_two_compiled_planes=bool(m['body_geomnum'][0]==2),world_has_static_tree=bool(adr>=0 and num==3));worldpos=a['cache_body_inertial_position'][0];worldR=a['cache_body_inertial_rotation'][0]
 for j in (0,1):
  prefix=f'pair{j}_';hg,pg=map(int,a[prefix+'actual_geom_ids']);P=a[prefix+'actual_geom_xpos'];R=a[prefix+'actual_geom_xmat'];normal=R[1,:,2];axis=R[0,:,2];size=a[prefix+'actual_geom_size'][0];point=P[0]-np.sign(float(np.dot(normal,axis)))*size[1]*axis-size[0]*normal;local=np.einsum('ji,nj->ni',worldR,np.stack((P[1],point))-worldpos,optimize=False);leaf=np.flatnonzero(m['bvh_nodeid'][adr:adr+num]==pg)+adr;h.require(len(leaf)==1,'One static BVH leaf per plane')
  for n in (adr,int(leaf[0])):
   box=m['bvh_aabb'][n];scale=max(1.,float(np.abs(box).max()),float(np.abs(local).max()));arithmetic=4096*np.finfo(float).eps*scale;excess=float(np.max(np.abs(local-box[:3])-box[3:]));checks[f'plane{j}_cache_node{n}_contains_geometry']=bool(np.all(box[3:]>=0) and excess<=arithmetic)
 h.require(all(checks.values()),'Version2 complete plane/cache preflight failed');return checks,rows

def base_fixture(f,directory,measurement,source,goal):
 meta=h.read(directory/'command_contract.json');static,m,c=snapshot_checks(directory,meta,source,goal);a=h.npz(directory/'trajectory.npz');N=f['steps'];tail=a['endpoint_time_s']>=f['duration_s']-.1-1e-12
 h.require(a['pre_qpos'].shape==a['endpoint_qpos'].shape==(N,67) and a['pre_qvel'].shape==a['endpoint_qvel'].shape==(N,51),'Recorded trajectory shape');h.near(a['pre_time_s'],np.arange(N)*f['dt_s'],1e-12,'Pre clocks');h.near(a['endpoint_time_s'],np.arange(1,N+1)*f['dt_s'],1e-12,'Endpoint clocks');h.exact(a['pre_qpos'][1:],a['endpoint_qpos'][:-1],'qpos temporal alignment');h.exact(a['pre_qvel'][1:],a['endpoint_qvel'][:-1],'qvel temporal alignment');h.exact(a['pre_qpos'][0],m['initial_state'],'Exact initial state');h.exact(m['native_target'],goal,'Unchanged target');h.principal(a['pre_qpos']);h.principal(a['endpoint_qpos']);h.require(not a['warning_counts'].any(),'Nonzero warning counts');reference=refs(f,m['initial_state'],goal,m['indentation_world_m']);h.near(a['reference_qpos'],reference[:-1],1e-14,'Exact declared reference trajectory');h.near(a['requested_ctrl'],h.controls(a['reference_qpos'],c),1e-14,'All actual named principal controls');h.exact(a['requested_ctrl'],a['effective_ctrl'],'Command clipping/mutation');h.near(a['bone_angle_error_rad'],h.angles(a['body_rotations_start'],source['source_body_rotations_world'][0]),1e-14,'All16 tracking angles');wrist=np.linalg.norm(a['endpoint_qpos'][:,:3]-goal[:3],axis=1);h.near(a['wrist_goal_error_m'],wrist,1e-14,'Wrist to unchanged goal')
 expected=gravity(m['body_mass'][c['body_ids']],a['start_body_COM_world'],a['anchors_start'],a['body_rotations_start']);h.near(a['start_expected_gravcomp'],expected,1e-12,'Sealed expected gravity mapping');delta=np.abs(a['start_qfrc_gravcomp']-expected);h.near(a['gravity_translation_error_N'],delta[:,:3].max(axis=1),1e-12,'Gravity force scalar');h.near(a['gravity_angular_error_Nm'],delta[:,3:].max(axis=1),1e-12,'Gravity torque scalar')
 metrics=dict(wrist_tail_max_m=float(wrist[tail].max()),bone_tail_max_rad=float(a['bone_angle_error_rad'][tail].max()));checks=dict(all16_bone_tracking=metrics['bone_tail_max_rad']<=GATES['tracking_bone_rad'],wrist_to_unchanged_goal=metrics['wrist_tail_max_m']<=GATES['tracking_wrist_m'],independent_gravity_force_map=bool(np.all(delta[:,:3]<=1e-8) and np.all(delta[:,3:]<=1e-10)));contact=None;approach=None;reads=0
 if f['kind']=='two_finger_planes':
  planechecks,planes=plane_preflight_v2(directory,m,c);h.require(meta['plane_preflight']==planechecks,'Plane preflight booleans differ');v,contact=contact_check(a,m,c);reads=contact['active_contact_reads'];checks['two_simultaneous_positive_tail_reactions']=bool(np.all(a['contact_active'][tail]) and np.all(a['normal_force_N'][tail]>GATES['positive_contact_force_N']))
  for field,key in [('contact_J_error','contact_J'),('contact_gap_error_m','contact_gap_m'),('contact_normal_error','contact_normal'),('generalized_translation_force_error_N','contact_translation_force_N'),('generalized_angular_force_error_Nm','contact_angular_force_Nm')]:checks[field]=bool(np.max(v[field])<=GATES[key])
  checks['summed_virtual_power']=bool(np.all(v['virtual_power_error_W']<=1e-10+1e-10*v['virtual_power_scale_W']))
  for field,key,scale in [('cone_primal','cone_primal_relative',v['cone_force_scale']),('cone_dual','cone_dual_relative',v['cone_gradient_scale']),('cone_complementarity','cone_complementarity_relative',v['cone_force_scale']*v['cone_gradient_scale'])]:checks[field]=bool(np.all(v[field]<=GATES[key]*scale))
  contact.update(plane_preflight=planes,tail_minimum_force_each_N=a['normal_force_N'][tail].min(axis=0).tolist())
 elif f['kind']=='guarded_sphere_table_approach':
  guard,actual,approach=approach_check(directory,a,m,c);h.scalar_equal(meta['approach_guard'],guard,'Reference path guard');checks['predeclared_swept_analytic_guard']=guard['passed'];checks['actual_trajectory_minimum_clearance']=bool(np.all(actual>=.002))
 else:h.require(not a['start_qfrc_constraint'].any(),'Free-motion unexpected constraint force')
 assessment=measurement['assessment'];h.require(assessment['checks']==checks and assessment['status']==('PASS' if all(checks.values()) else 'FAIL'),'Independent assessment differs');h.scalar_equal(assessment['metrics'],metrics,'Tail metrics');h.closed(assessment);h.closed(measurement);h.require(measurement['command_clipping'] is False,'Command clipping');h.exact(np.asarray(measurement['final_qpos']),a['endpoint_qpos'][-1],'Final endpoint qpos');h.require(meta['static_checks']==dict(qpos=True,anchors=True,rotations=True,transmission=True,zero_servo_force=True),'Static mapping status differs')
 residual=a['start_qfrc_bias']-a['start_qfrc_gravcomp'];return dict(fixture=f,status=assessment['status'],checks=checks,metrics=metrics,static=static,contact=contact,approach=approach,trajectory_rows=N,contact_reads=reads,moving_bias_minus_gravity_max_abs=float(np.abs(residual[np.any(a['pre_qvel']!=0,axis=1)]).max(initial=0.)),moving_coriolis_required_cancelled=False,maximum_gravity_translation_error_N=float(delta[:,:3].max()),maximum_gravity_torque_error_Nm=float(delta[:,3:].max()),final_qpos=measurement['final_qpos'],final_rotations=measurement['final_rotations'])

def completed_fixture(f,directory,measurement,source,goal):
 r=base_fixture(f,directory,measurement,source,goal)
 if f['kind']=='two_finger_planes':
  a=h.npz(directory/'trajectory.npz');m=h.npz(directory/'fixture_model.npz');c=h.read(directory/'command_contract.json')['compiled'];C,R,gids=v1.geom_frames(a,m,c);gaps=[]
  for j,b in enumerate((3,6)):
   hg=int(gids[b]);planes=np.flatnonzero((m['geom_bodyid']==0)&(m['geom_type']==0)&(m['geom_contype']==(1<<j)));h.require(len(planes)==1,'Unique declared plane');pg=int(planes[0]);normal=h.rotation(m['geom_quat'][pg])[:,2];gap=np.einsum('ni,i->n',C[:,b]-m['geom_pos'][pg],normal,optimize=False)-m['geom_size'][hg,0]-m['geom_size'][hg,1]*np.abs(np.einsum('ni,i->n',R[:,b,:,2],normal,optimize=False));gaps.append(gap)
  error=h.near(a['analytic_capsule_gap_m'],np.stack(gaps,axis=1),1e-12,'V2 every-pair analytic gap channel');r['new_all_pair_gap_witness_error_m']=error
  values,_=v1.contact_check(a,m,c);tail=a['endpoint_time_s']>=f['duration_s']-.1-1e-12;forces=a['normal_force_N'];active=a['contact_active'].astype(bool);r['contact']['force_each_finger_N']=[dict(bone=b,active_intervals=int(active[:,j].sum()),minimum_all_rows=float(forces[:,j].min()),minimum_active=float(forces[active[:,j],j].min()) if active[:,j].any() else None,mean_active=float(forces[active[:,j],j].mean()) if active[:,j].any() else None,minimum_tail=float(forces[tail,j].min()),mean_tail=float(forces[tail,j].mean())) for j,b in enumerate((3,6))]
  r['contact'].update(maximum_normal_error=float(values['contact_normal_error'].max()),maximum_gap_error_m=float(values['contact_gap_error_m'].max()),maximum_scaled_cone_primal=float(np.max(values['cone_primal']/values['cone_force_scale'])),maximum_scaled_cone_dual=float(np.max(values['cone_dual']/values['cone_gradient_scale'])),maximum_scaled_cone_complementarity=float(np.max(values['cone_complementarity']/(values['cone_force_scale']*values['cone_gradient_scale']))))
 return r

def prior_check(root):
 old=root/'research_artifacts/manosim_native_physics_profile_2026-10-16_v1';rp=old/'run_v1/report.json';ip=old/'run_v1/independent_readback.json';pp=old/'protocol.json'
 h.require(h.sha(ip)=='4f8b4a64035406022be569fbc75833e41860271978f5769c2f8e4101e7a0f312' and h.sha(rp)=='1b0937a8f683079ceba0e798296df00e054ae715ee5c7879ff4dba0c48e4b467','V1 failure identity changed');r=h.read(rp);i=h.read(ip);p=h.read(pp);freeze='510cbd4fb87b2432f76dc68374215e3444f78d6f';h.require(r['status']=='COMPLETED' and r['qualification']=='NOT_QUALIFIED' and i['reported_bounded_qualification']=='NOT_QUALIFIED','Preserved v1 outcomes changed');h.require(r['actual_counts']==dict(models=12,data=12,static_forwards=240,steps=9750,contact_force_reads=0),'V1 actual counts differ');h.require(h.git(root,'show',freeze+':'+str(pp.relative_to(root)))==pp.read_bytes(),'V1 protocol Git differs')
 for path,sha in p['active_bindings'].items():
  relative=str(Path(path).relative_to(root)) if Path(path).is_relative_to(root) else None
  if relative in p['selected_git_sources']:h.require(hashlib.sha256(h.git(root,'show',freeze+':'+relative)).hexdigest()==sha,'V1 selected archival source differs')
  else:h.require(h.sha(path)==sha,'V1 unchanged non-source bytes differ')
 for row in (r['worker_result'],r['raw_result']):h.rowcheck(row)
 for row in h.read(r['worker_result']['path'])['artifacts']:h.rowcheck(row)
 return dict(protocol=h.bind(pp),report=h.bind(rp),independent_receipt=h.bind(ip),qualification='NOT_QUALIFIED',actual_counts=r['actual_counts'],old_BVH_arrays_available=False,old_failure_cause='Source-backed explanation only; no new claims about old unrecorded arrays')

def cache_check(directory):
 m=h.npz(directory/'fixture_model.npz');a=h.npz(directory/'trajectory.npz');early=h.npz(directory/'prevalidation_compiled_model.npz');meta=h.read(directory/'command_contract.json');c=meta['compiled']
 for key in ('geom_pos','geom_quat','geom_sameframe'):h.exact(m[key],early[key],'Authored compile-time geometry unchanged before dynamics '+key)
 fields=tuple(k.removeprefix('final_model_') for k in a if k.startswith('final_model_'));required=set(cache.CACHE_FIELDS)
 h.require(required.issubset(fields),'Required final saved cache/geom operands absent');final={k:a['final_model_'+k] for k in fields};unchanged=cache.exact_unchanged(m,final,fields)
 bvh=cache.world_bvh_readback(m) if int(m['body_bvhnum'][0]) else None
 root=ET.parse(directory/'fixture.xml').getroot();world=root.findall('./worldbody/geom');worldgids=np.flatnonzero(m['geom_bodyid']==0);h.require(len(world)==len(worldgids),'World authored/compiled geom count')
 for xml,gid in zip(world,worldgids):
  h.near(m['geom_pos'][gid],[float(x) for x in xml.get('pos','0 0 0').split()],0,'World plane/obstacle position authored precompile');desired=v1.h.rotation(np.asarray([float(x) for x in xml.get('quat','1 0 0 0').split()]));h.near(v1.h.rotation(m['geom_quat'][gid]),desired,1e-12,'World quaternion authored precompile')
 if (directory/'plane_preflight.npz').exists():
  pre=h.npz(directory/'plane_preflight.npz')
  for key in fields:
   if 'cache_'+key in pre:h.exact(pre['cache_'+key],m[key],'Forward preflight cannot change compiled cache '+key)
  h.require(meta.get('world_geometry_runtime_mutations') is False and meta['world_cache']['runtime_world_geometry_mutation'] is False,'No hot geometry mutation declaration')
 return dict(fixture=str(directory),initial_final_byte_identity=unchanged,world_bvh=bvh,compiled_authored_world_geometries=len(world),physical_acceptance_thresholds_changed=False)

def verify(protocol,freeze,report_sha):
 protocol=Path(protocol).resolve();root=protocol.parents[2];p=h.read(protocol);prior=prior_check(root);h.require(p['schema_version']=='manosim_native_compensated_physics_v2','Unexpected profile');h.closed(p);h.require(p['method']['budget']==BUDGET and p['method']['plan']==plan() and p['method']['limits']==GATES,'Prospective plan/gates differ');before=h.closure(p,protocol,root,freeze);out=Path(p['output_root']);raw=Path(p['raw_root']);rp=out/'report.json';h.require(h.sha(rp)==report_sha,'Explicit expected report differs');initial_report=h.bind(rp);r=h.read(rp);h.closed(r);h.require(r['freeze_commit']==freeze and r['protocol_sha256']==h.sha(protocol),'Report freeze/identity');started=h.read(out/'STARTED.json');h.require(started['freeze_commit']==freeze and started['protocol_sha256']==h.sha(protocol) and started['budget']==BUDGET,'Started identity');worker=None;details=[];snapshots=[];static_rows=[];artifacts=[];partial=[];cache_rows=[]
 for key in ('worker_result','raw_result'):
  if r.get(key):h.rowcheck(r[key])
 if r.get('worker_result'):
  h.require(Path(r['worker_result']['path']).read_bytes()==Path(r['raw_result']['path']).read_bytes(),'Worker raw/tracked mismatch');worker=h.read(r['worker_result']['path']);h.closed(worker);h.require(worker['schema_version']=='manosim_native_compensated_physics_v2_worker' and worker['freeze_commit']==freeze and worker['protocol_sha256']==h.sha(protocol),'Worker identity');h.require(worker['cases']==r['cases'] and worker['fixture_schedule']==r['fixture_schedule'] and worker['actual_counts']==r['actual_counts'],'Worker/report content mismatch');h.require([x['fixture'] for x in worker['fixture_schedule']]==plan(),'Exact12fixture schedule');counts=worker['actual_counts'];h.require(all(isinstance(v,int) and 0<=v<=BUDGET[k] for k,v in counts.items()),'Operation budget escaped');h.require(all(sum(row[k] for row in worker['per_case_counts'].values())==v for k,v in counts.items()),'Per-case/global count mismatch');artifacts=worker['artifacts'];paths={}
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
    f=measurement['fixture'];h.require(f in plan() and f['index']==index,'Unplanned completed fixture');directory=raw/str(index)/(f['kind']+'_'+str(round(1/f['dt_s'])));rows.append(completed_fixture(f,directory,measurement,source,goal));cache_rows.append(cache_check(directory))
   convergence=[]
   if case['status']=='COMPLETED':
    h.require(len(rows)==6,'Completed6fixture denominator')
    for kind in KINDS:
     pair=[x for x in rows if x['fixture']['kind']==kind];dx=float(np.linalg.norm(np.asarray(pair[0]['final_qpos'])[:3]-np.asarray(pair[1]['final_qpos'])[:3]));dr=float(h.angles(pair[0]['final_rotations'],pair[1]['final_rotations']).max());convergence.append(dict(kind=kind,wrist_difference_m=dx,bone_difference_rad=dr,passed=dx<=.001 and dr<=np.deg2rad(1)))
    for a,b0 in zip(case['convergence'],convergence):h.scalar_equal(a,b0,'Convergence')
    qualification='PASS' if all(x['status']=='PASS' for x in rows) and all(x['passed'] for x in convergence) else 'NOT_QUALIFIED';h.require(case['qualification']==qualification,'Case qualification differs')
   else:qualification='NOT_QUALIFIED'
   h.require(worker['per_case_counts'][str(index)]['steps']==sum(x['trajectory_rows'] for x in rows) or r['status']=='HARD_STOP','Per-case interval/step count');details.append(dict(index=index,producer_status=case['status'],qualification=qualification,fixtures=rows,convergence=convergence,operation_counts=worker['per_case_counts'][str(index)]))
  if r['status']=='COMPLETED':
   h.require(len(artifacts)==80 and len(snapshots)==12 and len(static_rows)==12 and len(details)==2 and not extras,'Exact completed80artifact/12static inventory');h.require(worker['all12_static_checks_before_dynamics'] is True and not worker['forbidden_attempts'] and worker['status']=='COMPLETED' and worker['failure'] is None and r['failure'] is None and r['returncode']==0,'Completed worker transaction');h.require(all(counts[k]==BUDGET[k] for k in ('models','data','static_forwards','steps')),'Complete operation counts');h.require(sum(x['contact_reads'] for c in details for x in c['fixtures'])==counts['contact_force_reads'],'Actual contact-force read denominator');h.require(sum(x['trajectory_rows'] for c in details for x in c['fixtures'])==11250,'All11250recorded intervals');static_positions=[i for i,a in enumerate(artifacts) if Path(a['path']).name=='command_contract.json'];trajectory_positions=[i for i,a in enumerate(artifacts) if Path(a['path']).name=='trajectory.npz'];h.require(max(static_positions)<min(trajectory_positions),'All static artifacts must be published before any trajectories');h.require(all(x['status']=='COMPLETED_ASSESSED' for x in worker['fixture_schedule']),'Completed schedule');h.require(r['qualification']==('PASS' if all(c['qualification']=='PASS' for c in details) else 'NOT_QUALIFIED'),'Top qualification')
  else:h.require(r['status']=='HARD_STOP' and r['qualification']=='NOT_QUALIFIED','Hard stop remains unqualified')
 for a in artifacts:h.rowcheck(a)
 for a in r.get('partial_files',[]):h.rowcheck(a)
 for k in ('worker_result','raw_result'):
  if r.get(k):h.rowcheck(r[k])
 h.require(h.bind(rp)==initial_report,'Report changed during replay');after=h.closure(p,protocol,root,freeze);h.require(before==after,'Closure changed')
 return dict(schema_version=SCHEMA,status='VERIFIED_COMPLETED_BOUNDED_NATIVE_PROFILE_SAVED_READBACK' if r['status']=='COMPLETED' else 'VERIFIED_HARD_STOP_NATIVE_PROFILE_SAVED_READBACK',experiment_status=r['status'],reported_bounded_qualification=r['qualification'],protocol=h.bind(protocol),report=h.bind(rp),freeze_commit=freeze,closure_before=before,closure_after=after,actual_operation_counts=r['actual_counts'],fixture_schedule=r['fixture_schedule'],cases=details,prior_v1_unqualified=prior,compiled_cache_readbacks=cache_rows,static_preflight_readbacks=static_rows,compiled_snapshots=snapshots,artifact_count=len(artifacts),partial_artifacts=partial,worker_failure=worker.get('failure') if worker else None,parent_failure=r['failure'],physical_outcome=None,**{k:False for k in FLAGS},operations=dict(engine_calls=0,MANO_or_skin_calls=0,native_FK_or_geometry_API_calls=0,physics_steps=0),limits=['Saved-array arithmetic verifies the bounded record, not physical reality or task-grasp success.','Static-before-first-step228forwards is a frozen Engine gate assertion cross-checked against12preflights, ordered artifacts and recorded clocks; no engine rerun.','Reference rigid-translation certificate is continuous; servo-deviating actual trajectory clearance is only sampled once per step.','Contact friction/compliance checks are frozen producer assertions and compiled-model parameters; per-contact parameter vectors are not present in every trajectory row.','No full-skin articulation/forces parity, human-oracle quality, new task collection or training admission.'])

def numpy_scalar(x):
 if isinstance(x,np.generic):return x.item()
 raise TypeError('Unsupported receipt object '+type(x).__name__)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--freeze-commit',required=True);ap.add_argument('--expected-report-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();h.require(not a.output.exists(),'Write-once receipt');failure=None
 try:r=verify(a.protocol,a.freeze_commit,a.expected_report_sha256)
 except BaseException as e:failure=e;r=dict(schema_version=SCHEMA,status='VERIFICATION_HARD_STOP',failure=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()),physical_outcome=None,**{k:False for k in FLAGS})
 r['verifier_development_history']=dict(initial_attempt='Independent v1 fixture math expected only original plane-preflight keys; v2 legitimately adds compile-time BVH checks.',correction='Versioned adapter independently derives the new cache checks and requires exact full augmented-preflight dictionary equality. Old math source stays byte-identical.',experimental_sources_results_or_gates_changed=False)
 r['verifier_sources']=[dict(**h.bind(p),source=p.read_text()) for p in (Path(__file__),HERE/'cache_math.py',HERE/'array_helpers.py',HERE/'v1_saved_math.py')];a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(r,f,sort_keys=True,indent=2,allow_nan=False,default=numpy_scalar);f.write('\n')
 print(json.dumps(dict(status=r['status'],path=str(a.output),sha256=h.sha(a.output))));return 1 if failure else 0
if __name__=='__main__':raise SystemExit(main())
