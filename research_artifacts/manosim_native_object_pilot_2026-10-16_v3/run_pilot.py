"""Prospective native object pilot transaction; prepare hashes saved inputs only.

Preparation requires the exact independently qualified physical v2 bundle and
saved HUG generation receipt. Task execution separately requires root-authorized
protocol/source freeze; the bounded physical fixtures do not grant task or
scientific-training qualification.
"""
from pathlib import Path
import argparse,functools,hashlib,importlib.abc,importlib.metadata,importlib.util,json,os,subprocess,sys,time,traceback,warnings
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
RUNTIME=ROOT/'runs/manosim_mujoco_3_4_0_runtime_v1';sys.path.insert(0,str(RUNTIME))
if 'mujoco' in sys.modules and (sys.modules['mujoco'].__version__!='3.4.0' or not Path(sys.modules['mujoco'].__file__).resolve().is_relative_to(RUNTIME)):raise RuntimeError('Isolated pinned MuJoCo3.4.0 required')
import numpy as np
if np.__version__!='2.2.6':raise RuntimeError('Base NumPy2.2.6 required')
BASE=Path('/Users/nilakarthikesan/grasp-failure-prediction')
ASSET_ROOT=ROOT/'runs/hug_native_pilot_assets_2026-10-15_v1'
GENERATION=ROOT/'runs/manosim_native_hug_pilot_generation_2026-10-16_v2/generation_report.json'
GENERATION_SHA='f0a1e26b2aa73c3c3bca2a8ba6e5a8091b7b8e28fd215b802fc44eb18b94fc7b'
CATALOG=ROOT/'research_artifacts/manosim_hug_coverage_design_2026-10-15_v1/official_test_catalog_30.json'
PRIOR=ROOT/'research_artifacts/manosim_native_object_pilot_2026-10-16_v1'
PRIOR_COHORT=ROOT/'research_artifacts/manosim_native_pilot_cohort_audit_2026-10-16_v1/independent_readback.json'
PRIOR_FREEZE='959db6c119b67dbf66a6e2ad67f0da36b36d88eb'
PRIOR_REPORT_SHA='55ced44bb921f1ae0a76dd46768de3920132ece4a28ae9a75b552d2bbe2fa9c6'
PRIOR_MECHANICS_SHA='d093594729839b18d70a480b0e9125207e6b069d2a9c9f9f462295e1f5eac726'
PRIOR_COHORT_SHA='8caeb6dcc973b3488c6bb5097013ffa567c59232e398cdacd261e93dc4b1636b'
SOURCE_CAST_LOADER=BASE/'runs/hug_integration_check/hug/src/dataloader/grasp_dataset.py'
FOLDS=CATALOG.parent/'predictor_folds_30.json'
FOLDS_SHA='1d7a7549fd75f25d05a1292e3a9d75ff15f8f60ffe88b9d775e93923cb4887a4'
BUILD=ROOT/'research_artifacts/manosim_native_transfer_2026-10-15_v2/build_run_v1'
PROFILE=ROOT/'research_artifacts/manosim_native_physics_profile_2026-10-16_v2'
RAW_ROOT=ROOT/'runs/manosim_native_object_pilot_2026-10-16_v3/run_v1/raw'
SCHEMA='manosim_native_object_pilot_v3';BUDGET=dict(maximum_seconds=1800,outer_seconds=1830,per_rollout_seconds=60)
PHYSICS_REPORT_SHA='2e69170bbd42a19c601c0b0f1afc54292c27893561b6827c2f9f1fce77695618'
PHYSICS_FREEZE='4687918665fa89ebbc825ede07b445005a449b50'
CATALOG_SHA='04507e8834364d1451c5c34134d6ca1819efe9da45df5c093d8fa322f897626f'
PHYSICS_REPORT_SCHEMA='manosim_native_compensated_physics_v2_report'
PHYSICS_READBACK_SCHEMA='manosim_native_compensated_physics_independent_saved_readback_v2'
BLOCKED=('torch','manotorch','manopth','smplx','transformers','hug')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path):
 p=Path(path)
 if any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('Aliased artifact path')
 p=p.resolve()
 if not p.is_file():raise ValueError('Regular source/input artifact required')
 return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(path):
 def pairs(rows):
  out={}
  for k,v in rows:
   if k in out:raise ValueError('Duplicate JSON key')
   out[k]=v
  return out
 def constant(x):raise ValueError('Nonfinite JSON')
 v=json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=constant)
 def check(x):
  if isinstance(x,float) and not np.isfinite(x):raise ValueError('Nonfinite JSON float')
  if isinstance(x,dict):
   for value in x.values():check(value)
  if isinstance(x,list):
   for value in x:check(value)
 check(v);return v
def write(path,v):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
@functools.cache
def core():return load('native_object_pilot_case',HERE/'pilot_case.py')
def bootstrap():
 for p in (ROOT/'src',BASE/'runs/hug_integration_check/deps',BASE/'runs/hug_integration_check/hug-deps'):
  if str(p) not in sys.path:sys.path.append(str(p))
 sys.path.insert(0,str(ROOT/'src'))
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def verify_row(row):
 if bind(row['path'])!={k:row[k] for k in ('path','sha256','bytes')}:raise ValueError('Write-time artifact changed')
def verify(files):
 for path,digest in files.items():
  if sha(path)!=digest:raise ValueError('Active bound input changed: '+path)
def check_controls(path):
 value=read(path)
 if value.get('exit_code')!=0 or value.get('integration_calls')!=0:raise ValueError('Passing zero-integration manufactured controls required')
 return value

def runtime_files():
 files={Path(sys.executable).resolve(),BASE/'.venv/pyvenv.cfg'}
 for name in ('numpy','mujoco','Pillow','pydantic','pydantic_core','annotated_types','typing_extensions','typing_inspection'):
  d=importlib.metadata.distribution(name)
  for f in d.files or []:
   p=Path(d.locate_file(f)).resolve()
   if p.is_file() and '__pycache__' not in str(p) and (p.suffix in ('.py','.so','.dylib','.h') or '.dist-info/' in str(p)):files.add(p)
 return sorted(files)


def generation_evidence(receipt_path,receipt_sha):
 if sha(GENERATION)!=GENERATION_SHA or sha(receipt_path)!=receipt_sha:raise ValueError('Exact generation/report independent receipt hashes required')
 r=read(GENERATION);v=read(receipt_path)
 if (r.get('status')!='COMPLETE' or r.get('schema_version')!='manosim_native_hug_pilot_proposals_v2' or len(r.get('rows',[]))!=120 or r.get('errors') or r.get('counts')!={'primary':120,'repeat':1,'source_geometry':120}
  or v.get('schema_version')!='manosim_native_hug_generation_independent_saved_readback_v2' or v.get('status')!='VERIFIED_SAVED_NATIVE_TARGET_ALGEBRA' or v.get('report')!={'path':str(GENERATION),'sha256':GENERATION_SHA}
  or v.get('freeze_commit')!=r.get('freeze_commit') or v.get('saved_rows')!=120 or v.get('repeat_saved_arrays_exact') is not True or v.get('errors')):raise ValueError('Completed independently verified saved120 native proposals required')
 protocol=ROOT/'research_artifacts/manosim_native_pilot_2026-10-16_v2/generation_protocol.json'
 if sha(protocol)!=r['protocol_sha256'] or v.get('protocol')!={'path':str(protocol),'sha256':sha(protocol)} or git('show',r['freeze_commit']+':'+str(protocol.relative_to(ROOT)))!=protocol.read_bytes():raise ValueError('Original generation protocol/freeze differs')
 generation_protocol=read(protocol)
 cast_source=generation_protocol['bindings'][str(SOURCE_CAST_LOADER)]
 if sha(SOURCE_CAST_LOADER)!=cast_source['sha256'] or SOURCE_CAST_LOADER.stat().st_size!=cast_source['bytes']:raise ValueError('Pinned upstream K.float source changed')
 manifest=ASSET_ROOT/'download_manifest.json';expected=generation_protocol['bindings'][str(manifest)]
 if sha(manifest)!=expected['sha256'] or manifest.stat().st_size!=expected['bytes']:raise ValueError('Original saved download manifest changed')
 # HUG inference/checkpoint closure is archived lineage, not an active generator.
 return r,[GENERATION,Path(receipt_path),protocol,SOURCE_CAST_LOADER],dict(source_K_cast=dict(source=bind(SOURCE_CAST_LOADER),function='GraspDataset.__getitem__: camera_K=torch.from_numpy(K_np).float()',expected='exact float32 numeric cast of original capture camera.K',camera_world='exact original capture T_world_camera float64',tolerance_used=False),generation_freeze=r['freeze_commit'],generation_report=bind(GENERATION),independent_readback=bind(receipt_path),historical_generation_full_runtime_live_reverified=False,new_HUG_calls=0)


def prior_v1_evidence():
 protocol=PRIOR/'protocol.json';report=PRIOR/'run_v1/report.json';mechanics=PRIOR/'run_v1/independent_readback.json';cohort=PRIOR_COHORT
 if sha(report)!=PRIOR_REPORT_SHA or sha(mechanics)!=PRIOR_MECHANICS_SHA or sha(cohort)!=PRIOR_COHORT_SHA:raise ValueError('Exact independently verified v1 refusal lineage required')
 p=read(protocol);r=read(report);m=read(mechanics);c=read(cohort)
 if (r.get('status')!='COMPLETED' or r.get('freeze_commit')!=PRIOR_FREEZE or r.get('actual_counts',{}).get('steps')!=0 or r['actual_counts'].get('contact_force_reads')!=0
  or len(r.get('attempt_schedule',[]))!=135 or any(a.get('status')!='MAPPING_REFUSAL' or a.get('outcome') is not None for a in r['attempt_schedule'])
  or m.get('schema_version')!='manosim_native_object_pilot_independent_saved_readback_v1' or m.get('status')!='VERIFIED_COMPLETED_NATIVE_OBJECT_PILOT_SAVED_MECHANICS'
  or c.get('schema_version')!='manosim_native_pilot_cohort_independent_saved_readback_v1' or c.get('status')!='VERIFIED_SAVED_PILOT_COHORT_AND_PREFIX_READINESS'
  or any(x.get('report')!=bind(report) or x.get('protocol')!=bind(protocol) or x.get('freeze_commit')!=PRIOR_FREEZE or x.get('scientific_training_allowed') is not False for x in (m,c))):raise ValueError('Original all-refused v1/zero-step/no-training truth required')
 if git('show',PRIOR_FREEZE+':'+str(protocol.relative_to(ROOT)))!=protocol.read_bytes():raise ValueError('Original v1 protocol Git blob changed')
 for relative,digest in p['selected_git_sources'].items():
  if hashlib.sha256(git('show',PRIOR_FREEZE+':'+relative)).hexdigest()!=digest:raise ValueError('Archived v1 selected source Git blob changed')
 files={str(path):sha(path) for path in (protocol,report,mechanics,cohort)}
 for row in [r['worker_result'],*r['raw_artifacts']]:verify_row(row);files[row['path']]=row['sha256']
 if r.get('creation_time_journal'):verify_row(r['creation_time_journal']);files[r['creation_time_journal']['path']]=r['creation_time_journal']['sha256']
 return files,dict(protocol=bind(protocol),report=bind(report),mechanical_readback=bind(mechanics),cohort_readback=bind(cohort),freeze_commit=PRIOR_FREEZE,original_mapping_refusals=135,original_physics_steps=0,original_labels=0,archived_selected_source_blobs_verified=len(p['selected_git_sources']),historical_full_runtime_closure_live_reverified=False,prior_refusals_preserved=True,change='exact upstream float32 K-cast identity only; alignment allowance and all physical/task gates unchanged')


def physical_evidence(protocol_path,report_path,receipt_path,receipt_sha):
 if sha(receipt_path)!=receipt_sha:raise ValueError('Exact physical independent receipt digest required')
 p=read(protocol_path);r=read(report_path);v=read(receipt_path)
 if sha(report_path)!=PHYSICS_REPORT_SHA or r.get('freeze_commit')!=PHYSICS_FREEZE:raise ValueError('Exact corrected physical v2 report required')
 if (r.get('schema_version')!=PHYSICS_REPORT_SCHEMA or r.get('status')!='COMPLETED' or r.get('qualification')!='PASS' or len(r.get('cases',[]))!=2 or any(c.get('qualification')!='PASS' for c in r['cases'])
  or v.get('schema_version')!=PHYSICS_READBACK_SCHEMA or v.get('status')!='VERIFIED_COMPLETED_BOUNDED_NATIVE_PROFILE_SAVED_READBACK' or v.get('reported_bounded_qualification')!='PASS'
  or v.get('protocol')!=bind(protocol_path) or v.get('report')!=bind(report_path) or v.get('freeze_commit')!=r.get('freeze_commit') or r.get('protocol_sha256')!=sha(protocol_path)
  or any(r.get(k) is not False for k in core().modules().profile.FLAGS)):raise ValueError('Qualified gravity + two-finger + guarded-approach physical evidence required; failed fixtures refuse')
 if git('show',r['freeze_commit']+':'+str(Path(protocol_path).relative_to(ROOT)))!=Path(protocol_path).read_bytes():raise ValueError('Physical protocol not frozen at reported commit')
 files={str(Path(protocol_path).resolve()):sha(protocol_path),str(Path(report_path).resolve()):sha(report_path),str(Path(receipt_path).resolve()):receipt_sha}
 # Active physical source/runtime are byte-identical to the qualified profile.
 for helper in ('native_hand.py','profile_math.py'):
  path=str(PROFILE/helper)
  if p.get('active_bindings',{}).get(path)!=sha(path):raise ValueError('Actually imported qualified profile v2 helper bytes differ')
 for path,digest in p['active_bindings'].items():
  q=Path(path)
  if q.is_relative_to(PROFILE) or q.is_relative_to(RUNTIME) or q==ROOT/'src/grasp_failure_prediction/evaluation/manosim_capsule.py':files[path]=digest
 verify(files)
 worker=read(r['worker_result']['path'])
 for row in [r['worker_result'],r['raw_result'],*worker['artifacts']]:verify_row(row);files[row['path']]=row['sha256']
 return files,dict(protocol=bind(protocol_path),report=bind(report_path),independent_readback=bind(receipt_path),freeze_commit=r['freeze_commit'],bounded_qualification='PASS',task_or_training_admission_inherited=False)


def plan(rows,catalog,manifest):
 objects={o['object_id']:o for o in catalog['objects']};schedule=[];models={};observations={};seen=set()
 if len(objects)!=30 or len(rows)!=120:raise ValueError('Fixed30 object /120proposal plan required')
 for i,row in enumerate(rows):
  obj=row['object_path'];meta=objects[obj];observation=row['observation'];seed=row['seed'];identity=(obj,observation,seed)
  if identity in seen or seed not in (42,43) or row.get('grasp_label_available') is not False:raise ValueError('Exact unique seed42/43 unlabeled proposals required')
  seen.add(identity);path=Path(row['path']);path=path if path.is_absolute() else ROOT/path
  if not path.resolve().is_relative_to(GENERATION.parent) or sha(path)!=row['sha256']:raise ValueError('Saved proposal bytes/path differ')
  obs=ASSET_ROOT/observation
  if sha(obs)!=row['input_sha256']:raise ValueError('Original observation bytes differ')
  oid='obs_'+hashlib.sha256(observation.encode()).hexdigest()[:16];objectroot=ASSET_ROOT/obj/'sim_assets';props=read(objectroot/'properties.json');parts=sorted((objectroot/'collision').glob('*.obj'))
  observations[oid]=dict(path=str(obs),visual_mesh=str(objectroot/'visual.obj'),world_position=props['world_position'],object_id=obj)
  model_id='object_'+hashlib.sha256(obj.encode()).hexdigest()[:16]+'_2ms';models[model_id]=dict(object_id=obj,properties=str(objectroot/'properties.json'),object_xml=str(objectroot/'object.xml'),collision_meshes=list(map(str,parts)),dt_s=.002)
  category=meta['official_geometry_category'];geometry='geom_'+hashlib.sha256(json.dumps(meta['geometry_asset_files'],sort_keys=True).encode()).hexdigest();pid='proposal_'+row['sha256'];scenario=dict(object_id=obj,category=category,size=meta['official_size_bin'],replica=meta['cell_replica'],physics_dt_s=.002,gravity_compensation='hand_only',sensor_noise=0.,command_noise=0.,source_XML_sha256=sha(objectroot/'object.xml'),friction_intervention=False,table_placement='source_collision_min_z_plus_original_world_COM')
  scenario_sha=core().digest(scenario);attempt='primary_'+row['id'];seeds=dict(generator=seed,physics=0,sensor=0)
  training_identity=dict(attempt_id=attempt,category_id=category,object_id=obj,geometry_group_id=geometry,observation_group_id=oid,proposal_group_id=pid,scenario_id='scenario_'+scenario_sha[7:],lift_start_s=1.5,capture_sha256='sha256:'+sha(obs),scenario_sha256=scenario_sha,native_pose_sha256='sha256:'+row['sha256'],seeds=seeds)
  schedule.append(dict(attempt_id=attempt,cohort='primary',object_id=obj,category_id=category,size=meta['official_size_bin'],replica=meta['cell_replica'],geometry_group_id=geometry,observation_id=oid,proposal_group_id=pid,model_id=model_id,dt_s=.002,proposal=bind(path),capture_sha256='sha256:'+sha(obs),scenario_sha256=scenario_sha,seeds=seeds,scenario=scenario,training_identity=training_identity))
 # First observed capture, seed42, A object in every official category/size cell.
 for obj,meta in sorted(objects.items()):
  matches=[r for r in schedule if r['object_id']==obj]
  if len(matches)!=4 or len({r['observation_id'] for r in matches})!=2:raise ValueError('Every object needs two exact captures ×two seeds')
  if meta['cell_replica']!='A':continue
  candidate=min((r for r in matches if r['seeds']['generator']==42),key=lambda r:observations[r['observation_id']]['path']);row=json.loads(json.dumps(candidate));row.update(attempt_id='matched_1ms_'+candidate['attempt_id'],cohort='matched_1ms',matched_primary_id=candidate['attempt_id'],model_id=candidate['model_id'].replace('_2ms','_1ms'),dt_s=.001);row['scenario']['physics_dt_s']=.001;row['scenario_sha256']=core().digest(row['scenario']);models[row['model_id']]={**models[candidate['model_id']],'dt_s':.001};schedule.append(row)
 if len(schedule)!=135 or len(models)!=45 or len(observations)!=60 or len({(r['category_id'],r['size']) for r in schedule if r['cohort']=='matched_1ms'})!=15:raise ValueError('Fixed135/45/60/15cell quota differs')
 return schedule,models,observations


def prepare(args):
 bootstrap();generation,files,lineage=generation_evidence(args.generation_readback,args.generation_readback_sha256);physical,physics=physical_evidence(args.physics_protocol,args.physics_report,args.physics_readback,args.physics_readback_sha256);prior,prior_evidence=prior_v1_evidence()
 if sha(FOLDS)!=FOLDS_SHA:raise ValueError('Exact catalog seven-fold declaration required')
 if sha(CATALOG)!=CATALOG_SHA:raise ValueError('Exact official catalog identity required')
 catalog=read(CATALOG);manifest=read(ASSET_ROOT/'download_manifest.json')
 if catalog.get('dataset_revision')!='56ea3f100b1a4ec38bb35e986c0dd14d38977283' or manifest.get('observation_count')!=60 or manifest.get('objects')!=30:raise ValueError('Exact official30object/60capture asset manifest required')
 for row in manifest['files']:
  path=ASSET_ROOT/row['path']
  if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Original downloaded dataset asset changed')
 schedule,models,observations=plan(generation['rows'],catalog,manifest);x=core().modules();first=core().npz(schedule[0]['proposal']['path']);producer=read(BUILD/'builder_report.json');independent_builder=read(BUILD/'independent_readback.json')
 if independent_builder.get('status')!='VERIFIED_LITERAL_SHAPE_BUILD_ARTIFACTS_AND_PRODUCER' or independent_builder.get('builder_report_sha256')!=sha(BUILD/'builder_report.json'):raise ValueError('Independent literal shape build evidence required')
 root=Path(producer['generated_root']);shape=root/'assets/mano_rhand_shape.npy';capsule=root/'assets/literal_native/mano_rhand/capsule_hand.xml';assets=x.api.inspect_authored_assets(capsule,shape,dict(path=str(BUILD/'builder_report.json'),sha256=sha(BUILD/'builder_report.json')),expected_shape_sha256=producer['shape_asset_sha256'],native_shape_float32=first['shape'])
 hand=dict(capsule_xml=str(capsule),shape_asset=str(shape),expected_shape_sha256=producer['shape_asset_sha256'],assets=assets,rest_quaternions=first['rest_quaternions'].tolist())
 own=[p for p in HERE.iterdir() if p.is_file() and (p.suffix=='.py' or p.name=='controls.json')]
 if not (HERE/'controls.json').is_file():raise ValueError('Final manufactured controls receipt required before prepare')
 check_controls(HERE/'controls.json')
 required=[ROOT/'research_artifacts/manosim_scene_correspondence_diagnosis_2026-10-16_v1/authored_asset_transfer.py',ROOT/'research_artifacts/manosim_scene_correspondence_diagnosis_2026-10-16_v1/test_authored_asset_transfer.py',CATALOG,FOLDS]+[ROOT/'src/grasp_failure_prediction/evaluation'/name for name in ('manosim_recording.py','prediction_recording.py','native_task_labels.py','manosim_capsule.py','manosim_pose.py')]+[PROFILE/'native_hand.py',PROFILE/'profile_math.py',ROOT/'research_artifacts/manosim_native_source_pipeline_2026-10-15_v1/source_coordinates.py',ROOT/'research_artifacts/manosim_capsule_execution_2026-10-15_v3/execution_math.py',ROOT/'tests/test_native_task_labels.py',ROOT/'tests/test_manosim_capsule.py',ROOT/'src/grasp_failure_prediction/prediction/manosim_training.py',ROOT/'src/grasp_failure_prediction/prediction/model.py']
 v2=ROOT/'research_artifacts/manosim_native_object_pilot_2026-10-16_v2';v2_lineage=[v2/'protocol.json',v2/'run_v1/report.json',v2/'run_v1/independent_readback.json',ROOT/'research_artifacts/manosim_native_pilot_cohort_audit_2026-10-16_v2/independent_readback.json']
 all_files=set(v2_lineage)|set(map(Path,files))|set(map(Path,physical))|set(map(Path,prior))|set(map(Path,assets['files']))|{CATALOG,ASSET_ROOT/'download_manifest.json',ASSET_ROOT/'official_test_tree.json',BUILD/'builder_report.json',BUILD/'independent_readback.json'}|set(own+required+runtime_files())
 all_files|={ASSET_ROOT/f['path'] for f in manifest['files']}
 all_files|={Path(r['proposal']['path']) for r in schedule}
 active={str(p.resolve()):sha(p) for p in sorted(all_files)};verify(physical)
 source_hashes={str(p.relative_to(ROOT)):sha(p) for p in own+required};tracked=set(git('ls-files').decode().splitlines());selected={str(Path(p).relative_to(ROOT)):d for p,d in active.items() if Path(p).is_relative_to(ROOT) and str(Path(p).relative_to(ROOT)) in tracked};selected.update(source_hashes)
 config=dict(scene_correspondence_admission='exact authored HUG object mesh/frame/COM + native source/camera identity; RGBD reconstruction discrepancy diagnostic; no calibrated physical correspondence claim',phases=dict(reset=[0,.25],approach_motion=[.25,.75],approach_settle=[.75,1.],close=[1.,1.5],lift=[1.5,2.5],hold=[2.5,5.5]),lift_m=.15,control_dt_s=.01,primary_dt_s=.002,matched_dt_s=.001,zero_command_and_sensor_noise=True,reset_support_no_contact_drift_m=.001,preclose_tracking_m=.002,preclose_all16_angle_deg=2.,object_stock_solref=[.02,1.],object_stock_solimp=[.9,.95,.001,.5,2.],masks=dict(hand=[2,5],object=[1,6],table=[4,3]),contact_combine_primary='https://mujoco.readthedocs.io/en/3.4.0/modeling.html#contact-parameters',scene_scope='one target object +table only; other objects visible in sourceRGBD omitted; no clutter generalization claim',hand_compliance_preserved=True,object_original_XML_mass_inertia_preserved=True,all135_static_before_first_task_step=True,maximum_contacts_per_step=256,maximum_contact_penetration_m=.005,quarantine_reason_codes=sorted(core().QUARANTINE_CODES),quarantine_policy='only explicit fixed guard failures continue after full reset; warnings/nonfinite/quota/identity/unregistered calls global hard stop')
 p=dict(schema_version=SCHEMA+'_protocol',active_bindings=active,selected_git_sources=selected,source_hashes=source_hashes,task_configuration=config,hand=hand,profile_sha256=core().digest(dict(profile_id=x.profile.PROFILE_ID,source={str(PROFILE/'native_hand.py'):sha(PROFILE/'native_hand.py'),str(PROFILE/'profile_math.py'):sha(PROFILE/'profile_math.py')},physics_evidence=physics)),hand_version_sha256='sha256:'+assets['closure_sha256'],collider_sha256=core().digest(dict(capsule_XML_sha256=assets['capsule_xml_sha256'],asset_files=assets['files'],collision_masks=config['masks'])),controller_sha256=core().digest(dict(authored_actuators=assets['actuators'],trajectory_source=sha(HERE/'pilot_math.py'),configuration=config)),models=models,observations=observations,schedule=schedule,catalog_binding=bind(CATALOG),catalog_folds_binding=bind(FOLDS),catalog_folds_spec=read(FOLDS),budget={**core().COUNTS,**BUDGET},raw_root=str(RAW_ROOT),generation=lineage,physical_qualification=physics,prior_v1_refusals=prior_evidence,collection_scope='CONDITIONAL AUTHORED HUG ASSET SIMULATION; native capsule approximation; RGBD physical scene correspondence NOT calibrated; no full-skin or hardware transfer claim',physical_outcome=None,**core().FLAGS)
 write(HERE/'protocol.json',p);return dict(protocol_sha256=sha(HERE/'protocol.json'),active_bindings=len(active),selected_git_sources=len(selected),schedule=135,models=45,actual_operations=0)


def validate(protocol_path,freeze):
 protocol_path=Path(protocol_path).resolve();p=read(protocol_path)
 if p.get('schema_version')!=SCHEMA+'_protocol' or p['budget']!={**core().COUNTS,**BUDGET} or p['raw_root']!=str(RAW_ROOT):raise ValueError('Exact versioned pilot/budgets/raw root required')
 if git('rev-parse','HEAD').decode().strip()!=freeze or git('show',freeze+':'+str(protocol_path.relative_to(ROOT)))!=protocol_path.read_bytes():raise ValueError('Frozen HEAD/protocol differs')
 verify(p['active_bindings'])
 for relative,digest in p['selected_git_sources'].items():
  if sha(ROOT/relative)!=digest or hashlib.sha256(git('show',freeze+':'+relative)).hexdigest()!=digest:raise ValueError('Frozen selected scientific source differs')
 return p


class Engine:
 def __init__(self,mj,p,raw):
  self.mj=mj;self.p=p;self.raw=raw;self.counts={k:0 for k in core().COUNTS};self.start=time.monotonic();self.original={n:getattr(mj,n) for n in ('MjModel','MjData','mj_forward','mj_step','mj_resetData','mj_contactForce')};self.forbidden=dict(core().FORBIDDEN);self.active=None;self.notes=[];self.static_gate=False
  def blocked(*a,**kw):self.forbidden['unregistered_engine_calls']+=1;raise RuntimeError('Unregistered native engine operation')
  names={n for n in dir(mj) if n.startswith('mj_') and callable(getattr(mj,n)) and n not in ('mj_name2id','mj_version','mj_versionString')}|{'MjModel','MjData','MjSpec','set_mjcb_control','set_mjcb_passive'}
  for n in names:
   if hasattr(mj,n):setattr(mj,n,blocked)
 def charge(self,name,n=1):
  if time.monotonic()-self.start>self.p['budget']['maximum_seconds'] or self.counts[name]+n>self.p['budget'][name]:raise RuntimeError('Before-work hard quota: '+name)
  self.counts[name]+=n
 def note(self,message):self.notes.append(dict(message=message,counts=dict(self.counts)));print(json.dumps(self.notes[-1]),flush=True)
 def resolve(self,m,kind,name):
  value=self.mj.mj_name2id(m,getattr(self.mj.mjtObj,'mjOBJ_'+kind.upper()),name)
  if value<0:raise ValueError('Missing literal compiled name '+name)
  return int(value)
 def model(self,xml):self.charge('models');return self.original['MjModel'].from_xml_string(xml)
 def data(self,m):self.charge('data');return self.original['MjData'](m)
 def reset(self,m,d):self.charge('resets');self.original['mj_resetData'](m,d);self.check(d)
 def check(self,d):
  for name in ('qpos','qvel','qacc','ctrl','qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_constraint'):
   if not np.isfinite(np.asarray(getattr(d,name))).all():raise RuntimeError('Nonfinite numerical state: '+name)
  if not np.isfinite(d.time):raise RuntimeError('Nonfinite numerical clock')
  if np.any(np.asarray(d.qfrc_applied)) or np.any(np.asarray(d.xfrc_applied)):raise RuntimeError('Unplanned generalized/body applied load')
  if np.any(np.asarray(d.warning.number).copy()):raise RuntimeError('MuJoCo warning counter nonzero')
 def forward(self,m,d):
  self.charge('static_forwards');self.check(d)
  with warnings.catch_warnings(record=True) as w:warnings.simplefilter('always');self.original['mj_forward'](m,d)
  self.check(d)
  if w:raise RuntimeError('Python warning during static forward')
 def actual_chart(self,q):
  try:return core().modules().api.principal_ball_coordinates(np.asarray(q)[3:67].reshape(16,4))[0]
  except ValueError as error:
   if str(error)=='Principal ball actuator chart near pi; declared branch refusal':raise core().QuarantineError('ACTUAL_PRINCIPAL_CHART',str(error)) from error
   raise
 def step(self,m,d):
  if not self.static_gate:raise RuntimeError('All135 static gates must be evaluated before any step')
  self.check(d);previous_q=np.asarray(d.qpos).copy();previous_chart=self.actual_chart(previous_q);before=float(d.time);ctrl=np.asarray(d.ctrl).copy();counts=np.asarray(d.warning.number).copy();self.charge('steps')
  with warnings.catch_warnings(record=True) as w:warnings.simplefilter('always');self.original['mj_step'](m,d)
  try:
   self.check(d)
   if not np.array_equal(ctrl,d.ctrl):raise RuntimeError('Engine changed controller-owned controls')
   if w or not np.array_equal(counts,np.asarray(d.warning.number)) or abs(float(d.time)-before-float(m.opt.timestep))>1e-12:raise RuntimeError('Step warning or unexpected clock increment')
   core().modules().m.rotation(np.asarray(d.qpos)[70:74])
   for q in np.asarray(d.qpos)[3:67].reshape(16,4):core().modules().m.rotation(q)
   current_chart=self.actual_chart(d.qpos)
   if np.any(np.linalg.norm(current_chart-previous_chart,axis=1)>=np.pi):raise core().QuarantineError('ACTUAL_PRINCIPAL_CHART','Actual principal actuator branch jump refused')
  except Exception:
   name='numerical_failure_'+str(self.counts['steps'])+'.npz';self.raw.arrays(self.raw.root/name,{**{k:np.asarray(getattr(d,k)).copy() for k in ('qpos','qvel','qacc','ctrl','time')},'preceding_qpos':previous_q,'preceding_time_s':np.asarray(before),'warning_counters':np.asarray(d.warning.number).copy()},'failed_numeric_state_uninterpreted');raise
 def force(self,m,d,index):self.charge('contact_force_reads');out=np.empty(6);self.original['mj_contactForce'](m,d,index,out);return out


class ImportGuard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in BLOCKED:raise RuntimeError('Forbidden source-model/inference/training import '+fullname)


def worker(protocol_path,freeze,raw_root):
 p=validate(protocol_path,freeze);initial_protocol_sha=sha(protocol_path);bootstrap();sys.meta_path.insert(0,ImportGuard());raw_root=Path(raw_root)
 if raw_root!=RAW_ROOT or raw_root.exists():raise ValueError('Fresh exact ignored raw root required')
 raw_root.mkdir(parents=True);raw=core().Artifacts(raw_root);e=None;pilot=None;failure=None;measurement=None
 try:
  import mujoco
  if mujoco.__version__!='3.4.0' or not Path(mujoco.__file__).resolve().is_relative_to(RUNTIME):raise ValueError('Pinned isolated engine path required')
  e=Engine(mujoco,p,raw);pilot=core().Pilot(p,raw,e);measurement=pilot.run();raw.verify();validate(protocol_path,freeze)
 except BaseException as exc:
  failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());raw.json(raw.root/'failure.json',failure,'runtime_failure')
 results=[] if pilot is None else pilot.results;completed={r['attempt_id'] for r in results};active=None if pilot is None else pilot.active
 ledger=[next((r for r in results if r['attempt_id']==row['attempt_id']),dict(attempt_id=row['attempt_id'],status='HARD_STOPPED_PARTIAL' if row['attempt_id']==active else 'NOT_RUN',outcome=None)) for row in p['schedule']]
 result=dict(schema_version=SCHEMA+'_worker',protocol_sha256=initial_protocol_sha,freeze_commit=freeze,status='COMPLETED' if failure is None else 'HARD_STOP',failure=failure,measurement=measurement,attempt_schedule=ledger,actual_counts={} if e is None else e.counts,forbidden={} if e is None else e.forbidden,artifacts=raw.rows,all135_static_before_task=bool(pilot and len(pilot.prepared)==135),physical_outcome=None,**core().FLAGS)
 write(raw.root.parent/'worker_result.json',result);return result


def run(protocol_path,freeze):
 p=validate(protocol_path,freeze);initial_protocol_sha=sha(protocol_path);output=HERE/'run_v1'
 if output.exists() or RAW_ROOT.parent.exists():raise ValueError('Fresh exact output roots required; no retries')
 output.mkdir();write(output/'start.json',dict(protocol=bind(protocol_path),freeze_commit=freeze,status='STARTED',**core().FLAGS));failure=None;returncode=None;worker_result=None;start=time.monotonic()
 try:
  command=[sys.executable,str(HERE/'run_pilot.py'),'_worker','--protocol',str(protocol_path),'--freeze-commit',freeze,'--raw-root',str(RAW_ROOT)]
  with (output/'worker.log').open('xb') as log:
   process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
   try:returncode=process.wait(timeout=BUDGET['outer_seconds'])
   except subprocess.TimeoutExpired:process.kill();process.wait();raise RuntimeError('Outer1830s worker timeout')
  worker_result=read(RAW_ROOT.parent/'worker_result.json')
  if worker_result.get('protocol_sha256')!=initial_protocol_sha or worker_result.get('freeze_commit')!=freeze:raise ValueError('Child frozen transaction identity differs')
  for row in worker_result['artifacts']:verify_row(row)
  if any(worker_result['forbidden'].values()) or set(worker_result['forbidden'])!=set(core().FORBIDDEN):raise ValueError('Forbidden operation inventory mismatch')
  if returncode!=0 or worker_result['status']!='COMPLETED':raise RuntimeError('Worker hard stop: '+str(worker_result.get('failure')))
  ledger=worker_result['attempt_schedule']
  if len(ledger)!=135 or len({r['attempt_id'] for r in ledger})!=135 or {r['attempt_id'] for r in ledger}!={r['attempt_id'] for r in p['schedule']} or any(r['status'] not in ('COMPLETED','MAPPING_REFUSAL','RUNTIME_QUARANTINE') for r in ledger) or worker_result['all135_static_before_task'] is not True:raise ValueError('Completed135 slot denominator/static inventory differs')
  validate(protocol_path,freeze)
 except BaseException as exc:failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
 report=dict(schema_version=SCHEMA+'_report',status='COMPLETED' if failure is None else 'HARD_STOP',protocol_sha256=initial_protocol_sha,freeze_commit=freeze,returncode=returncode,failure=failure,actual_counts=None if worker_result is None else worker_result['actual_counts'],attempt_schedule=None if worker_result is None else worker_result['attempt_schedule'],worker_result=bind(RAW_ROOT.parent/'worker_result.json') if (RAW_ROOT.parent/'worker_result.json').is_file() else None,raw_artifacts=[] if worker_result is None else worker_result['artifacts'],creation_time_journal=bind(RAW_ROOT.parent/'artifact_index.jsonl') if (RAW_ROOT.parent/'artifact_index.jsonl').is_file() else None,elapsed_s=time.monotonic()-start,training_gate='CLOSED_PENDING_INDEPENDENT_NATIVE_TASK_QUALIFICATION',physical_outcome=None,**core().FLAGS)
 write(output/'report.json',report);print(json.dumps(dict(status=report['status'],report=bind(output/'report.json'))));return report


def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True);p=sub.add_parser('prepare')
 for name in ('generation-readback','physics-protocol','physics-report','physics-readback'):p.add_argument('--'+name,type=Path,required=True)
 for name in ('generation-readback-sha256','physics-readback-sha256'):p.add_argument('--'+name,required=True)
 for name in ('run','_worker'):
  q=sub.add_parser(name);q.add_argument('--protocol',type=Path,default=HERE/'protocol.json');q.add_argument('--freeze-commit',required=True)
  if name=='_worker':q.add_argument('--raw-root',type=Path,required=True)
 args=ap.parse_args()
 if args.command=='prepare':print(json.dumps(prepare(args)));return 0
 if args.command=='_worker':return 0 if worker(args.protocol,args.freeze_commit,args.raw_root)['status']=='COMPLETED' else 1
 return 0 if run(args.protocol,args.freeze_commit)['status']=='COMPLETED' else 1
if __name__=='__main__':raise SystemExit(main())
