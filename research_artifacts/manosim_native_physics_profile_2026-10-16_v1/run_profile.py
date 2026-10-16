"""Prospective12-fixture compensated profile; all declared static gates before steps."""
from pathlib import Path
import argparse,collections,functools,hashlib,importlib.abc,importlib.util,json,subprocess,sys,time,traceback,warnings
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
RUNTIME=ROOT/'runs/manosim_mujoco_3_4_0_runtime_v1';sys.path.insert(0,str(RUNTIME))
if 'mujoco' in sys.modules and (sys.modules['mujoco'].__version__!='3.4.0' or not Path(sys.modules['mujoco'].__file__).resolve().is_relative_to(RUNTIME)):raise RuntimeError('Wrong preloaded engine')
import numpy as np
if np.__version__!='2.2.6':raise RuntimeError('Unchanged NumPy2.2.6 required')
SOURCE=ROOT/'research_artifacts/manosim_native_source_pipeline_2026-10-15_v1';BUILD=ROOT/'research_artifacts/manosim_native_transfer_2026-10-15_v2/build_run_v1'
API=ROOT/'src/grasp_failure_prediction/evaluation/manosim_capsule.py'
INERTIA_SOURCE=ROOT/'research_artifacts/manosim_capsule_inertia_representation_2026-10-15_v1'
SCHEMA='manosim_native_compensated_physics_v1'
SOURCE_HASH='92b5f9bc2552b7a56cc0698aa981ae565f1701d665395095ae55ec6558c2ec9b';SOURCE_RECEIPT='9a5d3a551a29d45049aa360dd1012fab86c32a97f45b38c4477be77b32d5a13e'
BUDGET=dict(models=12,data=12,static_forwards=240,steps=9750,contact_force_reads=4500,maximum_seconds=600,outer_seconds=610)
PRIOR=ROOT/'research_artifacts/manosim_capsule_execution_2026-10-15_v1'
PRIOR_FREEZE='a4b447e1c0467bf877798dd64b090e8ad11af301'
PRIOR_READBACK_SHA='da0dc34c722d960adbcdbdbdad60fdb12b58cd8cc88780ac98580df59b05c51a'
PRIOR_V2=ROOT/'research_artifacts/manosim_capsule_execution_2026-10-15_v2'
PRIOR_V2_FREEZE='6830aefd3da36a6b19acc2ad922ae6ab61088409'
PRIOR_V2_READBACK_SHA='3c6683676ebd6ada331e7135fe07a301e5904876363e4a069bfb3e0e5d3fd36b'
WORLD_FRAME_SOURCE=ROOT/'research_artifacts/manosim_capsule_world_plane_frame_2026-10-15_v1'
BLOCKED=('torch','manotorch','manopth','smplx','hug','transformers','grasp_failure_prediction')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 def reject(v):raise ValueError('Nonfinite JSON')
 def pairs(rows):
  d={}
  for k,v in rows:
   if k in d:raise ValueError('Duplicate JSON key')
   d[k]=v
  return d
 value=json.loads(Path(p).read_text(),parse_constant=reject,object_pairs_hook=pairs)
 def finite(v):
  if isinstance(v,float) and not np.isfinite(v):raise ValueError('Nonfinite JSON float')
  if isinstance(v,dict):
   for x in v.values():finite(x)
  if isinstance(v,list):
   for x in v:finite(x)
 finite(value);return value
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
@functools.cache
def core():return load('native_compensated_physics_case',HERE/'physics_case.py')
def binding(p):
 p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def verify(files):
 for path,digest in files.items():
  if sha(path)!=digest:raise ValueError('Active required input changed: '+path)
def verify_row(row):
 if binding(row['path'])!={k:row[k] for k in ('path','sha256','bytes')}:raise ValueError('Write-time artifact changed')
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])

def runtime_files():
 import importlib.metadata
 files={Path(sys.executable).resolve(),Path('/Users/nilakarthikesan/grasp-failure-prediction/.venv/pyvenv.cfg')}
 for name in ('numpy','mujoco'):
  distribution=importlib.metadata.distribution(name)
  for f in distribution.files or []:
   p=Path(distribution.locate_file(f)).resolve()
   if p.is_file() and '__pycache__' not in str(p) and (p.suffix in ('.py','.so','.dylib','.h') or '.dist-info/' in str(p)):files.add(p)
 return sorted(files)


def archived_closure(p,freeze):
 """Old scientific source is immutable Git evidence; unchanged data stays live."""
 selected=p['selected_git_sources'];bindings=p['active_bindings'];data={}
 for relative,digest in selected.items():
  if bindings.get(str(ROOT/relative))!=digest or hashlib.sha256(git('show',freeze+':'+relative)).hexdigest()!=digest:raise ValueError('Archived selected source Git identity changed')
 for path,digest in bindings.items():
  relative=str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else None
  if relative not in selected:data[path]=digest
 verify(data);return data

def prior_failure_inputs():
 ip=PRIOR/'run_v1/independent_readback.json';sp=PRIOR/'protocol.json';rp=PRIOR/'run_v1/report.json'
 if sha(ip)!=PRIOR_READBACK_SHA:raise ValueError('Exact independent v1 failure receipt required')
 d=read(ip);p=read(sp);r=read(rp)
 if (d.get('schema_version')!='manosim_capsule_execution_independent_saved_readback_v1' or d.get('status')!='VERIFIED_HARD_STOP_CAPSULE_SAVED_READBACK'
  or d.get('freeze_commit')!=PRIOR_FREEZE or d.get('actual_operation_counts')!=dict(models=1,data=0,static_forwards=0,steps=0,contact_force_reads=0)
  or d.get('protocol')!=binding(sp) or d.get('report')!=binding(rp) or r.get('status')!='HARD_STOP' or r.get('qualification')!='NOT_QUALIFIED'
  or r.get('physical_outcome') is not None or d.get('physical_outcome') is not None or any(r.get(k) is not False for k in core().math.FLAGS)
  or any(d.get('operations',{}).values()) or not d.get('operations') or r.get('freeze_commit')!=PRIOR_FREEZE
  or git('show',PRIOR_FREEZE+':'+str(sp.relative_to(ROOT)))!=sp.read_bytes()):raise ValueError('Honest frozen v1 failure lineage required')
 files=archived_closure(p,PRIOR_FREEZE)
 for row in [r['worker_result'],r['raw_result'],*r['partial_files']]:verify_row(row);files[row['path']]=row['sha256']
 worker=read(r['worker_result']['path'])
 if worker.get('failure',{}).get('message')!='Authored compiled inertia tensor differs' or worker.get('actual_counts')!=d['actual_operation_counts']:raise ValueError('Original compiler-validation failure changed')
 for row in worker['artifacts']:verify_row(row);files[row['path']]=row['sha256']
 for path in (ip,sp,rp):files[str(path.resolve())]=sha(path)
 evidence=dict(independent_receipt=binding(ip),protocol=binding(sp),report=binding(rp),freeze_commit=PRIOR_FREEZE,
  status='VERIFIED_HARD_STOP_CAPSULE_SAVED_READBACK',actual_counts=d['actual_operation_counts'],prior_active_bindings=len(p['active_bindings']),
  historical_selected_source_blobs_verified=len(p['selected_git_sources']),unchanged_non_source_live_bindings=len(archived_closure(p,PRIOR_FREEZE)),
  prior_sources_live_required=False,original_failure=worker['failure']['message'],no_v1_pose_or_contact_pass=True)
 return files,evidence



def prior_v2_failure_inputs():
 ip=PRIOR_V2/'run_v1/independent_readback.json';sp=PRIOR_V2/'protocol.json';rp=PRIOR_V2/'run_v1/report.json'
 if sha(ip)!=PRIOR_V2_READBACK_SHA:raise ValueError('Exact independent v2 failure receipt required')
 d=read(ip);p=read(sp);r=read(rp);expected=dict(models=12,data=12,static_forwards=26,steps=1501,contact_force_reads=0)
 if (d.get('schema_version')!='manosim_capsule_execution_independent_saved_readback_v2' or d.get('status')!='VERIFIED_HARD_STOP_CAPSULE_SAVED_READBACK'
  or d.get('freeze_commit')!=PRIOR_V2_FREEZE or d.get('actual_operation_counts')!=expected or d.get('protocol')!=binding(sp) or d.get('report')!=binding(rp)
  or r.get('status')!='HARD_STOP' or r.get('qualification')!='NOT_QUALIFIED' or r.get('actual_counts')!=expected
  or r.get('physical_outcome') is not None or d.get('physical_outcome') is not None or any(r.get(k) is not False for k in core().math.FLAGS)
  or any(d.get('operations',{}).values()) or not d.get('operations') or r.get('freeze_commit')!=PRIOR_V2_FREEZE
  or git('show',PRIOR_V2_FREEZE+':'+str(sp.relative_to(ROOT)))!=sp.read_bytes()):raise ValueError('Honest frozen v2 failure lineage required')
 files=archived_closure(p,PRIOR_V2_FREEZE)
 for row in [r['worker_result'],r['raw_result'],*r['partial_files']]:verify_row(row);files[row['path']]=row['sha256']
 worker=read(r['worker_result']['path'])
 if worker.get('failure',{}).get('message')!='Declared5mm maximum contact penetration escaped' or worker.get('actual_counts')!=expected or worker.get('all12_static_checks_before_dynamics') is not True:raise ValueError('Original first-contact failure changed')
 for row in worker['artifacts']:verify_row(row);files[row['path']]=row['sha256']
 for path in (ip,sp,rp):files[str(path.resolve())]=sha(path)
 evidence=dict(independent_receipt=binding(ip),protocol=binding(sp),report=binding(rp),freeze_commit=PRIOR_V2_FREEZE,
  actual_counts=expected,historical_selected_source_blobs_verified=len(p['selected_git_sources']),original_failure=worker['failure']['message'],
  all12_source_mapping_checks_passed_before_steps=True,prior_contact_qualification=False,actual_failed_plane_frame_or_cache_available=False)
 return files,evidence


COMPLETED_V3=ROOT/'research_artifacts/manosim_capsule_execution_2026-10-15_v3'
COMPLETED_V3_FREEZE='9e7a0a72e338c0a493579e57c880bd7d0ae3bb66'
COMPLETED_V3_RECEIPT_SHA='42135771d8d8128e104f28effd9df18175054832a7828d983a631c1734bde7fd'

def completed_v3_inputs():
 ip=COMPLETED_V3/'run_v1/independent_readback.json';sp=COMPLETED_V3/'protocol.json';rp=COMPLETED_V3/'run_v1/report.json'
 if sha(ip)!=COMPLETED_V3_RECEIPT_SHA:raise ValueError('Exact independently completed capsule baseline required')
 d=read(ip);p=read(sp);r=read(rp)
 if (d.get('schema_version')!='manosim_capsule_execution_independent_saved_readback_v3' or d.get('status')!='VERIFIED_COMPLETED_BOUNDED_CAPSULE_SAVED_READBACK'
  or d.get('freeze_commit')!=COMPLETED_V3_FREEZE or d.get('protocol')!=binding(sp) or d.get('report')!=binding(rp)
  or r.get('status')!='COMPLETED' or r.get('qualification')!='PASS' or r.get('freeze_commit')!=COMPLETED_V3_FREEZE
  or r.get('actual_counts')!=dict(models=12,data=12,static_forwards=36,steps=10500,contact_force_reads=1492)
  or len(r.get('cases',[]))!=2 or any(x.get('qualification')!='PASS' for x in r['cases'])
  or git('show',COMPLETED_V3_FREEZE+':'+str(sp.relative_to(ROOT)))!=sp.read_bytes()):raise ValueError('Original uncompensated capsule baseline identity changed')
 files=archived_closure(p,COMPLETED_V3_FREEZE)
 for row in [r['worker_result'],r['raw_result']]:verify_row(row);files[row['path']]=row['sha256']
 worker=read(r['worker_result']['path'])
 for row in worker['artifacts']:verify_row(row);files[row['path']]=row['sha256']
 for path in (COMPLETED_V3/'capsule_case.py',COMPLETED_V3/'execution_math.py'):
  relative=str(path.relative_to(ROOT))
  if p['selected_git_sources'].get(relative)!=sha(path):raise ValueError('Executed imported baseline helper differs from frozen Git source')
 for path in (ip,sp,rp,COMPLETED_V3/'capsule_case.py',COMPLETED_V3/'execution_math.py'):
  files[str(path.resolve())]=sha(path)
 return files,dict(independent_receipt=binding(ip),protocol=binding(sp),report=binding(rp),freeze_commit=COMPLETED_V3_FREEZE,uncompensated_baseline_preserved=True,new_profile_does_not_inherit_task_admission=True)


def active_inputs(durable_path,durable_sha):
 durable_path=Path(durable_path).resolve()
 if sha(durable_path)!=durable_sha:raise ValueError('Explicit durable saved-readback hash required')
 d=read(durable_path);sp=SOURCE/'protocol.json';ip=SOURCE/'run_v1/independent_readback.json';rp=SOURCE/'run_v1/report.json'
 if (sha(sp)!=SOURCE_HASH or sha(ip)!=SOURCE_RECEIPT or d.get('schema_version')!='manosim_reconstruction_durable_readback_v1' or d.get('status')!='PASS'
  or d.get('prior_independent_receipt_sha256')!=SOURCE_RECEIPT or d.get('prior_protocol_sha256')!=SOURCE_HASH
  or d.get('prior_full_dependency_closure_live_reverified') is not False or d.get('collection_or_training_allowed') is not False
  or set(d.get('operations',{}))!={'native_model_constructions','MANO_or_skin_evaluations','FK_or_collision_queries','physics_steps'} or any(d['operations'].values())
  or [c['case'] for c in d.get('cases',[])]!=[20,28] or any(c.get('all_checks_pass') is not True for c in d['cases'])):raise ValueError('Completed durable reconstruction evidence required')
 source=read(sp);independent=read(ip);report=read(rp)
 if independent.get('report_sha256')!=sha(rp) or independent.get('status')!='VERIFIED_COMPLETED_SOURCE_CAMERA_PIPELINE_GEOMETRY_ONLY' or report.get('status')!='COMPLETED':raise ValueError('Original completed source identity changed')
 bindings={};selected={};specs={}
 def add(path,digest=None):
  path=Path(path).resolve()
  if not path.is_file() or path.is_symlink() or (digest and sha(path)!=digest):raise ValueError('Required durable input absent/changed')
  bindings[str(path)]=sha(path)
 for path,row in d['verified_durable_artifacts'].items():
  if sha(path)!=row['sha256'] or Path(path).stat().st_size!=row['bytes']:raise ValueError('Durable reconstruction artifact changed')
  add(path,row['sha256'])
 if sha(durable_path.parent/'audit.py')!=d.get('verifier_source_sha256'):raise ValueError('Durable verifier source version changed')
 for path in (sp,ip,rp,durable_path,durable_path.parent/'audit.py',BUILD/'builder_report.json',BUILD/'independent_readback.json'):add(path)
 producer=read(BUILD/'builder_report.json');builder_independent=read(BUILD/'independent_readback.json')
 if builder_independent.get('status')!='VERIFIED_LITERAL_SHAPE_BUILD_ARTIFACTS_AND_PRODUCER' or builder_independent.get('builder_report_sha256')!=sha(BUILD/'builder_report.json'):raise ValueError('Independent exact-shape build receipt required')
 root=Path(producer['generated_root']);shape=root/'assets/mano_rhand_shape.npy';capsule=root/'assets/literal_native/mano_rhand/capsule_hand.xml'
 api=load('authored_capsule_mapping',API)
 inertia_source_files=[INERTIA_SOURCE/name for name in ('mujoco_3_4_0_user_util.cc','source_binding.json','README.md')]
 inertia_source=read(INERTIA_SOURCE/'source_binding.json')
 if inertia_source.get('schema_version')!='manosim_capsule_inertia_representation_source_v1' or inertia_source.get('source_sha256')!=api.INERTIA_SOURCE_SHA256 or sha(inertia_source_files[0])!=api.INERTIA_SOURCE_SHA256 or inertia_source.get('source_inertia_changes') is not False or inertia_source.get('physical_gate_changes') is not False:raise ValueError('Pinned inertia compiler representation source required')
 for path in inertia_source_files:add(path)
 world_binding=read(WORLD_FRAME_SOURCE/'source_binding.json');world_source_files=[WORLD_FRAME_SOURCE/'source_binding.json',WORLD_FRAME_SOURCE/'README.md']
 if world_binding.get('schema_version')!='manosim_capsule_world_plane_frame_source_v1' or world_binding.get('sameframe_NONE')!=0 or world_binding.get('sameframe_BODYROT')!=3 or world_binding.get('physical_gate_changes') is not False:raise ValueError('Pinned world-plane cache source required')
 for row in world_binding['sources']:
  path=WORLD_FRAME_SOURCE/row['path']
  if path.parent!=WORLD_FRAME_SOURCE or sha(path)!=row['sha256']:raise ValueError('Pinned plane cache source bytes changed')
  world_source_files.append(path)
 for path in world_source_files:add(path)
 for index in (20,28):
  row=next(c for c in report['cases'] if c['index']==index);v=read(row['result'])
  if v.get('source_pipeline_geometry_check_passed') is not True or any(x is not True for x in v.get('checks',{}).values()):raise ValueError('Source geometry check incomplete')
  artifacts={Path(a['path']).name:a for a in v['artifacts']};b=artifacts['literal_source_buffers.npz'];r=artifacts['all109_native_source_engine_readbacks.npz']
  for a in (b,r):verify_row(a);add(a['path'],a['sha256'])
  with np.load(b['path'],allow_pickle=False) as file:beta=file['shape'].copy()
  assets=api.inspect_authored_assets(capsule,shape,dict(path=str(BUILD/'builder_report.json'),sha256=sha(BUILD/'builder_report.json')),expected_shape_sha256=producer['shape_asset_sha256'],native_shape_float32=beta)
  for path,digest in assets['files'].items():add(path,digest)
  specs[str(index)]=dict(literal_buffers=b,source_readbacks=r,capsule_xml=str(capsule),shape_asset=str(shape),expected_shape_sha256=producer['shape_asset_sha256'],builder_report=dict(path=str(BUILD/'builder_report.json'),sha256=sha(BUILD/'builder_report.json')),asset_closure_sha256=assets['closure_sha256'])
 completed_files,completed_evidence=completed_v3_inputs()
 for path,digest in completed_files.items():add(path,digest)
 prior_files,prior_evidence=prior_failure_inputs()
 for path,digest in prior_files.items():add(path,digest)
 prior_v2_files,prior_v2_evidence=prior_v2_failure_inputs()
 for path,digest in prior_v2_files.items():add(path,digest)
 for path in [API,SOURCE/'source_coordinates.py',ROOT/'src/grasp_failure_prediction/evaluation/manosim_pose.py',ROOT/'tests/test_manosim_capsule.py',*runtime_files()]:add(path)
 source_binding=read(HERE/'primary_sources/source_binding.json')
 if source_binding.get('revision')!='e55fff5dea6f1d5dd7963ca52eecc41d05ad0922' or source_binding.get('velocity_bias_compensated') is not False:raise ValueError('Pinned passive gravity source declaration required')
 for row in source_binding['sources']:
  path=HERE/'primary_sources'/row['path']
  if path.parent!=HERE/'primary_sources' or sha(path)!=row['sha256']:raise ValueError('Pinned gravity source changed')
 for path in sorted((HERE/'primary_sources').iterdir()):add(path)
 own=[p for p in HERE.iterdir() if p.is_file() and (p.suffix=='.py' or p.name in ('README.md','controls.json','manufactured_control_lineage.json','controls_before_positive_tail_guard.json','protocol_unused_prepare_v1.json'))]
 for p in own:add(p)
 tracked=set(git('ls-files').decode().splitlines())
 for path,digest in bindings.items():
  p=Path(path)
  if p.is_relative_to(ROOT):
   relative=str(p.relative_to(ROOT))
   if relative in tracked:selected[relative]=digest
 for path in [API,SOURCE/'source_coordinates.py',ROOT/'src/grasp_failure_prediction/evaluation/manosim_pose.py',ROOT/'tests/test_manosim_capsule.py',*inertia_source_files,*world_source_files,*sorted((HERE/'primary_sources').iterdir()),*own]:selected[str(path.relative_to(ROOT))]=sha(path)
 evidence=dict(completed_uncompensated_v3=completed_evidence,prior_v2_failure=prior_v2_evidence,world_plane_frame_source=[binding(path) for path in world_source_files],inertia_representation_source=[binding(path) for path in inertia_source_files],prior_failure=prior_evidence,durable_report=dict(path=str(durable_path),sha256=durable_sha),source_protocol_sha256=SOURCE_HASH,original_independent_receipt_sha256=SOURCE_RECEIPT,historical_full_dependency_closure_live_reverified=False,historical_missing_temp_paths_at_durable_audit=d['prior_dependency_paths_currently_missing'])
 return bindings,selected,specs,evidence


def method():
 c=core();return dict(profile_id=c.p.PROFILE_ID,plan=c.fixture_plan(),budget=BUDGET,per_case_budget=c.BUDGET,limits=c.LIMITS,flags=c.p.FLAGS,
  gravity_compensation=dict(body_gravcomp=1,jnt_actgravcomp=0,only16hand_bodies=True,qfrc_bias_feedforward=False,coriolis_compensated=False,external_forces=False),
  approach=dict(knot_count=c.PATH_KNOTS,root_start_shift_m=c.PATH_SHIFT.tolist(),sphere_center_m=c.SPHERE_CENTER.tolist(),sphere_radius_m=c.SPHERE_RADIUS,table_z_m=c.TABLE_Z,clearance_m=c.PATH_CLEARANCE,oracle_error_m=c.ORACLE_ERROR,reference_swept_bound=True,actual_trajectory_scope='once-per-step clearance and engine contact checks, no continuous servo-trajectory certificate'),
  multifinger=dict(native_bone_ids=[3,6],separate_analytic_planes=True,depth_each_m=.0005,maximum_root_move_m=.002,maximum_normal_condition=1e6,simultaneous_tail_s=.1),
  all12_static_checks_before_first_step=True,source_inertias_gains_commands_unchanged=True,new_task_collection_or_training_admission=False)
def prepare(protocol,durable,durable_sha):
 protocol=Path(protocol).resolve()
 if protocol!=HERE/'protocol.json' or protocol.exists() or (HERE/'run_v1').exists():raise ValueError('Fresh exact preparation only')
 controls=read(HERE/'controls.json')
 if controls.get('exit_code')!=0 or controls.get('integration_calls')!=0:raise ValueError('No-integration controls required')
 bindings,selected,specs,evidence=active_inputs(durable,durable_sha);c=core();xml={}
 for row in c.math.fixture_plan():
  key=str(row['index'])+'/'+row['kind']+'_'+str(round(1/row['dt_s']));xml[key]=hashlib.sha256(c.scene_xml(specs[str(row['index'])]['capsule_xml'],row['kind'],row['dt_s']).encode()).hexdigest()
 p=dict(schema_version=SCHEMA,protocol_path=str(protocol),active_bindings=bindings,selected_git_sources=selected,cases=specs,evidence=evidence,method=method(),fixture_xml_sha256=xml,raw_root=str(ROOT/'runs'/HERE.name/'run_v1/raw'),output_root=str(HERE/'run_v1'),physical_outcome=None,**c.math.FLAGS)
 verify(bindings);write(protocol,p);return p

def validate(protocol,freeze):
 p=read(protocol);bindings,selected,specs,evidence=active_inputs(p['evidence']['durable_report']['path'],p['evidence']['durable_report']['sha256'])
 if p.get('schema_version')!=SCHEMA or p.get('protocol_path')!=str(Path(protocol).resolve()) or p['active_bindings']!=bindings or p['selected_git_sources']!=selected or p['cases']!=specs or p['evidence']!=evidence or p['method']!=method() or p['physical_outcome'] is not None or any(p.get(k) is not False for k in core().math.FLAGS):raise ValueError('Frozen capsule method/active closure changed')
 if p['raw_root']!=str(ROOT/'runs'/HERE.name/'run_v1/raw') or p['output_root']!=str(HERE/'run_v1'):raise ValueError('Frozen once-only roots changed')
 expected={str(row['index'])+'/'+row['kind']+'_'+str(round(1/row['dt_s'])):hashlib.sha256(core().scene_xml(specs[str(row['index'])]['capsule_xml'],row['kind'],row['dt_s']).encode()).hexdigest() for row in core().math.fixture_plan()}
 if p['fixture_xml_sha256']!=expected:raise ValueError('Declared fixture XML changed')
 verify(bindings)
 if git('rev-parse','HEAD').decode().strip()!=freeze:raise ValueError('Exact HEAD hold required')
 for rel,digest in selected.items():
  if hashlib.sha256(git('show',freeze+':'+rel)).hexdigest()!=digest:raise ValueError('Required source not cleanly frozen')
 if git('show',freeze+':'+str(Path(protocol).relative_to(ROOT)))!=Path(protocol).read_bytes():raise ValueError('Protocol not frozen')
 return p

class ImportGuard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in BLOCKED:raise RuntimeError('Forbidden model/inference/training import '+fullname)

class Engine:
 def __init__(self,mj,p,notify):
  self.mj=mj;self.p=p;self.notify=notify;self.counts={k:0 for k in ('models','data','static_forwards','steps','contact_force_reads')};self.per_case={i:{k:0 for k in self.counts} for i in (20,28)};self.models={};self.pairs={};self.saved={};self.index=None;self.started=time.monotonic();self.execution_starts={};self.static_gate=False;self.forbidden=[]
 def deny(self,name):self.forbidden.append(name);raise RuntimeError('Unregistered/forbidden engine operation '+name)
 def charge(self,key):
  if time.monotonic()-self.started>BUDGET['maximum_seconds']:raise TimeoutError('600s overall worker bound')
  if self.index not in self.per_case or self.counts[key]>=BUDGET[key] or self.per_case[self.index][key]>=core().BUDGET[key]:return self.deny('quota_'+key)
  if key=='steps':
   if not self.static_gate:return self.deny('step_before_all12_static_gates')
   start=self.execution_starts.setdefault(self.index,time.monotonic())
   if time.monotonic()-start>300:raise TimeoutError('300s per-case dynamics bound')
  self.counts[key]+=1;self.per_case[self.index][key]+=1;self.notify(dict(event='engine_operation',counts=self.counts.copy(),per_case=self.per_case))
 def model(self,path):
  path=Path(path);key=str(path.parent.relative_to(Path(self.p['raw_root'])))
  if self.p['fixture_xml_sha256'].get(key)!=sha(path):return self.deny('unplanned_XML')
  self.charge('models')
  with warnings.catch_warnings(record=True) as emitted:
   warnings.simplefilter('always');m=self.saved['MjModel'].from_xml_path(str(path))
  if emitted:raise RuntimeError('Compiler warning')
  self.models[id(m)]=self.index;return m
 def data(self,m):
  if id(m) not in self.models or id(m) in self.pairs:return self.deny('extra_or_unknown_data')
  self.charge('data');d=self.saved['MjData'](m);self.pairs[id(m)]=(m,d);return d
 def state(self,d):
  for k in ('qpos','qvel','qacc','ctrl','qfrc_applied','xfrc_applied'):
   if not np.isfinite(getattr(d,k)).all():raise RuntimeError('Nonfinite actual native state/force')
  if np.max(np.abs(np.linalg.norm(np.asarray(d.qpos[3:]).reshape(16,4),axis=1)-1))>core().math.LIMITS['raw_ball_norm']:raise RuntimeError('Raw ball quaternion norm escaped')
  if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise RuntimeError('Unplanned external force/gravity feedforward')
 def operation(self,key,m,d):
  if id(m) not in self.pairs or self.pairs[id(m)][1] is not d or self.models[id(m)]!=self.index:return self.deny('unregistered_model_data')
  self.state(d);clock=float(d.time);warn=d.warning.number.copy();ctrl=d.ctrl.copy();protected={k:np.asarray(getattr(d,k)).copy() for k in ('qpos','qvel')} if key=='static_forwards' else {}
  if np.any(warn):raise RuntimeError('Nonzero warning baseline')
  self.charge(key)
  with warnings.catch_warnings(record=True) as emitted:
   warnings.simplefilter('always');self.saved['mj_step' if key=='steps' else 'mj_forward'](m,d)
  self.state(d)
  if emitted or any(not np.array_equal(getattr(d,k),v) for k,v in protected.items()) or not np.array_equal(warn,d.warning.number) or not np.array_equal(ctrl,d.ctrl) or abs((d.time-clock)-(float(m.opt.timestep) if key=='steps' else 0.))>1e-12:raise RuntimeError('Numerical warning/control/clock mutation')
 def forward(self,m,d):return self.operation('static_forwards',m,d)
 def step(self,m,d):return self.operation('steps',m,d)
 def force(self,m,d,i):
  if id(m) not in self.pairs or self.pairs[id(m)][1] is not d or not 1<=d.ncon<=2 or not 0<=i<d.ncon:return self.deny('unregistered_contact_force')
  self.charge('contact_force_reads');out=np.zeros(6);self.saved['mj_contactForce'](m,d,i,out)
  if not np.isfinite(out).all():raise RuntimeError('Nonfinite engine contact force')
  return out
 def __enter__(self):
  names={k for k in dir(self.mj) if k.startswith('mj_') and callable(getattr(self.mj,k)) and k not in ('mj_name2id','mj_version','mj_versionString')};names|={'MjModel','MjData','MjSpec','set_mjcb_control','set_mjcb_passive'}
  for name in names:
   if hasattr(self.mj,name):self.saved[name]=getattr(self.mj,name);setattr(self.mj,name,lambda *a,_name=name,**k:self.deny(_name))
  return self
 def __exit__(self,*a):
  for name,value in self.saved.items():setattr(self.mj,name,value)

def verify_outputs(rows,raw,complete):
 raw=Path(raw).resolve();seen=set()
 for row in rows:
  path=Path(row['path'])
  if path.is_symlink() or not path.resolve().is_relative_to(raw) or path.resolve() in seen:raise ValueError('Output escape/alias')
  verify_row(row);seen.add(path.resolve())
 extras={p.resolve() for p in raw.rglob('*') if p.is_file()}-seen-{raw/'progress.json',raw/'result.json'}
 if any(p.is_symlink() for p in raw.rglob('*')) or (complete and extras):raise ValueError('Unindexed/symlink completed output')
 return [binding(p) for p in sorted(extras)]

def worker(protocol,freeze):
 p=validate(protocol,freeze);raw=Path(p['raw_root']);raw.mkdir(parents=True,exist_ok=False);rows=[];artifacts=[];failure=None;started=time.monotonic();engine=None;static_completed=False;contexts=[];finder=ImportGuard();sys.meta_path.insert(0,finder)
 def notify(v):
  if v.get('event')=='artifact':artifacts.append(v['artifact'])
  tmp=raw/'progress.tmp';tmp.write_text(json.dumps(dict(last=v,elapsed_s=time.monotonic()-started),allow_nan=False)+'\n');tmp.replace(raw/'progress.json')
 try:
  import mujoco
  if mujoco.__version__!='3.4.0' or not Path(mujoco.__file__).resolve().is_relative_to(RUNTIME):raise RuntimeError('Exact isolated MuJoCo3.4 required')
  if any(getattr(mujoco,k)() is not None for k in ('get_mjcb_control','get_mjcb_passive')):raise RuntimeError('Unplanned callback already installed')
  c=core()
  with Engine(mujoco,p,notify) as engine:
   for index in (20,28):
    engine.index=index;directory=raw/str(index);directory.mkdir();contexts.append(c.prepare_case(index,p['cases'][str(index)],p,directory,engine,notify))
   if any(engine.counts[k]!=v for k,v in dict(models=12,data=12,static_forwards=228,steps=0).items()) or len(contexts)!=2 or sum(len(x['prepared']) for x in contexts)!=12:raise RuntimeError('Global static-before-dynamics inventory incomplete')
   static_completed=True;engine.static_gate=True;notify(dict(event='ALL12_STATIC_GATES_PASSED_BEFORE_FIRST_STEP',steps=0))
   for context in contexts:
    engine.index=context['index'];rows.append(c.execute_case(context,engine,notify))
  if any(engine.counts[k]!=BUDGET[k] for k in ('models','data','static_forwards','steps')) or engine.forbidden:raise RuntimeError('Actual operation inventory incomplete')
 except BaseException as e:
  failure=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
  for context in contexts:
   session=context.get('current_session');partial=context.get('partial_rows',[])
   if session is None or not context.get('current_fixture'):continue
   try:
    if partial:context['save'](session['directory']/'partial_trajectory.npz','FAILED_PARTIAL_UNINTERPRETED_aligned_recorded_rows',**{k:np.asarray([x[k] for x in partial]) for k in partial[0]})
    data=session['data'];context['save'](session['directory']/'partial_live_state.npz','FAILED_PARTIAL_UNINTERPRETED_actual_failure_state_no_engine_call',**{k:np.asarray(getattr(data,k)).copy() for k in ('qpos','qvel','qacc','ctrl','qfrc_applied','xfrc_applied')},warning_counts=data.warning.number.copy(),time_s=np.asarray(data.time))
   except BaseException as persistence:failure['partial_persistence_error']=dict(type=type(persistence).__name__,message=str(persistence))
 finally:
  if finder in sys.meta_path:sys.meta_path.remove(finder)
 result=dict(schema_version=SCHEMA+'_worker',status='HARD_STOP' if failure else 'COMPLETED',protocol_sha256=sha(protocol),freeze_commit=freeze,all12_static_checks_before_dynamics=static_completed,cases=rows,artifacts=artifacts,actual_counts=engine.counts if engine else {},per_case_counts=engine.per_case if engine else {},forbidden_attempts=engine.forbidden if engine else [],failure=failure,elapsed_s=time.monotonic()-started,physical_outcome=None,**core().math.FLAGS)
 result['remaining_case_indices_NOT_RUN']=[i for i in (20,28) if engine is None or engine.per_case[i]['steps']==0]
 for context in contexts:
  if context['index'] not in [r['index'] for r in rows]:rows.append(dict(index=context['index'],status='HARD_STOPPED_PARTIAL' if context.get('current_fixture') else 'STATIC_PREFLIGHT_PASSED_DYNAMICS_NOT_RUN',measurements=context.get('measurements',[]),failed_fixture=context.get('current_fixture'),physical_outcome=None,**core().math.FLAGS))
 indexed_names={str(Path(a['path']).relative_to(raw)) for a in artifacts};schedule=[]
 for fixture in core().math.fixture_plan():
  key=str(fixture['index'])+'/'+fixture['kind']+'_'+str(round(1/fixture['dt_s']))
  complete=any(m['fixture']==fixture for case_row in rows for m in case_row.get('measurements',[]))
  state='COMPLETED_ASSESSED' if complete else ('HARD_STOPPED_PARTIAL' if key+'/partial_live_state.npz' in indexed_names else ('STATIC_PREFLIGHT_PASSED_DYNAMICS_NOT_RUN' if key+'/command_contract.json' in indexed_names else ('HARD_STOPPED_STATIC_PREFLIGHT' if key+'/fixture.xml' in indexed_names else 'NOT_RUN')))
  schedule.append(dict(fixture=fixture,status=state))
 result['fixture_schedule']=schedule
 try:validate(protocol,freeze);result['unindexed_partial_files']=verify_outputs(artifacts,raw,result['status']=='COMPLETED')
 except BaseException as e:result.update(status='HARD_STOP',final_validation_failure=dict(type=type(e).__name__,message=str(e)));result['unindexed_partial_files']=[]
 write(raw/'result.json',result);return 0 if result['status']=='COMPLETED' else 1



def check_worker(result,p,protocol_hash,freeze,returncode,raw):
 c=core()
 if result.get('schema_version')!=SCHEMA+'_worker' or result.get('protocol_sha256')!=protocol_hash or result.get('freeze_commit')!=freeze or result.get('physical_outcome') is not None or any(result.get(k) is not False for k in c.math.FLAGS):raise ValueError('Worker identity/admission changed')
 if result.get('status') not in ('COMPLETED','HARD_STOP'):raise ValueError('Unknown worker transaction status')
 verify_outputs(result.get('artifacts',[]),raw,result['status']=='COMPLETED')
 if result['status']=='HARD_STOP':
  if returncode==0:raise ValueError('Failed worker returned success')
  return
 counts=result.get('actual_counts',{})
 if returncode or result.get('failure') or result.get('final_validation_failure') or result.get('unindexed_partial_files') or result.get('forbidden_attempts') or result.get('all12_static_checks_before_dynamics') is not True or any(counts.get(k)!=BUDGET[k] for k in ('models','data','static_forwards','steps')) or not 0<=counts.get('contact_force_reads',-1)<=BUDGET['contact_force_reads']:raise ValueError('Complete operation/static inventory inconsistent')
 if [v.get('index') for v in result.get('cases',[])]!=[20,28] or len(result.get('fixture_schedule',[]))!=12 or any(r['status']!='COMPLETED_ASSESSED' for r in result['fixture_schedule']) or [r['fixture'] for r in result['fixture_schedule']]!=c.math.fixture_plan():raise ValueError('Exact2case/12fixture completion required')
 expected_artifacts={str(f['index'])+'/'+f['kind']+'_'+str(round(1/f['dt_s']))+'/'+name for f in c.math.fixture_plan() for name in ('fixture.xml','prevalidation_compiled_model.npz','static_command.npz','fixture_model.npz','command_contract.json','trajectory.npz')}
 expected_artifacts|={str(f['index'])+'/'+f['kind']+'_'+str(round(1/f['dt_s']))+'/plane_preflight.npz' for f in c.math.fixture_plan() if f['kind']=='two_finger_planes'}
 expected_artifacts|={str(f['index'])+'/'+f['kind']+'_'+str(round(1/f['dt_s']))+'/approach_path.npz' for f in c.math.fixture_plan() if f['kind']=='guarded_sphere_table_approach'}
 if len(result['artifacts'])!=80 or {str(Path(a['path']).relative_to(Path(raw))) for a in result['artifacts']}!=expected_artifacts:raise ValueError('Exact80-artifact completion inventory required')
 for row in result['cases']:
  if row.get('status')!='COMPLETED' or row.get('forbidden_counts')!=c.FORBIDDEN or row.get('physical_outcome') is not None or any(row.get(k) is not False for k in c.math.FLAGS) or row.get('qualification') not in ('PASS','NOT_QUALIFIED') or len(row.get('measurements',[]))!=6 or len(row.get('convergence',[]))!=3:raise ValueError('Completed case identity/counts/flags changed')
  if [m['fixture'] for m in row['measurements']]!=[f for f in c.math.fixture_plan() if f['index']==row['index']]:raise ValueError('Declared fixture identities changed')
  for m in row['measurements']:
   gravity=m['assessment'].get('gravity_preflight',{}).get('checks',{})
   if not gravity or any(x is not True for x in gravity.values()):raise ValueError('Completed gravity preflight incomplete')
   plane=m['assessment'].get('plane_preflight')
   if m['fixture']['kind']=='two_finger_planes' and (not isinstance(plane,dict) or len(plane)<7 or any(x is not True for x in plane.values())):raise ValueError('Completed simultaneous-plane preflight incomplete')
   if m['fixture']['kind']=='guarded_sphere_table_approach' and m['assessment'].get('approach_guard',{}).get('passed') is not True:raise ValueError('Completed approach guard absent')
   if m['assessment']['status'] not in ('PASS','FAIL') or not all(m['assessment'].get('static_source_and_transmission_checks',{}).values()) or len(m['assessment'].get('static_source_and_transmission_checks',{}))!=5:raise ValueError('Completed fixture static gate incomplete')
   gates=m['assessment'].get('checks',{})
   if not gates or any(type(v) is not bool for v in gates.values()) or (m['assessment']['status']=='PASS')!=all(gates.values()):raise ValueError('Fixture assessment/gates inconsistent')
  expected_pass=all(m['assessment']['status']=='PASS' for m in row['measurements']) and all(x.get('passed') is True for x in row['convergence'])
  if (row['qualification']=='PASS')!=expected_pass:raise ValueError('Case qualification differs from fixed assessments')


def run(protocol,freeze):
 p=validate(protocol,freeze);out=Path(p['output_root']);raw=Path(p['raw_root'])
 if out.exists() or raw.exists():raise FileExistsError('Once-only roots already exist')
 out.mkdir();write(out/'STARTED.json',dict(protocol_sha256=sha(protocol),freeze_commit=freeze,budget=BUDGET));failure=None;result=None;code=None
 try:
  with (out/'run.log').open('x') as log:child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'_worker','--protocol',str(Path(protocol).resolve()),'--freeze-commit',freeze],stdout=log,stderr=subprocess.STDOUT,timeout=BUDGET['outer_seconds'])
  code=child.returncode;rp=raw/'result.json';result=read(rp);(out/'worker_result.json').write_bytes(rp.read_bytes());validate(protocol,freeze);check_worker(result,p,sha(protocol),freeze,code,raw)
  if result.get('schema_version')!=SCHEMA+'_worker' or result.get('protocol_sha256')!=sha(protocol) or result.get('freeze_commit')!=freeze or result.get('physical_outcome') is not None or any(result.get(k) is not False for k in core().math.FLAGS):raise ValueError('Worker identity/admission changed')
  if code or result['status']!='COMPLETED':raise RuntimeError('Worker hard stop retained')
 except BaseException as e:failure=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
 report=dict(schema_version=SCHEMA+'_report',status='HARD_STOP' if failure else 'COMPLETED',protocol_sha256=sha(protocol),freeze_commit=freeze,returncode=code,failure=failure,worker_result=binding(out/'worker_result.json') if (out/'worker_result.json').exists() else None,raw_result=binding(raw/'result.json') if (raw/'result.json').exists() else None,cases=result.get('cases',[]) if result else [],fixture_schedule=result.get('fixture_schedule',[]) if result else [],actual_counts=result.get('actual_counts',{}) if result else {},remaining_case_indices_NOT_RUN=result.get('remaining_case_indices_NOT_RUN',[20,28]) if result else [20,28],qualification='PASS' if not failure and result and len(result['cases'])==2 and all(c['qualification']=='PASS' for c in result['cases']) else 'NOT_QUALIFIED',partial_files=[binding(p) for p in raw.rglob('*') if p.is_file()] if failure and raw.exists() else [],physical_outcome=None,**core().math.FLAGS)
 write(out/'report.json',report);return report

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run','_worker'));parser.add_argument('--protocol',type=Path,default=HERE/'protocol.json');parser.add_argument('--durable-report',type=Path);parser.add_argument('--durable-report-sha256');parser.add_argument('--freeze-commit');a=parser.parse_args()
 if a.stage=='prepare':
  if not a.durable_report or not a.durable_report_sha256:parser.error('Explicit inspected durable receipt path/hash required')
  r=prepare(a.protocol,a.durable_report,a.durable_report_sha256)
 elif a.stage=='run':
  if not a.freeze_commit:parser.error('Exact freeze commit required')
  r=run(a.protocol,a.freeze_commit)
 else:
  if not a.freeze_commit:parser.error('Exact freeze commit required')
  raise SystemExit(worker(a.protocol,a.freeze_commit))
 print(json.dumps(dict(status=r.get('status','PREPARED_NOT_RUN'),schema_version=SCHEMA)))
