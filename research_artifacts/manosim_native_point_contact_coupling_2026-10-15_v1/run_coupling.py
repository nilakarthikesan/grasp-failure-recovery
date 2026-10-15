"""Once-only STATIC point-load coupling. Prepare requires completed source lineage."""
from pathlib import Path
import argparse,hashlib,importlib.abc,importlib.util,json,subprocess,sys,time,traceback,warnings
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
RUNTIME=ROOT/'runs/manosim_mujoco_3_4_0_runtime_v1';sys.path.insert(0,str(RUNTIME))
if 'mujoco' in sys.modules and (sys.modules['mujoco'].__version__!='3.4.0' or not Path(sys.modules['mujoco'].__file__).resolve().is_relative_to(RUNTIME)):raise RuntimeError('Wrong preloaded engine')
import numpy as np
if np.__version__!='2.2.6':raise RuntimeError('Unchanged base NumPy2.2.6 required')
SOURCE=ROOT/'research_artifacts/manosim_native_source_pipeline_2026-10-15_v1'
ANALYTIC=ROOT/'research_artifacts/manosim_compliant_contact_dynamics_2026-10-15_v1'
SKIN=ROOT/'research_artifacts/manosim_articulated_skin_2026-10-15_v1'
SCHEMA='manosim_native_camera_graph_static_materialpoint_coupling_v1'
SOURCE_REPORT_SCHEMA='manosim_native_source_camera_pipeline_geometry_v1_report'
SOURCE_READBACK_SCHEMA='manosim_native_source_pipeline_independent_saved_readback_v1'
SOURCE_READBACK_STATUS='VERIFIED_COMPLETED_SOURCE_CAMERA_PIPELINE_GEOMETRY_ONLY'
ANALYTIC_HASHES=('13b2d8e4ad214d325f7605f1ee3bafa477bebe77f8769a1fd6624fb19cc1a03b','944e5a2dcf231ccc54451f37000a76c956aa728eeb91bb4faa12bdcb06cdba8c','e370d13f5bfc824d6591ec6ba58e1090688c9710c709a3e0a13a57029f247d0b')
INDICES=(20,28)
BLOCKED=('manotorch','manopth','smplx','hug','transformers','grasp_failure_prediction')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 def reject(x):raise ValueError('Nonfinite JSON '+x)
 value=json.loads(Path(p).read_text(),parse_constant=reject)
 def check(v):
  if isinstance(v,float) and not np.isfinite(v):raise ValueError('Nonfinite JSON number')
  if isinstance(v,dict):
   for x in v.values():check(x)
  if isinstance(v,list):
   for x in v:check(x)
 check(value);return value

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def case_module():return load('frozen_static_coupling_case',HERE/'coupling_case.py')
def binding(p):
 p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def verify(bindings):
 for name,digest in bindings.items():
  if sha(name)!=digest:raise ValueError('Bound input changed: '+name)
def verify_inventory(inventory):
 for name,row in inventory.items():
  actual=binding(name)
  if actual['sha256']!=row['sha256'] or actual['bytes']!=row['bytes'] or ('path' in row and actual['path']!=row['path']):raise ValueError('Creation-time artifact changed: '+name)
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def old_git_sources(p,freeze,mode):
 selected=p['tracked_sources'] if mode=='relative_map' else {str(Path(n).relative_to(ROOT)):sha(n) for n in p['tracked_sources']}
 for relative,digest in selected.items():
  if hashlib.sha256(git('show',freeze+':'+relative)).hexdigest()!=digest or sha(ROOT/relative)!=digest:raise ValueError('Prior Git source changed')
 return dict(selected)


def prior(source_protocol_sha,source_readback_sha):
 sp,rp,ip=SOURCE/'protocol.json',SOURCE/'run_v1/report.json',SOURCE/'run_v1/independent_readback.json'
 if sha(sp)!=source_protocol_sha or sha(ip)!=source_readback_sha:raise ValueError('Explicit independently reviewed source hashes required')
 p,r,i=read(sp),read(rp),read(ip)
 if (r.get('schema_version')!=SOURCE_REPORT_SCHEMA or r.get('status')!='COMPLETED' or i.get('schema_version')!=SOURCE_READBACK_SCHEMA
  or i.get('status')!=SOURCE_READBACK_STATUS or i.get('active_protocol_sha256')!=sha(sp) or r.get('protocol_sha256')!=sha(sp)
  or i.get('report_sha256')!=sha(rp) or i.get('freeze_commit')!=r.get('freeze_commit') or i.get('all_bindings_verified_before') is not True or i.get('all_bindings_verified_after') is not True
  or i.get('expected_bindings')!=len(p['bindings']) or i.get('errors') or [x['index'] for x in r.get('cases',[])]!=list(INDICES)):
  raise ValueError('Completed independent source pipeline required before preparation')
 verify(p['bindings']);verify_inventory(i['artifact_inventory']);selected=old_git_sources(p,r['freeze_commit'],'relative_map');inputs=dict(p['bindings']);specs={}
 for index,row in zip(INDICES,r['cases'],strict=True):
  if row.get('status')!='COMPLETED' or row.get('returncode')!=0:raise ValueError('Both exact source cases must be completed')
  result=read(row['result']);raw=Path(row['raw_result'])
  if sha(row['result'])!=row['result_sha256'] or sha(raw)!=row['raw_result_sha256'] or Path(row['result']).read_bytes()!=raw.read_bytes():raise ValueError('Source scalar copies differ')
  if result.get('source_pipeline_geometry_check_passed') is not True or not result.get('checks') or any(v is not True for v in result['checks'].values()) or result.get('old64_jacobian_qualified_for_this_profile') is not False:raise ValueError('Source geometry gate incomplete or old tangent mislabeled')
  artifacts={Path(a['path']).name:a for a in result['artifacts']};buffers=artifacts['literal_source_buffers.npz'];readbacks=artifacts['all109_native_source_engine_readbacks.npz']
  for a in result['artifacts']:
   inv=i['artifact_inventory'].get(a['path'])
   if inv is None or inv['sha256']!=a['sha256'] or inv['bytes']!=a['bytes']:raise ValueError('Source array not creation-time independently indexed')
  # Read only already sealed labels to freeze the six point IDs; no geometry/model evaluation.
  with np.load(buffers['path'],allow_pickle=False) as a:faces=a['faces'];labels=a['native_face_labels']
  kin=load('static_point_selector',HERE/'contact_kinematics.py');choices=kin.fixed_material_points(faces,labels)
  original=p['cases'][str(index)]
  specs[str(index)]=dict(source_inventory_index=index,literal_buffers=buffers,source_readbacks=readbacks,original_compiled_readback=original['original_compiled_readback'],base_xml=original['base_xml'],saved_named_joint_addresses=original['saved_named_joint_addresses'],material_points=choices)
 for name,row in i['artifact_inventory'].items():inputs[name]=row['sha256']
 for path in (sp,rp,ip):inputs[str(path.resolve())]=sha(path)
 ap,ar,ai=ANALYTIC/'protocol.json',ANALYTIC/'run_v1/report.json',ANALYTIC/'run_v1/independent_readback_v2.json'
 if tuple(sha(x) for x in (ap,ar,ai))!=ANALYTIC_HASHES:raise ValueError('Exact separately qualified compliant benchmark required')
 pa,ra,ia=read(ap),read(ar),read(ai)
 if ra.get('status')!='COMPLETED_COMPLIANT_CONTACT_BENCHMARK' or ra.get('numerical_compliant_contact_qualification')!='PASS' or ia.get('status')!='VERIFIED_COMPLETED_COMPLIANT_CONTACT_BENCHMARK' or ia.get('all_bindings_verified_before') is not True or ia.get('all_bindings_verified_after') is not True or ia.get('errors') or ia.get('active_protocol_sha256')!=sha(ap) or ia.get('report_sha256')!=sha(ar):raise ValueError('Conditional analytic benchmark not independently complete')
 verify(pa['bindings']);verify_inventory(ia['artifact_inventory']);selected.update(old_git_sources(pa,ra['freeze_commit'],'absolute_list'))
 for name,digest in pa['bindings'].items():
  if name in inputs and inputs[name]!=digest:raise ValueError('Prior source closure conflict')
  inputs[name]=digest
 for name,row in ia['artifact_inventory'].items():inputs[name]=row['sha256']
 for path in (ap,ar,ai):inputs[str(path.resolve())]=sha(path)
 for name in inputs:
  path=Path(name)
  if path.is_relative_to(ROOT/'research_artifacts') and path.suffix in ('.json','.md') and '/run_v1/' in name:selected[str(path.relative_to(ROOT))]=inputs[name]
 return p,inputs,selected,specs


def own_sources():return [p for p in sorted(HERE.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name in ('README.md','controls.json'))]
def prepare(protocol,source_protocol_sha,source_readback_sha):
 protocol=Path(protocol).resolve()
 if protocol!=HERE/'protocol.json' or protocol.exists() or (HERE/'run_v1').exists():raise ValueError('Fresh exact preparation path only')
 source,bindings,selected,specs=prior(source_protocol_sha,source_readback_sha);case=case_module();controls=read(HERE/'controls.json')
 if controls.get('exit_code')!=0 or controls.get('integration_calls')!=0:raise ValueError('Successful no-integration manufactured controls required')
 for path in own_sources():bindings[str(path.resolve())]=sha(path);selected[str(path.relative_to(ROOT))]=sha(path)
 for path in (SKIN/'skin_kinematics.py',SKIN/'contact_law.py'):
  if str(path) not in bindings:raise ValueError('Frozen fullskin/law closure absent')
 profile_sources={str(p):sha(p) for p in (HERE/'contact_kinematics.py',SKIN/'skin_kinematics.py')}
 fixture_hashes={str(index):hashlib.sha256(case.scene_xml(specs[str(index)]['base_xml'])[0].encode()).hexdigest() for index in INDICES}
 p=dict(schema_version=SCHEMA,protocol_path=str(protocol),source_protocol_sha256=source_protocol_sha,source_readback_sha256=source_readback_sha,
  bindings=bindings,tracked_sources=selected,cases=specs,budget=case.BUDGET,gates=case.GATES,parameters=case.PARAMETERS,profile_source_hashes=profile_sources,fixture_xml_sha256=fixture_hashes,
  output_root=str(HERE/'run_v1'),raw_root=str(ROOT/'runs'/HERE.name/'run_v1/raw'),state_domain='mano_source_postAA32_exp64_camera64_named_state_v1',physical_profile='mano_source_initialized_camera_graph_continuous64_contact_v1',**case.FLAGS,physical_outcome=None)
 verify(bindings);write(protocol,p);return p


def validate(protocol,freeze):
 p=read(protocol);source,bindings,selected,specs=prior(p['source_protocol_sha256'],p['source_readback_sha256']);case=case_module()
 for f in own_sources():bindings[str(f.resolve())]=sha(f);selected[str(f.relative_to(ROOT))]=sha(f)
 if p.get('schema_version')!=SCHEMA or p.get('protocol_path')!=str(Path(protocol).resolve()) or p['bindings']!=bindings or p['tracked_sources']!=selected or p['cases']!=specs or p['budget']!=case.BUDGET or p['gates']!=case.GATES or p['parameters']!=case.PARAMETERS or p.get('physical_outcome') is not None or any(p.get(k) is not False for k in case.FLAGS):raise ValueError('Frozen static method/source closure changed')
 if p['output_root']!=str(HERE/'run_v1') or p['raw_root']!=str(ROOT/'runs'/HERE.name/'run_v1/raw') or p['physical_profile']!='mano_source_initialized_camera_graph_continuous64_contact_v1' or p['state_domain']!='mano_source_postAA32_exp64_camera64_named_state_v1':raise ValueError('Frozen roots/domain changed')
 if p['profile_source_hashes']!={str(f):sha(f) for f in (HERE/'contact_kinematics.py',SKIN/'skin_kinematics.py')} or p['fixture_xml_sha256']!={str(index):hashlib.sha256(case.scene_xml(specs[str(index)]['base_xml'])[0].encode()).hexdigest() for index in INDICES}:raise ValueError('Physical graph/fixture changed')
 verify(p['bindings'])
 if git('rev-parse','HEAD').decode().strip()!=freeze:raise ValueError('Exact HEAD hold required')
 for relative,digest in selected.items():
  if hashlib.sha256(git('show',freeze+':'+relative)).hexdigest()!=digest:raise ValueError('Committed source/evidence changed')
 if git('show',freeze+':'+str(Path(protocol).resolve().relative_to(ROOT)))!=Path(protocol).read_bytes():raise ValueError('Protocol not frozen in commit')
 return p


class ImportGuard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in BLOCKED:raise RuntimeError('Native source/inference/training import forbidden '+fullname)


class EngineGuard:
 def __init__(self,mj,p,index,notify):
  self.mj=mj;self.p=p;self.index=index;self.notify=notify;self.counts={k:0 for k in ('model_compilations','data_constructions','static_forward_calls','full_mass_matrix_calls')};self.forbidden=[];self.original={};self.models=[];self.pairs=[]
 def charge(self,key):
  if self.counts[key]>=self.p['budget'][key]:raise RuntimeError('Actual API quota exceeded before work '+key)
  self.counts[key]+=1;self.notify(dict(event='actual_engine_api',counts=dict(self.counts)))
 def deny(self,name):self.forbidden.append(name);raise RuntimeError('Forbidden engine API '+name)
 def __enter__(self):
  mj=self.mj
  names={name for name in dir(mj) if name.startswith('mj_') and callable(getattr(mj,name)) and name not in ('mj_name2id','mj_version','mj_versionString')}
  names.update(('MjModel','MjData','MjSpec','set_mjcb_control','set_mjcb_passive'))
  for name in names:
   if hasattr(mj,name):self.original[name]=getattr(mj,name);setattr(mj,name,lambda *a,_name=name,**k:self.deny(_name))
  guard=self
  class Model:
   @staticmethod
   def from_xml_path(path):
    if sha(path)!=guard.p['fixture_xml_sha256'][str(guard.index)]:return guard.deny('unplanned_XML')
    guard.charge('model_compilations')
    with warnings.catch_warnings(record=True) as emitted:
     warnings.simplefilter('always');m=guard.original['MjModel'].from_xml_path(path)
    if emitted:raise RuntimeError('Compiler warning')
    guard.models.append(m);return m
  def data(model):
   if not any(model is x for x in guard.models) or guard.pairs:return guard.deny('unregistered_or_extra_data')
   guard.charge('data_constructions');d=guard.original['MjData'](model);guard.pairs.append((model,d));return d
  def forward(model,data):
   if not any(model is m and data is d for m,d in guard.pairs):return guard.deny('unregistered_forward')
   warn=data.warning.number.copy()
   if np.any(warn) or data.time!=0:return guard.deny('warning_or_nonzero_time')
   before={k:np.asarray(getattr(data,k)).copy() for k in ('qpos','qvel','ctrl','qfrc_applied','xfrc_applied')};guard.charge('static_forward_calls')
   with warnings.catch_warnings(record=True) as emitted:
    warnings.simplefilter('always');guard.original['mj_forward'](model,data)
   if emitted or data.time!=0 or not np.array_equal(warn,data.warning.number) or any(not np.array_equal(getattr(data,k),v) for k,v in before.items()) or any(not np.isfinite(getattr(data,k)).all() for k in ('qpos','qvel','qacc','ctrl')):raise RuntimeError('Static numerical/state integrity failed')
  def fullM(model,dst,packed):
   if not any(model is m for m in guard.models) or dst.shape!=(51,51) or not np.isfinite(packed).all():return guard.deny('unregistered_mass_readout')
   guard.charge('full_mass_matrix_calls');guard.original['mj_fullM'](model,dst,packed)
   if not np.isfinite(dst).all():raise RuntimeError('Nonfinite dense mass matrix')
  mj.MjModel=Model;mj.MjData=data;mj.mj_forward=forward;mj.mj_fullM=fullM;return self
 def __exit__(self,*a):
  for name,value in self.original.items():setattr(self.mj,name,value)


def verify_outputs(rows,raw):
 seen=set();raw=Path(raw).resolve()
 for row in rows:
  p=Path(row['path'])
  if p.is_symlink() or not p.resolve().is_relative_to(raw) or p.resolve() in seen or binding(p)!={k:row[k] for k in ('path','sha256','bytes')}:raise ValueError('Output creation-time binding/path changed')
  seen.add(p.resolve())
 if any(p.is_symlink() for p in raw.rglob('*')):raise ValueError('Output symlink forbidden')
 extras={p.resolve() for p in raw.rglob('*') if p.is_file()}-seen-{raw/'progress.json',raw/'result.json'}
 if extras:raise ValueError('Unindexed output artifacts')



def check_result(v,p,protocol_hash,freeze,index,raw,returncode):
 case=case_module()
 if (v.get('schema_version')!=SCHEMA+'_case' or v.get('index')!=index or v.get('protocol_sha256')!=protocol_hash or v.get('freeze_commit')!=freeze
  or v.get('physical_outcome') is not None or any(v.get(k) is not False for k in case.FLAGS)):
  raise ValueError('Child identity/domain/admission changed')
 if v.get('status') not in ('COMPLETED','HARD_STOP'):raise ValueError('Unknown child transaction status')
 verify_outputs(v.get('artifacts',[]),raw)
 if v['status']=='HARD_STOP':
  if returncode==0:raise ValueError('Failed child returned success')
  return
 expected={k:n for k,n in p['budget'].items() if k!='maximum_seconds_per_case'}
 m=v.get('measurement',{})
 if (returncode!=0 or v.get('failure') is not None or v.get('final_validation_failure') or v.get('publication_error') or v.get('unindexed_partial_files')
  or v.get('counts')!=expected or v.get('actual_engine_counts')!={k:p['budget'][k] for k in ('model_compilations','data_constructions','static_forward_calls','full_mass_matrix_calls')}
  or v.get('forbidden_attempts') or m.get('status')!='COMPLETED' or m.get('forbidden_counts')!=case.FORBIDDEN
  or m.get('physical_outcome') is not None or any(m.get(k) is not False for k in case.FLAGS)
  or m.get('source32_derivative_qualified') is not False or m.get('root_rigid_wrench_equivalence_claimed') is not False
  or m.get('static_point_coupling_status') not in ('PASS','FAIL') or m.get('source32_alignment_status') not in ('PASS','FAIL')
  or m.get('material_points')!=p['cases'][str(index)]['material_points'] or m.get('named_layout')!=p['cases'][str(index)]['saved_named_joint_addresses']
  or len(m.get('conditions',[]))!=12):raise ValueError('Completed child counts/method/flags incomplete')
 expected_names={'physical_profile.json','surface_tangent_and_alignment.npz','native_contact_disabled.xml','original_model_and_named_layout.npz'}|{f'condition_{j:02d}.npz' for j in range(12)}
 if {Path(a['path']).name for a in v['artifacts']}!=expected_names or len(v['artifacts'])!=len(expected_names):raise ValueError('Complete child artifact inventory incomplete')
 for j,c in enumerate(m['conditions']):
  if c.get('condition')!=j or c.get('part')!=m['material_points'][j//2]['part'] or c.get('mode')!=('normal','slide')[j%2] or c.get('root_rigid_wrench_equivalence_claimed') is not False or type(c.get('passed')) is not bool:raise ValueError('Condition identity changed')
 for key in ('profile_hash','state_hash'):
  value=m.get(key)
  if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Unbound profile/state hash')


def worker(protocol,freeze,index):
 p=validate(protocol,freeze);case=case_module();raw=Path(p['raw_root'])/str(index);raw.mkdir(parents=True,exist_ok=False);started=time.monotonic();artifacts=[];counts={};failure=None
 def notify(v):
  nonlocal counts
  if time.monotonic()-started>p['budget']['maximum_seconds_per_case']:raise TimeoutError('Fixed static case budget exceeded')
  if v.get('event')=='operation':counts=dict(v['counts'])
  if v.get('event')=='artifact':artifacts.append(v['artifact'])
  path=raw/'progress.tmp';path.write_text(json.dumps(dict(last=v,elapsed_s=time.monotonic()-started),allow_nan=False)+'\n');path.replace(raw/'progress.json')
 finder=ImportGuard();sys.meta_path.insert(0,finder);guard=None;measurement=None
 try:
  import mujoco
  if mujoco.__version__!='3.4.0' or not Path(mujoco.__file__).resolve().is_relative_to(RUNTIME):raise RuntimeError('Wrong engine path/version')
  if any(getattr(mujoco,k)() is not None for k in ('get_mjcb_control','get_mjcb_passive')):raise RuntimeError('Unexpected engine force callback')
  with EngineGuard(mujoco,p,index,notify) as guard:measurement=case.run_case(p['cases'][str(index)],raw,p,notify)
  if measurement['counts']!={k:v for k,v in p['budget'].items() if k!='maximum_seconds_per_case'} or measurement['counts']!=counts or guard.counts!={k:p['budget'][k] for k in guard.counts} or guard.forbidden:raise ValueError('Independent API/scientific counts disagree')
  verify_outputs(artifacts,raw)
 except BaseException as e:failure=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
 finally:
  if finder in sys.meta_path:sys.meta_path.remove(finder)
 result=dict(schema_version=SCHEMA+'_case',index=index,status='HARD_STOP' if failure else 'COMPLETED',protocol_sha256=sha(protocol),freeze_commit=freeze,failure=failure,artifacts=artifacts,counts=counts,actual_engine_counts=guard.counts if guard else None,forbidden_attempts=guard.forbidden if guard else [],elapsed_s=time.monotonic()-started,physical_outcome=None,**case.FLAGS)
 if measurement is not None:
  if measurement['artifacts']!=artifacts:result.update(status='HARD_STOP',publication_error='Case artifact inventory differs')
  result['measurement']={k:v for k,v in measurement.items() if k not in ('artifacts','counts')}
 try:validate(protocol,freeze);verify_outputs(artifacts,raw)
 except BaseException as e:result.update(status='HARD_STOP',final_validation_failure=dict(type=type(e).__name__,message=str(e)))
 indexed={Path(a['path']).resolve() for a in artifacts};result['unindexed_partial_files']=[binding(x) for x in raw.rglob('*') if x.is_file() and x.resolve() not in indexed and x.name!='progress.json']
 if result['status']=='COMPLETED':
  try:check_result(result,p,sha(protocol),freeze,index,raw,0)
  except BaseException as e:result.update(status='HARD_STOP',publication_error=dict(type=type(e).__name__,message=str(e)))
 write(raw/'result.json',result);return 0 if result['status']=='COMPLETED' else 1


def run(protocol,freeze):
 p=validate(protocol,freeze);out=Path(p['output_root']);raw=Path(p['raw_root'])
 if out.exists() or raw.exists():raise FileExistsError('Once-only roots already exist')
 out.mkdir();raw.mkdir(parents=True);write(out/'STARTED.json',dict(protocol_sha256=sha(protocol),freeze_commit=freeze));rows=[];stopped=False
 for index in INDICES:
  if stopped:rows.append(dict(index=index,status='NOT_RUN'));continue
  row=dict(index=index,status='STARTED');log=out/f'case_{index}.log'
  try:
   with log.open('x') as stream:child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'_worker','--protocol',str(Path(protocol).resolve()),'--freeze-commit',freeze,'--index',str(index)],stdout=stream,stderr=subprocess.STDOUT,timeout=610)
   result=raw/str(index)/'result.json';v=read(result);check_result(v,p,sha(protocol),freeze,index,raw/str(index),child.returncode);target=out/str(index)/'result.json';target.parent.mkdir();target.write_bytes(result.read_bytes());row.update(status=v['status'],returncode=child.returncode,result=binding(target),raw_result=binding(result))
   if child.returncode or v['status']!='COMPLETED':stopped=True
  except BaseException as e:stopped=True;row.update(status='HARD_STOP',failure=dict(type=type(e).__name__,message=str(e)),partial_files=[binding(x) for x in (raw/str(index)).rglob('*') if x.is_file()])
  rows.append(row)
 final_failure=None
 try:
  validate(protocol,freeze)
  for row in rows:
   if 'result' not in row:continue
   for key in ('result','raw_result'):
    if binding(row[key]['path'])!=row[key]:raise ValueError('Scalar result changed')
   if Path(row['result']['path']).read_bytes()!=Path(row['raw_result']['path']).read_bytes():raise ValueError('Scalar copies differ')
   v=read(row['result']['path']);check_result(v,p,sha(protocol),freeze,row['index'],Path(p['raw_root'])/str(row['index']),row['returncode'])
 except BaseException as e:stopped=True;final_failure=dict(type=type(e).__name__,message=str(e))
 measured=[read(row['result']['path']).get('measurement',{}) for row in rows if 'result' in row]
 case=case_module();report=dict(schema_version=SCHEMA+'_report',status='HARD_STOP' if stopped else 'COMPLETED_STATIC_POINT_COUPLING',protocol_sha256=sha(protocol),freeze_commit=freeze,cases=rows,final_validation_failure=final_failure,static_point_coupling_status='PASS' if not stopped and len(measured)==2 and all(m.get('static_point_coupling_status')=='PASS' for m in measured) else 'NOT_QUALIFIED',source32_alignment_status='PASS' if not stopped and len(measured)==2 and all(m.get('source32_alignment_status')=='PASS' for m in measured) else 'NOT_QUALIFIED',physical_outcome=None,**case.FLAGS)
 write(out/'report.json',report);return report

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run','_worker'));parser.add_argument('--protocol',type=Path,default=HERE/'protocol.json');parser.add_argument('--source-protocol-sha');parser.add_argument('--source-readback-sha');parser.add_argument('--freeze-commit');parser.add_argument('--index',type=int,choices=INDICES);a=parser.parse_args()
 if a.stage=='prepare':
  if not a.source_protocol_sha or not a.source_readback_sha:parser.error('Both explicitly inspected source hashes required')
  result=prepare(a.protocol,a.source_protocol_sha,a.source_readback_sha)
 elif a.stage=='run':
  if not a.freeze_commit:parser.error('Exact freeze commit required')
  result=run(a.protocol,a.freeze_commit)
 else:
  if not a.freeze_commit or a.index is None:parser.error('Worker freeze/index required')
  raise SystemExit(worker(a.protocol,a.freeze_commit,a.index))
 print(json.dumps(dict(status=result.get('status','PREPARED_NOT_RUN'),schema_version=SCHEMA)))
