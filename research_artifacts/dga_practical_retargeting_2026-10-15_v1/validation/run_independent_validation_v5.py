"""Recount mapper outputs, certify conditional full-cube branch gates, export replay.

No optimization, inference, physics, labels, or training. Original native and
candidate archives remain immutable; fallback is explicitly recorded separately.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,hashlib,json,time,warnings
import numpy as np
from independent_validator import *
from independent_geometry_v5 import source_part,target_part,adaptive_compare_meshes
ROOT=Path('/private/tmp/dga_practical_validation_20261015_v1')
WORK=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
SOURCE=Path('/Users/nilakarthikesan/grasp-failure-prediction/local_assets/dexgrasp_anything/source-86353d710ca71d0f547f0d8b15cc8ca959c78163/assets/urdf/sr_grasp_description/urdf/shadowhand.urdf')
TARGET=WORK/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf'
def read(p):return json.loads(Path(p).read_text())
def verify(p,d):
 if sha(p)!=d.removeprefix('sha256:'):raise ValueError('hash mismatch '+str(p))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def check_files(records):
 for p,d in records.items():verify(p,d)

def main(args):
 policy=read(ROOT/'validation_policy_frozen_v5.json');check_files(policy['code_sha256'])
 report=read(args.input/'report.json');before={str(args.input/n):d for n,d in report['outputs'].items()};before[str(args.input/'report.json')]=sha(args.input/'report.json');check_files(before)
 with np.load(args.input/'poses.npz',allow_pickle=False) as z:poses={k:z[k] for k in z.files}
 total_native_rows=len(report['records'])
 if args.rows is not None:
  requested=set(args.rows)
  if len(requested)!=len(args.rows) or not requested.issubset({r['global_sample_index'] for r in report['records']}):raise ValueError('invalid selected-row partition')
  report['records']=[r for r in report['records'] if r['global_sample_index'] in requested]
  keep=np.array([int(g) in requested for g in poses['global_sample_indices']])
  poses={k:v[keep] for k,v in poses.items()}
 indices=poses['global_sample_indices'].tolist()
 if len(indices)!=len(set(indices)):raise ValueError('duplicate candidate rows')
 s,t=Tree(SOURCE,SOURCE_SHA),Tree(TARGET,TARGET_SHA);args.output.mkdir(exist_ok=False);geometry_dependencies={}
 compiler=CompiledKinematics(t,args.output/'target_compiled.mjb');records=[];final_q=[];final_roots=[];final_raw=[];final_indices=[];started=time.monotonic();counts=Counter()
 for row in report['records']:
  g=row['global_sample_index'];mp=Path(row['native_manifest']['path']);verify(mp,row['native_manifest']['sha256']);manifest=read(mp);native=mp.parent/manifest['output']['path'];verify(native,manifest['output']['sha256'])
  with np.load(native,allow_pickle=False) as z:raw=z['hand_qpos'][row['within_batch_sample_index']].copy()
  verify_raw=hashlib.sha256(raw.tobytes()).hexdigest()
  if verify_raw!=row['raw33_row_sha256'].removeprefix('sha256:'):raise ValueError('raw fingerprint mismatch')
  violations=s.violations(dict(zip(NAMES,raw[9:],strict=True)))
  record={'global_sample_index':g,'raw33_sha256':verify_raw,'source_domain_violations':violations,'native_manifest':str(mp),'native_output_sha256':sha(native)}
  if violations:
   if g in indices or row['status']!='SOURCE_DOMAIN_REJECTED_UNALTERED':raise ValueError('source rejection accounting mismatch')
   record['status']='SOURCE_DOMAIN_REJECTED_UNALTERED';records.append(record);counts['source_invalid']+=1;write(args.output/f'row_{g:02d}.json',record);continue
  if g not in indices:raise ValueError('missing source-valid candidate')
  i=indices.index(g);q=poses['target_joint_values'][i];baseline=poses['baseline_target_joint_values'][i];root=poses['object_from_target_root_world'][i];baseroot=poses['baseline_object_from_target_root_world'][i];object_from_input=poses['object_from_input'][i]
  if not np.array_equal(object_from_input,np.eye(4)):raise ValueError('current complete cube geometry only supports exact identity input transform')
  if not np.array_equal(q,np.array([row['selected']['q'][n] for n in NAMES])) or not np.array_equal(baseline,np.array([row['baseline']['q'][n] for n in NAMES])):raise ValueError('saved pose JSON-NPZ mismatch')
  for a,b in [('root','object_from_target_root_world'),('palm','object_from_target_palm'),('forearm','object_from_target_forearm')]:
   if not np.array_equal(poses[b][i],np.array(row['selected'][a])):raise ValueError('saved transform JSON-NPZ mismatch')
  if not np.array_equal(root,baseroot):raise ValueError('fixed-palm/wrist candidate changed root bytes')
  v=validate_saved(raw,q,root,s,t,baseline_joints=baseline,object_from_input=object_from_input,saved_raw=poses['source_hand_qpos'][i]);bv=validate_saved(raw,baseline,baseroot,s,t,baseline_joints=baseline,object_from_input=object_from_input,saved_raw=poses['source_hand_qpos'][i])
  for name,error in v['landmark_errors_m'].items():
   at=row['site_names'].index(name);saved=np.array(row['selected']['errors_m'])[at];expected=float(np.sqrt(np.sum(saved*saved)))
   if abs(error-expected)>1e-12:raise ValueError('independent landmark mismatch')
  record.update(status='CANDIDATE_STRUCTURAL_RECOUNT_PASS',candidate=v,baseline=bv,candidate_compiled=compiler.validate(q,root),baseline_compiled=compiler.validate(baseline,root),branches={})
  refpath=Path(row['mapping_reference']['path']);verify(refpath,row['mapping_reference']['sha256']);reference=read(refpath);runtime=refpath.parent/reference['arrays']['path'];verify(runtime,reference['arrays']['sha256'])
  dependencies={str(mp):row['native_manifest']['sha256'],str(native):manifest['output']['sha256'],str(refpath):row['mapping_reference']['sha256'],str(runtime):reference['arrays']['sha256'],**row['backend_dependency_manifest']['target_visual_assets']}
  check_files(dependencies);geometry_dependencies.update(dependencies)
  for link in t.xml.findall('link'):
   if not link.get('name').startswith(tuple(PARTS.values())):continue
   for mesh in link.findall('visual/geometry/mesh'):
    asset=str((t.path.parent/mesh.get('filename')).resolve())
    if asset not in dependencies:raise ValueError('unbound target mesh '+asset)
  sf={n:np.array(a) for n,a in bv['target_object_from_links'].items()};cf={n:np.array(a) for n,a in v['target_object_from_links'].items()};final=baseline.copy()
  for part,prefix in PARTS.items():
   ids=[j for j,n in enumerate(NAMES) if n.startswith(prefix.upper())];same=np.array_equal(q[ids],baseline[ids]);decision={'same_target_geometry_as_baseline':same}
   if same:
    decision.update(decision='PASS_BY_TARGET_GEOMETRY_IDENTITY',adopt_candidate=False,native_equivalence_claimed=False);counts['identity_branches']+=1
   else:
    source=source_part(runtime,row['within_batch_sample_index'],part);bm=target_part(t,sf,part);cm=target_part(t,cf,part)
    out=adaptive_compare_meshes(source,bm,cm,output_dir=args.output/'geometry',prefix=f'row{g}_{part}')
    decision.update(out);max_delta=v['baseline']['per_part_max_difference_m'][part];sse_delta=v['baseline']['per_part_sse_difference_m2'][part]
    decision.update(independent_max_landmark_change_m=max_delta,independent_landmark_sse_change_m2=sse_delta)
    strict=(out['candidate']['continuous_max_upper_m']<out['baseline']['continuous_max_lower_m'] or max_delta<0 or sse_delta<0)
    decision['independent_strict_improvement']=strict
    decision['adopt_candidate']=out['decision']=='PASS_NONREGRESSION' and max_delta<=0 and sse_delta<=0 and strict
    if decision['adopt_candidate']:final[ids]=q[ids];counts['certified_corrected_branches']+=1
    else:counts['explicit_baseline_fallback_branches']+=1
    counts[out['decision']]+=1
    print(json.dumps({'row':g,'part':part,'decision':out['decision'],'adopt_candidate':decision['adopt_candidate'],'queried_cells':sum(l['queried_cell_count'] for l in out['levels']),'elapsed_s':time.monotonic()-started}),flush=True)
   record['branches'][part]=decision
  fv=validate_saved(raw,final,root,s,t,baseline_joints=baseline,object_from_input=object_from_input,saved_raw=poses['source_hand_qpos'][i]);record['final']=fv;record['final_compiled']=compiler.validate(final,root);record['status']='STATIC_RELATIVE_BASELINE_GATE_PASS';record['absolute_native_fidelity']='NOT_ACCEPTED_OR_CERTIFIED';record['execution_eligible']=False
  final_indices.append(g);final_q.append(final);final_roots.append(root);final_raw.append(raw);counts['source_valid_mapped']+=1;records.append(record);write(args.output/f'row_{g:02d}.json',record)
  print(json.dumps({'row':g,'status':record['status'],'elapsed_s':time.monotonic()-started}),flush=True)
 check_files(before);check_files(policy['code_sha256']);check_files(geometry_dependencies)
 if len(records)!=len(report['records']) or set(indices)!=set(final_indices):raise ValueError('final total accounting')
 np.savez(args.output/'final_static_poses.npz',global_sample_indices=np.array(final_indices),source_hand_qpos=np.array(final_raw),target_joint_values=np.array(final_q),object_from_target_root_world=np.array(final_roots))
 result={'schema_version':'dga_practical_independent_static_validation_v5','generated_at_utc':datetime.now(timezone.utc).isoformat(),'policy_sha256':sha(ROOT/'validation_policy_frozen_v5.json'),'input_report_sha256':sha(args.input/'report.json'),'cohort_input':str(args.input),'records':records,'counts':dict(counts),'native_rows_in_partition':len(records),'parent_cohort_native_rows':total_native_rows,'partition_rows':[r['global_sample_index'] for r in records],'all_native_rows_accounted':len(records)==total_native_rows,'native_holdout':False,'geometry_fidelity_budget_acceptance':'NOT_CLAIMED','scope':'Engineering correctness plus conditional completecube per-branch nonregression relative explicit projection reference; not absolute native fidelity or physical adequacy. Baseline fallbacks retain source native distortions.','execution_eligible':False,'physics_steps':0,'inference_calls':0,'evaluation_data_generated':False,'training_calls':0,'elapsed_s':time.monotonic()-started,'input_hashes':before,'geometry_dependency_hashes':geometry_dependencies,'output_hashes':{str(p.relative_to(args.output)):sha(p) for p in args.output.rglob('*') if p.is_file()}}
 write(args.output/'report.json',result);print(json.dumps({'COMPLETE':dict(counts),'report_sha256':sha(args.output/'report.json'),'elapsed_s':result['elapsed_s']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--rows',nargs='+',type=int);args=p.parse_args()
 with warnings.catch_warnings():warnings.simplefilter('error');main(args)
