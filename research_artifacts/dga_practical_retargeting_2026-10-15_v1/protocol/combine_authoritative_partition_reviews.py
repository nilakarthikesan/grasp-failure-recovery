"""Combine independent archive reviews with exact all20 cohort accounting."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,xml.etree.ElementTree as ET
import numpy as np
NAMES=('WRJ2','WRJ1','FFJ4','FFJ3','FFJ2','FFJ1','MFJ4','MFJ3','MFJ2','MFJ1','RFJ4','RFJ3','RFJ2','RFJ1','LFJ5','LFJ4','LFJ3','LFJ2','LFJ1','THJ5','THJ4','THJ3','THJ2','THJ1')
TARGET=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(args):
 checks=[];bindings={str(Path(__file__).resolve()):sha(__file__)};cohorts={};summary={};bounds=0;branch_results=[]
 def check(name,ok,detail=None):
  checks.append({'check':name,'pass':bool(ok),'detail':detail})
  if not ok:raise ValueError(name+': '+str(detail))
 xml=ET.parse(TARGET).getroot();limits={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in xml.findall('joint') if j.get('type')=='revolute'}
 check('target_bytes_unchanged',sha(TARGET)=='29a50063cced2fba81dfe3d208ba54499fc12635c9189d0822f7caa2771c36de');bindings[str(TARGET)]=sha(TARGET)
 for receipt_path in args.receipt:
  receipt=read(receipt_path);bindings[str(receipt_path)]=sha(receipt_path)
  check('independent_receipt_pass',receipt['status']=='PASS' and not receipt['partial_row_only'],str(receipt_path))
  bound_reports=[p for p in receipt['source_bindings'] if p.endswith('/report.json')]
  check('one_authoritative_report_per_receipt',len(bound_reports)==1,str(receipt_path));rp=Path(bound_reports[0]);r=read(rp)
  for p,h in receipt['source_bindings'].items():check('recount_binding_still_matches',sha(p)==h,p);bindings[p]=h
  source=Path(r['cohort_input']);original=read(source/'report.json');bindings[str(source/'report.json')]=sha(source/'report.json')
  check('partition_binds_full_mapper_report',r['input_report_sha256'].removeprefix('sha256:')==sha(source/'report.json'),str(rp))
  if str(source) not in cohorts:cohorts[str(source)]={'expected':{z['global_sample_index']:z for z in original['records']},'observed':{},'source_valid':0,'source_invalid':0,'partitions':[]}
  cohort=cohorts[str(source)];cohort['partitions'].append(str(rp))
  with np.load(rp.parent/'final_static_poses.npz',allow_pickle=False) as z:final={k:z[k] for k in z.files}
  check('final_array_row_count',len(final['global_sample_indices'])==r['counts'].get('source_valid_mapped',0),str(rp))
  for row in r['records']:
   g=row['global_sample_index'];check('partition_row_unique',g not in cohort['observed'],[str(source),g]);cohort['observed'][g]=row
   check('partition_row_expected',g in cohort['expected'],[str(source),g]);old=cohort['expected'][g]
   check('native_identity_matches_mapper',row['raw33_sha256']==old['raw33_row_sha256'].removeprefix('sha256:'),[str(source),g])
   if row['source_domain_violations']:
    check('rejection_unchanged_and_accounted',row['status']=='SOURCE_DOMAIN_REJECTED_UNALTERED' and old['status']=='SOURCE_DOMAIN_REJECTED_UNALTERED' and g not in final['global_sample_indices'],[str(source),g]);cohort['source_invalid']+=1;continue
   cohort['source_valid']+=1;check('legal_static_status',row['status']=='STATIC_RELATIVE_BASELINE_GATE_PASS' and row['execution_eligible'] is False,[str(source),g])
   i=final['global_sample_indices'].tolist().index(g);q=dict(zip(NAMES,final['target_joint_values'][i],strict=True));raw=final['source_hand_qpos'][i]
   check('literal_target_limits',all(np.isfinite(v) and limits[n][0]<=v<=limits[n][1] for n,v in q.items()),[str(source),g])
   check('raw_final_hash',hashlib.sha256(raw.tobytes()).hexdigest()==row['raw33_sha256'],[str(source),g])
   check('wrists_exactly_native',all(q[n]==raw[9+k] for k,n in enumerate(NAMES[:2])),[str(source),g])
   for label in ['candidate_compiled','baseline_compiled','final_compiled']:
    c=row[label];check('independent_compiled_fk',c['status']=='PASS' and c['physics_steps']==0 and c['data_time_s']==0 and not c['blocked_entrypoint_attempts'] and max(c['maximum'].values())<=1e-9,[str(source),g,label])
   b=row['final']['baseline'];check('final_per_finger_landmark_nonregression',all(v<=0 for v in b['per_part_max_difference_m'].values()) and all(v<=0 for v in b['per_part_sse_difference_m2'].values()),[str(source),g])
   for part,br in row['branches'].items():
    key=('IDENTITY' if br['same_target_geometry_as_baseline'] else 'CERTIFIED_CORRECTION' if br['adopt_candidate'] else 'BASELINE_FALLBACK')
    summary[key]=summary.get(key,0)+1
  check('no_prohibited_operations',r['physics_steps']==0 and r['inference_calls']==0 and r['training_calls']==0 and r['evaluation_data_generated'] is False and r['execution_eligible'] is False,str(rp))
  bounds+=receipt['individual_leaf_upper_bounds_verified'];branch_results += [dict(x,cohort_input=str(source)) for x in receipt['nonidentity_branch_results']]
 for source,c in cohorts.items():check('complete_disjoint_native_cohort',set(c['observed'])==set(c['expected']),source)
 check('two_separate_cohorts',len(cohorts)==2)
 check('twenty_native_outputs',sum(len(c['observed']) for c in cohorts.values())==20)
 check('thirteen_source_valid_seven_rejected',sum(c['source_valid'] for c in cohorts.values())==13 and sum(c['source_invalid'] for c in cohorts.values())==7)
 check('sixty_five_finger_decisions',sum(summary.values())==65)
 result={'schema_version':'independent_dga_practical_all20_final_review_v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_STATIC_TESTED_SCOPE','check_count':len(checks),'checks':checks,'source_bindings':bindings,'native_outputs':20,'source_valid_static_outputs':13,'source_invalid_accounted':7,'finger_decision_counts':summary,'individual_leaf_upper_bounds_verified':bounds,'cohorts':{k:{'native_rows':len(c['observed']),'source_valid':c['source_valid'],'source_invalid':c['source_invalid'],'partitions':c['partitions']} for k,c in cohorts.items()},'nonidentity_branch_results':branch_results,'physics_steps':0,'evaluation_data_generated':False,'training_calls':0,'new_native_inference_calls':0,'execution_eligible':False,'scope':'Independent static implementation and conditional complete-cube per-finger worst-distance nonregression relative to explicit legal joint projection. All20 are development/regression outputs, not held-out native evidence. Does not establish absolute native fidelity, pointwise contact preservation or physical adequacy.'}
 with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':result['status'],'check_count':len(checks),'native_outputs':20,'finger_decision_counts':summary,'leaf_bounds_verified':bounds,'output':str(args.output),'sha256':sha(args.output)},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
