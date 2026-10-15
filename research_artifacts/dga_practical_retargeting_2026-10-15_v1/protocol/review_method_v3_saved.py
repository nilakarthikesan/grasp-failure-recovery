"""Hash and saved-array review of uncapped practical mapping; no geometry queries."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np

def sha(p):return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main(args):
 method=json.loads(args.method.read_text());report=json.loads((args.results/'report.json').read_text());checks=[]
 def check(name,ok,detail=None):
  checks.append({'check':name,'pass':bool(ok),'detail':detail})
  if not ok:raise ValueError(name+': '+str(detail))
 for p,h in method['dependency_sha256'].items():check('frozen_dependency_hash',sha(p)==h,p)
 check('report_method_binding',report['method_sha256']==sha(args.method))
 check('report_protocol_binding',report['protocol_sha256']==method['protocol']['sha256'])
 check('prospective_method_freeze',method['created_at_utc']<report['created_at_utc'])
 for p,h in report['outputs'].items():check('report_output_hash',sha(args.results/p)==h,p)
 before=load(args.previous/'poses.npz');after=load(args.results/'poses.npz')
 for k in ['global_sample_indices','source_hand_qpos','object_from_source_forearm','object_from_source_palm','object_from_input']+[n for n in after if n.startswith('baseline_')]:
  check('native_and_reference_bytes_unchanged',before[k].dtype==after[k].dtype and before[k].shape==after[k].shape and before[k].tobytes()==after[k].tobytes(),k)
 for k,a in after.items():check('finite_pose_array',np.isfinite(a).all(),k)
 old=load(args.previous/'fields.npz');new=load(args.results/'fields.npz');field_recount=[]
 check('field_archive_keys_match',set(old)==set(new))
 for k,a in new.items():
  check('finite_field_array',np.isfinite(a).all(),k)
  if k=='queries_object_m':check('fitting_queries_unchanged',np.array_equal(a,old[k]),k);continue
  check('capped_palm_wrist_fields_unchanged',np.array_equal(a[:,[0,6,7]],old[k][:,[0,6,7]]),k)
  if k.endswith('_source') or k.endswith('_baseline'):
   recapped=np.minimum(a[:,1:6],.004);difference=float(np.max(np.abs(recapped-old[k][:,1:6])))
   check('recapped_finger_fields_match_previous',difference<=1e-9,{'array':k,'max_abs_delta_m':difference})
   field_recount.append({'array':k,'recap_max_abs_delta_m':difference,'largest_uncapped_distance_m':float(a[:,1:6].max())})
 check('source_rejection_count',sum(r['status']=='SOURCE_DOMAIN_REJECTED_UNALTERED' for r in report['records'])==report['accounting']['source_domain_rejected'])
 check('mapped_source_count',len(after['global_sample_indices'])==report['accounting']['source_valid_mapped'])
 check('all_rows_accounted',len(report['records'])==report['accounting']['native_rows'])
 check('zero_prohibited_execution',all(v==0 for v in report['zero_execution'].values()))
 counts={'provisional_corrected_branches':0,'optimizer_exception_branches':0}
 for r in report['records']:
  for b in r.get('trace',{}).get('branches',[]):
   counts['provisional_corrected_branches']+=bool(b['adopted_correction']);counts['optimizer_exception_branches']+=bool(b.get('optimizer_exception'))
   if b['adopted_correction']:
    for key in ['maximum_landmark_error_m','landmark_SSE_m2','maximum_unsigned_field_error_m']:
     check('frozen_fitting_nonregression',b['selected_metrics'][key]<=b['baseline_metrics'][key],[r['global_sample_index'],b['part'],key])
 result={'schema_version':'independent_practical_uncapped_method_saved_review_v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),'method_sha256':sha(args.method),'report_sha256':sha(args.results/'report.json'),'previous_report_sha256':sha(args.previous/'report.json'),'status':'PASS','accounting':report['accounting'],'counts':counts,'checks':checks,'check_count':len(checks),'field_recount':field_recount,'geometry_queries':0,'physics_steps':0,'native_inference_calls':0,'training_calls':0,'scope':'Provenance, unchanged raw/reference bytes, uncapped-versus-capped field consistency and provisional fitting guards. Final independent surface admission is not inferred.'}
 with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'PASS','checks':len(checks),'output':str(args.output),'sha256':sha(args.output),'counts':counts},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['method','results','previous','output']:p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
