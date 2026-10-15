from pathlib import Path
import numpy as np,json,hashlib
from independent_validator import *
from run_independent_validation import SOURCE,TARGET,ROOT,read,verify,write
s,t=Tree(SOURCE,SOURCE_SHA),Tree(TARGET,TARGET_SHA);c=CompiledKinematics(t,ROOT/'target_compiled_numerical.mjb');records=[];counts={}
for label,path in [('recent16','results_v2'),('original4','results_original_four_regression_v1')]:
 d=Path('/private/tmp/dga_practical_retargeting_20261015_v1')/path;r=read(d/'report.json')
 with np.load(d/'poses.npz',allow_pickle=False)as z:a={k:z[k]for k in z.files}
 count={'native':len(r['records']),'mapped':0,'source_rejected':0}
 for row in r['records']:
  g=row['global_sample_index'];mp=Path(row['native_manifest']['path']);verify(mp,row['native_manifest']['sha256']);m=read(mp);p=mp.parent/m['output']['path'];verify(p,m['output']['sha256'])
  with np.load(p,allow_pickle=False)as z:raw=z['hand_qpos'][row['within_batch_sample_index']]
  viol=s.violations(dict(zip(NAMES,raw[9:],strict=True)));item={'cohort':label,'global_sample_index':g,'source_domain_violations':viol,'raw_sha256':hashlib.sha256(raw.tobytes()).hexdigest()}
  if viol:
   assert g not in a['global_sample_indices'];count['source_rejected']+=1
  else:
   i=a['global_sample_indices'].tolist().index(g);q=a['target_joint_values'][i];b=a['baseline_target_joint_values'][i];root=a['object_from_target_root_world'][i]
   v=validate_saved(raw,q,root,s,t,baseline_joints=b,object_from_input=a['object_from_input'][i],saved_raw=a['source_hand_qpos'][i]);item['independent_candidate']=v;item['compiled']=c.validate(q,root);count['mapped']+=1
   item['independent_baseline']=validate_saved(raw,b,a['baseline_object_from_target_root_world'][i],s,t,baseline_joints=b,object_from_input=a['object_from_input'][i],saved_raw=a['source_hand_qpos'][i])
   for name,e in v['landmark_errors_m'].items():assert abs(e-np.linalg.norm(np.array(row['selected']['errors_m'])[row['site_names'].index(name)]))<1e-12
  records.append(item)
 counts[label]=count
report={'schema_version':'independent_all20_practical_numerical_recount_v1','counts':counts,'records':records,'physics_steps':0,'inference':0,'new_native_holdout':False,'geometry_acceptance_not_assessed':True,'execution_eligible':False,'code_sha256':sha(__file__),'max_compiled_position_m':max(r['compiled']['maximum']['position_m']for r in records if 'compiled'in r),'max_compiled_rotation_rad':max(r['compiled']['maximum']['rotation_rad']for r in records if 'compiled'in r)}
write(ROOT/'numerical_all20_recount_v1.json',report);print(json.dumps({k:report[k]for k in ['counts','max_compiled_position_m','max_compiled_rotation_rad']},indent=2));print(sha(ROOT/'numerical_all20_recount_v1.json'))
