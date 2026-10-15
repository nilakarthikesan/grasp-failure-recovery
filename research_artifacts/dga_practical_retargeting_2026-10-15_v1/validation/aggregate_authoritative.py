"""Aggregate disjoint complete static validation partitions, never rescore outputs."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,numpy as np
from independent_validator import sha
R=Path('/private/tmp/dga_practical_validation_20261015_v1')
PARTITIONS={'recent16':[R/'recent16_authoritative_v3_part_a',R/'recent16_authoritative_v3_part_b'],'original4':[R/'original4_authoritative_v3']}
rows=[];counts=Counter();inputs={};native_count={};pose_arrays=[]
for cohort,dirs in PARTITIONS.items():
 seen=[]
 for directory in dirs:
  p=directory/'report.json';r=json.load(open(p));inputs[str(p)]=sha(p)
  assert r['physics_steps']==r['inference_calls']==r['training_calls']==0 and r['execution_eligible'] is False
  assert r['policy_sha256']==sha(R/'validation_policy_frozen_v5.json')
  for name,digest in r['input_hashes'].items():assert sha(name)==digest.removeprefix('sha256:')
  for name,digest in r['output_hashes'].items():assert sha(directory/name)==digest
  for path,digest in r['geometry_dependency_hashes'].items():assert sha(path)==digest.removeprefix('sha256:')
  with np.load(directory/'final_static_poses.npz',allow_pickle=False)as z:arr={k:z[k]for k in z.files}
  for j,g in enumerate(arr['global_sample_indices']):pose_arrays.append((cohort,int(g),{k:arr[k][j]for k in arr if k!='global_sample_indices'}))
  for original in r['records']:
   row=dict(original);row['cohort']=cohort;row['authoritative_partition']=str(directory);rows.append(row);seen.append(row['global_sample_index'])
  counts.update(r['counts'])
 expected=list(range(16 if cohort=='recent16' else 4));assert sorted(seen)==expected and len(seen)==len(set(seen));native_count[cohort]=len(seen)
valid=[r for r in rows if r['status']=='STATIC_RELATIVE_BASELINE_GATE_PASS'];rejected=[r for r in rows if r['status']=='SOURCE_DOMAIN_REJECTED_UNALTERED'];assert len(valid)==13 and len(rejected)==7
worst=max(r['final']['landmark_aggregate']['max_m']for r in valid);baseline_worst=max(r['baseline']['landmark_aggregate']['max_m']for r in valid)
comparison=[]
for r in valid:
 comparison.append({'cohort':r['cohort'],'global_sample_index':r['global_sample_index'],'baseline_max_landmark_mm':r['baseline']['landmark_aggregate']['max_m']*1000,'final_max_landmark_mm':r['final']['landmark_aggregate']['max_m']*1000,'baseline_rms_landmark_mm':r['baseline']['landmark_aggregate']['rms_m']*1000,'final_rms_landmark_mm':r['final']['landmark_aggregate']['rms_m']*1000,'baseline_sse_mm2':r['baseline']['landmark_aggregate']['sse_m2']*1e6,'final_sse_mm2':r['final']['landmark_aggregate']['sse_m2']*1e6,'certified_corrected_branches':[p for p,x in r['branches'].items()if x.get('adopt_candidate')],'final_per_finger_max_mm':{p:m['max_m']*1000 for p,m in r['final']['landmark_per_part'].items()}})
pose_arrays.sort(key=lambda x:(x[0],x[1]));np.savez(R/'authoritative_all20_final_static_poses.npz',cohort=np.array([a[0]for a in pose_arrays]),global_sample_indices=np.array([a[1]for a in pose_arrays]),**{k:np.stack([a[2][k]for a in pose_arrays])for k in pose_arrays[0][2]})
report={'schema_version':'dga_practical_authoritative_all20_static_summary_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'native_accounting_by_cohort':native_count,'counts':dict(counts),'all_native_rows_once':True,'source_valid_mapped':13,'source_invalid_retained':7,'source_invalid_indices_by_cohort':{c:[r['global_sample_index']for r in rejected if r['cohort']==c]for c in PARTITIONS},'cohort_accounting':{c:{'native_total':native_count[c],'source_valid_mapped':sum(r['cohort']==c for r in valid),'source_invalid_retained':sum(r['cohort']==c for r in rejected)}for c in PARTITIONS},'max_final_per_pose_rms_m':max(r['final']['landmark_aggregate']['rms_m']for r in valid),'max_baseline_per_pose_rms_m':max(r['baseline']['landmark_aggregate']['rms_m']for r in valid),'pooled_final_rms_m':float(np.sqrt(sum(r['final']['landmark_aggregate']['sse_m2']for r in valid)/(28*len(valid)))),'pooled_baseline_rms_m':float(np.sqrt(sum(r['baseline']['landmark_aggregate']['sse_m2']for r in valid)/(28*len(valid)))),'max_final_landmark_m':worst,'max_baseline_landmark_m':baseline_worst,'per_case_summary':comparison,'max_final_compiled_position_m':max(r['final_compiled']['maximum']['position_m']for r in valid),'max_final_compiled_rotation_rad':max(r['final_compiled']['maximum']['rotation_rad']for r in valid),'input_partition_reports':inputs,'policy_sha256':sha(R/'validation_policy_frozen_v5.json'),'final_pose_archive_sha256':sha(R/'authoritative_all20_final_static_poses.npz'),'scientific_scope':'Static representation correctness and conditional fullcube nonregression relativeexplicitprojectionreference only; no absolute native fidelity, unseenobject/nativeholdoutgeneralization, physics/controller/label adequacy or executionpermission.','native_development_only':True,'execution_eligible':False,'physics_steps':0,'inference_calls':0,'evaluation_data_generated':False,'training_calls':0,'records':rows}
(R/'authoritative_all20_summary_v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['counts','source_valid_mapped','source_invalid_retained','max_final_landmark_m','max_baseline_landmark_m','max_final_compiled_position_m','max_final_compiled_rotation_rad']},indent=2));print('SHA',sha(R/'authoritative_all20_summary_v1.json'))
