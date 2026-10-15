"""Bind carried prior palm diagnostics to the independently selected final states."""
from pathlib import Path
import json
import numpy as np
from independent_validator import sha
R=Path('/private/tmp/dga_practical_validation_20261015_v1')
M=Path('/private/tmp/dga_practical_retargeting_20261015_v1')
old=json.load(open(R/'native_palm_diagnostic_transfer_v1.json'))
summary=json.load(open(R/'authoritative_all20_summary_v1.json'))
with np.load(R/'authoritative_all20_final_static_poses.npz',allow_pickle=False) as z:final={k:z[k]for k in z.files}
with np.load(M/'results_v3/poses.npz',allow_pickle=False) as z:new={k:z[k]for k in z.files}
with np.load(M/'results_v2/poses.npz',allow_pickle=False) as z:before={k:z[k]for k in z.files}
rows=[]
for i,g in enumerate(new['global_sample_indices']):
 j=np.where((final['cohort']=='recent16')&(final['global_sample_indices']==g))[0].item()
 assert np.array_equal(final['source_hand_qpos'][j],new['source_hand_qpos'][i])
 assert np.array_equal(final['source_hand_qpos'][j],before['source_hand_qpos'][i])
 assert np.array_equal(final['object_from_target_root_world'][j],new['baseline_object_from_target_root_world'][i])
 assert np.array_equal(final['object_from_target_root_world'][j],before['baseline_object_from_target_root_world'][i])
 assert np.array_equal(final['target_joint_values'][j][:2],new['baseline_target_joint_values'][i][:2])
 assert np.array_equal(final['target_joint_values'][j][:2],before['baseline_target_joint_values'][i][:2])
 for k in ['object_from_source_palm','object_from_target_palm','baseline_object_from_target_palm']:
  assert np.array_equal(new[k][i],before[k][i])
 r=next(x for x in old['records'] if x['global_sample_index']==g)
 rows.append(dict(r,final_root_and_two_wrists_byte_identical=True,final_raw_byte_identical=True))
result={'schema_version':'dga_practical_final_palm_diagnostic_identity_v2','status':'PASS_IDENTITY_TRANSFER','inputs':{str(p):sha(p)for p in [R/'native_palm_diagnostic_transfer_v1.json',R/'authoritative_all20_summary_v1.json',R/'authoritative_all20_final_static_poses.npz',M/'results_v3/poses.npz',M/'results_v2/poses.npz']},'records':rows,'scope':'For all11 recent sourcevalid native poses, final root and wrist q exactly equal the prior practical and raw-reference fixed-palm geometry. Finger changes do not affect palm. Therefore prior palm overlap failures and unresolved directions persist unchanged. Original-four palm geometry not asserted by this receipt. No surface queries, physics, inference, evaluation data or training.'}
output=R/'final_palm_diagnostic_identity_v2.json';output.write_text(json.dumps(result,indent=2)+'\n');print(sha(output))
