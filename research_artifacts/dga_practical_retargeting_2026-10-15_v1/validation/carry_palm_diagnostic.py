from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np
from independent_validator import sha,SOURCE_SHA,TARGET_SHA
R=Path('/private/tmp/dga_practical_validation_20261015_v1');W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction');P=W/'research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_acceptance_feasibility_v2/surface';new=Path('/private/tmp/dga_practical_retargeting_20261015_v1/results_v2/poses.npz');summary=json.load(open(P/'summary_v2.json'))
assert sha(P/'unchanged_angle_poses_v1.npz')==summary['raw_pose_sha256'].removeprefix('sha256:')
with np.load(P/'unchanged_angle_poses_v1.npz',allow_pickle=False)as z:old={k:z[k]for k in z.files}
with np.load(new,allow_pickle=False)as z:a={k:z[k]for k in z.files}
print(old.keys())
rows=[]
for i,g in enumerate(a['global_sample_indices']):
 j=old['global_sample_indices'].tolist().index(int(g));r=next(x for x in summary['records']if x['global_sample_index']==g)
 for key in ['source_hand_qpos','object_from_source_palm','object_from_target_palm']:
  assert np.array_equal(a[key][i],old[key][j]),(key,int(g))
 assert np.array_equal(a['baseline_object_from_target_palm'][i],old['object_from_target_palm'][j])
 rows.append({'global_sample_index':int(g),'raw_and_source_target_palm_bytes_identical':True,'original_palm_nearset_and_occupancy_result':r['palm_strict_nearset_and_occupancy'],'original_palm_family_including_directions':r['palm_family_including_directions'],'interpretation':'Original declared-palm native difference carries unchanged to fixed-palm practical baseline/candidate; finger corrections cannot fix this.'})
result={'schema_version':'dga_practical_native_palm_diagnostic_transfer_v1','records':rows,'prior_scientific_protocol_sha256':summary['scientific_protocol_sha256'],'original_summary':{'path':str(P/'summary_v2.json'),'sha256':sha(P/'summary_v2.json')},'raw_poses_sha256':sha(P/'unchanged_angle_poses_v1.npz'),'new_practical_poses_sha256':sha(new),'target_urdf_sha256':TARGET_SHA,'source_urdf_sha256':SOURCE_SHA,'scope':'Same declared1sourcebox8targetboxes and exact same palm transforms. Palm collision geometry has no fingerjoint dependence. Strict prior FAIL does not become native-fidelity PASS under new pipeline. No recomputation/no new surfacequeries.','physics_steps':0}
(R/'native_palm_diagnostic_transfer_v1.json').write_text(json.dumps(result,indent=2)+'\n');print(sha(R/'native_palm_diagnostic_transfer_v1.json'))
