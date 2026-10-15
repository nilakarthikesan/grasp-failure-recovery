from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/private/tmp/compiled_hand_audit_20261015_v1/shadow');report=json.loads((R/'report.json').read_text());records=json.loads((R/'target_pose_records.json').read_text())['records']
rows=[]
for row in records:
 t=np.eye(4);t[:3,:3]=Rotation.from_quat(row['object_to_palm']['quaternion_xyzw']).as_matrix();t[:3,3]=row['object_to_palm']['translation_m']
 if row['variant']=='original_native':status='ORIGINAL_CANONICAL_FRAME_FAILURE_PRESERVED'
 elif not row['target_legal']:status='TARGET_JOINT_LIMIT_REJECTED_DIAGNOSTIC_ONLY'
 elif row['generator']=='unidexgrasp':status='SAVED_CONSUMER_VALID_SOURCE_LEGAL' if row['source_legal'] else 'SAVED_CONSUMER_VALID_SOURCE_ILLEGAL'
 else:status='OFFICIAL_SAVED_RECORD_STATIC_DOMAIN_LEGAL_NOT_NEW_ACCEPTANCE'
 rows.append({'case_id':row['id'],'generator':row['generator'],'variant':row['variant'],'original_status':status,'target_joint_targets_rad':row['joint_targets_rad'],'object_from_target_palm':t.tolist(),'source_joint_targets_rad':row['source_joint_targets_rad'],'source_hand_path':row['source_hand_path'],'source_legal':row['source_legal'],'target_legal':row['target_legal'],'source_tip_vectors_palm_m':row['source_tip_vectors_palm_m'],'target_tip_vectors_palm_m':row['target_tip_vectors_palm_m'],'source_tips_object_m':row['source_tips_object_m'],'input_file_hashes':report['inputs'],'original_record_index':row['native_row'],'seed':row['seed'],'object_scale':row['object_scale']})
p=R/'target_cases_v1.json';p.write_text(json.dumps(rows,indent=2)+'\n');print(len(rows),hashlib.sha256(p.read_bytes()).hexdigest())
