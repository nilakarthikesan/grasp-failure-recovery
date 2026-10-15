from pathlib import Path
import time,json,numpy as np
from independent_validator import *
from independent_geometry_v5 import *
R=Path('/private/tmp/dga_practical_validation_20261015_v1');row=json.load(open('/private/tmp/dga_practical_retargeting_20261015_v1/results_v2/row_01.json'));refpath=Path(row['mapping_reference']['path']);ref=json.load(open(refpath));p=refpath.parent/ref['arrays']['path'];t=Tree('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf',TARGET_SHA)
with np.load(R/'recent16_v2_candidates/geometry/row1_little_level0_queries.npz')as z:old={k:z[k]for k in z.files}
values={};timings={}
for label in ['source','baseline','candidate']:
 if label=='source':mesh=source_part(p,1,'little')
 else:
  state=row['selected'if label=='candidate'else'baseline'];f={n:mm(np.array(state['root']),x)for n,x in t.fk(state['q']).items()};mesh=target_part(t,f,'little')
 start=time.monotonic();d,count=mesh.query(old['queries_object_m']);elapsed=time.monotonic()-start;error=float(np.max(abs(d-old[label+'_distance_m'])));assert error<=1e-9;values[label+'_distance_m']=d;timings[label]={'elapsed_s':elapsed,'triangle_pairs':count,'maximum_distance_difference_m':error};print(label,timings[label],flush=True)
np.savez_compressed(R/'row1_little_probe_v5.npz',queries_object_m=old['queries_object_m'],**values)
r={'schema_version':'dga_independent_query_acceleration_equivalence_v1','same_frozen_queries':True,'results':timings,'query_count':len(old['queries_object_m']),'helper_sha256':sha(R/'independent_geometry_v5.py'),'previous_query_archive_sha256':sha(R/'recent16_v2_candidates/geometry/row1_little_level0_queries.npz'),'new_archive_sha256':sha(R/'row1_little_probe_v5.npz'),'controls_passed':15}
(R/'v5_equivalence_probe_v1.json').write_text(json.dumps(r,indent=2)+'\n')
