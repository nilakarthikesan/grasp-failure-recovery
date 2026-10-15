import json,time
from pathlib import Path
import numpy as np
from independent_validator import *
from independent_geometry_v5 import *
R=Path('/private/tmp/dga_practical_retargeting_20261015_v1/results_v1');OUT=Path('/private/tmp/dga_practical_validation_20261015_v1')
row=json.loads((R/'row_00.json').read_text());reference=json.loads(Path(row['mapping_reference']['path']).read_text());ref=Path(row['mapping_reference']['path']).parent/reference['arrays']['path'];target=Tree('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf',TARGET_SHA)
p,rho=grid(.001);start=time.monotonic();s=source_part(ref,0,'ring');ds,ns=s.query(p);print('source',time.monotonic()-start,ns,flush=True)
for stage in ['baseline','selected']:
 state=row[stage];frames={n:mm(np.array(state['root']),t) for n,t in target.fk(state['q']).items()};mesh=target_part(target,frames,'ring');start=time.monotonic();d,n=mesh.query(p);print(stage,time.monotonic()-start,n,max_interval(ds,d,rho),flush=True)
 np.savez_compressed(OUT/(stage+'_row0_ring_probe_1mm_v5.npz'),queries_object_m=p,source_distance_m=ds,target_distance_m=d)
