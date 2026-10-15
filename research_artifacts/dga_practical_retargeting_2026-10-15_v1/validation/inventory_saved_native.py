"""Read-only, no-inference inventory of project-owned DGA output archives."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
BASE=Path('/Users/nilakarthikesan/grasp-failure-prediction/local_assets/dexgrasp_anything')
WORK=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OUT=Path('/private/tmp/dga_practical_validation_20261015_v1')
SOURCE=BASE/'source-86353d710ca71d0f547f0d8b15cc8ca959c78163/assets/urdf/sr_grasp_description/urdf/shadowhand.urdf'
TARGET=WORK/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf'
NAMES=('WRJ2','WRJ1','FFJ4','FFJ3','FFJ2','FFJ1','MFJ4','MFJ3','MFJ2','MFJ1','RFJ4','RFJ3','RFJ2','RFJ1','LFJ5','LFJ4','LFJ3','LFJ2','LFJ1','THJ5','THJ4','THJ3','THJ2','THJ1')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def limits(p):
    return {j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(p).getroot().findall('joint') if j.get('type')=='revolute'}
sl,tl=limits(SOURCE),limits(TARGET)
def rejected(q,lim): return [n for n,v in zip(NAMES,q[9:],strict=True) if not lim[n][0]<=v<=lim[n][1]]
def group(p):
    x=str(p.relative_to(BASE)).split('/')[0]
    if x.startswith('independent_native'): return 'multidex_16_development'
    if x.startswith('multidex'): return 'multidex_4_development'
    if x.startswith('native_cohort'): return 'dexgraspnet_16_original'
    if x.startswith('native_smoke'): return 'dexgraspnet_2_original'
    return 'precision_control_duplicate_only'
files=[];unique={};groups=defaultdict(set)
for p in sorted(BASE.rglob('outputs.npz')):
    with np.load(p,allow_pickle=False) as z:
        if 'hand_qpos' not in z: continue
        arr=z['hand_qpos']
    assert arr.ndim==2 and arr.shape[1]==33 and arr.dtype==np.float64 and np.isfinite(arr).all()
    rowhash=[]
    for i,q in enumerate(arr):
        digest=hashlib.sha256(q.tobytes()).hexdigest();rowhash.append(digest)
        record={'raw33_sha256':digest,'source_invalid_joints':rejected(q,sl),'target_invalid_joints':rejected(q,tl),'occurrences':[]}
        unique.setdefault(digest,record)['occurrences'].append({'path':str(p),'index':i,'group':group(p)})
        groups[group(p)].add(digest)
    files.append({'path':str(p),'file_sha256':sha(p),'shape':list(arr.shape),'group':group(p),'fp32_geometry_reference_copy':'fp32_references' in p.parts,'row_sha256':rowhash})
summary={}
for name,hashes in groups.items():
    rows=[unique[k] for k in hashes]
    summary[name]={'unique_rows':len(rows),'source_valid':sum(not x['source_invalid_joints'] for x in rows),'source_and_target_valid':sum(not x['source_invalid_joints'] and not x['target_invalid_joints'] for x in rows)}
report={'schema_version':'dga_practical_existing_native_inventory_v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Only project DGA runtime asset outputs.npz archives; no unrelated files','source_urdf':{'path':str(SOURCE),'sha256':sha(SOURCE)},'target_urdf':{'path':str(TARGET),'sha256':sha(TARGET)},'archive_count':len(files),'unique_native_raw33_rows':len(unique),'groups':summary,'files':files,'rows':list(unique.values()),'independent_native_holdout_available':False,'object_coverage':'one previously inspected 50 mm cube; no new object generalization evidence','interpretation':['FP32 geometry reference exports are raw-pose duplicates, not independent native samples.','All 20 MultiDex outputs have been used or inspected during mapping development.','Original DexGraspNet checkpoint samples remain a separate generator variant.','New manufactured pose/frame/boundary controls test engineering behavior only, not native output or object generalization.'],'physics_steps':0,'inference_calls':0,'evaluation_data_generated':False,'training_calls':0}
p=OUT/'saved_native_inventory_v1.json';p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'path':str(p),'sha256':sha(p),'archive_count':len(files),'unique_rows':len(unique),'groups':summary},indent=2))
