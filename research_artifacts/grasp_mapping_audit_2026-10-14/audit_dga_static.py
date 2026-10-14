import hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

SRC=Path('/private/tmp/DexGrasp-Anything-86353d710ca71d0f547f0d8b15cc8ca959c78163/assets/urdf/sr_grasp_description/urdf/shadowhand.urdf')
DST=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf')
OUT=Path('/private/tmp/grasp-mapping-audit-2026-10-14')
OUT.mkdir(exist_ok=True)
def vec(e,k):return np.fromstring(e.get(k,'0 0 0'),sep=' ') if e is not None else np.zeros(3)
def rot(v):
    a=np.linalg.norm(v)
    if a==0:return np.eye(3)
    x,y,z=v/a; K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)
def joints(p):
    root=ET.parse(p).getroot();result={}
    for e in root.findall('joint'):
        o=e.find('origin');r=vec(o,'rpy');limit=e.find('limit')
        result[e.get('name')]={'type':e.get('type'),'parent':e.find('parent').get('link'),'child':e.find('child').get('link'),'xyz':vec(o,'xyz'),'rotation':rot(np.array([0,0,r[2]]))@rot(np.array([0,r[1],0]))@rot(np.array([r[0],0,0])),'axis':vec(e.find('axis'),'xyz'),'limits':None if limit is None else [float(limit.get('lower','0')),float(limit.get('upper','0'))]}
    return result
def fk(js,q):
    roots={v['parent'] for v in js.values()}-{v['child'] for v in js.values()};poses={x:np.eye(4) for x in roots}
    for _ in range(len(js)+1):
        for name,j in js.items():
            if j['parent'] not in poses or j['child'] in poses:continue
            T=np.eye(4);T[:3,:3]=j['rotation'];T[:3,3]=j['xyz']
            if j['type'] in ('revolute','continuous'):T[:3,:3]=T[:3,:3]@rot(j['axis']*q.get(name,0))
            poses[j['child']]=poses[j['parent']]@T
    return poses
s,t=joints(SRC),joints(DST); sn={k for k,j in s.items() if j['type']=='revolute'};tn={k for k,j in t.items() if j['type']=='revolute'}
diff=[]
for name in sorted(sn & tn):
    a,b=s[name],t[name];fields=[]
    for k in ['parent','child','type']:
        if a[k]!=b[k]:fields.append(k)
    for k in ['xyz','rotation','axis','limits']:
        if not np.allclose(a[k],b[k],atol=1e-9,rtol=0):fields.append(k)
    if fields:diff.append({'joint':name,'different_fields':fields})
names=sorted(sn&tn);rng=np.random.default_rng(20261014)
poses=[{k:0.0 for k in names}]
for _ in range(64):poses.append({k:float(rng.uniform(max(s[k]['limits'][0],t[k]['limits'][0]),min(s[k]['limits'][1],t[k]['limits'][1]))) for k in names})
tips=['thtip','fftip','mftip','rftip','lftip'];errs=[];palms=[]
for q in poses:
    sf,tf=fk(s,q),fk(t,q)
    # Compensate root conventions using each actual palm frame, preserving source object-to-palm.
    sp=np.linalg.inv(sf['palm']);tp=np.linalg.inv(tf['palm'])
    errs.append([float(np.linalg.norm((sp@sf[n])[:3,3]-(tp@tf[n])[:3,3])) for n in tips]);palms.append(sf['palm'][:3,3].tolist())
report={'date':'2026-10-14','source_commit':'86353d710ca71d0f547f0d8b15cc8ca959c78163','source_urdf_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'target_urdf_sha256':hashlib.sha256(DST.read_bytes()).hexdigest(),'kind':'synthetic_static_URDF_FK_audit_no_generator_no_physics','source_joint_count':len(sn),'target_joint_count':len(tn),'source_only_joints':sorted(sn-tn),'target_only_joints':sorted(tn-sn),'joint_differences':diff,'pose_count':len(poses),'sampling_seed':20261014,'fingertip_order':tips,'fingertip_max_error_m':np.max(errs,axis=0).tolist(),'mean_tip_error_m':float(np.mean(errs)),'maximum_tip_error_m':float(np.max(errs)),'source_forearm_to_palm_zero_translation_m':palms[0],'source_forearm_to_palm_translation_range_m':np.ptp(palms,axis=0).tolist(),'surface_equivalence':'not_tested','rotation_export':'six_dimensional_columns_after_inverse_random_object_rotation','native_generated_pose':'unavailable'}
(OUT/'dga_static_fk.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
