"""Additional independent FK/metric recount of the sole admitted correction.
No mapper imports, optimization, native inference, simulation, or mesh queries.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,xml.etree.ElementTree as ET
import numpy as np
P=Path('/private/tmp/dga_practical_protocol_20261015_v1')
V=Path('/private/tmp/dga_practical_validation_20261015_v1')
M=Path('/private/tmp/dga_practical_retargeting_20261015_v1')
TARGET=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf')
NAMES=('WRJ2','WRJ1','FFJ4','FFJ3','FFJ2','FFJ1','MFJ4','MFJ3','MFJ2','MFJ1','RFJ4','RFJ3','RFJ2','RFJ1','LFJ5','LFJ4','LFJ3','LFJ2','LFJ1','THJ5','THJ4','THJ3','THJ2','THJ1')
PARTS={'index':'ff','middle':'mf','ring':'rf','little':'lf','thumb':'th'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def arr(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def rotation(axis,angle):
 axis=np.array(axis,dtype=float);axis/=math.sqrt(sum(x*x for x in axis));x,y,z=axis;cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
 return np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*(cross@cross)
def origin(e):
 t=np.eye(4)
 if e is None:return t
 r,p,y=map(float,e.get('rpy','0 0 0').split());t[:3,:3]=rotation([0,0,1],y)@rotation([0,1,0],p)@rotation([1,0,0],r);t[:3,3]=list(map(float,e.get('xyz','0 0 0').split()));return t
def fk(path,q,root):
 xml=ET.parse(path).getroot();joints={e.find('child').get('link'):e for e in xml.findall('joint')};roots={e.get('name') for e in xml.findall('link')}-set(joints);assert len(roots)==1
 frames={next(iter(roots)):root.copy()}
 def get(name):
  if name not in frames:
   e=joints[name];t=origin(e.find('origin'))
   if e.get('type')=='revolute':
    motion=np.eye(4);motion[:3,:3]=rotation(list(map(float,e.find('axis').get('xyz').split())),q[e.get('name')]);t=t@motion
   else:assert e.get('type')=='fixed'
   frames[name]=get(e.find('parent').get('link'))@t
  return frames[name]
 for name in joints:get(name)
 return frames
checks=[];bindings={str(Path(__file__)):sha(__file__)}
def check(name,ok,detail=None):
 checks.append({'check':name,'pass':bool(ok),'detail':detail})
 if not ok:raise ValueError(name+': '+str(detail))
def bind(p):bindings[str(p)]=sha(p)
rpath=V/'recent16_authoritative_v3_part_a/row_04.json';r=read(rpath);bind(rpath)
mpath=M/'results_v3/report.json';mr=read(mpath)['records'][4];bind(mpath)
manifest=Path(r['native_manifest']);native=read(manifest);bind(manifest)
src=manifest.parent/native['source_hand']['path'];bind(src);bind(TARGET)
check('pinned_source_urdf',sha(src)==native['source_hand']['sha256'].removeprefix('sha256:'))
check('pinned_target_urdf',sha(TARGET)==r['final']['target_urdf_sha256'])
a=M/'results_v3/poses.npz';b=V/'recent16_authoritative_v3_part_a/final_static_poses.npz';bind(a);bind(b);poses=arr(a);finals=arr(b)
i=poses['global_sample_indices'].tolist().index(4);j=finals['global_sample_indices'].tolist().index(4);raw=poses['source_hand_qpos'][i];nativeq=dict(zip(NAMES,raw[9:]))
unit=lambda x:np.array(x)/math.sqrt(sum(float(v)*float(v) for v in x))
x=unit(raw[3:6]);y=unit(raw[6:9]);a=unit(x+y);b=unit(x-y);x=unit(a+b);y=unit(a-b)
root=np.eye(4);root[:3,:3]=np.column_stack([x,y,unit(np.cross(x,y))]);root[:3,3]=raw[:3]
check('object_frame_identity',np.array_equal(native['object_from_input'],np.eye(4)))
sf=fk(src,nativeq,root);metrics={};frames={}
for label,vals in [('baseline',poses['baseline_target_joint_values'][i]),('candidate',poses['target_joint_values'][i]),('final',finals['target_joint_values'][j])]:
 q=dict(zip(NAMES,vals));tf=fk(TARGET,q,poses['object_from_target_root_world'][i]);frames[label]=tf
 check('independent_scalar_FK_matches_saved',max(np.max(abs(tf[n]-np.array(r[label]['target_object_from_links'][n]))) for n in tf)<1e-14,label)
 lengths={n:float(np.linalg.norm(tf[n][:3,3]-sf[n][:3,3])) for n in r[label]['landmark_sites']}
 check('recomputed_landmarks_match_report',max(abs(lengths[n]-r[label]['landmark_errors_m'][n]) for n in lengths)<1e-14,label)
 metrics[label]={}
 for part,prefix in PARTS.items():
  v=np.array([d for n,d in lengths.items() if n.startswith(prefix)]);value={'max_m':float(max(v)),'sse_m2':float(v@v),'rms_m':float(np.sqrt(v@v/len(v)))};metrics[label][part]=value
  check('independent_per_finger_metrics',all(abs(value[k]-r[label]['landmark_per_part'][part][k])<1e-14 for k in value),[label,part])
for part in PARTS:
 check('independent_final_max_and_SSE_nonregression',all(metrics['final'][part][k]<=metrics['baseline'][part][k] for k in ['max_m','sse_m2']),part)
check('independent_thumb_strict_landmark_improvement',all(metrics['final']['thumb'][k]<metrics['baseline']['thumb'][k] for k in ['max_m','sse_m2']))
selected=poses['baseline_target_joint_values'][i].copy();selected[19:24]=poses['target_joint_values'][i,19:24]
check('final_exact_thumb_only_candidate_adoption',np.array_equal(selected,finals['target_joint_values'][j]))
check('unmodified_other_branch_geometry',all(np.array_equal(frames['final'][n],frames['baseline'][n]) for n in frames['final'] if not n.startswith('th')))
# Verify the saved helper includes the exact target visual primitive. Merely
# loading/triangulating geometry does not perform triangle queries or physics.
import sys
sys.path.insert(0,str(V))
import independent_geometry_v5 as helper
from independent_validator import Tree,TARGET_SHA
policy=read(V/'validation_policy_frozen_v5.json');bind(V/'validation_policy_frozen_v5.json')
for path,h in policy['code_sha256'].items():bind(path);check('frozen_validator_code',sha(path)==h.removeprefix('sha256:'),path)
tree=Tree(TARGET,TARGET_SHA);mesh=helper.target_part(tree,frames['final'],'thumb');e=tree.xml.find("link[@name='thbase']/visual");size=np.array(list(map(float,e.find('geometry/box').get('size').split())))
check('thumbbase_declared_1mm_visual_box',np.array_equal(size,[.001,.001,.001]))
vertex,face=helper.box_boundary(size);transform=frames['final']['thbase']@origin(e.find('origin'));expected=vertex@transform[:3,:3].T+transform[:3,3]
check('thumbbase_all_eight_transformed_vertices_present',all(np.min(np.linalg.norm(mesh.vertices-v,axis=1))<1e-15 for v in expected))
actual=mesh.triangles
check('thumbbase_all_twelve_boundary_triangles_present',all(np.min(np.max(np.linalg.norm(actual-expected[f][None],axis=2),axis=1))<1e-15 for f in face))
counts={}
for link in tree.xml.findall('link'):
 if link.get('name').startswith('th'):
  counts[link.get('name')]=len(link.findall('visual'))
check('no_target_visual_omitted_by_part_loader',sum(len(helper.obj(tree.path.parent/v.find('geometry/mesh').get('filename'))[1]) if v.find('geometry/mesh') is not None else 12 for link in tree.xml.findall('link') if link.get('name').startswith('th') for v in link.findall('visual'))==len(mesh.faces))
res={'schema_version':'independent_positive_thumb_review_v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','checks':checks,'check_count':len(checks),'source_bindings':bindings,'metrics_independently_recomputed':metrics,'target_thumb_visual_counts':counts,'target_thumb_triangle_count':len(mesh.faces),'geometry_supremum_interval_m':{k:r['branches']['thumb'][k] for k in ['baseline','candidate']},'full_surface_archive_recount_receipt':str(P/'recent16_part_a_v3_complete_geometry_recount_v2.json'),'scope':'Independent scalar Rodrigues/RPY FK and metrics; exact mixed-q branch choice; explicit inclusion of all target thumb visuals including 1mm box. Conditional fullsurface interval arithmetic separately recounted. Does not establish pointwise nonregression, native fidelity, or physical acceptance.','new_triangle_queries':0,'physics_steps':0,'training_calls':0,'inference_calls':0}
receipt=P/'recent16_part_a_v3_complete_geometry_recount_v2.json';bind(receipt)
out=P/'positive_row4_thumb_independent_review_v1.json'
with out.open('x') as f:json.dump(res,f,indent=2);f.write('\n')
print(json.dumps({'status':'PASS','check_count':len(checks),'output':str(out),'sha256':sha(out),'thumb_metrics':{k:v['thumb'] for k,v in metrics.items()}},indent=2))
