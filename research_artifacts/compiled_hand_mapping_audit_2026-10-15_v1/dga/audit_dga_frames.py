"""Saved pose conversion audit: own XML FK; no inference, fitting or physics."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
B=W/'research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_component_constraints_v1'
OUT=Path('/private/tmp/compiled_hand_audit_20261015_v1/dga')
sys.path.insert(0,str(W/'src'))
from grasp_failure_prediction.proposals.dexgrasp_anything import decode_dga_hand_pose,map_dga_candidate,DGAContractError

def sha(p): return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def record(p): return json.loads(Path(p).read_text())
def maxerr(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def assertclose(a,b):
 e=maxerr(a,b);assert e<1e-12,e;return e

def vector(node,name,default='0 0 0'):
 return np.fromstring(default if node is None else node.get(name,default),sep=' ')
def origin(node):
 M=np.eye(4);M[:3,:3]=Rotation.from_euler('xyz',vector(node,'rpy')).as_matrix();M[:3,3]=vector(node,'xyz');return M
class Tree:
 def __init__(self,p):
  self.p=Path(p);self.xml=ET.parse(p).getroot();self.links=[e.get('name') for e in self.xml.findall('link')];self.j={}
  for e in self.xml.findall('joint'):
   lim=e.find('limit');self.j[e.get('name')]={'parent':e.find('parent').get('link'),'child':e.find('child').get('link'),'origin':origin(e.find('origin')),'axis':vector(e.find('axis'),'xyz','1 0 0'),'type':e.get('type'),'limits':None if e.get('type')=='fixed' else (float(lim.get('lower')),float(lim.get('upper')))}
  roots=set(self.links)-{r['child'] for r in self.j.values()};assert len(roots)==1;self.root=roots.pop()
 def fk(self,q):
  f={self.root:np.eye(4)};pending=dict(self.j)
  while pending:
   before=len(pending)
   for n,j in list(pending.items()):
    if j['parent'] not in f: continue
    M=j['origin'].copy()
    if j['type']!='fixed':
     assert j['type']=='revolute';axis=j['axis']/np.linalg.norm(j['axis']);M[:3,:3]=M[:3,:3]@Rotation.from_rotvec(axis*q[n]).as_matrix()
    f[j['child']]=f[j['parent']]@M;del pending[n]
   assert len(pending)<before
  return f
 def violations(self,q):return {n:q[n] for n,j in self.j.items() if j['limits'] and not j['limits'][0]<=q[n]<=j['limits'][1]}

ledgerp=B/'native_validation/results_v1/native_validation_report.json';fitp=B/'native_mapping_results_v1/native_component_fit_report.json'
ledger=record(ledgerp);fit=record(fitp)
firstm=Path(ledger['records'][0]['native_manifest']['path']);manifest=record(firstm)
sourcep=firstm.parent/manifest['source_hand']['path'];targetp=W/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf'
S,T=Tree(sourcep),Tree(targetp)
assert S.root=='forearm' and T.root=='world'
assert sha(sourcep)=='sha256:932fca7c411ef9ae4e6a8f17a892b3629840b5b5e1254610e0616e184d1f95c7'
assert sha(targetp)=='sha256:29a50063cced2fba81dfe3d208ba54499fc12635c9189d0822f7caa2771c36de'
names=manifest['joint_names'];assert len(names)==24
inputs={str(p):ref(p) for p in [ledgerp,fitp,sourcep,targetp,W/'src/grasp_failure_prediction/proposals/dexgrasp_anything.py',Path(__file__)]}
cases=[];checks=[];provider_counts={};source_frames={}
for r in ledger['records']:
 i=r['global_sample_index'];mp=Path(r['native_manifest']['path']);m=record(mp);assert sha(mp)==r['native_manifest']['sha256'];rawp=mp.parent/m['output']['path'];assert sha(rawp)==m['output']['sha256']==r['raw_output_sha256']
 inputs[str(mp)]=ref(mp);inputs[str(rawp)]=ref(rawp)
 with np.load(rawp,allow_pickle=False) as z:
  if i==0: print('rawkeys',list(z))
  raw=np.asarray(z['hand_qpos'])[r['within_batch_sample_index']]
 assert 'sha256:'+hashlib.sha256(raw.tobytes(order='C')).hexdigest()==r['raw33_row_sha256']
 assert raw.shape==(33,)
 q=dict(zip(names,map(float,raw[9:])));assert q==r['joint_targets_rad']
 # independent symmetric column orthogonalization (not Gram-Schmidt)
 a,b=raw[3:6]/np.linalg.norm(raw[3:6]),raw[6:9]/np.linalg.norm(raw[6:9]);u=(a+b)/np.linalg.norm(a+b);v=(a-b)/np.linalg.norm(a-b)
 c1,c2=(u+v)/np.sqrt(2),(u-v)/np.sqrt(2)
 input_forearm=np.eye(4);input_forearm[:3,:3]=np.column_stack((c1,c2,np.cross(c1,c2)));input_forearm[:3,3]=raw[:3]
 object_forearm=np.asarray(m['object_from_input'])@input_forearm
 sf=S.fk(q);tf=T.fk(q);native={n:object_forearm@M for n,M in sf.items()};palm=native['palm'];root=palm@np.linalg.inv(tf['palm']);target={n:root@M for n,M in tf.items()};source_frames[i]=native
 se=assertclose(object_forearm,r['object_from_source_forearm']);pe=assertclose(palm,r['object_from_palm']);assertclose(palm,r['algebra']['transforms']['object_from_palm'])
 sv=S.violations(q);tv=T.violations(q)
 assert set(sv)=={v['joint'] for v in r['source_limit_violations']};assert set(tv)=={v['joint'] for v in r['target_limit_violations']}
 try:
  decoded=decode_dga_hand_pose(raw,object_from_input=m['object_from_input'],source_hand_urdf=sourcep)
  for n,M in decoded.object_from_links.items():se=max(se,assertclose(M,native[n]))
  mapped=map_dga_candidate(decoded,target_hand_urdf=targetp)
  assert not sv and not tv
  pe=max(pe,assertclose(mapped['object_from_target_root'],root));pstatus='CANDIDATE_ONLY_LIMIT_VALID'
 except DGAContractError as e:
  assert sv or tv
  assert e.code=='joint_limit_violation',e.code
  pstatus='REJECTED_SOURCE_LIMIT' if sv else 'REJECTED_TARGET_LIMIT'
 provider_counts[pstatus]=provider_counts.get(pstatus,0)+1
 cases.append({'case_id':f'dga_raw_multidex_{i:02d}','generator':'dexgrasp_anything','variant':'raw_native_palm_compensated_diagnostic','global_sample_index':i,'original_status':r['mapping_status'],'target_joint_targets_rad':q,'object_from_target_palm':palm.tolist(),'object_from_target_root':root.tolist(),'object_from_target_forearm':target['forearm'].tolist(),'source_fk_reference':{n:M.tolist() for n,M in native.items()},'target_fk_reference':{n:M.tolist() for n,M in target.items()},'source_limit_violations':sv,'target_limit_violations':tv,'input_file_hashes':[ref(mp),ref(rawp),ref(sourcep),ref(targetp)],'execution_eligible':False,'full_mapping_accepted':False,'diagnostic_limit_bypass_only':bool(sv or tv)})
 wrong={'source_forearm_as_target_world':object_forearm,'source_palm_as_target_world':palm,'target_forearm_as_target_world':target['forearm']}
 checks.append({'case_id':cases[-1]['case_id'],'provider_status':pstatus,'max_source_frame_component_error':se,'max_target_frame_component_error':pe,'wrong_root_negative_controls_palm_displacement_m':{n:float(np.linalg.norm((M@tf['palm'])[:3,3]-palm[:3,3])) for n,M in wrong.items()},'native_target_common_link_origin_max_error_m':max(float(np.linalg.norm(native[n][:3,3]-target[n][:3,3])) for n in set(native)&set(target))})
for r in fit['records']:
 i=r['global_sample_index'];state=r['selected_state'];q=r['target_joint_targets_rad'];assert q==state['q'];tf=T.fk(q);root=np.asarray(state['root']);target={n:root@M for n,M in tf.items()};native=source_frames[i]
 errors=[assertclose(target['palm'],state['palm']),assertclose(target['forearm'],state['forearm'])];assert not T.violations(q)
 for n in ['WRJ2','WRJ1']:assert q[n]==r['original_joint_targets_rad'][n]
 cases.append({'case_id':f'dga_component_corrected_multidex_{i:02d}','generator':'dexgrasp_anything','variant':'component_corrected_saved_unaccepted','global_sample_index':i,'original_status':'CORRECTED_SURFACE_PRESERVATION_FAILED_UNACCEPTED','target_joint_targets_rad':q,'object_from_target_palm':target['palm'].tolist(),'object_from_target_root':root.tolist(),'object_from_target_forearm':target['forearm'].tolist(),'source_fk_reference':{n:M.tolist() for n,M in native.items()},'target_fk_reference':{n:M.tolist() for n,M in target.items()},'source_limit_violations':{},'target_limit_violations':{},'input_file_hashes':[ref(fitp),ref(B/'native_mapping_results_v1/fit_poses.npz'),ref(sourcep),ref(targetp)],'execution_eligible':False,'full_mapping_accepted':False,'diagnostic_limit_bypass_only':False})
 checks.append({'case_id':cases[-1]['case_id'],'max_target_frame_component_error':max(errors),'both_original_wrists_preserved':True})
# Declared model differences: do not confuse them with implementation errors.
joint_differences=[]
for n in names:
 s,t=S.j[n],T.j[n];d={'joint':n,'source_limits':s['limits'],'target_limits':t['limits'],'origin_translation_difference_m':float(np.linalg.norm(s['origin'][:3,3]-t['origin'][:3,3])),'origin_rotation_difference_rad':float(Rotation.from_matrix(s['origin'][:3,:3].T@t['origin'][:3,:3]).magnitude()),'local_axis_max_component_difference':maxerr(s['axis'],t['axis'])}
 if s['limits']!=t['limits'] or d['origin_translation_difference_m'] or d['origin_rotation_difference_rad'] or d['local_axis_max_component_difference']:joint_differences.append(d)
geometry={}
for link in ['forearm','wrist','palm','thbase','thproximal','thmiddle','thdistal']:
 geometry[link]={}
 for name,tree in [('source',S),('target',T)]:
  e=tree.xml.find(f"link[@name='{link}']")
  geometry[link][name]={'visual':[ET.tostring(x,encoding='unicode') for x in e.findall('visual')],'collision':[ET.tostring(x,encoding='unicode') for x in e.findall('collision')]}
output={'schema_version':'compiled_target_static_case_export_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'source_hand':ref(sourcep),'target_hand':ref(targetp),'joint_names':names,'target_root_link':'world','cases':cases,'scope':'18 saved DGA poses, 16 raw including all domain rejects plus2 saved corrected; independent XML FK only; no geometry or execution acceptance','physics_steps':0,'inference_calls':0,'evaluation_episodes':0,'training_calls':0}
(OUT/'target_cases_v1.json').write_text(json.dumps(output,indent=2)+'\n')
report={'schema_version':'dga_saved_conversion_frame_audit_v1','status':'PASS_CONVERSION_FRAMES_ONLY_NO_NEW_DEFECT_FOUND','created_at_utc':datetime.now(timezone.utc).isoformat(),'inputs':list(inputs.values()),'case_export':ref(OUT/'target_cases_v1.json'),'source_root':S.root,'target_root':T.root,'native_row_count':16,'corrected_row_count':2,'provider_counts':provider_counts,'max_frame_component_error':max(max(c.get('max_source_frame_component_error',0),c.get('max_target_frame_component_error',0)) for c in checks),'checks':checks,'declared_joint_differences':joint_differences,'selected_declared_geometry':geometry,'compiled_mujoco_comparison':'PENDING_PARENT_SHARED_HARNESS','production_provider_state':'UNREGISTERED_STATIC_CANDIDATE; original DexGraspNet schema intentionally separate from MultiDex study','scientific_acceptance':False,'target_or_criteria_changed':False,'physics_steps':0,'inference_calls':0,'evaluation_episodes':0,'training_calls':0}
(OUT/'dga_conversion_audit_v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','native_row_count','corrected_row_count','provider_counts','max_frame_component_error']}))
