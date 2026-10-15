"""Independent static native FK and saved-output conversion audit; never advances time."""
from pathlib import Path
from xml.etree import ElementTree as E
import json, hashlib, sys, copy
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
R=Path('/private/tmp/compiled_hand_audit_20261015_v1/shadow')
B=W/'research_artifacts/unidexgrasp_mapping_validation_2026-10-15_native_precision_audit_v1'
U=B/'cloud_results/stages/03_validation_seed1000_count32_once/worker_output/_unidexgrasp_export'
DG=W/'research_artifacts/dexgrasp_anything_mapping_validation_2026-10-14_metadata_audit_v1/preprocessing/dexgraspnet_shadow_hand_wrist_free.xml'
DS=R/'dexgraspnet_official_dataset.npy'
TU=W/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf'
sys.path.insert(0,str(W/'src'))
def forbidden(*a,**k): raise AssertionError('Physics stepping forbidden in static conversion audit')
for name in ('mj_step','mj_step1','mj_step2'): setattr(mujoco,name,forbidden)
from grasp_failure_prediction.proposals.dexgraspnet import DexGraspNetRecordDecoder
from grasp_failure_prediction.proposals.unidexgrasp import decode_unidexgrasp_hand_pose
from grasp_failure_prediction.proposals.translation import translate_native_pose
from grasp_failure_prediction.proposals.schema import RigidTransform
from grasp_failure_prediction.evaluation.pose_validation import load_shadow_model
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vec(s,default): return np.asarray(default if s is None else [float(x) for x in s.split()])
def tr(p=(0,0,0),rot=None):
 t=np.eye(4);t[:3,3]=p;t[:3,:3]=np.eye(3) if rot is None else rot;return t
def quat(s):
 q=vec(s,[1,0,0,0]);return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
def screw(axis,q,p=(0,0,0)):
 a=np.asarray(axis); a=a/np.linalg.norm(a);return tr(p)@tr(rot=Rotation.from_rotvec(a*q).as_matrix())@tr(-np.asarray(p))
def mjcf_fk(root,q):
 frames={};sites={};axes={};anchors={}
 def rec(b,parent):
  t=parent@tr(vec(b.get('pos'),[0,0,0]),quat(b.get('quat')))
  for j in b.findall('joint'):
   n=j.get('name');ax=vec(j.get('axis'),[0,0,1]);p=vec(j.get('pos'),[0,0,0]);assert j.get('type','hinge')=='hinge'
   axes[n]=t[:3,:3]@ax; anchors[n]=(t@np.r_[p,1])[:3]
   t=t@screw(ax,q[n]-float(j.get('ref','0')),p)
  frames[b.get('name')]=t
  for s in b.findall('site'):sites[s.get('name')]=(t@np.r_[vec(s.get('pos'),[0,0,0]),1])[:3]
  for c in b.findall('body'):rec(c,t)
 for b in root.find('worldbody').findall('body'): rec(b,np.eye(4))
 return frames,sites,axes,anchors
ROOT=E.parse(TU).getroot();TJ={j.get('name'):j for j in ROOT.findall('joint')}
def urdf_fk(q):
 children={}
 for j in ROOT.findall('joint'): children.setdefault(j.find('parent').get('link'),[]).append(j)
 allchild={j.find('child').get('link') for j in ROOT.findall('joint')};roots={l.get('name') for l in ROOT.findall('link')}-allchild;assert len(roots)==1
 f={next(iter(roots)):np.eye(4)};axes={};anchors={}
 def rec(n):
  for j in children.get(n,[]):
   o=j.find('origin');r=vec(o.get('rpy') if o is not None else None,[0,0,0]);p=vec(o.get('xyz') if o is not None else None,[0,0,0]);t=f[n]@tr(p,Rotation.from_euler('xyz',r).as_matrix());name=j.get('name')
   if j.get('type')!='fixed':
    assert j.get('type')=='revolute';ax=vec(j.find('axis').get('xyz'),[0,0,1]);axes[name]=t[:3,:3]@ax;anchors[name]=t[:3,3].copy();t=t@screw(ax,q[name])
   child=j.find('child').get('link');f[child]=t;rec(child)
 rec(next(iter(roots)));return f,axes,anchors

def source_model(p,label):
 root=E.parse(p).getroot();out=copy.deepcopy(root)
 if out.find('option') is not None:out.find('option').attrib.pop('apirate',None)
 if out.find('size') is not None:out.find('size').attrib.pop('nstack',None)
 # Geometry is retained. Only unused rendering assets and execution attachments are removed.
 for tag in ('asset','contact','equality','tendon','actuator','sensor'):
  for el in out.findall(tag):out.remove(el)
 for el in out.iter():
  if el.tag in ('geom','site'):el.attrib.pop('material',None)
 path=R/(label+'_static_primitives.xml');E.ElementTree(out).write(path,encoding='utf-8')
 model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
 return root,model,data,path

def pose(T):return {'translation_m':T[:3,3].tolist(),'quaternion_xyzw':Rotation.from_matrix(T[:3,:3]).as_quat().tolist()}
def frompose(p):return tr(p['translation_m'],Rotation.from_quat(p['quaternion_xyzw']).as_matrix())
# Independent semantic mapping derived from named joint correspondence, not loaded YAML.
source_names=[f'robot0:{f}J{j}' for f,js in [('FF',range(3,-1,-1)),('MF',range(3,-1,-1)),('RF',range(3,-1,-1)),('LF',range(4,-1,-1)),('TH',range(4,-1,-1))] for j in js]
sign={'robot0:FFJ3':-1,'robot0:MFJ3':-1,'robot0:THJ1':-1,'robot0:THJ0':-1}
def mapping(q):return dict(WRJ2=0.,WRJ1=0.,**{n.split(':')[1][:-1]+str(int(n[-1])+1):v*sign.get(n,1) for n,v in q.items()})
def violations(q,limits):return {n:{'value':q[n],'lower':v[0],'upper':v[1]} for n,v in limits.items() if q[n]<v[0] or q[n]>v[1]}
TModel=load_shadow_model(TU.parent.parent);TData=mujoco.MjData(TModel)
target_limits={n:tuple(float(TJ[n].find('limit').attrib[k]) for k in ('lower','upper')) for n in mapping(dict.fromkeys(source_names,0.))}
tips=('thtip','fftip','mftip','rftip','lftip')
report={'schema_version':'native_shadow_compiled_static_audit_v1','mujoco_version':mujoco.__version__,'physics_steps':0,'model_calls':0,'retarget_optimization_calls':0,'new_evaluation_data':False,'inputs':{},'source_models':{},'rows':[],'maxima':{}}
record_rows=[]
def maxval(k,x):report['maxima'][k]=max(report['maxima'].get(k,0),float(x))
def audit_row(label,variant,index,source_path,root,SM,SD,sq,object_palm,producer_palm,producer_q,raw_path,scale=None,seed=None):
 tq=mapping(sq);assert tq==producer_q
 source_limits={j.get('name'):tuple(map(float,j.get('range').split())) for b in root.iter('body') for j in b.findall('joint')}
 sf,ss,sa,sp=mjcf_fk(root,sq);tf,ta,tp=urdf_fk(tq)
 for j in range(SM.njnt):SD.qpos[SM.jnt_qposadr[j]]=sq[SM.joint(j).name]
 mujoco.mj_forward(SM,SD);assert SD.time==0
 for j in range(TModel.njnt):TData.qpos[TModel.jnt_qposadr[j]]=tq[TModel.joint(j).name]
 mujoco.mj_forward(TModel,TData);assert TData.time==0
 se=0.;te=0.
 for name,F in sf.items():
  cb=SD.body(name);se=max(se,float(np.max(np.abs(F[:3,3]-cb.xpos))),float(np.max(np.abs(F[:3,:3]-cb.xmat.reshape(3,3)))))
 for b in range(1,TModel.nbody):
  name=TModel.body(b).name;F=tf[name];cb=TData.body(b);te=max(te,float(np.max(np.abs(F[:3,3]-cb.xpos))),float(np.max(np.abs(F[:3,:3]-cb.xmat.reshape(3,3)))))
 srcp=sf['robot0:palm']; targetp=tf['palm'];sv=np.asarray([(np.linalg.inv(srcp)@np.r_[ss['robot0:S_'+tip],1])[:3] for tip in tips]);tv=np.asarray([(np.linalg.inv(targetp)@tf[tip])[:3,3] for tip in tips]);err=np.linalg.norm(sv-tv,axis=1)
 pe=float(np.max(np.abs(object_palm-producer_palm)))
 assert max(se,te,pe)<1e-10
 maxval('source_xml_vs_compiled_max_absolute',se);maxval('target_xml_vs_compiled_max_absolute',te);maxval('independent_vs_adapter_root_max_absolute',pe);maxval(label+'_maximum_tip_error_m',max(err))
 rid=f'{label}_{variant}_{index:04d}'
 row={'id':rid,'generator':label,'variant':variant,'native_row':index,'seed':seed,'source_legal':not violations(sq,source_limits),'target_legal':not violations(tq,target_limits),'source_violations':violations(sq,source_limits),'target_violations':violations(tq,target_limits),'tip_error_m':dict(zip(tips,err.tolist())),'tip_gate_2mm_pass':bool(max(err)<=.002),'source_xml_vs_compiled_max_absolute':se,'target_xml_vs_compiled_max_absolute':te,'independent_vs_adapter_root_max_absolute':pe}
 report['rows'].append(row)
 record_rows.append(dict(row,source_input=str(raw_path),source_hand_path=str(source_path),source_joint_targets_rad=sq,joint_targets_rad=tq,object_to_palm=pose(object_palm),object_scale=scale,source_tip_vectors_palm_m=sv.tolist(),target_tip_vectors_palm_m=tv.tolist(),source_tips_object_m=(sv@object_palm[:3,:3].T+object_palm[:3,3]).tolist()))

for label,source_path in [('unidexgrasp',U/'source_hand.xml'),('dexgraspnet',DG)]:
 root,SM,SD,compiled=source_model(source_path,label)
 report['inputs'][str(source_path)]=sha(source_path)
 report['source_models'][label]={'original_sha256':sha(source_path),'static_primitive_model':str(compiled),'static_model_sha256':sha(compiled),'joint_names':[SM.joint(i).name for i in range(SM.njnt)],'ngeom':SM.ngeom,'geoms':[{'name':SM.geom(i).name,'body':SM.body(SM.geom_bodyid[i]).name,'type':int(SM.geom_type[i]),'size':SM.geom_size[i].tolist(),'pos':SM.geom_pos[i].tolist(),'quat':SM.geom_quat[i].tolist(),'contype':int(SM.geom_contype[i]),'conaffinity':int(SM.geom_conaffinity[i])} for i in range(SM.ngeom)]}
 if label=='unidexgrasp':
  mf=json.loads((U/'native_manifest.json').read_text());shift=frompose(mf['object_to_input'])
  for variant,raw in [('original_native',U/'model_output.npz'),('existing_cpu64_derivative',B/'derived_validation_cpu64_01/derived_output.npz')]:
   report['inputs'][str(raw)]=sha(raw)
   with np.load(raw,allow_pickle=False) as ar:rows=ar['hand_pose'].copy()
   for i,v in enumerate(rows):
    sq=dict(zip(source_names,map(float,v[6:])));ind=shift@tr(v[:3],Rotation.from_rotvec(v[3:6]).as_matrix())@mjcf_fk(root,sq)[0]['robot0:palm']
    native=decode_unidexgrasp_hand_pose(v,object_to_input=RigidTransform.model_validate(mf['object_to_input']),source_hand_mjcf=source_path,source_hand_sha256='sha256:'+sha(source_path),kinematics_output=R/'unidex_production_kinematic.xml');translated=translate_native_pose(native,'dexgraspnet_shadow_to_shadow_hand_right_v1')
    audit_row(label,variant,i,source_path,root,SM,SD,sq,ind,frompose(translated.object_to_palm.model_dump()),translated.joint_targets_rad,raw,scale=mf['object_scale'],seed=1000+i)
 else:
  assert sha(DS)=='e31d3247f2e2be416160d22f1a963144bb7c496c19453ec722258a1685bee646';report['inputs'][str(DS)]=sha(DS);rows=np.load(DS,allow_pickle=True)
  for i,rec in enumerate(rows):
   q=rec['qpos'];sq={n:float(q[n]) for n in source_names};ind=tr([q['WRJT'+a] for a in 'xyz'],Rotation.from_euler('xyz',[q['WRJR'+a] for a in 'xyz']).as_matrix())
   bundle={'schema_version':'dexgraspnet_replay_bundle_v1','catalog_object_key':'core_mug_8570d9a8d24cb0acbebd3c0c0c70fb03','source_commit':'bd1b13d7248729af117e1d46aaa6266b147a3c7b','record_path':DS.name,'record_sha256':'sha256:'+sha(DS),'record_index':i,'object_scale':float(rec['scale']),'source_kinematics_path':compiled.name,'source_kinematics_sha256':'sha256:'+sha(compiled)}
   bp=R/'dexgraspnet_diagnostic_decode_bundle.json';bp.write_text(json.dumps(bundle));native=DexGraspNetRecordDecoder().decode(bp);translated=translate_native_pose(native,'dexgraspnet_shadow_to_shadow_hand_right_v1')
   audit_row(label,'official_saved_record',i,source_path,root,SM,SD,sq,ind,frompose(translated.object_to_palm.model_dump()),translated.joint_targets_rad,DS,scale=float(rec['scale']))
report['inputs'][str(TU)]=sha(TU)
report['summary']={}
for label in ('unidexgrasp','dexgraspnet'):
 for variant in sorted({r['variant'] for r in report['rows'] if r['generator']==label}):
  rows=[r for r in report['rows'] if r['generator']==label and r['variant']==variant];report['summary'][label+':'+variant]={'rows':len(rows),'source_legal':sum(r['source_legal'] for r in rows),'target_legal':sum(r['target_legal'] for r in rows),'jointly_legal':sum(r['source_legal'] and r['target_legal'] for r in rows),'pass_2mm_tip_gate':sum(r['tip_gate_2mm_pass'] for r in rows)}
report['status']='PASS_CONVERSION_RECONSTRUCTION_CHECKS_NOT_CONTACT_EQUIVALENCE'
(R/'target_pose_records.json').write_text(json.dumps({'schema_version':'native_shadow_target_pose_records_v1','tip_order':tips,'records':record_rows},indent=2)+'\n')
(R/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'summary':report['summary'],'maxima':report['maxima'],'record_count':len(record_rows),'report_sha256':sha(R/'report.json')},indent=2))
