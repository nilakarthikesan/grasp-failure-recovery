"""Independent XML/SciPy FK validation. No mapper, learned model or physics imports.

The functions accept saved pose values; they do not optimize, clip or repair them.
Compiled checks use mj_kinematics only and can be enabled by a separate driver.
"""
from pathlib import Path
import hashlib, math, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

NAMES=('WRJ2','WRJ1','FFJ4','FFJ3','FFJ2','FFJ1','MFJ4','MFJ3','MFJ2','MFJ1','RFJ4','RFJ3','RFJ2','RFJ1','LFJ5','LFJ4','LFJ3','LFJ2','LFJ1','THJ5','THJ4','THJ3','THJ2','THJ1')
PARTS={'index':'ff','middle':'mf','ring':'rf','little':'lf','thumb':'th'}
SOURCE_SHA='932fca7c411ef9ae4e6a8f17a892b3629840b5b5e1254610e0616e184d1f95c7'
TARGET_SHA='29a50063cced2fba81dfe3d208ba54499fc12635c9189d0822f7caa2771c36de'
NUMERICAL_ATOL=1e-9

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mm(a,b):return np.einsum('ij,jk->ik',a,b)
def norm(x):return math.sqrt(sum(float(v)*float(v) for v in x))
def inv(t):
 o=np.eye(4);o[:3,:3]=t[:3,:3].T;o[:3,3]=-np.einsum('ij,j->i',t[:3,:3].T,t[:3,3]);return o

def checked(values,shape,label):
 a=np.asarray(values)
 if a.shape!=shape or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():raise ValueError('invalid '+label)
 return a.astype(float)

def rigid(t):
 t=checked(t,(4,4),'rigid transform')
 if not np.array_equal(t[3],[0,0,0,1]) or np.max(abs(mm(t[:3,:3].T,t[:3,:3])-np.eye(3)))>NUMERICAL_ATOL or abs(np.linalg.det(t[:3,:3])-1)>NUMERICAL_ATOL:raise ValueError('not proper rigid transform')
 return t

def source_rotation(values):
 """Scalar-normalized symmetric two-column 6D construction, independently coded."""
 v=checked(values,(6,),'native rotation6d')
 def unit(x):
  length=norm(x)
  if length<1e-8:raise ValueError('degenerate native rotation')
  return np.array([float(i)/length for i in x])
 a=unit(v[:3]);b=unit(v[3:]);m=unit(a+b);d=unit(a-b);x=unit(m+d);y=unit(m-d)
 r=np.column_stack([x,y,unit(np.cross(x,y))]);t=np.eye(4);t[:3,:3]=r;rigid(t);return r

class Tree:
 def __init__(self,path,expected_sha):
  self.path=Path(path);self.sha256=sha(path)
  if self.sha256!=expected_sha:raise ValueError('wrong URDF bytes')
  self.xml=ET.parse(path).getroot();self.links={e.get('name') for e in self.xml.findall('link')};self.joints={};self.child_joint={}
  def vector(e,k,default):return np.array([float(x) for x in (e.get(k,default) if e is not None else default).split()])
  for j in self.xml.findall('joint'):
   typ=j.get('type')
   if typ not in ['fixed','revolute'] or j.find('mimic') is not None:raise ValueError('unsupported joint')
   o=j.find('origin');t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',vector(o,'rpy','0 0 0')).as_matrix();t[:3,3]=vector(o,'xyz','0 0 0')
   lim=j.find('limit');entry={'parent':j.find('parent').get('link'),'child':j.find('child').get('link'),'origin':t,'axis':vector(j.find('axis'),'xyz','1 0 0'),'type':typ,'limits':None if typ=='fixed' else (float(lim.get('lower')),float(lim.get('upper')))}
   self.joints[j.get('name')]=entry;self.child_joint[entry['child']]=entry
  roots=self.links-set(self.child_joint)
  if len(roots)!=1:raise ValueError('invalid root')
  self.root=next(iter(roots));self.names={n for n,j in self.joints.items() if j['type']=='revolute'}
  if self.names!=set(NAMES):raise ValueError('joint names mismatch')
 def violations(self,q):
  if set(q)!=self.names:raise ValueError('incomplete joints')
  return [n for n in NAMES if not math.isfinite(q[n]) or not self.joints[n]['limits'][0]<=q[n]<=self.joints[n]['limits'][1]]
 def fk(self,q):
  if set(q)!=self.names or not all(math.isfinite(v) for v in q.values()):raise ValueError('invalid joints')
  result={self.root:np.eye(4)};pending=dict(self.joints)
  while pending:
   old=len(pending)
   for n,j in list(pending.items()):
    if j['parent'] not in result:continue
    t=j['origin'].copy()
    if j['type']=='revolute':t[:3,:3]=mm(t[:3,:3],Rotation.from_rotvec(j['axis']/norm(j['axis'])*q[n]).as_matrix())
    result[j['child']]=mm(result[j['parent']],t);del pending[n]
   if len(pending)==old:raise ValueError('invalid tree')
  return result

def decode_source(raw,tree,object_from_input=None):
 raw=checked(raw,(33,),'native33');q=dict(zip(NAMES,raw[9:],strict=True))
 violations=tree.violations(q)
 if violations:raise ValueError('source domain '+','.join(violations))
 pose=np.eye(4);pose[:3,:3]=source_rotation(raw[3:9]);pose[:3,3]=raw[:3]
 root=mm(rigid(np.eye(4) if object_from_input is None else object_from_input),pose)
 return q,{n:mm(root,t) for n,t in tree.fk(q).items()}

def metrics(errors):
 lengths=np.sqrt(np.einsum('ij,ij->i',errors,errors))
 return {'max_m':float(lengths.max()),'rms_m':float(np.sqrt(np.mean(lengths*lengths))),'mean_m':float(lengths.mean()),'sse_m2':float(np.sum(errors*errors))}

def validate_saved(raw,target_joints,object_from_target_root_world,source_tree,target_tree,*,baseline_joints=None,object_from_input=None,saved_raw=None):
 raw=checked(raw,(33,),'native33');qs,sf=decode_source(raw,source_tree,object_from_input)
 qt=dict(zip(NAMES,checked(target_joints,(24,),'target24'),strict=True)) if not isinstance(target_joints,dict) else target_joints
 violations=target_tree.violations(qt)
 if violations:raise ValueError('target domain '+','.join(violations))
 if any(qt[n]!=qs[n] for n in ['WRJ2','WRJ1']):raise ValueError('wrists changed')
 if saved_raw is not None:
  a=np.asarray(saved_raw)
  if a.dtype!=raw.dtype or a.shape!=(33,) or a.tobytes()!=raw.tobytes():raise ValueError('native bytes changed')
 root=rigid(object_from_target_root_world);tf={n:mm(root,t) for n,t in target_tree.fk(qt).items()}
 palm_error=float(np.max(abs(sf['palm']-tf['palm'])))
 if palm_error>NUMERICAL_ATOL:raise ValueError('palm not preserved')
 sites=['palm']+sorted(n for n in sf if n in tf and n.startswith(tuple(PARTS.values())))
 if len(sites)!=28:raise ValueError('landmark set mismatch')
 deltas=np.array([tf[n][:3,3]-sf[n][:3,3] for n in sites]);errors=np.sqrt(np.einsum('ij,ij->i',deltas,deltas))
 parts={part:metrics(np.array([tf[n][:3,3]-sf[n][:3,3] for n in sites if n.startswith(prefix)])) for part,prefix in PARTS.items()}
 result={'source_raw33_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'source_and_target_domains_valid':True,'wrists_retained':True,'palm_max_component_error':palm_error,'landmark_sites':sites,'landmark_errors_m':dict(zip(sites,errors.tolist(),strict=True)),'landmark_aggregate':metrics(deltas),'landmark_per_part':parts,'source_urdf_sha256':source_tree.sha256,'target_urdf_sha256':target_tree.sha256,'target_object_from_links':{n:t.tolist() for n,t in tf.items()},'baseline':None}
 if baseline_joints is not None:
  qb=dict(zip(NAMES,checked(baseline_joints,(24,),'baseline24'),strict=True)) if not isinstance(baseline_joints,dict) else baseline_joints
  if target_tree.violations(qb):raise ValueError('baseline target domain')
  expected={n:max(target_tree.joints[n]['limits'][0],min(target_tree.joints[n]['limits'][1],qs[n])) for n in NAMES}
  if any(qb[n]!=expected[n] for n in NAMES):raise ValueError('baseline not exact declared interval projection')
  bf={n:mm(root,t) for n,t in target_tree.fk(qb).items()};bed=np.array([bf[n][:3,3]-sf[n][:3,3] for n in sites]);bem=metrics(bed)
  bp={part:metrics(np.array([bf[n][:3,3]-sf[n][:3,3] for n in sites if n.startswith(prefix)])) for part,prefix in PARTS.items()}
  changed=[n for n in NAMES if qt[n]!=qs[n]];legal_branches=[part for part,prefix in PARTS.items() if all(target_tree.joints[n]['limits'][0]<=qs[n]<=target_tree.joints[n]['limits'][1] for n in NAMES if n.startswith(prefix.upper()))]
  for part in legal_branches:
   prefix=PARTS[part].upper()
   if any(qt[n]!=qs[n] for n in NAMES if n.startswith(prefix)):raise ValueError('already legal branch modified')
  # Numerical comparison is independent; report the signed difference without hiding near-boundary cases.
  result['baseline']={'landmark_aggregate':bem,'landmark_per_part':bp,'max_landmark_difference_m':result['landmark_aggregate']['max_m']-bem['max_m'],'max_landmark_nonregression_with_1nm_arithmetic_allowance':result['landmark_aggregate']['max_m']<=bem['max_m']+1e-9,'per_part_max_difference_m':{p:parts[p]['max_m']-bp[p]['max_m'] for p in PARTS},'per_part_sse_difference_m2':{p:parts[p]['sse_m2']-bp[p]['sse_m2'] for p in PARTS},'per_part_sse_nonregression_exact_computed':{p:parts[p]['sse_m2']<=bp[p]['sse_m2'] for p in PARTS},'unchanged_legal_branches':legal_branches,'changed_joint_names':changed}
 return result

class CompiledKinematics:
 """Current live MjSpec importer checked against independent XML FK, no solver."""
 def __init__(self,target_tree,output_mjb=None):
  import mujoco
  from grasp_failure_prediction.evaluation.pose_validation import load_shadow_spec
  self.mj=mujoco;self.tree=target_tree;self.block_attempts=[]
  def blocked(name):
   def run(*args,**kwargs):self.block_attempts.append(name);raise RuntimeError('prohibited dynamics '+name)
   return run
  self.blocked_names=['mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_inverse','mj_inverseSkip','mj_integratePos','mj_Euler','mj_RungeKutta']
  for n in self.blocked_names:
   if hasattr(mujoco,n):setattr(mujoco,n,blocked(n))
  self.spec=load_shadow_spec(target_tree.path.parent.parent,floating_root=True);self.model=self.spec.compile();self.data=mujoco.MjData(self.model)
  self.ids={n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in NAMES}
  if any(i<0 for i in self.ids.values()):raise ValueError('compiled joint missing')
  for n,i in self.ids.items():
   if not np.array_equal(self.model.jnt_range[i],target_tree.joints[n]['limits']):raise ValueError('compiled limit drift '+n)
  free=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')
  if self.model.jnt_type[free]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('floating root absent')
  self.freeadr=int(self.model.jnt_qposadr[free]);self.mjb_sha256=None
  if output_mjb is not None:
   mujoco.mj_saveModel(self.model,str(output_mjb),None);self.mjb_sha256=sha(output_mjb)
 def frame(self,name):
  mj=self.mj;i=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_BODY,name)
  if i>=0:
   t=np.eye(4);t[:3,:3]=self.data.xmat[i].reshape(3,3);t[:3,3]=self.data.xpos[i];return t
  joint=self.tree.child_joint[name]
  if joint['type']!='fixed':raise ValueError('compiled movable body missing')
  return mm(self.frame(joint['parent']),joint['origin'])
 def validate(self,qt,root):
  if not isinstance(qt,dict):qt=dict(zip(NAMES,qt,strict=True))
  if self.tree.violations(qt):raise ValueError('compiled pose domain')
  tf={n:mm(root,t) for n,t in self.tree.fk(qt).items()}
  # Free q stores the ACTUAL forearm frame; fixed URDF world_joint has a
  # nonidentity yaw that must already be incorporated exactly once.
  forearm=tf['forearm'];m,d,mj=self.model,self.data,self.mj;d.qpos[:]=m.qpos0;d.qvel[:]=0
  for n,i in self.ids.items():d.qpos[int(m.jnt_qposadr[i])]=qt[n]
  d.qpos[self.freeadr:self.freeadr+3]=forearm[:3,3];q=Rotation.from_matrix(forearm[:3,:3]).as_quat();d.qpos[self.freeadr+3:self.freeadr+7]=q[[3,0,1,2]]
  mj.mj_kinematics(m,d)
  if d.time!=0 or np.count_nonzero(d.qvel) or self.block_attempts:raise ValueError('dynamic state changed')
  errors={}
  for n,t in tf.items():
   if n=='world':continue
   actual=self.frame(n);relative=mm(actual[:3,:3].T,t[:3,:3]);skew=np.array([relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]])*.5
   angle=math.atan2(norm(skew),max(-1,min(1,(np.trace(relative)-1)/2)))
   errors[n]={'position_m':norm(actual[:3,3]-t[:3,3]),'rotation_rad':angle,'max_matrix_component':float(np.max(abs(actual-t)))}
  maximum={k:max(r[k] for r in errors.values()) for k in next(iter(errors.values()))}
  if max(maximum.values())>1e-9:raise ValueError('compiled FK discrepancy')
  return {'status':'PASS','links_checked':len(errors),'maximum':maximum,'per_link':errors,'source_target_limit_identity':True,'mujoco_version':mj.__version__,'compiled_mjb_sha256':self.mjb_sha256,'physics_steps':0,'data_time_s':float(d.time),'blocked_entrypoint_attempts':self.block_attempts.copy()}
