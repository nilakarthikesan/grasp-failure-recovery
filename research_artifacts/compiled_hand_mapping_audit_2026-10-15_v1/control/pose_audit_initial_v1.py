"""Independent static URDF-to-compiled-MuJoCo audit. No dynamics or solver calls."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import mujoco

TARGET=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf')
TARGET_SHA='sha256:29a50063cced2fba81dfe3d208ba54499fc12635c9189d0822f7caa2771c36de'
ROOT=Path('/private/tmp/compiled_hand_audit_20261015_v1')
BLOCKED=['mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_inverse','mj_inverseSkip','mj_integratePos','mj_Euler','mj_RungeKutta']
BLOCK_ATTEMPTS=[]
def refuse(name):
 def f(*a,**kw):
  BLOCK_ATTEMPTS.append(name)
  raise RuntimeError('Forbidden dynamics/solver entry point: '+name)
 return f
for n in BLOCKED:
 if hasattr(mujoco,n):setattr(mujoco,n,refuse(n))

def sha(p):return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def vec(n,k,default='0 0 0'):return np.fromstring(default if n is None else n.get(k,default),sep=' ')
def transform(R=None,t=None):
 a=np.eye(4)
 if R is not None:a[:3,:3]=R
 if t is not None:a[:3,3]=t
 return a
def origin(n):return transform(Rotation.from_euler('xyz',vec(n,'rpy')).as_matrix(),vec(n,'xyz'))
def from_quat(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
def to_quat(R):
 # Same representation API as runner; this does not call its stepping methods.
 q=np.empty(4);mujoco.mju_mat2Quat(q,np.asarray(R).reshape(9));return q

def rigid_inv(M):
 r=M[:3,:3];return transform(r.T,-r.T@M[:3,3])
def errors(A,B):
 R=A[:3,:3]@B[:3,:3].T
 skew=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])*.5
 return {'translation_m':float(np.linalg.norm(A[:3,3]-B[:3,3])),'rotation_rad':float(np.arctan2(np.linalg.norm(skew),max(-1.,min(1.,(np.trace(R)-1.)*.5)))),'max_matrix_component':float(np.max(np.abs(A-B)))}
def maximum(vals):
 return {k:max(float(v[k]) for v in vals) for k in vals[0]} if vals else {}
def obj_vertices(p):
 rows=[]
 for line in Path(p).read_text().splitlines():
  fields=line.split()
  if fields and fields[0]=='v':rows.append(list(map(float,fields[1:4])))
 return np.asarray(rows)
def points(M,v):return v@M[:3,:3].T+M[:3,3]
class Tree:
 def __init__(self,path):
  self.path=path;self.xml=ET.parse(path).getroot();self.links=[e.get('name') for e in self.xml.findall('link')];self.joints={};self.child_joint={};self.collisions=[];self.mesh_files={}
  for e in self.xml.findall('joint'):
   lim=e.find('limit');n=e.get('name');d={'parent':e.find('parent').get('link'),'child':e.find('child').get('link'),'origin':origin(e.find('origin')),'axis':vec(e.find('axis'),'xyz','1 0 0'),'type':e.get('type'),'limits':None if e.get('type')=='fixed' else [float(lim.get('lower')),float(lim.get('upper'))]};self.joints[n]=d;self.child_joint[d['child']]=d
  self.names=[n for n,j in self.joints.items() if j['type']=='revolute']
  for e in self.xml.findall('link'):
   for i,c in enumerate(e.findall('collision')):
    g=c.find('geometry');assert g is not None and len(g)==1
    primitive=list(g)[0];kind=primitive.tag;d={'link':e.get('name'),'ordinal':i,'origin':origin(c.find('origin')),'kind':kind,'key':f"{e.get('name')}/collision/{i}"}
    if kind=='mesh':
     p=(path.parent/primitive.get('filename')).resolve();scale=vec(primitive,'scale','1 1 1');d.update({'path':p,'scale':scale,'vertices':obj_vertices(p)*scale});self.mesh_files[str(p)]=ref(p)
    elif kind=='box':d['size']=vec(primitive,'size')/2
    elif kind=='cylinder':d['size']=np.array([float(primitive.get('radius')),float(primitive.get('length'))/2,0.])
    elif kind=='sphere':d['size']=np.array([float(primitive.get('radius')),0.,0.])
    else:raise ValueError(kind)
    self.collisions.append(d)
 def fk(self,q):
  f={'world':np.eye(4)};pending=dict(self.joints)
  while pending:
   n0=len(pending)
   for n,j in list(pending.items()):
    if j['parent'] not in f:continue
    local=j['origin'].copy()
    if j['type']!='fixed':local[:3,:3]=local[:3,:3]@Rotation.from_rotvec(j['axis']/np.linalg.norm(j['axis'])*q[n]).as_matrix()
    f[j['child']]=f[j['parent']]@local;del pending[n]
   assert len(pending)<n0
  return f
 def violations(self,q):return [n for n in self.names if not self.joints[n]['limits'][0]<=q[n]<=self.joints[n]['limits'][1]]

def body_frame(model,data,name,tree):
 b=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
 if b>=0:return transform(data.xmat[b].reshape(3,3),data.xpos[b])
 j=tree.child_joint[name];assert j['type']=='fixed',(name,j['type'])
 return body_frame(model,data,j['parent'],tree)@j['origin']

def model_setup(path,tree):
 m=mujoco.MjModel.from_binary_path(str(path));d=mujoco.MjData(m)
 assert m.nq>=31
 named={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in tree.names};assert all(i>=0 for i in named.values())
 addr={n:int(m.jnt_qposadr[i]) for n,i in named.items()};free=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint');assert m.jnt_type[free]==mujoco.mjtJoint.mjJNT_FREE;rootaddr=int(m.jnt_qposadr[free])
 mapping=[];decl=[]
 kinds={'box':mujoco.mjtGeom.mjGEOM_BOX,'sphere':mujoco.mjtGeom.mjGEOM_SPHERE,'cylinder':mujoco.mjtGeom.mjGEOM_CYLINDER,'mesh':mujoco.mjtGeom.mjGEOM_MESH}
 for col in tree.collisions:
  body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,col['link']);ids=[i for i in range(m.ngeom) if m.geom_bodyid[i]==body];assert len(ids)==len([v for v in tree.collisions if v['link']==col['link']]);gid=ids[col['ordinal']];assert m.geom_type[gid]==kinds[col['kind']]
  info={'gid':gid,'col':col};compiled_local=transform(from_quat(m.geom_quat[gid]),m.geom_pos[gid]);e={'key':col['key'],'kind':col['kind'],'geom_id':gid}
  if col['kind']=='mesh':
   mid=int(m.geom_dataid[gid]);a=int(m.mesh_vertadr[mid]);cnt=int(m.mesh_vertnum[mid]);vv=m.mesh_vert[a:a+cnt].astype(float).copy();Mmesh=transform(from_quat(m.mesh_quat[mid]),m.mesh_pos[mid]);info.update({'mesh_id':mid,'vertices':vv,'mesh_transform':Mmesh});decl_local=points(Mmesh,vv)
   vf=col['vertices'];forward=cKDTree(vf).query(decl_local);backward=cKDTree(decl_local).query(vf);info['vertex_forward']=forward[1];info['vertex_backward']=backward[1]
   e.update({'compiled_vertex_count':len(vv),'declared_vertex_count':len(vf),'bidirectional_vertex_cloud_difference_m':float(max(forward[0].max(),backward[0].max())),'normalization_frame_mesh_pos':m.mesh_pos[mid].tolist(),'normalization_frame_mesh_quat_wxyz':m.mesh_quat[mid].tolist(),'topology_or_collision_exterior_certified':False})
   original_compiled=compiled_local@rigid_inv(Mmesh)
  else:
   info['size']=m.geom_size[gid].copy();e['size_max_component_error_m']=float(np.max(np.abs(m.geom_size[gid]-col['size'])));original_compiled=compiled_local
  e['local_declared_frame_error']=errors(original_compiled,col['origin']);mapping.append(info);decl.append(e)
 assert len(mapping)==45
 ranges=[{'joint':n,'declared_rad':tree.joints[n]['limits'],'compiled_rad':m.jnt_range[i].tolist(),'max_component_error_rad':float(np.max(np.abs(m.jnt_range[i]-tree.joints[n]['limits'])))} for n,i in named.items()]
 return {'m':m,'d':d,'addr':addr,'named':named,'rootaddr':rootaddr,'mapping':mapping,'static':{'model':ref(path),'bodies':m.nbody,'joints':m.njnt,'collision_geoms':len(mapping),'joint_ranges':ranges,'declared_collider_import':decl,'max_joint_range_error_rad':max(x['max_component_error_rad'] for x in ranges),'max_analytic_size_error_m':max(x.get('size_max_component_error_m',0) for x in decl),'max_mesh_vertex_cloud_difference_m':max(x.get('bidirectional_vertex_cloud_difference_m',0) for x in decl)}}

def apply_pose(info,q,palm):
 m,d=info['m'],info['d'];d.qpos[:]=m.qpos0;d.qvel[:]=0
 for n,v in q.items():d.qpos[info['addr'][n]]=v
 a=info['rootaddr'];d.qpos[a:a+7]=[0,0,0,1,0,0,0]
 mujoco.mj_kinematics(m,d)
 pb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'palm');R0=d.xmat[pb].reshape(3,3).copy();t0=d.xpos[pb].copy()
 root_R=palm[:3,:3]@R0.T;root_t=palm[:3,3]-root_R@t0
 d.qpos[a:a+3]=root_t;d.qpos[a+3:a+7]=to_quat(root_R)
 mujoco.mj_kinematics(m,d)
 assert d.time==0. and np.array_equal(d.qvel,np.zeros(m.nv))
 assert all(d.qpos[info['addr'][n]]==v for n,v in q.items())
 return root_t,d.qpos[a+3:a+7].copy()

def case_compare(case,info,tree):
 m,d=info['m'],info['d'];q={n:float(case['target_joint_targets_rad'][n]) for n in tree.names};assert set(q)==set(case['target_joint_targets_rad'])
 palm=np.asarray(case['object_from_target_palm'],float);assert palm.shape==(4,4) and np.isfinite(palm).all()
 ortho=float(np.max(np.abs(palm[:3,:3].T@palm[:3,:3]-np.eye(3))));det=float(np.linalg.det(palm[:3,:3]));assert max(ortho,abs(det-1.))<1e-5,(case['case_id'],ortho,det)
 tf=tree.fk(q);expectedroot=palm@np.linalg.inv(tf['palm']);expected={n:expectedroot@M for n,M in tf.items()}
 rpos,rquat=apply_pose(info,q,palm)
 normalizedpalm=palm.copy();normalizedpalm[:3,:3]=from_quat(to_quat(palm[:3,:3]));normroot=normalizedpalm@np.linalg.inv(tf['palm']);normexpected={n:normroot@M for n,M in tf.items()}
 links={n:errors(body_frame(m,d,n,tree),expected[n]) for n in tree.links if n!='world'}
 norms={n:errors(body_frame(m,d,n,tree),normexpected[n]) for n in tree.links if n!='world'}
 geoms={}
 for item in info['mapping']:
  gid=item['gid'];col=item['col'];actual=transform(d.geom_xmat[gid].reshape(3,3),d.geom_xpos[gid]);wanted=expected[col['link']]@col['origin'];e={}
  if col['kind']=='mesh':
   original=actual@rigid_inv(item['mesh_transform']);av=points(actual,item['vertices']);ev=points(wanted,col['vertices']);e['vertex_cloud_max_difference_m']=float(max(np.linalg.norm(av-ev[item['vertex_forward']],axis=1).max(),np.linalg.norm(ev-av[item['vertex_backward']],axis=1).max()))
  else:original=actual;e['size_max_component_error_m']=float(np.max(np.abs(item['size']-col['size'])))
  e.update(errors(original,wanted));geoms[col['key']]=e
 raw_viol=tree.violations(q);compiled_viol=[n for n,i in info['named'].items() if not m.jnt_range[i,0]<=q[n]<=m.jnt_range[i,1]]
 src=case.get('source_fk_reference');source_target=None
 if src:
  common=set(src)&set(expected)-{'world'};source_target={n:float(np.linalg.norm(np.asarray(src[n])[:3,3]-body_frame(m,d,n,tree)[:3,3])) for n in sorted(common)}
 frozen_refs=case.get('target_fk_reference');saved_ref_error=None
 if frozen_refs:saved_ref_error=max(float(np.max(np.abs(expected[n]-np.asarray(v)))) for n,v in frozen_refs.items())
 return {'case_id':case['case_id'],'generator':case['generator'],'variant':case.get('variant'),'original_status':case['original_status'],'diagnostic_only':True,'raw_requested_palm_rotation_orthogonality_max_component':ortho,'raw_requested_palm_rotation_determinant':det,'palm_quaternion_roundtrip_max_component':float(np.max(np.abs(palm-normalizedpalm))),'maximum_link_error':maximum(list(links.values())),'maximum_link_error_normalized_reference':maximum(list(norms.values())),'link_errors':links,'maximum_collider_frame_error':maximum([{k:v[k] for k in ['translation_m','rotation_rad','max_matrix_component']} for v in geoms.values()]),'maximum_mesh_vertex_cloud_error_m':max(v.get('vertex_cloud_max_difference_m',0) for v in geoms.values()),'collider_errors':geoms,'target_limits_violations_urdf':raw_viol,'target_limits_violations_compiled':compiled_viol,'limit_classification_same':set(raw_viol)==set(compiled_viol),'source_to_compiled_link_origin_errors_m':source_target,'provided_target_fk_reference_max_component_error':saved_ref_error,'actual_free_forearm_position_m':rpos.tolist(),'actual_free_forearm_quaternion_wxyz':rquat.tolist(),'data_time_s':float(d.time),'execution_eligible':False}

def controls(tree):
 result=[]
 for i,fraction in enumerate([.17,.53,.83]):
  q={n:(1-fraction)*j['limits'][0]+fraction*j['limits'][1] for n,j in tree.joints.items() if j['type']=='revolute'}
  r=Rotation.from_euler('xyz',[[.37,-.28,1.2],[-.41,.32,-1.06],[.93,.47,2.18]][i]).as_matrix();p=transform(r,[[.12,-.1,.34],[-.2,.07,.15],[.05,.08,.22]][i]);assert q['WRJ1'] and q['WRJ2']
  result.append({'case_id':f'manufactured_nonzero_wrists_rotated_root_{i}','generator':'manufactured_static_control','variant':'nonzero_wrists_world_rotation','original_status':'MANUFACTURED_CONVERSION_CONTROL_ONLY','target_joint_targets_rad':q,'object_from_target_palm':p.tolist()})
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--case-file',action='append',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();start=time.time();assert sha(TARGET)==TARGET_SHA
 tree=Tree(TARGET);allcases=[];inputs=[ref(TARGET),ref(Path(__file__))]+list(tree.mesh_files.values())
 for name in args.case_file:
  file=Path(name);data=json.loads(file.read_text());rows=data if isinstance(data,list) else data['cases'];allcases+=rows;inputs.append(ref(file))
 assert len({c['case_id'] for c in allcases})==len(allcases)
 native_count=len(allcases);allcases+=controls(tree)
 models={k:model_setup(ROOT/'control'/f'{k}_{args.stage}.mjb',tree) for k in ['base','tabletop','actuated']}
 results={};summaries={}
 for k,info in models.items():
  rows=[case_compare(c,info,tree) for c in allcases];results[k]={'static':info['static'],'cases':rows};inputs.append(info['static']['model'])
  summaries[k]={'case_count':len(rows),'maximum_link_error':maximum([r['maximum_link_error'] for r in rows]),'maximum_link_error_normalized_reference':maximum([r['maximum_link_error_normalized_reference'] for r in rows]),'maximum_collider_frame_error':maximum([r['maximum_collider_frame_error'] for r in rows]),'maximum_mesh_vertex_cloud_error_m':max(r['maximum_mesh_vertex_cloud_error_m'] for r in rows),'max_joint_range_error_rad':info['static']['max_joint_range_error_rad'],'max_analytic_size_error_m':info['static']['max_analytic_size_error_m'],'limit_classification_disagreements':[r['case_id'] for r in rows if not r['limit_classification_same']],'raw_rotation_projection_max_component':max(r['palm_quaternion_roundtrip_max_component'] for r in rows)}
  print(json.dumps({'model':k,**summaries[k]}),flush=True)
 # Test the entry-point guard after all actual comparisons. This deliberate attempt is not a physics call.
 try:mujoco.mj_step(None,None)
 except RuntimeError:pass
 assert BLOCK_ATTEMPTS==['mj_step']
 report={'schema_version':'cross_generator_compiled_target_static_pose_audit_v1','stage':args.stage,'created_at_utc':datetime.now(timezone.utc).isoformat(),'mujoco_version':mujoco.__version__,'native_saved_case_count':native_count,'manufactured_control_count':3,'case_count_with_controls':len(allcases),'links_compared_per_case_excluding_world':len(tree.links)-1,'collision_geoms_compared_per_case':45,'inputs':inputs,'summary':summaries,'results':results,'scope':'Target XML FK versus compiled static kinematics, declared analytic collider parameters and OBJ vertex-cloud placement. Vertex agreement does not establish mesh topology, convex-hull collision exterior, surface/contact equivalence, dynamic success, generator acceptance or evaluation eligibility. Invalid-domain rows remain diagnostic only. Root compensation reproduces production algebra using compiled palm, with no clipping.','execution_eligible':False,'target_or_criteria_changed':False,'zero_execution':{'physics_steps':0,'solver_calls':0,'mujoco_kinematics_only':True,'inference_calls':0,'fit_calls':0,'evaluation_episodes':0,'training_calls':0,'data_time_unchanged':True,'blocked_api_names':BLOCKED,'deliberate_guard_probe_attempts':BLOCK_ATTEMPTS},'elapsed_seconds':time.time()-start}
 args.output.write_text(json.dumps(report,indent=2)+'\n');print('REPORT',str(args.output),sha(args.output))
if __name__=='__main__':main()
