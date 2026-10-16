"""Pure saved-cache arithmetic; no producer or engine imports/calls.
Finite mjMAXVAL AABBs are compiler bookkeeping, not physical tolerances.
"""
from pathlib import Path
import math
import numpy as np
CACHE_FIELDS=('geom_pos','geom_quat','geom_sameframe','geom_aabb','body_bvhadr','body_bvhnum','bvh_aabb','bvh_child','bvh_nodeid')

def require(x,m):
 if not x:raise ValueError(m)
def rotation(q):
 q=np.asarray(q,dtype=float);require(q.shape==(4,) and np.isfinite(q).all() and abs(math.fsum(float(v)*float(v) for v in q)-1)<=1e-12,'Proper literal compiled quaternion');w,x,y,z=q;n=math.fsum(float(v)*float(v) for v in q);s=2/n
 return np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],[s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],[s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])
def dot(A,b):return np.array([math.fsum(float(x)*float(y) for x,y in zip(row,b)) for row in A])
def exact_unchanged(before,after,fields=CACHE_FIELDS):
 for field in fields:
  a=np.asarray(before[field]);b=np.asarray(after[field]);require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'Postcompile cache/geometry mutation '+field)
 return dict(fields=list(fields),all_bytes_unchanged=True)
def world_bvh_readback(a):
 """Replay every world-body node from stored local geom AABBs and frames.

 Bound256u/(1-256u) times the largest absolute transformed corner magnitude
 encloses small3D quaternion/matrix/bounds arithmetic. This is solely an
 arithmetic comparison of finite compiler caches; no contact gate changes.
 """
 pos=np.asarray(a['geom_pos'],float);quat=np.asarray(a['geom_quat'],float);boxes=np.asarray(a['geom_aabb'],float);body=np.asarray(a['geom_bodyid']);start=int(a['body_bvhadr'][0]);count=int(a['body_bvhnum'][0]);allboxes=np.asarray(a['bvh_aabb'],float);children=np.asarray(a['bvh_child']);ids=np.asarray(a['bvh_nodeid']);require(allboxes.ndim==2 and allboxes.shape[1]==6 and boxes.shape==(len(pos),6),'Saved BVH array shapes');require(np.isfinite(np.r_[pos.ravel(),quat.ravel(),boxes.ravel(),allboxes.ravel()]).all() and np.all(boxes[:,3:]>=0) and np.all(allboxes[:,3:]>=0),'Finite nonnegative cache boxes')
 require(start>=0 and count>0 and start+count<=len(allboxes),'Nonempty world BVH range');require(np.array_equal(np.asarray(a['body_ipos'])[0],[0.,0.,0.]) and np.array_equal(np.asarray(a['body_iquat'])[0],[1.,0.,0.,0.]),'Original world inertial frame must be literal identity')
 visited=set();leafs=[];rows=[];unit=np.finfo(float).eps/2;gamma=256*unit/(1-256*unit)
 def recurse(local):
  require(0<=local<count and local not in visited,'BVH cycle/shared/outside node');visited.add(local);k=start+local;gid=int(ids[k])
  if gid>=0:
   require(gid<len(body) and int(body[gid])==0,'World leaf must reference world geom');require(np.all(children[k]==-1),'Leaf cannot have children');R=rotation(quat[gid]);center=dot(R,boxes[gid,:3])+pos[gid];half=dot(np.abs(R),boxes[gid,3:]);leafs.append(gid)
  else:
   ch=children[k];require(ch.shape==(2,) and np.all(ch>=0) and ch[0]!=ch[1],'Exactly two distinct internal children');l=recurse(int(ch[0]));r=recurse(int(ch[1]));lo=np.minimum(l[:3]-l[3:],r[:3]-r[3:]);hi=np.maximum(l[:3]+l[3:],r[:3]+r[3:]);center=(lo+hi)/2;half=(hi-lo)/2
  expected=np.r_[center,half];actual=allboxes[k];scale=max(1.,float(np.max(np.abs(expected[:3])+expected[3:])),float(np.max(np.abs(pos)))+1);bound=gamma*scale;error=float(np.max(np.abs(actual-expected)));require(error<=bound,'Compiled world BVH does not reflect authored world geometry')
  rows.append(dict(local_node=local,global_node=k,geom_id=gid,maximum_component_error_m=error,arithmetic_bound_m=bound));return expected
 recurse(0);require(len(visited)==count,'Unreachable declared world BVH node');require(len(leafs)==len(set(leafs)) and set(leafs)==set(np.flatnonzero((body==0)&((np.asarray(a['geom_contype'])!=0)|(np.asarray(a['geom_conaffinity'])!=0)))),'World leaf inventory must cover all collision-enabled world geoms')
 return dict(world_body_bvhadr=start,world_body_bvhnum=count,leaf_geom_ids=sorted(leafs),nodes=rows,maximum_component_error_m=max(r['maximum_component_error_m'] for r in rows),arithmetic_rule='gamma256 times finite transformed box magnitude; never a contact/fidelity allowance',no_runtime_geometry_mutation_proven_by_this_function=False)
