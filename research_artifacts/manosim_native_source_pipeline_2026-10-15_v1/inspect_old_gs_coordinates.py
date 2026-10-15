"""Saved parameter/quaternion coordinate algebra only; noFK/skin/model."""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
p=json.loads((ROOT/'research_artifacts/manosim_articulated_skin_2026-10-15_v1/protocol.json').read_text())
def mul(a,b):
 w,x,y,z=a;v,i,j,k=b
 return np.array([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v])
def matrixq(R):
 t=np.trace(R)
 if t>0:
  s=2*np.sqrt(1+t);q=np.array([s/4,(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,(R[1,0]-R[0,1])/s])
 else:
  k=np.argmax(np.diag(R))
  if k==0:
   s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]);q=np.array([(R[2,1]-R[1,2])/s,s/4,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s])
  elif k==1:
   s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]);q=np.array([(R[0,2]-R[2,0])/s,(R[0,1]+R[1,0])/s,s/4,(R[1,2]+R[2,1])/s])
  else:
   s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]);q=np.array([(R[1,0]-R[0,1])/s,(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,s/4])
 return q/np.linalg.norm(q)
def log_near(q,raw):
 n=np.linalg.norm(q[1:]);theta=2*np.arctan2(n,q[0]);axis=q[1:]/n
 candidates=np.array([axis*(theta+2*math.pi*k) for k in (-1,0,1)])
 i=np.argmin(np.sum((candidates-raw)**2,axis=1));return candidates[i]
for i in (20,28):
 sp=p['cases'][str(i)]
 with np.load(sp['decoded_reference']['path'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
 with np.load(sp['compiled_readback']['path'],allow_pickle=False) as z:q0=z['literal_qpos'].copy()
 camera=d['camera_world'];raw=d['actual_full_poses'].reshape(16,3)
 q=q0[3:].reshape(16,4).copy();cq=matrixq(camera[:3,:3]);cq[1:]*=-1;q[0]=mul(cq,q[0])
 result=np.array([log_near(v,r) for v,r in zip(q,raw)]).astype(np.float32)
 recovered_t=(camera[:3,:3].T@(q0[:3]-camera[:3,3])).astype(np.float32)
 mismatches=np.argwhere(result.view(np.uint32)!=raw.view(np.uint32))
 print(json.dumps(dict(index=i,exact_AA32_bytes=bool(np.array_equal(result.view(np.uint8),raw.view(np.uint8))),mismatch_coordinates=mismatches.tolist(),original_root=raw[0].tolist(),recovered_root=result[0].tolist(),root_float32_bitpatterns_original=raw[0].view(np.uint32).tolist(),root_float32_bitpatterns_recovered=result[0].view(np.uint32).tolist(),maximum_AA_difference=float(np.max(np.abs(result.astype(float)-raw))),translation_exact=bool(np.array_equal(recovered_t.view(np.uint8),d['params_translation_m'].view(np.uint8))),rotation_source_profile='savednativeGS64rootvsactualsourceAA32',new_model_or_skin_operations=0)))
