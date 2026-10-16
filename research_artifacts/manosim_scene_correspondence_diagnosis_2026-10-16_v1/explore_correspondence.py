"""EXPLORATORY fixed-source geometry/pixel diagnostics; no fitted transforms/gates."""
import importlib.util,hashlib,json,time,io
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
P=ROOT/'research_artifacts/manosim_native_object_pilot_2026-10-16_v2/protocol.json'
S=Path('/private/tmp/manosim_object_pilot_saved_reader_v2/verify_saved_pilot.py');sp=importlib.util.spec_from_file_location('saved',S);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mesh(path):
 v=[];f=[]
 for line in Path(path).read_text().splitlines():
  x=line.split()
  if x and x[0]=='v':v.append(list(map(float,x[1:4])))
  if x and x[0]=='f':f.append([int(i.split('/')[0])-1 for i in x[1:]])
 return np.asarray(v),np.asarray(f)
def rays_depth(uv,K,tri):
 rays=np.c_[(uv[:,0]-K[0,2])/K[0,0],(uv[:,1]-K[1,2])/K[1,1],np.ones(len(uv))]
 a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a;tv=-a;qv=np.cross(tv,e1);num=np.einsum('ij,ij->i',e2,qv,optimize=False);out=[]
 for r in np.array_split(rays,max(1,(len(rays)+15)//16)):
  pv=np.cross(r[:,None,:],e2[None]);det=np.einsum('tj,ptj->pt',e1,pv,optimize=False);ok=np.abs(det)>1e-15;inv=np.divide(1.,det,out=np.zeros_like(det),where=ok);u=np.einsum('tj,ptj->pt',tv,pv,optimize=False)*inv;v=np.einsum('pj,tj->pt',r,qv,optimize=False)*inv;t=num[None]*inv
  ok&=(u>=0)&(v>=0)&(u+v<=1)&(t>0);out.append(np.where(ok,t,np.inf).min(1))
 return np.concatenate(out)
def quant(a):return dict(n=int(len(a)),median=float(np.median(a)),q90=float(np.quantile(a,.9)),max=float(np.max(a))) if len(a) else dict(n=0)
p=m.read(P);start=time.monotonic();rows=[];bindings={str(P):digest(P),str(S):digest(S)}
for i,(oid,spec) in enumerate(sorted(p['observations'].items())):
 data=m.observation(spec['path']);depth=np.asarray(Image.open(io.BytesIO(data['depth']))).astype(float)/1000;mask=np.asarray(Image.open(io.BytesIO(data['object_mask'])).convert('L'))>0;K=np.asarray(data['camera']['K']);T=np.asarray(data['T_world_camera']);v,f=mesh(spec['visual_mesh']);world=v+np.asarray(spec['world_position']);cam=np.einsum('nj,ji->ni',world-T[:3,3],T[:3,:3],optimize=False);tri=cam[f]
 sil=np.zeros(mask.shape,np.uint8);projected=np.empty((len(v),2));projected[:,0]=K[0,0]*cam[:,0]/cam[:,2]+K[0,2];projected[:,1]=K[1,1]*cam[:,1]/cam[:,2]+K[1,2]
 for face in f:
  if np.all(cam[face,2]>0):cv2.fillConvexPoly(sil,np.rint(projected[face]*256).astype(np.int32),1,shift=8)
 valid=mask&(depth>0)&(depth<65.535);er=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&valid;ids=np.flatnonzero(er);chosen=ids[np.linspace(0,len(ids)-1,min(128,len(ids)),dtype=int)];yy,xx=np.unravel_index(chosen,mask.shape);uv=np.c_[xx,yy];pred=rays_depth(uv,K,tri);obs=depth[yy,xx];hit=np.isfinite(pred);delta=obs[hit]-pred[hit]
 inter=np.count_nonzero(mask&(sil>0));union=np.count_nonzero(mask|(sil>0));diagonal=float(np.linalg.norm(v.max(0)-v.min(0)));comcam=T[:3,:3].T@(np.asarray(spec['world_position'])-T[:3,3]);comuv=(K@comcam);comuv=comuv[:2]/comuv[2]
 row=dict(observation_id=oid,object_id=spec['object_id'],capture=spec['path'],mesh=spec['visual_mesh'],mask_pixels=int(mask.sum()),predicted_silhouette_pixels=int(sil.sum()),silhouette_mask_recall=inter/int(mask.sum()),silhouette_precision=inter/int(sil.sum()) if sil.sum() else 0,silhouette_IoU=inter/union,mesh_bbox_diagonal_m=diagonal,selected_rays=int(len(pred)),ray_hits=int(hit.sum()),signed_ray_depth_error_m=quant(delta),absolute_ray_depth_error_m=quant(np.abs(delta)),median_absolute_depth_error_relative_diagonal=float(np.median(np.abs(delta))/diagonal) if len(delta) else None,COM_projected_uv=comuv.tolist(),mask_centroid_uv=[float(np.flatnonzero(mask.any(0)).mean()),float(np.flatnonzero(mask.any(1)).mean())],observed_depth_m=obs.tolist(),ray_depth_m=[float(x) if np.isfinite(x) else None for x in pred],pixel_uv=uv.tolist())
 rows.append(row)
 for path in (spec['path'],spec['visual_mesh']):bindings[path]=digest(path)
 if i%10==0:print('progress',i+1,'elapsed',round(time.monotonic()-start,2),flush=True)
report=dict(schema_version='hug_scene_correspondence_EXPLORATORY_v1',exploratory=True,acceptance_or_protocol_changed=False,transform_fitting=False,models_or_physics_calls=0,source_sha256=digest(__file__),source=Path(__file__).read_text(),inputs=bindings,views=rows,elapsed_s=time.monotonic()-start,summary=dict(views=len(rows),median_silhouette_recall=float(np.median([r['silhouette_mask_recall'] for r in rows])),min_silhouette_recall=min(r['silhouette_mask_recall'] for r in rows),median_IoU=float(np.median([r['silhouette_IoU'] for r in rows])),median_perview_abs_ray_error_m=float(np.median([r['absolute_ray_depth_error_m']['median'] for r in rows if r['ray_hits']])),median_perview_relative_error=float(np.median([r['median_absolute_depth_error_relative_diagonal'] for r in rows if r['ray_hits']])),total_rays=sum(r['selected_rays'] for r in rows),hits=sum(r['ray_hits'] for r in rows)))
for path,h in bindings.items():assert digest(path)==h
out=Path('/private/tmp/hug_scene_correspondence_exploratory_2026-10-16_v1.json');assert not out.exists();out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'report':str(out),'sha256':digest(out),'summary':report['summary'],'elapsed_s':report['elapsed_s']}),flush=True)
