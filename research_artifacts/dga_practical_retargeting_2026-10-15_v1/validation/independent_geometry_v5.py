"""Uncapped closed-triangle distance and complete cube-grid comparison.

No clipped optimizer field, simulator, inference, fit or success metric is used.
F64 query errors are conditional on a documented 1 nm per-distance allowance;
this is not a directed-rounding exact arithmetic certificate.
"""
from pathlib import Path
import json,math,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from independent_validator import PARTS,mm

FRACTIONS=(9/23,7/19)
GRIDS_M=(.001,.0005,.00025)
DISTANCE_ERROR_M=1e-9
BROADPHASE_MARGIN_M=1e-11

def closest_pairs(p,t):
 """Independent vectorized Ericson region tests with segment fallback."""
 a,b,c=t[:,0],t[:,1],t[:,2];ab=b-a;ac=c-a;ap=p-a
 dot=lambda x,y:np.einsum('ij,ij->i',x,y)
 d1=dot(ab,ap);d2=dot(ac,ap);bp=p-b;d3=dot(ab,bp);d4=dot(ac,bp);cp=p-c;d5=dot(ab,cp);d6=dot(ac,cp)
 vc=d1*d4-d3*d2;vb=d5*d2-d1*d6;va=d3*d6-d5*d4
 best=np.full(len(p),np.inf)
 for x,y in [(a,b),(b,c),(c,a)]:
  e=y-x;den=dot(e,e);u=np.divide(dot(p-x,e),den,out=np.zeros_like(den),where=den>0);q=x+np.clip(u,0,1)[:,None]*e;best=np.minimum(best,dot(p-q,p-q))
 # Any point projecting to the closed triangle interior has nonnegative
 # barycentric numerators; all other regions are covered by the segments.
 den=va+vb+vc;n=np.cross(ab,ac);normal2=dot(n,n)
 inside=(va>=0)&(vb>=0)&(vc>=0)&(den>0)&(normal2>0)
 if np.any(inside):
  q=a[inside]+ab[inside]*(vb[inside]/den[inside])[:,None]+ac[inside]*(vc[inside]/den[inside])[:,None]
  residual=p[inside]-q;best[inside]=np.minimum(best[inside],dot(residual,residual))
 if not np.isfinite(best).all():raise ValueError('nonfinite triangle distance')
 return np.sqrt(np.maximum(best,0))

class MeshUnion:
 def __init__(self,vertices,faces):
  self.vertices=np.asarray(vertices,float);self.faces=np.asarray(faces,np.int64)
  if self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or self.faces.ndim!=2 or self.faces.shape[1]!=3 or not len(self.faces) or not np.isfinite(self.vertices).all():raise ValueError('invalid mesh')
  self.triangles=self.vertices[self.faces];self.centers=self.triangles.mean(axis=1);self.radii=np.linalg.norm(self.triangles-self.centers[:,None],axis=2).max(axis=1)
  self.referenced_vertex_indices=np.unique(self.faces)
  self.triangle_lower=self.triangles.min(axis=1);self.triangle_upper=self.triangles.max(axis=1)
  self.vertex_tree=cKDTree(self.vertices[self.referenced_vertex_indices]);self.center_tree=cKDTree(self.centers);self.max_radius=float(self.radii.max())
  bins=np.floor(np.log2(np.maximum(self.radii,1e-4)/1e-4)).astype(np.int64)
  self.radius_groups=[]
  for code in np.unique(bins):
   ids=np.flatnonzero(bins==code);self.radius_groups.append((ids,cKDTree(self.centers[ids]),float(self.radii[ids].max())))
 def query(self,points,chunk=128):
  points=np.asarray(points,float)
  if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('invalid queries')
  result=[];pairs=0
  for start in range(0,len(points),chunk):
   p=points[start:start+chunk];upper=self.vertex_tree.query(p,workers=1)[0]
   allrows=[];allcols=[]
   for group_ids,group_tree,group_radius in self.radius_groups:
    hits=group_tree.query_ball_point(p,upper+group_radius+BROADPHASE_MARGIN_M,eps=0,workers=1,return_sorted=True)
    sizes=np.array([len(h) for h in hits]);allrows.append(np.repeat(np.arange(len(p)),sizes));allcols.append(group_ids[np.concatenate([np.array(h,np.int64) for h in hits])])
   rows=np.concatenate(allrows);cols=np.concatenate(allcols);delta=p[rows]-self.centers[cols]
   keep=np.einsum('ij,ij->i',delta,delta)<=(upper[rows]+self.radii[cols]+BROADPHASE_MARGIN_M)**2;rows=rows[keep];cols=cols[keep]
   # Every triangle is inside its AABB. This lower bound can only discard a
   # triangle farther than the existing surface-vertex upper bound.
   gap=np.maximum(np.maximum(self.triangle_lower[cols]-p[rows],p[rows]-self.triangle_upper[cols]),0.)
   keep=np.einsum('ij,ij->i',gap,gap)<=(upper[rows]+BROADPHASE_MARGIN_M)**2
   rows=rows[keep];cols=cols[keep];pairs+=len(cols)
   distance=closest_pairs(p[rows],self.triangles[cols]);best=np.full(len(p),np.inf);np.minimum.at(best,rows,distance)
   if not np.isfinite(best).all() or np.any(best>upper+BROADPHASE_MARGIN_M):raise ValueError('uncapped search completeness witness failure')
   result.append(best)
  return np.concatenate(result),pairs

def combine(meshes):
 vertices=[];faces=[];count=0
 for v,f in meshes:vertices.append(v);faces.append(f+count);count+=len(v)
 if not vertices:raise ValueError('empty anatomical mesh')
 return MeshUnion(np.vstack(vertices),np.vstack(faces))

def source_part(path,sample,part):
 prefix=PARTS[part];meshes=[]
 with np.load(path,allow_pickle=False) as z:
  for k in sorted(z.files):
   if k.startswith('vertices__'+prefix):
    link=k.removeprefix('vertices__');v=z[k][sample];f=z['faces__'+link];meshes.append((v,f))
 return combine(meshes)

def obj(path):
 vertices=[];faces=[]
 for line in Path(path).read_text().splitlines():
  w=line.split()
  if not w:continue
  if w[0]=='v':vertices.append([float(x) for x in w[1:4]])
  elif w[0]=='f':
   ids=[int(x.split('/')[0]) for x in w[1:]];ids=[x-1 if x>0 else len(vertices)+x for x in ids]
   for i in range(1,len(ids)-1):faces.append([ids[0],ids[i],ids[i+1]])
 return np.asarray(vertices,float),np.asarray(faces,np.int64)

def box_boundary(size):
 import itertools
 size=np.asarray(size,float)
 if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):raise ValueError('invalid declared visual box')
 signs=list(itertools.product([-1,1],repeat=3));v=np.asarray(signs)*size/2;f=[]
 for axis in range(3):
  other=[k for k in range(3)if k!=axis]
  for sign in [-1,1]:
   ids=[]
   for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]:
    corner=[0,0,0];corner[axis]=sign;corner[other[0]]=a;corner[other[1]]=b;ids.append(signs.index(tuple(corner)))
   f.extend([[ids[0],ids[1],ids[2]],[ids[0],ids[2],ids[3]]])
 return v,np.asarray(f,np.int64)

def target_part(tree,object_frames,part):
 meshes=[]
 for link in tree.xml.findall('link'):
  name=link.get('name')
  if not name.startswith(PARTS[part]):continue
  for visual in link.findall('visual'):
   mesh=visual.find('geometry/mesh')
   if mesh is not None:
    v,f=obj(tree.path.parent/mesh.get('filename'));scale=np.array([float(x) for x in mesh.get('scale','1 1 1').split()]);v=v*scale
   else:
    box=visual.find('geometry/box')
    if box is None:raise ValueError('nonmesh/nonbox target visual unsupported')
    v,f=box_boundary([float(x) for x in box.get('size').split()])
   origin=visual.find('origin');t=np.eye(4)
   if origin is not None:
    t[:3,:3]=Rotation.from_euler('xyz',[float(x) for x in origin.get('rpy','0 0 0').split()]).as_matrix();t[:3,3]=[float(x) for x in origin.get('xyz','0 0 0').split()]
   t=mm(object_frames[name],t);v=np.einsum('ij,nj->ni',t[:3,:3],v)+t[:3,3];meshes.append((v,f))
 return combine(meshes)

def grid(h,half=.025):
 cells=round(2*half/h)
 if abs(cells*h-2*half)>1e-12:raise ValueError('grid mismatch')
 u=-half+(np.arange(cells)+FRACTIONS[0])*h;v=-half+(np.arange(cells)+FRACTIONS[1])*h;uv=np.array(np.meshgrid(u,v,indexing='ij')).reshape(2,-1).T;points=[]
 for axis in range(3):
  other=[i for i in range(3) if i!=axis]
  for sign in [-1,1]:
   p=np.zeros((len(uv),3));p[:,axis]=sign*half;p[:,other]=uv;points.append(p)
 rho=h*math.sqrt(max(FRACTIONS[0],1-FRACTIONS[0])**2+max(FRACTIONS[1],1-FRACTIONS[1])**2)
 return np.vstack(points),rho

def max_interval(source,target,rho):
 delta=np.abs(source-target);maximum=float(delta.max())
 return {'sampled_max_m':maximum,'continuous_max_lower_m':max(0,maximum-2*DISTANCE_ERROR_M),'continuous_max_upper_m':maximum+2*rho+2*DISTANCE_ERROR_M,'covering_radius_m':rho,'metric':'sup over complete cube surface of abs(unsigned distance target part - unsigned distance native part)','conditional_distance_error_per_query_m':DISTANCE_ERROR_M}

def compare_meshes(source,baseline,candidate,*,output_dir,prefix):
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);levels=[]
 for h in GRIDS_M:
  p,rho=grid(h);s,ns=source.query(p);b,nb=baseline.query(p);c,nc=candidate.query(p)
  bi=max_interval(s,b,rho);ci=max_interval(s,c,rho)
  passed=ci['continuous_max_upper_m']<=bi['continuous_max_lower_m'];failed=ci['continuous_max_lower_m']>bi['continuous_max_upper_m']
  name=prefix+f'_{h*1000:g}mm.npz';np.savez_compressed(out/name,queries_object_m=p,source_distance_m=s,baseline_distance_m=b,candidate_distance_m=c)
  levels.append({'grid_m':h,'query_count':len(p),'triangle_pairs':ns+nb+nc,'baseline':bi,'candidate':ci,'decision':'PASS_NONREGRESSION' if passed else 'FAIL_REGRESSION' if failed else 'UNRESOLVED','archive':str(out/name)})
  if passed or failed:break
 return {'part':prefix,'levels':levels,'decision':levels[-1]['decision'],'fitting_queries_used':False,'uncapped_distances':True,'conditional_f64_not_exact_arithmetic':True}

def adaptive_compare_meshes(source,baseline,candidate,*,output_dir,prefix):
 """Equivalent complete-cover max bounds with safe nonmaximal-cell pruning.

 Each queried cell bounds both residual fields by the same 2-Lipschitz rule.
 Children partition the parent exactly. Unrefined leaves retain their upper
 bounds. Parent upper bounds remain valid on every child and may tighten it.
 """
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 cells=np.array([(face,i,j) for face in range(6) for i in range(50) for j in range(50)],dtype=np.int64)
 inherited=np.full((len(cells),2),np.inf);leaves=[];archives=[];lower=np.zeros(2);all_levels=[]
 for level,h in enumerate(GRIDS_M):
  rho=h*math.sqrt(max(FRACTIONS[0],1-FRACTIONS[0])**2+max(FRACTIONS[1],1-FRACTIONS[1])**2)
  p=np.zeros((len(cells),3))
  for face in range(6):
   mask=cells[:,0]==face;axis=face//2;sgn=-1 if face%2==0 else 1;other=[k for k in range(3) if k!=axis]
   p[mask,axis]=sgn*.025;p[np.ix_(mask,other)]=-.025+(cells[mask,1:]+np.array(FRACTIONS))*h
  s,ns=source.query(p);b,nb=baseline.query(p);c,nc=candidate.query(p);residual=np.column_stack([abs(s-b),abs(s-c)])
  lower=np.maximum(lower,np.maximum(0,residual.max(axis=0)-2*DISTANCE_ERROR_M))
  upper=np.minimum(inherited,residual+2*rho+2*DISTANCE_ERROR_M)
  if np.any(upper+1e-14<residual-2*DISTANCE_ERROR_M):raise ValueError('inconsistent enclosing bounds')
  file=out/(prefix+f'_level{level}_queries.npz');np.savez_compressed(file,cells=cells,queries_object_m=p,source_distance_m=s,baseline_distance_m=b,candidate_distance_m=c,cell_upper_m=upper,inherited_upper_m=inherited)
  archives.append(str(file));all_levels.append({'grid_m':h,'queried_cell_count':len(cells),'triangle_pairs':ns+nb+nc,'covering_radius_m':rho,'sampled_global_lower_m':lower.tolist(),'query_archive':str(file)})
  active_bounds=[x[2] for x in leaves]+[upper];global_upper=np.vstack(active_bounds).max(axis=0)
  passed=global_upper[1]<=lower[0];failed=lower[1]>global_upper[0]
  if passed or failed or level==len(GRIDS_M)-1:
   leaves.append((level,cells,upper));break
  refine=(upper[:,0]>lower[0])|(upper[:,1]>lower[1])
  if np.any(~refine):leaves.append((level,cells[~refine],upper[~refine]))
  chosen=cells[refine];bounds=upper[refine]
  if not len(chosen):break
  # Fixed lexicographic children at exactly half the parent side length.
  cells=np.vstack([np.column_stack([chosen[:,0],2*chosen[:,1]+di,2*chosen[:,2]+dj]) for di,dj in [(0,0),(0,1),(1,0),(1,1)]])
  inherited=np.vstack([bounds]*4)
 # Every complete fine-grid tile must belong to exactly one retained leaf.
 occupancy=np.zeros((6,200,200),np.uint8);leaf_rows=[];leaf_bounds=[]
 for level,cc,uu in leaves:
  factor=2**(2-level)
  for (face,i,j),bound in zip(cc,uu,strict=True):
   occupancy[face,i*factor:(i+1)*factor,j*factor:(j+1)*factor]+=1;leaf_rows.append([level,face,i,j]);leaf_bounds.append(bound)
 if not np.all(occupancy==1):raise ValueError('adaptive cube partition does not cover each tile exactly once')
 upper=np.asarray(leaf_bounds).max(axis=0)
 decision='PASS_NONREGRESSION' if upper[1]<=lower[0] else 'FAIL_REGRESSION' if lower[1]>upper[0] else 'UNRESOLVED'
 lf=out/(prefix+'_leaf_cover.npz');np.savez_compressed(lf,leaf_cells=np.asarray(leaf_rows,np.int64),leaf_upper_m=leaf_bounds,global_lower_m=lower,global_upper_m=upper,coverage_counts=occupancy)
 return {'part':prefix,'decision':decision,'baseline':{'continuous_max_lower_m':float(lower[0]),'continuous_max_upper_m':float(upper[0])},'candidate':{'continuous_max_lower_m':float(lower[1]),'continuous_max_upper_m':float(upper[1])},'levels':all_levels,'leaf_cover_archive':str(lf),'full_cube_coverage_verified':True,'finest_grid_m':GRIDS_M[len(all_levels)-1],'uncapped_distances':True,'fitting_queries_used':False,'conditional_f64_not_exact_arithmetic':True,'distance_error_per_query_m':DISTANCE_ERROR_M,'pruning':'only retain unrefined cells whose both upperbounds cannot exceed corresponding existing global lower; every leaf bound retained in final maximum'}
