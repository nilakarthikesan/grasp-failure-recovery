"""Pure continuous64 CAMERA-graph material points; never a source32 derivative.

No assets, MANO layer, engine, search or integration are constructed here.
The caller provides immutable literal buffers and the frozen complete LBS core.
"""
from dataclasses import dataclass
import hashlib,json
import numpy as np
import torch

PROFILE='mano_source_initialized_camera_graph_continuous64_contact_v1'
TANGENT='mano_named_tangent_3slides_16local_angular_velocities_v1'
PARTS=('palm','index','middle','ring','little','thumb')
NORMALS=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
TANGENTS=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
BARYCENTRIC=np.array([1/3,1/3,1/3],dtype=np.float64)
PENETRATION_M=1e-4
FD_STEP=1e-6


def digest(value):
 a=np.ascontiguousarray(value);return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def array(value,shape,name):
 a=np.asarray(value)
 if a.shape!=shape or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Explicit finite float64 '+name+' required')
 return a

def face_normals(vertices,faces):
 v=array(vertices,(778,3),'full vertices');f=np.asarray(faces)
 if f.shape!=(1552,3) or f.dtype.kind not in 'iu' or np.any(f<0) or np.any(f>=778):raise ValueError('All literal1552faces required')
 tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1)
 if not np.isfinite(cross).all() or not np.isfinite(length).all() or np.any(length<=1e-15):raise ValueError('Nonfinite/degenerate facet refused without replacement')
 return cross/length[:,None]


def fixed_material_points(faces,labels):
 f=np.asarray(faces);l=np.asarray(labels)
 if f.shape!=(1552,3) or f.dtype.kind not in 'iu' or np.any(f<0) or np.any(f>=778) or l.shape!=(1552,) or l.dtype.kind not in 'iu' or set(l)!=set(range(6)):raise ValueError('Exact original faces/six anatomical labels required')
 return [dict(part=part,part_id=i,face_id=int(np.flatnonzero(l==i)[0]),vertex_ids=f[np.flatnonzero(l==i)[0]].tolist(),barycentric=BARYCENTRIC.tolist(),normal_world=NORMALS[i].tolist(),sliding_tangent_world=TANGENTS[i].tolist(),plane_penetration_m=PENETRATION_M) for i,part in enumerate(PARTS)]


class CameraGraph:
 """Exact geometry/tangent profile: world state→camera LBS→world placement.

 This smooth float64 model uses literal sourcefloat32 buffers explicitly cast
 to double. Rounding of AA32/source operations is not differentiated or fitted.
 """
 def __init__(self,*,core,qpos,rest_quaternions,camera_world,beta32,arrays32,source_hashes,count=None):
  self.core=core;self.q=array(qpos,(67,),'source-initialized namedqpos').copy();rest=array(rest_quaternions,(16,4),'rest body quaternions');C=array(camera_world,(4,4),'cameraWorld')
  if not np.array_equal(C[3],[0.,0.,0.,1.]) or np.max(np.abs(C[:3,:3].T@C[:3,:3]-np.eye(3)))>1e-12 or abs(np.linalg.det(C[:3,:3])-1)>1e-12:raise ValueError('Proper original camera required')
  if np.max(np.abs((rest*rest).sum(1)-1))>1e-12 or np.max(np.abs((self.q[3:].reshape(16,4)**2).sum(1)-1))>1e-12:raise ValueError('Literal unit named ball quaternions required')
  beta=np.asarray(beta32)
  if beta.dtype!=np.float32 or beta.shape!=(1,10) or not np.isfinite(beta).all():raise ValueError('Exact literal float32beta row required')
  if set(arrays32)!=set(core.ARRAY_KEYS):raise ValueError('Exact literal source buffer inventory required')
  tensors={}
  for key,value in arrays32.items():
   a=np.asarray(value)
   if a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Literal float32 source buffers required')
   tensors[key]=torch.from_numpy(a.copy()).double()
  self.arrays=tensors;self.beta=torch.from_numpy(beta.copy()).double();core.validate_arrays(tensors,self.beta)
  self.base=torch.from_numpy(self.q.copy());self.rest=torch.from_numpy(rest.copy());self.C=torch.from_numpy(C.copy());self.count=count
  payload=dict(schema=PROFILE,tangent_schema=TANGENT,precision='literalfloat32_buffers_cast64; cameraLBS64_then_world64; noAA32quantizationderivative',source_hashes=source_hashes,
   camera_sha256=digest(C),rest_sha256=digest(rest),beta_sha256=digest(beta),buffers={k:digest(v) for k,v in arrays32.items()})
  self.profile_hash=canonical(payload);self.profile_payload=payload;self.state_hash=canonical(dict(profile=self.profile_hash,qpos_sha256=digest(self.q)))
 def vertices(self,delta,kind):
  if not isinstance(delta,torch.Tensor) or delta.dtype!=torch.float64 or delta.shape!=(51,) or not bool(torch.isfinite(delta).all()):raise ValueError('Explicit float64 named51 tangent required')
  if self.count:self.count(kind)
  joint=self.core.quaternion_product(self.base[3:].reshape(16,4),self.core.rotation_increment(delta[3:].reshape(16,3)))
  body=self.core.quaternion_product(self.rest,joint);Rworld=self.core.quaternion_matrices(body)
  Rcamera=torch.cat(((self.C[:3,:3].T@Rworld[0])[None],Rworld[1:]),dim=0)
  tcamera=self.C[:3,:3].T@(self.base[:3]+delta[:3]-self.C[:3,3])
  v_camera=self.core.skin_state(Rcamera,tcamera,self.beta,self.arrays)['vertices']
  result=torch.einsum('ij,vj->vi',self.C[:3,:3],v_camera)+self.C[:3,3]
  if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite camera-graph skin')
  return result
 def baseline(self):
  result=self.vertices(torch.zeros(51,dtype=torch.float64),'baseline_skin_evaluations');self.registered_vertices_hash=digest(result.detach().numpy());return result
 def jacobian(self):
  columns=[];zero=torch.zeros(51,dtype=torch.float64)
  for k in range(51):
   direction=torch.zeros_like(zero);direction[k]=1.
   with torch.autograd.forward_ad.dual_level():
    dual=torch.autograd.forward_ad.make_dual(zero,direction);_,value=torch.autograd.forward_ad.unpack_dual(self.vertices(dual,'jvp_skin_evaluations'));columns.append(value.clone())
  result=torch.stack(columns,-1);self.registered_J_hash=digest(result.detach().numpy());return result
 def finite_difference(self):
  columns=[]
  for k in range(51):
   delta=torch.zeros(51,dtype=torch.float64);delta[k]=FD_STEP
   columns.append((self.vertices(delta,'fd_skin_evaluations')-self.vertices(-delta,'fd_skin_evaluations'))/(2*FD_STEP))
  return torch.stack(columns,-1)
 def force_vjp(self,forces):
  f=torch.from_numpy(array(forces,(len(self.arrays['template']),3),'fixed vertex forces').copy());delta=torch.zeros(51,dtype=torch.float64,requires_grad=True)
  return torch.autograd.grad((self.vertices(delta,'vjp_skin_evaluations')*f).sum(),delta)[0]


@dataclass(frozen=True)
class ContactPoint:
 profile_hash:str
 state_hash:str
 face_id:int
 vertex_ids:tuple
 barycentric:np.ndarray
 point_world_m:np.ndarray
 point_J:np.ndarray
 buffer_hash:str
 def verify(self,profile_hash,state_hash):
  if profile_hash!=self.profile_hash or state_hash!=self.state_hash:raise ValueError('Mixed geometry/tangent profile or state refused')
  current=canonical(dict(face_id=self.face_id,vertex_ids=list(self.vertex_ids),b=digest(self.barycentric),p=digest(self.point_world_m),J=digest(self.point_J)))
  if current!=self.buffer_hash:raise ValueError('Contact point buffer mutated')
 def load(self,force,velocity,*,profile_hash,state_hash):
  self.verify(profile_hash,state_hash);F=array(force,(3,),'held world force');v=array(velocity,(51,),'named velocity')
  tau=np.einsum('ij,i->j',self.point_J,F,optimize=False);point_v=np.einsum('ij,j->i',self.point_J,v,optimize=False)
  return tau,point_v,float(tau@v-F@point_v)


def material_point(graph,vertices,jacobian,selection):
 V=np.asarray(vertices);J=np.asarray(jacobian)
 if V.shape!=(len(graph.arrays['template']),3) or J.shape!=(len(V),3,51) or V.dtype!=np.float64 or J.dtype!=np.float64 or not np.isfinite(V).all() or not np.isfinite(J).all():raise ValueError('Matching finite64 vertex/J arrays required')
 if digest(V)!=getattr(graph,'registered_vertices_hash',None) or digest(J)!=getattr(graph,'registered_J_hash',None):raise ValueError('Unregistered/mixed geometry and tangent buffers refused')
 ids=np.asarray(selection['vertex_ids'],dtype=np.int64);b=array(np.asarray(selection['barycentric'],dtype=np.float64),(3,),'fixed barycentric')
 if ids.shape!=(3,) or np.any(ids<0) or np.any(ids>=len(V)) or np.any(b<0) or abs(b.sum()-1)>1e-12:raise ValueError('Literal material triangle required')
 p=np.einsum('i,ij->j',b,V[ids],optimize=False);jp=np.einsum('i,ijk->jk',b,J[ids],optimize=False);b=b.copy();p=p.copy();jp=jp.copy()
 for a in (b,p,jp):a.setflags(write=False)
 h=canonical(dict(face_id=selection['face_id'],vertex_ids=ids.tolist(),b=digest(b),p=digest(p),J=digest(jp)))
 return ContactPoint(graph.profile_hash,graph.state_hash,selection['face_id'],tuple(ids.tolist()),b,p,jp,h)
