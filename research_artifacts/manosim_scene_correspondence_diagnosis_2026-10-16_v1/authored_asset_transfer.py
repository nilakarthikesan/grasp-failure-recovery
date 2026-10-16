"""Pure compiled authored-object vertex/frame transfer; no engine operations.

5um is the existing numerical reconstruction limit. This is NOT an RGB-D noise
or physical-contact fidelity allowance. Source poses/meshes are never adjusted.
"""
from pathlib import Path
from itertools import product
from xml.etree import ElementTree as ET
import hashlib
import numpy as np
LIMIT_M=5e-6
ROTATION_LIMIT=1e-6
MESH_OPERANDS=('mesh_vertadr','mesh_vertnum','mesh_faceadr','mesh_facenum','mesh_vert','mesh_face','mesh_pos','mesh_quat','geom_pos','geom_quat','geom_bodyid','geom_dataid','geom_type','body_pos','body_quat','body_parentid')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(v,message):
 if not bool(v):raise ValueError(message)
def shaped(a,shape):
 a=np.asarray(a,dtype=float);require(a.shape==shape and np.isfinite(a).all(),'Finite shaped source/compiled asset operand');return a
def rotation(q):
 w,x,y,z=shaped(q,(4,));n=w*w+x*x+y*y+z*z;require(abs(n-1)<=1e-12,'Proper compiled asset quaternion');s=2/n
 return np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],[s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],[s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])
def source_mesh(path):
 vertices=[];faces=[]
 for line in Path(path).read_text().splitlines():
  p=line.split()
  if p and p[0]=='v':vertices.append(list(map(float,p[1:4])))
  elif p and p[0]=='f':
   require(len(p)==4,'Literal triangular OBJ required');faces.append([int(x.split('/')[0])-1 for x in p[1:]])
 v=np.asarray(vertices,dtype=np.float32);f=np.asarray(faces,dtype=np.int64)
 require(v.ndim==2 and v.shape[1]==3 and len(v)>0 and np.isfinite(v).all() and f.ndim==2 and f.shape[1]==3 and len(f)>0 and f.min()>=0 and f.max()<len(v),'Valid literal source OBJ')
 return v.astype(np.float64),f

def epsilon_coverage(a,b,tolerance_m=LIMIT_M):
 """Exact existential <=epsilon distance both directions, using 27 grid cells.

 Any Euclidean neighbor <=epsilon lies in the same/adjacent cell on each axis.
 No match truncation or fitting. A returned max is the max LOCAL nearest match;
 the boolean is equivalent to bidirectional Hausdorff distance <=epsilon.
 """
 require(tolerance_m==LIMIT_M,'Original5um numerical reconstruction limit only')
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 require(a.ndim==b.ndim==2 and a.shape[1]==b.shape[1]==3 and len(a)>0 and len(b)>0 and np.isfinite(a).all() and np.isfinite(b).all(),'Finite nonempty vertex sets')
 offsets=tuple(product((-1,0,1),repeat=3))
 def one(x,y):
  grid={}
  for i,c in enumerate(np.floor(y/tolerance_m).astype(np.int64)):grid.setdefault(tuple(c),[]).append(i)
  worst=0.
  for point,c in zip(x,np.floor(x/tolerance_m).astype(np.int64),strict=True):
   ids=[]
   for d in offsets:ids.extend(grid.get((c[0]+d[0],c[1]+d[1],c[2]+d[2]),()))
   require(bool(ids),'Compiled/source vertex outside5um neighborhood')
   delta=y[ids]-point;distance=float(np.sqrt(np.min(np.einsum('ij,ij->i',delta,delta,optimize=False))))
   require(distance<=tolerance_m,'Compiled/source vertex exceeds5um reconstruction limit');worst=max(worst,distance)
  return worst
 return dict(source_to_compiled_max_match_m=one(a,b),compiled_to_source_max_match_m=one(b,a),numerical_limit_m=tolerance_m,source_vertices=len(a),compiled_vertices=len(b),bidirectional_epsilon_coverage=True)

def validate_authored_object_transfer(model,resolve_name,object_xml,world_position,object_body_id):
 """Read existing model arrays only. resolve_name(kind,name) is name lookup."""
 object_xml=Path(object_xml).resolve();tree=ET.parse(object_xml).getroot();body=tree.find('./worldbody/body');require(body is not None and len(tree.findall('./worldbody//body'))==1,'One authored object body')
 world=shaped(world_position,(3,));bid=int(object_body_id);require(int(model.body_parentid[bid])==0,'Authored object is world-root body')
 require(np.max(np.abs(shaped(model.body_pos[bid],(3,))-world))<=LIMIT_M and np.max(np.abs(rotation(model.body_quat[bid])-np.eye(3)))<=ROTATION_LIMIT,'Compiled original object root frame changed')
 meshes={x.get('name'):x for x in tree.findall('./asset/mesh')};receipts=[];seen=set()
 for i,g in enumerate(body.findall('geom')):
  require(g.get('type')=='mesh' and g.get('mesh') in meshes,'Literal authored mesh geom');source=meshes[g.get('mesh')]
  require(np.array_equal(np.fromstring(g.get('pos','0 0 0'),sep=' '),[0,0,0]) and np.array_equal(np.fromstring(g.get('quat','1 0 0 0'),sep=' '),[1,0,0,0]) and not any(k in g.attrib for k in ('euler','axisangle','xyaxes','zaxis')),'Original identity source geom frame')
  require(np.array_equal(np.fromstring(source.get('scale','1 1 1'),sep=' '),[1,1,1]),'Original unit source mesh scale')
  gid=int(resolve_name('geom','pilot_object_geom_'+str(i)));mid=int(resolve_name('mesh','pilot_object__'+g.get('mesh')))
  require(gid not in seen and int(model.geom_bodyid[gid])==bid and int(model.geom_dataid[gid])==mid and int(model.geom_type[gid])==7,'Compiled authored mesh identity/ownership');seen.add(gid)
  adr=int(model.mesh_vertadr[mid]);num=int(model.mesh_vertnum[mid]);compiled=np.asarray(model.mesh_vert)[adr:adr+num].astype(float);require(compiled.shape==(num,3),'Compiled mesh vertex address range')
  pos=shaped(model.geom_pos[gid],(3,));rot=rotation(model.geom_quat[gid]);require(np.max(np.abs(pos-shaped(model.mesh_pos[mid],(3,))))<=LIMIT_M and np.max(np.abs(rot-rotation(model.mesh_quat[mid])))<=ROTATION_LIMIT,'Compiler geom/mesh recenter frame differs')
  restored=np.einsum('nj,ij->ni',compiled,rot,optimize=False)+pos
  path=(object_xml.parent/source.get('file')).resolve();vertices,faces=source_mesh(path);metric=epsilon_coverage(vertices,restored)
  receipts.append(dict(source_mesh=str(path),source_mesh_sha256=sha(path),source_vertex_precision='literal OBJ parsed tofloat32 (compiler input)',source_faces=len(faces),compiled_faces=int(model.mesh_facenum[mid]),geom_id=gid,mesh_id=mid,mesh_vertex_address=adr,mesh_vertex_count=num,source_geom_frame_identity=True,compiled_geom_position=pos.tolist(),compiled_geom_quaternion=np.asarray(model.geom_quat[gid]).tolist(),**metric))
 require(len(receipts)>0,'Authored object mesh inventory empty')
 return dict(schema_version='manosim_authored_object_compiled_vertex_frame_transfer_v1',source_object_xml=str(object_xml),source_object_xml_sha256=sha(object_xml),object_world_position=world.tolist(),object_body_id=bid,geoms=receipts,numerical_reconstruction_limit_m=LIMIT_M,rotation_component_limit=ROTATION_LIMIT,compiled_asset_vertex_frame_transfer_passed=True,RGBD_correspondence_qualified=False,full_surface_contact_parity_claimed=False,triangle_topology_or_collision_hull_independently_replayed=False,geometry_or_pose_fitting=False)
