"""Independent bounded MJCF/OBJ scene reference. No simulator/project imports."""
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree
from xml_reference import transform,quaternion_matrix_wxyz,transform_points,symmetric_vertex_distance

def vector(value,n,default):
 v=np.asarray(default,float) if value is None else np.fromstring(value,sep=' ')
 if v.shape!=(n,) or not np.isfinite(v).all():raise ValueError('Finite explicit vector required')
 return v

def frame(node):
 if any(n in node.attrib for n in ('euler','axisangle','xyaxes','zaxis')):raise ValueError('Reference currently supports position/quaternion frames only')
 return transform(vector(node.get('pos'),3,[0,0,0]),quaternion_matrix_wxyz(vector(node.get('quat'),4,[1,0,0,0])))

def read_obj(path,scale):
 vertices=[];faces=[]
 for line in Path(path).read_text().splitlines():
  f=line.split()
  if not f:continue
  if f[0]=='v':vertices.append([float(x) for x in f[1:4]])
  elif f[0]=='f':
   ids=[int(x.split('/')[0]) for x in f[1:]]
   if len(ids)!=3 or any(i<=0 for i in ids):raise ValueError('Bounded reference requires positive-index OBJ triangles')
   faces.append([i-1 for i in ids])
 v=np.asarray(vertices,float)*np.asarray(scale,float);f=np.asarray(faces,int)
 if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3 or np.any(f<0) or np.any(f>=len(v)):raise ValueError('Invalid OBJ')
 return v,f

def read_object(path):
 path=Path(path);root=ET.parse(path).getroot();body=root.find('worldbody/body');compiler=root.find('compiler')
 if any(n.tag not in {'compiler','asset','worldbody'} for n in root) or any('class' in n.attrib or 'childclass' in n.attrib for n in root.iter()):raise ValueError('Defaults/includes or inherited MJCF settings are outside bounded reference')
 if compiler is None or set(compiler.attrib)-{'angle','meshdir'}:raise ValueError('Unhandled compiler options')
 if body is None or set(body.attrib)-{'name','pos','quat'} or any(n.tag not in {'geom','inertial','freejoint'} for n in body):raise ValueError('Unhandled body options')
 if len(body.findall('freejoint'))!=1 or len(body.findall('inertial'))!=1:raise ValueError('Exactly one free joint and explicit inertial required')
 if any(n.tag!='mesh' for n in root.find('asset')):raise ValueError('Only explicit mesh assets supported')
 if compiler.get('angle')!='radian' or len(root.findall('.//body'))!=1:raise ValueError('Only declared radian single-body object supported')
 meshes={};files=[path]
 for mesh in root.findall('asset/mesh'):
  if set(mesh.attrib)-{'name','file','scale','refpos','refquat'}:raise ValueError('Unhandled mesh options')
  if 'refpos' in mesh.attrib or 'refquat' in mesh.attrib:raise ValueError('Mesh reference frames require a separately extended independent reference')
  p=(path.parent/compiler.get('meshdir','.')/mesh.get('file')).resolve();scale=vector(mesh.get('scale'),3,[1,1,1]);v,f=read_obj(p,scale);meshes[mesh.get('name')]={'path':str(p),'vertices':v,'faces':f};files.append(p)
 rows=[];collision=0;visual=0
 for j,g in enumerate(body.findall('geom')):
  if set(g.attrib)-{'name','type','mesh','pos','quat','contype','conaffinity','friction','margin','gap','group','rgba'}:raise ValueError('Unhandled geom options')
  if g.get('type')!='mesh':raise ValueError('Explicit object mesh required')
  ct=int(g.get('contype','1'));ca=int(g.get('conaffinity','1'));active=bool(ct or ca)
  name=('object_geom' if collision==0 else f'object_collision_{collision:03}') if active else f'object_visual_{visual:03}'
  if active:collision+=1
  else:visual+=1
  rows.append({'ordinal':j,'compiled_name':name,'mesh':meshes[g.get('mesh')],'local_frame':frame(g),'collision_active':active,'contype':ct,'conaffinity':ca,'friction':vector(g.get('friction'),3,[1,.005,.0001]),'margin':float(g.get('margin','0')),'gap':float(g.get('gap','0')),'group':int(g.get('group','0'))})
 inertial=body.find('inertial');mass=float(inertial.get('mass'));pose=frame(inertial)
 if 'fullinertia' in inertial.attrib:
  xx,yy,zz,xy,xz,yz=vector(inertial.get('fullinertia'),6,None);I=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
 else:I=np.diag(vector(inertial.get('diaginertia'),3,None))
 I=pose[:3,:3]@I@pose[:3,:3].T
 return {'path':str(path),'files':list(dict.fromkeys(str(p) for p in files)),'rows':rows,'native_mass':mass,'inertial_position':pose[:3,3],'inertia_tensor_body':I,'default_body_frame':frame(body)}

def triangle_connectivity(source_vertices,source_faces,compiled_world_vertices,compiled_faces,world_source_frame,tolerance):
 expected=transform_points(world_source_frame,source_vertices)
 # Duplicate raw OBJ vertices are one geometric vertex for this comparison.
 unique,inverse=np.unique(expected,axis=0,return_inverse=True);distance,which=cKDTree(unique).query(compiled_world_vertices)
 def canon(face):
  q=tuple(int(x) for x in face);return min(q,q[1:]+q[:1],q[2:]+q[:2])
 a=Counter(canon(inverse[f]) for f in source_faces);b=Counter(canon(which[f]) for f in compiled_faces)
 return {'maximum_compiled_vertex_to_source_m':float(distance.max()),'oriented_geometric_triangle_multisets_equal':a==b,'source_triangle_count':len(source_faces),'compiled_triangle_count':len(compiled_faces),'passed':bool(float(distance.max())<=tolerance and a==b)}
