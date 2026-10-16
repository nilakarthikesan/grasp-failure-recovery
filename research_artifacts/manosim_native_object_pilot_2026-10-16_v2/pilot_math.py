"""Prospective native-object pilot math. No engine/model work on import."""
from pathlib import Path
import hashlib,io,pickle
from xml.etree import ElementTree as ET
import numpy as np
PARENTS=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
TASK_SECONDS=5.5
MAX_VISUAL_TRIANGLES=10000
QUERY_COUNT=32
MATCH_LIMITS=dict(object_COM_m=.002,wrist_position_m=.002,bone_angle_rad=float(np.deg2rad(2)),failure_time_s=.01)


def finite(x,shape):
 a=np.asarray(x,dtype=np.float64)
 if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite shaped pilot operand required')
 return a

def rotation(q):
 w,x,y,z=finite(q,(4,));norm=w*w+x*x+y*y+z*z
 if abs(norm-1)>1e-8:raise ValueError('Raw unit quaternion required')
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def product(a,b):
 a=finite(a,(4,));b=finite(b,(4,));return np.r_[a[0]*b[0]-a[1:]@b[1:],a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]

def endpoint_pose(qpos,rest0,object_ipos):
 q=finite(qpos,(74,));return q[:3].copy(),rotation(product(rest0,q[3:7])),q[67:70]+rotation(q[70:74])@finite(object_ipos,(3,))

def bone_rotations(qpos,rest):
 q=finite(qpos,(67,));rest=finite(rest,(16,4));world=[]
 for i in range(16):
  R=rotation(product(rest[i],q[3+4*i:7+4*i]));world.append(R if i==0 else world[PARENTS[i]]@R)
 return np.asarray(world)

def phase(t):return 'reset' if t<.25 else 'approach' if t<1. else 'close_fingers' if t<1.5 else 'lift' if t<2.5 else 'hold'

def target(t,goal,opened,start,u):
 if t<.25:return start.copy()
 if t<.75:return u.trajectory(start,opened,u.smooth_fraction(t-.25,.5))
 if t<1.:return opened.copy()
 if t<1.5:return u.trajectory(opened,goal,u.smooth_fraction(t-1.,.5))
 q=goal.copy()
 if t>=1.5:q[:3]+=[0,0,.15*u.smooth_fraction(t-1.5,1.)]
 return q


class ArrayUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={('numpy','ndarray'),('numpy','dtype'),('_codecs','encode')}|{(m,n) for m in ('numpy.core.multiarray','numpy._core.multiarray') for n in ('_reconstruct','scalar')}|{(m,'_frombuffer') for m in ('numpy.core.numeric','numpy._core.numeric')}
  if (module,name) not in allowed:raise ValueError('Unsupported observation pickle global')
  return super().find_class(module,name)


def observation(path):
 d=ArrayUnpickler(io.BytesIO(Path(path).read_bytes())).load()
 if not isinstance(d,dict) or d.get('grasp') is not None:raise ValueError('Unlabeled upstream observation required')
 return d


def obj_mesh(path):
 vertices=[];faces=[]
 for line in Path(path).read_text().splitlines():
  parts=line.split()
  if not parts:continue
  if parts[0]=='v':vertices.append([float(x) for x in parts[1:4]])
  elif parts[0]=='f':
   if len(parts)!=4:raise ValueError('Source triangle mesh required; no retriangulation')
   face=[int(x.split('/')[0])-1 for x in parts[1:]]
   if min(face)<0:raise ValueError('Literal positive OBJ indices required')
   faces.append(face)
 v=np.asarray(vertices,dtype=float);f=np.asarray(faces,dtype=np.int64)
 if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3 or not len(f) or np.max(f)>=len(v):raise ValueError('Invalid source mesh')
 return v,f


def nearest_surface(points,triangles):
 """Unsigned point-triangle query, including degenerate segment fallback."""
 points=np.asarray(points,dtype=float);tri=np.asarray(triangles,dtype=float)
 if points.ndim!=2 or points.shape[1]!=3 or tri.ndim!=3 or tri.shape[1:]!=(3,3) or not len(tri) or not np.isfinite(points).all() or not np.isfinite(tri).all():raise ValueError('Finite source triangle/query arrays required')
 a,b,c=tri[:,0],tri[:,1],tri[:,2];u=b-a;v=c-a;uu=np.einsum('ij,ij->i',u,u,optimize=False);uv=np.einsum('ij,ij->i',u,v,optimize=False);vv=np.einsum('ij,ij->i',v,v,optimize=False);den=uu*vv-uv*uv
 ids=[];nearest=[];distances=[]
 for p in points:
  pa=p-a;pu=np.einsum('ij,ij->i',pa,u,optimize=False);pv=np.einsum('ij,ij->i',pa,v,optimize=False);s=np.divide(pu*vv-pv*uv,den,out=np.zeros(len(tri)),where=den>0);t=np.divide(pv*uu-pu*uv,den,out=np.zeros(len(tri)),where=den>0)
  interior=(den>0)&(s>=0)&(t>=0)&(s+t<=1);plane=a+s[:,None]*u+t[:,None]*v;candidates=[]
  for x,y in ((a,b),(b,c),(c,a)):
   edge=y-x;d=np.einsum('ij,ij->i',edge,edge,optimize=False);h=np.divide(np.einsum('ij,ij->i',p-x,edge,optimize=False),d,out=np.zeros(len(tri)),where=d>0);candidates.append(x+np.clip(h,0,1)[:,None]*edge)
  candidates.append(np.where(interior[:,None],plane,candidates[0]));stack=np.stack(candidates,1);d2=np.einsum('nki,nki->nk',stack-p,stack-p,optimize=False);which=d2.argmin(axis=1);closest=stack[np.arange(len(tri)),which];best=int(np.argmin(d2[np.arange(len(tri)),which]));ids.append(best);nearest.append(closest[best]);distances.append(float(np.sqrt(d2[best,which[best]])))
 return np.asarray(ids),np.asarray(nearest),np.asarray(distances)


def alignment_inputs(data):
 from PIL import Image
 depth=np.asarray(Image.open(io.BytesIO(data['depth'])));mask=np.asarray(Image.open(io.BytesIO(data['object_mask'])).convert('L'))>0
 K=finite(data['camera']['K'],(3,3));T=finite(data['T_world_camera'],(4,4))
 if depth.dtype!=np.uint16 or depth.shape!=mask.shape or depth.shape!=(data['camera']['height'],data['camera']['width']) or K[0,0]<=0 or K[1,1]<=0 or not np.array_equal(K[2],[0,0,1]) or K[0,1]!=0 or K[1,0]!=0:raise ValueError('Literal uint16-mm depth and calibrated zero-skew K required')
 padded=np.pad(mask,2);eroded=np.logical_and.reduce([padded[i:i+len(mask),j:j+mask.shape[1]] for i in range(5) for j in range(5)])
 valid=eroded&(depth>0)&(depth<65535);pixels=np.flatnonzero(valid)
 if len(pixels)<QUERY_COUNT:return dict(passed_support=False,pixel_ids=pixels,valid_mask=valid)
 selected=pixels[np.linspace(0,len(pixels)-1,QUERY_COUNT,dtype=int)];row,col=np.unravel_index(selected,depth.shape);z=depth[row,col].astype(float)/1000;ray=np.stack(((col-K[0,2])/K[0,0],(row-K[1,2])/K[1,1],np.ones(QUERY_COUNT)),1);camera=z[:,None]*ray;world=np.einsum('ij,nj->ni',T[:3,:3],camera,optimize=False)+T[:3,3]
 allowance=.001*np.linalg.norm(ray,axis=1)+.5*z*np.sqrt(K[0,0]**-2+K[1,1]**-2)
 if not np.isfinite(world).all() or np.max(np.abs(T[:3,:3].T@T[:3,:3]-np.eye(3)))>1e-10 or abs(np.linalg.det(T[:3,:3])-1)>1e-10 or not np.array_equal(T[3],[0,0,0,1]):raise ValueError('Proper literal camera and finite backprojection required')
 return dict(passed_support=True,pixel_ids=selected,pixel_uv=np.stack((col,row),1),valid_mask=valid,depth_m=z,camera_rays=ray,points_camera=camera,points_world=world,allowances_m=allowance,K=K,cameraWorld64=T)


def scene_xml(hand_xml,object_xml,world_position,table_z,dt,profile):
 if dt not in (.002,.001) or not np.isfinite(table_z):raise ValueError('Declared finite task scene required')
 hand_xml=Path(hand_xml).resolve();object_xml=Path(object_xml).resolve();h=ET.fromstring(profile.gravity_profile_xml(hand_xml.read_text()));o=ET.parse(object_xml).getroot()
 if o.findall('.//include') or len(o.findall('./worldbody//body'))!=1 or len(o.findall('./worldbody/body'))!=1 or len(o.findall('./worldbody//freejoint'))!=1 or o.findall('./worldbody//joint') or o.find('actuator') is not None or o.find('default') is not None:raise ValueError('Single original uncompensated source free object required')
 for mesh in h.findall('./asset/mesh'):mesh.set('file',str((hand_xml.parent/mesh.get('file')).resolve()))
 assets=h.find('asset');names={}
 for mesh in o.findall('./asset/mesh'):
  if not np.array_equal(np.fromstring(mesh.get('scale','1 1 1'),sep=' '),[1,1,1]) or any(k in mesh.attrib for k in ('refpos','refquat')):raise ValueError('Literal unscaled source mesh frame required')
  names[mesh.get('name')]='pilot_object__'+mesh.get('name');mesh.set('name',names[mesh.get('name')]);mesh.set('file',str((object_xml.parent/mesh.get('file')).resolve()));assets.append(mesh)
 body=o.find('./worldbody/body')
 if not np.array_equal(np.fromstring(body.get('quat','1 0 0 0'),sep=' '),[1,0,0,0]) or any(k in body.attrib for k in ('euler','axisangle','xyaxes','zaxis')) or not np.array_equal(np.fromstring(body.get('pos','0 0 0'),sep=' '),[0,0,0]):raise ValueError('Original flat identity source object frame required')
 body.set('name','pilot_object');body.set('pos',' '.join(map(str,finite(world_position,(3,)))));body.set('quat','1 0 0 0');body.set('gravcomp','0');body.find('freejoint').set('name','pilot_object_free')
 inertial=body.find('inertial')
 if inertial is None or not np.array_equal(np.fromstring(inertial.get('pos','0 0 0'),sep=' '),[0,0,0]):raise ValueError('Original COM-centred object inertial frame required')
 for i,g in enumerate(body.findall('geom')):
  if g.get('type')!='mesh' or not np.array_equal(np.fromstring(g.get('pos','0 0 0'),sep=' '),[0,0,0]) or not np.array_equal(np.fromstring(g.get('quat','1 0 0 0'),sep=' '),[1,0,0,0]) or any(k in g.attrib for k in ('euler','axisangle','xyaxes','zaxis')):raise ValueError('Original world-aligned local source geom frame required')
  visual=g.get('contype')=='0' and g.get('conaffinity')=='0'
  # Imported geoms must retain original stock compliance, not hand defaults.
  for key,value in dict(solref='.02 1',solimp='.9 .95 .001 .5 2',priority='0',condim='3').items():
   if key not in g.attrib:g.set(key,value)
  g.set('name','pilot_object_geom_'+str(i));g.set('mesh',names[g.get('mesh')]);g.set('contype','0' if visual else '1');g.set('conaffinity','0' if visual else '6')
 for g in h.findall('./worldbody//geom'):g.set('contype','2');g.set('conaffinity','5')
 h.find('worldbody').append(body);ET.SubElement(h.find('worldbody'),'geom',name='pilot_table',type='plane',size='2 2 .01',pos=f'0 0 {table_z}',contype='4',conaffinity='3',condim='3',friction='1 .005 .0001',solref='.02 1',solimp='.9 .95 .001 .5 2',priority='0')
 opt=h.find('option')
 if opt is None:opt=ET.SubElement(h,'option')
 for k,v in dict(timestep=str(dt),gravity='0 0 -9.81',integrator='implicitfast',cone='elliptic',jacobian='dense',solver='Newton',iterations='100',tolerance='1e-8').items():opt.set(k,v)
 flag=opt.find('flag')
 if flag is None:flag=ET.SubElement(opt,'flag')
 flag.set('contact','enable');flag.set('energy','enable');return ET.tostring(h,encoding='unicode')+'\n'


def object_model_receipt(model,body_id,source_xml,world_position,api):
 """Literal source mass/frame plus compiler inertia representation postcondition."""
 source=ET.parse(source_xml).getroot().find('./worldbody/body/inertial');mass=float(source.get('mass'));ipos=np.fromstring(source.get('pos','0 0 0'),sep=' ');full=np.fromstring(source.get('fullinertia'),sep=' ')
 if not np.isfinite(mass) or mass<=0 or mass!=float(model.body_mass[body_id]) or not np.array_equal(ipos,model.body_ipos[body_id]) or float(model.body_gravcomp[body_id])!=0 or not np.array_equal(np.asarray(model.qpos0)[67:70],finite(world_position,(3,))) or not np.array_equal(np.asarray(model.qpos0)[70:74],[1,0,0,0]):raise ValueError('Original uncompensated free object mass/COM/frame differs')
 inertia=api.inertia_representation_receipt(full,np.asarray(model.body_inertia)[body_id],np.asarray(model.body_iquat)[body_id])
 return dict(source_fullinertia_kg_m2=full.tolist(),compiled_principal_inertia_kg_m2=np.asarray(model.body_inertia)[body_id].tolist(),compiled_inertia_quaternion=np.asarray(model.body_iquat)[body_id].tolist(),mass_kg=mass,COM_local_m=ipos.tolist(),original_world_COM_m=list(world_position),inertia_representation=inertia,object_gravity_compensation=False)


def catalog_seven_fold_manifest(schedule,spec,catalog_binding,fold_binding):
 """Join the fixed catalog object roles before outcomes; separate from trainer5."""
 primary=[r for r in schedule if r['cohort']=='primary'];objects={r['object_id'] for r in primary};folds=spec.get('folds',[])
 if len(primary)!=120 or len(objects)!=30 or len(folds)!=7 or len({f['fold_id'] for f in folds})!=7 or spec.get('schema_version')!='hug_native_small_pilot_object_category_fold_spec_v1' or spec.get('catalog_sha256')!=catalog_binding['sha256'] or spec.get('training_performed') is not False or spec.get('no_labels_loaded') is not True:raise ValueError('Exact prospective catalog seven folds required')
 output=[]
 for fold in folds:
  roles={role:list(fold[role+'_object_ids']) for role in ('train','validation','test')};sets={k:set(v) for k,v in roles.items()}
  if any(len(roles[k])!=len(sets[k]) or not sets[k] for k in sets) or set.union(*sets.values())!=objects or any(sets[a]&sets[b] for a,b in (('train','validation'),('train','test'),('validation','test'))):raise ValueError('Physical object roles must be exhaustive and disjoint')
  assignments={row['attempt_id']:next(role for role,ids in sets.items() if row['object_id'] in ids) for row in primary}
  output.append(dict(fold_id=fold['fold_id'],generalization_claim=fold.get('generalization_claim'),object_membership=roles,assignments=assignments))
 return dict(schema_version='manosim_native_pilot_catalog_seven_folds_v1',catalog=catalog_binding,source_fold_spec=fold_binding,selection_uses_outcomes=False,cohort='primary120_only_matched1ms_excluded',trainer_five_category_manifest_unchanged=True,seven_fold_training_API_implemented=False,folds=output)


def source_cast_camera_operands(saved_K,saved_camera,capture_K,capture_camera):
 """Exact pinned upstream camera_K.float() cast, no geometric allowance."""
 K=finite(capture_K,(3,3));T=finite(capture_camera,(4,4));saved=np.asarray(saved_K);camera=np.asarray(saved_camera);expected=K.astype(np.float32)
 passed=bool(saved.shape==(3,3) and saved.dtype==np.float32 and np.isfinite(saved).all() and np.array_equal(saved,expected) and camera.shape==(4,4) and camera.dtype==np.float64 and np.isfinite(camera).all() and np.array_equal(camera,T))
 return dict(source_capture_K64=K,expected_generator_K32=expected,saved_proposal_K=saved,source_capture_cameraWorld64=T,saved_proposal_cameraWorld64=camera,capture_camera_and_K32_cast_pass=np.asarray(passed))
