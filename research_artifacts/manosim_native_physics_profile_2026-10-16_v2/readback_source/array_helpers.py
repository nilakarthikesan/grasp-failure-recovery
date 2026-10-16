"""Independent saved-array primitives copied from our prior verifier, no producer imports."""

from pathlib import Path

import hashlib,json,subprocess,math

from xml.etree import ElementTree as ET

import numpy as np

PARENTS=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)

FLAGS=('accepted_mapping','native_task_contact_qualified','full_skin_parity_claimed','new_collection_or_training_allowed')

LIMITS=dict(static_anchor_m=5e-6,static_rotation_component=1e-6,transmission_length=1e-10,zero_servo_force=1e-8,tracking_wrist_m=.002,tracking_bone_rad=float(np.deg2rad(2)),gravity_balance_fraction=.01,convergence_wrist_m=.001,convergence_bone_rad=float(np.deg2rad(1)),contact_gap_m=1e-9,contact_normal=1e-10,contact_J=1e-10,virtual_power_absolute_W=1e-10,virtual_power_relative=1e-10,contact_translation_force_N=1e-8,contact_angular_force_Nm=1e-10,cone_primal_relative=1e-8,cone_dual_relative=1e-7,cone_complementarity_relative=1e-7,positive_contact_force_N=1e-8,maximum_penetration_m=.005,clock_s=1e-12,raw_ball_norm=1e-8)

def require(x,message):
 if not bool(x):raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
 def pairs(rows):
  out={}
  for k,v in rows:
   require(k not in out,'Duplicate JSON key');out[k]=v
  return out
 def reject(value):raise ValueError('Nonfinite JSON constant '+value)
 def check(x):
  if isinstance(x,float):require(np.isfinite(x),'Nonfinite JSON exponent')
  elif isinstance(x,dict):
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
 out=json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=reject);check(out);return out

def bind(path):
 p=Path(path).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)

def rowcheck(row):
 p=Path(row['path']);require(p.is_absolute() and not p.is_symlink(),'Absolute nonsymlink artifact required');require(bind(p)=={k:row[k] for k in ('path','sha256','bytes')},'Artifact hash/bytes changed: '+str(p))

def npz(path):
 with np.load(path,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
 for k,v in a.items():require(v.dtype.kind in 'biufSU','Object/unsupported sealed operand '+k);require(v.dtype.kind in 'SU' or np.isfinite(v).all(),'Nonfinite sealed operand '+k)
 return a

def near(a,b,tol,message):
 a=np.asarray(a);b=np.asarray(b);require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),message+' shape/finiteness');d=float(np.max(np.abs(a-b))) if a.size else 0.;require(d<=tol,message+': '+str(d));return d

def exact(a,b,message):
 a=np.asarray(a);b=np.asarray(b);require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),message)

def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])

def closed(value):
 require(value.get('physical_outcome') is None,'Physical outcome must remain null')
 for key in FLAGS:require(value.get(key) is False,'Admission flag changed: '+key)

def closure(p,protocol,root,freeze):
 require(git(root,'rev-parse','HEAD').decode().strip()==freeze,'Exact HEAD hold required')
 for path,digest in p['active_bindings'].items():require(sha(path)==digest,'Active binding changed '+path)
 for relative,digest in p['selected_git_sources'].items():require(hashlib.sha256(git(root,'show',freeze+':'+relative)).hexdigest()==digest,'Frozen source blob changed '+relative)
 require(git(root,'show',freeze+':'+str(protocol.relative_to(root)))==protocol.read_bytes(),'Protocol Git blob differs')
 return dict(active_bindings=len(p['active_bindings']),selected_git_sources=len(p['selected_git_sources']),protocol_blob_verified=True,exact_HEAD=freeze)

def principal(q):
 q=np.asarray(q,float);require(q.shape[-1]==67,'Named67 qpos required');balls=q[...,3:].reshape(q.shape[:-1]+(16,4));norm=np.sqrt(np.einsum('...ij,...ij->...i',balls,balls,optimize=False));require(np.max(np.abs(norm-1))<=1e-8,'Raw ball norm escaped');balls=balls/norm[...,None];s=np.sqrt(np.einsum('...ij,...ij->...i',balls[...,1:],balls[...,1:],optimize=False));nz=s>=1e-15;axis=np.zeros_like(balls[...,1:]);np.divide(balls[...,1:],s[...,None],out=axis,where=nz[...,None]);angle=2*np.arctan2(np.where(nz,s,0),balls[...,0]);angle=np.where(angle>np.pi,angle-2*np.pi,angle);require(np.all(np.pi-np.abs(angle)>1e-6),'Principal command chart crossed exclusion');return axis*angle[...,None]

def controls(q,contract):
 q=np.asarray(q,float);rv=principal(q);gear=np.asarray(contract['gears']);ids=np.asarray(contract['actuator_ids']);require(gear.shape==(51,6) and set(ids)==set(range(51)),'Named actuator contract malformed');canonical=np.empty(q.shape[:-1]+(51,));canonical[...,:3]=q[...,:3]*gear[:3,0];canonical[...,3:]=np.sum(np.repeat(rv,3,axis=-2)*gear[3:,:3],axis=-1);out=np.empty_like(canonical);out[...,ids]=canonical;return out

def angles(actual,target):
 relative=np.einsum('...bji,...bjk->...bik',np.asarray(target),np.asarray(actual),optimize=False);s=np.linalg.norm(np.stack((relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],relative[...,1,0]-relative[...,0,1]),axis=-1),axis=-1)/2;c=(np.trace(relative,axis1=-2,axis2=-1)-1)/2;return np.arctan2(s,np.clip(c,-1,1))

def qmul(a,b):return np.r_[a[0]*b[0]-a[1:]@b[1:],a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]

def ramp(t,T):
 x=np.clip(np.asarray(t)/T,0,1);return x*x*x*(10+x*(-15+6*x))

def scalar_equal(actual,expected,label):
 require(set(actual)==set(expected),label+' keys differ')
 for k,v in expected.items():
  if isinstance(v,bool):require(actual[k] is v,label+' decision differs '+k)
  elif isinstance(v,list):near(actual[k],v,1e-12,label+' '+k)
  elif isinstance(v,(float,int)):require(abs(actual[k]-v)<=1e-12*max(1,abs(v)),label+' scalar differs '+k)
  else:require(actual[k]==v,label+' differs '+k)

def rotation(q):
 q=np.asarray(q,dtype=float);require(q.shape==(4,) and np.isfinite(q).all() and abs(float(q@q)-1)<=1e-12,'Compiled quaternion invalid');w,x,y,z=q;k=2/float(q@q);return np.array([[1-k*(y*y+z*z),k*(x*y-z*w),k*(x*z+y*w)],[k*(x*y+z*w),1-k*(x*x+z*z),k*(y*z-x*w)],[k*(x*z-y*w),k*(y*z+x*w),1-k*(x*x+y*y)]])

def inertia_arithmetic(full,principal,q):
 # Independent arithmetic only, no eigensolver/compiler invocation. Both
 # termination tests of pinned user_util.cc611-700 are retained prospectively.
 import math
 f=np.asarray(full,dtype=float);eig=np.asarray(principal,dtype=float);require(f.shape==(6,) and eig.shape==(3,) and np.isfinite(f).all() and np.isfinite(eig).all() and np.all(eig>0),'Finite positive inertia operands required');I=np.array([[f[0],f[3],f[4]],[f[3],f[1],f[5]],[f[4],f[5],f[2]]]);R=rotation(q)
 D=np.array([[math.fsum(float(R[k,i])*float(I[k,l])*float(R[l,j]) for k in range(3) for l in range(3)) for j in range(3)] for i in range(3)]);diag=np.diag(D);off=D-np.diag(diag);unit=np.finfo(float).eps/2;g32=32*unit/(1-32*unit);g128=128*unit/(1-128*unit);low=np.nextafter((1-1e-12)/(1+g32),0.);sq=np.nextafter(low*low,0.);t2=np.nextafter(np.nextafter(1/sq,np.inf)-1,np.inf);t=np.nextafter(np.sqrt(t2),np.inf);coefficient=np.nextafter(t/(1-t2)*(1+g32),np.inf);scale=float(np.max(np.sum(np.abs(I),axis=1)));arithmetic=g128*scale;span=float(diag.max()-diag.min());bound=max(1e-12,float(coefficient)*span)+arithmetic;diag_error=float(np.max(np.abs(diag-eig)));off_error=float(np.max(np.abs(off)));reconstructed=np.array([[math.fsum(float(R[i,k])*float(eig[k])*float(R[j,k]) for k in range(3)) for j in range(3)] for i in range(3)]);require(np.isfinite([arithmetic,span,bound,diag_error,off_error]).all() and scale>0,'Invalid inertia arithmetic')
 return dict(source_in_compiled_frame_kg_m2=D.tolist(),diagonal_error_kg_m2=diag_error,diagonal_arithmetic_bound_kg_m2=arithmetic,maximum_offdiagonal_kg_m2=off_error,jacobi_offdiagonal_bound_kg_m2=bound,diagonal_span_kg_m2=span,cosine_stop_coefficient=float(coefficient),gamma32=g32,gamma128=g128,maximum_reconstructed_tensor_error_kg_m2=float(np.max(np.abs(reconstructed-I))),output_eigenframe_postconditions_passed=bool(diag_error<=arithmetic and off_error<=bound))

def compiled_snapshot_check(snapshot_path,xml_path,declared_metadata=None,context=None):
 a=npz(snapshot_path);root=ET.parse(xml_path).getroot();bodies={b.get('name'):b for b in root.findall('./worldbody//body')};names=[f'mano_body_{i}' for i in range(16)];require(a['source_body_names'].tolist()==names,'Exact named source-body order required');ids=np.asarray(a['named_body_ids']);require(ids.shape==(16,) and len(set(map(int,ids)))==16 and np.all(ids>0),'Named compiled body identities');source=[];rows=[]
 if context is not None:
  pp=context['protocol'];ii=context['index'];spec=pp['cases'][str(ii)];literal=npz(spec['literal_buffers']['path']);require(a['validation_performed'].shape==() and a['validation_performed'].item() is False,'Snapshot must precede mapping validation');near(a['model_dimensions'],np.array([67,51,51,19,17,17]),0,'Exact early compiled dimensions');require(str(a['fixture_xml_sha256'].item())==sha(xml_path),'Early XML identity');require(str(a['authored_capsule_xml_sha256'].item())==sha(spec['capsule_xml']),'Early authored XML identity');require(str(a['asset_closure_sha256'].item())==spec['asset_closure_sha256'],'Early asset closure identity');require(str(a['mapping_controller_source_sha256'].item())==pp['active_bindings'][str(context['root']/'src/grasp_failure_prediction/evaluation/manosim_capsule.py')],'Early controller-source identity');exact(a['rest_quaternions'],literal['rest_quaternions'],'Early literal rest quaternions');jn=['wrist_t'+x for x in 'xyz']+['wrist_rot']+[f'mano_joint_{j}' for j in range(1,16)];an=['wrist_motor_t'+x for x in 'xyz']+['wrist_motor_r'+x for x in 'xyz']+[f'mano_motor_{j}_{x}' for j in range(1,16) for x in 'xyz'];require(a['source_joint_names'].tolist()==jn and a['source_actuator_names'].tolist()==an,'Early named state/controller order');require(set(map(int,a['named_joint_ids']))==set(range(19)) and set(map(int,a['named_actuator_ids']))==set(range(51)),'Early named-ID permutation')
 for i,name in enumerate(names):
  body=bodies[name];inertial=body.find('inertial');require(inertial is not None,'Explicit source inertia required');full=np.asarray([float(x) for x in inertial.get('fullinertia').split()]);require(full.shape==(6,),'Six source fullinertia values');source.append(full);bid=int(ids[i]);require(float(a['body_mass'][bid])==float(inertial.get('mass')),'Source mass changed');near(a['body_ipos'][bid],[float(x) for x in inertial.get('pos','0 0 0').split()],0,'Source inertial origin');near(a['body_pos'][bid],[float(x) for x in body.get('pos','0 0 0').split()],0,'Source body origin');source_q=np.asarray([float(x) for x in body.get('quat','1 0 0 0').split()]);near(rotation(a['body_quat'][bid]),rotation(source_q),128*np.finfo(float).eps,'Source body rest rotation');require(a['body_parentid'][bid]==(0 if i==0 else ids[PARENTS[i]]),'Source anatomical tree changed');row=inertia_arithmetic(full,a['body_inertia'][bid],a['body_iquat'][bid]);row.update(body_name=name,body_id=bid);rows.append(row)
  if declared_metadata is not None:
   m=declared_metadata[i];require(m['body_name']==name and m['body_id']==bid and m['rule']=='mujoco_3_4_0_mjuu_eig3_postcondition_v1' and m['primary_source_sha256']=='f9d5ef77317707039f12658e620ff3393f1eb8a0e7087111e4740207f2ee2522','Inertia compiler-source identity changed');near(m['source_fullinertia_kg_m2'],full,0,'Metadata source inertia');near(m['compiled_principal_kg_m2'],a['body_inertia'][bid],0,'Metadata principal inertia');near(m['compiled_iquat'],a['body_iquat'][bid],0,'Metadata compiled inertia quaternion');near(m['source_in_compiled_frame_kg_m2'],row['source_in_compiled_frame_kg_m2'],row['diagonal_arithmetic_bound_kg_m2'],'Independent inertia-frame arithmetic');require(m['actual_compiler_iteration_count'] is None and m['compiler_maximum_iterations']==500 and m['iteration_limit_is_not_assumed_converged'] is True and m['source_inertia_changed'] is False,'No inferred compiler convergence/source mutation');require(m['output_eigenframe_postconditions_passed']==row['output_eigenframe_postconditions_passed'],'Independent inertia decision differs')
   for key in ('diagonal_error_kg_m2','maximum_offdiagonal_kg_m2','maximum_reconstructed_tensor_error_kg_m2','diagonal_span_kg_m2'):require(abs(m[key]-row[key])<=row['diagonal_arithmetic_bound_kg_m2']*4,'Independent inertia residual differs '+key)
   for key in ('diagonal_arithmetic_bound_kg_m2','jacobi_offdiagonal_bound_kg_m2','cosine_stop_coefficient','gamma32','gamma128'):require(abs(m[key]-row[key])<=max(abs(row[key])*1e-13,np.finfo(float).tiny),'Independent inertia bound differs '+key)
 near(a['source_fullinertia6'],np.asarray(source),0,'Snapshot source fullinertia differs from literal XML')
 if 'source_body_mass' in a:near(a['source_body_mass'],[float(bodies[name].find('inertial').get('mass')) for name in names],0,'Snapshot source masses differ from XML')
 return dict(snapshot=bind(snapshot_path),body_count=16,all_output_postconditions_passed=all(r['output_eigenframe_postconditions_passed'] for r in rows),bodies=rows,actual_compiler_iteration_count=None,scope='Source-equivalent approximate compiler eigenframe, not physical parameter calibration')
