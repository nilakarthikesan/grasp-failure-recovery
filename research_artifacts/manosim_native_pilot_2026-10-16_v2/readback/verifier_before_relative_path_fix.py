"""Independent saved-only HUG native-target replay: stdlib + NumPy only.
Never imports producer/source-coordinate helpers, Torch, MANO or an engine.
Runs only on a parent-authorized completed/failing saved generation report.
"""
from pathlib import Path
import argparse,hashlib,io,json,math,pickle,subprocess,time
import numpy as np
PARENTS=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
GATES=dict(proper_camera_and_state=1e-12,world_reexpression_m=1e-12,source_anchor_m=5e-6,source_rotation_component=1e-6)

def require(x,msg):
 if not x:raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while data:=f.read(1024*1024):h.update(data)
 return h.hexdigest()
def digest(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def arr(a,shape,dtype):
 a=np.asarray(a);require(a.shape==shape and a.dtype==np.dtype(dtype) and np.isfinite(a).all(),'Literal finite array contract differs '+str(shape));return a

def mul(a,b):
 aw,ax,ay,az=a;bw,bx,by,bz=b
 return np.array([aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw])
def quat_matrix(q):
 q=np.asarray(q,dtype=float);n=float(sum(float(x)*float(x) for x in q));require(abs(n-1)<=1e-12,'State/rest quaternion not proper');w,x,y,z=q;s=2/n
 return np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],[s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],[s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])
def exp_matrix(aa):
 a=np.asarray(aa,dtype=float);theta=math.sqrt(math.fsum(float(v)*float(v) for v in a));x,y,z=a
 K=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
 if theta==0:return np.eye(3)
 return np.eye(3)+(math.sin(theta)/theta)*K+((1-math.cos(theta))/(theta*theta))*mat3(K,K)
def mat3(A,B):return np.array([[math.fsum(float(A[i,k])*float(B[k,j]) for k in range(3)) for j in range(3)] for i in range(3)])
def dot3(R,v):return np.array([math.fsum(float(R[i,j])*float(v[j]) for j in range(3)) for i in range(3)])
def world(R,points,t):return np.array([dot3(R,v)+t for v in points])
def camera(c):
 c=arr(c,(4,4),'float64');require(np.array_equal(c[3],[0,0,0,1]),'Nonhomogeneous camera');r=c[:3,:3]
 require(np.max(np.abs(mat3(r.T,r)-np.eye(3)))<=1e-12 and abs(np.linalg.det(r)-1)<=1e-12,'Nonrigid camera');return r,c[:3,3]
def neutral_joints(buffers):
 template=arr(buffers['template'],(778,3),'float32').astype(float);directions=arr(buffers['shapedirs'],(778,3,10),'float32').astype(float);beta=arr(buffers['shape'],(1,10),'float32')[0].astype(float);reg=arr(buffers['joint_regressor'],(16,778),'float32').astype(float)
 shaped=template+np.einsum('vci,i->vc',directions,beta,optimize=False)
 return np.einsum('jv,vc->jc',reg,shaped,optimize=False)
def source_fk(aa,translation,camera_world,joints):
 R,t=camera(camera_world);local=np.array([exp_matrix(a) for a in aa.reshape(16,3)]);rot=[];pos=[]
 for i,p in enumerate(PARENTS):
  if p<0:rot.append(mat3(R,local[i]));pos.append(dot3(R,translation.astype(float))+t)
  else:rot.append(mat3(rot[p],local[i]));pos.append(pos[p]+dot3(rot[p],joints[i]-joints[p]))
 return np.array(pos),np.array(rot),local

class RestrictedObservation(pickle.Unpickler):
 def find_class(self,m,n):
  allow={('numpy','ndarray'),('numpy','dtype'),('_codecs','encode')}
  allow|={(x,y) for x in ('numpy.core.multiarray','numpy._core.multiarray') for y in ('_reconstruct','scalar')}
  allow|={(x,'_frombuffer') for x in ('numpy.core.numeric','numpy._core.numeric')}
  require((m,n) in allow,'Unsupported observation pickle global')
  return super().find_class(m,n)
def observation(path):
 obj=RestrictedObservation(io.BytesIO(Path(path).read_bytes())).load();require(type(obj) is dict and obj.get('grasp') is None,'Expected source unlabeled observation');return obj

def row_check(row,arrays,buffers,source_observation):
 a=arrays;aa=arr(a['postAA32'],(48,),'float32');t=arr(a['translation32'],(3,),'float32');pred=arr(a['prediction99'],(99,),'float32');q=arr(a['native_qpos'],(67,),'float64');rest=arr(a['rest_quaternions'],(16,4),'float64');shape=arr(a['shape'],(1,10),'float32');R,shift=camera(a['cameraWorld64'])
 require(t.tobytes()==pred[:3].tobytes(),'Translation differs from original HUG prediction')
 require(shape.tobytes()==buffers['shape'].tobytes() and rest.tobytes()==buffers['rest_quaternions'].tobytes(),'Qualified shape/rest identity differs')
 require(np.array_equal(np.asarray(source_observation['T_world_camera'],dtype=np.float64),a['cameraWorld64']),'Saved camera differs from literal observation')
 require(np.array_equal(np.asarray(source_observation['camera']['K'],dtype=np.float32),a['K']),'K differs from official eval conversion')
 require(np.array_equal(np.asarray(source_observation['condition_point'],dtype=np.float32)[:2],a['query_point'][:2]),'Query point differs from stored source')
 arr(a['vertices_camera32'],(778,3),'float32');arr(a['vertices_world'],(778,3),'float64');arr(a['source_anchors_world'],(16,3),'float64');arr(a['source_rotations_world'],(16,3,3),'float64')
 got=world(R,a['vertices_camera32'].astype(float),shift);worlderr=float(np.max(np.abs(got-a['vertices_world'])));require(worlderr<=GATES['world_reexpression_m'],'Camera/world vertex algebra failed')
 expected_root=dot3(R,t.astype(float))+shift;rooterr=float(np.max(np.abs(expected_root-q[:3])));require(rooterr<=1e-12,'Native root translation changed')
 joints=neutral_joints(buffers);anchors,rotations,local=source_fk(aa,t,a['cameraWorld64'],joints)
 targetlocal=np.array([quat_matrix(mul(rest[i],q[3+4*i:7+4*i])) for i in range(16)])
 target_expected=local.copy();target_expected[0]=mat3(R,local[0]);qerr=float(np.max(np.abs(targetlocal-target_expected)));require(qerr<=1e-12,'67-state body-local/rest rotation differs from source AA32')
 # Independent hierarchy algebra from saved source-neutral buffers. This is
 # not a call to either a MANO model or simulator FK and not a skin replay.
 anchorerr=float(np.max(np.abs(anchors-a['source_anchors_world'])));rerr=float(np.max(np.abs(rotations-a['source_rotations_world'])));require(anchorerr<=5e-6 and rerr<=1e-6,'Source anchor/body-R transfer outside original numerical gates')
 md=row['mapping'];require(md['qpos_sha256']==digest(q) and md['shape_digest']==digest(shape) and md['rest_digest']==digest(rest),'Mapping receipt array digest differs')
 sr=md['source_receipt'];require(sr['original_AA32_sha256']==digest(aa) and sr['original_translation32_sha256']==digest(t) and sr['camera64_sha256']==digest(a['cameraWorld64']),'Source receipt operands differ')
 require(sr['original_input_byte_guard']['original_axisangle_bytes_exact'] and sr['original_input_byte_guard']['original_translation_bytes_exact'],'Producer original-byte guard absent')
 require(all(md[k] is False for k in ('accepted_mapping','collection_or_training_allowed','physical_model_qualified')) and row['grasp_label_available'] is False,'Unexpected task/admission claim')
 return dict(id=row['id'],world_vertex_component_error_m=worlderr,root_translation_error_m=rooterr,target_rotation_component_error=qerr,source_anchor_component_error_m=anchorerr,source_absolute_rotation_component_error=rerr,source_arrays_finite=True,task_label=None)

def audit(root,protocol_path,report_path,freeze,expected_report_sha,out):
 start=time.monotonic();root=Path(root);protocol_path=Path(protocol_path);report_path=Path(report_path)
 def git(*args):return subprocess.check_output(['git','-C',str(root),*args])
 require(git('rev-parse','HEAD').decode().strip()==freeze,'Frozen HEAD changed');require(sha(report_path)==expected_report_sha,'Report differs')
 p=read(protocol_path);r=read(report_path);psha=sha(protocol_path);require(r['freeze_commit']==freeze and r['protocol_sha256']==psha,'Report protocol/freeze differs')
 def inputs():
  for name,b in p['bindings'].items():require(Path(name).is_file() and Path(name).stat().st_size==b['bytes'] and sha(name)==b['sha256'],'Bound bytes changed: '+name)
 inputs();tracked=set(git('ls-tree','-r','--name-only',freeze).decode().splitlines());selected=[]
 for path in [protocol_path]+[Path(x) for x in p['bindings'] if Path(x).is_relative_to(root)]:
  rel=path.relative_to(root).as_posix()
  if rel in tracked:require(hashlib.sha256(git('show',freeze+':'+rel)).hexdigest()==sha(path),'Selected frozen Git blob differs');selected.append(rel)
 with np.load(p['source_buffers'],allow_pickle=False) as z:buffers={k:z[k].copy() for k in z.files}
 plan=[(obs,seed) for obs in p['observations'] for seed in p['seeds']];results=[];expected_files=set();tensors={}
 for i,row in enumerate(r['rows']):
  require((row['observation'],row['seed'])==plan[i],'Saved row is outside prospective slot order');path=Path(row['path']);require(path.parent==report_path.parent and sha(path)==row['sha256'],'Output row bytes/path differ');expected_files.add(str(path));input_path=Path(p['asset_root'])/row['observation'];require(sha(input_path)==row['input_sha256'],'Observation bytes differ')
  require(row['id']==f'{i//len(p["seeds"]):02d}_seed{row["seed"]}' and row['object_path']==str(Path(row['observation']).parent),'Object/slot identity differs')
  tensors.setdefault(row['observation'],set()).add(row['input_tensor_hash'])
  with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  results.append(row_check(row,a,buffers,observation(input_path)))
 require(all(len(v)==1 for v in tensors.values()),'Input tensor hash varies with grasp seed')
 require(all(r[k]==0 for k in ('physics_steps','retarget_calls','training_calls')) and r['training_allowed'] is False and r['closure_verified_after'] is True,'Unexpected work/admission')
 inventory=r['partial_artifact_inventory']
 for name,b in inventory.items():require(Path(name).parent==report_path.parent and sha(name)==b['sha256'] and Path(name).stat().st_size==b['bytes'],'Creation inventory differs')
 if r['status']=='COMPLETE':
  require(len(results)==120 and r['counts']==dict(primary=120,repeat=1,source_geometry=120) and not r['errors'],'Incomplete work declared complete')
  repeat=report_path.parent/'first_seed_repeat.npz';require(str(repeat) in inventory,'Repeat arrays missing from original inventory')
  with np.load(repeat,allow_pickle=False) as z:
   require(set(z.files)=={'original','repeated'},'Repeat operand inventory differs');original=arr(z['original'],(99,),'float32');repeated=arr(z['repeated'],(99,),'float32');require(original.tobytes()==repeated.tobytes(),'Actual repeat predictions differ')
  with np.load(r['rows'][0]['path'],allow_pickle=False) as z:require(original.tobytes()==z['prediction99'].tobytes(),'Repeat does not reference first primary proposal')
  expected_files.add(str(repeat));require(set(inventory)==expected_files and {str(x) for x in report_path.parent.glob('*.npz')}==expected_files,'Complete output has missing/unindexed/unexpected artifacts')
 inputs();require(sha(report_path)==expected_report_sha and sha(protocol_path)==psha and git('rev-parse','HEAD').decode().strip()==freeze,'Final immutable closure differs')
 summary={key:max((row[key] for row in results),default=None) for key in ('world_vertex_component_error_m','root_translation_error_m','target_rotation_component_error','source_anchor_component_error_m','source_absolute_rotation_component_error')}
 receipt=dict(schema_version='manosim_native_hug_generation_independent_saved_readback_v2',status='VERIFIED_SAVED_NATIVE_TARGET_ALGEBRA' if r['status']=='COMPLETE' else 'VERIFIED_PARTIAL_NATIVE_TARGET_ALGEBRA',freeze_commit=freeze,protocol=dict(path=str(protocol_path),sha256=psha),report=dict(path=str(report_path),sha256=expected_report_sha),input_bindings_verified_before_after=len(p['bindings']),selected_git_blobs=selected,planned_slots=len(plan),saved_rows=len(results),producer_status=r['status'],producer_counts=r['counts'],errors=r['errors'],gates=GATES,summary=summary,rows=results,limitations=['Pure saved-state coordinate/hierarchy algebra only; no new simulator readback or MANO skin output is generated. The6D->AA32 learned-output conversion itself is bound to frozen source rather than rerun with Torch.','Input tensor digest consistency is checked across seeds; RGB/depth preprocessing is not rerun.','Actual post-AA capture and original-byte-recovery are bound producer assertions unless separately persisted operands permit independent recovery.','No physical success/failure labels, task/controller/skin contact qualification or collection/training admission.'],verifier=dict(source=Path(__file__).read_text(),sha256=sha(__file__),stdlib_numpy_only=True,engine_MANO_inference_geometry_calls=0),elapsed_s=time.monotonic()-start)
 Path(out).write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(output=str(out),sha256=sha(out),summary=summary,saved_rows=len(results))))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--report',required=True);ap.add_argument('--freeze',required=True);ap.add_argument('--report-sha256',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();audit(a.root,a.protocol,a.report,a.freeze,a.report_sha256,a.output)
