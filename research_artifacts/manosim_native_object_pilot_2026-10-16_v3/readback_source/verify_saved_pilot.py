"""Independent saved-only pilot readback: no producer imports or model queries.

The reader uses hashes, XML declarations and already saved arrays. Quaternion
and point-Jacobian algebra does not construct a model or evaluate joint anchors.
Per-step pair-force sums remain source-bound; stored first-positive witnesses
receive an independent 57D virtual-work check. No positive task admission.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, importlib.abc, io, json, math, pickle, subprocess, sys
from xml.etree import ElementTree as ET
import numpy as np

class NoScientificRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'mujoco','torch','manotorch','manopth','smplx','trimesh','grasp_failure_prediction'}:
            raise RuntimeError('Saved-only reader forbids producer/model/geometry runtime: '+name)
sys.meta_path.insert(0, NoScientificRuntime())

FLAGS=('full_skin_parity_claimed','hardware_calibrated','scientific_training_allowed','native_task_qualified')
PARENTS=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
PRIORITY=('approach_collision','no_contact','failed_acquisition','drop','slip_during_hold')

def require(test, message):
    if not bool(test): raise ValueError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    def unique(pairs):
        out={}
        for k,v in pairs:
            require(k not in out,'Duplicate JSON key'); out[k]=v
        return out
    value=json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
    def finite(x):
        if isinstance(x,float): require(math.isfinite(x),'Nonfinite JSON')
        elif isinstance(x,dict):
            for v in x.values(): finite(v)
        elif isinstance(x,list):
            for v in x: finite(v)
    finite(value); return value
def arrays(path):
    with np.load(path,allow_pickle=False) as f:
        out={k:f[k].copy() for k in f.files}
    for k,a in out.items():
        if a.dtype.kind in 'fci': require(np.isfinite(a).all(),'Nonfinite saved array '+str(path)+':'+k)
    return out
def binding(path):
    p=Path(path).resolve(); return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def verify_row(row):
    p=Path(row['path']); require(not p.is_symlink() and p.is_file(),'Missing/aliased artifact')
    require(binding(p)=={k:row[k] for k in ('path','sha256','bytes')},'Creation-time artifact changed: '+str(p))
def close(a,b,atol=1e-12,message='Saved array mismatch'):
    a=np.asarray(a); b=np.asarray(b)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),message+' layout/nonfinite')
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    require(error<=atol,message+': '+str(error)); return error
def canonical(value): return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
class ObservationReader(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','ndarray'),('numpy','dtype'),('_codecs','encode')}|{(m,n) for m in ('numpy.core.multiarray','numpy._core.multiarray') for n in ('_reconstruct','scalar')}|{(m,'_frombuffer') for m in ('numpy.core.numeric','numpy._core.numeric')}
        require((module,name) in allowed,'Unsupported observation global')
        return super().find_class(module,name)
def observation(path):
    d=ObservationReader(io.BytesIO(Path(path).read_bytes())).load();require(isinstance(d,dict) and d.get('grasp') is None,'Unlabeled capture required');return d
def R(q):
    q=np.asarray(q,dtype=float);require(q.shape[-1]==4 and np.isfinite(q).all(),'Quaternion layout')
    require(np.max(np.abs(np.sum(q*q,axis=-1)-1))<=1e-8,'Raw quaternion not unit')
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1).reshape(q.shape[:-1]+(3,3))
def product(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return np.concatenate(((a[...,0]*b[...,0]-np.sum(a[...,1:]*b[...,1:],axis=-1))[...,None],a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])),axis=-1)
def angle(a,b):
    c=np.einsum('...ji,...jk->...ik',a,b,optimize=False)
    sine=np.linalg.norm(np.stack((c[...,2,1]-c[...,1,2],c[...,0,2]-c[...,2,0],c[...,1,0]-c[...,0,1]),axis=-1),axis=-1)/2
    return np.arctan2(sine,np.clip((np.trace(c,axis1=-2,axis2=-1)-1)/2,-1,1))
def bone_rotations(q,rest):
    local=R(product(np.asarray(rest),np.asarray(q)[3:67].reshape(16,4))); out=[]
    for i in range(16):out.append(local[i] if i==0 else out[PARENTS[i]]@local[i])
    return np.asarray(out)
def principal(q):
    q=np.asarray(q,dtype=float);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    v=np.linalg.norm(q[...,1:],axis=-1);theta=2*np.arctan2(v,q[...,0]);theta=np.where(theta>np.pi,theta-2*np.pi,theta)
    axis=np.divide(q[...,1:],v[...,None],out=np.zeros_like(q[...,1:]),where=v[...,None]>=1e-15)
    return axis*theta[...,None]
def phase(t):return 'reset' if t<.25 else 'approach' if t<1. else 'close_fingers' if t<1.5 else 'lift' if t<2.5 else 'hold'
def endpoint(q,rest,ipos):
    q=np.asarray(q);return q[...,:3],R(product(np.asarray(rest)[0],q[...,3:7])),q[...,67:70]+np.einsum('...ij,j->...i',R(q[...,70:74]),ipos,optimize=False)

def independent_label(st,ft,dt):
    """Index-based replay of frozen labels; returns earliest event only."""
    M=round(5.5/dt); require(len(st['qpos'])==M+1 and len(ft['force_time_s'])==M,'Complete label denominator')
    close_index=round(1/dt); lift=round(1.5/dt); hold=round(2.5/dt)
    obj=st['object_com_world_m'];w=st['wrist_position_world_m'];rot=st['wrist_rotation_world']
    z0=float(obj[0,2]);ho=ft['hand_object_normal_force_N'];ht=ft['hand_table_normal_force_N'];ot=ft['object_table_normal_force_N']
    require(np.all(ho>=0)&np.all(ht>=0)&np.all(ot>=0),'Signed pair normal force')
    events=[]
    def add(kind,indices):
        if len(indices): events.append((int(indices[0]),PRIORITY.index(kind),kind))
    ticks=np.arange(M);contact=ho>.001;support=ot>.001
    add('approach_collision',np.flatnonzero(((ticks<close_index)&contact)|(ht>.001)))
    if not np.any(contact&(ticks>=close_index)&(ticks<=lift)):events.append((lift,1,'no_contact'))
    if obj[hold,2]<z0+.10 or support[hold]:events.append((hold,2,'failed_acquisition'))
    add('drop',np.flatnonzero((ticks>hold)&support))
    raised=np.flatnonzero(obj[:,2]>=z0+.10)
    if len(raised):add('drop',np.flatnonzero((np.arange(M+1)>=raised[0])&(obj[:,2]<z0+.08)))
    relative=np.einsum('nji,nj->ni',rot,obj-w,optimize=False)
    shifts=np.linalg.norm(relative[hold:]-relative[hold],axis=1);slips=np.flatnonzero(shifts>.02)
    if len(slips):events.append((hold+int(slips[0]),4,'slip_during_hold'))
    events.sort();event=events[0] if events else None
    return dict(success=not events,failure_type=None if event is None else event[2],first_failure_time_s=None if event is None else event[0]*dt)

def check_journal(path,rows):
    ledger={}
    for line in Path(path).read_text().splitlines():
        r=json.loads(line)
        if r.get('event')=='artifact_relocation':
            old=ledger.pop(r['original_path']);require((old['sha256'],old['bytes'])==(r['sha256'],r['bytes']),'Relocation changed bytes');old.update(path=r['path'],original_publication_path=r['original_path']);ledger[r['path']]=old
        else:require(r['path'] not in ledger,'Journal duplicate publication');ledger[r['path']]=r
    require(ledger=={r['path']:r for r in rows},'Creation journal differs from terminal artifact index')

def static_checks(a,source,compiled,contract,alignment):
    """Recompute saved static predicates without a new source/FK evaluation."""
    require(np.array_equal(a['expected_anchors'],source['source_anchors_world']) and np.array_equal(a['expected_rotations'],source['source_rotations_world']),'Static reference arrays differ from sealed native output')
    aerr=float(np.max(np.abs(a['actual_anchors']-a['expected_anchors'])));rerr=float(np.max(np.abs(a['actual_rotations']-a['expected_rotations'])))
    chart=principal(a['command_knots'][:,3:].reshape(-1,16,4));margin=np.pi-np.linalg.norm(chart,axis=-1)
    chart_ok=bool(np.min(margin)>1e-6 and np.max(np.linalg.norm(np.diff(chart,axis=0),axis=-1),initial=0)<np.pi)
    meta=contract['scene']['compiled_hand'];acts=np.asarray(meta['actuator_ids']);command=np.r_[a['goal'][:3],principal(a['goal'][3:].reshape(16,4)).ravel()];expected=np.empty(51);expected[acts]=command
    close(a['command_ctrl'],expected,2e-14,'Native principal angle-axis transmission command')
    q=a['actual_qpos'];wp,wr,oc=endpoint(q,source['rest_quaternions'],compiled['body_ipos'][17])
    endpoint_ok=bool(np.max(np.abs(wp-a['actual_anchors'][0]))<=1e-12 and np.max(np.abs(wr-a['actual_rotations'][0]))<=1e-12)
    checks=dict(qpos=np.array_equal(q[:67],a['goal']),anchors=aerr<=5e-6,rotations=rerr<=1e-6,transmission=bool(np.max(np.abs(a['actual_actuator_length']-a['command_ctrl']))<=1e-10),zero_servo_force=bool(np.max(np.abs(a['actual_actuator_force']))<=1e-8),endpoint_formula=endpoint_ok,all_control_chart_knots=chart_ok,depth_visual_alignment=alignment)
    return checks,aerr,rerr

def alignment_readback(a):
    if not bool(a['passed_support']):return False
    require(a['pixel_uv'].shape==(32,2) and a['nearest_triangles'].shape==(32,3,3),'Fixed32 alignment witnesses')
    K=a['K'];uv=a['pixel_uv'];z=a['depth_m'];rays=np.c_[(uv[:,0]-K[0,2])/K[0,0],(uv[:,1]-K[1,2])/K[1,1],np.ones(32)]
    close(a['camera_rays'],rays,1e-14,'Saved depth rays');close(a['points_camera'],z[:,None]*rays,1e-14,'Depth backprojection')
    T=a['cameraWorld64'];world=np.einsum('ij,nj->ni',T[:3,:3],a['points_camera'],optimize=False)+T[:3,3]
    close(a['points_world'],world,1e-14,'Camera/world alignment transform')
    allowance=.001*np.linalg.norm(rays,axis=1)+.5*z*np.sqrt(K[0,0]**-2+K[1,1]**-2)
    close(a['allowances_m'],allowance,1e-14,'Depth bin +half-pixel diagonal allowance')
    close(a['distances_m'],np.linalg.norm(a['points_world']-a['nearest_points'],axis=1),1e-12,'Saved nearest witness distance')
    # Verify each recorded point lies on its recorded triangle. This is NOT a
    # new nearest-triangle search and does not claim independent global minima.
    tri=a['nearest_triangles'];v=tri[:,1]-tri[:,0];w=tri[:,2]-tri[:,0];p=a['nearest_points']-tri[:,0]
    vv=np.sum(v*v,axis=1);ww=np.sum(w*w,axis=1);vw=np.sum(v*w,axis=1);pv=np.sum(p*v,axis=1);pw=np.sum(p*w,axis=1);den=vv*ww-vw*vw
    require(np.all(den>0),'Recorded alignment nearest triangle is degenerate')
    u=(pv*ww-pw*vw)/den;t=(pw*vv-pv*vw)/den
    close(p,u[:,None]*v+t[:,None]*w,1e-10,'Nearest point on recorded triangle plane')
    require(np.all(u>=-1e-8)&np.all(t>=-1e-8)&np.all(u+t<=1+1e-8),'Nearest point outside recorded triangle')
    return bool(np.all(a['distances_m']<=a['allowances_m']))

def camera_cast_contract(source,captured,saved=None):
    K=np.asarray(source['K']);T=np.asarray(source['cameraWorld64']);expected=np.asarray(captured['camera']['K'],dtype=np.float64).astype(np.float32);captureT=np.asarray(captured['T_world_camera'],dtype=np.float64)
    passed=bool(K.dtype==np.float32 and K.shape==(3,3) and np.isfinite(K).all() and np.array_equal(K,expected) and T.dtype==np.float64 and T.shape==(4,4) and np.isfinite(T).all() and np.array_equal(T,captureT))
    if saved is not None:
        for key,value in dict(source_capture_K64=np.asarray(captured['camera']['K'],dtype=np.float64),expected_generator_K32=expected,saved_proposal_K=K,source_capture_cameraWorld64=captureT,saved_proposal_cameraWorld64=T,capture_camera_and_K32_cast_pass=np.asarray(passed)).items():
            a=np.asarray(saved[key]);v=np.asarray(value);require(a.dtype==v.dtype and a.shape==v.shape and a.tobytes()==v.tobytes(),'V2 exact K-cast saved operand differs: '+key)
    return passed

def same_npz_arrays(path_a,path_b,extra_keys=()):
    a=arrays(path_a);b=arrays(path_b);require(set(b)==set(a)|set(extra_keys),'Legacy/updated saved field inventory')
    for k,v in a.items():require(v.dtype==b[k].dtype and v.shape==b[k].shape and v.tobytes()==b[k].tobytes(),'Legacy semantic array changed: '+str(path_a)+':'+k)
    return len(a)

def verify_legacy_v1(protocol,report,raw):
    prior=protocol['prior_v1_refusals']
    for k in ('protocol','report','mechanical_readback','cohort_readback'):verify_row(prior[k])
    p=read(prior['protocol']['path']);r=read(prior['report']['path']);oldraw=Path(p['raw_root'])
    require(prior['original_mapping_refusals']==135 and prior['original_physics_steps']==0 and prior['original_labels']==0 and prior['prior_refusals_preserved'] is True,'Truthful old-refusal lineage')
    require(r['freeze_commit']==prior['freeze_commit']=='959db6c119b67dbf66a6e2ad67f0da36b36d88eb' and r['protocol_sha256']==sha(prior['protocol']['path']),'Original active freeze/protocol')
    require(r['actual_counts']['steps']==0,'Original refusal run must not create physical evidence')
    require(p['schedule']==protocol['schedule'] and p['models']==protocol['models'] and p['observations']==protocol['observations'],'K-only version changed plan/poses/inputs')
    fields=0
    for oid in protocol['observations']:fields+=same_npz_arrays(oldraw/'alignment'/(oid+'.npz'),raw/'alignment'/(oid+'.npz'))
    extra=('source_capture_K64','expected_generator_K32','saved_proposal_K','source_capture_cameraWorld64','saved_proposal_cameraWorld64','capture_camera_and_K32_cast_pass')
    for row in protocol['schedule']:fields+=same_npz_arrays(oldraw/row['attempt_id']/'source_chart_gate.npz',raw/row['attempt_id']/'source_chart_gate.npz',extra)
    for mid in protocol['models']:fields+=same_npz_arrays(oldraw/'models'/mid/'compiled_model.npz',raw/'models'/mid/'compiled_model.npz',('mesh_vertadr','mesh_vertnum','mesh_faceadr','mesh_facenum'))
    return dict(original_protocol=prior['protocol'],original_report=prior['report'],original_mechanical_readback=prior['mechanical_readback'],alignment_artifacts_exact=60,source_pose_artifacts_common_arrays_exact=135,compiled_model_artifacts_exact=45,semantic_arrays_byte_exact=fields,original_refusals_retained=135,zero_transform_or_geometry_change=True)

def source_obj_vertices(path):
    values=[]
    for line in Path(path).read_text().splitlines():
        p=line.split()
        if p and p[0]=='v':values.append(list(map(float,p[1:4])))
    return np.asarray(values,dtype=np.float32).astype(np.float64)

def reexpress_vertices(points,rot,pos):
    """Explicit scalar cross-check avoids this runtime's spurious BLAS flags."""
    result=np.einsum('nj,ij->ni',points,rot,optimize=False)+pos
    scalar=np.asarray([[math.fsum(float(p[j])*float(rot[i,j]) for j in range(3))+float(pos[i]) for i in range(3)] for p in points])
    unit=np.finfo(float).eps/2;gamma8=8*unit/(1-8*unit);bound=np.nextafter(gamma8*(np.einsum('nj,ij->ni',np.abs(points),np.abs(rot),optimize=False)+np.abs(pos)),np.inf)
    require(np.isfinite(result).all() and np.all(np.abs(result-scalar)<=bound),'Saved vertex re-expression differs from independent scalar sum')
    return result,float(np.max(np.abs(result-scalar))),float(bound.max())

def authored_asset_readback(compiled,receipt):
    """Independent kd-tree comparison of SAVED vertex arrays, no scene query."""
    from scipy.spatial import cKDTree
    require(receipt['compiled_asset_vertex_frame_transfer_passed'] is True and receipt['RGBD_correspondence_qualified'] is False and receipt['geometry_or_pose_fitting'] is False,'Scoped authored-asset receipt')
    require(receipt['numerical_reconstruction_limit_m']==5e-6 and receipt['rotation_component_limit']==1e-6,'Original transfer numerical limits')
    world=np.asarray(receipt['object_world_position']);bid=receipt['object_body_id'];close(compiled['body_pos'][bid],world,5e-6,'Compiled original object body position')
    metrics=[]
    for row in receipt['geoms']:
        require(sha(row['source_mesh'])==row['source_mesh_sha256'],'Source OBJ identity');v=source_obj_vertices(row['source_mesh']);mid=row['mesh_id'];gid=row['geom_id'];adr=int(compiled['mesh_vertadr'][mid]);num=int(compiled['mesh_vertnum'][mid])
        require(adr==row['mesh_vertex_address'] and num==row['mesh_vertex_count'] and compiled['geom_dataid'][gid]==mid and compiled['geom_bodyid'][gid]==bid,'Saved mesh identity/address')
        points=compiled['mesh_vert'][adr:adr+num].astype(np.float64);rot=R(compiled['geom_quat'][gid]);restored,arithmetic_error,arithmetic_bound=reexpress_vertices(points,rot,compiled['geom_pos'][gid])
        close(compiled['geom_pos'][gid],compiled['mesh_pos'][mid],5e-6,'Compiled geom/mesh recenter position');close(rot,R(compiled['mesh_quat'][mid]),1e-6,'Compiled geom/mesh recenter rotation')
        d1=cKDTree(restored).query(v,k=1,eps=0,workers=1)[0];d2=cKDTree(v).query(restored,k=1,eps=0,workers=1)[0]
        require(float(max(d1.max(),d2.max()))<=5e-6,'Independent authored vertex frame transfer exceeds5um')
        metrics.append(dict(geom_id=gid,mesh_id=mid,source_vertices=len(v),compiled_vertices=num,source_to_compiled_max_m=float(d1.max()),compiled_to_source_max_m=float(d2.max()),explicit_einsum_vs_math_fsum_error_m=arithmetic_error,arithmetic_gamma8_bound_max_m=arithmetic_bound))
    return metrics

def contact_witness(a,compiled,contract):
    """Point Jacobian from SAVED actual anchors/body frames; no FK/engine."""
    meta=contract['scene']['compiled_hand'];body=list(map(int,meta['body_ids']));joints=list(map(int,meta['joint_ids']));obj=int(contract['scene']['object_body_id'])
    anchors=a['start_hand_anchors'];rot=a['start_hand_rotations'];point=a['contact_position'];objectR=a['start_object_body_rotation'];objectp=a['start_object_body_origin']
    require(anchors.shape==(16,3) and rot.shape==(16,3,3),'Contact anchor witness layout')
    close(np.einsum('nji,njk->nik',rot,rot,optimize=False),np.broadcast_to(np.eye(3),(16,3,3)),1e-8,'Witness proper hand frames')
    close(objectR.T@objectR,np.eye(3),1e-8,'Witness proper object frame')
    def body_jac(b):
        J=np.zeros((3,57))
        if b==0:return J
        if b==obj:
            J[:,:][:,51:54]=np.eye(3)
            J[:,54:57]=np.cross(objectR.T,point-objectp).T
            return J
        require(b in body,'Contact body outside fixed scene')
        i=body.index(b);J[:,:3]=np.eye(3)
        while i>=0:
            J[:,3+3*i:6+3*i]=np.cross(rot[i].T,point-anchors[i]).T;i=PARENTS[i]
        return J
    b0,b1=map(int,a['body_ids']);frame=a['contact_frame'];expected=frame@(body_jac(b1)-body_jac(b0))
    require(int(a['contact_dim'])==3,'Expected explicit elliptic3 contact')
    require(int(a['efc_address'])>=0 and 'efc_J' in a,'Positive force must have active constraint')
    je=close(a['efc_J'],expected,1e-10,'Independent57D point contact Jacobian')
    force=a['contact_force6'];require(force.shape==(6,) and force[0]>.001,'Positive witness force')
    close(force[3:],np.zeros(3),1e-12,'No spin/rolling contact components')
    fe=close(a['efc_force'],force[:3],1e-10,'Elliptic contact force coordinates')
    tau=np.asarray(a['efc_J']).T@a['efc_force'];v=a['start_qvel'];worldf=frame.T@force[:3]
    error=abs(float(tau@v-worldf@((body_jac(b1)-body_jac(b0))@v)))
    require(error<=1e-10*max(1.,float(np.linalg.norm(tau)*np.linalg.norm(v))),'Contact virtual work mismatch')
    return dict(jacobian_max_error=je,force_coordinate_error=fe,virtual_work_error_W=error)

def verify_recording(folder,row,st,ft,status):
    p=read(folder/'attempt.json');meta=p['attempt'];require(meta['attempt_id']==row['attempt_id'],'Recording identity')
    require(p['contract_sha256']==canonical(p['contract'])==meta['contract_sha256'],'Recording contract hash')
    require(meta['provenance_hashes']['native_pose']=='sha256:'+row['proposal']['sha256'],'Native proposal provenance')
    require(meta['provenance_hashes']['capture']==row['capture_sha256'],'Capture provenance')
    require(meta['provenance_hashes']['scenario']==row['scenario_sha256'],'Scenario provenance')
    admission=p['native_admission'];require(admission.get('admitted') is False if 'admitted' in admission else not admission.get('new_collection_or_training_allowed',False),'Native admission unexpectedly open')
    for file,digest in p['artifacts_sha256'].items():require('sha256:'+sha(folder/file)==digest,'Recording payload hash')
    if status=='MAPPING_REFUSAL':
        require(meta['execution_status']=='not_executed' and meta['outcome'] is None and p['recording'] is None and not p['artifacts_sha256'],'Refusal has physical sample/label');return 0
    require(meta['execution_status']==('completed' if status=='COMPLETED' else 'runtime_error'),'Wrong native execution status')
    if status!='COMPLETED':require(meta['outcome'] is None,'Runtime quarantine received physical label')
    if not p['recording']:return 0
    a=arrays(folder/'observables.npz');N=len(a['time_s']);require(p['recording']['samples']==N,'Control sample count')
    close(a['time_s'],np.arange(1,N+1)*.01,1e-9,'Control endpoint clock')
    close(a['action_interval_start_s'],np.arange(N)*.01,1e-9,'Control start clock')
    require(list(a['phase'])==[phase(i*.01) for i in range(N)],'Recorded phase timing')
    if st is not None:
        take=np.rint(np.asarray(a['time_s'])/row['dt_s']).astype(int)
        close(a['hand_qpos'],st['qpos'][take,:67],0,'Recorded raw endpoint qpos')
        close(a['hand_qvel'],st['qvel'][take,:51],0,'Recorded raw endpoint qvel')
    if ft is not None:
        take=np.rint(np.asarray(a['time_s'])/row['dt_s']).astype(int)-1
        close(a['effective_ctrl'],ft['actual_ctrl'][take],0,'Recorded actually applied control')
        close(a['requested_ctrl'],a['effective_ctrl'],0,'Unclipped requested/effective control')
    return N

def check_states(st,ft,row,compiled,rest,complete):
    q=st['qpos'];v=st['qvel'];dt=row['dt_s'];N=len(q)
    require(q.shape==(N,74) and v.shape==(N,57),'Global74/57 state layout')
    close(st['state_time_s'],np.arange(N)*dt,1e-9,'Physics endpoint clock')
    wp,wr,obj=endpoint(q,rest,compiled['body_ipos'][-1])
    close(st['wrist_position_world_m'],wp,0,'Raw endpoint wrist position')
    close(st['wrist_rotation_world'],wr,1e-14,'Raw endpoint wrist rotation')
    close(st['object_com_world_m'],obj,1e-14,'Raw endpoint object COM')
    if ft is not None:
        M=len(ft['force_time_s']);require(M<=N-1,'Force intervals exceed endpoint states')
        close(ft['force_time_s'],np.arange(M)*dt,1e-9,'Independent start force clock')
        close(ft['start_qvel'],v[:M],0,'Saved preceding-interval velocities')
        require(np.all(ft['contact_count']>=0)&np.all(ft['contact_count']<=256),'Contact quota')
        for kind in ('hand_object','hand_table','object_table'):require(np.all(ft[kind+'_normal_force_N']>=0),'Negative aggregate normal force')
        require(np.max(ft['maximum_penetration_m'],initial=0)<=.005,'Completed saved force escaped5mm guard')
        if 'pair_peak_force6' in ft:
            for i,kind in enumerate(('hand_object','hand_table','object_table')):require(np.all(ft['pair_peak_force6'][:,i,0]<=ft[kind+'_normal_force_N']+1e-10),'Pair peak exceeds aggregate')
    if complete:require(N==round(5.5/dt)+1 and len(ft['force_time_s'])==N-1,'Completed physics interval denominator')
    return N-1

def inspect(protocol,report,root):
    raw=Path(protocol['raw_root']);worker=read(report['worker_result']['path']);verify_row(report['worker_result'])
    require(worker['protocol_sha256']==report['protocol_sha256'] and worker['freeze_commit']==report['freeze_commit'],'Parent/child transaction identity')
    require(worker['status']==report['status']=='COMPLETED','This receipt requires complete bounded transaction')
    require(set(worker['forbidden'])=={'MANO_calls','HUG_calls','retarget_calls','predictor_fits','unregistered_engine_calls'} and not any(worker['forbidden'].values()),'Forbidden operation inventory')
    require(all(worker[k] is False and report[k] is False for k in FLAGS),'Closed admission flags changed')
    rows=worker['artifacts'];require(len({r['path'] for r in rows})==len(rows),'Duplicate terminal artifact')
    for row in rows:require(Path(row['path']).resolve().is_relative_to(raw),'Artifact outside declared raw root');verify_row(row)
    require({str(p.resolve()) for p in raw.rglob('*') if p.is_file()}=={r['path'] for r in rows},'Unindexed raw artifacts')
    verify_row(report['creation_time_journal']);check_journal(report['creation_time_journal']['path'],rows)
    schedule=protocol['schedule'];ledger=worker['attempt_schedule'];require(ledger==report['attempt_schedule'] and len(ledger)==135,'135 slot parent/child denominator')
    results={r['attempt_id']:r for r in ledger};require(len(results)==135 and set(results)=={r['attempt_id'] for r in schedule},'Unique fixed slot identities')
    require(Counter(r['cohort'] for r in schedule)=={'primary':120,'matched_1ms':15},'Fixed primary/matched denominator')
    require(worker['all135_static_before_task'] is True,'All static checks completed flag')
    metrics=[];total_steps=0;total_samples=0;witnesses=[];refusals=Counter();label_counts=Counter();alignment={};alignment_metrics=[];capture={};query_count=0
    for oid,spec in protocol['observations'].items():
        a=arrays(raw/'alignment'/(oid+'.npz'));passed=alignment_readback(a);alignment[oid]=passed;capture[oid]=observation(spec['path'])
        reported=worker['measurement']['alignment'][oid];require(reported['passed']==passed,'Independent depth alignment predicate')
        require(reported['checks']['support']==bool(a['passed_support']),'Depth support predicate')
        if bool(a['passed_support']):
            query_count+=32*sum(line.lstrip().startswith('f ') for line in Path(spec['visual_mesh']).read_text().splitlines())
            alignment_metrics.append(dict(observation_id=oid,passed=passed,maximum_distance_m=float(a['distances_m'].max()),maximum_allowance_m=float(a['allowances_m'].max()),outside_allowance_points=int(np.count_nonzero(a['distances_m']>a['allowances_m'])),cameraWorld64=a['cameraWorld64'].tolist()))
    model_arrays={};contracts={};asset_metrics={}
    for mid in protocol['models']:
        folder=raw/'models'/mid;a=arrays(folder/'compiled_model.npz');c=read(folder/'model_contract.json');model_arrays[mid]=a;contracts[mid]=c
        require(c['scene']['global_dimensions']==[74,57,51,20],'Compiled dimensions')
        require(c['scene']['object_joint_id']==19 and c['scene']['object_body_id']==17,'Appended one free object identity')
        require(a['jnt_type'][19]==0 and a['jnt_qposadr'][19]==67 and a['jnt_dofadr'][19]==51,'Compiled object addresses')
        require(a['body_gravcomp'][17]==0 and np.all(a['body_gravcomp'][c['scene']['compiled_hand']['body_ids']]==1),'Gravity compensation routing')
        require(float(a['timestep_s'])==protocol['models'][mid]['dt_s'],'Model dt identity')
        asset_metrics[mid]=authored_asset_readback(a,read(folder/'authored_asset_transfer.json'))
    for row in schedule:
        aid=row['attempt_id'];folder=raw/aid;result=results[aid];status=result['status'];require(status in ('COMPLETED','MAPPING_REFUSAL','RUNTIME_QUARANTINE'),'Unfinished attempt')
        require(read(folder/'result.json')==result,'Raw result/ledger identity')
        static=arrays(folder/'source_chart_gate.npz');source=arrays(row['proposal']['path']);verify_row(row['proposal']);require(np.array_equal(static['goal'],source['native_qpos']),'Static source native q equality')
        checks=result['static_checks'];expected,ae,re=static_checks(static,source,model_arrays[row['model_id']],contracts[row['model_id']],alignment[row['observation_id']]);captured=capture[row['observation_id']]
        expected.pop('depth_visual_alignment');expected['authored_object_scene_contract']=True
        expected['capture_camera_and_K32_cast']=camera_cast_contract(source,captured,static)
        require(checks==expected,'Independent static refusal predicates differ: '+aid+' '+str((checks,expected)))
        if status=='MAPPING_REFUSAL':require(not all(checks.values()) and result['outcome'] is None,'Fabricated mapping refusal');refusals.update(k for k,v in checks.items() if not v)
        else:require(all(checks.values()),'Executed failed static check')
        st=ft=None
        if status=='COMPLETED':st=arrays(folder/'physics_state.npz');ft=arrays(folder/'physics_forces.npz')
        elif status=='RUNTIME_QUARANTINE':
            require(result['outcome'] is None and result['reason_code'] in protocol['task_configuration']['quarantine_reason_codes'],'Runtime null guard reason')
            if (folder/'partial_physics_state.npz').is_file():st=arrays(folder/'partial_physics_state.npz')
            if (folder/'partial_physics_forces.npz').is_file():ft=arrays(folder/'partial_physics_forces.npz')
        if st is not None:
            steps=check_states(st,ft,row,model_arrays[row['model_id']],source['rest_quaternions'],status=='COMPLETED');total_steps+=steps
            require(result.get('actual_steps')==steps,'Per-slot actual step denominator')
        total_samples+=verify_recording(folder/'recording',row,st,ft,status)
        if status=='COMPLETED':
            outcome=independent_label(st,ft,row['dt_s']);saved=read(folder/'label_report.json');require({k:saved['outcome'][k] for k in outcome}==outcome=={k:result['outcome'][k] for k in outcome},'Independent label mismatch')
            label_counts['success' if outcome['success'] else outcome['failure_type']]+=1
            require(read(folder/'recording/attempt.json')['attempt']['outcome']==result['outcome'],'Recording outcome/ledger identity')
            if float(np.max(ft['hand_object_normal_force_N'][:round(.25/row['dt_s'])]))<=.001:
                index=round(.25/row['dt_s']);require(np.linalg.norm(st['object_com_world_m'][index]-st['object_com_world_m'][0])<=.001,'Completed support drift guard')
                require(float(angle(R(st['qpos'][0,70:74]),R(st['qpos'][index,70:74])))<=np.deg2rad(2),'Completed support angle guard')
            index=round(1/row['dt_s']);require(np.linalg.norm(st['qpos'][index,:3]-static['opened'][:3])<=.002,'Completed preclose wrist guard')
            require(float(angle(bone_rotations(st['qpos'][index,:67],source['rest_quaternions']),bone_rotations(static['opened'],source['rest_quaternions'])).max())<=np.deg2rad(2),'Completed preclose bone guard')
        for path in sorted(folder.glob('*_first_positive_witness.npz')):
            a=arrays(path);index=int(round(float(a['force_time_s'])/row['dt_s']));require(st is not None and 0<=index<len(st['qpos']),'Contact start clock index')
            close(a['start_qpos'],st['qpos'][index],0,'Contact witness preceding qpos');close(a['start_qvel'],st['qvel'][index],0,'Contact witness preceding qvel')
            close(a['start_hand_rotations'],bone_rotations(st['qpos'][index,:67],source['rest_quaternions']),1e-12,'Contact saved right-tangent/body rotation clock')
            close(a['start_object_body_origin'],st['qpos'][index,67:70],1e-12,'Contact object origin clock');close(a['start_object_body_rotation'],R(st['qpos'][index,70:74]),1e-12,'Contact object rotation clock')
            w=contact_witness(a,model_arrays[row['model_id']],contracts[row['model_id']]);witnesses.append(dict(path=str(path),**w))
        metrics.append(dict(attempt_id=aid,status=status,source_anchor_max_error_m=ae,source_rotation_max_error=re,physical_outcome=result['outcome'],camera_world_max_element_difference=float(np.max(np.abs(source['cameraWorld64']-captured['T_world_camera']))),K_max_element_difference=float(np.max(np.abs(source['K']-captured['camera']['K'])))))
    counts=worker['actual_counts'];require(counts['models']==counts['data']==45,'45model/data count')
    require(counts['steps']==total_steps,'Global actual task step denominator')
    require(counts['alignment_triangle_point_tests']==query_count,'Actual declared alignment triangle-point denominator')
    require(all(0<=counts[k]<=protocol['budget'][k] for k in counts),'Before-work quotas')
    legacy=verify_legacy_v1(protocol,report,raw)
    return dict(status='VERIFIED_COMPLETED_NATIVE_OBJECT_PILOT_V3_SAVED_MECHANICS_CONDITIONAL_AUTHORED_ASSETS',attempts=135,primary=120,matched=15,attempt_status_counts=dict(Counter(r['status'] for r in ledger)),null_physical_outcomes=sum(r['outcome'] is None for r in ledger),label_counts=dict(label_counts),refused_static_checks=dict(refusals),actual_counts=counts,independent_steps=total_steps,recorded_control_samples=total_samples,source_static_metrics=metrics,alignment_diagnostic_only=alignment_metrics,authored_asset_vertex_readback=asset_metrics,contact_witnesses=witnesses,legacy_v1_comparison=legacy,limitations=['Per-step force sums are source-bound; stored first-positive contact mechanics independently checked when present.','Saved nearest-point membership/distances are checked; no new global nearest-triangle search is run.','Saved source/compiled vertex sets are compared independently using exact kd-tree distance.','No new model, engine, skin or joint-anchor FK evaluation.','RGBD correspondence remains unqualified; no hardware/full-skin equivalence or task/training admission is granted by this reader.'],**{k:False for k in FLAGS})

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--report-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Independent receipt must be write-once');require(sha(a.report)==a.report_sha256,'Authorized report digest')
    protocol=read(a.protocol);report=read(a.report);require(report['protocol_sha256']==sha(a.protocol),'Active protocol identity');source=Path(__file__).read_text()
    freeze=report['freeze_commit'];require(subprocess.check_output(['git','-C',str(a.root),'rev-parse','HEAD']).decode().strip()==freeze,'Held freeze HEAD')
    def all_bindings():
        for path,digest in protocol['active_bindings'].items():require(sha(path)==digest,'Active source/input changed: '+path)
        for relative,digest in protocol['selected_git_sources'].items():require(hashlib.sha256(subprocess.check_output(['git','-C',str(a.root),'show',freeze+':'+relative])).hexdigest()==digest==sha(a.root/relative),'Selected Git blob differs')
        require(hashlib.sha256(subprocess.check_output(['git','-C',str(a.root),'show',freeze+':'+str(a.protocol.relative_to(a.root))])).hexdigest()==sha(a.protocol),'Protocol Git blob differs')
    all_bindings();result=inspect(protocol,report,a.root);all_bindings();require(sha(a.report)==a.report_sha256 and Path(__file__).read_text()==source,'Report/verifier changed during readback')
    prior=a.output.parent/'independent_readback.json';require(sha(prior)=='f693ed8352c0fe5a06649d48f0ba35a68ec8d906a66033cdcb82ab63c25a0cb4','Preserved first reader receipt identity')
    receipt=dict(schema_version='manosim_native_object_pilot_independent_saved_readback_v3',protocol=binding(a.protocol),report=binding(a.report),freeze_commit=freeze,expected_bindings=len(protocol['active_bindings']),selected_git_blobs=len(protocol['selected_git_sources']),all_bindings_verified_before=True,all_bindings_verified_after=True,prior_readback=binding(prior),readback_arithmetic_repair='Only saved vertex re-expression: BLAS matmul emitted divide/overflow/invalid warnings with finite outputs; new optimize=False einsum checked for EVERY coordinate against math.fsum under gamma8. Scientific artifacts/gates unchanged.',artifact_inventory={r['path']:{'sha256':r['sha256'],'bytes':r['bytes']} for r in read(report['worker_result']['path'])['artifacts']},verifier=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(source.encode()).hexdigest(),source=source),**result)
    with a.output.open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=receipt['status'],receipt=binding(a.output),attempt_status_counts=receipt['attempt_status_counts'],actual_counts=receipt['actual_counts'])))
if __name__=='__main__':main()
