"""Source-initialized proper-quaternion domain and literalfloat32 inputs.

Pure coordinate algebra only. No model, FK, skin or geometry operations.
Equivalent2pi branches are selected in coordinate space, never by a surface.
"""
from __future__ import annotations
import math
import numpy as np

DOMAIN='mano_source_postAA32_exp64_camera64_named_state_v1'
BRANCHES=(-1,0,1)


def array(x,shape,dtype,name):
    a=np.asarray(x)
    if a.shape!=shape or a.dtype!=np.dtype(dtype) or not np.isfinite(a).all():
        raise ValueError('Finite literal '+name+' '+str(shape)+' '+dtype+' required')
    return a


def multiply(a,b):
    w,x,y,z=np.moveaxis(a,-1,0);v,i,j,k=np.moveaxis(b,-1,0)
    return np.stack((w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v),axis=-1)


def conjugate(q):
    out=np.array(q,copy=True);out[...,1:]*=-1;return out


def unit_quaternions(q,shape):
    q=array(q,shape,'float64','unit quaternion')
    if np.any(np.abs(np.sum(q*q,axis=-1)-1)>1e-12):raise ValueError('Proper literal unit quaternion required; no pose normalization')
    return q


def exp_quaternion(aa):
    a=np.asarray(aa,dtype=np.float64)
    if a.shape[-1]!=3 or not np.isfinite(a).all():raise ValueError('Finite threecomponent axisangles required')
    theta=np.sqrt(np.sum(a*a,axis=-1));scale=np.full_like(theta,.5);nz=theta!=0
    scale[nz]=np.sin(theta[nz]/2)/theta[nz]
    return np.concatenate((np.cos(theta/2)[...,None],a*scale[...,None]),axis=-1)


def rotation_matrices(q):
    q=unit_quaternions(q,q.shape)
    w,x,y,z=np.moveaxis(q,-1,0);s=2/np.sum(q*q,axis=-1)
    return np.stack((1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w),s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w),s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)),axis=-1).reshape(q.shape[:-1]+(3,3))


def camera_quaternion(camera):
    t=array(camera,(4,4),'float64','original cameraWorld')
    if not np.array_equal(t[3],np.array([0.,0.,0.,1.])):raise ValueError('Literal homogeneous camera required')
    r=t[:3,:3]
    if np.max(np.abs(np.einsum('ji,jk->ik',r,r,optimize=False)-np.eye(3)))>1e-12 or abs(float(np.linalg.det(r))-1)>1e-12:
        raise ValueError('Proper original camera rotation required; no projection')
    trace=float(np.trace(r))
    if trace>0:
        s=2*np.sqrt(1+trace);q=np.array([s/4,(r[2,1]-r[1,2])/s,(r[0,2]-r[2,0])/s,(r[1,0]-r[0,1])/s])
    else:
        k=int(np.argmax(np.diag(r)))
        if k==0:
            s=2*np.sqrt(1+r[0,0]-r[1,1]-r[2,2]);q=np.array([(r[2,1]-r[1,2])/s,s/4,(r[0,1]+r[1,0])/s,(r[0,2]+r[2,0])/s])
        elif k==1:
            s=2*np.sqrt(1+r[1,1]-r[0,0]-r[2,2]);q=np.array([(r[0,2]-r[2,0])/s,(r[0,1]+r[1,0])/s,s/4,(r[1,2]+r[2,1])/s])
        else:
            s=2*np.sqrt(1+r[2,2]-r[0,0]-r[1,1]);q=np.array([(r[1,0]-r[0,1])/s,(r[0,2]+r[2,0])/s,(r[1,2]+r[2,1])/s,s/4])
    raw=q.copy();norm=float(np.linalg.norm(q))
    if abs(norm-1)>1e-12:raise ValueError('Camera matrix conversion outside representation precision')
    q=q/norm # Explicit arithmetic normalization of a proper-matrix conversion, never SO3 fitting.
    error=float(np.max(np.abs(rotation_matrices(q)-r)))
    if error>1e-12:raise ValueError('Camera representation roundtrip failed')
    return q,dict(raw_quaternion_norm=norm,normalization_max_change=float(np.max(np.abs(q-raw))),matrix_roundtrip_error=error)


def nearest_equivalent_axis_angle(q,reference32):
    q=unit_quaternions(q,(4,));reference32=array(reference32,(3,),'float32','original axisangle reference')
    reference=reference32.astype(np.float64)
    if np.linalg.norm(reference)>2*math.pi+1e-6:raise ValueError('Source quaternion-log branch domain<=2pi required')
    # q and-q are the same rotation. Use the canonical hemisphere so theta
    # lies in[0,pi]; with reference norm<=2pi, k=-1,0,1 includes a nearest
    # equivalent coordinate even when a perturbation crosses2pi.
    nonzero=np.flatnonzero(q)
    if q[0]<0 or (q[0]==0 and q[nonzero[0]]<0):q=-q
    length=float(np.linalg.norm(q[1:]));theta=2*math.atan2(length,float(q[0]))
    if length==0:
        rn=float(np.linalg.norm(reference));axis=reference/rn if rn else np.array([1.,0.,0.])
    else:axis=q[1:]/length
    candidates=np.array([axis*(theta+2*math.pi*k) for k in BRANCHES])
    selected=int(np.argmin(np.sum((candidates-reference)**2,axis=1)))
    return candidates[selected],BRANCHES[selected]


def source_initialized_qpos(original_aa32,original_t32,camera,rest_quaternions):
    aa=array(original_aa32,(48,),'float32','original post-mean axisangles').reshape(16,3)
    t=array(original_t32,(3,),'float32','original camera translation')
    rest=unit_quaternions(rest_quaternions,(16,4));cq,conversion=camera_quaternion(camera)
    body=exp_quaternion(aa.astype(np.float64));body[0]=multiply(cq,body[0])
    joint=multiply(conjugate(rest),body)
    q=np.empty(67,dtype=np.float64);q[:3]=np.einsum('ij,j->i',camera[:3,:3],t.astype(float),optimize=False)+camera[:3,3];q[3:]=joint.reshape(64)
    return q,dict(domain=DOMAIN,root='literalcameraWorld64 times Exp64(original postAA32)',
        locals='Exp64(original postAA32)',source_input_values_unchanged=True,camera_representation=conversion,
        old_GS64_state_domain_reused=False,geometry_fitting_or_offset=False)


def recover_source_inputs(qpos,rest_quaternions,camera,original_aa32):
    q=array(qpos,(67,),'float64','actual named qpos');rest=unit_quaternions(rest_quaternions,(16,4))
    joint=unit_quaternions(q[3:].reshape(16,4),(16,4));cq,conversion=camera_quaternion(camera)
    body=multiply(rest,joint);body_camera=body.copy();body_camera[0]=multiply(conjugate(cq),body[0])
    original=array(original_aa32,(48,),'float32','original post-mean axisangles').reshape(16,3)
    recovered=[];branches=[]
    for quat,ref in zip(body_camera,original):
        aa,branch=nearest_equivalent_axis_angle(quat,ref);recovered.append(aa);branches.append(branch)
    aa64=np.asarray(recovered);aa32=aa64.astype(np.float32).reshape(48)
    t64=np.einsum('ji,j->i',camera[:3,:3],q[:3]-camera[:3,3],optimize=False);t32=t64.astype(np.float32)
    r_actual=rotation_matrices(body_camera);r_cast=rotation_matrices(exp_quaternion(aa32.astype(float).reshape(16,3)))
    return dict(axis_angles_float32=aa32,translation_camera_float32=t32,
        quaternion_log_axis_angles_float64=aa64,equivalent_branch_indices=np.asarray(branches,dtype=np.int64),
        camera_translation_float64=t64,actual_body_camera_rotations=r_actual,
        float32_axisangle_rotation_roundtrip_errors=np.max(np.abs(r_cast-r_actual),axis=(1,2)),
        camera_representation=conversion,
        quaternion_hemisphere_rule='nonnegative scalar; exactzero scalar uses first nonzero component positive; doublecover only')


def require_original_bytes(record,original_aa32,original_t32):
    aa=array(original_aa32,(48,),'float32','original axisangles')
    t=array(original_t32,(3,),'float32','original translation')
    gotaa=array(record['axis_angles_float32'],(48,),'float32','recovered axisangles')
    gott=array(record['translation_camera_float32'],(3,),'float32','recovered translation')
    if gotaa.tobytes()!=aa.tobytes() or gott.tobytes()!=t.tobytes():
        raise ValueError('Exact q0 source AA32/translation32 byte identity required before all source/model work')
    return dict(original_axisangle_bytes_exact=True,original_translation_bytes_exact=True,
        branch_selection='nearest original among equivalent2pi coordinates only',geometry_lookup_or_offset=False)


def motion_states(q0,grid):
    q=array(q0,(67,),'float64','new source initialized q0');rows=[]
    for row in grid:
        delta=array(np.asarray(row['delta51'],dtype=np.float64),(51,),'float64','declared local tangent')
        out=q.copy();out[:3]+=delta[:3]
        out[3:]=multiply(q[3:].reshape(16,4),exp_quaternion(delta[3:].reshape(16,3))).reshape(64)
        rows.append(out)
    if len(rows)!=109 or not np.array_equal(np.asarray(grid[0]['delta51']),np.zeros(51)):raise ValueError('Exact finite109 grid includingq0 required')
    return np.asarray(rows)


def world_from_camera32(camera_vertices32,camera):
    """Transform only AFTER the caller's literal sourcefloat32 t addition."""
    vertices=np.asarray(camera_vertices32)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or vertices.dtype!=np.dtype('float32') or not np.isfinite(vertices).all():
        raise ValueError('Sourcefloat32 camera vertices afterfloat32 translation required')
    camera_quaternion(camera) # Validation only; use the ORIGINAL matrix below.
    return np.einsum('ij,kj->ki',camera[:3,:3],vertices,optimize=False)+camera[:3,3]
