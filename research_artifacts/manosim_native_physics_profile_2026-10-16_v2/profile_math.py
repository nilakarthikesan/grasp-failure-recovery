"""Prospective pure-array rules. No native/model/FK/contact or stepping imports."""
import hashlib
from xml.etree import ElementTree as ET
import numpy as np
PROFILE_ID='manosim_authored_capsule_pure_gravity_compensation_v1'
PARENTS=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
BODY_NAMES=tuple(f'mano_body_{i}' for i in range(16))
FLAGS=dict(accepted_mapping=False,native_task_contact_qualified=False,full_skin_parity_claimed=False,new_collection_or_training_allowed=False)
FORCE_LIMIT_N=1e-8
TORQUE_LIMIT_NM=1e-10


def finite(x,shape):
    a=np.asarray(x,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite shaped profile operand required')
    return a

def gravity_profile_xml(text):
    """Explicit new body-level profile. Preserve source masses/gains/joint poses."""
    root=ET.fromstring(text);bodies=root.findall('./worldbody//body');joints=root.findall('./worldbody//joint')
    if len(bodies)!=16 or {b.get('name') for b in bodies}!=set(BODY_NAMES) or len(joints)!=19:raise ValueError('Original native16-body/19-joint hand required')
    before=[ET.tostring(x) for x in root.findall('.//inertial')+root.findall('./actuator/*')]
    for body in bodies:body.set('gravcomp','1')
    for joint in joints:joint.set('actuatorgravcomp','false')
    if before!=[ET.tostring(x) for x in root.findall('.//inertial')+root.findall('./actuator/*')]:raise AssertionError('Inertia/gain mutation forbidden')
    return ET.tostring(root,encoding='unicode')+'\n'


def compensated_model_receipt(model,contract):
    """Existing compiled arrays only. This function never mutates the model."""
    ids=np.asarray(contract.body_ids,dtype=int);joints=np.asarray(contract.joint_ids,dtype=int)
    expected=np.zeros_like(np.asarray(model.body_gravcomp));expected[ids]=1.
    gravity=finite(model.opt.gravity,(3,))
    if not np.array_equal(gravity,[0.,0.,-9.81]) or not np.array_equal(model.body_gravcomp,expected) or np.any(np.asarray(model.jnt_actgravcomp)[joints]) or int(model.ngravcomp)!=16:raise ValueError('Explicit pure-gravity hand-only model profile required')
    return dict(profile_id=PROFILE_ID,body_ids=ids.tolist(),body_gravcomp=np.asarray(model.body_gravcomp).tolist(),jnt_actgravcomp=np.asarray(model.jnt_actgravcomp).tolist(),gravity_world_m_s2=gravity.tolist(),qfrc_bias_feedforward=False,coriolis_compensated=False,servo_offsets=False,per_step_qpos_writes=False,**FLAGS)


def point_jacobian(point,body,anchors,rotations):
    p=finite(point,(3,));a=finite(anchors,(16,3));R=finite(rotations,(16,3,3));i=int(body)
    if i!=body or not 0<=i<16:raise ValueError('Named native body index required')
    J=np.zeros((3,51));J[:,:3]=np.eye(3)
    while i>=0:
        J[:,3+3*i:6+3*i]=np.cross(R[i].T,p-a[i]).T;i=PARENTS[i]
    return J


def gravity_compensation_force(masses,centers,anchors,rotations,gravity):
    """Sum exact per-body -m*g point loads through the named51 tangent map."""
    mass=finite(masses,(16,));p=finite(centers,(16,3));g=finite(gravity,(3,))
    if np.any(mass<=0):raise ValueError('Positive original hand masses required')
    tau=np.zeros(51)
    for i in range(16):tau+=np.einsum('ij,i->j',point_jacobian(p[i],i,anchors,rotations),-mass[i]*g,optimize=False)
    return tau


def compensation_checks(expected,actual_gravcomp,bias,passive,qvel):
    expected=finite(expected,(51,));actual=finite(actual_gravcomp,(51,));bias=finite(bias,(51,));passive=finite(passive,(51,));v=finite(qvel,(51,))
    delta=np.abs(actual-expected);stationary=bool(np.all(v==0))
    checks=dict(translational_gravity_force=bool(delta[:3].max()<=FORCE_LIMIT_N),angular_gravity_force=bool(delta[3:].max()<=TORQUE_LIMIT_NM))
    if stationary:
        residual=np.abs(passive-bias)
        checks.update(stationary_translation_bias_cancelled=bool(residual[:3].max()<=FORCE_LIMIT_N),stationary_angular_bias_cancelled=bool(residual[3:].max()<=TORQUE_LIMIT_NM))
    return dict(checks=checks,stationary=stationary,moving_bias_minus_gravity=(bias-actual).tolist(),moving_residual_required_zero=False)


def two_plane_indentation(normals,depth_m=.0005,max_displacement_m=.002,max_condition=1e6):
    """Declared geometry-only minimum-norm root move; no contact outcomes used."""
    N=finite(normals,(2,3))
    if np.max(np.abs(np.einsum('ij,ij->i',N,N,optimize=False)-1))>1e-12:raise ValueError('Unit planned plane normals required')
    if not np.isfinite([depth_m,max_displacement_m,max_condition]).all() or depth_m<=0 or max_displacement_m<=0 or max_condition<1:raise ValueError('Positive declared indentation bounds required')
    c=float(np.einsum('i,i->',N[0],N[1],optimize=False));den=1-abs(c)
    condition=np.inf if den<=0 else (1+abs(c))/den
    if not np.isfinite(condition) or condition>max_condition:raise ValueError('Declared two-plane system ill-conditioned; no replacement normals')
    determinant=1-c*c;rhs=np.full(2,-depth_m);weights=np.array([rhs[0]-c*rhs[1],rhs[1]-c*rhs[0]])/determinant
    delta=np.einsum('ij,i->j',N,weights,optimize=False)
    if np.linalg.norm(delta)>max_displacement_m:raise ValueError('Declared root indentation bound exceeded; no rescaling')
    if np.max(np.abs(np.einsum('ij,j->i',N,delta,optimize=False)-rhs))>128*np.finfo(float).eps*max(1.,depth_m):raise ValueError('Minimum-norm indentation arithmetic failed')
    return delta,dict(normal_condition=condition,depth_each_m=depth_m,root_displacement_m=delta.tolist(),maximum_displacement_m=max_displacement_m,outcome_selection_used=False)


def rigid_translation_guard(distances,offsets,*,required_clearance_m,oracle_error_m,oracle_id):
    """Continuous-in-segment lower bound only for qualified rigid translation.

    Caller must supply signed separation lower-distance measurements for every
    hand collider/obstacle at all knots. Only exact analytic oracle IDs are
    supported here; sampled mesh/GJK distances do not inherit an error bound.
    """
    d=np.asarray(distances,dtype=np.float64);x=np.asarray(offsets,dtype=np.float64)
    if d.ndim!=3 or len(d)<2 or min(d.shape[1:])<1 or x.shape!=(len(d),3) or not np.isfinite(d).all() or not np.isfinite(x).all():raise ValueError('Complete finite knot/collider/obstacle distance domain required')
    if oracle_id!='analytic_plane_sphere_hand_bounds_v1' or not np.isfinite([required_clearance_m,oracle_error_m]).all() or required_clearance_m<0 or oracle_error_m<0:raise ValueError('Qualified declared analytic distance oracle required')
    radius=np.linalg.norm(np.diff(x,axis=0),axis=1)/2
    lower=np.minimum(d[:-1],d[1:])-radius[:,None,None]-oracle_error_m
    return dict(passed=bool(np.all(lower>=required_clearance_m)),minimum_segment_clearance_lower_m=float(lower.min()),required_clearance_m=float(required_clearance_m),oracle_error_m=float(oracle_error_m),oracle_id=oracle_id,knot_count=len(d),collider_count=d.shape[1],obstacle_count=d.shape[2],rotation_and_finger_motion_allowed=False,object_query_fitting=False,sampled_only=False,proof_scope='1-Lipschitz signed separation under rigid root translation and fixed obstacles; only stated analytic oracle')


def analytic_hand_separations(centers,rotations,sizes,palm_vertices,sphere_center,sphere_radius,table_z):
    """16×2 certified geometric lower distances: table and fixed sphere.

    Fifteen capsule distances are analytic. The convex palm uses exact vertex
    support against the table and an enclosing-sphere distance lower bound.
    Convexification/relabeling or object-query fitting is never performed.
    """
    c=finite(centers,(16,3));R=finite(rotations,(16,3,3));size=finite(sizes,(16,3));v=np.asarray(palm_vertices,dtype=np.float64);s=finite(sphere_center,(3,))
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all() or np.max(np.abs(c))>10 or np.max(np.abs(s))>10 or np.max(np.abs(v))>10:raise ValueError('Bounded compiled convex palm vertices required')
    if np.max(np.abs(np.einsum('bji,bjk->bik',R,R,optimize=False)-np.eye(3)))>1e-10 or np.any(np.linalg.det(R)<0) or np.any(size[1:,0]<=0) or np.any(size[1:,1]<0) or not np.isfinite([sphere_radius,table_z]).all() or sphere_radius<=0:raise ValueError('Proper rigid capsule/sphere/table geometry required')
    distances=np.empty((16,2));local_center=(v.min(axis=0)+v.max(axis=0))/2
    palm_center=c[0]+np.einsum('ij,j->i',R[0],local_center,optimize=False);radius=float(np.linalg.norm(v-local_center,axis=1).max())
    world_vertices=np.einsum('ij,kj->ki',R[0],v,optimize=False)+c[0]
    distances[0]=[float(world_vertices[:,2].min()-table_z),float(np.linalg.norm(palm_center-s)-radius-sphere_radius)]
    for i in range(1,16):
        axis=R[i,:,2];half=size[i,1];radius_i=size[i,0]
        t=float(np.clip(np.einsum('i,i->',s-c[i],axis,optimize=False),-half,half));closest=c[i]+t*axis
        distances[i]=[float(c[i,2]-abs(axis[2])*half-radius_i-table_z),float(np.linalg.norm(closest-s)-radius_i-sphere_radius)]
    if not np.isfinite(distances).all():raise ValueError('Nonfinite analytic separation')
    return distances,dict(palm_bound='enclosing sphere over every compiled convex-palm vertex; table exact support',palm_bound_center_world_m=palm_center.tolist(),palm_bound_radius_m=radius,capsule_count=15,palm_count=1,convex_palm_vertex_count=len(v),geometry_or_contact_query_calls=0)
