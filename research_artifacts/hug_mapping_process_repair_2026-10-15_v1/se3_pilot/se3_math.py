"""Thirty-coordinate bounded placement; no simulation or optimization imports."""
import numpy as np

ROTATION_SCALE_RAD = .025
ROTATION_COMPONENT_LIMIT_RAD = float(np.deg2rad(15.))

def decode_state(state, original_q):
    x=np.asarray(state,dtype=float);q0=np.asarray(original_q,dtype=float)
    if x.shape!=(30,) or q0.shape!=(24,) or not np.isfinite(x).all() or not np.isfinite(q0).all():
        raise ValueError('finite30-state and24 original joints required')
    return q0+.025*x[:24],.005*x[24:27],ROTATION_SCALE_RAD*x[27:30]

def skew(v):
    v=np.asarray(v,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('finite3-vector required')
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def left_jacobian_so3(omega):
    w=np.asarray(omega,dtype=float);K=skew(w);theta2=float(w@w)
    if theta2<1e-8:
        A=.5-theta2/24.+theta2*theta2/720.
        B=1./6.-theta2/120.+theta2*theta2/5040.
    else:
        theta=np.sqrt(theta2);A=(1.-np.cos(theta))/theta2;B=(theta-np.sin(theta))/(theta2*theta)
    return np.eye(3)+A*K+B*(K@K)

def rotvec_quaternion_wxyz(omega):
    w=np.asarray(omega,dtype=float);skew(w);theta=float(np.linalg.norm(w))
    return np.r_[np.cos(theta/2.),.5*np.sinc(theta/(2.*np.pi))*w]

def left_compose_root_quaternion(original, omega):
    q=np.asarray(original,dtype=float);w=np.asarray(omega,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q),1.,rtol=0,atol=1e-12):
        raise ValueError('unit finite original quaternion required')
    skew(w)
    # Preserve the exact original bytes; zero rotation is the old method.
    if np.array_equal(w,np.zeros(3)):return q.copy()
    d=rotvec_quaternion_wxyz(w)
    result=np.r_[d[0]*q[0]-np.dot(d[1:],q[1:]),d[0]*q[1:]+q[0]*d[1:]+np.cross(d[1:],q[1:])]
    return result/np.linalg.norm(result)

def rotation_bounds_scaled():
    hi=ROTATION_COMPONENT_LIMIT_RAD/ROTATION_SCALE_RAD
    while hi*ROTATION_SCALE_RAD>ROTATION_COMPONENT_LIMIT_RAD:hi=np.nextafter(hi,0.)
    return np.tile([-hi,hi],(3,1))

def rotation_point_jacobian(point_minus_palm_world, omega):
    return -skew(point_minus_palm_world)@left_jacobian_so3(omega)*ROTATION_SCALE_RAD
