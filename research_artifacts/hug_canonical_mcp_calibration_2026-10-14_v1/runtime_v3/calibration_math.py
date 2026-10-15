"""Pure source-only MCP-frame algebra; no object/contact/quality inputs."""
import numpy as np

def anatomical_frame(mcp,origin,*,minimum_longitudinal_m,minimum_projected_lateral_m):
    p=np.asarray(mcp,float);o=np.asarray(origin,float)
    if p.shape!=(3,3) or o.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(o).all():raise ValueError('three finite MCPs in index,middle,ring order and one origin required')
    center=p.mean(0);longitudinal=center-o;long_length=float(np.linalg.norm(longitudinal))
    if long_length<=minimum_longitudinal_m:raise ValueError('canonical longitudinal axis is degenerate')
    z=longitudinal/long_length;lateral=p[0]-p[2];projected=lateral-z*np.dot(z,lateral);lat_length=float(np.linalg.norm(projected))
    if lat_length<=minimum_projected_lateral_m:raise ValueError('canonical lateral axis is degenerate')
    x=projected/lat_length;y=np.cross(z,x);R=np.column_stack((x,y,z))
    if np.max(np.abs(R.T@R-np.eye(3)))>1e-12 or abs(np.linalg.det(R)-1)>1e-12:raise ValueError('proper orthonormal anatomical frame required')
    F=np.eye(4);F[:3,:3]=R;F[:3,3]=center
    return F,{'longitudinal_length_m':long_length,'projected_lateral_length_m':lat_length,'lateral_index_minus_ring_m':lateral.tolist(),'MCP_centroid_m':center.tolist(),'frame_determinant':float(np.linalg.det(R))}

def rigid_inverse(T):
    v=np.asarray(T,float);out=np.eye(4);out[:3,:3]=v[:3,:3].T;out[:3,3]=-out[:3,:3]@v[:3,3];return out

def calibrate(mano_mcp,mano_wrist,palm_mcp,**condition):
    FM,dm=anatomical_frame(mano_mcp,mano_wrist,**condition);FP,dp=anatomical_frame(palm_mcp,np.zeros(3),**condition)
    C=FM@rigid_inverse(FP)
    assert abs(np.linalg.det(C[:3,:3])-1)<1e-12
    assert np.max(np.abs(C@FP-FM))<1e-12
    return C,FM,FP,dm,dp

def calibrated_vectors(landmarks_mano,C,R_BP0):
    # Translation is explicit. Subtracting transformed native wrist would cancel it.
    palm=(np.asarray(landmarks_mano)-C[:3,3])@C[:3,:3]
    return palm,palm@np.asarray(R_BP0).T
