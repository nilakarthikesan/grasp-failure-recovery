import numpy as np
import pytest
from calibration_math import anatomical_frame,calibrate,calibrated_vectors,rigid_inverse
K=dict(minimum_longitudinal_m=1e-6,minimum_projected_lateral_m=1e-6)

def test_identity_no_shift():
 p=np.array([[.03,0,.09],[.01,0,.10],[-.01,0,.09]])
 C,*_=calibrate(p,np.zeros(3),p,**K);assert np.allclose(C,np.eye(4),atol=1e-15)

def test_rigid_translation_rotation_and_world_roundtrip():
 p=np.array([[.03,0,.09],[.01,0,.10],[-.01,0,.09]]);R=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);t=np.array([.01,-.02,.005])
 C,FM,FP,*_=calibrate(p@R.T+t,t,p,**K)
 assert np.allclose(C[:3,:3],R,atol=1e-14);assert np.allclose(C[:3,3],t,atol=1e-14)
 L=np.array([[.05,.02,.13],[-.02,.04,.1]]);RB=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
 vp,vb=calibrated_vectors(L,C,RB);world=vb@(C[:3,:3]@RB.T).T+C[:3,3]
 assert np.max(np.abs(world-L))<1e-14
 assert np.max(np.abs(vp[0]-(L[0]-C[:3,3])@C[:3,:3]))<1e-14
 # The old native-wrist origin would wrongly erase the calibrated translation.
 old_cancel=vp-(-C[:3,3])@C[:3,:3]
 assert np.max(np.abs(old_cancel-vp))>.001
 assert np.max(np.abs(rigid_inverse(C)@C-np.eye(4)))<1e-14

@pytest.mark.parametrize('points,origin',[([[1,0,0],[0,0,0],[-1,0,0]],[0,0,0]),([[0,0,1],[0,0,2],[0,0,3]],[0,0,0])])
def test_degenerate_axes_refused(points,origin):
 with pytest.raises(ValueError):anatomical_frame(points,origin,**K)

def test_index_ring_sign_and_proper_handedness():
 p=np.array([[.03,0,.09],[.01,0,.10],[-.01,0,.09]]);F,_=anatomical_frame(p,np.zeros(3),**K)
 assert np.dot(F[:3,0],p[0]-p[2])>0
 assert np.max(np.abs(np.cross(F[:3,0],F[:3,1])-F[:3,2]))<1e-14
 assert np.linalg.det(F[:3,:3])>0
