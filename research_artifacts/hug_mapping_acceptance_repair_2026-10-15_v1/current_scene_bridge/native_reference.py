"""Independent camera/native/operator/source-XML palm composition.

No project or MuJoCo imports. Raw source rotations are retained; the final
quaternion uses the declared normalized matrix-to-quaternion convention.
"""
import numpy as np
from xml_reference import quaternion_matrix_wxyz,transform
A=np.array([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])
def quaternion_from_matrix(R):
 R=np.asarray(R,float);assert R.shape==(3,3) and np.isfinite(R).all()
 # Positive trace, otherwise largest diagonal branch; quaternion normalized explicitly, without an
 # SVD projection that would silently change how the current helper treats
 # small nonorthogonality in a stored float-valued native frame.
 choices=np.array([np.trace(R),R[0,0]-R[1,1]-R[2,2],R[1,1]-R[0,0]-R[2,2],R[2,2]-R[0,0]-R[1,1]])
 k=0 if choices[0]>0 else 1+int(np.argmax(np.diag(R)));q=np.zeros(4);q[k]=np.sqrt(1+choices[k])/2;s=4*q[k]
 if k==0:q[1:]=[(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,(R[1,0]-R[0,1])/s]
 elif k==1:q[[0,2,3]]=[(R[2,1]-R[1,2])/s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s]
 elif k==2:q[[0,1,3]]=[(R[0,2]-R[2,0])/s,(R[0,1]+R[1,0])/s,(R[1,2]+R[2,1])/s]
 else:q[[0,1,2]]=[(R[1,0]-R[0,1])/s,(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s]
 return q/np.linalg.norm(q)
def expected_native_palm(camera,wrist,fixed_xml_palm):
 camera=np.asarray(camera,float);wrist=np.asarray(wrist,float)
 for value in [camera,wrist]:assert value.shape==(4,4) and np.isfinite(value).all() and np.array_equal(value[3],[0,0,0,1])
 world=np.einsum('ij,jk->ik',camera,wrist)
 rawR=np.einsum('ij,jk,kl->il',world[:3,:3],A.T,fixed_xml_palm[:3,:3])
 q=quaternion_from_matrix(rawR)
 return {'raw_world_wrist':world,'raw_world_palm_rotation':rawR,'quaternion_wxyz':q,'palm_world':transform(world[:3,3],quaternion_matrix_wxyz(q)),
         'raw_palm_rotation_orthogonality_max_error':float(np.max(np.abs(rawR.T@rawR-np.eye(3)))),
         'raw_palm_rotation_determinant':float(np.linalg.det(rawR)),
         'normalization_max_rotation_matrix_change':float(np.max(np.abs(quaternion_matrix_wxyz(q)-rawR)))}
