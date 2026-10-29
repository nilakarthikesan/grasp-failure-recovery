import numpy as np
from grasp_failure_prediction.evaluation.phase1_wuji import rigid_fit, load_proposal, COMMIT

def test_rigid_fit_preserves_metric_points_and_proper_rotation():
    source=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]])
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    target=source@rotation.T+[.1,.2,.3]
    actual,translation=rigid_fit(source,target)
    np.testing.assert_allclose(source@actual.T+translation,target,atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(actual),1)

def test_camera_world_transform_does_not_apply_native_wrist_twice(tmp_path):
    points=np.arange(63,dtype=float).reshape(21,3)/1000
    camera=np.eye(4);camera[:3,3]=[1,2,3]
    wrist=np.eye(4);wrist[:3,3]=[10,20,30]
    np.save(tmp_path/'T_world_camera.npy',camera)
    np.savez(tmp_path/'proposal.npz',landmarks_3d=points,T_camera_wrist=wrist)
    result,_,_=load_proposal(tmp_path,tmp_path)
    np.testing.assert_allclose(result,points+[1,2,3])
    assert COMMIT=='5c4dfa16917355d073e2e1d7e3d03897b6cf9508'
