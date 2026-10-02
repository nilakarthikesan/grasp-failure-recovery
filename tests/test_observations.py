import numpy as np
from grasp_failure_prediction.integrations.observations import validate_arrays

def example(**changes):
    args=dict(rgb=np.zeros((8,8,3),dtype='uint8'),depth=np.ones((8,8),dtype='float32'),
        K=np.array([[10.,0,4],[0,10,4],[0,0,1]]),mask=np.ones((8,8),dtype='uint8'),
        point=[4,4],registered=True,calibration_matches=True)
    args.update(changes); return validate_arrays(**args)

def test_valid_does_not_claim_model_execution():
    r=example(); assert r['preflight_passed']; assert not r['hug_inference_tested']

def test_shape_and_units_rejected():
    assert not example(depth=np.ones((4,4),dtype='uint16'))['preflight_passed']

def test_bad_calibration_rejected():
    assert not example(K=np.zeros((3,3)))['preflight_passed']

def test_invalid_selection_depth_rejected():
    d=np.ones((8,8),dtype='float32');d[4,4]=0
    assert not example(depth=d)['preflight_passed']

def test_point_and_registration_rejected():
    assert not example(point=[99,99])['preflight_passed']
    assert not example(registered=False)['preflight_passed']

def test_mask_and_nonfinite_rejected():
    assert not example(mask=np.zeros((8,8)))['preflight_passed']
    assert not example(depth=np.full((8,8),np.nan))['preflight_passed']
