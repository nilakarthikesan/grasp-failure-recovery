import numpy as np
import pytest
from grasp_failure_prediction.integrations.observations import validate_depth_encoding


def test_hug_depth_encoding_matches_metric_depth_with_quantization():
    metric = np.array([[0., .50149, .72351]], dtype=np.float32)
    encoded = np.rint(metric * 1000).astype(np.uint16)
    result = validate_depth_encoding(metric, encoded)
    assert result['depth_encoding_checked']
    assert result['maximum_quantization_error_m'] < .000501


def test_stale_or_wrong_unit_hug_depth_is_rejected():
    with pytest.raises(ValueError, match='half a millimeter'):
        validate_depth_encoding(np.array([[.5]], dtype=np.float32), np.array([[5]], dtype=np.uint16))


def test_invalid_depth_is_not_replaced_with_a_valid_encoded_pixel():
    with pytest.raises(ValueError):
        validate_depth_encoding(np.array([[0.]], dtype=np.float32), np.array([[1]], dtype=np.uint16))
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
