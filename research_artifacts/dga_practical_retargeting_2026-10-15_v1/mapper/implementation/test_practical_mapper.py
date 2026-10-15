import numpy as np
import pytest
from practical_mapper import explicit_box_projection, allowed, preference

def test_projection_is_explicit_minimum_coordinatewise_and_keeps_original():
    original=np.array([-2.,.4,3.]);original_bytes=original.tobytes()
    bounds=np.array([[-1.,1.],[-1.,1.],[-1.,1.]])
    projected,changed=explicit_box_projection(original,bounds)
    assert projected.tolist()==[-1.,.4,1.]
    assert changed.tolist()==[0,2]
    assert original.tobytes()==original_bytes
    again,rechanged=explicit_box_projection(projected,bounds)
    assert np.array_equal(again,projected) and len(rechanged)==0
    for shift in (-.01,0.,.01):
        alternative=np.array([-.99,.4+shift,.99])
        assert np.sum((projected-original)**2)<=np.sum((alternative-original)**2)

@pytest.mark.parametrize('values,bounds',[
    ([np.nan],[[0,1]]),([0],[[1,0]]),([0],[[0,np.inf]]),([0,1],[[0,1]])])
def test_projection_refuses_malformed(values,bounds):
    with pytest.raises(ValueError):explicit_box_projection(values,bounds)

def metrics(landmark=2.,field=3.,sse=10.,mse=7.):
    return {'maximum_landmark_error_m':landmark,'maximum_unsigned_field_error_m':field,
            'landmark_SSE_m2':sse,'unsigned_field_MSE_m2':mse}

def test_no_aggregate_override_of_maximum_guards():
    baseline=metrics()
    assert not allowed(metrics(landmark=2.01,field=0.,sse=0.,mse=0.),baseline)
    assert not allowed(metrics(landmark=0.,field=3.01,sse=0.,mse=0.),baseline)
    assert not allowed(metrics(landmark=0.,field=0.,sse=10.01,mse=0.),baseline)
    assert not allowed(metrics(field=np.nan),baseline)
    assert allowed(baseline,baseline) and not allowed(baseline,baseline,strict=True)
    assert allowed(metrics(landmark=1.9),baseline,strict=True)

def test_preference_does_not_replace_eligibility():
    baseline=metrics();lower_objective_but_bad=metrics(landmark=100,field=.1,sse=0,mse=0)
    assert preference(lower_objective_but_bad)<preference(baseline)
    assert not allowed(lower_objective_but_bad,baseline)
