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

class FakeContext:
    """Manufactured branch contract fixture, not native/geometry evidence."""
    def __init__(self):
        self.names=('WRJ2','WRJ1','FFJ4','FFJ3','FFJ2','FFJ1','MFJ4','MFJ3','MFJ2','MFJ1',
                    'RFJ4','RFJ3','RFJ2','RFJ1','LFJ5','LFJ4','LFJ3','LFJ2','LFJ1',
                    'THJ5','THJ4','THJ3','THJ2','THJ1')
        self.fingers=self.names[2:];self.qsource=dict.fromkeys(self.names,0.);self.qsource['FFJ4']=-2.
        self.target_model={'joints':{n:{'limit':{'lower':-1.,'upper':1.}} for n in self.names}}
        self.sites=['palm','fftip','mftip','rftip','lftip','thtip'];self.queries=np.zeros((856,3));self.source_palm=np.eye(4)
    def source_identity(self):return 'fixture-original'
    def verify_immutable_source(self):assert self.qsource['FFJ4']==-2.
    def certify_payload(self,state):return state
    def make_state(self,x,current,block):
        from practical_mapper import PREFIX
        q=dict(self.qsource);q.update(zip(self.fingers,x[:22]))
        errors=np.zeros((6,3))
        for i,prefix in enumerate(PREFIX.values(),start=1):
            name=next(n for n in self.names if n.startswith(prefix));errors[i,0]=q[name]-self.qsource[name]
        return {'variables':x.copy(),'q':q,'errors_m':errors,'palm':np.eye(4),'root':np.eye(4),
                'forearm':np.eye(4),'delta':np.eye(4),'landmark_metrics':{},
                'strict_domain_valid':True,'original_wrists_preserved':True,
                'rigid_frames_valid':True,'independent_fk_valid':True}
    def evaluate_fields(self,state,current=None,block=None):
        result=np.zeros((856,8));result[:,1]=abs(state['q']['FFJ4']-self.qsource['FFJ4']);return result

@pytest.mark.parametrize('exception',[ValueError('invalid solver output'),FloatingPointError('nonfinite workspace'),RuntimeError('solver crash')])
def test_solver_exception_selects_reference_and_preserves_accounting(monkeypatch,exception):
    import practical_mapper
    def broken(*args,**kwargs):raise exception
    monkeypatch.setattr(practical_mapper,'least_squares',broken)
    ctx=FakeContext();baseline,selected,trace=practical_mapper.map_pose(ctx,np.zeros((856,8)))
    assert selected is baseline
    assert selected['q']['FFJ4']==-1. and ctx.qsource['FFJ4']==-2.
    assert trace['branches'][0]['optimizer_exception']['type']==type(exception).__name__
    assert not trace['branches'][0]['adopted_correction']
    assert all(x['status']=='IDENTITY_BRANCH_RETAINED' for x in trace['branches'][1:])
