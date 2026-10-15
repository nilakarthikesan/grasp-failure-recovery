from copy import deepcopy
import numpy as np
import pytest
from grasp_failure_prediction.evaluation.constrained_surface_refinement import ConstrainedEvaluation,native_direction_residual
from keyed_direction_evidence import KeyedDirectionEvidence


def setup_case():
    q=np.zeros(24)
    refs={'palm':{'available':True,'native_score':np.array([.2,-.3]),'weights':np.array([.1,1/6-.1])},'thumb':{'available':False}}
    def make(scores=(.25,-.2),valid=(True,True),tips=None):
        r=native_direction_residual(refs['palm']['native_score'],np.array(scores),refs['palm']['weights']*6,np.array(valid))/np.sqrt(6)
        v=ConstrainedEvaluation(np.r_[9.,r,8.],np.ones((5,3))*.001 if tips is None else tips)
        d={'normal_diagnostics':{'palm':{'available':True,'native_score':refs['palm']['native_score'].tolist(),
            'candidate_score':list(scores),'candidate_valid':list(valid),'residual':r.tolist()},'thumb':{'available':False}}}
        return v,d
    return q,refs,make


def test_cached_q_lookup_ignores_later_record():
    q,r,make=setup_case();s=KeyedDirectionEvidence(q,r,normal_block_start=1,normal_block_count=2)
    v,d=make();s.record(q,v,d);q1=q.copy();q1[3]=.01;v1,d1=make(scores=(.8,-.1));s.record(q1,v1,d1)
    assert s.assess(q,v)['accepted']
    assert not s.assess(q1,v1)['accepted']
    assert s.assess(q,v)['metadata_callback_sequence']==1


def test_exact_bytes_not_nearest_pose():
    q,r,make=setup_case();s=KeyedDirectionEvidence(q,r,normal_block_start=1,normal_block_count=2);v,d=make();s.record(q,v,d)
    q1=q.copy();q1[0]=np.nextafter(0.,1.)
    with pytest.raises(ValueError,match='no full callback'):s.assess(q1,v)
    q1=q.copy();q1[0]=-0.
    with pytest.raises(ValueError,match='no full callback'):s.assess(q1,v)


@pytest.mark.parametrize('changed',['residual','tips'])
def test_guard_requires_complete_value_agreement(changed):
    q,r,make=setup_case();s=KeyedDirectionEvidence(q,r,normal_block_start=1,normal_block_count=2);v,d=make();s.record(q,v,d)
    residual=v.residuals.copy();tips=v.fingertip_error_vectors_m.copy()
    if changed=='residual':residual[-1]+=.01
    else:tips[0,2]+=.01
    with pytest.raises(ValueError,match='guard value differs'):s.assess(q,ConstrainedEvaluation(residual,tips))


def test_input_mutation_detached_and_assessment_detached():
    q,r,make=setup_case();s=KeyedDirectionEvidence(q,r,normal_block_start=1,normal_block_count=2);v,d=make();s.record(q,v,d)
    r['palm']['native_score'][:]=.9;d['normal_diagnostics']['palm']['candidate_score'][0]=.9
    result=s.assess(q,v);assert result['accepted'];result['parts']['palm']['candidate_direction_sse']=100
    assert s.assess(q,v)['parts']['palm']['candidate_direction_sse']<1
    audit=s.audit();audit['callbacks'][0]['normal_diagnostics']['palm']['candidate_score'][0]=.9
    assert s.assess(q,v)['accepted']


@pytest.mark.parametrize('changed',['residual','tips','normal','diagnostic'])
def test_repeat_q_changed_evidence_refused(changed):
    q,r,make=setup_case();s=KeyedDirectionEvidence(q,r,normal_block_start=1,normal_block_count=2);v,d=make();s.record(q,v,d)
    residual=v.residuals.copy();tips=v.fingertip_error_vectors_m.copy();dd=deepcopy(d)
    if changed=='residual':residual[-1]+=.01
    elif changed=='tips':tips[0,2]+=.01
    elif changed=='normal':dd['normal_diagnostics']['palm']['candidate_valid'][0]=False
    else:dd['extra']='changed'
    with pytest.raises(ValueError):s.record(q,ConstrainedEvaluation(residual,tips),dd)
    assert s.callback_count==1


def test_fresh_terminal_sequence_is_not_cached_sequence():
    q,r,make=setup_case();s=KeyedDirectionEvidence(q,r,normal_block_start=1,normal_block_count=2);v,d=make();s.record(q,v,d)
    assert s.assess(q,v)['metadata_callback_sequence']==1
    s.record(q,v,d);assert s.assess(q,v)['metadata_callback_sequence']==2
    assert s.audit()['callbacks'][1]['repeat_exact_q_callback']


@pytest.mark.parametrize('changed',['native','normal_residual','full_facing','missing_part','availability','boolean'])
def test_direction_metadata_agreement_and_shapes(changed):
    q,r,make=setup_case();s=KeyedDirectionEvidence(q,r,normal_block_start=1,normal_block_count=2);v,d=make();res=v.residuals.copy()
    if changed=='native':d['normal_diagnostics']['palm']['native_score'][0]=.9
    elif changed=='normal_residual':d['normal_diagnostics']['palm']['residual'][0]+=.1
    elif changed=='full_facing':res[1]+=.1
    elif changed=='missing_part':d['normal_diagnostics'].pop('thumb')
    elif changed=='availability':d['normal_diagnostics']['palm']['available']=False
    elif changed=='boolean':d['normal_diagnostics']['palm']['candidate_valid']=[1,1]
    with pytest.raises(ValueError):s.record(q,ConstrainedEvaluation(res,v.fingertip_error_vectors_m),d)


def test_first_q_must_be_original():
    q,r,make=setup_case();s=KeyedDirectionEvidence(q,r,normal_block_start=1,normal_block_count=2);v,d=make();q[0]=.01
    with pytest.raises(ValueError,match='first full callback'):s.record(q,v,d)


@pytest.mark.parametrize('bad_direction',[False,True])
def test_solver_integration_selects_or_refuses_and_freshly_rechecks(bad_direction):
    from feasible_surface_refinement import FeasibleRefinementConfig,solve_feasible_refinement
    q,r,make=setup_case();store=KeyedDirectionEvidence(q,r,normal_block_start=1,normal_block_count=2)
    def evaluation(x):
        scores=(.8,-.1) if bad_direction and x[0]>0 else (.25,-.2)
        v,d=make(scores=scores)
        residual=v.residuals.copy();residual[0]=9.-x[0]
        result=ConstrainedEvaluation(residual,v.fingertip_error_vectors_m)
        store.record(x,result,d);return result
    seed=q.copy();seed[0]=.01
    result,audit=solve_feasible_refinement(q,np.tile([-1.,1.],(24,1)),evaluation,
        evaluate_fingertips=lambda x:np.ones((5,3))*.001,
        config=FeasibleRefinementConfig(maximum_pattern_sweeps=0,maximum_full_geometry_evaluations=4),
        predeclared_seed_qpos_rad=[seed],candidate_guard=store.assess)
    assert bool(np.array_equal(result,q))==bad_direction
    last=store.audit()['callbacks'][-1]
    final=ConstrainedEvaluation(last['residuals'],last['fingertip_error_vectors_m'])
    terminal=store.assess(result,final)
    assert terminal['accepted']
    assert terminal['metadata_callback_sequence']==audit['counters']['full']==store.callback_count
    assert last['repeat_exact_q_callback']
