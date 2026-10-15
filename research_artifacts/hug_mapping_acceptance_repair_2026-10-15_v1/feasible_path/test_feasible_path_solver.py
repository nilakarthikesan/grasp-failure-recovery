"""Manufactured callback-only cases; never import a simulator or native input."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import copy
import gzip
import json
import unittest
import numpy as np
import feasible_path_solver as s


class FakeCase:
    def __init__(self, mode='normal'):
        self.index=99; self.mode=mode
        self.q0=np.zeros(24); self.limits=np.tile([-.35,.35],(24,1))
        self.bounds=np.tile([-1.,1.],(30,1)); self.domain={'world_displacement_AABB_m':np.tile([-.005,.005],(3,1))}
        self.caps=np.full(5,.005); self.raw_calls=0; self.exact_calls=0; self.tip_jacobian_calls=0
        self.exact_states=[]
    def raw(self,x):
        self.raw_calls+=1; x=np.asarray(x,float)
        errors=np.zeros((10,3)); errors[:5,0]=.001+(x[0]**2 if self.mode=='nonlinear_tips' else 0.)
        surf=np.array([.1,x[0]-.5,.1,.1,.1,.1,.1])
        if self.mode=='nonlinear_hard': surf[0]=.02-x[0]
        return {'state_scaled':x.copy(),'surface_margin':surf,'task_error_vectors_m':errors}
    def exact(self,x):
        self.exact_calls+=1; self.exact_states.append(np.asarray(x).copy()); r=self.raw(x); x=np.asarray(x)
        prox=np.array([not(self.mode=='lost_pass' and x[0]>.02),x[0]>=.5])
        valid=np.array([not(self.mode=='lost_valid' and x[0]>.02)])
        ori=valid.copy(); passed=np.r_[prox,ori]
        rank=[int(not passed.all()),int((~valid).sum()),max(0.,.5-x[0]),int((~passed).sum()),float(x@x)]
        if self.mode=='fresh_mismatch' and self.exact_calls==3: rank[2]+=1.
        r.update(proximity={'passed':prox},exact_orientation={'passed':ori},
                 exact_orientation_fields={'valid':valid},literal_surface_pass=passed,
                 rank=rank,in_domain=s.literal_domain(self,x),all_witness_constraints_pass=bool(passed.all()))
        return r
    def tip_jacobian(self,x):
        self.tip_jacobian_calls+=1; J=np.zeros((5,3,30))
        if self.mode=='nonlinear_tips': J[:,0,0]=2*x[0]
        return J


def model(step):
    d=np.zeros(30); d[0]=step
    return {'step_scaled':d,'epigraph':1.,'usable_finite_trust_step':True,'status':0,
            'success':True,'message':'manufactured direction','iterations':1,'objective':1.,
            'linearized_constraint_min':0.,'linearized_constraints':np.ones(13),'slack_column':np.zeros(13)}


def config(jac=1, raw=200, exact=30):
    c=copy.deepcopy(s.DEFAULT_CONFIGURATION)
    c.update(maximum_analytic_tip_jacobians_per_case=jac,
             maximum_raw_callback_executions_per_case=raw,maximum_exact_checks_per_case=exact)
    return c


def run_fake(case, c=None, step=None, side_effect=None):
    c=config() if c is None else c
    ref=case.exact(np.zeros(30))
    with TemporaryDirectory() as out, patch.dict(s.DEFAULT_CONFIGURATION,c,clear=True):
        if step is not None or side_effect is not None:
            with patch.object(s,'local_minimax_step',return_value=model(step or 0.),side_effect=side_effect):
                x,e,r=s.solve(case,np.zeros(30),ref,c,out)
        else: x,e,r=s.solve(case,np.zeros(30),ref,c,out)
        rows=[json.loads(line) for line in gzip.open(r['trace_path'],'rt')]
    return x,e,r,rows


class Controls(unittest.TestCase):
    def test_actual_linear_qp_makes_feasible_surface_progress(self):
        x,e,r,rows=run_fake(FakeCase(),config(jac=2))
        self.assertGreater(x[0],.49)
        self.assertEqual(r['counts']['accepted_transitions'],2)
        self.assertLess(tuple(e['rank']),tuple(r['start_exact']['rank']))
        self.assertTrue(all(np.diff([a['rank'][2] for a in r['accepted_path']])<0.))
        self.assertEqual([a['event'] for a in rows],list(range(1,len(rows)+1)))

    def test_nonlinear_tip_caps_force_dyadic_backtracking(self):
        x,e,r,rows=run_fake(FakeCase('nonlinear_tips'),step=.25)
        self.assertEqual(x[0],.0625)
        self.assertEqual(r['accepted_path'][1]['dyadic_exponent'],2)
        self.assertTrue(s.exact_tip_status(e,np.full(5,.005))['passed'])
        self.assertTrue(any(a.get('reason')=='original_tip_caps_or_mean' for a in rows))

    def test_raw_hard_violation_refused_before_exact(self):
        case=FakeCase('nonlinear_hard'); x,e,r,rows=run_fake(case,step=.25)
        self.assertEqual(x[0],.015625)
        self.assertTrue(all(a[0]<=.02 for a in case.exact_states))
        self.assertTrue(any(a.get('reason')=='actual_buffered_hard_surface' for a in rows))

    def test_invalid_orientation_cannot_be_accepted(self):
        x,e,r,rows=run_fake(FakeCase('lost_valid'),step=.25)
        self.assertEqual(x[0],.015625)
        self.assertTrue(e['exact_orientation_fields']['valid'].all())
        self.assertTrue(any(a.get('reason')=='exact_gates' for a in rows))

    def test_passing_row_loss_cannot_be_accepted(self):
        x,e,r,rows=run_fake(FakeCase('lost_pass'),step=.25)
        self.assertEqual(x[0],.015625)
        self.assertTrue(e['literal_surface_pass'][0])

    def test_rank_compares_current_not_only_reference(self):
        x,e,r,rows=run_fake(FakeCase(),config(jac=2),side_effect=[model(.125),model(-.0625)])
        self.assertEqual(x[0],.125)
        self.assertEqual(r['counts']['accepted_transitions'],1)
        self.assertTrue(any(a.get('reason')=='no_strict_current_rank_improvement' for a in rows))

    def test_finite_difference_probes_never_selected(self):
        case=FakeCase();x,e,r,rows=run_fake(case,step=0.)
        np.testing.assert_array_equal(x,np.zeros(30))
        self.assertEqual(r['counts']['accepted_transitions'],0)
        self.assertTrue(any(np.any(a.get('state_scaled',[0])) for a in rows if a['kind']=='raw_callback'))
        self.assertTrue(all(np.array_equal(q,np.zeros(30)) for q in case.exact_states))

    def test_fresh_mismatch_falls_back_to_rechecked_start(self):
        # One reference callback occurs before solver; the candidate is callback3,
        # so trigger mismatch at actual final callback4 rather than candidate.
        case=FakeCase()
        original=case.exact
        def exact(x):
            r=original(x)
            if case.exact_calls==4:r['rank'][2]+=1.
            return r
        case.exact=exact
        x,e,r,rows=run_fake(case,step=.125)
        np.testing.assert_array_equal(x,np.zeros(30))
        self.assertIsNotNone(r['refused_terminal_state'])
        self.assertEqual(r['selected_source_exact_event'],r['accepted_path'][0]['event'])
        self.assertEqual(rows[-1]['reason'],'fresh_initial_segment_fallback')

    def test_exact_tip_equality_and_one_ulp_refusal(self):
        raw={'task_error_vectors_m':np.tile([.010,0,0],(5,1))}
        self.assertTrue(s.exact_tip_status(raw,np.full(5,.010))['passed'])
        raw['task_error_vectors_m'][0,0]=np.nextafter(.010,np.inf)
        self.assertFalse(s.exact_tip_status(raw,np.full(5,.010))['passed'])

    def test_zero_norm_jacobian_and_analytic_norm_derivative(self):
        errors=np.zeros((5,3));J=np.ones((5,3,30))
        np.testing.assert_array_equal(s.norm_jacobian(errors,J),np.zeros((5,30)))
        errors[:,0]=1.
        np.testing.assert_array_equal(s.norm_jacobian(errors,J),np.ones((5,30)))

    def test_full_state_key_separates_rotation_and_translation(self):
        x=np.zeros(30);a=s.state_key(x);x[24]=np.nextafter(0.,1.);b=s.state_key(x)
        x=np.zeros(30);x[29]=np.nextafter(0.,1.);c=s.state_key(x)
        self.assertEqual(len({a,b,c}),3)
        with self.assertRaises(ValueError):s.state_key(np.zeros(27))

    def test_domain_rejects_physical_or_translation_escape(self):
        case=FakeCase();x=np.zeros(30);x[0]=1.00000000001
        self.assertFalse(s.literal_domain(case,x))
        case.bounds[:]=[-100,100];x[:]=0;x[24]=2.
        self.assertFalse(s.literal_domain(case,x))
        x[:]=0;x[0]=14.000000001
        self.assertFalse(s.literal_domain(case,x))

    def test_local_slack_column_is_zero_for_protected_rows(self):
        case=FakeCase();ref=case.exact(np.zeros(30));p=s.protected_surface_constraint_policy(ref)
        J=np.zeros((7,30));J[1,0]=1.
        count=(case.raw_calls,case.exact_calls,case.tip_jacobian_calls)
        m=s.local_minimax_step(np.zeros(30),ref,J,np.zeros((5,3,30)),case.bounds,.25,case.caps,p,s.DEFAULT_CONFIGURATION)
        np.testing.assert_array_equal(m['slack_column'][6:],p['slack_coefficients'])
        self.assertEqual(count,(case.raw_calls,case.exact_calls,case.tip_jacobian_calls))

    def test_exact_budget_reserves_fresh_return(self):
        x,e,r,rows=run_fake(FakeCase('lost_valid'),config(jac=3,exact=5),step=.25)
        self.assertEqual(r['termination'],'exact_check_budget')
        self.assertLessEqual(r['counts']['exact_checks'],5)
        np.testing.assert_array_equal(x,np.zeros(30))
        self.assertEqual(rows[-1]['reason'],'fresh_returned_accepted_state')

    def test_bad_local_optimizer_results_produce_finite_refusal(self):
        from types import SimpleNamespace
        case=FakeCase();ref=case.exact(np.zeros(30));p=s.protected_surface_constraint_policy(ref)
        for z,objective in [(np.full(31,np.nan),np.nan),(np.zeros(30),0.),(np.zeros(31),np.inf)]:
            bad=SimpleNamespace(x=z,fun=objective,status=9,success=False,message='bad',nit=3)
            with patch.object(s,'minimize',return_value=bad):
                m=s.local_minimax_step(np.zeros(30),ref,np.zeros((7,30)),np.zeros((5,3,30)),case.bounds,.25,case.caps,p,s.DEFAULT_CONFIGURATION)
            self.assertFalse(m['usable_finite_trust_step'])
            self.assertIsNone(m['linearized_constraint_min'])
            json.dumps(s.plain(m),allow_nan=False)

    def test_nonfinite_rank_or_hard_margin_refused(self):
        with self.assertRaises(ValueError):s.exact_rank({'rank':[0,0,np.nan,0,0]})
        case=FakeCase();ref=case.exact(np.zeros(30));p=s.protected_surface_constraint_policy(ref)
        ref['surface_margin'][0]=np.nan
        with self.assertRaises(ValueError):s.hard_raw_status(ref,p)

if __name__=='__main__':unittest.main(verbosity=2)
