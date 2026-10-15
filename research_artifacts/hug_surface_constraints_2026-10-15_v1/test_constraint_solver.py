"""Manufactured solver tests only; no research cases or scientific fits."""
from pathlib import Path
from types import SimpleNamespace
import gzip,json,tempfile,unittest
import numpy as np
import constraint_solver as s
from constraint_core import exact_tip_status,diagnostic_rank

CONFIG={'maximum_raw_callback_executions_per_case':3000,'maximum_exact_checks_per_case':90,
 'terminal_raw_and_exact_reserve':2,'maximum_iterations_per_start':40,'surface_difference_step_scaled':1e-4,
 'raw_cache_size':128,'epigraph_slack_upper':1000.,'tip_inward_target_m':1e-9,'optimizer_ftol':1e-9}
class FakeCase:
    index=999
    def __init__(self):
        self.raw_calls=self.exact_calls=self.tip_jacobian_calls=0
        self.q0=np.zeros(24);self.bounds=np.tile([-1.,1.],(27,1));self.limits=np.tile([-1.,1.],(24,1));self.caps=np.full(5,.006)
        self.exact_states=[]
    def raw(self,x):
        self.raw_calls+=1;x=np.array(x,float);e=np.zeros((10,3));e[:5,0]=.005
        margin=np.array([x[24]-.5]);passed=margin>=0
        return {'state_scaled':x,'surface_margin':margin,'task_error_vectors_m':e,'tip_status':exact_tip_status(e[:5],self.caps),'proximity':{'passed':passed}}
    def tip_jacobian(self,x):self.tip_jacobian_calls+=1;return np.zeros((5,3,27))
    def exact(self,x):
        self.exact_calls+=1;v=self.raw(x);self.exact_states.append(np.array(x))
        eligible=np.array([x[25]<=.2]);passed=v['proximity']['passed']&eligible
        v.update(in_domain=True,exact_orientation_fields={'valid':eligible},exact_orientation={'passed':eligible},
            all_witness_constraints_pass=bool(passed.all()),rank=diagnostic_rank(eligible,v['surface_margin'],passed,x))
        return v

class SolverTests(unittest.TestCase):
    def starts(self):return [{'label':'original','state_scaled':np.zeros(27)},{'label':'fixed_second','state_scaled':np.r_[np.zeros(24),.2,0.,0.]}]
    def test_full27_cache_key(self):
        x=np.zeros(27);y=x.copy();y[-1]=np.nextafter(0.,1.)
        self.assertNotEqual(s.state_key(x),s.state_key(y))
    def test_nonfinite_key_rejected(self):
        with self.assertRaises(ValueError):s.state_key(np.full(27,np.nan))
    def test_exact_zero_norm_derivative(self):
        self.assertTrue(np.array_equal(s.norm_jacobian(np.zeros((5,3)),np.ones((5,3,27))),np.zeros((5,27))))
    def test_numerical_zero_and_nonzero_norm_derivative(self):
        e=np.zeros((5,3));e[1,0]=2.;J=np.ones((5,3,27));got=s.norm_jacobian(e,J)
        self.assertTrue(np.array_equal(got[0],np.zeros(27)));self.assertTrue(np.array_equal(got[1],np.ones(27)))
    def test_real_toy_minimax_finds_fixed_witness(self):
        c=FakeCase()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            x,full,a=s.solve(c,self.starts(),CONFIG,d)
            self.assertTrue(full['all_witness_constraints_pass']);self.assertGreaterEqual(x[24],.5)
            self.assertTrue(any(np.array_equal(x,z) for z in c.exact_states));self.assertLessEqual(a['counts']['raw_callback_executions'],3000)
            self.assertEqual(a['counts']['raw_callback_executions'],c.raw_calls)
            events=[json.loads(l) for l in gzip.open(a['trace_path'],'rt')]
            self.assertEqual(len(events),a['counts']['trace_events']);self.assertEqual(events[-1]['kind'],'selected_fresh_terminal')
    def test_raw_probe_cannot_be_selected(self):
        original=s.minimize
        def probe_only(fun,z0,**kw):
            z=z0.copy();z[24]=1.;z[-1]=0.;kw['constraints'][0]['fun'](z)
            return SimpleNamespace(status=9,success=False,message='manufactured raw-probe only',nit=0,nfev=1,njev=0,x=z0.copy())
        s.minimize=probe_only
        try:
            c=FakeCase()
            with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                x,full,a=s.solve(c,self.starts(),CONFIG,d)
                self.assertEqual(x[24],.2);self.assertFalse(full['all_witness_constraints_pass'])
                self.assertFalse(any(z[24]==1 for z in c.exact_states))
        finally:s.minimize=original
    def test_invalid_checkpoint_cannot_win_by_score(self):
        original=s.minimize
        def invalid(fun,z0,**kw):
            z=z0.copy();z[24]=1.;z[25]=.8;z[-1]=0.;kw['callback'](z)
            return SimpleNamespace(status=0,success=True,message='manufactured ineligible state',nit=1,nfev=1,njev=0,x=z)
        s.minimize=invalid
        try:
            c=FakeCase()
            with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                x,full,a=s.solve(c,self.starts(),CONFIG,d)
                self.assertEqual(x[25],0.);self.assertEqual(x[24],.2)
                self.assertTrue(any(z[25]==.8 for z in c.exact_states))
        finally:s.minimize=original
    def test_terminal_fallback_updates_source_metadata(self):
        original=s.minimize
        class ChangedCase(FakeCase):
            def exact(self,x):
                v=super().exact(x)
                # Original baseline,start0,terminal0,start1,terminal1 then
                # selected fresh terminal. Change ONLY this manufactured call.
                if self.exact_calls==6:v['rank']=tuple([99]+list(v['rank'][1:]))
                return v
        def unchanged(fun,z0,**kw):return SimpleNamespace(status=9,success=False,message='fixture',nit=0,nfev=0,njev=0,x=z0.copy())
        s.minimize=unchanged
        try:
            c=ChangedCase()
            with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                x,full,a=s.solve(c,self.starts(),CONFIG,d)
                self.assertTrue(np.array_equal(x,np.zeros(27)))
                self.assertEqual(a['selected_source_exact_checkpoint']['reason'],'fresh_original_fallback')
                self.assertEqual(a['selected_source_exact_checkpoint']['event'],a['actual_returned_fresh_event'])
                self.assertEqual(a['refused_preterminal_selection']['state_scaled'][24],.2)
        finally:s.minimize=original

if __name__=='__main__':unittest.main(verbosity=2)
