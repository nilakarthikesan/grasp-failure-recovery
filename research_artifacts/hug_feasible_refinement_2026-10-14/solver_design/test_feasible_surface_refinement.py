"""Manufactured mathematics only: no saved HUG proposals, simulator or data."""
from dataclasses import replace

import numpy as np
import pytest

from feasible_surface_refinement import FeasibleRefinementConfig, solve_feasible_refinement
from grasp_failure_prediction.evaluation.constrained_surface_refinement import ConstrainedEvaluation


BASE = FeasibleRefinementConfig(maximum_pattern_sweeps=2, maximum_full_geometry_evaluations=10,
    maximum_FK_evaluations=1500, maximum_restorations=20, restoration_maxiter=20)
Q0 = np.zeros(24)
LIMITS = np.tile([-1., 1.], (24, 1))


def curved_callbacks(*, goal=.05, disagreement=False, change_shape=False):
    seen_full = []
    def tips(q):
        errors = np.zeros((5, 3))
        errors[:, 0] = .005 + .01*q[0]
        errors[:, 1] = .01*q[1]
        return errors
    def full(q):
        seen_full.append(q.copy())
        errors = tips(q)
        if disagreement and q[1] != 0:
            errors[0, 0] += 1e-6
        residual = [100.*(q[1]-goal)]
        if change_shape and q[1] != 0:
            residual.append(0.)
        return ConstrainedEvaluation(residual, errors)
    return full, tips, seen_full


def run(full, tips, config=BASE, **kwargs):
    return solve_feasible_refinement(Q0, LIMITS, full, evaluate_fingertips=tips, config=config, **kwargs)


def test_curved_feasible_manifold_restores_tangent_steps_and_improves():
    full, tips, seen = curved_callbacks()
    q, report = run(full, tips)
    assert report['surface_candidate_returned'] and q[1] > 0
    assert q[0] < 0  # curvature restoration, not just trusting a tangent line
    assert report['final']['objective'] < report['baseline']['objective']
    assert report['final']['exact_computed_engineering_feasible']
    assert all(np.linalg.norm(tips(p), axis=1).max() <= .005 for p in seen)
    assert report['counters']['restorations'] > 0
    assert report['feasibility_error_allowance_m'] == 0
    assert not report['mapping_fidelity_acceptance']


def test_seed_only_projects_an_infeasible_nonlinear_seed_then_improves():
    full, tips, seen = curved_callbacks()
    raw = np.zeros(24); raw[1] = .05
    assert np.linalg.norm(tips(raw), axis=1).max() > .005
    q, report = run(full, tips, replace(BASE, maximum_pattern_sweeps=0,
                                      maximum_full_geometry_evaluations=4),
                    predeclared_seed_qpos_rad=[raw])
    assert report['surface_candidate_returned'] and q[1] > 0 and q[0] < 0
    assert report['final']['exact_computed_engineering_feasible']
    assert report['final']['objective'] < report['baseline']['objective']
    assert report['termination'] == 'predeclared_seed_only_stage_complete_no_certificate'
    assert len(report['proposals']) == 1 and report['proposals'][0]['sweep'] == 0
    assert report['proposals'][0]['restoration'] == 'cheap_FK_SLSQP_projection'
    assert not report['proposals'][0]['proposal_assessment']['exact_computed_engineering_feasible']
    assert all(np.linalg.norm(tips(p), axis=1).max() <= .005 for p in seen)
    assert len(seen) == 3  # original, projected candidate, fresh final
    assert np.array_equal(raw[1:], np.r_[.05, np.zeros(22)])


def test_seed_only_no_seed_falls_back_with_fresh_original():
    full, tips, seen = curved_callbacks()
    q, report = run(full, tips, replace(BASE, maximum_pattern_sweeps=0))
    assert np.array_equal(q, Q0) and len(seen) == 2
    assert report['proposals'] == []


def test_changed_fresh_final_surface_objective_forces_fresh_original_fallback():
    _, tips, _ = curved_callbacks()
    calls = []
    def full(q):
        calls.append(q.copy())
        residual = 100.*(q[1]-.05)
        if len(calls) == 3:
            residual += 100.  # changed fresh selected geometry callback
        return ConstrainedEvaluation([residual], tips(q))
    raw = np.zeros(24); raw[1] = .05
    q, report = run(full, tips, replace(BASE, maximum_pattern_sweeps=0,
                                      maximum_full_geometry_evaluations=4),
                    predeclared_seed_qpos_rad=[raw])
    assert np.array_equal(q, Q0) and np.array_equal(calls[-1], Q0)
    assert len(calls) == 4 and report['counters']['full'] == 4
    assert report['termination'] == 'fresh_final_verification_refusal'
    assert 'fresh final objective' in report['callback_or_geometry_refusal']
    assert report['final']['objective'] == report['baseline']['objective']


@pytest.mark.parametrize('seed', [np.ones(24), [np.full(24, np.nan)], [np.ones(24)]])
def test_invalid_or_outside_seed_is_rejected_before_callbacks(seed):
    full, tips, seen = curved_callbacks()
    with pytest.raises(ValueError, match='predeclared seeds'):
        run(full, tips, predeclared_seed_qpos_rad=seed)
    assert seen == []


def test_new_method_has_explicit_hard_budget_and_fresh_terminal():
    full, tips, seen = curved_callbacks()
    q, report = run(full, tips, replace(BASE, maximum_full_geometry_evaluations=4))
    assert len(seen) == report['counters']['full'] <= 4
    assert report['counters']['FK'] <= BASE.maximum_FK_evaluations
    assert report['final_callbacks_fresh_outside_cache']
    calls = [e for e in report['events'] if e['kind'] == 'full_callback']
    assert calls[-1]['fresh'] and calls[-1]['terminal']
    assert np.array_equal(np.array(calls[-1]['qpos_rad']), q)


def test_no_cost_improvement_returns_exact_original():
    _, tips, _ = curved_callbacks()
    full = lambda q: ConstrainedEvaluation([0.], tips(q))
    q, report = run(full, tips)
    assert np.array_equal(q, Q0) and report['exact_original_fallback']
    assert report['final']['exact_computed_engineering_feasible']


def test_zero_native_tip_cap_never_receives_an_acceptance_allowance():
    seen = []
    def tips(q):
        errors = np.zeros((5, 3)); errors[:, 0] = q[0]; return errors
    def full(q):
        seen.append(q.copy()); return ConstrainedEvaluation([q[0]-.1], tips(q))
    q, report = run(full, tips, replace(BASE, maximum_restorations=5))
    assert q[0] == 0
    assert report['final']['exact_computed_engineering_feasible']
    assert all(p[0] == 0 for p in seen)


def test_fk_full_disagreement_discards_any_new_pose():
    full, tips, _ = curved_callbacks(disagreement=True)
    q, report = run(full, tips)
    assert np.array_equal(q, Q0)
    assert 'disagree' in report['callback_or_geometry_refusal']
    assert report['final']['exact_computed_engineering_feasible']


def test_frozen_residual_shape_change_is_a_refusal():
    full, tips, _ = curved_callbacks(change_shape=True)
    q, report = run(full, tips)
    assert np.array_equal(q, Q0)
    assert 'shape' in report['callback_or_geometry_refusal']


def test_strict_mean_guard_original_infeasibility_reported_without_search():
    tips = lambda q: np.tile([.012, 0., 0.], (5, 1))
    full = lambda q: ConstrainedEvaluation([1.], tips(q))
    q, report = run(full, tips)
    assert np.array_equal(q, Q0)
    assert report['termination'] == 'original_mean_guard_infeasible_no_search'
    assert not report['final']['exact_computed_engineering_feasible']
    assert report['proposals'] == []


def test_physical_and_original_delta_bounds_enforced_before_every_callback():
    full, tips, _ = curved_callbacks()
    physical = LIMITS.copy(); physical[:, 0] = -.002; physical[:, 1] = .002
    seen = []
    def checked_tips(q):
        assert np.all(q >= -.002) and np.all(q <= .002)
        assert np.max(np.abs(q)) <= BASE.joint_delta_limit_rad
        seen.append(q.copy()); return tips(q)
    q, report = solve_feasible_refinement(Q0, physical, full, evaluate_fingertips=checked_tips, config=BASE)
    assert np.all(q >= -.002) and np.all(q <= .002) and seen
    assert np.array_equal(physical[:, 0], np.full(24, -.002))


def test_callers_arrays_preserved_and_callback_arguments_readonly():
    full, tips, _ = curved_callbacks()
    initial, bounds = Q0.copy(), LIMITS.copy()
    def readonly(q):
        assert not q.flags.writeable
        return tips(q)
    solve_feasible_refinement(initial, bounds, full, evaluate_fingertips=readonly, config=BASE)
    assert np.array_equal(initial, Q0) and np.array_equal(bounds, LIMITS)


def test_callback_making_argument_writable_and_mutating_is_refused():
    def mutation(q):
        q.setflags(write=True); q[0] += 1; return np.zeros((5, 3))
    with pytest.raises(ValueError, match='modified'):
        run(lambda q: ConstrainedEvaluation([0.], np.zeros((5, 3))), mutation)


def test_full_callback_resetting_readonly_and_mutating_is_refused():
    _, tips, _ = curved_callbacks()
    def full(q):
        assert not q.flags.writeable
        q.setflags(write=True); q[0] += .01
        return ConstrainedEvaluation([1.], tips(q))
    with pytest.raises(ValueError, match='full callback modified'):
        run(full, tips)


@pytest.mark.parametrize('bad', [np.ones((4, 3)), np.full((5, 3), np.nan)])
def test_invalid_fk_output_refused(bad):
    with pytest.raises(ValueError, match='five finite'):
        run(lambda q: ConstrainedEvaluation([0.], np.zeros((5, 3))), lambda q: bad)


def test_deterministic_manufactured_search():
    full1, tips1, _ = curved_callbacks()
    full2, tips2, _ = curved_callbacks()
    q1, r1 = run(full1, tips1)
    q2, r2 = run(full2, tips2)
    assert np.array_equal(q1, q2)
    assert r1 == r2


@pytest.mark.parametrize('kwargs', [dict(maximum_full_geometry_evaluations=2),
    dict(maximum_FK_evaluations=2), dict(restoration_inward_target_m=0),
    dict(restoration_inward_target_m=.010),
    dict(maximum_pattern_sweeps=True), dict(f_scale=np.nan)])
def test_bad_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        FeasibleRefinementConfig(**kwargs)
