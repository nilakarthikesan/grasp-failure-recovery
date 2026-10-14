"""Manufactured static constraints: no simulator, generator, outcome or eval."""
from types import SimpleNamespace
import importlib
import json

import numpy as np
import pytest

api = importlib.import_module("grasp_failure_prediction.evaluation.constrained_surface_refinement")
from grasp_failure_prediction.evaluation.constrained_surface_refinement import (
    ConstrainedEvaluation,
    ConstrainedRefinementConfig,
    native_direction_residual,
    solve_constrained_refinement,
)


Q0 = np.zeros(24)
LIMITS = np.tile([-1.0, 1.0], (24, 1))
BASE_ERRORS = np.full(5, 0.005)


def problem(target=None, baseline_errors=BASE_ERRORS):
    desired = np.zeros(24) if target is None else np.array(target, copy=True)
    base = np.array(baseline_errors, copy=True)
    def errors(q):
        value = np.zeros((5, 3))
        value[:, 0] = base + 0.010 * q[:5]
        return value
    def evaluate(q):
        assert not q.flags.writeable
        return ConstrainedEvaluation(q - desired, errors(q))
    return evaluate, errors


def mock_result(monkeypatch, x, *, success=True, status=0, nit=1):
    def minimize(*args, **kwargs):
        assert kwargs["method"] == "SLSQP"
        assert kwargs["options"]["maxiter"] == 120
        assert kwargs["options"]["eps"] == 1e-5
        assert kwargs["options"]["ftol"] == 1e-9
        np.testing.assert_array_equal(args[1], Q0)
        return SimpleNamespace(x=np.array(x, copy=True), success=success,
                               status=status, message="manufactured result",
                               nit=nit, nfev=2, njev=1)
    monkeypatch.setattr(api, "minimize", minimize)


def test_joint_solve_improves_available_surface_motion_without_worsening_any_tip():
    target = np.zeros(24)
    target[[0, 5]] = 0.2
    evaluate, errors = problem(target)
    q, report = solve_constrained_refinement(Q0, LIMITS, evaluate,
                                             evaluate_fingertips=errors)
    assert report["surface_candidate_returned"]
    assert q[5] > 0.19
    assert np.all(np.linalg.norm(errors(q), axis=1) <= BASE_ERRORS + 1e-8)
    assert q[0] <= 1e-6
    assert report["final_assessment"]["objective"] < report["baseline"]["objective"]
    assert report["optimized_joint_count"] == 24
    assert report["mapping_fidelity_acceptance"] is False
    assert report["full_FK_callback_agreement_checks"] >= 1
    assert report["full_FK_callback_agreement_max_error_m"] == 0.0


def test_direction_can_improve_without_spending_another_fingers_error_budget():
    def errors(q):
        values = np.zeros((5, 3))
        values[:, 0] = BASE_ERRORS + 0.010 * q[:5]
        return values
    def evaluate(q):
        # Increasing q[0] would improve facing but worsen finger zero. A
        # simultaneous alternative q[5] adjustment preserves all five tips.
        score = np.clip(0.5 - 4.0 * q[0] + 4.0 * q[5], -1.0, 1.0)
        direction = native_direction_residual([-0.5], [score], [1.0], [True])
        return ConstrainedEvaluation(np.r_[direction, 0.02 * q], errors(q))
    q, report = solve_constrained_refinement(Q0, LIMITS, evaluate,
                                             evaluate_fingertips=errors)
    assert report["surface_candidate_returned"]
    assert np.linalg.norm(evaluate(q).residuals) < np.linalg.norm(evaluate(Q0).residuals)
    assert 0.5 - 4.0 * q[0] + 4.0 * q[5] < 0.0
    assert np.all(np.linalg.norm(errors(q), axis=1) <= BASE_ERRORS + 1e-8)
    assert not report["mapping_fidelity_acceptance"]


def test_lower_cost_raw_result_is_refused_despite_acceptable_mean(monkeypatch):
    raw = Q0.copy()
    raw[0], raw[1], raw[5] = 0.1, -0.1, 0.2
    evaluate, errors = problem(raw)
    assert np.linalg.norm(errors(raw), axis=1).mean() == pytest.approx(0.005)
    mock_result(monkeypatch, raw)
    q, report = solve_constrained_refinement(Q0, LIMITS, evaluate)
    np.testing.assert_array_equal(q, Q0)
    assert report["raw_assessment"]["objective"] < report["baseline"]["objective"]
    assert report["raw_assessment"]["exact_existing_mean_guard_pass"]
    assert not report["raw_assessment"]["allowance_individual_tip_nonworsening_pass"][0]
    assert "individual_native_relative_tip_constraint_violated" in report["refusal_reasons"]
    assert report["exact_original_fallback"]


def test_existing_mean_guard_is_strict_even_within_tip_arithmetic_allowance(monkeypatch):
    raw = Q0.copy()
    raw[:5] = 5e-7
    raw[5] = 0.2
    evaluate, _ = problem(raw, baseline_errors=np.full(5, 0.010))
    mock_result(monkeypatch, raw)
    q, report = solve_constrained_refinement(Q0, LIMITS, evaluate)
    np.testing.assert_array_equal(q, Q0)
    assert all(report["raw_assessment"]["allowance_individual_tip_nonworsening_pass"])
    assert not report["raw_assessment"]["allowance_existing_mean_guard_pass"]
    assert "existing_mean_fingertip_guard_violated" in report["refusal_reasons"]


def test_tip_roundoff_allowance_is_reported_as_conditional_not_exact_pass(monkeypatch):
    raw = Q0.copy()
    raw[0], raw[5] = 5e-7, 0.2
    evaluate, _ = problem(raw)
    mock_result(monkeypatch, raw)
    q, report = solve_constrained_refinement(Q0, LIMITS, evaluate)
    np.testing.assert_array_equal(q, raw)
    assert report["surface_candidate_returned"]
    assert not report["raw_assessment"]["exact_individual_tip_nonworsening_pass"][0]
    assert all(report["raw_assessment"]["allowance_individual_tip_nonworsening_pass"])
    assert report["final_feasibility_is_conditional_on_engineering_arithmetic_allowance"]
    assert not report["mapping_fidelity_acceptance"]


@pytest.mark.parametrize("raw_value, bounds", [(0.4, LIMITS), (0.3, np.tile([-0.2, 0.2], (24, 1)))])
def test_solver_result_outside_trust_or_physical_bounds_is_never_clipped_into_acceptance(monkeypatch, raw_value, bounds):
    raw = Q0.copy()
    raw[5] = raw_value
    evaluate, _ = problem(raw)
    mock_result(monkeypatch, raw)
    q, report = solve_constrained_refinement(Q0, bounds, evaluate)
    np.testing.assert_array_equal(q, Q0)
    assert report["raw_assessment"] is None
    assert report["refusal_reasons"] == ["raw_candidate_outside_physical_or_trust_bounds"]


def test_feasible_candidate_without_lower_objective_returns_exact_baseline(monkeypatch):
    raw = Q0.copy()
    raw[0] = -0.1
    evaluate, _ = problem()
    mock_result(monkeypatch, raw)
    q, report = solve_constrained_refinement(Q0, LIMITS, evaluate)
    np.testing.assert_array_equal(q, Q0)
    assert report["raw_assessment"]["allowance_engineering_feasible"]
    assert report["refusal_reasons"] == ["frozen_surface_objective_not_improved"]


def test_feasible_lower_cost_budget_exhausted_output_is_honestly_labeled(monkeypatch):
    raw = Q0.copy()
    raw[5] = 0.2
    evaluate, _ = problem(raw)
    mock_result(monkeypatch, raw, success=False, status=9, nit=120)
    q, report = solve_constrained_refinement(Q0, LIMITS, evaluate)
    np.testing.assert_array_equal(q, raw)
    assert report["surface_candidate_returned"]
    assert report["budget_exhausted"]
    assert not report["solver_success"]
    assert report["candidate_returned_despite_solver_nonconvergence"]
    assert not report["mapping_fidelity_acceptance"]


def test_invalid_solver_raw_target_is_refused_and_audit_is_json_serializable(monkeypatch):
    raw = Q0.copy()
    raw[5] = np.nan
    evaluate, _ = problem()
    mock_result(monkeypatch, raw)
    q, report = solve_constrained_refinement(Q0, LIMITS, evaluate)
    np.testing.assert_array_equal(q, Q0)
    assert report["refusal_reasons"] == ["raw_candidate_nonfinite_or_invalid_shape"]
    assert report["raw_candidate_qpos_rad"] is None
    json.dumps(report, allow_nan=False)


def test_input_arrays_and_callback_snapshots_are_independent():
    target = Q0.copy()
    target[5] = 0.2
    evaluate, errors = problem(target)
    q0, bounds = Q0.copy(), LIMITS.copy()
    def checked(q):
        assert not q.flags.writeable
        with pytest.raises(ValueError):
            q[0] = 999.0
        return evaluate(q)
    q, _ = solve_constrained_refinement(q0, bounds, checked, evaluate_fingertips=errors)
    np.testing.assert_array_equal(q0, Q0)
    np.testing.assert_array_equal(bounds, LIMITS)
    q[:] = 999.0
    np.testing.assert_array_equal(q0, Q0)


def test_callback_that_reenables_writing_and_mutates_q_is_refused():
    def altered(q):
        q.setflags(write=True)
        q[5] = 0.1
        return ConstrainedEvaluation(q, np.zeros((5, 3)))
    with pytest.raises(ValueError, match="modified its joint argument"):
        solve_constrained_refinement(Q0, LIMITS, altered)


def test_cached_evaluation_container_copies_source_arrays():
    residual = np.ones(3)
    errors = np.ones((5, 3))
    value = ConstrainedEvaluation(residual, errors)
    residual[:] = 999.0
    errors[:] = 999.0
    np.testing.assert_array_equal(value.residuals, np.ones(3))
    np.testing.assert_array_equal(value.fingertip_error_vectors_m, np.ones((5, 3)))
    assert not value.residuals.flags.writeable
    assert not value.fingertip_error_vectors_m.flags.writeable


def test_later_invalid_geometry_refuses_candidate_and_preserves_baseline(monkeypatch):
    evaluate, _ = problem()
    def sometimes_invalid(q):
        if q[5] > 0.05:
            return ConstrainedEvaluation([np.nan], np.zeros((5, 3)))
        return evaluate(q)
    def minimize(fun, initial, **kwargs):
        trial = initial.copy()
        trial[5] = 0.1
        fun(trial)
        raise AssertionError("invalid geometry should have raised")
    monkeypatch.setattr(api, "minimize", minimize)
    q, report = solve_constrained_refinement(Q0, LIMITS, sometimes_invalid)
    np.testing.assert_array_equal(q, Q0)
    assert report["exact_original_fallback"]
    assert report["refused_geometry_evaluations"]
    assert report["refusal_reasons"] == ["optimizer_or_geometry_evaluation_refused"]


def test_changed_residual_domain_is_refused_instead_of_changing_the_objective(monkeypatch):
    evaluate, _ = problem()
    def changed(q):
        if q[5] > 0.05:
            return ConstrainedEvaluation([0.0], np.zeros((5, 3)))
        return evaluate(q)
    def minimize(fun, initial, **kwargs):
        trial = initial.copy()
        trial[5] = 0.1
        fun(trial)
    monkeypatch.setattr(api, "minimize", minimize)
    q, report = solve_constrained_refinement(Q0, LIMITS, changed)
    np.testing.assert_array_equal(q, Q0)
    assert "residual shape changed" in report["optimizer_exception"]


def test_FK_only_callback_is_used_and_agreement_is_enforced(monkeypatch):
    evaluate, errors = problem()
    full_calls, cheap_calls = [], []
    def full(q):
        full_calls.append(q.copy())
        return evaluate(q)
    def cheap(q):
        cheap_calls.append(q.copy())
        assert not q.flags.writeable
        return errors(q)
    def minimize(fun, initial, **kwargs):
        trial = initial.copy()
        trial[5] = 0.1
        kwargs["constraints"][0]["fun"](trial)
        assert len(full_calls) == 1
        assert len(cheap_calls) == 2  # baseline plus constraint-only trial
        fun(trial)
        return SimpleNamespace(x=initial.copy(), success=True, status=0,
                               message="manufactured no improvement", nit=1, nfev=1, njev=1)
    monkeypatch.setattr(api, "minimize", minimize)
    _, report = solve_constrained_refinement(Q0, LIMITS, full, evaluate_fingertips=cheap)
    assert report["full_FK_callback_agreement_checks"] == 2
    assert report["full_FK_callback_agreement_max_error_m"] == 0.0


def test_disagreeing_baseline_FK_only_callback_is_refused():
    evaluate, errors = problem()
    def wrong(q):
        result = errors(q)
        result[0, 0] += 1e-6
        return result
    with pytest.raises(ValueError, match="callbacks disagree"):
        solve_constrained_refinement(Q0, LIMITS, evaluate, evaluate_fingertips=wrong)


def test_candidate_full_FK_disagreement_is_recorded_and_falls_back(monkeypatch):
    evaluate, errors = problem()
    def wrong(q):
        result = errors(q)
        if q[5] > 0.05:
            result[0, 0] += 1e-6
        return result
    def minimize(fun, initial, **kwargs):
        trial = initial.copy()
        trial[5] = 0.1
        kwargs["constraints"][0]["fun"](trial)
        fun(trial)
    monkeypatch.setattr(api, "minimize", minimize)
    q, report = solve_constrained_refinement(Q0, LIMITS, evaluate, evaluate_fingertips=wrong)
    np.testing.assert_array_equal(q, Q0)
    assert "callbacks disagree" in report["optimizer_exception"]
    assert report["full_FK_callback_agreement_max_error_m"] > 1e-12


def test_feasible_major_iterate_survives_later_infeasible_final_result(monkeypatch):
    target = Q0.copy()
    target[5] = 0.3
    evaluate, _ = problem(target)
    earlier, best, probe, raw = [Q0.copy() for _ in range(4)]
    earlier[5], best[5], probe[5] = 0.1, 0.2, 0.3
    raw[0], raw[5] = 0.1, 0.3  # Raw surface goal fits better but harms a tip.
    def minimize(fun, initial, **kwargs):
        kwargs["callback"](earlier)
        fun(probe)  # The lower-cost finite-difference/line-search probe is ineligible.
        kwargs["callback"](best)
        return SimpleNamespace(x=raw, success=False, status=9, message="budget",
                               nit=120, nfev=4, njev=1)
    monkeypatch.setattr(api, "minimize", minimize)
    q, report = solve_constrained_refinement(Q0, LIMITS, evaluate)
    np.testing.assert_array_equal(q, best)
    assert report["selected_candidate_source"] == "major_iteration"
    assert report["selected_major_iteration"] == 2
    assert report["raw_final_refusal_reasons"] == ["individual_native_relative_tip_constraint_violated"]
    assert report["selection_refusal_reasons"] == []
    assert report["candidate_returned_despite_solver_nonconvergence"]
    assert len(report["major_iteration_assessments"]) == 2
    assert not report["mapping_fidelity_acceptance"]


def test_lower_feasible_final_result_can_replace_a_feasible_major_iterate(monkeypatch):
    raw, earlier = Q0.copy(), Q0.copy()
    raw[5], earlier[5] = 0.2, 0.1
    evaluate, _ = problem(raw)
    def minimize(fun, initial, **kwargs):
        kwargs["callback"](earlier)
        return SimpleNamespace(x=raw, success=True, status=0, message="done",
                               nit=2, nfev=3, njev=1)
    monkeypatch.setattr(api, "minimize", minimize)
    q, report = solve_constrained_refinement(Q0, LIMITS, evaluate)
    np.testing.assert_array_equal(q, raw)
    assert report["selected_candidate_source"] == "raw_final"
    assert report["selected_major_iteration"] is None
    assert report["raw_final_eligible_lower_objective_candidate"]


@pytest.mark.parametrize("initial, bounds", [
    (np.zeros(23), LIMITS), (np.full(24, np.nan), LIMITS),
    (np.full(24, 1.0 + 5e-10), LIMITS), (Q0, np.ones((24, 2))),
    (Q0, np.zeros((23, 2))),
])
def test_invalid_initial_contract_is_refused_before_callbacks(initial, bounds):
    calls = []
    def evaluate(q):
        calls.append(q)
        return ConstrainedEvaluation(q, np.zeros((5, 3)))
    with pytest.raises(ValueError):
        solve_constrained_refinement(initial, bounds, evaluate)
    assert not calls


@pytest.mark.parametrize("residuals, errors", [
    ([], np.zeros((5, 3))), ([np.nan], np.zeros((5, 3))),
    (np.zeros((2, 2)), np.zeros((5, 3))), ([1.0], np.zeros((4, 3))),
    ([1.0], np.full((5, 3), np.inf)),
])
def test_invalid_geometry_container_is_rejected(residuals, errors):
    with pytest.raises(ValueError):
        ConstrainedEvaluation(residuals, errors)


def test_invalid_callback_contract_is_refused_explicitly():
    with pytest.raises(ValueError, match="must be callable"):
        solve_constrained_refinement(Q0, LIMITS, None)
    with pytest.raises(ValueError, match="return ConstrainedEvaluation"):
        solve_constrained_refinement(Q0, LIMITS, lambda q: (q, np.zeros((5, 3))))


def test_numpy_config_values_are_normalized_for_json_audit():
    conf = ConstrainedRefinementConfig(maxiter=np.int64(120), eps=np.float64(1e-5))
    assert type(conf.maxiter) is int
    assert type(conf.eps) is float


def test_native_away_targets_and_invalid_direction_sites_are_not_dropped():
    native = np.array([-0.5, 0.25])
    candidate = np.array([-0.5, np.nan])
    weights = np.array([0.25, 0.75])
    residual = native_direction_residual(native, candidate, weights, [True, False])
    assert residual.shape == native.shape
    assert residual[0] == 0.0
    assert residual[1] == pytest.approx(2.0 * np.sqrt(0.75))
    introduced_facing = native_direction_residual([-0.5], [0.5], [1.0], [True])
    assert introduced_facing[0] == 1.0
    np.testing.assert_array_equal(native, [-0.5, 0.25])


def test_cosine_endpoint_arithmetic_allowance_does_not_change_direction_sign_targets():
    score = -1.0 - 5e-13
    residual = native_direction_residual([score], [score], [1.0], [True])
    assert residual[0] == 0.0


@pytest.mark.parametrize("native,candidate,weights,valid", [
    ([], [], [], []), ([np.nan], [0.0], [1.0], [True]),
    ([0.0], [np.nan], [1.0], [True]), ([0.0], [0.0], [0.0], [True]),
    ([0.0], [0.0], [0.5], [True]), ([0.0], [0.0], [1.0, 1.0], [True]),
    ([1.1], [0.0], [1.0], [True]), ([0.0], [-1.1], [1.0], [True]),
])
def test_invalid_fixed_direction_contract_is_rejected(native, candidate, weights, valid):
    with pytest.raises(ValueError):
        native_direction_residual(native, candidate, weights, valid)


@pytest.mark.parametrize("change", [
    {"ftol": True}, {"eps": 0.0}, {"maxiter": False},
    {"maxiter": 1.5}, {"constraint_scale_m": np.inf},
    {"feasibility_error_allowance_m": -1.0}, {"evaluation_cache_size": 0},
    {"fingertip_callback_agreement_allowance_m": np.nan},
])
def test_invalid_solver_config_is_rejected(change):
    with pytest.raises(ValueError):
        ConstrainedRefinementConfig(**change)
