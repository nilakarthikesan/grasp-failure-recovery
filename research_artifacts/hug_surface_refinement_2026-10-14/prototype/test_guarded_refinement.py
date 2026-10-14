"""Engineering-guard tests, deliberately separate from fidelity acceptance."""
import numpy as np
import pytest

from grasp_failure_prediction.evaluation.surface_refinement_guard import (
    GuardConfig,
    guarded_joint_selection,
)


BOUNDS = np.array([[-1.0, 1.0], [-1.0, 1.0]])
BASELINE = np.array([0.0, 0.0])
RAW = np.array([0.2, -0.1])


def assessment(objective, tip_error):
    return {"objective": objective, "mean_fingertip_error_m": tip_error}


def test_largest_fixed_alpha_satisfying_both_independent_conditions_is_selected():
    seen = []
    def assess(q):
        alpha = q[0] / RAW[0]
        seen.append(alpha)
        if alpha == 0.0:
            return assessment(1.0, 0.006)
        return assessment(1.0 - alpha, 0.006 + 0.008 * alpha)
    q, report = guarded_joint_selection(BASELINE, RAW, BOUNDS, assess)
    np.testing.assert_allclose(q, BASELINE + 0.5 * (RAW - BASELINE))
    np.testing.assert_allclose(seen, [0.0, 1.0, 0.5])
    assert report["selected_alpha"] == 0.5
    assert report["surface_candidate_returned"]
    assert not report["mapping_fidelity_acceptance"]
    rejected, selected = report["attempted_alphas"]
    assert rejected["objective_improved"]
    assert not rejected["passed_existing_engineering_guard"]
    assert rejected["reasons"] == ["existing_mean_fingertip_error_guard_exceeded"]
    assert selected["accepted"]


def test_lower_geometric_objective_alone_cannot_bypass_fingertip_guard():
    def assess(q):
        alpha = q[0] / RAW[0]
        return assessment(1.0 - alpha, 0.006 if alpha == 0.0 else 0.011)
    q, report = guarded_joint_selection(BASELINE, RAW, BOUNDS, assess)
    np.testing.assert_array_equal(q, BASELINE)
    assert report["selected_alpha"] == 0.0
    assert not report["surface_candidate_returned"]
    assert len(report["attempted_alphas"]) == len(GuardConfig().alphas)
    assert all(row["objective_improved"] for row in report["attempted_alphas"][:-1])
    assert all(not row["passed_existing_engineering_guard"] for row in report["attempted_alphas"][:-1])


def test_acceptable_fingertips_alone_cannot_bypass_frozen_objective_condition():
    def assess(q):
        return assessment(1.0, 0.001)
    q, report = guarded_joint_selection(BASELINE, RAW, BOUNDS, assess)
    np.testing.assert_array_equal(q, BASELINE)
    assert report["selected_alpha"] == 0.0
    assert all(row["passed_existing_engineering_guard"] for row in report["attempted_alphas"])
    assert all(not row["objective_improved"] for row in report["attempted_alphas"])
    assert report["attempted_alphas"][-1]["reasons"] == ["original_mapping_retained"]


def test_baseline_above_tip_guard_is_retained_with_explicit_failed_guard():
    # Fallback is a refusal to replace the pose, not approval of a baseline
    # which already violates the existing engineering condition.
    def assess(q):
        return assessment(1.0, 0.020)
    q, report = guarded_joint_selection(BASELINE, RAW, BOUNDS, assess)
    np.testing.assert_array_equal(q, BASELINE)
    assert report["attempted_alphas"][-1]["passed_existing_engineering_guard"] is False
    assert report["mapping_fidelity_acceptance"] is False


def test_exact_tip_guard_boundary_is_eligible_but_tiny_objective_change_is_not():
    conf = GuardConfig(minimum_objective_improvement=0.01)
    def assess(q):
        alpha = q[0] / RAW[0]
        return assessment(1.0 if alpha == 0 else 0.995, 0.010)
    _, report = guarded_joint_selection(BASELINE, RAW, BOUNDS, assess, conf)
    assert report["selected_alpha"] == 0.0
    assert all(row["passed_existing_engineering_guard"] for row in report["attempted_alphas"])
    assert all(not row["objective_improved"] for row in report["attempted_alphas"])


def test_joint_inputs_and_bounds_are_preserved_and_result_is_independent():
    q0, qr, limits = BASELINE.copy(), RAW.copy(), BOUNDS.copy()
    def assess(q):
        return assessment(1.0 - q[0], 0.006)
    selected, _ = guarded_joint_selection(q0, qr, limits, assess)
    np.testing.assert_array_equal(q0, BASELINE)
    np.testing.assert_array_equal(qr, RAW)
    np.testing.assert_array_equal(limits, BOUNDS)
    selected[:] = 999.0
    np.testing.assert_array_equal(q0, BASELINE)
    np.testing.assert_array_equal(qr, RAW)


def test_assessor_receives_readonly_target_copy_and_reused_dictionary_is_snapshotted():
    shared = {}
    def assess(q):
        assert not q.flags.writeable
        with pytest.raises(ValueError):
            q[:] = 999.0
        alpha = q[0] / RAW[0]
        shared.clear()
        shared.update(assessment(1.0 - alpha, 0.006 + 0.008 * alpha))
        shared["nested"] = {"alpha": alpha}
        return shared
    q, report = guarded_joint_selection(BASELINE, RAW, BOUNDS, assess)
    np.testing.assert_allclose(q, 0.5 * RAW)
    assert report["baseline"]["objective"] == 1.0
    assert report["baseline"]["nested"]["alpha"] == 0.0
    assert report["attempted_alphas"][0]["assessment"]["objective"] == 0.0
    shared["nested"]["alpha"] = 999.0
    assert report["attempted_alphas"][-1]["assessment"]["nested"]["alpha"] == 0.5


def test_mutable_alpha_input_cannot_change_the_frozen_guard_sequence():
    alphas = [1.0, 0.5, 0.0]
    conf = GuardConfig(alphas=alphas)
    alphas[1] = 0.9
    assert conf.alphas == (1.0, 0.5, 0.0)


def test_numerically_small_physical_bound_excess_is_refused_before_assessment():
    calls = []
    raw = RAW.copy()
    raw[0] = 1.0 + 5e-10
    def assess(q):
        calls.append(q.copy())
        return assessment(0.0, 0.001)
    with pytest.raises(ValueError, match="physical limits"):
        guarded_joint_selection(BASELINE, raw, BOUNDS, assess, GuardConfig(joint_delta_limit_rad=2.0))
    assert not calls


@pytest.mark.parametrize("q0, qr, limits", [
    (np.array([np.nan, 0.0]), RAW, BOUNDS),
    (BASELINE, np.array([0.1]), BOUNDS),
    (BASELINE, RAW, np.ones((2, 2))),
    (BASELINE, np.array([2.0, 0.0]), BOUNDS),
    (BASELINE, np.array([0.36, 0.0]), BOUNDS),
])
def test_invalid_or_out_of_domain_candidate_is_rejected_before_assessment(q0, qr, limits):
    calls = []
    def assess(q):
        calls.append(q.copy())
        return assessment(1.0, 0.001)
    with pytest.raises(ValueError):
        guarded_joint_selection(q0, qr, limits, assess)
    assert not calls


@pytest.mark.parametrize("bad", [
    {}, {"objective": np.nan, "mean_fingertip_error_m": 0.0},
    {"objective": 0.0, "mean_fingertip_error_m": -0.01},
    {"objective": [1.0], "mean_fingertip_error_m": 0.0},
    {"objective": True, "mean_fingertip_error_m": 0.0},
    None,
])
def test_unusable_assessment_refuses_selection(bad):
    with pytest.raises(ValueError, match="assessment must provide"):
        guarded_joint_selection(BASELINE, RAW, BOUNDS, lambda q: bad)


@pytest.mark.parametrize("change", [
    {"alphas": (0.5, 0.0)}, {"alphas": (1.0, 0.5)},
    {"alphas": (1.0, 0.5, 0.5, 0.0)}, {"alphas": (1.0, np.nan, 0.0)},
    {"maximum_mean_fingertip_error_m": 0.0},
    {"minimum_objective_improvement": -1.0}, {"joint_delta_limit_rad": np.inf},
    {"maximum_mean_fingertip_error_m": True}, {"alphas": (True, 0.0)},
    {"alphas": np.array(1.0)},
])
def test_invalid_fixed_guard_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        GuardConfig(**change)
