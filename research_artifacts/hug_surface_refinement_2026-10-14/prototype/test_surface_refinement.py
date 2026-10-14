"""Proposal-preservation tests for the experimental static refinement.

These use manufactured geometry callbacks only: no generator, simulator, eval
record or outcome label is involved. Imports are the eventual project API.
"""
import copy

import numpy as np
import pytest

from grasp_failure_prediction.evaluation.surface_refinement import (
    RefinementConfig,
    area_quantile_sites,
    refine_part,
    select_sites,
)


def proposal_case(**overrides):
    # A planar query's closest hand point lies on the negative side. Equal
    # unsigned distances alone cannot distinguish this native poor relation
    # from a replacement surface on the positive side.
    data = {
        "initial_q": np.array([0.01]),
        "physical_bounds": np.array([[-0.04, 0.04]]),
        "native_distances": np.array([0.01]),
        "native_closest_points": np.array([[0.0, 0.0, -0.01]]),
        "native_facing": np.array([-0.6]),
        "normal_eligible": np.array([True]),
        "query_weights": np.array([1.0]),
        "original_anchor_positions": np.array([[0.0, 0.0, 0.0]]),
    }
    data.update(overrides)
    return data


def plane_evaluator(q):
    return (
        np.abs(q),
        np.array([[0.0, 0.0, q[0]]]),
        np.array([-0.6]),
        np.zeros((1, 3)),
    )


def config(**kwargs):
    # A separately manufactured objective weight makes the planar side-change
    # tractable. This does not alter defaults or the frozen saved-grasp run.
    vector_weight = kwargs.pop("closest_vector_weight", 3.0)
    return RefinementConfig(anchor_weight=0.0, joint_weight=0.0,
                            closest_vector_weight=vector_weight, **kwargs)


def test_closest_vector_disambiguates_equal_distance_opposite_sides():
    args = proposal_case()
    distance_only, unchanged = refine_part(
        **args, evaluate=plane_evaluator,
        config=config(closest_vector_weight=0.0),
    )
    np.testing.assert_array_equal(distance_only, args["initial_q"])
    assert not unchanged["accepted_objective_improvement"]

    candidate, diagnostic = refine_part(
        **args, evaluate=plane_evaluator, config=config(),
    )
    assert candidate[0] == pytest.approx(-0.01, abs=1e-7)
    assert diagnostic["accepted_objective_improvement"]
    assert diagnostic["after_robust_cost"] < diagnostic["before_robust_cost"]
    assert diagnostic["does_not_mean_mapping_acceptance"]


def test_native_away_facing_and_negative_side_are_retained_as_targets():
    # A signed positional component may encode native penetration/approach
    # side; it is deliberately not replaced by a zero-penetration goal. An
    # away-facing native normal is likewise not replaced by +1 contact reward.
    def evaluator(q):
        distance, closest, facing, anchors = plane_evaluator(q)
        return distance, closest, np.clip(q * 60.0, -1.0, 1.0), anchors
    candidate, diagnostic = refine_part(
        **proposal_case(), evaluate=evaluator,
        config=config(normal_weight=1.0),
    )
    assert candidate[0] < 0.0
    assert diagnostic["after_component_sse"]["facing"] < 1e-10
    assert evaluator(candidate)[2][0] == pytest.approx(-0.6, abs=1e-5)
    assert diagnostic["native_normal_eligible_site_count"] == 1


def test_default_local_objective_improvement_does_not_certify_side_preservation():
    candidate, diagnostic = refine_part(
        **proposal_case(), evaluate=plane_evaluator,
        config=config(closest_vector_weight=1.0),
    )
    assert diagnostic["accepted_objective_improvement"]
    assert diagnostic["does_not_mean_mapping_acceptance"]
    # The least-squares solver remains on the wrong side in a local basin.
    # This deliberate limitation test prevents presenting its flag as a proof.
    assert 0.0 < candidate[0] < 0.01


@pytest.mark.parametrize("physical, target, expected", [
    ((-0.005, 0.04), -0.02, -0.005),
    ((-0.04, 0.04), -0.02, 0.008),
])
def test_physical_and_local_trust_bounds_limit_a_requested_correction(physical, target, expected):
    trust = 0.002 if expected == 0.008 else 0.35
    args = proposal_case(
        physical_bounds=np.array([physical]),
        native_distances=np.array([abs(target)]),
        native_closest_points=np.array([[0.0, 0.0, target]]),
    )
    candidate, diagnostic = refine_part(
        **args, evaluate=plane_evaluator,
        config=config(joint_delta_limit_rad=trust),
    )
    assert candidate[0] == pytest.approx(expected, abs=2e-7)
    assert diagnostic["accepted_objective_improvement"]
    assert physical[0] - 1e-9 <= candidate[0] <= physical[1] + 1e-9
    assert abs(candidate[0] - args["initial_q"][0]) <= trust + 1e-9


def test_input_arrays_are_not_modified_by_a_cooperative_evaluator():
    args = proposal_case()
    saved = copy.deepcopy(args)
    candidate, _ = refine_part(**args, evaluate=plane_evaluator, config=config())
    for name in args:
        np.testing.assert_array_equal(args[name], saved[name])
    candidate[0] = 100.0
    np.testing.assert_array_equal(args["initial_q"], saved["initial_q"])


def test_no_geometric_improvement_returns_baseline_with_explicit_status():
    args = proposal_case(
        initial_q=np.array([-0.01]),
    )
    candidate, diagnostic = refine_part(**args, evaluate=plane_evaluator, config=config())
    np.testing.assert_array_equal(candidate, args["initial_q"])
    assert not diagnostic["accepted_objective_improvement"]
    assert diagnostic["does_not_mean_mapping_acceptance"]


@pytest.mark.parametrize("change", [
    {"initial_q": np.array([np.nan])},
    {"initial_q": np.array([[0.01]])},
    {"initial_q": np.array([0.05])},
    {"physical_bounds": np.array([[0.0, 0.0]])},
    {"physical_bounds": np.array([[-np.inf, 1.0]])},
    {"native_distances": np.array([-0.001])},
    {"native_closest_points": np.zeros((1, 2))},
    {"native_facing": np.array([np.nan])},
    {"query_weights": np.array([0.0])},
    {"query_weights": np.array([0.5])},
    {"original_anchor_positions": np.empty((0, 3))},
])
def test_invalid_source_contract_is_rejected_before_geometry_evaluation(change):
    calls = []
    def evaluator(q):
        calls.append(q.copy())
        return plane_evaluator(q)
    with pytest.raises(ValueError):
        refine_part(**proposal_case(**change), evaluate=evaluator, config=config())
    assert not calls


@pytest.mark.parametrize("bad_kind", ["shape", "nonfinite"])
def test_invalid_geometry_callback_is_refused(bad_kind):
    def evaluator(q):
        values = list(plane_evaluator(q))
        if bad_kind == "shape":
            values[1] = np.zeros((2, 3))
        else:
            values[1][0, 2] = np.nan
        return values
    with pytest.raises(ValueError, match="static.*(shape|evaluation)"):
        refine_part(**proposal_case(), evaluate=evaluator, config=config())


def test_negative_evaluator_unsigned_distances_are_rejected():
    def evaluator(q):
        distance, closest, facing, anchors = plane_evaluator(q)
        return -distance, closest, facing, anchors
    with pytest.raises(ValueError, match="unsigned and nonnegative"):
        refine_part(**proposal_case(), evaluate=evaluator, config=config())


def test_geometry_callback_cannot_modify_solver_target_or_source_input():
    args = proposal_case()
    saved = args["initial_q"].copy()
    def evaluator(q):
        assert not q.flags.writeable
        with pytest.raises(ValueError):
            q[0] = 999.0
        return plane_evaluator(q)
    candidate, diagnostic = refine_part(**args, evaluate=evaluator, config=config())
    np.testing.assert_array_equal(args["initial_q"], saved)
    assert candidate[0] < 0.0
    assert diagnostic["accepted_objective_improvement"]


def test_numerically_small_outside_bound_source_is_refused_without_clipping():
    args = proposal_case(initial_q=np.array([0.04 + 5e-10]))
    calls = []
    def evaluator(q):
        calls.append(q.copy())
        return plane_evaluator(q)
    with pytest.raises(ValueError, match="URDF bounds"):
        refine_part(**args, evaluate=evaluator, config=config())
    assert not calls
    assert args["initial_q"][0] > args["physical_bounds"][0, 1]


def test_fixed_area_quadrature_retains_large_cells_without_claiming_independence():
    sites = area_quantile_sites(np.array([True, True]), np.array([9.0, 1.0]), 10)
    np.testing.assert_array_equal(sites, np.array([0] * 9 + [1]))
    assert len(np.unique(sites)) < len(sites)
    assert area_quantile_sites(np.zeros(2, bool), np.ones(2), 3).size == 0


def test_site_selection_includes_native_near_baseline_near_and_far_guard_sites():
    conf = RefinementConfig(near_band_m=0.01, near_sites=4, broad_sites=3, near_mass=0.75)
    nd = np.array([0.001, 0.020, 0.100])
    sd = np.array([0.020, 0.001, 0.100])
    area = np.ones(3)
    sites, weights, diagnostic = select_sites(nd, sd, area, conf)
    assert {0, 1} <= set(sites[:4])
    assert 2 in sites[4:]
    assert weights[:4].sum() == pytest.approx(0.75)
    assert weights[4:].sum() == pytest.approx(0.25)
    assert weights.sum() == pytest.approx(1.0)
    assert diagnostic["near_source_cell_count"] == 2


def test_no_near_sites_falls_back_to_full_area_quadrature():
    conf = RefinementConfig(near_sites=4, broad_sites=2)
    sites, weights, diagnostic = select_sites(np.ones(2), np.ones(2), np.ones(2), conf)
    np.testing.assert_array_equal(sites, [0, 1])
    np.testing.assert_allclose(weights, [0.5, 0.5])
    assert diagnostic["near_bin_count"] == 0


@pytest.mark.parametrize("native, original, area", [
    ([], [], []), ([0.0, -1.0], [0.0, 0.0], [1.0, 1.0]),
    ([[0.0]], [[0.0]], [[1.0]]), ([0.0], [0.0], [np.nan]),
])
def test_invalid_site_domains_are_rejected(native, original, area):
    with pytest.raises(ValueError):
        select_sites(native, original, area, RefinementConfig())


@pytest.mark.parametrize("change", [
    {"distance_scale_m": 0.0}, {"closest_vector_weight": -1.0},
    {"normal_weight": np.nan}, {"near_mass": 1.0},
    {"max_nfev": True}, {"near_sites": 1.5}, {"loss": "linear"},
    {"normal_weight": True}, {"distance_scale_m": True}, {"near_mass": np.bool_(True)},
])
def test_invalid_frozen_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        RefinementConfig(**change)
