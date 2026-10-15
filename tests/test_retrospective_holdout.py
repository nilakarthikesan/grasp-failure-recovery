import numpy as np
import pytest

from grasp_failure_prediction.prediction.retrospective_holdout import (
    calibration, group_uncertainty, object_folds,
)


def identities():
    return (np.repeat(["a", "b", "c"], 8),
            np.repeat(["a0", "a1", "b0", "b1", "c0", "c1"], 4),
            np.repeat([f"p{i}" for i in range(12)], 2))


def test_object_observation_and_proposal_groups_do_not_leak():
    geometry, observation, proposal = identities()
    folds = object_folds(geometry, observation, proposal)
    assert len(folds) == 3
    for fold in folds:
        indices = fold["indices"]
        assert len(np.concatenate(list(indices.values()))) == len(geometry)
        assert set(geometry[indices["test"]]) == {fold["held_geometry"]}
        assert not set(geometry[indices["train"]]) & set(geometry[indices["test"]])
        for values in (observation, proposal):
            groups = [set(values[index]) for index in indices.values()]
            assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))


def test_membership_depends_on_identities_only_and_is_order_invariant():
    values = identities()
    original = object_folds(*values)
    reverse = object_folds(*(value[::-1] for value in values))
    for a, b in zip(original, reverse):
        for role in a["indices"]:
            assert set(values[2][a["indices"][role]]) == set(values[2][::-1][b["indices"][role]])


def test_rejects_cross_geometry_observation_and_cross_capture_proposal():
    values = list(identities())
    values[1][8] = values[1][0]
    with pytest.raises(ValueError, match="conflicting"):
        object_folds(*values)
    values = list(identities())
    values[2][4] = values[2][0]
    with pytest.raises(ValueError, match="conflicting"):
        object_folds(*values)


def test_requires_training_validation_captures():
    with pytest.raises(ValueError, match="two captures"):
        object_folds(["a", "b"], ["a0", "b0"], ["p0", "p1"])


def test_single_class_metrics_and_uncertainty_remain_undefined():
    result = group_uncertainty(np.zeros(4), np.full(4, 0.2), ["a", "a", "b", "b"], repeats=20)
    assert result["percentile_95"]["failure_roc_auc"] == {"valid_repeats": 0, "interval": None}
    assert result["percentile_95"]["success_false_alarm_rate"]["interval"] is None


def test_fixed_calibration_edges_include_zero_and_one():
    result = calibration(np.array([0, 1]), np.array([0.0, 1.0]))
    assert result["expected_calibration_error"] == 0
    assert sum(row["n"] for row in result["bins"]) == 2


def test_group_bootstrap_is_deterministic():
    args = ([0, 1, 0, 1], [0.2, 0.8, 0.7, 0.4], ["a", "a", "b", "b"])
    assert group_uncertainty(*args, repeats=20) == group_uncertainty(*args, repeats=20)
