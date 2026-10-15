import pytest

from grasp_failure_prediction.evaluation.scenario_sampler import (
    ScenarioAxes, freeze_category_folds, sample_scenarios,
)


def objects():
    return [dict(object_id=f"object_{i}", category_id=f"category_{i}",
                 geometry_group_id=f"geometry_{i}", observation_group_ids=[f"capture_{i}_a", f"capture_{i}_b"])
            for i in range(4)]


def test_schedule_is_reproducible_bounded_and_reports_independent_units():
    args = dict(seed=7, attempts_per_object=16, generation_seeds=(10, 11))
    a = sample_scenarios(objects(), **args)
    assert a == sample_scenarios(objects(), **args)
    assert a != sample_scenarios(objects(), **{**args, "seed": 8})
    assert a["counts"]["attempts"] == 64
    assert a["counts"]["objects"] == 4
    assert a["counts"]["observations"] <= 8
    assert a["counts"]["proposal_families"] <= 16
    assert not a["outcome_selection"]


def test_holdout_keeps_every_proposal_and_capture_in_one_partition():
    schedule = sample_scenarios(objects(), seed=2, attempts_per_object=32, generation_seeds=(10, 11))
    folds = freeze_category_folds(schedule)
    assert len(folds["folds"]) == 4
    for fold in folds["folds"]:
        for field in ("object_id", "observation_group_id", "proposal_group_id", "geometry_group_id"):
            seen = {}
            for row in schedule["scenarios"]:
                role = fold["assignments"][row["scenario_id"]]
                assert seen.setdefault(row[field], role) == role
        assert set(fold["assignments"].values()) == {"train", "validation", "test"}


@pytest.mark.parametrize("kwargs", [dict(seed=-1), dict(attempts_per_object=0),
    dict(attempts_per_object=99999), dict(generation_seeds=(1, 1)),
    dict(axes=ScenarioAxes(friction_coefficients=(0.,))),
    dict(axes=ScenarioAxes(approach_modes=("unknown",)))])
def test_invalid_or_unbounded_schedule_refused(kwargs):
    args = dict(seed=2, attempts_per_object=4, generation_seeds=(10,))
    with pytest.raises(ValueError):
        sample_scenarios(objects(), **{**args, **kwargs})


def test_too_few_categories_refused_before_training():
    schedule = sample_scenarios(objects()[:2], seed=2, attempts_per_object=4, generation_seeds=(10,))
    with pytest.raises(ValueError, match="three"):
        freeze_category_folds(schedule)


def test_identity_aliases_cannot_leak_across_categories():
    source = objects()
    source[1]["geometry_group_id"] = source[0]["geometry_group_id"]
    with pytest.raises(ValueError, match="category"):
        sample_scenarios(source, seed=2, attempts_per_object=4, generation_seeds=(10,))
    source = objects()
    source[1]["observation_group_ids"] = source[0]["observation_group_ids"]
    with pytest.raises(ValueError, match="observation"):
        sample_scenarios(source, seed=2, attempts_per_object=4, generation_seeds=(10,))
