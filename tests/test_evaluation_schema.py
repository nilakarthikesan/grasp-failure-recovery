from __future__ import annotations

import pytest
from pydantic import ValidationError

from grasp_failure_prediction.evaluation.schema import (
    EvaluationCase,
    canonical_sha256,
)


def valid_case() -> dict:
    return {
        "schema_version": "eval_case_v1",
        "case_id": "object01_grasp003_mass018_seed9000",
        "environment": {"id": "adroit_shadow_tabletop_v1"},
        "embodiment": {"id": "shadow_hand_right"},
        "execution_protocol": {"id": "fixed_grasp_lift_v1"},
        "object": {
            "id": "object01",
            "mass_kg": 0.18,
            "friction_profile_id": "nominal_v1",
        },
        "grasp": {
            "source": "hug",
            "id": "grasp_003",
            "prediction_path": "grasps/object01/grasp_003.pkl",
        },
        "initial_condition": {
            "id": "pose_001",
            "object_position_m": [0.0, 0.0, 0.03],
            "object_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "motion_profile_id": "nominal_lift_v1",
        "seed": 9000,
    }


def test_case_schema_accepts_reviewed_contract() -> None:
    case = EvaluationCase.model_validate(valid_case())
    assert case.environment.id == "adroit_shadow_tabletop_v1"
    assert case.execution_protocol.id == "fixed_grasp_lift_v1"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("object", "mass_kg"), 0),
        (("seed",), -1),
        (("environment", "id"), "Unknown Environment"),
        (("grasp", "prediction_path"), "../outside.pkl"),
    ],
)
def test_case_schema_rejects_invalid_values(path: tuple[str, ...], value) -> None:
    payload = valid_case()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


def test_case_schema_rejects_unknown_fields() -> None:
    payload = valid_case()
    payload["silent_override"] = True
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


def test_canonical_hash_ignores_mapping_order() -> None:
    left = {"b": 2, "a": {"y": 1, "x": 0}}
    right = {"a": {"x": 0, "y": 1}, "b": 2}
    assert canonical_sha256(left) == canonical_sha256(right)
    assert canonical_sha256(left).startswith("sha256:")
