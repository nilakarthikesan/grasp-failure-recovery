from __future__ import annotations

import pytest

from grasp_failure_prediction.evaluation.registry import (
    RegistryError,
    load_environment_registry,
    load_protocol_registry,
    resolve_case,
)
from grasp_failure_prediction.evaluation.schema import EvaluationCase
from test_evaluation_schema import valid_case


def test_versioned_registries_load_reviewed_entries() -> None:
    environments = load_environment_registry()
    protocols = load_protocol_registry()
    assert environments.ids == ("adroit_shadow_tabletop_v1",)
    assert protocols.ids == ("fixed_grasp_lift_v1",)
    assert (
        protocols.resolve("fixed_grasp_lift_v1")
        .parameters.maximum_mean_fingertip_error_m
        == 0.01
    )


def test_case_resolution_adds_reproducible_hashes() -> None:
    resolved = resolve_case(EvaluationCase.model_validate(valid_case()))
    assert resolved.environment_config_hash.startswith("sha256:")
    assert resolved.execution_protocol_config_hash.startswith("sha256:")
    assert len(resolved.environment_config_hash) == 71


def test_unknown_environment_fails_instead_of_falling_back() -> None:
    payload = valid_case()
    payload["environment"]["id"] = "missing_environment_v1"
    with pytest.raises(RegistryError, match="unknown registry ID"):
        resolve_case(EvaluationCase.model_validate(payload))


def test_hash_mismatch_fails_before_execution() -> None:
    payload = valid_case()
    payload["execution_protocol"]["expected_config_hash"] = "sha256:" + "0" * 64
    with pytest.raises(RegistryError, match="configuration hash mismatch"):
        resolve_case(EvaluationCase.model_validate(payload))


def test_environment_embodiment_mismatch_is_rejected() -> None:
    payload = valid_case()
    payload["embodiment"]["id"] = "different_hand"
    with pytest.raises(RegistryError, match="does not match environment hand"):
        resolve_case(EvaluationCase.model_validate(payload))
