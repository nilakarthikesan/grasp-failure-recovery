from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from grasp_failure_prediction.evaluation.schema import (
    EvaluationCase,
    ObjectGeometryReference,
    canonical_sha256,
)
from test_evaluation_schema import valid_case


def geometry_reference() -> dict:
    return {
        "schema_version": "object_geometry_ref_v1",
        "kind": "mjcf_asset",
        "mjcf_path": "runs/reviewed_apple/sim_assets/object.xml",
        "expected_content_hash": "sha256:" + "a" * 64,
    }


def test_historic_cube_case_parses_without_attesting_geometry_match():
    case = EvaluationCase.model_validate(valid_case())
    assert case.object.id == "object01"
    assert case.object.geometry is None
    assert case.initial_condition.object_position_m == (0.0, 0.0, 0.03)
    assert case.initial_condition.object_orientation_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_explicit_geometry_reference_preserves_pinned_closure_and_other_fields():
    payload = valid_case()
    payload["object"]["geometry"] = geometry_reference()
    case = EvaluationCase.model_validate(payload)
    geometry = case.object.geometry
    assert isinstance(geometry, ObjectGeometryReference)
    assert geometry.mjcf_path == Path("runs/reviewed_apple/sim_assets/object.xml")
    assert geometry.expected_content_hash == "sha256:" + "a" * 64
    assert case.object.mass_kg == 0.18
    assert case.grasp.prediction_path == Path("grasps/object01/grasp_003.pkl")
    assert case.model_dump(mode="json")["object"]["geometry"] == geometry_reference()


def test_version_defaults_are_explicit_after_validation():
    reference = geometry_reference()
    del reference["schema_version"]
    del reference["kind"]
    geometry = ObjectGeometryReference.model_validate(reference)
    assert geometry.schema_version == "object_geometry_ref_v1"
    assert geometry.kind == "mjcf_asset"


@pytest.mark.parametrize("path", [
    "/outside/object.xml", "../object.xml", "runs/../object.xml", "", ".",
    "runs/reviewed_apple", "runs/reviewed_apple/object.obj",
])
def test_geometry_path_rejects_absolute_traversal_or_non_mjcf_file(path):
    reference = geometry_reference()
    reference["mjcf_path"] = path
    with pytest.raises(ValidationError, match="mjcf_path"):
        ObjectGeometryReference.model_validate(reference)


@pytest.mark.parametrize("path", ["assets/object.xml", "assets/object.mjcf"])
def test_geometry_path_allows_supported_relative_asset_suffixes(path):
    reference = geometry_reference()
    reference["mjcf_path"] = path
    assert ObjectGeometryReference.model_validate(reference).mjcf_path == Path(path)


@pytest.mark.parametrize("hash_value", [
    None, "a" * 64, "sha256:" + "a" * 63, "sha256:" + "g" * 64,
    "sha256:" + "A" * 64,
])
def test_expected_dependency_closure_hash_is_required_and_strict(hash_value):
    reference = geometry_reference()
    reference["expected_content_hash"] = hash_value
    with pytest.raises(ValidationError):
        ObjectGeometryReference.model_validate(reference)


def test_missing_hash_cannot_silently_enable_unpinned_geometry():
    reference = geometry_reference()
    del reference["expected_content_hash"]
    with pytest.raises(ValidationError, match="expected_content_hash"):
        ObjectGeometryReference.model_validate(reference)


@pytest.mark.parametrize(("field", "value"), [
    ("schema_version", "object_geometry_ref_v2"),
    ("kind", "box"), ("unchecked_override", True),
])
def test_unreviewed_geometry_contracts_and_unknown_fields_are_rejected(field, value):
    reference = geometry_reference()
    reference[field] = value
    with pytest.raises(ValidationError):
        ObjectGeometryReference.model_validate(reference)


def test_reference_is_immutable_and_changes_case_content_hash():
    original = EvaluationCase.model_validate(valid_case())
    payload = valid_case()
    payload["object"]["geometry"] = geometry_reference()
    explicit = EvaluationCase.model_validate(payload)
    assert canonical_sha256(original) != canonical_sha256(explicit)
    with pytest.raises(ValidationError):
        explicit.object.geometry.expected_content_hash = "sha256:" + "b" * 64


def test_geometry_reference_does_not_create_new_object_identity_or_scene_evidence():
    # This schema records a declaration. Runtime preflight must verify the
    # bytes/scene contract; renaming this case is not proof of independent data.
    payload = valid_case()
    payload["object"]["geometry"] = geometry_reference()
    payload["object"]["id"] = "reviewed_apple"
    case = EvaluationCase.model_validate(payload)
    assert case.object.id == "reviewed_apple"
    assert case.object.geometry.expected_content_hash == "sha256:" + "a" * 64
    assert not hasattr(case.object.geometry, "scene_verified")
