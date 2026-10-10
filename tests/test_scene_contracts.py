from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from grasp_failure_prediction.evaluation.scene_contracts import (
    build_scene_contract,
    validate_scene_contract,
)
from grasp_failure_prediction.evaluation.schema import canonical_sha256


def digest(letter="a"):
    return "sha256:" + letter * 64


def arguments():
    return {
        "object_id": "reviewed_apple",
        "geometry_content_hash": digest("a"),
        "mass_kg": 0.2304,
        "object_position_m": [0.0, 0.0, 0.046],
        "object_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "environment_config_hash": digest("b"),
        "camera_sha256": digest("c"),
        "observation_files_sha256": {
            "rgb.png": digest("d"), "depth.png": digest("e"),
            "depth_m.npy": digest("f"), "intrinsics.npy": digest("a"),
            "object_mask.png": digest("b"), "T_world_camera.npy": digest("c"),
        },
    }


def test_capture_contract_round_trips_json_and_validates_against_the_scene():
    expected = arguments()
    contract = build_scene_contract(**expected)
    assert contract["schema_version"] == "source_scene_contract_v1"
    assert list(contract["observation_files_sha256"]) == sorted(expected["observation_files_sha256"])
    round_trip = json.loads(json.dumps(contract, allow_nan=False))
    assert validate_scene_contract(round_trip, **expected) is None


@pytest.mark.parametrize(("field", "value"), [
    ("object_id", "different_object"),
    ("geometry_content_hash", digest("b")),
    ("mass_kg", 0.18),
    ("object_position_m", [0.0, 0.0, 0.03]),
    ("object_orientation_xyzw", [0.0, 0.0, 1.0, 0.0]),
    ("environment_config_hash", digest("a")),
])
def test_geometry_mass_pose_identity_and_environment_mismatches_fail_closed(field, value):
    original = arguments()
    contract = build_scene_contract(**original)
    changed = deepcopy(original)
    changed[field] = value
    with pytest.raises(ValueError, match=field + " mismatch"):
        validate_scene_contract(contract, **changed)


def test_changed_camera_bytes_cannot_reuse_a_saved_proposal_contract():
    original = arguments()
    contract = build_scene_contract(**original)
    changed = deepcopy(original)
    changed["camera_sha256"] = digest("d")
    changed["observation_files_sha256"]["T_world_camera.npy"] = digest("d")
    with pytest.raises(ValueError, match="camera_sha256 mismatch"):
        validate_scene_contract(contract, **changed)


@pytest.mark.parametrize("file_name", ["rgb.png", "depth.png", "depth_m.npy", "intrinsics.npy", "object_mask.png"])
def test_any_changed_observation_bytes_break_full_content_binding(file_name):
    original = arguments()
    contract = build_scene_contract(**original)
    changed = deepcopy(original)
    changed["observation_files_sha256"][file_name] = digest("c")
    with pytest.raises(ValueError, match="observation_files_sha256 mismatch: " + file_name.replace(".", r"\.")):
        validate_scene_contract(contract, **changed)


def test_hash_map_order_is_irrelevant_but_extra_files_are_still_bound():
    original = arguments()
    original["observation_files_sha256"]["capture_settings.json"] = digest("a")
    contract = build_scene_contract(**original)
    reordered = deepcopy(original)
    reordered["observation_files_sha256"] = dict(reversed(list(original["observation_files_sha256"].items())))
    assert validate_scene_contract(contract, **reordered) is None
    assert canonical_sha256(contract) == canonical_sha256(build_scene_contract(**reordered))
    del reordered["observation_files_sha256"]["capture_settings.json"]
    with pytest.raises(ValueError, match="capture_settings"):
        validate_scene_contract(contract, **reordered)


@pytest.mark.parametrize("orientation", [
    [0.0, 0.0, 0.0, -1.0],
    [0.0, 0.0, 0.0, 1.0000005],
])
def test_equivalent_quaternion_sign_or_tiny_norm_rounding_does_not_change_pose(orientation):
    original = arguments()
    contract = build_scene_contract(**original)
    changed = deepcopy(original)
    changed["object_orientation_xyzw"] = orientation
    assert validate_scene_contract(contract, **changed) is None
    assert canonical_sha256(contract) == canonical_sha256(build_scene_contract(**changed))


def test_180_degree_equivalent_quaternion_sign_has_identical_canonical_hash():
    original = arguments()
    original["object_orientation_xyzw"] = [1.0, 0.0, 0.0, 0.0]
    changed = deepcopy(original)
    changed["object_orientation_xyzw"] = [-1.0, -0.0, -0.0, -0.0]
    assert canonical_sha256(build_scene_contract(**original)) == canonical_sha256(build_scene_contract(**changed))
    validate_scene_contract(build_scene_contract(**original), **changed)


def test_numpy_capture_measurements_are_supported_without_mutating_them():
    expected = arguments()
    expected["object_position_m"] = np.array(expected["object_position_m"])
    expected["object_orientation_xyzw"] = np.array(expected["object_orientation_xyzw"])
    original_orientation = expected["object_orientation_xyzw"].copy()
    contract = build_scene_contract(**expected)
    validate_scene_contract(contract, **expected)
    np.testing.assert_array_equal(expected["object_orientation_xyzw"], original_orientation)


def test_position_mass_and_orientation_tolerances_allow_only_tiny_differences():
    original = arguments()
    contract = build_scene_contract(**original)
    changed = deepcopy(original)
    changed["object_position_m"][0] = 5e-7
    changed["mass_kg"] += 5e-13
    changed["object_orientation_xyzw"] = [5e-7, 0.0, 0.0, np.sqrt(1 - (5e-7)**2)]
    validate_scene_contract(contract, **changed)
    changed["object_position_m"] = [9e-7, 9e-7, 0.046]
    with pytest.raises(ValueError, match="object_position_m mismatch"):
        validate_scene_contract(contract, **changed)


@pytest.mark.parametrize(("field", "value"), [
    ("mass_kg", 0.0), ("mass_kg", -0.1), ("mass_kg", float("inf")),
    ("mass_kg", float("nan")), ("mass_kg", True), ("mass_kg", "0.2304"),
    ("object_position_m", [0.0, 0.0]),
    ("object_position_m", [0.0, float("inf"), 0.0]),
    ("object_position_m", [0.0, 0.0, float("nan")]),
    ("object_position_m", [0.0, 0.0, "0.046"]),
    ("object_orientation_xyzw", [0.0, 0.0, 0.0, 0.0]),
    ("object_orientation_xyzw", [0.0, 0.0, 0.0, 2.0]),
    ("object_orientation_xyzw", [0.0, 0.0, float("nan"), 1.0]),
    ("object_orientation_xyzw", [0.0, 0.0, 1.0]),
    ("object_orientation_xyzw", [False, 0.0, 0.0, 1.0]),
])
def test_invalid_capture_measurements_cannot_build_a_contract(field, value):
    expected = arguments()
    expected[field] = value
    with pytest.raises(ValueError, match=field):
        build_scene_contract(**expected)


@pytest.mark.parametrize("field", ["geometry_content_hash", "environment_config_hash", "camera_sha256"])
@pytest.mark.parametrize("value", [1, None, "a" * 64, "sha256:" + "g" * 64])
def test_top_level_hashes_are_typed_and_tagged(field, value):
    expected = arguments()
    expected[field] = value
    with pytest.raises(ValueError, match=field):
        build_scene_contract(**expected)


def test_partial_empty_malformed_or_internally_inconsistent_file_maps_are_rejected():
    for invalid_map in ({}, {"rgb.png": digest()}, {"rgb.png": 1}):
        expected = arguments()
        expected["observation_files_sha256"] = invalid_map
        with pytest.raises(ValueError, match="observation_files_sha256"):
            build_scene_contract(**expected)
    expected = arguments()
    expected["observation_files_sha256"]["T_world_camera.npy"] = digest("a")
    with pytest.raises(ValueError, match="camera_sha256 must match"):
        build_scene_contract(**expected)


@pytest.mark.parametrize("name", ["../outside.npy", "/outside.npy", "", "x\\outside.npy"])
def test_additional_observation_names_cannot_escape_capture_directory(name):
    expected = arguments()
    expected["observation_files_sha256"][name] = digest("a")
    with pytest.raises(ValueError, match="relative file paths"):
        build_scene_contract(**expected)


def test_missing_version_unknown_fields_or_non_mapping_contract_cannot_validate():
    expected = arguments()
    contract = build_scene_contract(**expected)
    invalid = deepcopy(contract)
    del invalid["schema_version"]
    with pytest.raises(ValueError, match="schema_version"):
        validate_scene_contract(invalid, **expected)
    invalid = deepcopy(contract)
    invalid["scene_verified"] = True
    with pytest.raises(ValueError, match="scene_verified"):
        validate_scene_contract(invalid, **expected)
    invalid = deepcopy(contract)
    invalid["schema_version"] = "source_scene_contract_v2"
    with pytest.raises(ValueError, match="schema_version"):
        validate_scene_contract(invalid, **expected)
    with pytest.raises(ValueError, match="JSON mapping"):
        validate_scene_contract(json.dumps(contract), **expected)
