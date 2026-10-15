"""Bind a captured HUG observation to the object scene executed by a case.

An observation sidecar is a content binding, not proof that an external capture
was truthful. The capture implementation must obtain the geometry digest and
actual settled pose from the renderer. Contracts are copied unchanged into each
proposal folder; validation does not invent contracts for historic recordings.
"""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, StrictStr, ValidationError, field_validator, model_validator

from .schema import ID_PATTERN, SHA256_PATTERN, StrictModel


SCHEMA_VERSION = "source_scene_contract_v1"
POSITION_TOLERANCE_M = 1e-6
QUATERNION_TOLERANCE = 1e-6
MASS_TOLERANCE_KG = 1e-12
REQUIRED_OBSERVATION_FILES = frozenset({
    "rgb.png", "depth.png", "depth_m.npy", "intrinsics.npy",
    "object_mask.png", "T_world_camera.npy",
})

_FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
_TaggedHash = Annotated[StrictStr, Field(pattern=SHA256_PATTERN)]


def _normalized_orientation(value) -> np.ndarray:
    orientation = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(orientation))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("object_orientation_xyzw must be a finite nonzero quaternion")
    if abs(norm - 1.0) > QUATERNION_TOLERANCE:
        raise ValueError("object_orientation_xyzw must be a unit quaternion within 1e-6")
    return orientation / norm


def _canonical_orientation(value) -> list[float]:
    orientation = _normalized_orientation(value)
    # q and -q encode the same rotation. Choose a reproducible sign (w first,
    # followed by x/y/z for 180-degree rotations) for canonical content hashes.
    for component in (orientation[3], *orientation[:3]):
        if abs(component) > 1e-15:
            if component < 0:
                orientation = -orientation
            break
    orientation[orientation == 0] = 0.0
    return orientation.tolist()


class _SceneContract(StrictModel):
    schema_version: Literal["source_scene_contract_v1"]
    object_id: StrictStr = Field(pattern=ID_PATTERN)
    geometry_content_hash: _TaggedHash
    mass_kg: _FiniteNumber = Field(gt=0.0)
    object_position_m: tuple[_FiniteNumber, _FiniteNumber, _FiniteNumber]
    object_orientation_xyzw: tuple[_FiniteNumber, _FiniteNumber, _FiniteNumber, _FiniteNumber]
    environment_config_hash: _TaggedHash
    camera_sha256: _TaggedHash
    observation_files_sha256: dict[StrictStr, _TaggedHash]

    @field_validator("object_orientation_xyzw")
    @classmethod
    def require_unit_orientation(cls, value):
        _normalized_orientation(value)
        return value

    @field_validator("observation_files_sha256")
    @classmethod
    def require_complete_relative_file_map(cls, value):
        missing = REQUIRED_OBSERVATION_FILES - value.keys()
        if missing:
            raise ValueError("observation_files_sha256 is missing required files: "
                             + ", ".join(sorted(missing)))
        for name in value:
            path = PurePosixPath(name)
            if (not name or path.is_absolute() or ".." in path.parts
                    or not path.name or "\\" in name):
                raise ValueError("observation_files_sha256 names must be relative file paths")
        return value

    @model_validator(mode="after")
    def require_camera_binding(self):
        if self.observation_files_sha256["T_world_camera.npy"] != self.camera_sha256:
            raise ValueError("camera_sha256 must match observation_files_sha256['T_world_camera.npy']")
        return self


def _validated_contract(payload) -> _SceneContract:
    if not isinstance(payload, dict):
        raise ValueError("scene contract must be a JSON mapping")
    try:
        return _SceneContract.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid scene contract: {exc}") from exc


def build_scene_contract(
    *,
    object_id,
    geometry_content_hash,
    mass_kg,
    object_position_m,
    object_orientation_xyzw,
    environment_config_hash,
    camera_sha256,
    observation_files_sha256,
) -> dict:
    """Build a JSON-safe binding from verified capture measurements.

    Geometry hashes cover the MJCF dependency closure. Pose must be the actual
    captured object pose, rather than the requested pre-settling pose. Every
    input byte hash is mandatory; additional relative files remain part of the
    binding. Unit quaternions are normalized and sign-canonicalized so q/-q do
    not create a different scene contract.
    """

    model = _validated_contract({
        "schema_version": SCHEMA_VERSION,
        "object_id": object_id,
        "geometry_content_hash": geometry_content_hash,
        "mass_kg": mass_kg,
        "object_position_m": object_position_m,
        "object_orientation_xyzw": object_orientation_xyzw,
        "environment_config_hash": environment_config_hash,
        "camera_sha256": camera_sha256,
        "observation_files_sha256": observation_files_sha256,
    })
    result = model.model_dump(mode="json")
    result["object_orientation_xyzw"] = _canonical_orientation(model.object_orientation_xyzw)
    result["observation_files_sha256"] = dict(sorted(model.observation_files_sha256.items()))
    return result


def validate_scene_contract(
    contract,
    *,
    object_id,
    geometry_content_hash,
    mass_kg,
    object_position_m,
    object_orientation_xyzw,
    environment_config_hash,
    camera_sha256,
    observation_files_sha256,
) -> None:
    """Reject a capture/case mismatch before labeling a new scene.

    Content hashes and the complete file-hash mapping must match exactly.
    Positions use Euclidean tolerance 1e-6 m, mass uses absolute tolerance
    1e-12 kg, and normalized quaternion distance uses tolerance 1e-6 with q/-q
    equivalence. Invalid or missing data never produces a fallback contract.
    """

    actual = _validated_contract(contract)
    expected = _validated_contract(build_scene_contract(
        object_id=object_id, geometry_content_hash=geometry_content_hash,
        mass_kg=mass_kg, object_position_m=object_position_m,
        object_orientation_xyzw=object_orientation_xyzw,
        environment_config_hash=environment_config_hash,
        camera_sha256=camera_sha256, observation_files_sha256=observation_files_sha256,
    ))
    for field in ("object_id", "geometry_content_hash", "environment_config_hash", "camera_sha256"):
        if getattr(actual, field) != getattr(expected, field):
            raise ValueError(f"scene contract {field} mismatch")
    if not math.isclose(actual.mass_kg, expected.mass_kg, rel_tol=0, abs_tol=MASS_TOLERANCE_KG):
        raise ValueError("scene contract mass_kg mismatch")
    position_difference = np.linalg.norm(
        np.asarray(actual.object_position_m) - np.asarray(expected.object_position_m)
    )
    if position_difference > POSITION_TOLERANCE_M:
        raise ValueError("scene contract object_position_m mismatch")
    actual_orientation = _normalized_orientation(actual.object_orientation_xyzw)
    expected_orientation = _normalized_orientation(expected.object_orientation_xyzw)
    orientation_difference = min(
        np.linalg.norm(actual_orientation - expected_orientation),
        np.linalg.norm(actual_orientation + expected_orientation),
    )
    if orientation_difference > QUATERNION_TOLERANCE:
        raise ValueError("scene contract object_orientation_xyzw mismatch")
    if actual.observation_files_sha256 != expected.observation_files_sha256:
        differing = sorted(set(actual.observation_files_sha256) | set(expected.observation_files_sha256))
        differing = [name for name in differing
                     if actual.observation_files_sha256.get(name) != expected.observation_files_sha256.get(name)]
        raise ValueError("scene contract observation_files_sha256 mismatch: " + ", ".join(differing))
