"""Strict, versioned schemas shared by evaluation manifests and results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ID_PATTERN = r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$"
SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class StrictModel(BaseModel):
    """Base model that rejects undeclared fields and mutable instances."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RegistryReference(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    expected_config_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)


class EmbodimentReference(StrictModel):
    id: str = Field(pattern=ID_PATTERN)


class ObjectCase(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    mass_kg: float = Field(gt=0.0)
    friction_profile_id: str = Field(pattern=ID_PATTERN)


class GraspCase(StrictModel):
    source: Literal["hug"] = "hug"
    id: str = Field(pattern=ID_PATTERN)
    prediction_path: Path

    @field_validator("prediction_path")
    @classmethod
    def require_relative_prediction_path(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("prediction_path must be a repository-relative path")
        return value


class InitialCondition(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    object_position_m: tuple[float, float, float]
    object_orientation_xyzw: tuple[float, float, float, float]

    @field_validator("object_orientation_xyzw")
    @classmethod
    def require_nonzero_quaternion(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if sum(component * component for component in value) <= 1e-12:
            raise ValueError("object orientation quaternion must be nonzero")
        return value


class EvaluationCase(StrictModel):
    schema_version: Literal["eval_case_v1"] = "eval_case_v1"
    case_id: str = Field(pattern=ID_PATTERN)
    environment: RegistryReference
    embodiment: EmbodimentReference
    execution_protocol: RegistryReference
    object: ObjectCase
    grasp: GraspCase
    initial_condition: InitialCondition
    motion_profile_id: str = Field(pattern=ID_PATTERN)
    seed: int = Field(ge=0, le=2**32 - 1)


class ResolvedExecution(StrictModel):
    environment_id: str = Field(pattern=ID_PATTERN)
    environment_config_hash: str = Field(pattern=SHA256_PATTERN)
    embodiment_id: str = Field(pattern=ID_PATTERN)
    execution_protocol_id: str = Field(pattern=ID_PATTERN)
    execution_protocol_config_hash: str = Field(pattern=SHA256_PATTERN)
    execution_protocol_parameters: dict[str, float | int | str]
    mujoco_version: str = Field(min_length=1)
    code_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")


class RetargetingMetrics(StrictModel):
    library: str = Field(min_length=1)
    config_id: str = Field(pattern=ID_PATTERN)
    mean_fingertip_error_m: float = Field(ge=0.0)
    maximum_fingertip_error_m: float = Field(ge=0.0)


class OutcomeMetrics(StrictModel):
    acquired_object: bool
    maximum_lift_m: float = Field(ge=0.0)
    hold_duration_s: float = Field(ge=0.0)
    final_object_height_m: float


class ArtifactReferences(StrictModel):
    trajectory: Path
    resolved_case: Path
    video: Path | None = None
    simulator_state: Path | None = None


class EvaluationResult(StrictModel):
    schema_version: Literal["eval_result_v1"] = "eval_result_v1"
    case_id: str = Field(pattern=ID_PATTERN)
    status: Literal["completed", "validation_failed", "runtime_failed"]
    success: bool
    failure_type: str | None = Field(default=None, pattern=ID_PATTERN)
    resolved_execution: ResolvedExecution
    retargeting: RetargetingMetrics
    outcome: OutcomeMetrics
    artifacts: ArtifactReferences


def canonical_json(value: BaseModel | dict[str, Any]) -> bytes:
    """Serialize a model or mapping deterministically for content hashing."""

    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: BaseModel | dict[str, Any]) -> str:
    """Return a tagged SHA-256 digest of canonical JSON data."""

    return f"sha256:{hashlib.sha256(canonical_json(value)).hexdigest()}"
