"""Strict schemas at the boundary between grasp generation and execution."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from grasp_failure_prediction.evaluation.schema import (
    ID_PATTERN,
    SHA256_PATTERN,
    EmbodimentReference,
    InitialCondition,
    ObjectCase,
    OutcomeMetrics,
    RegistryReference,
    ResolvedExecution,
    StrictModel,
)


def _relative(value: Path | None, field: str) -> Path | None:
    if value is not None and (value.is_absolute() or ".." in value.parts or not value.name):
        raise ValueError(f"{field} must be a relative file path")
    return value


class ProviderReference(RegistryReference):
    """Reference to one immutable provider configuration."""


class RequestObject(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    geometry_content_hash: str = Field(pattern=SHA256_PATTERN)


class RequestInput(StrictModel):
    role: str = Field(pattern=ID_PATTERN)
    path: Path

    @field_validator("path")
    @classmethod
    def relative_input(cls, value: Path) -> Path:
        result = _relative(value, "provider input path")
        assert result is not None
        return result


class ProposalRequest(StrictModel):
    schema_version: Literal["proposal_request_v1"] = "proposal_request_v1"
    request_id: str = Field(pattern=ID_PATTERN)
    provider: ProviderReference
    object: RequestObject
    embodiment_id: Literal["shadow_hand_right"] = "shadow_hand_right"
    observation_path: Path | None = None
    provider_input_path: Path | None = None
    provider_inputs: tuple[RequestInput, ...] = ()
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    sample_count: int = Field(default=1, ge=1, le=1024)

    @field_validator("observation_path")
    @classmethod
    def relative_observation(cls, value: Path | None) -> Path | None:
        return _relative(value, "observation_path")

    @field_validator("provider_input_path")
    @classmethod
    def relative_provider_input(cls, value: Path | None) -> Path | None:
        return _relative(value, "provider_input_path")

    @field_validator("provider_inputs")
    @classmethod
    def unique_input_roles(cls, value):
        roles = [item.role for item in value]
        if len(set(roles)) != len(roles):
            raise ValueError("provider input roles must be unique")
        return value


class ProviderProvenance(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    config_hash: str = Field(pattern=SHA256_PATTERN)
    code_revision: str = Field(min_length=1)
    adapter_code_revision: str = Field(min_length=1)
    checkpoint_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)


class GenerationRecord(StrictModel):
    seed: int = Field(ge=0, le=2**32 - 1)
    sample_index: int = Field(ge=0)


class ProposalObjectBinding(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    geometry_content_hash: str = Field(pattern=SHA256_PATTERN)


class RigidTransform(StrictModel):
    """T_object_palm: palm coordinates expressed in the MuJoCo object body frame."""

    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    @field_validator("translation_m")
    @classmethod
    def finite_translation(cls, value):
        if not all(math.isfinite(component) for component in value):
            raise ValueError("translation_m must be finite")
        return value

    @field_validator("quaternion_xyzw")
    @classmethod
    def unit_quaternion(cls, value):
        if not all(math.isfinite(component) for component in value):
            raise ValueError("quaternion_xyzw must be finite")
        norm = math.sqrt(sum(component * component for component in value))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError("quaternion_xyzw must be unit length within 1e-6")
        return value


class InputBinding(StrictModel):
    geometry_group_id: str = Field(pattern=SHA256_PATTERN)
    observation_group_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    generation_request_group_id: str = Field(pattern=SHA256_PATTERN)
    input_manifest_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    scene_contract_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)


class NativeArtifactReference(StrictModel):
    role: str = Field(pattern=ID_PATTERN)
    path: Path
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: Path) -> Path:
        result = _relative(value, "native artifact path")
        assert result is not None
        return result


class ProviderValidation(StrictModel):
    status: Literal["passed", "not_applicable"]
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)

    @field_validator("metrics")
    @classmethod
    def finite_metrics(cls, value):
        for key, item in value.items():
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"provider validation metric {key!r} is non-finite")
        return value


class GraspProposal(StrictModel):
    """One execution-ready right Shadow Hand target.

    The palm link is the URDF/MuJoCo body named ``palm``. ``T_object_palm``
    maps palm-frame coordinates into the registered MuJoCo object body frame.
    All distances are meters and all joint values are radians.
    """

    schema_version: Literal["grasp_proposal_v1"] = "grasp_proposal_v1"
    proposal_id: str = Field(pattern=ID_PATTERN)
    provider: ProviderProvenance
    generation: GenerationRecord
    object: ProposalObjectBinding
    embodiment_id: Literal["shadow_hand_right"] = "shadow_hand_right"
    palm_link: Literal["palm"] = "palm"
    pose_frame: Literal["mujoco_object_body"] = "mujoco_object_body"
    object_to_palm: RigidTransform
    joint_targets_rad: dict[str, float]
    compatible_execution_protocols: tuple[str, ...]
    approach_semantics: Literal["fixed_world_negative_z"] = "fixed_world_negative_z"
    input_binding: InputBinding
    provider_validation: ProviderValidation
    provider_metrics: dict[str, Any] = Field(default_factory=dict)
    native_artifacts: tuple[NativeArtifactReference, ...] = ()

    @field_validator("joint_targets_rad")
    @classmethod
    def finite_named_joints(cls, value):
        if not value:
            raise ValueError("joint_targets_rad must not be empty")
        for name, position in value.items():
            if not name or not isinstance(name, str):
                raise ValueError("joint target names must be non-empty strings")
            if isinstance(position, bool) or not math.isfinite(float(position)):
                raise ValueError(f"joint target {name!r} must be finite")
        return value

    @field_validator("compatible_execution_protocols")
    @classmethod
    def protocols_are_ids(cls, value):
        if not value:
            raise ValueError("at least one compatible execution protocol is required")
        if len(set(value)) != len(value) or any(
            re.fullmatch(ID_PATTERN, item) is None for item in value
        ):
            raise ValueError("compatible execution protocols must be unique valid IDs")
        return value


class ProposalCaseReference(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    proposal_path: Path
    expected_proposal_hash: str = Field(pattern=SHA256_PATTERN)

    @field_validator("proposal_path")
    @classmethod
    def relative_proposal(cls, value: Path) -> Path:
        result = _relative(value, "proposal_path")
        assert result is not None
        return result


class EvaluationCaseV2(StrictModel):
    schema_version: Literal["eval_case_v2"] = "eval_case_v2"
    case_id: str = Field(pattern=ID_PATTERN)
    environment: RegistryReference
    embodiment: EmbodimentReference
    execution_protocol: RegistryReference
    object: ObjectCase
    grasp: ProposalCaseReference
    initial_condition: InitialCondition
    motion_profile_id: str = Field(pattern=ID_PATTERN)
    seed: int = Field(ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def require_shadow(self):
        if self.embodiment.id != "shadow_hand_right":
            raise ValueError("grasp_proposal_v1 currently supports shadow_hand_right only")
        quaternion = self.initial_condition.object_orientation_xyzw
        norm = math.sqrt(sum(component * component for component in quaternion))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-6:
            raise ValueError("eval_case_v2 object orientation must be a unit quaternion")
        return self


class ProposalManifestEntry(StrictModel):
    proposal_id: str = Field(pattern=ID_PATTERN)
    proposal_path: Path
    proposal_hash: str = Field(pattern=SHA256_PATTERN)
    generation_seed: int = Field(ge=0, le=2**32 - 1)
    proposal_sample_index: int = Field(ge=0)
    geometry_group_id: str = Field(pattern=SHA256_PATTERN)
    observation_group_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    generation_request_group_id: str = Field(pattern=SHA256_PATTERN)


class ProposalManifest(StrictModel):
    schema_version: Literal["proposal_manifest_v1"] = "proposal_manifest_v1"
    request_id: str = Field(pattern=ID_PATTERN)
    provider_id: str = Field(pattern=ID_PATTERN)
    provider_config_hash: str = Field(pattern=SHA256_PATTERN)
    proposals: tuple[ProposalManifestEntry, ...]


class ProposalExecutionValidation(StrictModel):
    joint_count: int = Field(gt=0)
    minimum_joint_limit_margin_rad: float = Field(ge=0.0)
    self_contact_count: int = Field(ge=0)
    provider_validation_status: Literal["passed", "not_applicable"]
    provider_metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)


class ResolvedProviderMetadata(StrictModel):
    kind: Literal["model", "optimizer", "dataset_replay"]
    translation_strategy_id: str = Field(pattern=ID_PATTERN)
    source_revision: str = Field(min_length=1)
    license_statuses: dict[str, Literal["verified", "unverified", "not_applicable"]]


class ProposalPreflightRejection(StrictModel):
    schema_version: Literal["proposal_preflight_rejection_v1"] = "proposal_preflight_rejection_v1"
    status: Literal["translation_failed", "proposal_rejected"]
    stage: Literal["translation", "structural", "kinematic", "scene_preflight"]
    code: str = Field(pattern=ID_PATTERN)
    message: str = Field(min_length=1)
    task_label: None = None


class ProposalRejectionArtifact(StrictModel):
    schema_version: Literal["proposal_rejection_artifact_v1"] = "proposal_rejection_artifact_v1"
    case_id: str = Field(pattern=ID_PATTERN)
    proposal_id: str = Field(pattern=ID_PATTERN)
    provider_id: str = Field(pattern=ID_PATTERN)
    proposal_hash: str = Field(pattern=SHA256_PATTERN)
    rejection: ProposalPreflightRejection


class ProposalArtifactReferences(StrictModel):
    trajectory: Path
    resolved_case: Path
    proposal: Path
    video: Path | None = None
    simulator_state: Path | None = None


class ProposalExecutionResult(StrictModel):
    schema_version: Literal["eval_result_v2"] = "eval_result_v2"
    case_id: str = Field(pattern=ID_PATTERN)
    status: Literal["completed", "validation_failed", "runtime_failed"]
    success: bool
    failure_type: str | None = Field(default=None, pattern=ID_PATTERN)
    resolved_execution: ResolvedExecution
    proposal_id: str = Field(pattern=ID_PATTERN)
    provider_id: str = Field(pattern=ID_PATTERN)
    provider_config_hash: str = Field(pattern=SHA256_PATTERN)
    # Optional only for backward loading of eval_result_v2 artifacts written
    # before provider classification was added. New executions always populate it.
    provider_metadata: ResolvedProviderMetadata | None = None
    proposal_hash: str = Field(pattern=SHA256_PATTERN)
    generation_seed: int = Field(ge=0, le=2**32 - 1)
    proposal_sample_index: int = Field(ge=0)
    geometry_group_id: str = Field(pattern=SHA256_PATTERN)
    observation_group_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    generation_request_group_id: str = Field(pattern=SHA256_PATTERN)
    proposal_validation: ProposalExecutionValidation
    outcome: OutcomeMetrics
    artifacts: ProposalArtifactReferences
