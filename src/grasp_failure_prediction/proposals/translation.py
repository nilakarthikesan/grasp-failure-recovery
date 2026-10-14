"""Fail-closed native-decoder and reusable hand-translation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import yaml
from pydantic import Field, model_validator

from grasp_failure_prediction.evaluation.registry import Registry, RegistryError
from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter
from grasp_failure_prediction.evaluation.schema import SHA256_PATTERN, StrictModel

from .schema import RigidTransform


class MappingEvidence(StrictModel):
    role: Literal["source_asset", "target_asset", "joint_mapping", "retargeter_config", "implementation"]
    location: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    availability: Literal["repository", "python_package", "external_licensed"]


class TranslationStrategySpec(StrictModel):
    schema_version: Literal["translation_strategy_spec_v1"]
    strategy_id: str = Field(min_length=1)
    strategy_kind: Literal["exact_kinematic_mapping", "official_retargeter"]
    source_hand_model_id: str = Field(min_length=1)
    target_hand_model_id: Literal["shadow_hand_right"]
    source_root_link: str = Field(min_length=1)
    target_root_link: Literal["palm"]
    source_asset_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    target_asset_hash: str = Field(pattern=SHA256_PATTERN)
    joint_mapping_path: str | None = None
    retargeter_id: str | None = None
    maximum_mean_fingertip_error_m: float | None = Field(default=None, gt=0.0)
    evidence: tuple[MappingEvidence, ...]

    @model_validator(mode="after")
    def complete_strategy(self):
        roles = {item.role for item in self.evidence}
        if not {"target_asset", "implementation"}.issubset(roles):
            raise ValueError("translation strategy requires target-asset and implementation evidence")
        if self.strategy_kind == "exact_kinematic_mapping":
            if self.source_asset_hash is None or self.joint_mapping_path is None:
                raise ValueError("exact mapping requires source asset hash and joint mapping")
            if not {"source_asset", "joint_mapping"}.issubset(roles):
                raise ValueError("exact mapping requires source-asset and joint-mapping evidence")
        elif self.retargeter_id is None or self.maximum_mean_fingertip_error_m is None:
            raise ValueError("official retargeter requires ID and residual threshold")
        return self


def load_translation_registry() -> Registry[TranslationStrategySpec]:
    resource = files(__package__).joinpath("configs", "translations.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise RegistryError("translation registry must contain a mapping")
    entries = {}
    for strategy_id, payload in raw.items():
        spec = TranslationStrategySpec.model_validate(payload)
        if spec.strategy_id != strategy_id:
            raise RegistryError(f"translation key {strategy_id!r} does not match embedded ID")
        entries[strategy_id] = spec
    return Registry(entries)


@dataclass(frozen=True)
class NativeHandPose:
    source_hand_model_id: str
    source_root_link: str
    object_to_root: RigidTransform
    joint_targets_rad: dict[str, float]
    payload: Any = None


@dataclass(frozen=True)
class TranslatedHandPose:
    object_to_palm: RigidTransform
    joint_targets_rad: dict[str, float]
    validation_metrics: dict[str, float | int | str | bool]


class NativeDecoder(Protocol):
    def decode(self, path: Path) -> NativeHandPose: ...


class TranslationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class SyntheticShadowRecordDecoder:
    """Decode the project-authored strict named-Shadow replay record."""

    schema_version = "shadow_saved_record_v1"

    def decode(self, path: Path) -> NativeHandPose:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.schema_version:
            raise TranslationError("native_schema_unsupported", "saved Shadow record has the wrong schema version")
        return NativeHandPose(
            source_hand_model_id="shadow_hand_right_dex_urdf_7304c7f",
            source_root_link="palm",
            object_to_root=RigidTransform.model_validate(payload["object_to_palm"]),
            joint_targets_rad={str(k): float(v) for k, v in payload["joint_targets_rad"].items()},
        )


def translate_native_pose(native: NativeHandPose, strategy_id: str) -> TranslatedHandPose:
    spec = load_translation_registry().resolve(strategy_id)
    if native.source_hand_model_id != spec.source_hand_model_id or native.source_root_link != spec.source_root_link:
        raise TranslationError("source_hand_mismatch", "native source hand/root does not match translation strategy")
    if spec.strategy_kind != "exact_kinematic_mapping":
        raise TranslationError("strategy_input_unsupported", "official retargeter requires native landmark payload")
    mapping_resource = files(__package__).joinpath(spec.joint_mapping_path)
    mapping_digest = "sha256:" + hashlib.sha256(mapping_resource.read_bytes()).hexdigest()
    recorded = next(item.sha256 for item in spec.evidence if item.role == "joint_mapping")
    if mapping_digest != recorded:
        raise TranslationError("mapping_evidence_hash_mismatch", "registered joint mapping evidence changed")
    with mapping_resource.open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle)
    joint_mapping = mapping["joint_mapping"]
    if set(native.joint_targets_rad) != set(joint_mapping):
        raise TranslationError("native_joint_set_mismatch", "native named joint set does not match exact mapping")
    return TranslatedHandPose(
        object_to_palm=native.object_to_root,
        joint_targets_rad={target: native.joint_targets_rad[source] for source, target in joint_mapping.items()},
        validation_metrics={"translation_strategy_id": strategy_id, "mapping_kind": spec.strategy_kind},
    )


def translate_mano_with_official_retargeter(grasp: Any, strategy_id: str):
    spec = load_translation_registry().resolve(strategy_id)
    if spec.strategy_kind != "official_retargeter" or spec.source_hand_model_id != "mano_right_hug_landmarks_v1":
        raise TranslationError("strategy_kind_mismatch", "MANO decoder requires the registered official retargeter")
    retargeter = ShadowHandRetargeter()
    config_digest = "sha256:" + hashlib.sha256(retargeter.config_path.read_bytes()).hexdigest()
    recorded = next(item.sha256 for item in spec.evidence if item.role == "retargeter_config")
    if config_digest != recorded:
        raise TranslationError("retargeter_evidence_hash_mismatch", "official retargeter configuration changed")
    return retargeter.retarget(grasp)
