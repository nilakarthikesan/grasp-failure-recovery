"""Versioned registry for grasp proposal providers."""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

import yaml
from pydantic import Field

from grasp_failure_prediction.evaluation.registry import Registry, RegistryError
from grasp_failure_prediction.evaluation.schema import StrictModel


class ProviderSpec(StrictModel):
    schema_version: Literal["proposal_provider_spec_v1"]
    provider_id: str
    adapter: str | None = None
    embodiment_id: Literal["shadow_hand_right"]
    execution: Literal["local", "local_or_external", "external"]
    required_inputs: tuple[str, ...]
    compatible_execution_protocols: tuple[str, ...]
    approach_semantics: Literal["fixed_world_negative_z"]
    maximum_mean_fingertip_error_m: float | None = Field(default=None, gt=0.0)


def load_provider_registry() -> Registry[ProviderSpec]:
    resource = files(__package__).joinpath("configs", "providers.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise RegistryError("provider registry must contain a mapping")
    entries: dict[str, ProviderSpec] = {}
    for provider_id, payload in raw.items():
        spec = ProviderSpec.model_validate(payload)
        if spec.provider_id != provider_id:
            raise RegistryError(
                f"provider key {provider_id!r} does not match embedded ID {spec.provider_id!r}"
            )
        entries[provider_id] = spec
    return Registry(entries)
