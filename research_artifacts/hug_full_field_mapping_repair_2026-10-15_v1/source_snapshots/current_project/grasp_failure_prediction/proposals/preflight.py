"""Versioned tolerances for provider-neutral proposal preflight."""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

import yaml
from pydantic import Field

from grasp_failure_prediction.evaluation.registry import Registry, RegistryError
from grasp_failure_prediction.evaluation.schema import StrictModel


class ProposalPreflightSpec(StrictModel):
    schema_version: Literal["proposal_preflight_spec_v1"]
    preflight_id: str
    maximum_object_to_palm_distance_m: float = Field(gt=0.0)


def load_preflight_registry() -> Registry[ProposalPreflightSpec]:
    resource = files(__package__).joinpath("configs", "preflight.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise RegistryError("proposal preflight registry must contain a mapping")
    entries = {}
    for entry_id, payload in raw.items():
        spec = ProposalPreflightSpec.model_validate(payload)
        if spec.preflight_id != entry_id:
            raise RegistryError(f"preflight key {entry_id!r} does not match embedded ID")
        entries[entry_id] = spec
    return Registry(entries)
