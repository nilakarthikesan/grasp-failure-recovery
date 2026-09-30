"""Versioned environment and execution-protocol registries."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Generic, Literal, TypeVar

import yaml
from pydantic import Field

from .schema import EvaluationCase, StrictModel, canonical_sha256


class RegistryError(ValueError):
    """Raised when a registry entry is unknown, malformed, or mismatched."""


class SimulatorSpec(StrictModel):
    type: Literal["mujoco"]
    physics_timestep_s: float = Field(gt=0.0)
    control_timestep_s: float = Field(gt=0.0)


class SceneSpec(StrictModel):
    model_path: str = Field(min_length=1)
    table_size_m: tuple[float, float, float]


class RobotSpec(StrictModel):
    arm_model: str = Field(min_length=1)
    hand_model: str = Field(min_length=1)
    joint_order_path: str = Field(min_length=1)
    controller_id: str = Field(min_length=1)


class ContractSpec(StrictModel):
    observation: str = Field(min_length=1)
    action: str = Field(min_length=1)
    outcome: str = Field(min_length=1)


class CameraSpec(StrictModel):
    id: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)


class EnvironmentSpec(StrictModel):
    schema_version: Literal["environment_spec_v1"]
    environment_id: str
    runner: str = Field(min_length=1)
    simulator: SimulatorSpec
    scene: SceneSpec
    robot: RobotSpec
    contracts: ContractSpec
    cameras: tuple[CameraSpec, ...]


class ProtocolParameters(StrictModel):
    pregrasp_distance_m: float = Field(gt=0.0)
    palm_height_above_object_m: float = Field(gt=0.0)
    approach_duration_s: float = Field(gt=0.0)
    grip_command: float = Field(ge=-1.0, le=1.0)
    grip_command_type: Literal["normalized_position"]
    close_duration_s: float = Field(gt=0.0)
    lift_height_m: float = Field(gt=0.0)
    lift_duration_s: float = Field(gt=0.0)
    hold_duration_s: float = Field(gt=0.0)
    total_timeout_s: float = Field(gt=0.0)


class SuccessSpec(StrictModel):
    minimum_lift_m: float = Field(gt=0.0)
    required_hold_s: float = Field(gt=0.0)
    maximum_object_drop_m: float = Field(ge=0.0)


class ExecutionProtocolSpec(StrictModel):
    schema_version: Literal["execution_protocol_v1"]
    execution_protocol_id: str
    states: tuple[
        Literal[
            "reset",
            "load_object",
            "retarget",
            "move_to_pregrasp",
            "approach",
            "close_fingers",
            "lift",
            "hold",
            "score",
        ],
        ...,
    ]
    parameters: ProtocolParameters
    success: SuccessSpec


SpecT = TypeVar("SpecT", EnvironmentSpec, ExecutionProtocolSpec)


class Registry(Generic[SpecT]):
    """An immutable mapping of IDs to validated, content-addressed specs."""

    def __init__(self, entries: dict[str, SpecT]) -> None:
        self._entries = dict(entries)

    def resolve(self, entry_id: str, expected_hash: str | None = None) -> SpecT:
        try:
            spec = self._entries[entry_id]
        except KeyError as exc:
            raise RegistryError(f"unknown registry ID: {entry_id}") from exc
        actual_hash = canonical_sha256(spec)
        if expected_hash is not None and expected_hash != actual_hash:
            raise RegistryError(
                f"configuration hash mismatch for {entry_id}: "
                f"expected {expected_hash}, resolved {actual_hash}"
            )
        return spec

    def config_hash(self, entry_id: str) -> str:
        return canonical_sha256(self.resolve(entry_id))

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))


@dataclass(frozen=True)
class ResolvedCase:
    case: EvaluationCase
    environment: EnvironmentSpec
    environment_config_hash: str
    execution_protocol: ExecutionProtocolSpec
    execution_protocol_config_hash: str


def _load_yaml(name: str) -> dict:
    resource = files(__package__).joinpath("configs", name)
    with resource.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise RegistryError(f"registry {name} must contain a mapping")
    return payload


def load_environment_registry() -> Registry[EnvironmentSpec]:
    raw = _load_yaml("environments.yaml")
    entries: dict[str, EnvironmentSpec] = {}
    for entry_id, payload in raw.items():
        spec = EnvironmentSpec.model_validate(payload)
        if spec.environment_id != entry_id:
            raise RegistryError(
                f"environment key {entry_id!r} does not match embedded ID "
                f"{spec.environment_id!r}"
            )
        entries[entry_id] = spec
    return Registry(entries)


def load_protocol_registry() -> Registry[ExecutionProtocolSpec]:
    raw = _load_yaml("execution_protocols.yaml")
    entries: dict[str, ExecutionProtocolSpec] = {}
    for entry_id, payload in raw.items():
        spec = ExecutionProtocolSpec.model_validate(payload)
        if spec.execution_protocol_id != entry_id:
            raise RegistryError(
                f"protocol key {entry_id!r} does not match embedded ID "
                f"{spec.execution_protocol_id!r}"
            )
        entries[entry_id] = spec
    return Registry(entries)


def resolve_case(
    case: EvaluationCase,
    environments: Registry[EnvironmentSpec] | None = None,
    protocols: Registry[ExecutionProtocolSpec] | None = None,
) -> ResolvedCase:
    """Resolve and hash both registry references before simulation starts."""

    environments = environments or load_environment_registry()
    protocols = protocols or load_protocol_registry()
    environment = environments.resolve(
        case.environment.id, case.environment.expected_config_hash
    )
    protocol = protocols.resolve(
        case.execution_protocol.id,
        case.execution_protocol.expected_config_hash,
    )
    if environment.robot.hand_model != case.embodiment.id:
        raise RegistryError(
            f"embodiment {case.embodiment.id!r} does not match environment hand "
            f"{environment.robot.hand_model!r}"
        )
    return ResolvedCase(
        case=case,
        environment=environment,
        environment_config_hash=canonical_sha256(environment),
        execution_protocol=protocol,
        execution_protocol_config_hash=canonical_sha256(protocol),
    )
