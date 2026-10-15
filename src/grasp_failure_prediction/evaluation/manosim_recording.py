"""Pure recording contract for native right MANO/MANOSim coordinates.

This module runs no simulator, decoder, controller or predictor. Named hand
coordinates are gathered by a future adapter; scene qpos/DOF/actuator addresses
must never be inferred from these serialization positions. Quaternions are raw
wxyz ball qpos (extra rotations after MJCF body rest quaternions), not controls.
Asset identities and well-formed records do not certify transfer or physics.
There is deliberately no supported positive native admission certifier here.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from .prediction_recording import MappingDecision, MappingStatus, PHASES, PhysicalOutcome


NATIVE_DOMAIN = "manosim_right_native_67qpos_51qvel_51ctrl_v1"
CONTRACT_SCHEMA = "manosim_native_recording_contract_v1"
RECORDING_SCHEMA = "manosim_native_recording_v1"
UPSTREAM_REVISION = "c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac"
SLIDES = ("wrist_tx", "wrist_ty", "wrist_tz")
BALLS = ("wrist_rot",) + tuple(f"mano_joint_{i}" for i in range(1, 16))
QPOS_NAMES = SLIDES + tuple(f"{joint}:quat_{axis}" for joint in BALLS for axis in "wxyz")
QVEL_NAMES = tuple(f"{joint}:velocity" for joint in SLIDES) + tuple(
    f"{joint}:angular_{axis}" for joint in BALLS for axis in "xyz")
CTRL_NAMES = tuple(f"wrist_motor_t{axis}" for axis in "xyz") + tuple(
    (f"wrist_motor_r{axis}" if i == 0 else f"mano_motor_{i}_{axis}")
    for i in range(16) for axis in "xyz")
_SHA = re.compile(r"sha256:[a-f0-9]{64}")
QUATERNION_NORM_ATOL = 1e-8
CLOCK_ATOL_S = 1e-9
OBSERVABLE_KEYS = frozenset({
    "domain", "contract_sha256", "qpos_names", "qvel_names", "ctrl_names",
    "time_s", "action_interval_start_s", "phase", "phase_start_s", "phase_elapsed_s",
    "hand_qpos", "hand_qvel", "requested_ctrl", "effective_ctrl",
})
DIAGNOSTICS_POLICY = "separate simulator-clock artifact; forbidden as predictor input"
_PAYLOAD_KEYS = frozenset({"schema_version", "contract", "contract_sha256", "attempt",
    "native_admission", "artifacts_sha256", "recording", "diagnostics_policy"})


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_sha256(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Channel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: StrictStr
    unit: StrictStr
    frame: StrictStr


def _channels(names, units, frames):
    return tuple(Channel(name=name, unit=unit, frame=frame)
                 for name, unit, frame in zip(names, units, frames, strict=True))


QPOS_CHANNELS = _channels(QPOS_NAMES, ("m",) * 3 + ("1",) * 64,
    ("root_body_rest_slide_axes",) * 3 + ("ball_extra_after_body_rest_wxyz",) * 64)
QVEL_CHANNELS = _channels(QVEL_NAMES, ("m/s",) * 3 + ("rad/s",) * 48,
    ("native_slide_dof",) * 3 + ("native_ball_tangent_dof",) * 48)
CTRL_CHANNELS = _channels(CTRL_NAMES, ("m",) * 3 + ("rad",) * 48,
    ("scalar_slide_joint_transmission",) * 3 + ("scalar_ball_joint_gear_axis",) * 48)


class NativeRecordingContract(BaseModel):
    """Immutable serialization identity, separate from compiled scene addresses."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[CONTRACT_SCHEMA] = CONTRACT_SCHEMA
    domain: Literal[NATIVE_DOMAIN] = NATIVE_DOMAIN
    hand_id: Literal["manosim_right_native"] = "manosim_right_native"
    handedness: Literal["right"] = "right"
    upstream_revision: Literal[UPSTREAM_REVISION] = UPSTREAM_REVISION
    assets_sha256: StrictStr
    shape_sha256: StrictStr
    collider_sha256: StrictStr
    controller_sha256: StrictStr
    compiled_model_sha256: StrictStr
    controller_protocol_id: StrictStr
    sampling_protocol_id: StrictStr
    control_interval_s: float = Field(strict=True, gt=0, allow_inf_nan=False)
    quaternion_norm_atol: Literal[QUATERNION_NORM_ATOL] = QUATERNION_NORM_ATOL
    clock_atol_s: Literal[CLOCK_ATOL_S] = CLOCK_ATOL_S
    qpos_channels: tuple[Channel, ...] = QPOS_CHANNELS
    qvel_channels: tuple[Channel, ...] = QVEL_CHANNELS
    ctrl_channels: tuple[Channel, ...] = CTRL_CHANNELS
    state_timing: Literal["completed_interval_endpoint"] = "completed_interval_endpoint"
    control_timing: Literal["final_requested_and_actual_written_ctrl_of_preceding_interval"] = (
        "final_requested_and_actual_written_ctrl_of_preceding_interval")
    control_policy: Literal["no_inferred_clipping_or_pose_to_ctrl_conversion"] = (
        "no_inferred_clipping_or_pose_to_ctrl_conversion")

    @field_validator("assets_sha256", "shape_sha256", "collider_sha256", "controller_sha256", "compiled_model_sha256")
    @classmethod
    def digest(cls, value):
        if not _SHA.fullmatch(value):
            raise ValueError("explicit tagged SHA-256 required")
        return value

    @field_validator("controller_protocol_id", "sampling_protocol_id")
    @classmethod
    def identifier(cls, value):
        if not value.strip():
            raise ValueError("explicit nonempty versioned protocol identity required")
        return value

    @model_validator(mode="after")
    def exact_layout(self):
        if (self.qpos_channels != QPOS_CHANNELS or self.qvel_channels != QVEL_CHANNELS
                or self.ctrl_channels != CTRL_CHANNELS):
            raise ValueError("exact independent native 67/51/51 named units/frame layouts required")
        return self

    @property
    def contract_sha256(self):
        return canonical_sha256(self.model_dump(mode="json"))


class NativeAttemptMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attempt_id: StrictStr
    domain: Literal[NATIVE_DOMAIN] = NATIVE_DOMAIN
    hand_id: Literal["manosim_right_native"] = "manosim_right_native"
    source_generator: Literal["hug"] = "hug"
    contract_sha256: StrictStr
    object_id: StrictStr
    geometry_group_id: StrictStr
    observation_group_id: StrictStr
    proposal_group_id: StrictStr
    mapping: MappingDecision
    execution_status: Literal["not_executed", "completed", "runtime_error"]
    outcome: PhysicalOutcome | None = None
    provenance_hashes: dict[str, StrictStr]

    @field_validator("attempt_id", "object_id", "geometry_group_id", "observation_group_id", "proposal_group_id")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("nonempty identity required")
        return value

    @model_validator(mode="after")
    def label_separation(self):
        hashes = self.provenance_hashes
        if (not _SHA.fullmatch(self.contract_sha256) or not {"code", "configuration", "inputs"} <= hashes.keys()
                or any(not key.strip() or not _SHA.fullmatch(value) for key, value in hashes.items())):
            raise ValueError("named tagged provenance hashes and contract SHA-256 required")
        if self.mapping.status is not MappingStatus.OK:
            if self.execution_status != "not_executed" or self.outcome is not None:
                raise ValueError("mapping/setup refusals require not_executed and outcome null")
        elif self.execution_status == "completed":
            if self.outcome is None or not {"native_pose", "label_rule"} <= hashes.keys():
                raise ValueError("completed label requires native-pose/label-rule provenance and outcome")
        elif self.outcome is not None:
            raise ValueError("unexecuted/runtime-error outcomes must be null")
        return self


def native_admission(contract: NativeRecordingContract, evidence: Mapping | None = None) -> dict:
    """Assess identity only; no available evidence can activate a certifier.

    In particular a caller's PASS flag is not permission to collect. A future
    native certifier requires a separately implemented/versioned admission gate.
    """
    contract = _contract(contract)
    digest = evidence.get("contract_sha256") if isinstance(evidence, Mapping) else None
    matches = isinstance(digest, str) and digest == contract.contract_sha256
    return {"admitted": False, "reason": "no_supported_native_transfer_certifier_v1",
            "contract_sha256": contract.contract_sha256, "evidence_contract_matches": bool(matches)}


def _contract(value):
    return NativeRecordingContract.model_validate(value.model_dump(mode="json"))


def _numeric(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite numeric shape {shape}")
    return a.astype(np.float64, copy=True)


def _named(value, names, name):
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"{name} requires exact native named scalar coordinates")
    return np.asarray([_numeric(value[key], (), name).item() for key in names], dtype=np.float64)


def _quaternions(qpos):
    quaternions = qpos[..., 3:].reshape(-1, 16, 4)
    norms = np.sqrt(np.einsum("...i,...i->...", quaternions, quaternions, optimize=False))
    if not np.isfinite(norms).all() or np.any(np.abs(norms - 1.) > QUATERNION_NORM_ATOL):
        raise ValueError("raw native ball wxyz quaternions must be unit length; no normalization")


def validate_native_observables(arrays: Mapping, contract: NativeRecordingContract) -> None:
    contract = _contract(contract)
    if set(arrays) != OBSERVABLE_KEYS:
        raise ValueError("native observable allowlist differs; diagnostics are forbidden")
    for key, expected in (("domain", NATIVE_DOMAIN), ("contract_sha256", contract.contract_sha256)):
        if np.asarray(arrays[key]).shape != () or np.asarray(arrays[key]).item() != expected:
            raise ValueError("native domain/contract identity mismatch")
    for key, expected in (("qpos_names", QPOS_NAMES), ("qvel_names", QVEL_NAMES), ("ctrl_names", CTRL_NAMES)):
        names = np.asarray(arrays[key])
        if names.dtype.kind not in "US" or names.shape != (len(expected),) or tuple(names.tolist()) != expected:
            raise ValueError("native channel names/order mismatch")
    times = np.asarray(arrays["time_s"])
    if times.ndim != 1 or not len(times):
        raise ValueError("native recording requires endpoint samples")
    times = _numeric(times, times.shape, "endpoint times")
    starts = _numeric(arrays["action_interval_start_s"], times.shape, "interval starts")
    if (starts[0] != 0 or np.any(times <= starts) or np.any(np.diff(times) <= 0)
            or not np.allclose(starts[1:], times[:-1], rtol=0, atol=CLOCK_ATOL_S)
            or not np.allclose(times - starts, contract.control_interval_s, rtol=0, atol=CLOCK_ATOL_S)):
        raise ValueError("native interval endpoints must be positive, contiguous and contract aligned")
    phases = np.asarray(arrays["phase"])
    if phases.shape != times.shape or phases.dtype.kind not in "US" or not set(phases.tolist()) <= PHASES:
        raise ValueError("native phases must be named and endpoint aligned")
    phase_starts = _numeric(arrays["phase_start_s"], times.shape, "phase starts")
    elapsed = _numeric(arrays["phase_elapsed_s"], times.shape, "phase elapsed")
    expected_starts = np.empty(len(times)); current = 0.
    for i in range(len(times)):
        if i and phases[i] != phases[i - 1]:
            current = starts[i]
        expected_starts[i] = current
    if (not np.array_equal(phase_starts, expected_starts)
            or not np.allclose(elapsed, times - phase_starts, rtol=0, atol=CLOCK_ATOL_S)):
        raise ValueError("native phase clock differs from observed phase transitions")
    qpos = _numeric(arrays["hand_qpos"], (len(times), 67), "native qpos")
    _quaternions(qpos)
    for key in ("hand_qvel", "requested_ctrl", "effective_ctrl"):
        _numeric(arrays[key], (len(times), 51), key)


class NativeRecorder:
    """Array-only recorder; admission and pose-to-control conversion are absent."""
    def __init__(self, contract: NativeRecordingContract):
        self._contract_value = _contract(contract)
        self._rows = []
        self._diagnostics = []
        self._written = False

    @property
    def contract(self):
        return self._contract_value

    @property
    def sample_count(self):
        return len(self._rows)

    def append(self, *, time_s, action_interval_start_s, phase, qpos_by_name,
               qvel_by_name, requested_ctrl_by_name, effective_ctrl_by_name):
        if self._written:
            raise ValueError("native recording already finalized")
        qpos = _named(qpos_by_name, QPOS_NAMES, "qpos")
        _quaternions(qpos)
        row = {"time_s": _numeric(time_s, (), "time").item(),
               "action_interval_start_s": _numeric(action_interval_start_s, (), "interval start").item(),
               "phase": phase, "hand_qpos": qpos,
               "hand_qvel": _named(qvel_by_name, QVEL_NAMES, "qvel"),
               "requested_ctrl": _named(requested_ctrl_by_name, CTRL_NAMES, "requested ctrl"),
               "effective_ctrl": _named(effective_ctrl_by_name, CTRL_NAMES, "effective ctrl")}
        row["phase_start_s"] = (self._rows[-1]["phase_start_s"]
            if self._rows and phase == self._rows[-1]["phase"] else row["action_interval_start_s"])
        row["phase_elapsed_s"] = row["time_s"] - row["phase_start_s"]
        candidate = self._arrays(self._rows + [row])
        validate_native_observables(candidate, self.contract)
        self._rows.append(row)

    def append_diagnostics(self, *, time_s, values: Mapping):
        """Separate simulator-clock samples; never returned by arrays()/loader."""
        if self._written:
            raise ValueError("native recording already finalized")
        time = _numeric(time_s, (), "diagnostic time").item()
        if time < 0 or (self._diagnostics and time <= self._diagnostics[-1]["time_s"]):
            raise ValueError("diagnostic clocks must be nonnegative and strictly increasing")
        if not isinstance(values, Mapping) or not values or any(not isinstance(k, str) or not k or k == "time_s" for k in values):
            raise ValueError("nonempty named diagnostic fields required; time_s is reserved")
        row = {key: _numeric(value, np.asarray(value).shape, key) for key, value in values.items()}
        if self._diagnostics and (row.keys() != self._diagnostics[0].keys() - {"time_s"}
                or any(value.shape != self._diagnostics[0][key].shape for key, value in row.items())):
            raise ValueError("diagnostic fields/shapes must remain aligned")
        self._diagnostics.append(dict(row, time_s=time))

    def _arrays(self, rows):
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        arrays.update(domain=np.asarray(NATIVE_DOMAIN), contract_sha256=np.asarray(self.contract.contract_sha256),
                      qpos_names=np.asarray(QPOS_NAMES), qvel_names=np.asarray(QVEL_NAMES), ctrl_names=np.asarray(CTRL_NAMES))
        return arrays

    def arrays(self):
        if not self._rows:
            raise ValueError("native recording has no samples")
        arrays = self._arrays(self._rows)
        validate_native_observables(arrays, self.contract)
        return arrays

    def write(self, output_dir, metadata: NativeAttemptMetadata):
        if self._written:
            raise ValueError("native recording already finalized")
        metadata = NativeAttemptMetadata.model_validate(metadata.model_dump(mode="json"))
        contract = _contract(self.contract)
        if metadata.contract_sha256 != contract.contract_sha256:
            raise ValueError("native attempt/recording contract mismatch")
        if metadata.execution_status == "not_executed" and (self._rows or self._diagnostics):
            raise ValueError("unexecuted native refusal cannot contain observations/diagnostics")
        if metadata.execution_status == "completed" and not self._rows:
            raise ValueError("completed native attempt requires samples")
        output = Path(output_dir)
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("refusing to overwrite a nonempty native attempt directory")
        arrays = self.arrays() if self._rows else None
        output.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, values in (("observables.npz", arrays), ("diagnostics.npz",
                {key: np.stack([row[key] for row in self._diagnostics]) for key in self._diagnostics[0]}
                if self._diagnostics else None)):
            if values is not None:
                np.savez_compressed(output / name, **values)
                files[name] = file_sha256(output / name)
        payload = {"schema_version": RECORDING_SCHEMA, "contract": contract.model_dump(mode="json"),
            "contract_sha256": contract.contract_sha256, "attempt": metadata.model_dump(mode="json"),
            "native_admission": native_admission(contract), "artifacts_sha256": files,
            "recording": None if arrays is None else {"samples": len(self._rows)},
            "diagnostics_policy": DIAGNOSTICS_POLICY}
        with (output / "attempt.json").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        self._written = True
        return payload


def load_native_attempt(output_dir):
    """Verify native identity/artifact bytes; return observables, never diagnostics."""
    root = Path(output_dir)
    if root.is_symlink() or (root / "attempt.json").is_symlink():
        raise ValueError("native attempt paths cannot be symlinks")
    payload = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != RECORDING_SCHEMA:
        raise ValueError("unsupported native attempt schema; Shadow recordings are not convertible")
    if set(payload) != _PAYLOAD_KEYS or payload["diagnostics_policy"] != DIAGNOSTICS_POLICY:
        raise ValueError("native attempt field/diagnostic policy mismatch")
    contract = NativeRecordingContract.model_validate(payload["contract"])
    metadata = NativeAttemptMetadata.model_validate(payload["attempt"])
    if payload.get("contract_sha256") != contract.contract_sha256 or metadata.contract_sha256 != contract.contract_sha256:
        raise ValueError("native contract digest mismatch")
    if payload.get("native_admission") != native_admission(contract):
        raise ValueError("native recording cannot grant positive admission")
    files = payload.get("artifacts_sha256")
    if not isinstance(files, dict) or not set(files) <= {"observables.npz", "diagnostics.npz"}:
        raise ValueError("unexpected native artifacts")
    if {path.name for path in root.iterdir()} != {"attempt.json", *files}:
        raise ValueError("unindexed native attempt artifacts")
    for name, digest in files.items():
        if (root / name).is_symlink() or file_sha256(root / name) != digest:
            raise ValueError("native artifact hash mismatch")
    if metadata.execution_status == "not_executed" and files:
        raise ValueError("unexecuted native attempt cannot have artifacts")
    if "observables.npz" not in files:
        if payload.get("recording") is not None or metadata.execution_status == "completed":
            raise ValueError("native completed/sample metadata requires observables")
        return metadata, contract, payload, None
    with np.load(root / "observables.npz", allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    validate_native_observables(arrays, contract)
    recording = payload.get("recording")
    if (recording != {"samples": len(arrays["time_s"])}
            or type(recording.get("samples")) is not int):
        raise ValueError("native sample count mismatch")
    return metadata, contract, payload, arrays
