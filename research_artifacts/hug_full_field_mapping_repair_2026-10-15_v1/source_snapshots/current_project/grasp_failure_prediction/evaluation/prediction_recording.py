"""Versioned HUG attempt records with isolated robot observables and labels.

This opt-in format does not migrate historical noisy_rollout_pilot_v3 files.
Controls are endpoint samples of the preceding interval, not every physics
substep. Contacts and privileged simulator state live in a different artifact.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator, model_validator


RECORDING_SCHEMA = "hug_prediction_recording_v1"
OBSERVABLE_KEYS = frozenset({
    "time_s", "action_interval_start_s", "phase", "joint_names",
    "target_limits_rad", "hand_qpos", "hand_qvel",
    "requested_target_rad", "applied_target_rad",
})
PHASES = frozenset({
    "reset", "load_object", "retarget", "move_to_pregrasp", "approach",
    "close_fingers", "lift", "hold", "score",
})
PHYSICAL_FAILURES = frozenset({
    "approach_collision", "no_contact", "failed_acquisition",
    "slip_during_hold", "drop", "timeout",
})
_SHA256 = re.compile(r"sha256:[a-f0-9]{64}")


class MappingStatus(str, Enum):
    OK = "ok"
    REFUSED_INVALID_RETARGET = "refused_invalid_retarget"
    REFUSED_GUARD = "refused_guard"
    UNREACHABLE_PREGRASP = "unreachable_pregrasp"
    SETUP_ERROR = "setup_error"


class MappingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: MappingStatus
    code: StrictStr | None = None
    message: StrictStr | None = None

    @model_validator(mode="after")
    def validate_reason(self):
        if self.status is MappingStatus.OK:
            if self.code is not None or self.message is not None:
                raise ValueError("an ok mapping cannot carry a refusal reason")
        elif not self.code or not self.message:
            raise ValueError("a refused mapping requires a code and message")
        return self


class PhysicalOutcome(BaseModel):
    """A completed physical task label; setup/runtime errors are not outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    success: StrictBool
    failure_type: StrictStr | None = None
    label_rule_id: StrictStr
    first_failure_time_s: float | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def validate_physical_label(self):
        if not self.label_rule_id.strip():
            raise ValueError("a physical outcome requires a versioned label rule ID")
        if self.success:
            if self.failure_type is not None or self.first_failure_time_s is not None:
                raise ValueError("success cannot carry a failure type or failure time")
        elif self.failure_type not in PHYSICAL_FAILURES:
            raise ValueError("failure_type must be a physical task failure")
        if self.first_failure_time_s is not None and (
            not np.isfinite(self.first_failure_time_s) or self.first_failure_time_s < 0
        ):
            raise ValueError("failure time must be finite and nonnegative")
        return self


class AttemptMetadata(BaseModel):
    """Denominator-preserving metadata independent of any recorded trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_id: StrictStr
    source_generator: Literal["hug"] = "hug"
    hand_id: StrictStr
    execution_protocol_id: StrictStr
    object_id: StrictStr
    geometry_group_id: StrictStr
    observation_group_id: StrictStr
    proposal_group_id: StrictStr
    mapping: MappingDecision
    execution_status: Literal["not_executed", "completed", "runtime_error"]
    outcome: PhysicalOutcome | None = None
    provenance_hashes: dict[str, StrictStr]

    @field_validator(
        "attempt_id", "hand_id", "execution_protocol_id", "object_id",
        "geometry_group_id", "observation_group_id", "proposal_group_id",
    )
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("identifiers must be nonempty")
        return value

    @field_validator("provenance_hashes")
    @classmethod
    def valid_hashes(cls, value):
        if not {"code", "configuration", "inputs"} <= set(value):
            raise ValueError("provenance requires code, configuration and input hashes")
        if any(not key or not _SHA256.fullmatch(digest) for key, digest in value.items()):
            raise ValueError("provenance must contain named sha256 hashes")
        return value

    @model_validator(mode="after")
    def isolate_outcome(self):
        if self.mapping.status is not MappingStatus.OK:
            if self.execution_status != "not_executed" or self.outcome is not None:
                raise ValueError("mapping refusals must be unexecuted with outcome null")
        elif self.execution_status == "completed":
            if self.outcome is None:
                raise ValueError("completed physical execution requires an outcome")
            if not {"model", "assets", "native_pose", "mapped_pose", "label_rule"} <= set(self.provenance_hashes):
                raise ValueError("completed executions require model/asset/pose/label-rule hashes")
        elif self.outcome is not None:
            raise ValueError("unexecuted/runtime-error attempts must have outcome null")
        return self


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _numeric(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite numeric array with shape {shape}")
    return array.astype(np.float64, copy=True)


def validate_observables(arrays: dict[str, np.ndarray]) -> None:
    """Fail closed on missing/extra fields, alignment or control-limit errors."""
    if set(arrays) != OBSERVABLE_KEYS:
        raise ValueError("recording observable fields differ from the strict allowlist")
    times = arrays["time_s"]
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("recording needs at least one control endpoint")
    times = _numeric(times, times.shape, "time_s")
    if np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("control endpoints must be positive and strictly increasing")
    starts = _numeric(arrays["action_interval_start_s"], times.shape, "interval starts")
    if (np.any(starts < 0) or np.any(starts >= times)
            or not np.allclose(starts[1:], times[:-1], rtol=0, atol=1e-9)):
        raise ValueError("control intervals must be positive, contiguous and aligned")
    phases, names = arrays["phase"], arrays["joint_names"]
    if (phases.shape != times.shape or phases.dtype.kind not in "US"
            or not set(phases.tolist()) <= PHASES):
        raise ValueError("phases must be named and aligned with control endpoints")
    if (names.ndim != 1 or not len(names) or names.dtype.kind not in "US"
            or not all(names.tolist()) or len(set(names.tolist())) != len(names)):
        raise ValueError("joint names must be nonempty and unique")
    limits = _numeric(arrays["target_limits_rad"], (len(names), 2), "target limits")
    if np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("target lower limits cannot exceed upper limits")
    for field in ("hand_qpos", "hand_qvel", "requested_target_rad", "applied_target_rad"):
        _numeric(arrays[field], (len(times), len(names)), field)
    expected = np.clip(arrays["requested_target_rad"], limits[:, 0], limits[:, 1])
    if not np.array_equal(arrays["applied_target_rad"], expected):
        raise ValueError("applied endpoint targets must equal actual clipped requested controls")


class PredictionRecorder:
    """Collect immutable, named robot-state samples without running simulation."""

    def __init__(self, joint_names, target_limits_rad) -> None:
        names = np.asarray(joint_names)
        if (names.ndim != 1 or not len(names) or names.dtype.kind not in "US"
                or not all(names.tolist()) or len(set(names.tolist())) != len(names)):
            raise ValueError("joint names must be nonempty and unique")
        self.joint_names = names.astype(str).copy()
        self.target_limits_rad = _numeric(target_limits_rad, (len(names), 2), "target limits")
        if np.any(self.target_limits_rad[:, 0] > self.target_limits_rad[:, 1]):
            raise ValueError("target lower limits cannot exceed upper limits")
        self._rows: list[dict] = []
        self._diagnostics: list[dict] = []
        self._written = False

    @property
    def sample_count(self) -> int:
        return len(self._rows)

    def append(self, *, time_s: float, action_interval_start_s: float, phase: str,
               hand_qpos, hand_qvel, requested_target_rad, applied_target_rad,
               diagnostics: dict[str, np.ndarray] | None = None) -> None:
        if self._written:
            raise ValueError("cannot append to a finalized recording")
        if (isinstance(time_s, bool) or isinstance(action_interval_start_s, bool)
                or not np.isfinite(time_s) or not np.isfinite(action_interval_start_s)
                or action_interval_start_s < 0 or time_s <= action_interval_start_s
                or (self._rows and not np.isclose(action_interval_start_s,
                    self._rows[-1]["time_s"], rtol=0, atol=1e-9))):
            raise ValueError("control intervals must be finite, positive and contiguous")
        if phase not in PHASES:
            raise ValueError("unknown execution phase")
        fields = {"hand_qpos": hand_qpos, "hand_qvel": hand_qvel,
                  "requested_target_rad": requested_target_rad,
                  "applied_target_rad": applied_target_rad}
        row = {name: _numeric(value, (len(self.joint_names),), name)
               for name, value in fields.items()}
        expected = np.clip(row["requested_target_rad"], self.target_limits_rad[:, 0],
                           self.target_limits_rad[:, 1])
        if not np.array_equal(row["applied_target_rad"], expected):
            raise ValueError("applied endpoint target differs from the clipped actual control")
        diagnostic = {}
        for name, value in (diagnostics or {}).items():
            if not isinstance(name, str) or not name or name == "time_s":
                raise ValueError("diagnostic names must be nonempty; time_s is reserved")
            array = np.asarray(value)
            diagnostic[name] = _numeric(array, array.shape, name)
        if self._diagnostics and (
            diagnostic.keys() != self._diagnostics[0].keys()
            or any(value.shape != self._diagnostics[0][key].shape
                   for key, value in diagnostic.items())
        ):
            raise ValueError("diagnostic fields and shapes must align across samples")
        row.update(time_s=float(time_s), action_interval_start_s=float(action_interval_start_s), phase=phase)
        self._rows.append(row)
        self._diagnostics.append(diagnostic)

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._rows:
            raise ValueError("recording has no samples")
        arrays = {key: np.stack([row[key] for row in self._rows]) for key in self._rows[0]}
        arrays.update(joint_names=self.joint_names.copy(), target_limits_rad=self.target_limits_rad.copy())
        validate_observables(arrays)
        return arrays

    def write(self, output_dir: str | Path, metadata: AttemptMetadata) -> dict:
        """Write one fresh attempt; no append or historical schema conversion."""
        if self._written:
            raise ValueError("recording has already been finalized")
        if metadata.mapping.status is not MappingStatus.OK or metadata.execution_status == "not_executed":
            raise ValueError("unexecuted/refused attempts cannot contain physical samples")
        if not self._rows and metadata.execution_status == "runtime_error":
            payload = write_incomplete_attempt(output_dir, metadata)
            self._written = True
            return payload
        arrays = self.arrays()
        payload = _write_attempt(output_dir, metadata, arrays, self._diagnostics)
        self._written = True
        return payload


def _write_attempt(output_dir, metadata, arrays=None, diagnostics=None) -> dict:
    # Validate again: even model_construct()/a mutated provenance dictionary is
    # not a way to bypass physical-label isolation at the artifact boundary.
    metadata = AttemptMetadata.model_validate(metadata.model_dump(mode="json"))
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("refusing to overwrite a nonempty attempt directory")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    if arrays is not None:
        np.savez_compressed(output / "observables.npz", **arrays)
        files["observables.npz"] = file_sha256(output / "observables.npz")
        if diagnostics and diagnostics[0]:
            values = {key: np.stack([row[key] for row in diagnostics]) for key in diagnostics[0]}
            values["time_s"] = arrays["time_s"]
            np.savez_compressed(output / "diagnostics.npz", **values)
            files["diagnostics.npz"] = file_sha256(output / "diagnostics.npz")
    payload = {
        "schema_version": RECORDING_SCHEMA, "attempt": metadata.model_dump(mode="json"),
        "artifacts_sha256": files,
        "recording": None if arrays is None else {
            "samples": len(arrays["time_s"]),
            "measurement_timing": "joint state at the completed control-interval endpoint",
            "command_timing": "final effective clipped servo target of the preceding interval",
            "command_resolution": "control endpoints; physics-substep interpolation is not recorded",
            "diagnostics_policy": "separate artifact, forbidden as predictor input",
        },
    }
    write_json(output / "attempt.json", payload)
    return payload


def write_unexecuted_attempt(output_dir: str | Path, metadata: AttemptMetadata) -> dict:
    """Keep mapping refusals/setup errors in the attempted denominator."""
    if metadata.execution_status != "not_executed":
        raise ValueError("an unexecuted attempt requires execution_status=not_executed")
    return _write_attempt(output_dir, metadata)


def write_incomplete_attempt(output_dir: str | Path, metadata: AttemptMetadata) -> dict:
    """Preserve a refusal or runtime error even when no sample was reached.

    A runtime error has no physical outcome. It remains in the execution-attempt
    denominator whether it occurred before or after the first control endpoint.
    Use PredictionRecorder.write instead when a partial trace is available.
    """
    if metadata.execution_status not in {"not_executed", "runtime_error"}:
        raise ValueError("only incomplete attempts can be written without physical samples")
    return _write_attempt(output_dir, metadata)


def load_attempt(output_dir: str | Path) -> tuple[AttemptMetadata, dict, dict[str, np.ndarray] | None]:
    """Read labels and observable allowlist; never open the diagnostic artifact."""
    root = Path(output_dir)
    payload = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != RECORDING_SCHEMA:
        raise ValueError("unsupported attempt recording schema")
    metadata = AttemptMetadata.model_validate(payload["attempt"])
    files = payload.get("artifacts_sha256")
    if not isinstance(files, dict) or not set(files) <= {"observables.npz", "diagnostics.npz"}:
        raise ValueError("unexpected attempt artifacts")
    if metadata.execution_status == "not_executed":
        if files or payload.get("recording") is not None:
            raise ValueError("unexecuted attempts cannot contain recorded artifacts")
        return metadata, payload, None
    if metadata.execution_status == "runtime_error" and not files:
        if payload.get("recording") is not None:
            raise ValueError("a runtime error without samples must have recording null")
        return metadata, payload, None
    if "observables.npz" not in files or not isinstance(payload.get("recording"), dict):
        raise ValueError("executed attempts require recorded observables")
    if files["observables.npz"] != file_sha256(root / "observables.npz"):
        raise ValueError("observable artifact hash mismatch")
    with np.load(root / "observables.npz", allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    validate_observables(arrays)
    if payload["recording"].get("samples") != len(arrays["time_s"]):
        raise ValueError("recording sample count mismatch")
    return metadata, payload, arrays
