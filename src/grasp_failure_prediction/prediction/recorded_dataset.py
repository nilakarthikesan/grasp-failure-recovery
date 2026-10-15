"""Export causal HUG prefixes from hug_prediction_recording_v1 attempts.

Cutoffs are an explicit protocol input. This format assigns no automatic split
and never reuses the historical scene IDs or observation_holdout_v1 policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from grasp_failure_prediction.evaluation.prediction_recording import (
    AttemptMetadata, MappingStatus, PhysicalOutcome, RECORDING_SCHEMA,
    file_sha256, load_attempt, write_json,
)


DATASET_SCHEMA = "hug_recorded_prefix_dataset_v1"
FEATURE_FIELDS = ("hand_qpos", "hand_qvel", "applied_target_rad")
ARRAY_KEYS = frozenset({
    "X", "y", "episode_ids", "group_ids", "observation_group_ids", "geometry_group_ids",
})


class PrefixSpec(BaseModel):
    """A declared close-only history ending exactly at a causal cutoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_id: StrictStr
    prediction_time_s: float = Field(strict=True, gt=0, allow_inf_nan=False)
    history_steps: StrictInt = Field(ge=2)
    control_timestep_s: float = Field(strict=True, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def sufficient_time(self):
        if not self.protocol_id.strip():
            raise ValueError("prefix protocol ID must be nonempty")
        if self.prediction_time_s < self.history_steps * self.control_timestep_s - 1e-9:
            raise ValueError("prefix cutoff cannot precede the declared history")
        return self


@dataclass(frozen=True)
class RecordedDataset:
    X: np.ndarray
    y: np.ndarray
    episode_ids: np.ndarray
    group_ids: np.ndarray
    observation_group_ids: np.ndarray
    geometry_group_ids: np.ndarray
    manifest: dict


def _prefix(arrays: dict[str, np.ndarray], spec: PrefixSpec) -> tuple[np.ndarray, list[float]]:
    times, phases = arrays["time_s"], arrays["phase"]
    prefix = np.flatnonzero(times <= spec.prediction_time_s + 1e-9)
    if len(prefix) < spec.history_steps:
        raise ValueError("insufficient samples before prediction cutoff")
    if not np.isclose(times[prefix[-1]], spec.prediction_time_s, rtol=0, atol=1e-9):
        raise ValueError("missing prediction-cutoff endpoint")
    selected = prefix[-spec.history_steps:]
    if (np.any(np.isin(phases[prefix], ["lift", "hold", "score"]))
            or not np.all(phases[selected] == "close_fingers")):
        raise ValueError("prediction history must be closing, before any lift/hold/score")
    if not np.allclose(np.diff(times[selected]), spec.control_timestep_s, rtol=0, atol=1e-9):
        raise ValueError("prediction history does not match the declared sample interval")
    values = np.concatenate([arrays[field][selected] for field in FEATURE_FIELDS], axis=1)
    with np.errstate(over="ignore"):
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("predictor values overflow float32")
    return values.reshape(-1), times[selected].tolist()


def _input_contract(spec: PrefixSpec, names: list[str]) -> dict:
    return {
        **spec.model_dump(mode="json"), "joint_names": names, "fields": list(FEATURE_FIELDS),
        "features": spec.history_steps * 3 * len(names), "dtype": "float32",
        "flatten_order": "time-major, then field, then joint",
        "excluded_features": [
            "requested targets", "contacts", "diagnostics", "full simulator state",
            "object state/properties", "outcome/failure labels", "mapping/filter status",
            "native/mapped pose metadata", "phase/time", "provenance/seeds",
        ],
    }


def export_recorded_dataset(attempt_dirs: list[str | Path], output_dir: str | Path,
                            *, prefix_spec: PrefixSpec) -> dict:
    """Keep every attempt in counts; only completed eligible physical labels train."""
    spec = PrefixSpec.model_validate(prefix_spec.model_dump(mode="json"))
    roots = [Path(path).resolve() for path in attempt_dirs]
    output = Path(output_dir).resolve()
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("attempt directories must be nonempty and unique")
    if any(output == root or output in root.parents or root in output.parents for root in roots):
        raise ValueError("derived output must be separate from all raw attempt directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("refusing to overwrite a nonempty dataset directory")
    seen, names, cohort, execution_cohort = set(), None, None, None
    features, labels, rows, excluded = [], [], [], []
    mapped_count = executed_count = labeled_count = 0
    for root in roots:
        attempt, payload, arrays = load_attempt(root)
        if attempt.attempt_id in seen:
            raise ValueError("duplicate attempt_id")
        seen.add(attempt.attempt_id)
        source = {"attempt_id": attempt.attempt_id,
                  "attempt_sha256": file_sha256(root / "attempt.json"),
                  "hand_id": attempt.hand_id,
                  "execution_protocol_id": attempt.execution_protocol_id,
                  "mapping_status": attempt.mapping.status.value,
                  "execution_status": attempt.execution_status}
        current_execution = (attempt.hand_id, attempt.execution_protocol_id)
        if execution_cohort is None:
            execution_cohort = current_execution
        elif execution_cohort != current_execution:
            raise ValueError("one dataset cannot mix hand/controller/physical label protocols")
        mapped_count += attempt.mapping.status is MappingStatus.OK
        executed_count += attempt.execution_status in {"completed", "runtime_error"}
        labeled_count += attempt.outcome is not None
        if attempt.mapping.status is not MappingStatus.OK:
            excluded.append(dict(source, reason="mapping_refusal", code=attempt.mapping.code))
            continue
        if attempt.execution_status != "completed" or attempt.outcome is None:
            excluded.append(dict(source, reason="unexecuted_or_runtime_error"))
            continue
        outcome = attempt.outcome
        current_cohort = {
            "hand_id": attempt.hand_id, "execution_protocol_id": attempt.execution_protocol_id,
            "label_rule_id": outcome.label_rule_id,
            "label_rule_sha256": attempt.provenance_hashes["label_rule"],
        }
        if cohort is None:
            cohort = current_cohort
        elif cohort != current_cohort:
            raise ValueError("one dataset cannot mix hand/controller/physical label protocols")
        if (outcome.failure_type == "approach_collision"
                or (outcome.first_failure_time_s is not None
                    and outcome.first_failure_time_s <= spec.prediction_time_s + 1e-9)):
            excluded.append(dict(source, reason="failure_at_or_before_cutoff",
                                 failure_type=outcome.failure_type))
            continue
        current_names = arrays["joint_names"].astype(str).tolist()
        if names is None:
            names = current_names
        elif names != current_names:
            raise ValueError("joint ordering differs across attempt recordings")
        feature, observed_times = _prefix(arrays, spec)
        features.append(feature)
        labels.append(int(outcome.success))
        rows.append({
            **source, "source_schema": RECORDING_SCHEMA,
            "object_id": attempt.object_id, "group_id": attempt.proposal_group_id,
            "observation_group_id": attempt.observation_group_id,
            "geometry_group_id": attempt.geometry_group_id,
            "source_task_label": int(outcome.success), "failure_type": outcome.failure_type,
            "first_failure_time_s": outcome.first_failure_time_s,
            "observed_times_s": observed_times, "provenance_hashes": attempt.provenance_hashes,
            "observables_sha256": payload["artifacts_sha256"]["observables.npz"],
        })
    if not rows:
        raise ValueError("no completed cutoff-eligible physical attempts to export")
    # A proposal family may never silently acquire a different observation or
    # geometry identity. A downstream split must group these families together.
    sources, observations = {}, {}
    for row in rows:
        binding = (row["observation_group_id"], row["geometry_group_id"], row["object_id"])
        if sources.setdefault(row["group_id"], binding) != binding:
            raise ValueError("one proposal group has conflicting observation/geometry/object identities")
        observation_binding = (row["geometry_group_id"], row["object_id"])
        if observations.setdefault(row["observation_group_id"], observation_binding) != observation_binding:
            raise ValueError("one observation group has conflicting geometry/object identities")
    arrays = {
        "X": np.stack(features), "y": np.asarray(labels, dtype=np.uint8),
        "episode_ids": np.asarray([row["attempt_id"] for row in rows]),
        **{key: np.asarray([row[field] for row in rows]) for key, field in (
            ("group_ids", "group_id"), ("observation_group_ids", "observation_group_id"),
            ("geometry_group_ids", "geometry_group_id"))},
    }
    manifest = {
        "schema_version": DATASET_SCHEMA, "source_schema": RECORDING_SCHEMA,
        "exporter_sha256": file_sha256(Path(__file__)), "cohort": cohort,
        "input_contract": _input_contract(spec, names),
        "label_convention": "1=final task success; 0=completed final task failure",
        "prediction_scope": "close-prefix task-outcome forecasting; not physical slip onset",
        "split_policy": "unassigned; freeze a separate grouped split protocol before fitting",
        "episodes": rows, "excluded": excluded,
        "counts": {"attempted": len(roots), "mapped": int(mapped_count),
                   "executed": int(executed_count), "physically_labeled": int(labeled_count),
                   "included": len(rows), "excluded": len(excluded),
                   "success": sum(labels), "failure": len(labels) - sum(labels),
                   "proposal_groups": len(sources),
                   "observation_groups": len(set(arrays["observation_group_ids"].tolist())),
                   "geometry_groups": len(set(arrays["geometry_group_ids"].tolist()))},
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "dataset.npz", **arrays)
    manifest["dataset_sha256"] = file_sha256(output / "dataset.npz")
    write_json(output / "manifest.json", manifest)
    return manifest


def load_recorded_dataset(output_dir: str | Path) -> RecordedDataset:
    """Validate the new feature allowlist without invoking historical split logic."""
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA or manifest.get("source_schema") != RECORDING_SCHEMA:
        raise ValueError("unsupported recorded-prefix dataset schema")
    if manifest.get("dataset_sha256") != file_sha256(root / "dataset.npz"):
        raise ValueError("recorded-prefix dataset hash mismatch")
    contract = manifest["input_contract"]
    spec = PrefixSpec.model_validate({key: contract[key] for key in PrefixSpec.model_fields})
    names = contract["joint_names"]
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or not all(isinstance(name, str) and name for name in names)
            or contract != _input_contract(spec, names)):
        raise ValueError("recorded-prefix feature contract differs from the strict allowlist")
    with np.load(root / "dataset.npz", allow_pickle=False) as source:
        if set(source.files) != ARRAY_KEYS:
            raise ValueError("unexpected predictor arrays")
        arrays = {key: source[key].copy() for key in source.files}
    X, y = arrays["X"], arrays["y"]
    if (y.ndim != 1 or X.dtype != np.float32 or X.ndim != 2 or X.shape != (len(y), contract["features"])
            or not np.isfinite(X).all() or y.dtype != np.uint8 or y.shape != (len(X),)
            or not np.isin(y, [0, 1]).all()):
        raise ValueError("invalid recorded-prefix features or physical labels")
    rows = manifest["episodes"]
    if len(rows) != len(X) or len({row["attempt_id"] for row in rows}) != len(rows):
        raise ValueError("recorded-prefix episode identity mismatch")
    for key, field in (("episode_ids", "attempt_id"), ("group_ids", "group_id"),
                       ("observation_group_ids", "observation_group_id"),
                       ("geometry_group_ids", "geometry_group_id")):
        if (arrays[key].shape != y.shape or arrays[key].dtype.kind not in "US"
                or arrays[key].tolist() != [row[field] for row in rows]):
            raise ValueError("recorded-prefix grouping identities are misaligned")
    if (any(row["mapping_status"] != "ok" or row["execution_status"] != "completed"
            or type(row["source_task_label"]) is not int for row in rows)
            or y.tolist() != [row["source_task_label"] for row in rows]):
        raise ValueError("only mapped completed physical labels are trainable")
    cohort = manifest["cohort"]
    sources, observations = {}, {}
    for row in rows:
        outcome = PhysicalOutcome(
            success=bool(row["source_task_label"]), failure_type=row["failure_type"],
            label_rule_id=cohort["label_rule_id"], first_failure_time_s=row["first_failure_time_s"],
        )
        AttemptMetadata(
            attempt_id=row["attempt_id"], hand_id=cohort["hand_id"],
            execution_protocol_id=cohort["execution_protocol_id"], object_id=row["object_id"],
            geometry_group_id=row["geometry_group_id"], observation_group_id=row["observation_group_id"],
            proposal_group_id=row["group_id"], mapping={"status": "ok"},
            execution_status="completed", outcome=outcome, provenance_hashes=row["provenance_hashes"],
        )
        if (row["provenance_hashes"]["label_rule"] != cohort["label_rule_sha256"]
                or outcome.failure_type == "approach_collision"
                or (outcome.first_failure_time_s is not None
                    and outcome.first_failure_time_s <= spec.prediction_time_s + 1e-9)):
            raise ValueError("recorded-prefix label protocol or causal eligibility mismatch")
        expected_times = spec.prediction_time_s + np.arange(1 - spec.history_steps, 1) * spec.control_timestep_s
        if (np.asarray(row["observed_times_s"]).shape != (spec.history_steps,)
                or not np.allclose(row["observed_times_s"], expected_times, rtol=0, atol=1e-9)):
            raise ValueError("recorded-prefix observation times violate the causal cutoff")
        binding = (row["observation_group_id"], row["geometry_group_id"], row["object_id"])
        observation_binding = (row["geometry_group_id"], row["object_id"])
        if (sources.setdefault(row["group_id"], binding) != binding
                or observations.setdefault(row["observation_group_id"], observation_binding) != observation_binding):
            raise ValueError("recorded-prefix grouping sources conflict")
    excluded = manifest["excluded"]
    audit_rows = rows + excluded
    if len({row["attempt_id"] for row in audit_rows}) != len(audit_rows):
        raise ValueError("included and excluded attempt identities must be unique")
    if any((row["hand_id"], row["execution_protocol_id"]) !=
           (cohort["hand_id"], cohort["execution_protocol_id"]) for row in audit_rows):
        raise ValueError("recorded-prefix attempt denominators mix hand/controller protocols")
    for row in excluded:
        MappingStatus(row["mapping_status"])
        if row["execution_status"] not in {"not_executed", "completed", "runtime_error"}:
            raise ValueError("unknown excluded execution status")
        if row["mapping_status"] != "ok":
            if row["execution_status"] != "not_executed" or row["reason"] != "mapping_refusal":
                raise ValueError("excluded mapping refusals cannot be physically executed")
        elif row["reason"] != ("failure_at_or_before_cutoff" if row["execution_status"] == "completed"
                                else "unexecuted_or_runtime_error"):
            raise ValueError("excluded attempt reason disagrees with execution status")
    expected_counts = {
        "attempted": len(audit_rows), "mapped": sum(row["mapping_status"] == "ok" for row in audit_rows),
        "executed": sum(row["execution_status"] in {"completed", "runtime_error"} for row in audit_rows),
        "physically_labeled": sum(row["execution_status"] == "completed" for row in audit_rows),
        "included": len(rows), "excluded": len(excluded), "success": int(y.sum()),
        "failure": len(y) - int(y.sum()), "proposal_groups": len(sources),
        "observation_groups": len(observations),
        "geometry_groups": len(set(arrays["geometry_group_ids"].tolist())),
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("recorded-prefix attempt denominator mismatch")
    return RecordedDataset(**arrays, manifest=manifest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-manifest", type=Path, required=True,
                        help="JSON list of attempt directories relative to this manifest")
    parser.add_argument("--prefix-protocol", type=Path, required=True,
                        help="JSON PrefixSpec declared before collection/analysis")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = json.loads(args.attempts_manifest.read_text(encoding="utf-8"))
    if not isinstance(paths, list) or not all(isinstance(path, str) and path for path in paths):
        raise ValueError("attempts manifest must be a nonempty list of directory paths")
    spec = PrefixSpec.model_validate_json(args.prefix_protocol.read_text(encoding="utf-8"))
    report = export_recorded_dataset(
        [args.attempts_manifest.parent / path for path in paths], args.output, prefix_spec=spec,
    )
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
