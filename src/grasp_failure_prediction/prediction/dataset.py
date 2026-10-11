"""Export a causal robot-state prefix and retrospective observation holdouts.

Raw simulator diagnostics are deliberately never predictor columns. The labels
retain the collected task rule; they are not physical slip-onset supervision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


PREDICTION_TIME_S = 2.96
HISTORY_STEPS = 20
CONTROL_TIMESTEP_S = 0.04
FIELDS = ("hand_qpos", "hand_qvel", "commanded_target_rad")
SCENE_ROLES = {9100: "train", 9101: "validation", 9102: "train", 9103: "test"}
ARRAY_KEYS = {
    "X", "y", "episode_ids", "group_ids", "observation_group_ids", "geometry_group_ids"
}
LABEL_RULE = {
    "minimum_lift_m": 0.14,
    "required_hold_s": 2.0,
    "maximum_object_drop_m": 0.02,
}


@dataclass(frozen=True)
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    episode_ids: np.ndarray
    group_ids: np.ndarray
    observation_group_ids: np.ndarray
    geometry_group_ids: np.ndarray
    manifest: dict
    splits: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _label_rule() -> dict:
    return dict(LABEL_RULE, convention="1=task success; 0=completed task failure",
                scope="frozen final task outcome, not physical slip or future-horizon loss",
                additional_checks="manipulation contact, no approach collision, no timeout")


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def split_observation_groups(rows: list[dict]) -> dict:
    """Assign declared capture roles without consulting labels or outcomes."""
    seen_ids: set[str] = set()
    observation_sources: dict[str, tuple[int, str]] = {}
    proposal_sources: dict[str, tuple[str, str, int]] = {}
    scene_observations: dict[int, set[str]] = {seed: set() for seed in SCENE_ROLES}
    partitions = {
        role: {"episode_ids": [], "observation_group_ids": [], "group_ids": [],
               "geometry_group_ids": []}
        for role in ("train", "validation", "test")
    }
    for row in rows:
        episode = _text(row.get("episode_id"), "episode_id")
        if episode in seen_ids:
            raise ValueError("duplicate episode_id")
        seen_ids.add(episode)
        seed = row.get("scene_seed")
        if type(seed) is not int or seed not in SCENE_ROLES:
            raise ValueError("scene_seed must identify declared capture 9100–9103")
        obs = _text(row.get("observation_group_id"), "observation_group_id")
        group = _text(row.get("group_id"), "group_id")
        geometry = _text(row.get("geometry_group_id"), "geometry_group_id")
        source = (seed, geometry)
        if observation_sources.setdefault(obs, source) != source:
            raise ValueError("one observation group has conflicting scene/geometry sources")
        proposal_source = (obs, geometry, seed)
        if proposal_sources.setdefault(group, proposal_source) != proposal_source:
            raise ValueError("one proposal group crosses observation sources")
        scene_observations[seed].add(obs)
        partition = partitions[SCENE_ROLES[seed]]
        partition["episode_ids"].append(episode)
        for key, value in (("observation_group_ids", obs), ("group_ids", group),
                           ("geometry_group_ids", geometry)):
            partition[key].append(value)
    if any(len(groups) != 1 for groups in scene_observations.values()):
        raise ValueError("the declared split requires exactly four observation groups")
    scene_geometries = {
        seed: observation_sources[next(iter(groups))][1]
        for seed, groups in scene_observations.items()
    }
    if (scene_geometries[9100] != scene_geometries[9101]
            or scene_geometries[9102] != scene_geometries[9103]
            or scene_geometries[9100] == scene_geometries[9102]):
        raise ValueError("declared cube/apple scene roles must share two distinct geometries")
    for partition in partitions.values():
        for key in ("observation_group_ids", "group_ids", "geometry_group_ids"):
            partition[key] = sorted(set(partition[key]))
        partition["counts"] = {
            "episodes": len(partition["episode_ids"]),
            "observation_groups": len(partition["observation_group_ids"]),
            "proposal_groups": len(partition["group_ids"]),
            "geometry_groups": len(partition["geometry_group_ids"]),
        }
    return dict(partitions=partitions, schema_version="observation_holdout_v1",
                scene_roles={str(seed): role for seed, role in SCENE_ROLES.items()},
                policy="declared scenes, not outcome-stratified; retrospective exploratory holdout",
                limitation="validation is cube B, test is apple B; both geometries occur in train; "
                "source outcomes were previously audited, so this is not an untouched final benchmark")


def _feature_columns(joint_names: list[str]) -> list[dict]:
    return [
        {"column": frame * 72 + field_index * 24 + joint_index,
         "history_index": frame,
         "frame_offset_s": (frame - HISTORY_STEPS + 1) * CONTROL_TIMESTEP_S,
         "field": field, "joint_name": joint,
         "unit": "rad/s" if field == "hand_qvel" else "rad"}
        for frame in range(HISTORY_STEPS)
        for field_index, field in enumerate(FIELDS)
        for joint_index, joint in enumerate(joint_names)
    ]


def _prefix(episode: Path) -> tuple[np.ndarray, list[str], float]:
    with np.load(episode / "trajectory.npz", allow_pickle=False) as trajectory, \
            np.load(episode / "actions.npz", allow_pickle=False) as actions:
        times = trajectory["time_s"]
        action_times = actions["time_s"]
        if (times.ndim != 1 or times.dtype.kind not in "fiu" or not len(times)
                or action_times.dtype.kind not in "fiu" or action_times.shape != times.shape
                or not np.array_equal(times, action_times)
                or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
            raise ValueError("trajectory/action timestamps must be equal, finite and increasing")
        state = trajectory["state"]
        if state.shape != times.shape or state.dtype.kind not in "US":
            raise ValueError("trajectory phases must align with timestamps")
        state = state.astype(str)
        prefix = np.flatnonzero(times <= PREDICTION_TIME_S + 1e-9)
        if len(prefix) < HISTORY_STEPS:
            raise ValueError("insufficient samples before prediction cutoff")
        if not np.isclose(times[prefix[-1]], PREDICTION_TIME_S, rtol=0, atol=1e-9):
            raise ValueError("missing prediction-cutoff endpoint")
        selected = prefix[-HISTORY_STEPS:]
        if (not np.all(state[selected] == "close_fingers")
                or np.any(np.isin(state[prefix], ["lift", "hold", "score"]))):
            raise ValueError("selected history must be CLOSE, before any lift/hold/score")
        if not np.allclose(np.diff(times[selected]), CONTROL_TIMESTEP_S, rtol=0, atol=1e-9):
            raise ValueError("selected history must use 40 ms control intervals")
        names = actions["joint_names"]
        if (names.shape != (24,) or names.dtype.kind not in "US"
                or len(set(names.tolist())) != 24 or not all(names.tolist())):
            raise ValueError("expected 24 unique named hand joints")
        blocks = []
        for values in (trajectory["hand_qpos"], actions["hand_qvel"],
                       actions["commanded_target_rad"]):
            if values.shape != (len(times), 24) or values.dtype.kind not in "fiu":
                raise ValueError("observable arrays must have timestamp-aligned (T, 24) shape")
            selected_values = values[selected]
            if not np.all(np.isfinite(selected_values)):
                raise ValueError("selected prefix contains nonfinite observable values")
            converted = selected_values.astype(np.float32)
            if not np.all(np.isfinite(converted)):
                raise ValueError("selected prefix overflows float32")
            blocks.append(converted)
        return np.concatenate(blocks, axis=1).reshape(-1), names.astype(str).tolist(), float(times[selected[-1]])


def export_dataset(runs_root: str | Path, output_dir: str | Path, *,
                   prediction_time_s: float = PREDICTION_TIME_S,
                   history_steps: int = HISTORY_STEPS) -> dict:
    """Export the frozen 20×72 pre-lift task-outcome baseline into fresh output."""
    if (type(history_steps) is not int or history_steps != HISTORY_STEPS
            or isinstance(prediction_time_s, bool)
            or not np.isfinite(prediction_time_s)
            or not np.isclose(prediction_time_s, PREDICTION_TIME_S, rtol=0, atol=1e-9)):
        raise ValueError("this contract fixes prediction_time_s=2.96 and history_steps=20")
    root, output = Path(runs_root).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("derived output must be separate from the raw collection")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("refusing to overwrite nonempty output")
    index_path = root / "index.jsonl"
    indexed = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    if not indexed or not all(isinstance(row, dict) for row in indexed):
        raise ValueError("collection index must contain episode records")
    rows, features, labels, excluded = [], [], [], []
    provenance_sources = {}
    joint_names, scorer_hash, source_commit = None, None, None
    seen = set()
    provenance_keys = (
        "geometry_content_hash", "geometry_asset_files_sha256", "proposal_sha256",
        "camera_sha256", "observation_files_sha256", "observation_manifest_sha256",
        "scene_contract_sha256", "scene_contract_status", "inference_report_sha256",
        "hug_inference_provenance", "source_file_sha256", "runtime_versions", "observation_source",
    )
    for entry in indexed:
        episode_id = entry.get("episode_id")
        if not isinstance(episode_id, str) or not re.fullmatch(r"episode_[0-9]{6}", episode_id):
            raise ValueError("unsafe or malformed episode_id")
        if episode_id in seen:
            raise ValueError("duplicate indexed episode_id")
        seen.add(episode_id)
        episode = root / episode_id
        if entry.get("status") != "completed" or entry.get("label") is None:
            if entry.get("label") is not None:
                raise ValueError("error records must remain unlabeled")
            excluded.append({"episode_id": episode_id, "reason": "error_or_unlabeled"})
            continue
        metadata = _json(episode / "collection_metadata.json")
        if metadata.get("schema_version") != "noisy_rollout_pilot_v3":
            raise ValueError("unsupported source collection metadata schema")
        if metadata.get("episode_id") != episode_id:
            raise ValueError("index/metadata episode identity is inconsistent")
        label = metadata.get("label")
        if (metadata.get("status") != "completed" or type(label) is not int
                or label not in (0, 1) or type(entry.get("label")) is not int
                or entry["label"] != label):
            raise ValueError("completed index/metadata require consistent integer labels 0/1")
        result = _json(episode / "result.json")
        if (result.get("status") != "completed" or type(result.get("success")) is not bool
                or int(result["success"]) != label
                or result.get("failure_type") != metadata.get("failure_type")):
            raise ValueError("metadata/result task labels are inconsistent")
        resolved = _json(episode / "resolved_case.json")
        if resolved["execution_protocol"]["success"] != LABEL_RULE:
            raise ValueError("source task thresholds differ from the frozen label rule")
        seed = resolved["case"]["seed"]
        current_scorer = _text(metadata["source_file_sha256"]["scoring.py"], "scorer hash")
        current_commit = _text(result["resolved_execution"]["code_commit"], "source commit")
        if (not re.fullmatch(r"sha256:[a-f0-9]{64}", current_scorer)
                or not re.fullmatch(r"[a-f0-9]{40}", current_commit)):
            raise ValueError("invalid recorded scorer hash/source commit")
        if scorer_hash is None:
            scorer_hash, source_commit = current_scorer, current_commit
        if (current_scorer, current_commit) != (scorer_hash, source_commit):
            raise ValueError("mixed label scorer/source commits are unsupported")
        feature, names, observed_at = _prefix(episode)
        if joint_names is None:
            joint_names = names
        if names != joint_names:
            raise ValueError("joint ordering differs across episodes")
        row = {key: metadata[key] for key in ("group_id", "observation_group_id", "geometry_group_id")}
        for key in row:
            if entry.get(key) != row[key]:
                raise ValueError("index/metadata grouping identities are inconsistent")
        source_provenance = {key: metadata.get(key) for key in provenance_keys}
        source_provenance["code_commit"] = current_commit
        provenance_id = _canonical_sha256(source_provenance)
        provenance_sources[provenance_id] = source_provenance
        row.update(episode_id=episode_id, scene_seed=seed, observed_at_s=observed_at,
                   source_task_label=label, source_provenance_id=provenance_id,
                   source_schema=metadata["schema_version"],
                   raw_files_sha256={name: _sha256(episode / name) for name in (
                       "actions.npz", "trajectory.npz", "collection_metadata.json",
                       "result.json", "resolved_case.json")})
        rows.append(row)
        features.append(feature)
        labels.append(label)
    if not rows:
        raise ValueError("no completed labeled episodes to export")
    splits = split_observation_groups(rows)
    X, y = np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.uint8)
    arrays = {
        "X": X, "y": y,
        "episode_ids": np.asarray([row["episode_id"] for row in rows]),
        "group_ids": np.asarray([row["group_id"] for row in rows]),
        "observation_group_ids": np.asarray([row["observation_group_id"] for row in rows]),
        "geometry_group_ids": np.asarray([row["geometry_group_id"] for row in rows]),
    }
    manifest = {
        "schema_version": "task_prefix_dataset_v1", "source_collection_name": root.name,
        "source_files_sha256": {name: _sha256(root / name) for name in ("index.jsonl", "plan.json", "summary.json")},
        "source_commit": source_commit, "label_scorer_sha256": scorer_hash,
        "exporter_sha256": _sha256(Path(__file__)),
        "label_rule": _label_rule(),
        "input_contract": {
            "prediction_time_s": PREDICTION_TIME_S, "history_steps": HISTORY_STEPS,
            "control_timestep_s": CONTROL_TIMESTEP_S, "joint_names": joint_names,
            "fields": list(FIELDS), "flatten_order": "time-major, then field, then joint",
            "dtype": "float32", "features": 1440,
            "frame_offsets_s": [(i - HISTORY_STEPS + 1) * CONTROL_TIMESTEP_S for i in range(HISTORY_STEPS)],
            "excluded_features": ["outcomes", "events", "contacts", "object state/properties",
                                  "root commands", "noise settings/seeds", "phase/time", "full simulator state"],
        },
        "feature_columns": _feature_columns(joint_names), "episodes": rows, "excluded": excluded,
        "provenance_sources": provenance_sources,
        "counts": {"indexed": len(indexed), "included": len(rows), "excluded": len(excluded),
                   "success": int(y.sum()), "failure": int(len(y) - y.sum()),
                   "observation_groups": len(set(arrays["observation_group_ids"].tolist())),
                   "proposal_groups": len(set(arrays["group_ids"].tolist())),
                   "geometry_groups": len(set(arrays["geometry_group_ids"].tolist()))},
        "partition_class_counts": {
            role: {"success": sum(labels[i] for i, row in enumerate(rows) if row["episode_id"] in splits["partitions"][role]["episode_ids"]),
                   "failure": sum(1 - labels[i] for i, row in enumerate(rows) if row["episode_id"] in splits["partitions"][role]["episode_ids"])}
            for role in ("train", "validation", "test")},
        "limitations": ["retrospective exploratory observation holdout; source outcomes already inspected",
                        "validation cube B and test apple B; both geometries present in training",
                        "initial RGB-D not included as model input; no per-frame vision or tactile sensing",
                        "four initial observation families; no unseen-geometry, hardware or recovery result"],
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "dataset.npz", **arrays)
    _write_json(output / "splits.json", splits)
    manifest.update(dataset_sha256=_sha256(output / "dataset.npz"),
                    splits_sha256=_sha256(output / "splits.json"))
    _write_json(output / "manifest.json", manifest)
    return manifest


def load_dataset(output_dir: str | Path) -> DatasetBundle:
    """Validate a portable export before exposing its fixed observable columns."""
    root = Path(output_dir)
    manifest, splits = _json(root / "manifest.json"), _json(root / "splits.json")
    if manifest.get("schema_version") != "task_prefix_dataset_v1":
        raise ValueError("unsupported exported dataset schema")
    if manifest.get("label_rule") != _label_rule():
        raise ValueError("exported label rule differs from the frozen task thresholds")
    scorer_hash = manifest.get("label_scorer_sha256")
    if (not isinstance(scorer_hash, str)
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", scorer_hash)):
        raise ValueError("invalid frozen scorer hash")
    if (manifest.get("dataset_sha256") != _sha256(root / "dataset.npz")
            or manifest.get("splits_sha256") != _sha256(root / "splits.json")):
        raise ValueError("exported dataset/split hash mismatch")
    contract = manifest["input_contract"]
    if (contract["prediction_time_s"] != PREDICTION_TIME_S
            or contract["history_steps"] != HISTORY_STEPS or contract["fields"] != list(FIELDS)
            or contract["features"] != 1440 or len(contract["joint_names"]) != 24
            or len(set(contract["joint_names"])) != 24
            or not all(isinstance(name, str) and name for name in contract["joint_names"])
            or contract["control_timestep_s"] != CONTROL_TIMESTEP_S
            or contract["dtype"] != "float32"
            or contract["flatten_order"] != "time-major, then field, then joint"
            or contract["frame_offsets_s"] != [(i - HISTORY_STEPS + 1) * CONTROL_TIMESTEP_S for i in range(HISTORY_STEPS)]
            or manifest["feature_columns"] != _feature_columns(contract["joint_names"])):
        raise ValueError("exported feature contract does not match the observable whitelist")
    with np.load(root / "dataset.npz", allow_pickle=False) as data:
        if set(data.files) != ARRAY_KEYS:
            raise ValueError("unexpected exported arrays")
        arrays = {key: data[key].copy() for key in ARRAY_KEYS}
    X, y = arrays["X"], arrays["y"]
    if (X.dtype != np.float32 or X.ndim != 2 or X.shape[1] != 1440
            or not np.all(np.isfinite(X)) or y.dtype != np.uint8
            or y.shape != (len(X),) or not np.all(np.isin(y, [0, 1]))):
        raise ValueError("invalid exported features/labels")
    for key in ARRAY_KEYS - {"X", "y"}:
        if arrays[key].shape != y.shape or arrays[key].dtype.kind not in "US":
            raise ValueError("exported identifiers must be aligned string arrays")
    rows = manifest["episodes"]
    if (any(type(row.get("source_task_label")) is not int for row in rows)
            or y.tolist() != [row["source_task_label"] for row in rows]):
        raise ValueError("exported labels differ from recorded source task labels")
    provenance_sources = manifest.get("provenance_sources", {})
    if not isinstance(provenance_sources, dict) or not provenance_sources:
        raise ValueError("missing provenance sources")
    for identity, provenance in provenance_sources.items():
        if (not isinstance(provenance, dict) or identity != _canonical_sha256(provenance)
                or provenance.get("source_file_sha256", {}).get("scoring.py") != scorer_hash
                or provenance.get("code_commit") != manifest.get("source_commit")):
            raise ValueError("provenance source hash/scorer/commit mismatch")
    references = {row.get("source_provenance_id") for row in rows}
    if references != set(provenance_sources):
        raise ValueError("invalid provenance source references")
    for key, row_key in (("episode_ids", "episode_id"), ("group_ids", "group_id"),
                         ("observation_group_ids", "observation_group_id"),
                         ("geometry_group_ids", "geometry_group_id")):
        if arrays[key].astype(str).tolist() != [row[row_key] for row in rows]:
            raise ValueError("manifest/array identifiers are inconsistent")
    if splits != split_observation_groups(rows):
        raise ValueError("exported observation split violates the declared group policy")
    excluded = manifest["excluded"]
    expected_counts = {
        "indexed": len(X) + len(excluded), "included": len(X), "excluded": len(excluded),
        "success": int(y.sum()), "failure": int(len(y) - y.sum()),
        "observation_groups": len(set(arrays["observation_group_ids"].tolist())),
        "proposal_groups": len(set(arrays["group_ids"].tolist())),
        "geometry_groups": len(set(arrays["geometry_group_ids"].tolist())),
    }
    expected_classes = {
        role: {"success": sum(int(y[i]) for i, row in enumerate(rows) if row["episode_id"] in splits["partitions"][role]["episode_ids"]),
               "failure": sum(1 - int(y[i]) for i, row in enumerate(rows) if row["episode_id"] in splits["partitions"][role]["episode_ids"])}
        for role in ("train", "validation", "test")}
    if (manifest.get("counts") != expected_counts
            or manifest.get("partition_class_counts") != expected_classes):
        raise ValueError("exported label/group counts disagree with manifest")
    return DatasetBundle(**arrays, manifest=manifest, splits=splits)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-time", type=float, default=PREDICTION_TIME_S)
    parser.add_argument("--history-steps", type=int, default=HISTORY_STEPS)
    args = parser.parse_args(argv)
    manifest = export_dataset(args.runs_root, args.output,
                              prediction_time_s=args.prediction_time, history_steps=args.history_steps)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
