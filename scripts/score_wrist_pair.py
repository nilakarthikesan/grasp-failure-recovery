"""Score one declared wrist pair from causal prefixes, then read task outcomes.

No fitting, threshold selection, altered inputs or simulator execution occurs.
Guard stops remain unlabeled and unscored even if partial motion was saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from grasp_failure_prediction.prediction.dataset import _prefix
from grasp_failure_prediction.prediction.model import (
    load_model, outcome_metrics, predict_episode, summarize_prefix,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def score(runs_root: Path, pose_manifest: Path, output: Path):
    repository = Path(__file__).resolve().parents[1]
    if output.exists():
        raise FileExistsError("Choose a new score output directory")
    plan = json.loads((runs_root / "plan.json").read_text())
    poses = json.loads(pose_manifest.read_text())
    if sha(pose_manifest) != plan["pose_manifest_sha256"]:
        raise ValueError("Pose manifest changed after declaration")
    if plan["pair_id"] != poses["first_pair_id"] or plan["postures"] != ["A", "B"]:
        raise ValueError("Expected the first declared eligible pair, A and B")
    frozen = dict(plan["frozen_files_sha256"])
    frozen.update(plan["paired_source_sha256"])
    for name, digest in frozen.items():
        if sha(repository / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    model_path = repository / poses["model_contract"]["model_file"]
    if sha(model_path) != poses["model_contract"]["model_sha256"]:
        raise ValueError("Frozen model changed")
    original_report = json.loads((model_path.parent / "report.json").read_text())
    raw_paths = sorted(p for p in runs_root.rglob("*") if p.is_file())
    raw_before = {str(p.relative_to(runs_root)): sha(p) for p in raw_paths}
    output.mkdir(parents=True)
    (output / "plan.json").write_bytes((runs_root / "plan.json").read_bytes())
    (output / "pose_manifest.json").write_bytes(pose_manifest.read_bytes())
    model = load_model(model_path)
    wrist_indices = np.array([0, 1, 24, 25, 48, 49])
    wrist_features = np.concatenate([wrist_indices + 72 * block for block in range(4)])
    active_wrist = np.isin(np.flatnonzero(model.normalizer.active), wrist_features)
    records, excluded, inputs = [], [], []
    for posture in plan["postures"]:
        episode = runs_root / posture
        metadata = json.loads((episode / "collection_metadata.json").read_text())
        if metadata["status"] != "completed":
            if metadata.get("label") is not None:
                raise ValueError("Stopped attempt must not have a binary label")
            excluded.append(dict(posture=posture, status=metadata["status"],
                                 label_success=None, p_success=None, p_failure=None,
                                 failure_alert=None, guard_event=metadata.get("runtime_guard_event"),
                                 reason=metadata.get("error")))
            continue
        if metadata["noise_amplitude_rad"] != 0.0:
            raise ValueError("Pair must use zero noise")
        prefix, names, cutoff = _prefix(episode)
        if names != original_report["input_contract"]["joint_names"]:
            raise ValueError("Joint order differs from frozen model")
        destination = output / "prediction_inputs" / posture
        destination.mkdir(parents=True)
        with np.load(episode / "trajectory.npz", allow_pickle=False) as trajectory, \
                np.load(episode / "actions.npz", allow_pickle=False) as actions:
            selected = np.flatnonzero(trajectory["time_s"] <= 2.96 + 1e-9)[-20:]
            if np.any(actions["episode_bias_rad"] != 0):
                raise ValueError("Nonzero command bias in zero-noise pair")
            np.savez_compressed(destination / "trajectory.npz", **{
                key: trajectory[key][selected] for key in ("time_s", "state", "hand_qpos")})
            np.savez_compressed(destination / "actions.npz", time_s=actions["time_s"][selected],
                                joint_names=actions["joint_names"],
                                hand_qvel=actions["hand_qvel"][selected],
                                commanded_target_rad=actions["commanded_target_rad"][selected])
        portable, portable_names, portable_cutoff = _prefix(destination)
        if not np.array_equal(prefix, portable) or names != portable_names or cutoff != portable_cutoff:
            raise ValueError("Portable causal input differs from raw prefix")
        prediction = predict_episode(model_path, destination)
        summary = summarize_prefix(prefix[None, :])
        terms = model.normalizer.transform(summary)[0] * model.weights
        records.append(dict(posture=posture, status="completed", observed_through_s=cutoff,
                            p_success=prediction["p_success"], p_failure=prediction["p_failure"],
                            failure_alert=prediction["p_failure"] >= 0.5,
                            success_logit=float(terms.sum() + model.intercept),
                            wrist_contribution=float(terms[active_wrist].sum()),
                            nonwrist_contribution=float(terms[~active_wrist].sum()),
                            intercept=float(model.intercept)))
        inputs.append(prefix)
    # This file is saved before outcome files are opened. Only the whitelisted
    # earlier observations entered prediction; full motion stays retrospective.
    write(output / "predictions_before_outcomes.json", records)
    for row in records:
        episode = runs_root / row["posture"]
        result = json.loads((episode / "result.json").read_text())
        metadata = json.loads((episode / "collection_metadata.json").read_text())
        if result["status"] != "completed" or type(result["success"]) is not bool:
            raise ValueError("Invalid completed task outcome")
        label = int(result["success"])
        if label != metadata["label"]:
            raise ValueError("Label and result disagree")
        row.update(label_success=label, failure_type=result["failure_type"],
                   classification=("false_alarm" if label and row["failure_alert"] else
                                   "correct_success" if label else
                                   "detected_failure" if row["failure_alert"] else "missed_failure"),
                   measured_task_outcome=result)
    metrics = (outcome_metrics(np.array([r["label_success"] for r in records]),
                              np.array([r["p_success"] for r in records])) if records else None)
    delta = None
    if len(records) == 2:
        a, b = records
        delta = dict(success_logit_B_minus_A=b["success_logit"] - a["success_logit"],
                     wrist_contribution_B_minus_A=b["wrist_contribution"] - a["wrist_contribution"],
                     nonwrist_contribution_B_minus_A=b["nonwrist_contribution"] - a["nonwrist_contribution"],
                     p_failure_B_minus_A=b["p_failure"] - a["p_failure"],
                     both_successful=bool(a["label_success"] and b["label_success"]),
                     alerts_differ=a["failure_alert"] != b["failure_alert"])
    raw_after = {str(p.relative_to(runs_root)): sha(p) for p in raw_paths}
    if raw_after != raw_before or any(sha(repository / name) != digest for name, digest in frozen.items()):
        raise ValueError("Original sources or recordings changed during scoring")
    report = dict(schema_version="frozen_wrist_pair_model_diagnostic_v1", date="2026-10-12",
                  pair_id=plan["pair_id"], pose_manifest_sha256=sha(pose_manifest),
                  planned=2, completed_scored=len(records), stopped_unscored=len(excluded),
                  prediction_time_s=2.96, history_steps=20, failure_alert_threshold=0.5,
                  model_unchanged=True, no_training_or_tuning=True, offline_causal_prefix_scoring=True,
                  records=records, excluded=excluded, paired_difference=delta, metrics=metrics,
                  frozen_file_sha256=frozen, raw_source_sha256=raw_before,
                  scope="One declared development pair. Outcomes concern grasp/lift/hold, not slip timing or recovery.",
                  comparison_qualification="Use pose-manifest training-range flags; two unfamiliar poses test sensitivity, not a familiar-versus-unfamiliar transfer contrast.")
    write(output / "report.json", report)
    if records:
        np.savez_compressed(output / "dataset.npz", X=np.stack(inputs),
                            y=np.array([r["label_success"] for r in records]),
                            postures=np.array([r["posture"] for r in records]),
                            joint_names=np.array(names))
        with (output / "predictions.csv").open("w", newline="") as stream:
            columns = [key for key in records[0] if key != "measured_task_outcome"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: value for key, value in row.items() if key in columns} for row in records)
    write(output / "manifest.json", dict(schema_version="wrist_pair_diagnostic_manifest_v1",
          files_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
          frozen_file_sha256=frozen, raw_source_sha256=raw_before))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pose-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.runs_root.resolve(), args.pose_manifest.resolve(), args.output.resolve())
    print(json.dumps({key: result[key] for key in ("pair_id", "completed_scored", "stopped_unscored", "paired_difference")}, indent=2))
