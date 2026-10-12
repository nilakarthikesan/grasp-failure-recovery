"""Score existing v3 development checks without training or new simulation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

from grasp_failure_prediction.prediction.dataset import _prefix
from grasp_failure_prediction.prediction.model import outcome_metrics, predict_episode


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def score(runs_root: Path, reference: Path, model: Path, output: Path) -> dict:
    repository = Path(__file__).resolve().parents[1]
    runs_root, reference, model, output = [p.resolve() for p in (runs_root, reference, model, output)]
    if output.exists():
        raise FileExistsError("use a new diagnostic output directory")
    reference_manifest = json.loads((reference / "manifest.json").read_text())
    for filename in ("plan.json", "summary.json", "protocol.json", "final_dynamic_replay_audit.json"):
        if sha(reference / filename) != reference_manifest["files_sha256"][filename]:
            raise ValueError(f"reference hash mismatch: {filename}")
    plan = json.loads((reference / "plan.json").read_text())
    previous = json.loads((reference / "summary.json").read_text())
    if plan["schema_version"] != "mug_clearance_validation_plan_v1":
        raise ValueError("unexpected source plan")
    entries = plan["episodes"]
    if len(entries) != 27 or len({e["episode_id"] for e in entries}) != 27:
        raise ValueError("expected the original 27 distinct planned checks")
    if {p.parent.name for p in runs_root.glob("*/collection_metadata.json")} != {e["episode_id"] for e in entries}:
        raise ValueError("raw inventory differs from the declared plan")
    for filename in ("plan.json", "summary.json"):
        if sha(runs_root / filename) != sha(reference / filename):
            raise ValueError(f"raw source differs from published {filename}")
    expected = {row["episode_id"]: row for row in previous["episodes"]}
    frozen_paths = [model, model.parent / "report.json",
                    repository / "src/grasp_failure_prediction/prediction/model.py",
                    repository / "src/grasp_failure_prediction/prediction/dataset.py"]
    frozen_before = {str(p): sha(p) for p in frozen_paths}
    training_report = json.loads((model.parent / "report.json").read_text())
    if model.name != "proprioception_and_commands.npz":
        raise ValueError("this diagnostic uses the saved state and command model")
    if sha(model) != "sha256:" + training_report["models"]["proprioception_and_commands"]["model_sha256"]:
        raise ValueError("model differs from its frozen report")
    if (training_report["prediction_time_s"] != 2.96 or training_report["history_steps"] != 20
            or sha(frozen_paths[2]) != "sha256:" + training_report["training_source_sha256"]["model.py"]):
        raise ValueError("prediction contract or implementation changed")

    raw_paths = [runs_root / f for f in ("plan.json", "summary.json")]
    for entry in entries:
        raw_paths.extend(p for p in (runs_root / entry["episode_id"]).iterdir() if p.is_file())
    raw_before = {str(p.relative_to(runs_root)): sha(p) for p in sorted(raw_paths)}
    output.mkdir(parents=True)
    write_json(output / "plan.json", dict(
        schema_version="corrected_mug_frozen_model_diagnostic_plan_v1", date="2026-10-12",
        source_plan_sha256=sha(reference / "plan.json"), source_summary_sha256=sha(reference / "summary.json"),
        source_runs_root=str(runs_root), reference_bundle=str(reference), model_path=str(model),
        frozen_before=frozen_before, raw_before=raw_before, analysis_script_sha256=sha(Path(__file__)),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        prediction_time_s=2.96, history_steps=20, failure_alert_threshold=0.5,
        selection="All original planned checks; score every completed eligible case; retain rejections separately",
        scope="Retrospective development diagnostic under corrected v3; no new simulation, training, tuning or untouched benchmark",
    ))
    records, excluded, X = [], [], []
    for entry in entries:
        episode_id = entry["episode_id"]
        episode = (runs_root / episode_id).resolve()
        if episode.parent != runs_root:
            raise ValueError("episode path escapes the declared run")
        metadata = json.loads((episode / "collection_metadata.json").read_text())
        old = expected[episode_id]
        if (metadata["status"] != old["status"] or metadata["label"] != old["label"]
                or metadata["noise_amplitude_rad"] != entry["noise_amplitude_rad"]
                or metadata["noise_seed"] != entry["noise_seed"]
                or metadata["execution_protocol_config_hash"] != reference_manifest["protocol_config_hash"]):
            raise ValueError(f"source metadata mismatch: {episode_id}")
        if metadata["status"] != "completed":
            if metadata["label"] is not None or any((episode / f).exists() for f in ("trajectory.npz", "actions.npz", "result.json")):
                raise ValueError("rejected source unexpectedly has completed inputs or a label")
            excluded.append(dict(episode_id=episode_id, object_id=metadata["object_id"],
                                 noise_amplitude_rad=entry["noise_amplitude_rad"], status=metadata["status"],
                                 label_success=None, p_failure=None, failure_alert=None,
                                 reason="Guard stopped execution; no complete eligible prefix or task outcome",
                                 guard_event=metadata.get("runtime_guard_event"),
                                 metadata_sha256=sha(episode / "collection_metadata.json")))
            continue
        prefix, names, cutoff = _prefix(episode)
        if names != training_report["input_contract"]["joint_names"]:
            raise ValueError("joint order differs from the original model")
        prefix_dir = output / "prediction_inputs" / episode_id
        prefix_dir.mkdir(parents=True)
        with np.load(episode / "trajectory.npz", allow_pickle=False) as trajectory, \
                np.load(episode / "actions.npz", allow_pickle=False) as actions:
            selected = np.flatnonzero(trajectory["time_s"] <= 2.96 + 1e-9)[-20:]
            np.savez_compressed(prefix_dir / "trajectory.npz", **{
                key: trajectory[key][selected] for key in ("time_s", "state", "hand_qpos")})
            np.savez_compressed(prefix_dir / "actions.npz", time_s=actions["time_s"][selected],
                                joint_names=actions["joint_names"], hand_qvel=actions["hand_qvel"][selected],
                                commanded_target_rad=actions["commanded_target_rad"][selected])
            if np.any(actions["episode_bias_rad"][:2] != 0):
                raise ValueError("wrist command noise is outside this collection's contract")
        portable_prefix, portable_names, portable_cutoff = _prefix(prefix_dir)
        if not np.array_equal(portable_prefix, prefix) or portable_names != names or portable_cutoff != cutoff:
            raise ValueError("portable input differs from raw causal prefix")
        prediction = predict_episode(model, prefix_dir)
        records.append(dict(episode_id=episode_id, object_id=metadata["object_id"],
                            role=entry["object_role"], observation_group_id=metadata["observation_group_id"],
                            proposal_group_id=metadata["group_id"], noise_amplitude_rad=entry["noise_amplitude_rad"],
                            p_success=prediction["p_success"], p_failure=prediction["p_failure"],
                            failure_alert=prediction["p_failure"] >= 0.5,
                            observed_through_s=prediction["observed_through_s"]))
        X.append(prefix)

    # Predictions are complete before outcome files are read for comparison.
    for row in records:
        episode = runs_root / row["episode_id"]
        result = json.loads((episode / "result.json").read_text())
        if result["status"] != "completed" or type(result["success"]) is not bool:
            raise ValueError("invalid completed outcome")
        label = int(result["success"])
        if label != expected[row["episode_id"]]["label"]:
            raise ValueError("result and published label disagree")
        row["label_success"] = label
        row["failure_type"] = result["failure_type"]
        row["classification"] = ("false_alarm" if label and row["failure_alert"] else
                                 "missed_failure" if not label and not row["failure_alert"] else
                                 "correct_success" if label else "detected_failure")

    def metrics(rows):
        values = outcome_metrics(np.array([r["label_success"] for r in rows]), np.array([r["p_success"] for r in rows]))
        values["p_failure_range"] = [min(r["p_failure"] for r in rows), max(r["p_failure"] for r in rows)]
        return values

    mugs = [r for r in records if r["role"] == "mug"]
    per_noise = []
    for amplitude in (0.0, 0.15, 0.6):
        rows = [r for r in mugs if r["noise_amplitude_rad"] == amplitude]
        stops = [r for r in excluded if r["noise_amplitude_rad"] == amplitude]
        per_noise.append(dict(noise_amplitude_rad=amplitude, planned=8, completed=len(rows),
                              stopped=len(stops), metrics=metrics(rows)))
    controls = [r for r in records if r["role"] == "control"]
    if len(records) != 26 or len(excluded) != 1 or len(mugs) != 23 or len(controls) != 3:
        raise ValueError("source membership differs from the declared diagnostic")
    frozen_after = {str(p): sha(p) for p in frozen_paths}
    raw_after = {str(p.relative_to(runs_root)): sha(p) for p in sorted(raw_paths)}
    if frozen_after != frozen_before or raw_after != raw_before:
        raise ValueError("frozen model, implementation or original recordings changed")
    report = dict(schema_version="corrected_mug_frozen_model_diagnostic_v1", date="2026-10-12",
                  planned=27, scored=26, stopped_unscored=1, overall=metrics(records),
                  mug=metrics(mugs), mug_by_noise=per_noise, controls=controls,
                  frozen_model_and_prediction_sources_unchanged=True, original_raw_files_unchanged=True,
                  frozen_after=frozen_after, prediction_time_s=2.96, failure_alert_threshold=0.5,
                  execution_protocol_id="fixed_grasp_lift_v3_table_clearance",
                  execution_protocol_config_hash=reference_manifest["protocol_config_hash"],
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  scope="Offline diagnostic of reused controller-development scenes and proposals",
                  limitations=["Two mug observation families and eight related proposals; noise draws are paired",
                               "Changed v3 wrist/controller conditions; not the historical unchanged v2 evaluation",
                               "Requested noise bounds do not equal all applied offsets because joint limits clip commands",
                               "Failure scores are not calibrated physical probabilities; no tuning or fitting performed",
                               "Eventual task outcome, not physical slip onset, live warning lead time or recovery"])
    columns = ["episode_id", "object_id", "role", "observation_group_id", "proposal_group_id",
               "noise_amplitude_rad", "observed_through_s", "label_success", "failure_type", "p_success",
               "p_failure", "failure_alert", "classification"]
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output / "dataset.npz", X=np.stack(X),
                        y=np.array([r["label_success"] for r in records], dtype=np.int64),
                        episode_ids=np.array([r["episode_id"] for r in records]),
                        joint_names=np.array(training_report["input_contract"]["joint_names"]))
    write_json(output / "report.json", report)
    write_json(output / "excluded.json", excluded)
    write_json(output / "manifest.json", dict(schema_version="corrected_mug_diagnostic_manifest_v1",
        raw_source_file_sha256=raw_before, frozen_file_sha256=frozen_before,
        files_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
        model_used_for_inference="proprioception_and_commands", no_training=True,
        input_features="Only 20 closing samples of 24 positions, 24 velocities and 24 applied commands"))
    return report


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, default=repository / "research_artifacts/mug_table_clearance_v1")
    parser.add_argument("--model", type=Path, default=repository / "research_artifacts/task_outcome_baseline_v1/proprioception_and_commands.npz")
    args = parser.parse_args()
    report = score(args.runs_root, args.reference_bundle, args.model, args.output)
    print(json.dumps({key: report[key] for key in ("planned", "scored", "stopped_unscored", "mug_by_noise", "controls")}, indent=2))


if __name__ == "__main__":
    main()
