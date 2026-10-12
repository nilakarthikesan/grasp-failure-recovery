"""Score the seven predeclared remaining wrist pairs without changing the model.

All new predictions are saved before any task-result JSON is parsed. This
analysis never executes a simulator, selects poses, fits or tunes a model.
Stopped attempts stay null and unscored, even if they have a partial prefix.
The historical first-pair bundle is verified and combined without rewriting it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from grasp_failure_prediction.prediction.dataset import _prefix
from grasp_failure_prediction.prediction.model import (
    load_model, outcome_metrics, predict_episode, summarize_prefix,
)


MANIFEST_SHA256 = "354f245ba6663974c92a132895709ea8d41dd5578942dc7af8e526f75dbb8ee1"
MODEL_SHA256 = "16c9941a51130daea818c6ad227ef1e3a02e4be9ed1c7aa5a8e8f3e023428beb"
POSTURES = ("A", "B")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value: str) -> str:
    return value.removeprefix("sha256:")


def inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def bound(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"Bound path escapes its source root: {relative}")
    return path


def verify_files(repository: Path, frozen: dict[str, str]) -> None:
    for relative, expected in frozen.items():
        if sha(bound(repository, relative)) != digest(expected):
            raise ValueError(f"Frozen experiment source changed: {relative}")


def file_bindings(plan: dict) -> dict[str, str]:
    frozen = dict(plan["frozen_files_sha256"])
    for relative, expected in plan["paired_source_sha256"].items():
        if relative in frozen and digest(frozen[relative]) != digest(expected):
            raise ValueError("Conflicting frozen source bindings")
        frozen[relative] = expected
    return frozen


def verify_first_bundle(first: Path, expected_report_sha: str) -> dict:
    if sha(first / "report.json") != digest(expected_report_sha):
        raise ValueError("Historical first-pair report differs from declaration")
    saved = read(first / "manifest.json")
    expected = saved["files_sha256"]
    present = {name for name in inventory(first) if name != "manifest.json"}
    if present != set(expected):
        raise ValueError("Historical first-pair payload inventory changed")
    for relative, value in expected.items():
        if sha(bound(first, relative)) != digest(value):
            raise ValueError(f"Historical first-pair payload changed: {relative}")
    return read(first / "report.json")


def metrics(records: list[dict]) -> dict | None:
    if not records:
        return None
    return outcome_metrics(np.array([row["label_success"] for row in records]),
                           np.array([row["p_success"] for row in records]))


def difference(records: list[dict]) -> dict | None:
    lookup = {row["posture"]: row for row in records}
    if set(lookup) != set(POSTURES):
        return None
    a, b = [lookup[posture] for posture in POSTURES]
    return dict(success_logit_B_minus_A=b["success_logit"] - a["success_logit"],
                wrist_contribution_B_minus_A=b["wrist_contribution"] - a["wrist_contribution"],
                nonwrist_contribution_B_minus_A=b["nonwrist_contribution"] - a["nonwrist_contribution"],
                p_failure_B_minus_A=b["p_failure"] - a["p_failure"],
                both_successful=bool(a["label_success"] and b["label_success"]),
                alerts_differ=a["failure_alert"] != b["failure_alert"])


def summarize_pairs(pair_reports: list[dict]) -> dict:
    records = [row for pair in pair_reports for row in pair["records"]]
    stops = [row for pair in pair_reports for row in pair["excluded"]]
    matrix = dict(both_success=0, A_success_B_failure=0,
                  A_failure_B_success=0, both_failure=0, incomplete=0)
    warnings = dict(both_warn=0, neither_warn=0,
                    A_warn_B_clear=0, A_clear_B_warn=0)
    both_success_ids, changed_ids = [], []
    for pair in pair_reports:
        lookup = {row["posture"]: row for row in pair["records"]}
        if set(lookup) != set(POSTURES):
            matrix["incomplete"] += 1
            continue
        a, b = [lookup[posture] for posture in POSTURES]
        key = ("both_success" if a["label_success"] and b["label_success"] else
               "A_success_B_failure" if a["label_success"] else
               "A_failure_B_success" if b["label_success"] else "both_failure")
        matrix[key] += 1
        if a["label_success"] != b["label_success"]:
            changed_ids.append(pair["pair_id"])
        if key == "both_success":
            both_success_ids.append(pair["pair_id"])
            warning_key = ("both_warn" if a["failure_alert"] and b["failure_alert"] else
                           "A_warn_B_clear" if a["failure_alert"] else
                           "A_clear_B_warn" if b["failure_alert"] else "neither_warn")
            warnings[warning_key] += 1
    return dict(planned_pairs=len(pair_reports), planned_attempts=2 * len(pair_reports),
                completed_scored=len(records), stopped_unscored=len(stops),
                paired_outcome_matrix=matrix,
                both_success_pair_ids=both_success_ids,
                outcome_changing_pair_ids=changed_ids,
                both_success_warning_matrix=warnings,
                overall_metrics=metrics(records),
                metrics_by_posture={posture: metrics([row for row in records
                                                      if row["posture"] == posture])
                                    for posture in POSTURES},
                failure_recall_qualification=("Undefined when there are no completed failures; "
                                              "small development counts do not estimate population reliability."),
                scope="Predeclared reused development proposals; not an untouched generalization benchmark.")


def score(runs_root: Path, pose_manifest: Path, first_bundle: Path,
          output: Path, scoring_declaration: Path | None = None) -> dict:
    repository = Path(__file__).resolve().parents[1]
    runs_root, pose_manifest, first_bundle, output = [path.resolve() for path in
                                                    (runs_root, pose_manifest, first_bundle, output)]
    if output.exists():
        raise FileExistsError("Choose a new score output directory")
    if any(root == output or root in output.parents for root in (runs_root, first_bundle)):
        raise ValueError("Derived scores must be outside preserved recordings and first bundle")
    declaration_path = (scoring_declaration or runs_root / "scoring_declaration.json").resolve()
    declaration = read(declaration_path)
    own_relative = str(Path(__file__).resolve().relative_to(repository))
    if (declaration["script_path"] != own_relative
            or digest(declaration["script_sha256"]) != sha(Path(__file__).resolve())
            or digest(declaration["runs_plan_sha256"]) != sha(runs_root / "plan.json")
            or digest(declaration["pose_manifest_sha256"]) != MANIFEST_SHA256
            or declaration.get("no_training_or_tuning") is not True):
        raise ValueError("Scoring analysis is not bound to its final declared implementation")
    analysis_frozen = declaration.get("analysis_files_sha256", {})
    if own_relative in analysis_frozen and digest(analysis_frozen[own_relative]) != sha(Path(__file__).resolve()):
        raise ValueError("Scoring implementation has conflicting analysis bindings")
    verify_files(repository, analysis_frozen)
    if sha(pose_manifest) != MANIFEST_SHA256:
        raise ValueError("Frozen pose manifest changed")
    poses, plan = read(pose_manifest), read(runs_root / "plan.json")
    expected_cases = [case for case in poses["cases"]
                      if case["pair_id"] != poses["first_pair_id"]]
    pair_ids = [case["pair_id"] for case in expected_cases]
    if (len(expected_cases) != 7 or not all(case["eligible"] for case in expected_cases)
            or len(set(pair_ids)) != 7 or plan["pair_ids"] != pair_ids
            or plan["postures"] != list(POSTURES)
            or digest(plan["pose_manifest_sha256"]) != MANIFEST_SHA256
            or digest(plan["first_report_sha256"]) != digest(declaration["first_report_sha256"])
            or plan["noise_amplitude_rad"] != 0.0):
        raise ValueError("Expected exactly the seven declared remaining eligible zero-noise pairs")
    present_pairs = {path.name for path in runs_root.iterdir() if path.is_dir()}
    if present_pairs != set(pair_ids):
        raise ValueError("Raw pair inventory differs from the exact declaration")
    contract = poses["model_contract"]
    if (contract["prediction_time_s"] != 2.96 or contract["history_steps"] != 20
            or contract["failure_alert_threshold"] != 0.5
            or not contract["weights_normalization_features_unchanged"]
            or not contract["outcomes_not_inputs"]):
        raise ValueError("Frozen model input and threshold contract changed")
    frozen = file_bindings(plan)
    verify_files(repository, frozen)
    model_path = bound(repository, contract["model_file"])
    if sha(model_path) != MODEL_SHA256 or digest(contract["model_sha256"]) != MODEL_SHA256:
        raise ValueError("Original model bytes changed")
    original_report = read(model_path.parent / "report.json")
    raw_before, first_before = inventory(runs_root), inventory(first_bundle)
    # Verify the historical payload now, without using its outcomes in selection
    # or scoring. Its fixed scores/outcomes are combined only after new inference.
    if sha(first_bundle / "report.json") != digest(declaration["first_report_sha256"]):
        raise ValueError("Historical first-pair report binding differs")
    if sha(first_bundle / "pose_manifest.json") != MANIFEST_SHA256:
        raise ValueError("Historical first-pair pose selection differs")
    model = load_model(model_path)
    active_indices = np.flatnonzero(model.normalizer.active)
    active_wrist = active_indices % 24 < 2
    output.mkdir(parents=True)
    shutil.copyfile(runs_root / "plan.json", output / "plan.json")
    shutil.copyfile(pose_manifest, output / "pose_manifest.json")
    shutil.copyfile(declaration_path, output / "scoring_declaration.json")
    records, excluded, inputs = [], [], []
    for case in expected_cases:
        pair_id = case["pair_id"]
        pair_root = runs_root / pair_id
        pair_plan = read(pair_root / "plan.json")
        pair_frozen = file_bindings(pair_plan)
        if (pair_plan["pair_id"] != pair_id or pair_plan["postures"] != list(POSTURES)
                or digest(pair_plan["pose_manifest_sha256"]) != MANIFEST_SHA256
                or pair_plan["noise_amplitude_rad"] != 0.0
                or pair_frozen != frozen):
            raise ValueError(f"Pair declaration differs: {pair_id}")
        if {path.name for path in pair_root.iterdir() if path.is_dir()} != set(POSTURES):
            raise ValueError(f"Expected exactly A and B raw recordings: {pair_id}")
        for posture in POSTURES:
            episode = pair_root / posture
            metadata = read(episode / "collection_metadata.json")
            if (metadata["pair_id"] != pair_id or metadata["posture"] != posture
                    or digest(metadata["frozen_pose_manifest_sha256"]) != MANIFEST_SHA256
                    or metadata["noise_amplitude_rad"] != 0.0
                    or metadata["noise_mode"] != "disabled"
                    or metadata["predictor_evaluated_by_executor"] is not False):
                raise ValueError("Execution identity, predictor isolation or zero-noise condition changed")
            if (episode / "actions.npz").exists():
                with np.load(episode / "actions.npz", allow_pickle=False) as actions:
                    if any(np.any(actions[key] != 0.0) for key in
                           ("episode_bias_rad", "requested_noise_rad", "applied_noise_rad")):
                        raise ValueError("Nonzero saved disturbance in a zero-noise pair")
                    if actions["joint_names"].tolist() != contract["joint_names"]:
                        raise ValueError("Saved joint order differs from frozen model")
                    if not np.allclose(actions["commanded_target_rad"][:, :2],
                                       case["poses"][posture]["wrist_rad"],
                                       rtol=0, atol=poses["gates"]["pose_reproduction_absolute_tolerance"]):
                        raise ValueError("Saved wrist commands differ from the declared pose")
            if metadata["status"] != "completed":
                if metadata.get("label") is not None or (episode / "result.json").exists():
                    raise ValueError("Stopped attempt must have no task result or binary label")
                excluded.append(dict(pair_id=pair_id, posture=posture, status=metadata["status"],
                                     label_success=None, p_success=None, p_failure=None,
                                     failure_alert=None, guard_event=metadata.get("runtime_guard_event"),
                                     reason=metadata.get("error")))
                continue
            prefix, names, cutoff = _prefix(episode)
            if names != original_report["input_contract"]["joint_names"]:
                raise ValueError("Prefix joint order differs from frozen model")
            destination = output / "prediction_inputs" / pair_id / posture
            destination.mkdir(parents=True)
            with np.load(episode / "trajectory.npz", allow_pickle=False) as trajectory, \
                    np.load(episode / "actions.npz", allow_pickle=False) as actions:
                selected = np.flatnonzero(trajectory["time_s"] <= 2.96 + 1e-9)[-20:]
                np.savez_compressed(destination / "trajectory.npz", **{
                    key: trajectory[key][selected] for key in ("time_s", "state", "hand_qpos")})
                np.savez_compressed(destination / "actions.npz", time_s=actions["time_s"][selected],
                                    joint_names=actions["joint_names"],
                                    hand_qvel=actions["hand_qvel"][selected],
                                    commanded_target_rad=actions["commanded_target_rad"][selected])
            portable, portable_names, portable_cutoff = _prefix(destination)
            if (not np.array_equal(prefix, portable) or names != portable_names
                    or cutoff != portable_cutoff):
                raise ValueError("Portable causal input differs from raw prefix")
            prediction = predict_episode(model_path, destination)
            terms = model.normalizer.transform(summarize_prefix(prefix[None, :]))[0] * model.weights
            records.append(dict(pair_id=pair_id, posture=posture, status="completed",
                                observed_through_s=cutoff, p_success=prediction["p_success"],
                                p_failure=prediction["p_failure"],
                                failure_alert=prediction["p_failure"] >= 0.5,
                                success_logit=float(terms.sum() + model.intercept),
                                wrist_contribution=float(terms[active_wrist].sum()),
                                nonwrist_contribution=float(terms[~active_wrist].sum()),
                                intercept=float(model.intercept)))
            inputs.append(prefix)
    # All new inference is complete before the first task outcome is parsed.
    write(output / "predictions_before_outcomes.json", records)
    for row in records:
        episode = runs_root / row["pair_id"] / row["posture"]
        result, metadata = read(episode / "result.json"), read(episode / "collection_metadata.json")
        if result["status"] != "completed" or type(result["success"]) is not bool:
            raise ValueError("Invalid completed task outcome")
        label = int(result["success"])
        if label != metadata["label"]:
            raise ValueError("Binary task label and saved result disagree")
        row.update(label_success=label, failure_type=result["failure_type"],
                   classification=("false_alarm" if label and row["failure_alert"] else
                                   "correct_success" if label else
                                   "detected_failure" if row["failure_alert"] else "missed_failure"),
                   measured_task_outcome=result)
    first_report = verify_first_bundle(first_bundle, declaration["first_report_sha256"])
    if (first_report["pair_id"] != poses["first_pair_id"]
            or first_report["pose_manifest_sha256"] != MANIFEST_SHA256
            or first_report["prediction_time_s"] != 2.96
            or first_report["failure_alert_threshold"] != 0.5
            or first_report["model_unchanged"] is not True):
        raise ValueError("Historical first-pair report contract differs")
    pair_reports = []
    for pair_id in pair_ids:
        pair_output = output / "pairs" / pair_id
        pair_output.mkdir(parents=True)
        shutil.copyfile(runs_root / pair_id / "plan.json", pair_output / "plan.json")
        shutil.copyfile(pose_manifest, pair_output / "pose_manifest.json")
        subset = [row for row in records if row["pair_id"] == pair_id]
        stops = [row for row in excluded if row["pair_id"] == pair_id]
        for row in subset:
            source = output / "prediction_inputs" / pair_id / row["posture"]
            shutil.copytree(source, pair_output / "prediction_inputs" / row["posture"])
        old_style_subset = [{key: value for key, value in row.items() if key != "pair_id"}
                            for row in subset]
        old_style_stops = [{key: value for key, value in row.items() if key != "pair_id"}
                           for row in stops]
        pair_report = dict(schema_version="frozen_wrist_pair_model_diagnostic_v1", date="2026-10-12",
                           pair_id=pair_id, pose_manifest_sha256=MANIFEST_SHA256,
                           planned=2, completed_scored=len(subset), stopped_unscored=len(stops),
                           prediction_time_s=2.96, history_steps=20, failure_alert_threshold=0.5,
                           model_unchanged=True, no_training_or_tuning=True,
                           offline_causal_prefix_scoring=True, records=old_style_subset,
                           excluded=old_style_stops, paired_difference=difference(subset),
                           metrics=metrics(subset), frozen_file_sha256=frozen,
                           raw_source_sha256=inventory(runs_root / pair_id),
                           scope="One predeclared remaining development pair; final grasp/lift/hold outcomes, not slip timing or recovery.",
                           comparison_qualification="Both predeclared wrist settings lie outside original training-command ranges; this tests sensitivity, not familiar-versus-unfamiliar transfer.")
        write(pair_output / "report.json", pair_report)
        write(pair_output / "predictions_before_outcomes.json", [
            {key: value for key, value in row.items()
             if key not in ("label_success", "failure_type", "classification", "measured_task_outcome", "pair_id")}
            for row in subset])
        pair_reports.append(pair_report)
    combined = [first_report, *pair_reports]
    report = dict(schema_version="frozen_remaining_wrist_pairs_model_diagnostic_v1", date="2026-10-12",
                  first_pair_id=poses["first_pair_id"], pair_ids=pair_ids, planned=14,
                  completed_scored=len(records), stopped_unscored=len(excluded),
                  pose_manifest_sha256=MANIFEST_SHA256, prediction_time_s=2.96,
                  history_steps=20, failure_alert_threshold=0.5, model_unchanged=True,
                  no_training_or_tuning=True, all_new_predictions_saved_before_outcome_reads=True,
                  offline_causal_prefix_scoring=True, records=records, excluded=excluded,
                  paired_reports=pair_reports,
                  remaining_seven=summarize_pairs(pair_reports),
                  combined_eight=summarize_pairs(combined),
                  first_pair_report_sha256=sha(first_bundle / "report.json"),
                  first_bundle_files_sha256=first_before,
                  scoring_declaration_sha256=sha(declaration_path),
                  scoring_script_sha256=sha(Path(__file__).resolve()),
                  analysis_files_sha256=analysis_frozen,
                  frozen_file_sha256=frozen, raw_source_sha256=raw_before,
                  qualification="A is the nearest-training eligible candidate among the evaluated candidates; both poses remain outside training. No model improvement, physical slip forecast, recovery or broad transfer claim is tested.")
    write(output / "report.json", report)
    write(output / "excluded.json", excluded)
    if records:
        np.savez_compressed(output / "dataset.npz", X=np.stack(inputs),
                            y=np.array([row["label_success"] for row in records]),
                            pair_ids=np.array([row["pair_id"] for row in records]),
                            postures=np.array([row["posture"] for row in records]),
                            joint_names=np.array(contract["joint_names"]))
        with (output / "predictions.csv").open("w", newline="") as stream:
            columns = [key for key in records[0] if key != "measured_task_outcome"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: value for key, value in row.items() if key in columns}
                             for row in records)
    verify_files(repository, frozen)
    verify_files(repository, analysis_frozen)
    if (inventory(runs_root) != raw_before or inventory(first_bundle) != first_before
            or sha(Path(__file__).resolve()) != digest(declaration["script_sha256"])):
        raise ValueError("Original recordings, first bundle or declared scoring code changed")
    write(output / "manifest.json", dict(schema_version="wrist_pair_remaining_manifest_v1",
          files_sha256=inventory(output), frozen_file_sha256=frozen,
          raw_source_sha256=raw_before, first_bundle_files_sha256=first_before,
          scoring_declaration_sha256=sha(declaration_path), unchanged_model_sha256=MODEL_SHA256))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pose-manifest", type=Path, required=True)
    parser.add_argument("--first-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scoring-declaration", type=Path,
                        help="Final analysis-code declaration; defaults to runs-root/scoring_declaration.json")
    args = parser.parse_args()
    result = score(args.runs_root, args.pose_manifest, args.first_bundle,
                   args.output, args.scoring_declaration)
    print(json.dumps({key: result[key] for key in
                      ("completed_scored", "stopped_unscored", "remaining_seven", "combined_eight")}, indent=2))


if __name__ == "__main__":
    main()
