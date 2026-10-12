"""Independently audit the seven remaining saved zero-noise wrist pairs.

This script never collects a new attempt, changes a model, or retunes a
threshold. Predictions are reconstructed by manual arithmetic from the causal
input whitelist. Full-state geometry and 2 ms command replay are retrospective
checks, kept separate from those inputs. The earlier first-pair report is
verified by its original hash and is not re-executed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_wrist_pair import (
    ACTION_INPUT_KEYS, ARM_NAMES, MODEL_SHA256, TRAJECTORY_INPUT_KEYS,
    assert_close, bound_path, digest, json_read, morphology, prefix_arrays,
    sha, sigmoid, stopped_geometry, summarize, tracking_comparison,
)
from audit_mug_table_clearance import audit as replay_audit, compare


FIRST_REPORT = "research_artifacts/wrist_pair_diagnostic_v1/report.json"
FIRST_REPORT_SHA256 = "80c1c82d57bf75356e931724b76417091e22376968060eacb04c7e05b7bfafa5"
POSE_MANIFEST_SHA256 = "354f245ba6663974c92a132895709ea8d41dd5578942dc7af8e526f75dbb8ee1"


def count_outcomes(records: list[dict]) -> dict:
    confusion = dict(true_positive=0, false_positive=0,
                     true_negative=0, false_negative=0)
    for row in records:
        key = ("false_positive" if row["label_success"] and row["failure_alert"] else
               "true_negative" if row["label_success"] else
               "true_positive" if row["failure_alert"] else "false_negative")
        confusion[key] += 1
    success = sum(row["label_success"] for row in records)
    failure = len(records) - success
    return dict(completed=len(records), successes=success, failures=failure,
                failure_confusion=confusion,
                success_false_alarm_rate=(confusion["false_positive"] / success
                                          if success else None),
                failure_recall=(confusion["true_positive"] / failure
                                if failure else None))


def verify_metric_counts(records: list[dict], metrics: dict | None) -> dict:
    """Reconstruct confusion and descriptive metrics without model helpers."""
    values = count_outcomes(records)
    if not records:
        if metrics is not None:
            raise ValueError("empty completed inventory received model metrics")
        return values
    for key in ("successes", "failures", "failure_confusion",
                "success_false_alarm_rate", "failure_recall"):
        if metrics[key] != values[key]:
            raise ValueError(f"independent outcome counts differ: {key}")
    if metrics["episodes"] != len(records):
        raise ValueError("reported metric denominator differs")
    y = np.asarray([r["label_success"] for r in records], dtype=float)
    p = np.asarray([r["p_success"] for r in records], dtype=float)
    c = values["failure_confusion"]
    bounded = np.clip(p, 1e-12, 1 - 1e-12)
    actual = dict(accuracy=(c["true_positive"] + c["true_negative"]) / len(records),
                  log_loss=float(-np.mean(y * np.log(bounded) + (1 - y) * np.log1p(-bounded))),
                  brier_score=float(np.mean((p - y) ** 2)),
                  failure_precision=(c["true_positive"] / (c["true_positive"] + c["false_positive"])
                                     if c["true_positive"] + c["false_positive"] else 0.0))
    for key, value in actual.items():
        assert_close(metrics[key], value, "independent " + key)
    # A direct pairwise concordance computation handles probability ties and
    # avoids importing the model's ROC implementation.
    failed, successful = (1 - p)[y == 0], (1 - p)[y == 1]
    auc = (float(np.mean((failed[:, None] > successful[None, :])
                         + 0.5 * (failed[:, None] == successful[None, :])))
           if len(failed) and len(successful) else None)
    if auc is None:
        if metrics["failure_roc_auc"] is not None:
            raise ValueError("single-class inventory received an ROC AUC")
    else:
        assert_close(metrics["failure_roc_auc"], auc, "independent pairwise ROC AUC")
    return dict(**values, **actual, failure_roc_auc=auc)


def aligned_geometry(first: dict, second: dict) -> tuple[dict, dict]:
    """Keep an interrupted arm's shared saved prefix explicitly partial."""
    count = min(len(first["time_s"]), len(second["time_s"]))
    if not count:
        raise ValueError("partial tracking needs at least one common saved endpoint")
    result = []
    for geometry in (first, second):
        result.append({key: value[:count] if isinstance(value, np.ndarray) else value
                       for key, value in geometry.items()})
    return result[0], result[1]


def verify_pair_summary(pairs: list[dict], reported: dict) -> dict:
    matrix = dict(both_success=0, A_success_B_failure=0,
                  A_failure_B_success=0, both_failure=0, incomplete=0)
    warning_matrix = dict(both_warn=0, neither_warn=0,
                          A_warn_B_clear=0, A_clear_B_warn=0)
    completed = []
    for pair in pairs:
        records = [r for r in pair["independent_records"] if r["status"] == "completed"]
        completed.extend(records)
        if len(records) != 2:
            matrix["incomplete"] += 1
            continue
        lookup = {row["posture"]: row for row in records}
        a, b = [lookup[arm] for arm in ARM_NAMES]
        key = ("both_success" if a["label_success"] and b["label_success"] else
               "A_success_B_failure" if a["label_success"] else
               "A_failure_B_success" if b["label_success"] else "both_failure")
        matrix[key] += 1
        if key == "both_success":
            warning_key = ("both_warn" if a["failure_alert"] and b["failure_alert"] else
                           "A_warn_B_clear" if a["failure_alert"] else
                           "A_clear_B_warn" if b["failure_alert"] else "neither_warn")
            warning_matrix[warning_key] += 1
    if (reported["paired_outcome_matrix"] != matrix
            or reported["both_success_warning_matrix"] != warning_matrix
            or reported["planned_pairs"] != len(pairs)
            or reported["planned_attempts"] != 2 * len(pairs)
            or reported["completed_scored"] != len(completed)
            or reported["stopped_unscored"] != 2 * len(pairs) - len(completed)):
        raise ValueError("independent aggregate paired outcome/warning inventory differs")
    values = verify_metric_counts(completed, reported["overall_metrics"])
    per_arm = {arm: verify_metric_counts([row for row in completed if row["posture"] == arm],
                                        reported["metrics_by_posture"][arm])
               for arm in ARM_NAMES}
    return dict(**values, paired_outcome_matrix=matrix,
                both_success_warning_matrix=warning_matrix, metrics_by_posture=per_arm)


def audit(runs_root: Path, pose_manifest: Path, scores: Path, output: Path,
          scoring_declaration: Path | None = None) -> dict:
    repository = Path(__file__).resolve().parents[1]
    runs_root, pose_manifest, scores, output = [p.resolve() for p in
                                              (runs_root, pose_manifest, scores, output)]
    if output.exists():
        raise FileExistsError("independent audit output must be a new directory")
    runtime_assets = next((p for p in runs_root.parents if (p / "runs").is_dir()), None)
    if runtime_assets is None:
        raise ValueError("cannot identify the runtime-assets root")
    sources = {}

    def verify(path: Path, expected: str | None = None):
        actual = sha(path)
        if expected is not None and actual != digest(expected):
            raise ValueError(f"bound source changed: {path}")
        if path in sources and sources[path] != actual:
            raise ValueError(f"source changed while binding inventory: {path}")
        sources[path] = actual
        return actual

    def verify_repository_bindings(value: dict):
        for field in ("frozen_files_sha256", "paired_source_sha256",
                      "frozen_file_sha256", "analysis_files_sha256"):
            for name, expected in value.get(field, {}).items():
                verify(bound_path(repository, name), expected)

    plan = json_read(runs_root / "plan.json")
    manifest = json_read(pose_manifest)
    report = json_read(scores / "report.json")
    manifest_sha = verify(pose_manifest, POSE_MANIFEST_SHA256)
    if (manifest["schema_version"] != "explicit_wrist_pair_pose_manifest_v1"
            or report["schema_version"] != "frozen_remaining_wrist_pairs_model_diagnostic_v1"):
        raise ValueError("unsupported manifest or remaining-pairs score schema")
    pair_ids = [row["pair_id"] for row in manifest["cases"]
                if row["eligible"] and row["pair_id"] != manifest["first_pair_id"]]
    if (len(pair_ids) != 7 or plan["pair_ids"] != pair_ids
            or report["pair_ids"] != pair_ids or plan["postures"] != list(ARM_NAMES)
            or plan["noise_amplitude_rad"] != 0.0
            or digest(plan["pose_manifest_sha256"]) != manifest_sha
            or digest(report["pose_manifest_sha256"]) != manifest_sha
            or report["prediction_time_s"] != 2.96
            or report["history_steps"] != 20 or report["failure_alert_threshold"] != 0.5):
        raise ValueError("declared remaining-pair inventory or model contract differs")
    verify(scores / "pose_manifest.json", manifest_sha)
    verify(scores / "plan.json", verify(runs_root / "plan.json"))
    verify(Path(__file__).resolve())
    verify_repository_bindings(plan)
    verify_repository_bindings(report)
    for name, expected in manifest["source_sha256"].items():
        kind, relative = name.split(":", 1)
        root = {"repository": repository, "runtime_assets": runtime_assets}[kind]
        verify(bound_path(root, relative), expected)
    for name, expected in report["raw_source_sha256"].items():
        verify(bound_path(runs_root, name), expected)
    first_path = repository / FIRST_REPORT
    first_sha = verify(first_path, FIRST_REPORT_SHA256)
    first_report = json_read(first_path)
    if first_report["pair_id"] != manifest["first_pair_id"]:
        raise ValueError("preserved first-pair identity differs")
    if digest(plan["first_report_sha256"]) != first_sha:
        raise ValueError("execution declaration changed the preserved first report")
    first_bundle = first_path.parent
    verify(first_bundle / "manifest.json", plan["first_bundle_manifest_sha256"])
    for name, expected in report["first_bundle_files_sha256"].items():
        verify(bound_path(first_bundle, name), expected)
    # These hashes are separately frozen after collection and before scoring,
    # rather than falsely treating unfinished analysis code as pre-execution.
    declaration_path = (scoring_declaration.resolve() if scoring_declaration is not None
                        else runs_root / "scoring_declaration.json")
    if not declaration_path.is_file():
        raise ValueError("missing before-analysis code declaration")
    declaration = json_read(declaration_path)
    verify(declaration_path)
    if (digest(declaration["runs_plan_sha256"]) != sha(runs_root / "plan.json")
            or digest(declaration["pose_manifest_sha256"]) != manifest_sha
            or digest(declaration["first_report_sha256"]) != first_sha):
        raise ValueError("analysis declaration differs from saved experiment sources")
    verify_repository_bindings(declaration)
    verify(scores / "scoring_declaration.json", sha(declaration_path))
    analysis_bindings = declaration.get("analysis_files_sha256", {})
    own_name = str(Path(__file__).resolve().relative_to(repository))
    if own_name not in analysis_bindings:
        raise ValueError("independent auditor code was not bound before analysis")
    contract = manifest["model_contract"]
    if (contract["history_steps"] != 20 or contract["prediction_time_s"] != 2.96
            or contract["failure_alert_threshold"] != 0.5
            or contract["fields"] != ["hand_qpos", "hand_qvel", "commanded_target_rad"]
            or contract["weights_normalization_features_unchanged"] is not True
            or contract["outcomes_not_inputs"] is not True):
        raise ValueError("frozen model input declaration differs")
    model_path = bound_path(repository, contract["model_file"])
    verify(model_path, MODEL_SHA256)
    if digest(contract["model_sha256"]) != MODEL_SHA256:
        raise ValueError("original fitted model identity differs")
    with np.load(model_path, allow_pickle=False) as model:
        mean, scale = model["mean"].copy(), model["scale"].copy()
        active, weights = model["active"].copy(), model["weights"].copy()
        intercept = float(model["intercept"])
    wrist_mask = np.arange(288) % 24 < 2
    if (mean.shape != (288,) or scale.shape != (288,) or active.shape != (288,)
            or active.dtype != np.bool_ or active.sum() != 269 or weights.shape != (269,)
            or np.sum(active & wrist_mask) != 20 or np.any(scale <= 0)):
        raise ValueError("original model summary dimensions differ")
    pilot = repository / "research_artifacts/object_scene_pilot_v1"
    split_path, dataset_path = pilot / "splits.json", pilot / "dataset.npz"
    train_ids = json_read(split_path)["partitions"]["train"]["episode_ids"]
    with np.load(dataset_path, allow_pickle=False) as dataset:
        lookup = {str(value): i for i, value in enumerate(dataset["episode_ids"])}
        train_features = summarize(dataset["X"][[lookup[value] for value in train_ids]])
    if len(train_features) != 520 or len(set(train_ids)) != 520:
        raise ValueError("training-only normalization inventory differs")
    manual_mean, manual_std = train_features.mean(axis=0), train_features.std(axis=0)
    manual_active = manual_std > 1e-8
    if (not np.array_equal(manual_mean, mean) or not np.array_equal(manual_active, active)
            or not np.array_equal(np.where(manual_active, manual_std, 1.0), scale)):
        raise ValueError("independent original training normalization differs")
    verify(split_path)
    verify(dataset_path)
    training_min, training_max = train_features.min(axis=0), train_features.max(axis=0)
    for path in sorted(runs_root.rglob("*")) + sorted(scores.rglob("*")):
        if path.is_file():
            verify(path)
    top_rows = {(r["pair_id"], r["posture"]): r for r in report["records"]}
    top_excluded = {(r["pair_id"], r["posture"]): r for r in report["excluded"]}
    if (len(top_rows) != len(report["records"]) or len(top_excluded) != len(report["excluded"])
            or set(top_rows) & set(top_excluded)
            or set(top_rows) | set(top_excluded) != {(p, arm) for p in pair_ids for arm in ARM_NAMES}):
        raise ValueError("every remaining treatment must occur exactly once, including stops")
    independent_rows, pair_results, tracking_arrays = [], [], {}
    maximum_probability_error = 0.0
    for pair_id in pair_ids:
        pair = next(row for row in manifest["cases"] if row["pair_id"] == pair_id)
        pair_root = runs_root / pair_id
        pair_scores = scores / "pairs" / pair_id
        pair_report = json_read(pair_scores / "report.json")
        pair_plan = json_read(pair_root / "plan.json")
        if (pair_report["schema_version"] != "frozen_wrist_pair_model_diagnostic_v1"
                or pair_report["pair_id"] != pair_id or pair_plan["pair_id"] != pair_id
                or digest(pair_report["pose_manifest_sha256"]) != manifest_sha
                or digest(pair_plan["pose_manifest_sha256"]) != manifest_sha
                or pair_report["prediction_time_s"] != 2.96
                or pair_report["failure_alert_threshold"] != 0.5):
            raise ValueError("pair-local identity or cutoff differs")
        verify(pair_scores / "plan.json", sha(pair_root / "plan.json"))
        verify(pair_scores / "pose_manifest.json", manifest_sha)
        verify_repository_bindings(pair_plan)
        verify_repository_bindings(pair_report)
        for name, expected in pair_report["raw_source_sha256"].items():
            verify(bound_path(pair_root, name), expected)
        scored = {r["posture"]: r for r in pair_report["records"]}
        excluded = {r["posture"]: r for r in pair_report["excluded"]}
        rows, geometry, commands, dynamic, stopped, command_deviations = [], {}, {}, [], [], {}
        for arm in ARM_NAMES:
            episode = pair_root / arm
            metadata = json_read(episode / "collection_metadata.json")
            if (metadata["pair_id"] != pair_id or metadata["posture"] != arm
                    or digest(metadata["frozen_pose_manifest_sha256"]) != manifest_sha
                    or metadata["noise_amplitude_rad"] != 0.0
                    or metadata["noise_mode"] != "disabled"
                    or metadata["predictor_evaluated_by_executor"] is not False
                    or digest(metadata["original_case_sha256"]) != digest(pair["case_sha256"])):
                raise ValueError("execution identity, model isolation or zero-noise condition differs")
            for key, expected in pair["source_binding"].items():
                if metadata.get(key) != expected:
                    raise ValueError(f"recorded proposal/observation/object source differs: {key}")
            assert_close(metadata["intended_world_palm_position_m"],
                         pair["intended_world_palm_position_m"], "palm intention")
            assert_close(metadata["intended_world_palm_quaternion_wxyz"],
                         pair["intended_world_palm_quaternion_wxyz"], "palm orientation intention")
            wrist = pair["poses"][arm]["wrist_rad"]
            assert_close(metadata["execution_protocol"]["explicit_wrist_target_rad"],
                         wrist, "declared treatment", 0.0)
            certificate = metadata.get("frozen_pose_certificate")
            if certificate is not None and (certificate["explicit_angles_unchanged"] is not True
                                           or certificate["automatic_substitution"] is not False):
                raise ValueError("executor substituted the declared treatment")
            actions_path = episode / "actions.npz"
            if actions_path.exists():
                with np.load(actions_path, allow_pickle=False) as actions:
                    if any(np.any(actions[key] != 0.0) for key in
                           ("episode_bias_rad", "requested_noise_rad", "applied_noise_rad")):
                        raise ValueError("nonzero applied disturbance in a zero-noise pair")
                    assert_close(actions["commanded_target_rad"][:, :2],
                                 np.tile(wrist, (len(actions["time_s"]), 1)),
                                 "wrist commands at declared pose tolerance",
                                 manifest["gates"]["pose_reproduction_absolute_tolerance"])
                    commands[arm] = actions["commanded_target_rad"].copy()
                    command_deviations[arm] = (np.max(np.abs(commands[arm][:, :2] - wrist), axis=0).tolist()
                                               if len(commands[arm]) else None)
                if (episode / "scene.mjb").exists() and (episode / "trajectory.npz").exists():
                    geometry[arm] = morphology(episode)
            if metadata["status"] != "completed":
                row = excluded.get(arm)
                if (metadata["label"] is not None or row is None or row["label_success"] is not None
                        or row["p_failure"] is not None or (episode / "result.json").exists()
                        or (pair_scores / "prediction_inputs" / arm).exists()
                        or (scores / "prediction_inputs" / pair_id / arm).exists()
                        or (pair_id, arm) not in top_excluded):
                    raise ValueError("stopped execution received an outcome label or model score")
                rows.append(dict(pair_id=pair_id, posture=arm, status=metadata["status"],
                                 label_success=None, p_failure=None, failure_alert=None,
                                 scope="Stopped arm excluded from model metrics; saved partial tracking is retrospective only"))
                stopped.append(dict(posture=arm,
                                    saved_guard_endpoint=stopped_geometry(episode, metadata)))
                continue
            if arm not in scored or not certificate or (pair_id, arm) not in top_rows:
                raise ValueError("completed arm lacks a score or geometry certificate")
            prefix, trajectory_input, action_input = prefix_arrays(episode)
            if action_input["joint_names"].tolist() != contract["joint_names"]:
                raise ValueError("prediction joint order differs")
            for portable in (pair_scores / "prediction_inputs" / arm,
                             scores / "prediction_inputs" / pair_id / arm):
                for filename, expected, whitelist in (
                    ("trajectory.npz", trajectory_input, TRAJECTORY_INPUT_KEYS),
                    ("actions.npz", action_input, ACTION_INPUT_KEYS),
                ):
                    with np.load(portable / filename, allow_pickle=False) as values:
                        if set(values.files) != whitelist:
                            raise ValueError("portable prefix includes a privileged field")
                        if any(not np.array_equal(values[key], array) for key, array in expected.items()):
                            raise ValueError("portable model inputs differ from raw pre-lift samples")
            features = summarize(prefix[None])[0]
            standardized = (features - mean) / scale
            contributions = np.zeros(288)
            contributions[active] = standardized[active] * weights
            wrist_term = float(contributions[wrist_mask].sum())
            other_term = float(contributions[~wrist_mask].sum())
            logit = float(np.dot(standardized[active], weights) + intercept)
            p_success = sigmoid(logit)
            calculated = dict(success_logit=logit, wrist_contribution=wrist_term,
                              nonwrist_contribution=other_term, intercept=intercept,
                              p_success=p_success, p_failure=1.0 - p_success)
            assert_close(logit, wrist_term + other_term + intercept, "independent attribution sum")
            for key, value in calculated.items():
                for reported in (scored[arm], top_rows[pair_id, arm]):
                    assert_close(value, reported[key], f"{pair_id}/{arm} reconstructed {key}")
            maximum_probability_error = max(maximum_probability_error,
                                            abs(p_success - scored[arm]["p_success"]))
            result = json_read(episode / "result.json")
            label = int(result["success"])
            if (type(result["success"]) is not bool or result["status"] != "completed"
                    or metadata["label"] != label or scored[arm]["label_success"] != label
                    or top_rows[pair_id, arm]["label_success"] != label
                    or scored[arm]["measured_task_outcome"] != result):
                raise ValueError("completed task label or stored outcome differs")
            if label and (result["outcome"]["maximum_lift_m"] < 0.14
                          or result["outcome"]["hold_duration_s"] < 2.0):
                raise ValueError("success does not satisfy frozen lift/hold minimum")
            alert = calculated["p_failure"] >= 0.5
            if scored[arm]["failure_alert"] != alert or top_rows[pair_id, arm]["failure_alert"] != alert:
                raise ValueError("threshold alert differs")
            rows.append(dict(pair_id=pair_id, posture=arm, status="completed", label_success=label,
                             failure_alert=alert, **calculated,
                             active_wrist_features_outside_training_range=int(np.sum(
                                 active & wrist_mask & ((features < training_min) | (features > training_max)))),
                             prefix_start_s=2.20, observed_through_s=2.96))
            replay = replay_audit(episode)
            if (not replay["all_qpos_qvel_bitwise_identical"]
                    or not replay["all_integration_states_bitwise_identical"]
                    or replay["first_any_hand_table_contact"] is not None):
                raise ValueError("saved-command replay differs or touches the table")
            reference = (runtime_assets / "runs/mug_table_clearance_validation_2026-10-11"
                         / "final_checks_27" / pair_id)
            if arm == "B" and reference.is_dir():
                replay["v3_reference_comparison"] = compare(reference, episode)
                replay["v3_reference_scope"] = "Array comparison against the prior zero-noise v3 reference; protocol identity remains v4"
            dynamic.append(dict(posture=arm, **replay))
        finger_commands_equal = None
        if set(commands) == set(ARM_NAMES):
            count = min(len(commands["A"]), len(commands["B"]))
            finger_commands_equal = np.array_equal(commands["A"][:count, 2:], commands["B"][:count, 2:])
            if not finger_commands_equal:
                raise ValueError("22 applied finger commands differ between treatments")
        completed = [r for r in rows if r["status"] == "completed"]
        tracking, tracking_scope = None, None
        if set(geometry) == set(ARM_NAMES):
            first, second = aligned_geometry(geometry["A"], geometry["B"])
            tracking, arrays = tracking_comparison(first, second)
            tracking_scope = ("completed_pair" if len(completed) == 2
                              else "partial_shared_saved_prefix; interrupted arm remains unlabeled and unscored")
            tracking["comparison_scope"] = tracking_scope
            tracking_arrays[pair_id] = arrays
        differences = None
        if len(completed) == 2:
            a, b = completed
            differences = dict(success_logit_B_minus_A=b["success_logit"] - a["success_logit"],
                               wrist_contribution_B_minus_A=b["wrist_contribution"] - a["wrist_contribution"],
                               nonwrist_contribution_B_minus_A=b["nonwrist_contribution"] - a["nonwrist_contribution"],
                               p_failure_B_minus_A=b["p_failure"] - a["p_failure"],
                               both_successful=bool(a["label_success"] and b["label_success"]),
                               alerts_differ=a["failure_alert"] != b["failure_alert"])
            for key, value in differences.items():
                if isinstance(value, bool):
                    if pair_report["paired_difference"][key] != value:
                        raise ValueError("reported paired boolean differs")
                else:
                    assert_close(pair_report["paired_difference"][key], value,
                                 "independent paired difference " + key)
        elif pair_report["paired_difference"] is not None:
            raise ValueError("incomplete pair received a completed-pair comparison")
        metrics = verify_metric_counts(completed, pair_report["metrics"])
        pair_results.append(dict(pair_id=pair_id, independent_records=rows,
                                 completed=len(completed), stopped_unscored=len(excluded),
                                 independently_reconstructed_metrics=metrics, paired_difference=differences,
                                 identical_22_applied_finger_commands_verified=finger_commands_equal,
                                 maximum_wrist_command_deviation_from_manifest_rad=command_deviations,
                                 actual_motion_tracking=tracking, independent_dynamic_replays=dynamic,
                                 stopped_guard_endpoint_checks=stopped))
        independent_rows.extend(rows)
        print(json.dumps(dict(pair_id=pair_id, completed=len(completed),
                              stopped_unscored=len(excluded),
                              false_alarms=metrics["failure_confusion"]["false_positive"])), flush=True)
    completed = [r for r in independent_rows if r["status"] == "completed"]
    remaining_counts = verify_pair_summary(pair_results, report["remaining_seven"])
    first_summary = dict(pair_id=first_report["pair_id"],
                         independent_records=[*first_report["records"], *first_report["excluded"]])
    combined_counts = verify_pair_summary([first_summary, *pair_results], report["combined_eight"])
    for path, before in sources.items():
        if sha(path) != before:
            raise ValueError(f"original source, recording or score changed during independent audit: {path}")
    output.mkdir(parents=True)
    for pair_id, arrays in tracking_arrays.items():
        destination = output / "pairs" / pair_id
        destination.mkdir(parents=True)
        np.savez_compressed(destination / "morphology_tracking.npz", **arrays)
    result = dict(schema_version="independent_remaining_wrist_pairs_audit_v1",
                  pair_ids=pair_ids, planned_pairs=7, planned_attempts=14,
                  pose_manifest_sha256=manifest_sha, independent_records=independent_rows,
                  pairs=pair_results, remaining_seven=remaining_counts, combined_eight=combined_counts,
                  stopped_unscored=len(top_excluded), preserved_first_pair_report_sha256=first_sha,
                  first_pair_not_reexecuted_by_this_auditor=True,
                  training_rows=520, active_features=269, active_wrist_features=20,
                  training_only_normalization_bitwise_reproduced=True,
                  portable_prefix_whitelist_and_raw_equivalence_verified=True,
                  maximum_independent_probability_error=maximum_probability_error,
                  unchanged_model_sha256=MODEL_SHA256,
                  all_available_saved_commands_zero_noise_verified=True,
                  wrist_command_comparison_tolerance_rad=manifest["gates"]["pose_reproduction_absolute_tolerance"],
                  completed_physics_substeps_checked=sum(
                      replay["physics_substeps_checked"] for pair in pair_results
                      for replay in pair["independent_dynamic_replays"]),
                  all_completed_qpos_qvel_and_integration_states_bitwise_identical=True,
                  completed_hand_table_contacts_found=False,
                  original_sources_unchanged=True,
                  analysis_code_declaration=declaration,
                  source_sha256={str(path): value for path, value in sorted(sources.items())},
                  scope="Seven predeclared development pairs, separately from preserved first pair; no new collection, fitting, familiarity claim, metric tuning or population generalization estimate")
    (output / "independent_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pose-manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scoring-declaration", type=Path)
    args = parser.parse_args()
    result = audit(args.runs_root, args.pose_manifest, args.scores, args.output,
                   args.scoring_declaration)
    print(json.dumps({key: result[key] for key in
                     ("remaining_seven", "combined_eight", "stopped_unscored",
                      "completed_physics_substeps_checked", "maximum_independent_probability_error")}, indent=2))


if __name__ == "__main__":
    main()
