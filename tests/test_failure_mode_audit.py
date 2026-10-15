"""Manufactured cohorts/probabilities only; no optimizer, dynamics or real fits."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation.prediction_recording import (
    AttemptMetadata, PredictionRecorder, write_unexecuted_attempt,
)
from grasp_failure_prediction.evaluation.scenario_sampler import freeze_category_folds
from grasp_failure_prediction.prediction import failure_mode_audit as audit
from grasp_failure_prediction.prediction import model as models
from grasp_failure_prediction.prediction import recorded_training as training
from grasp_failure_prediction.prediction import retrospective_holdout as retrospective
from grasp_failure_prediction.prediction.recorded_dataset import PrefixSpec, export_recorded_dataset


@pytest.fixture(autouse=True)
def no_real_integrations_or_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no real fitting or dynamics in manufactured audit tests")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(audit, "fit_logistic", forbidden)
    monkeypatch.setattr(models, "fit_logistic", forbidden)
    monkeypatch.setattr(training, "fit_logistic", forbidden)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def cohort(tmp_path, request):
    options = getattr(request, "param", {})
    rows = []
    for category in range(3):
        for family in range(4):
            for variant in range(2):
                number = len(rows)
                rows.append(dict(attempt_id=f"attempt_{number:03d}", scenario_id=f"scenario_{number:03d}",
                    object_id=f"object_{category}", category_id=f"cat_{category}",
                    geometry_group_id=f"geometry_{category}", observation_group_id=f"capture_{category}",
                    proposal_group_id=f"family_{category}_{0 if options.get('single_family_category') == category else family}", generation_seed=family,
                    mass_multiplier=.75 if variant else 1.25, friction_coefficient=.5,
                    table_offset_m=0., approach_mode="grasp_derived", sensor_noise_std_rad=0., sensor_noise_seed=number))
    for source, family in ((4, "family_0_2"), (22, "family_2_3"), (0, "family_only_null")):
        number = len(rows)
        rows.append(dict(rows[source], attempt_id=f"attempt_{number:03d}",
                         scenario_id=f"scenario_{number:03d}", proposal_group_id=family))
    schedule = dict(schema_version="hug_scenario_schedule_v1", outcome_selection=False, scenarios=rows)
    paths = []
    hash_value = "sha256:" + "a" * 64
    for index, scenario in enumerate(rows):
        success = bool(index % 2)
        failure = None if success else ("drop" if scenario["generation_seed"] == 0 else "failed_acquisition")
        if options.get("single_class_category") == int(scenario["category_id"][-1]):
            success, failure = False, "failed_acquisition"
        if index == 24:
            success, failure = False, "approach_collision"
        metadata = dict(attempt_id=scenario["attempt_id"], hand_id="manufactured_shadow",
            execution_protocol_id="manufactured_approach", object_id=scenario["object_id"],
            geometry_group_id=scenario["geometry_group_id"], observation_group_id=scenario["observation_group_id"],
            proposal_group_id=scenario["proposal_group_id"], mapping=dict(status="ok"), execution_status="completed",
            outcome=dict(success=success, failure_type=failure, label_rule_id="manufactured_rules",
                first_failure_time_s=.1 if index == 24 else None),
            provenance_hashes={**{name:hash_value for name in ("code", "configuration", "inputs", "model", "assets",
                "native_pose", "mapped_pose", "label_rule")}, "scenario":audit.digest(scenario)})
        path = tmp_path / "attempts" / scenario["attempt_id"]
        if index >= 25:
            metadata.update(mapping=dict(status="refused_guard", code="manufactured", message="static test refusal"),
                            execution_status="not_executed", outcome=None)
            write_unexecuted_attempt(path, AttemptMetadata(**metadata))
        else:
            recorder = PredictionRecorder(["j0", "j1"], [[-10., 10.], [-10., 10.]])
            for step in range(1, 5):
                qpos = np.full(2, index * .01 + step * .1)
                recorder.append(time_s=step * .04, action_interval_start_s=(step - 1) * .04,
                    phase="close_fingers", hand_qpos=qpos, hand_qvel=qpos * .2,
                    requested_target_rad=np.full(2, .4), applied_target_rad=np.full(2, .4),
                    diagnostics=dict(privileged_object_state=np.full(3, 999.)))
            recorder.write(path, AttemptMetadata(**metadata))
        paths.append(path)
    dataset = tmp_path / "dataset"
    export_recorded_dataset(paths, dataset, prefix_spec=PrefixSpec(protocol_id="manufactured_prefix",
        prediction_time_s=.16, history_steps=4, control_timestep_s=.04))
    folds = freeze_category_folds(schedule)
    binding = dict(schema_version="hug_attempt_scenario_binding_v1", schedule_sha256=audit.digest(schedule),
        attempt_to_scenario={row["attempt_id"]:row["scenario_id"] for row in rows})
    files = [tmp_path / (name + ".json") for name in ("schedule", "folds", "binding")]
    for path, value in zip(files, (schedule, folds, binding)):
        write(path, value)
    data = training.load_training_data(dataset, *files)
    report = dict(schema_version=training.SCHEMA, status="completed_with_skips", source_bindings=data.source_bindings,
        specification=dict(warning_threshold=.5), input_contract=data.dataset.manifest["input_contract"], folds=[])
    report_path = tmp_path / "frozen_models" / "report.json"
    for fold in data.folds:
        row = dict(fold_id=fold["fold_id"], test_category=fold["test_category"], validation_category=fold["validation_category"],
            partitions={role:dict(episode_ids=data.dataset.episode_ids[indices].tolist(), n=len(indices),
                row_indices_sha256=audit.digest(indices.tolist())) for role,indices in fold["indices"].items()}, models=[])
        for name in training.MODEL_NAMES:
            for seed in (0, 1, 2):
                item = dict(name=name, seed=seed, status="skipped", reason="manufactured unfit model")
                if name == "summary_logistic" and seed == 0:
                    indices = fold["indices"]["test"]
                    directory = Path(fold["fold_id"]) / name / "seed_0"
                    prediction_path = report_path.parent / directory / "predictions.json"
                    write(prediction_path, dict(episode_ids=data.dataset.episode_ids[indices].tolist(),
                        y_success=data.dataset.y[indices].tolist(), p_success=np.full(len(indices), .5).tolist(), warning_threshold=.5))
                    item.update(status="completed", model_directory=str(directory), predictions_sha256=audit.sha(prediction_path))
                row["models"].append(item)
        report["folds"].append(row)
    write(report_path, report)
    return data, report_path, tmp_path / "protocol.json"


def prepared(cohort):
    data, report, path = cohort
    return audit.prepare_audit(data, report, path)


def test_holdout_keeps_all_family_variants_and_success_controls(cohort):
    data = cohort[0]
    result = audit.failure_family_folds(data)
    assert result["dataset_denominators"] == data.dataset.manifest["counts"]
    assert result["dataset_denominators"]["attempted"] == 27
    assert result["dataset_denominators"]["included"] == 24
    for fold in result["folds"]:
        partitions = fold["partitions"]
        assert sum(p["planned"] for p in partitions.values()) == 27
        assert sum(p["included"] for p in partitions.values()) == 24
        roles = [set(p["families"]) for p in partitions.values()]
        assert all(not roles[i] & roles[j] for i in range(3) for j in range(i))
        assert set().union(*roles) == {r["proposal_group_id"] for r in data.schedule["scenarios"]}
        if fold["held_failure"] in ("drop", "failed_acquisition"):
            assert fold["status"] == "READY"
            assert partitions["test"]["success"] > 0
            assert partitions["test"]["failure"] > 0
            held = set(fold["held_families"])
            for row in data.dataset.manifest["episodes"]:
                if row["failure_type"] == fold["held_failure"]:
                    assert row["group_id"] in held
                if row["group_id"] in held:
                    assert row["attempt_id"] in partitions["test"]["eligible_episode_ids"]
    early = result["folds"][0]
    assert early["reason"] == "held_mode_has_no_eligible_prefix"
    assert early["observed_cause_attempts"] == 1
    assert early["eligible_cause_attempts"] == 0
    assert early["partitions"]["test"]["excluded_by_reason"] == {"failure_at_or_before_cutoff":1}
    assert "family_only_null" not in early["held_families"]
    assert result["folds"][1]["reason"] == "missing_failure_mode"


def test_validation_hash_rule_is_exact_and_deterministic_without_class_repair(cohort):
    result = audit.failure_family_folds(cohort[0])
    assert result == audit.failure_family_folds(cohort[0])
    for fold in result["folds"]:
        remaining = set(fold["family_roles"]) - set(fold["held_families"])
        ordered = sorted(remaining, key=lambda family:(audit.digest(dict(audit=audit.SCHEMA, seed=20261015,
            held_failure=fold["held_failure"], family=family)), family))
        n = int(np.ceil(len(remaining) * .2))
        assert set(fold["partitions"]["validation"]["families"]) == set(ordered[:n])


def test_single_class_test_is_explicit_skip_even_with_both_fit_roles(cohort):
    data = cohort[0]
    # Deliberate manufactured snapshot update; real callers cannot mutate loaded arrays.
    for index,row in enumerate(data.dataset.manifest["episodes"]):
        if row["group_id"].endswith("_0"):
            row.update(source_task_label=0, failure_type="drop")
            data.dataset.y[index] = 0
    data.source_bindings["loaded_data_sha256"] = training._loaded_data_sha256(data)
    fold = next(f for f in audit.failure_family_folds(data)["folds"] if f["held_failure"] == "drop")
    assert fold["reason"] == "single_class_test"
    assert fold["partitions"]["train"]["success"] and fold["partitions"]["validation"]["success"]


@pytest.mark.parametrize("changes", [dict(model_kind="small_gru"), dict(regularization_grid=(.1,)),
    dict(max_iterations=401), dict(warning_threshold=.4), dict(validation_fraction=.3),
    dict(family_hash_seed=7), dict(bootstrap_repeats=10), dict(bootstrap_seed=8), dict(bootstrap_seed=20261015.)])
def test_exact_versioned_budget(changes):
    with pytest.raises(ValueError, match="fixed"):
        replace(audit.AuditSpec(), **changes)


def test_prepare_is_no_fit_and_immutable_and_requires_commit(cohort):
    value = prepared(cohort)
    assert value["requires_commit_before_fit"] is True
    assert value["specification"]["model_kind"] == "summary_logistic"
    assert value["category_prediction_inventory"]["entries"][0]["families"]
    assert len(value["category_prediction_inventory"]["entries"]) == 45
    assert audit.validate_audit(cohort[0], cohort[2], require_commit=False)[0] == value
    with pytest.raises(FileExistsError):
        prepared(cohort)
    with pytest.raises(subprocess.CalledProcessError):
        audit.validate_audit(cohort[0], cohort[2])


def test_stale_loaded_labels_refused_before_fold_or_fit(cohort):
    data = cohort[0]
    data.dataset.y[0] = 1
    with pytest.raises(ValueError, match="loaded features/labels"):
        audit.failure_family_folds(data)


def test_explicit_scheduled_attempt_ids_must_match_binding(cohort):
    data = cohort[0]
    a, b = data.schedule["scenarios"][:2]
    a["attempt_id"], b["attempt_id"] = b["attempt_id"], a["attempt_id"]
    data.source_bindings["loaded_data_sha256"] = training._loaded_data_sha256(data)
    with pytest.raises(ValueError, match="adapter binding"):
        audit.failure_family_folds(data)


@pytest.mark.parametrize("mutation", ["cohort", "partition", "prediction_order", "missing_seed", "probability", "threshold"])
def test_frozen_prediction_admission_is_exact(cohort, mutation):
    data, report_path, _ = cohort
    report = json.loads(report_path.read_text())
    if mutation == "cohort":
        report["source_bindings"]["loaded_data_sha256"] = "sha256:" + "b" * 64
    elif mutation == "partition":
        report["folds"][0]["partitions"]["test"]["episode_ids"].reverse()
    elif mutation == "missing_seed":
        report["folds"][0]["models"].pop()
    else:
        model = next(r for r in report["folds"][0]["models"] if r["status"] == "completed")
        path = report_path.parent / model["model_directory"] / "predictions.json"
        prediction = json.loads(path.read_text())
        if mutation == "prediction_order":
            prediction["episode_ids"].reverse()
        elif mutation == "probability":
            prediction["p_success"][0] = 1.1
        else:
            prediction["warning_threshold"] = .4
        write(path, prediction)
        model["predictions_sha256"] = audit.sha(path)
    write(report_path, report)
    with pytest.raises(ValueError):
        prepared(cohort)


def test_bound_prediction_mutation_is_rejected_after_declaration(cohort):
    value = prepared(cohort)
    path = next(Path(e["predictions_path"]) for e in value["category_prediction_inventory"]["entries"] if e["status"] == "completed")
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        audit.frozen_category_uncertainty(cohort[0], value)


def test_source_closure_and_data_identity_are_checked_for_direct_uncertainty(cohort):
    value = prepared(cohort)
    value["source_files"].pop(next(iter(value["source_files"])))
    with pytest.raises(ValueError, match="source/cohort"):
        audit.frozen_category_uncertainty(cohort[0], value)


@pytest.mark.parametrize("cohort", [dict(single_family_category=0)], indirect=True)
def test_one_family_category_is_explicit_uncertainty_skip(cohort, monkeypatch):
    value = prepared(cohort)
    monkeypatch.setattr(audit, "group_uncertainty", lambda *a, **k: dict(percentile_95={}))
    result = audit.frozen_category_uncertainty(cohort[0], value)
    row = next(r for r in result["rows"] if r["category"] == "cat_0" and r["model_kind"] == "summary_logistic" and r["seed"] == 0)
    assert row["status"] == "SKIPPED_TOO_FEW_FAMILIES"
    assert row["uncertainty"] is None and row["point_metrics"]["episodes"] == 8


@pytest.mark.parametrize("cohort", [dict(single_class_category=1)], indirect=True)
def test_single_class_category_keeps_undefined_ranking_and_draw_counts(cohort, monkeypatch):
    value = prepared(cohort)
    monkeypatch.setattr(audit, "group_uncertainty", lambda *a, **k: dict(percentile_95={
        "failure_roc_auc":dict(valid_repeats=0, interval=None),
        "success_false_alarm_rate":dict(valid_repeats=0, interval=None),
        "failure_recall":dict(valid_repeats=1000, interval=[1.,1.])}))
    result = audit.frozen_category_uncertainty(cohort[0], value)
    row = next(r for r in result["rows"] if r["category"] == "cat_1" and r["status"] == "COMPLETED")
    assert row["point_metrics"]["failure_roc_auc"] is None
    assert row["point_metrics"]["failure_average_precision"] is None
    assert row["uncertainty"]["percentile_95"]["failure_roc_auc"]["undefined_repeats"] == 1000


def test_bootstrap_uses_frozen_rows_whole_families_exact_budget_without_inference(cohort, monkeypatch):
    value = prepared(cohort)
    seen = []
    def measured(y, probability, families, *, seed, repeats):
        seen.append((y.copy(), probability.copy(), families.copy(), seed, repeats))
        return dict(independent_groups=len(np.unique(families)), seed=seed, repeats=repeats,
            percentile_95=dict(failure_roc_auc=dict(valid_repeats=900, interval=[.2, .8])))
    monkeypatch.setattr(audit, "group_uncertainty", measured)
    result = audit.frozen_category_uncertainty(cohort[0], value)
    assert result["fitting_calls"] == result["prediction_calls"] == 0
    assert result["seed_repetitions_are_independent_datasets"] is False
    assert len(seen) == 3
    for fold, (y, p, groups, seed, repeats) in zip(cohort[0].folds, seen):
        ids = fold["indices"]["test"]
        np.testing.assert_array_equal(y, cohort[0].dataset.y[ids])
        np.testing.assert_array_equal(groups, cohort[0].dataset.group_ids[ids])
        np.testing.assert_array_equal(p, np.full(len(y), .5))
        assert seed == 20261015 and repeats == 1000
    completed = [r for r in result["rows"] if r["status"] == "COMPLETED"]
    assert all(r["uncertainty"]["percentile_95"]["failure_roc_auc"]["undefined_repeats"] == 100 for r in completed)
    assert sum(r["status"] == "SKIPPED_ORIGINAL_MODEL" for r in result["rows"]) == 42


def test_reused_estimator_resamples_entire_family_with_multiplicity(monkeypatch):
    seen = []
    def metric(y, probability):
        seen.append(probability.tolist())
        return dict(failure_roc_auc=.5, success_false_alarm_rate=None, failure_recall=1.)
    monkeypatch.setattr(retrospective, "outcome_metrics", metric)
    result = retrospective.group_uncertainty([0, 1, 0, 1], [.1, .2, .3, .4], ["a", "a", "b", "b"],
                                           seed=20261015, repeats=1000)
    assert len(seen) == 1000
    assert all(row in [[.1,.2,.1,.2], [.1,.2,.3,.4], [.3,.4,.1,.2], [.3,.4,.3,.4]] for row in seen)
    assert any(row == [.1,.2,.1,.2] for row in seen)
    assert result["percentile_95"]["success_false_alarm_rate"]["valid_repeats"] == 0
    assert result["percentile_95"]["success_false_alarm_rate"]["interval"] is None


def test_summary_audit_uses_only_fit_rows_fixed_grid_and_stronger_l2_tie(cohort, tmp_path, monkeypatch):
    value = prepared(cohort)
    monkeypatch.setattr(audit, "_committed", lambda *args: dict(manufactured_commit_check_only=True))
    calls = []
    def fake_fit(X, y, *, regularization, max_iterations):
        calls.append((X.copy(), y.copy(), regularization, max_iterations))
        normalizer = models.Normalizer.fit(X)
        return models.LogisticModel(normalizer, np.zeros(int(normalizer.active.sum())), 0., regularization, 1, True)
    monkeypatch.setattr(audit, "fit_logistic", fake_fit)
    monkeypatch.setattr(audit, "frozen_category_uncertainty", lambda *a, **k: dict(manufactured_skip_bootstrap=True))
    result = audit.run_audit(cohort[0], cohort[2], tmp_path / "fitted")
    assert result["status"] == "COMPLETED"
    ready = [f for f in value["failure_fold_plan"]["folds"] if f["status"] == "READY"]
    assert len(calls) == 4 * len(ready)
    assert result["actual_candidate_fits"] == result["planned_candidate_fit_budget"] == len(calls)
    features = models.summarize_prefix(cohort[0].dataset.X, history_steps=4, joint_count=2)
    for index, fold in enumerate(ready):
        train = fold["partitions"]["train"]["row_indices"]
        for X, y, regularization, maximum in calls[4*index:4*index+4]:
            np.testing.assert_array_equal(X, features[train])
            np.testing.assert_array_equal(y, cohort[0].dataset.y[train])
            assert maximum == 400
        completed = next(r for r in result["folds"] if r["fold_id"] == fold["fold_id"])
        assert completed["selection"]["selected_candidate"] == 3
        saved = models.load_model(tmp_path / "fitted" / fold["fold_id"] / "model.npz")
        np.testing.assert_array_equal(saved.normalizer.mean, features[train].mean(axis=0))
    with pytest.raises(FileExistsError):
        audit.run_audit(cohort[0], cohort[2], tmp_path / "fitted")


def test_fit_error_is_hard_stop_without_replacement_or_later_fit(cohort, tmp_path, monkeypatch):
    prepared(cohort)
    monkeypatch.setattr(audit, "_committed", lambda *args: dict(manufactured_commit_check_only=True))
    calls = []
    def error(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("manufactured optimizer failure")
    monkeypatch.setattr(audit, "fit_logistic", error)
    monkeypatch.setattr(audit, "frozen_category_uncertainty", lambda *a, **k: dict(manufactured_skip_bootstrap=True))
    result = audit.run_audit(cohort[0], cohort[2], tmp_path / "failed")
    assert len(calls) == 1 and result["status"] == "HARD_STOP"
    assert result["actual_candidate_fits"] == 1
    assert sum(r["status"] == "ERROR" for r in result["folds"]) == 1
    assert sum(r["status"] == "NOT_RUN" for r in result["folds"]) == 1
    assert (tmp_path / "failed" / "report.json").exists()


def uncertainty_args(cohort, output):
    return ["--dataset", "unused_manufactured_dataset", "--schedule", "unused_schedule",
            "--category-folds", "unused_folds", "--attempt-binding", "unused_binding",
            "--protocol", str(cohort[2]), "uncertainty", "--output", str(output)]


def test_direct_uncertainty_rechecks_prediction_bytes_after_calculation(cohort, monkeypatch):
    value = prepared(cohort)
    path = next(Path(row["predictions_path"]) for row in value["category_prediction_inventory"]["entries"]
                if row["status"] == "completed")
    def mutate(*args, **kwargs):
        path.write_text(path.read_text() + " ")
        return dict(percentile_95={})
    monkeypatch.setattr(audit, "group_uncertainty", mutate)
    with pytest.raises(ValueError, match="bound audit source/input changed"):
        audit.frozen_category_uncertainty(cohort[0], value)


def test_uncertainty_cli_revalidates_before_publishing(cohort, tmp_path, monkeypatch):
    value = prepared(cohort)
    path = next(Path(row["predictions_path"]) for row in value["category_prediction_inventory"]["entries"]
                if row["status"] == "completed")
    monkeypatch.setattr(training, "load_training_data", lambda *args: cohort[0])
    monkeypatch.setattr(audit, "_committed", lambda *args: dict(manufactured_commit_check_only=True))
    def mutate(*args, **kwargs):
        path.write_text(path.read_text() + " ")
        return dict(schema_version=audit.BOOTSTRAP_SCHEMA, manufactured_no_calculation=True)
    monkeypatch.setattr(audit, "frozen_category_uncertainty", mutate)
    output = tmp_path / "must_not_publish.json"
    with pytest.raises(ValueError, match="prediction hash mismatch"):
        audit.main(uncertainty_args(cohort, output))
    assert not output.exists()


def test_uncertainty_cli_rejects_existing_output_before_calculation(cohort, tmp_path, monkeypatch):
    prepared(cohort)
    monkeypatch.setattr(training, "load_training_data", lambda *args: cohort[0])
    def forbidden(*args, **kwargs):
        pytest.fail("existing output must refuse before validation/calculation")
    monkeypatch.setattr(audit, "validate_audit", forbidden)
    monkeypatch.setattr(audit, "frozen_category_uncertainty", forbidden)
    output = tmp_path / "existing.json"
    output.write_text("preserve me")
    with pytest.raises(FileExistsError, match="new uncertainty output"):
        audit.main(uncertainty_args(cohort, output))
    assert output.read_text() == "preserve me"
