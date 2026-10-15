"""Manufactured data and tiny fits; no real collection or model experiment."""
from copy import deepcopy
from dataclasses import replace
import json

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation.prediction_recording import (
    AttemptMetadata, PredictionRecorder, write_unexecuted_attempt, file_sha256,
)
from grasp_failure_prediction.evaluation.scenario_sampler import sample_scenarios, freeze_category_folds
from grasp_failure_prediction.prediction.recorded_dataset import PrefixSpec, export_recorded_dataset
from grasp_failure_prediction.prediction import recorded_training as rt


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no dynamics in manufactured training tests")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    # Only Git chronological-commit checking is replaced for manufactured tests.
    # All structured spec/source/data/hash checks run unchanged.
    monkeypatch.setattr(rt, "_committed_protocol_receipt", lambda path: {"manufactured_test_only": True})
    objects = [dict(object_id=f"object_{i}", category_id=f"cat_{i}", geometry_group_id=f"geom_{i}",
                    observation_group_ids=[f"capture_{i}_a", f"capture_{i}_b"]) for i in range(3)]
    schedule = sample_scenarios(objects, seed=9, attempts_per_object=6, generation_seeds=(0, 1, 2))
    folds = freeze_category_folds(schedule)
    dirs, mapping = [], {}
    hash_value = "sha256:" + "a" * 64
    for i, scenario in enumerate(schedule["scenarios"]):
        attempt_id = f"manufactured_{i}"
        mapping[attempt_id] = scenario["scenario_id"]
        success = bool(i % 2)
        meta = dict(attempt_id=attempt_id, hand_id="manufactured_shadow_v1", execution_protocol_id="manufactured_execution_v1",
                    object_id=scenario["object_id"], geometry_group_id=scenario["geometry_group_id"],
                    observation_group_id=scenario["observation_group_id"], proposal_group_id=scenario["proposal_group_id"],
                    mapping={"status": "ok"}, execution_status="completed",
                    outcome=dict(success=success, failure_type=None if success else "failed_acquisition",
                                 label_rule_id="manufactured_rule_v1", first_failure_time_s=None if success else .2),
                    provenance_hashes={**{key: hash_value for key in ("code", "configuration", "inputs", "model", "assets",
                                                                   "native_pose", "mapped_pose", "label_rule")},
                                       "scenario": rt.canonical_sha256(scenario)})
        root = tmp_path / "attempts" / attempt_id
        if i == 17:
            meta.update(mapping={"status": "refused_guard", "code": "synthetic", "message": "manufactured refusal"},
                        execution_status="not_executed", outcome=None)
            write_unexecuted_attempt(root, AttemptMetadata(**meta))
        else:
            recorder = PredictionRecorder(["j0", "j1"], [[-1., 1.], [-1., 1.]])
            sign = 1 if success else -1
            for k in range(1, 6):
                values = np.full(2, sign * .1 * k if k < 5 else 999.)
                target = np.full(2, sign * .2)
                recorder.append(time_s=k * .04, action_interval_start_s=(k - 1) * .04,
                                phase="close_fingers" if k < 5 else "lift", hand_qpos=values,
                                hand_qvel=values * .3, requested_target_rad=target, applied_target_rad=target,
                                diagnostics={"future_object_state": np.full(2, 9999.)})
            recorder.write(root, AttemptMetadata(**meta))
        dirs.append(root)
    dataset_dir = tmp_path / "dataset"
    export_recorded_dataset(dirs, dataset_dir, prefix_spec=PrefixSpec(
        protocol_id="manufactured_prefix_v1", prediction_time_s=.16, history_steps=4, control_timestep_s=.04))
    binding = dict(schema_version="hug_attempt_scenario_binding_v1", schedule_sha256=rt.canonical_sha256(schedule),
                   attempt_to_scenario=mapping)
    paths = []
    for name, value in (("schedule", schedule), ("folds", folds), ("binding", binding)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        paths.append(path)
    protocol = tmp_path / "manufactured_protocol.md"
    return dataset_dir, paths, schedule, folds, binding, protocol


def load(sources):
    return rt.load_training_data(sources[0], *sources[1])


def tiny_spec(**changes):
    return replace(rt.TrainingSpec(), regularization_grid=(.1,), include_gru=False, **changes)


def prepare(sources, data, spec=None):
    spec = spec or tiny_spec()
    rt.prepare_training_protocol(data, sources[-1], specification=spec)
    return sources[-1]


def rewrite_manufactured_labels(sources, predicate):
    """Create a new valid synthetic exported file, then reload through adapter."""
    root = sources[0]
    manifest = json.loads((root / "manifest.json").read_text())
    with np.load(root / "dataset.npz", allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    for i, row in enumerate(manifest["episodes"]):
        if predicate(row):
            arrays["y"][i] = 0
            row.update(source_task_label=0, failure_type="failed_acquisition", first_failure_time_s=.2)
    manifest["counts"].update(success=int(arrays["y"].sum()), failure=len(arrays["y"]) - int(arrays["y"].sum()))
    np.savez_compressed(root / "dataset.npz", **arrays)
    manifest["dataset_sha256"] = file_sha256(root / "dataset.npz")
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_adapter_binds_every_planned_attempt_and_strict_features(sources):
    data = load(sources)
    assert data.dataset.X.shape == (17, 24)
    assert data.dataset.manifest["counts"]["attempted"] == 18
    assert len(data.folds) == 3
    assert np.max(abs(data.dataset.X)) < 1  # future 999 and diagnostic 9999 excluded.
    for fold in data.folds:
        assert sum(len(rows) for rows in fold["indices"].values()) == 17
        for key in ("geometry_group_ids", "observation_group_ids", "group_ids"):
            groups = [set(getattr(data.dataset, key)[rows]) for rows in fold["indices"].values()]
            assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))


def test_frozen_fold_tampering_refused(sources):
    dataset, _, schedule, folds, binding, _ = sources
    altered = deepcopy(folds)
    first = next(iter(altered["folds"][0]["assignments"]))
    altered["folds"][0]["assignments"][first] = "train"
    if altered == folds:
        altered["folds"][0]["assignments"][first] = "validation"
    with pytest.raises(ValueError, match="exact frozen schedule"):
        rt._adapt_dataset(load(sources).dataset, schedule, altered, binding)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_hash"])
def test_binding_refuses_missing_repeat_or_foreign_scenario(sources, kind):
    _, _, schedule, folds, binding, _ = sources
    altered = deepcopy(binding)
    keys = list(altered["attempt_to_scenario"])
    if kind == "missing":
        del altered["attempt_to_scenario"][keys[0]]
    elif kind == "duplicate":
        altered["attempt_to_scenario"][keys[0]] = altered["attempt_to_scenario"][keys[1]]
    else:
        altered["schedule_sha256"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError):
        rt._adapt_dataset(load(sources).dataset, schedule, folds, altered)


def test_no_relabelling_geometry_as_new_category(sources):
    _, _, schedule, _, binding, _ = sources
    altered = deepcopy(schedule)
    for row in altered["scenarios"]:
        if row["category_id"] == "cat_1":
            row["geometry_group_id"] = "geom_0"
    new_binding = deepcopy(binding)
    new_binding["schedule_sha256"] = rt.canonical_sha256(altered)
    with pytest.raises(ValueError, match="conflicting geometry_group_id"):
        rt._adapt_dataset(load(sources).dataset, altered, freeze_category_folds(altered), new_binding)


def test_scenario_hash_must_be_recorded_at_collection(sources):
    data = load(sources)
    manifest = deepcopy(data.dataset.manifest)
    del manifest["episodes"][0]["provenance_hashes"]["scenario"]
    bundle = replace(data.dataset, manifest=manifest)
    with pytest.raises(ValueError, match="hash-bound"):
        rt._adapt_dataset(bundle, sources[2], sources[3], sources[4])


@pytest.mark.parametrize("changes", [dict(seeds=(0,)), dict(warning_threshold=.4),
    dict(regularization_grid=(.001,.01,.1,1.,10.)), dict(gru_candidates=((65,.001),)),
    dict(gru_epochs=0), dict(gru_weight_decay=-1)])
def test_predeclared_training_budget_validated(changes):
    with pytest.raises(ValueError):
        replace(rt.TrainingSpec(), **changes)


def test_tiny_manufactured_fits_save_readback_and_training_only_normalization(sources, tmp_path):
    data = load(sources)
    prepare(sources, data)
    output = tmp_path / "fitted"
    report = rt.train_comparison(data, output, specification=tiny_spec(), protocol_path=sources[-1])
    assert report["status"] == "completed_with_skips"
    for fold, adapted in zip(report["folds"], data.folds):
        assert len(fold["models"]) == 15
        train = adapted["indices"]["train"]
        test = adapted["indices"]["test"]
        for item in fold["models"]:
            if item["name"] == "small_gru":
                assert item["status"] == "skipped" and "disabled" in item["reason"]
                continue
            assert item["status"] == "completed", item
            model_dir = output / item["model_directory"]
            probability = rt.predict_saved_model(model_dir, data.dataset.X[test])
            assert np.isfinite(probability).all()
            assert item["metrics"]["episodes"] == len(test)
            assert "success_false_alarm_rate" in item["metrics"]
            assert "expected_calibration_error" in item["calibration"]
            if item["name"] != "training_prevalence":
                features = rt._features(data.dataset.X, item["name"], data.dataset.manifest["input_contract"])
                normalizer = rt._load_normalizer(model_dir / "normalizer.npz")
                np.testing.assert_array_equal(normalizer.mean, features[train].mean(axis=0))
                metadata = json.loads((model_dir / "model.json").read_text())
                assert len(metadata["selection"]["candidates"]) == 1
                assert metadata["selection_sha256"] == rt.canonical_sha256(metadata["selection"])
        for name in rt.MODEL_NAMES[:-1]:
            members = [item for item in fold["models"] if item["name"] == name]
            assert len({item["predictions_sha256"] for item in members}) == 1
    with pytest.raises(FileExistsError):
        rt.train_comparison(data, output, specification=tiny_spec(), protocol_path=sources[-1])


def test_class_degenerate_fitting_is_explicit_skip(sources, tmp_path):
    rewrite_manufactured_labels(sources, lambda row: True)
    data = load(sources)
    prepare(sources, data)
    report = rt.train_comparison(data, tmp_path / "degen", specification=tiny_spec(), protocol_path=sources[-1])
    assert all(item["status"] == "skipped" and item["reason"] == "single_class_train"
               and item["metrics"] is None for fold in report["folds"] for item in fold["models"])


def test_single_class_test_skips_ranking_but_preserves_actual_warning_counts(sources, tmp_path):
    rewrite_manufactured_labels(sources, lambda row: row["geometry_group_id"] == "geom_0")
    data = load(sources)
    prepare(sources, data)
    report = rt.train_comparison(data, tmp_path / "one_class", specification=tiny_spec(), protocol_path=sources[-1])
    completed = [item for item in report["folds"][0]["models"] if item["status"] == "completed"]
    assert completed
    for item in completed:
        assert item["ranking_status"] == "skipped_single_class_test"
        assert item["metrics"]["failure_roc_auc"] is None
        assert item["metrics"]["success_false_alarm_rate"] is None
        assert item["metrics"]["failure_confusion"]["false_positive"] == 0


def test_model_hash_tampering_refused(sources, tmp_path):
    data = load(sources)
    prepare(sources, data)
    report = rt.train_comparison(data, tmp_path / "fit", specification=tiny_spec(), protocol_path=sources[-1])
    item = next(x for x in report["folds"][0]["models"] if x["status"] == "completed")
    root = tmp_path / "fit" / item["model_directory"]
    with (root / "model.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        rt.predict_saved_model(root, data.dataset.X[:2])


def test_optional_gru_tiny_seeded_fit_and_safe_numpy_readback(sources, tmp_path):
    pytest.importorskip("torch")
    data = load(sources)
    # One fixed epoch/candidate, all frozen folds; manufactured test only.
    spec = replace(tiny_spec(), include_gru=True, gru_candidates=((2,.01),), gru_epochs=1)
    prepare(sources, data, spec)
    report = rt.train_comparison(data, tmp_path / "gru_fit", specification=spec, protocol_path=sources[-1])
    members = [x for x in report["folds"][0]["models"] if x["name"] == "small_gru"]
    assert len(members) == 3
    for item in members:
        assert item["status"] == "completed", item
        root = tmp_path / "gru_fit" / item["model_directory"]
        with np.load(root / "model.npz", allow_pickle=False) as value:
            assert "gru.weight_ih_l0" in value.files
        assert np.isfinite(rt.predict_saved_model(root, data.dataset.X[:2])).all()


def test_protocol_preparation_does_not_fit_and_refuses_overwrite(sources, monkeypatch):
    data = load(sources)
    monkeypatch.setattr(rt, "fit_logistic", lambda *args, **kwargs: pytest.fail("preparation attempted fitting"))
    declaration = rt.prepare_training_protocol(data, sources[-1], specification=tiny_spec())
    assert declaration["requires_commit_before_fit"] is True
    assert declaration["source_bindings"] == data.source_bindings
    with pytest.raises(FileExistsError):
        rt.prepare_training_protocol(data, sources[-1], specification=tiny_spec())


@pytest.mark.parametrize("drift", ["specification", "source_files", "source_bindings"])
def test_protocol_drift_refused_before_output_or_fit(sources, tmp_path, monkeypatch, drift):
    data = load(sources)
    prepare(sources, data)
    declaration = json.loads(sources[-1].read_text())
    if drift == "specification":
        declaration[drift]["include_gru"] = True
    elif drift == "source_files":
        key = next(iter(declaration[drift]))
        declaration[drift][key] = "sha256:" + "b" * 64
    else:
        declaration[drift]["dataset_sha256"] = "sha256:" + "b" * 64
    sources[-1].write_text(json.dumps(declaration))
    monkeypatch.setattr(rt, "fit_logistic", lambda *args, **kwargs: pytest.fail("drift reached fitting"))
    with pytest.raises(ValueError, match="protocol mismatch"):
        rt.train_comparison(data, tmp_path / "never", specification=tiny_spec(), protocol_path=sources[-1])
    assert not (tmp_path / "never").exists()


def test_in_memory_feature_or_fold_mutation_refused(sources, tmp_path):
    data = load(sources)
    prepare(sources, data)
    data.dataset.X[0, 0] += 1
    with pytest.raises(ValueError, match="changed after adapter"):
        rt.train_comparison(data, tmp_path / "never", specification=tiny_spec(), protocol_path=sources[-1])
    assert not (tmp_path / "never").exists()


def test_bound_file_mutation_refused_before_fit(sources, tmp_path):
    data = load(sources)
    prepare(sources, data)
    with sources[1][0].open("a") as handle:
        handle.write("\n")  # Same JSON semantics, different frozen file bytes.
    with pytest.raises(ValueError, match="file changed"):
        rt.train_comparison(data, tmp_path / "never", specification=tiny_spec(), protocol_path=sources[-1])
    assert not (tmp_path / "never").exists()


def test_uncommitted_protocol_outside_git_is_refused(tmp_path):
    path = tmp_path / "uncommitted.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="commit the prepared"):
        rt._committed_protocol_receipt(path)
