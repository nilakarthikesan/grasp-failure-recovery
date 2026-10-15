"""Manufactured recording/export inputs; analyzer never integrates or fits."""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import analyze_hug_build_study as analyzer
from grasp_failure_prediction.evaluation.prediction_recording import (
    AttemptMetadata, PredictionRecorder, write_incomplete_attempt,
)
from grasp_failure_prediction.evaluation.scenario_sampler import sample_scenarios, freeze_category_folds
from grasp_failure_prediction.prediction.recorded_dataset import export_recorded_dataset, PrefixSpec


@pytest.fixture(autouse=True)
def no_physics_or_fits(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("study analyzer tests forbid real integration and fitting")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)
    from grasp_failure_prediction.prediction import recorded_training, model
    monkeypatch.setattr(recorded_training, "train_comparison", forbidden)
    monkeypatch.setattr(model, "fit_logistic", forbidden)


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path/"study"
    root.mkdir()
    source = tmp_path/"synthetic_source.py"
    source.write_text("# manufactured source identity only\n")
    protocol = dict(quotas=dict(study_attempts=18, maximum_task_steps_per_attempt=20),
                    source_bindings={str(source):analyzer.collector.sha(source)}, input_bindings={}, asset_bindings={})
    analyzer.collector.write(root/"preparation.json",protocol)
    monkeypatch.setattr(analyzer.collector,"read_protocol",lambda path:protocol)
    objects = [dict(object_id=f"object_{i}", category_id=f"category_{i}", geometry_group_id=f"geometry_{i}",
                    observation_group_ids=[f"capture_{i}_a",f"capture_{i}_b"]) for i in range(3)]
    schedule = sample_scenarios(objects,seed=91,attempts_per_object=6,generation_seeds=(0,1,2))
    mapping = {}
    for i,row in enumerate(schedule["scenarios"]):
        row["attempt_id"] = f"manufactured_{i}"
        mapping[row["attempt_id"]] = row["scenario_id"]
    folds = freeze_category_folds(schedule)
    binding = dict(schema_version="hug_attempt_scenario_binding_v1",schedule_sha256=analyzer.collector.digest(schedule),
                   attempt_to_scenario=mapping)
    for filename,value in (("study_schedule.json",schedule),("study_folds.json",folds),
                           ("study_attempt_scenario_binding.json",binding)):
        analyzer.collector.write(root/filename,value)
    generation = dict(status="COMPLETED_BOUNDED_GENERATION",artifact_bindings={},
                       conditions=dict(study=dict(schedule=schedule,folds=folds,attempt_binding=binding)))
    analyzer.collector.write(root/"generation.json",generation)
    rows,dirs = [],[]
    hash_value = "sha256:"+"a"*64
    for i,scenario in enumerate(schedule["scenarios"]):
        path = root/"raw"/scenario["attempt_id"]
        path.mkdir(parents=True)
        refusal = i == 17
        success = i%2 == 0
        outcome = None if refusal else dict(success=success, failure_type=None if success else "failed_acquisition",
                                            label_rule_id="manufactured_rule_v1", first_failure_time_s=None)
        hashes = {key:hash_value for key in ("code","configuration","inputs","model","assets","native_pose","mapped_pose","label_rule")}
        hashes["scenario"] = analyzer.collector.digest(scenario)
        if not refusal:
            analyzer.collector.write(path/"scenario_realization.json",dict(scenario=scenario))
            hashes["scenario_realization"] = analyzer.collector.sha(path/"scenario_realization.json")
        metadata = AttemptMetadata(attempt_id=scenario["attempt_id"],hand_id="manufactured_shadow",
            execution_protocol_id="manufactured_controller_v1",object_id=scenario["object_id"],
            geometry_group_id=scenario["geometry_group_id"],observation_group_id=scenario["observation_group_id"],
            proposal_group_id=scenario["proposal_group_id"],mapping=dict(status="unreachable_pregrasp",code="manufactured_refusal",message="static refusal") if refusal else dict(status="ok"),
            execution_status="not_executed" if refusal else "completed",outcome=outcome,provenance_hashes=hashes)
        recording = path/"prediction_recording"
        if refusal:
            write_incomplete_attempt(recording,metadata)
        else:
            recorder = PredictionRecorder(["joint_0"],[[-1.,1.]])
            for k in range(1,6):
                recorder.append(time_s=k*.04,action_interval_start_s=(k-1)*.04,
                    phase="close_fingers" if k<5 else "hold",hand_qpos=[.1 if success else -.1],hand_qvel=[0.],
                    requested_target_rad=[.2],applied_target_rad=[.2],diagnostics={})
            recorder.write(recording,metadata)
        row = dict(attempt_id=scenario["attempt_id"],scenario_id=scenario["scenario_id"],
                   status="APPROACH_REFUSED" if refusal else "COMPLETED",outcome=outcome,
                   physics_steps=0 if refusal else 20,recording_path=str(recording),
                   attempt_sha256=analyzer.collector.sha(recording/"attempt.json"))
        if not refusal:
            row["scenario_realization_sha256"] = hashes["scenario_realization"]
        rows.append(row);dirs.append(recording)
    dataset = export_recorded_dataset(dirs,root/"study_prefixes",prefix_spec=PrefixSpec(
        protocol_id="manufactured_prefix_v1",prediction_time_s=.16,history_steps=4,control_timestep_s=.04))
    raw_bindings = {str(path):analyzer.collector.sha(path) for path in (root/"raw").rglob("*") if path.is_file()}
    summary = dict(schema_version="hug_build_collection_result_v1",condition="study",status="COMPLETED_BOUNDED_QUOTA",error=None,
        preparation_sha256=analyzer.collector.sha(root/"preparation.json"),generation_sha256=analyzer.collector.sha(root/"generation.json"),
        schedule_sha256=analyzer.collector.digest(schedule),quota=18,attempts=rows,counts=dict(COMPLETED=17,APPROACH_REFUSED=1),
        physics_steps=340,dataset=dataset,raw_artifact_bindings=raw_bindings,
        filters=dict(artifact_bindings={},proposals=dict(synthetic=dict(filter_status="accepted"))))
    analyzer.collector.write(root/"study_summary.json",summary)
    return SimpleNamespace(root=root,protocol=protocol,summary=summary,source=source)


def test_analysis_retains_nulls_coverage_and_frozen_class_eligibility(study,tmp_path):
    output = tmp_path/"analysis.json"
    result = analyzer.analyze(study.root,output)
    assert result["status"] == "COMPLETED_READ_ONLY_ANALYSIS"
    assert result["failure_taxonomy"]["counts"]["attempted"] == 18
    assert result["failure_taxonomy"]["counts"]["null_outcome"] == 1
    assert result["failure_taxonomy"]["counts"]["physically_labeled"] == 17
    assert result["coverage"]["scheduled"]["attempts"] == 18
    assert result["coverage"]["executed"]["attempts"] == result["coverage"]["included_prefixes"]["attempts"] == 17
    assert result["coverage"]["scheduled"]["geometries"] == 3
    assert all(row["two_classes"] for row in result["per_category_included_classes"].values())
    assert all(row["eligible_for_predeclared_comparison"] for row in result["frozen_fold_eligibility"])
    assert result["read_only_guard"]["integration_api_attempts"] == result["model_fits"] == 0
    assert not result["schedule_changed"] and not result["folds_changed"]
    with pytest.raises(FileExistsError):
        analyzer.analyze(study.root,output)


def test_analysis_refuses_incomplete_or_error_summary_before_output(study,tmp_path):
    summary = dict(study.summary,status="HARD_STOP",error=dict(message="manufactured fault"))
    (study.root/"study_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="completed study quota"):
        analyzer.analyze(study.root,tmp_path/"never.json")
    assert not (tmp_path/"never.json").exists()


def test_analysis_refuses_changed_bound_source_or_raw_bytes(study,tmp_path):
    study.source.write_text("changed")
    with pytest.raises(ValueError,match="frozen source/input changed"):
        analyzer.analyze(study.root,tmp_path/"never.json")


def test_analysis_refuses_changed_raw_recording_bytes(study,tmp_path):
    path = Path(study.summary["attempts"][0]["recording_path"])/"attempt.json"
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError,match="frozen source/input changed"):
        analyzer.analyze(study.root,tmp_path/"never.json")


def test_analysis_refuses_reassigned_frozen_fold(study,tmp_path):
    path = study.root/"study_folds.json"
    folds = json.loads(path.read_text())
    first = next(iter(folds["folds"][0]["assignments"]))
    folds["folds"][0]["assignments"][first] = "foreign"
    path.write_text(json.dumps(folds))
    with pytest.raises(ValueError,match="differ from frozen"):
        analyzer.analyze(study.root,tmp_path/"never.json")


def test_single_class_test_keeps_fit_eligible_but_ranking_undefined():
    data = SimpleNamespace(dataset=SimpleNamespace(y=np.array([0,1,0,1,0,0],dtype=np.uint8)),folds=(
        dict(fold_id="held_fixed",test_category="held",validation_category="validation",
             indices=dict(train=np.array([0,1]),validation=np.array([2,3]),test=np.array([4,5]))),))
    result = analyzer.fold_eligibility(data)[0]
    assert result["eligible_for_predeclared_comparison"]
    assert result["test_ranking"] == dict(status="UNDEFINED_SINGLE_CLASS_TEST",failure_roc_auc=None,failure_average_precision=None)
    data.dataset.y[3] = 0
    result = analyzer.fold_eligibility(data)[0]
    assert not result["eligible_for_predeclared_comparison"] and result["skip_reasons"] == ["single_class_validation"]


def test_empty_test_prevents_model_comparison_without_repairing_folds():
    data = SimpleNamespace(dataset=SimpleNamespace(y=np.array([0,1,0,1],dtype=np.uint8)),folds=(
        dict(fold_id="held_empty",test_category="held",validation_category="validation",
             indices=dict(train=np.array([0,1]),validation=np.array([2,3]),test=np.array([],dtype=int))),))
    result = analyzer.fold_eligibility(data)[0]
    assert not result["eligible_for_predeclared_comparison"] and result["skip_reasons"] == ["empty_test"]
    assert result["test_ranking"]["status"] == "UNDEFINED_EMPTY_TEST"


def test_all_three_step_apis_are_blocked_and_counted():
    with analyzer.no_physics() as counts:
        for name in ("mj_step","mj_step1","mj_step2"):
            with pytest.raises(RuntimeError,match="forbids physics"):
                getattr(mujoco,name)(None,None)
    assert counts["integration_api_attempts"] == 3
