"""Manufactured frozen predictions/identities; no real fitting or dynamics."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

import audit_hug_historical_prediction as audit
from grasp_failure_prediction.prediction import model, retrospective_holdout as loo


@pytest.fixture(autouse=True)
def no_physics(monkeypatch):
    def forbidden(*a,**kw):pytest.fail("historical audit tests forbid real physics")
    for name in ("mj_step","mj_step1","mj_step2"):monkeypatch.setattr(mujoco,name,forbidden)


def cohort_manifests():
    ref = dict(embodiment_id="shadow_hand_right",environment=dict(environment_id="adroit_shadow_tabletop_v2",
        runner="grasp_failure_prediction.evaluation.runner:ActuatedShadowRunner"),
        execution_protocol=dict(execution_protocol_id="fixed_grasp_lift_v2"),source_resolved_case_sha256="a"*64)
    ref["environment_config_hash"] = audit.canonical(ref["environment"])
    ref["execution_protocol_config_hash"] = audit.canonical(ref["execution_protocol"])
    source = dict(source_file_sha256={name:"sha256:"+"b"*64 for name in ("runner.py","noisy_rollouts.py","scoring.py")})
    return [dict(episodes=[dict(raw_files_sha256={"resolved_case.json":"sha256:"+"a"*64})],provenance_sources={"old":source}),
            dict(execution_reference=ref,provenance_sources={"fresh":deepcopy(source)})]


def test_original_same_controller_cohort_is_explicit():
    result = audit.execution_cohort(cohort_manifests())
    assert result["execution_protocol_id"] == "fixed_grasp_lift_v2"
    assert result["hand_id"] == "shadow_hand_right"


@pytest.mark.parametrize("change", ["new_controller","changed_source","missing_original_reference"])
def test_changed_or_unbound_controller_cannot_enter_historical_pool(change):
    manifests = cohort_manifests()
    if change == "new_controller":
        manifests[1]["execution_reference"]["execution_protocol"]["execution_protocol_id"] = "hug_grasp_approach_v1"
    elif change == "changed_source":
        manifests[1]["provenance_sources"]["fresh"]["source_file_sha256"]["runner.py"] = "sha256:"+"c"*64
    else:
        manifests[1]["execution_reference"]["source_resolved_case_sha256"] = "d"*64
    with pytest.raises(ValueError):audit.execution_cohort(manifests)


def test_read_only_guard_forbids_optimizers_and_all_step_apis():
    original = model.fit_logistic
    with audit.guarded() as counts:
        for name in ("mj_step","mj_step1","mj_step2"):
            with pytest.raises(RuntimeError,match="forbids physics"):getattr(mujoco,name)(None,None)
        with pytest.raises(RuntimeError,match="forbids fitting"):model.fit_logistic(None,None)
        with pytest.raises(RuntimeError,match="forbids fitting"):loo.select_model(None,None,None,None)
    assert counts == dict(integration_api_attempts=3,optimization_attempts=2)
    assert model.fit_logistic is original


def test_explicit_fit_scope_counts_mock_calls_without_real_optimization(monkeypatch):
    monkeypatch.setattr(model,"fit_logistic",lambda *a,**kw:"manufactured result")
    with audit.guarded(allow_fitting=True) as counts:
        assert model.fit_logistic(None,None) == "manufactured result"
    assert counts["optimization_attempts"] == 1 and counts["integration_api_attempts"] == 0


@pytest.fixture
def archived(tmp_path):
    root = tmp_path/"archive";root.mkdir();(root/"source").mkdir()
    (root/"source/retrospective_holdout.py").write_text("# manufactured source snapshot\n")
    (root/"source/HUG_RETROSPECTIVE_EVALUATION_V1.md").write_text("Manufactured frozen declaration.\n")
    data = dict(y=np.tile([0,1],6).astype(np.uint8),geometry_group_ids=np.repeat(["a","b","c"],4),
                observation_group_ids=np.repeat(["a0","a1","b0","b1","c0","c1"],2),
                group_ids=np.repeat([f"p{i}" for i in range(6)],2))
    ids = [f"manufactured/episode_{i:06d}" for i in range(12)]
    report = dict(schema_version="hug_retrospective_object_holdout_v1",commit="manufactured_previous_fit",attempts=12,successes=6,failures=6,folds=[],
        source_sha256=audit.sha(root/"source/retrospective_holdout.py")[7:],
        protocol_sha256=audit.sha(root/"source/HUG_RETROSPECTIVE_EVALUATION_V1.md")[7:])
    with (root/"predictions.csv").open("w",newline="") as handle:
        writer = csv.writer(handle);writer.writerow(["held_geometry","model","episode_id","success","p_success"])
        for fold in loo.object_folds(data["geometry_group_ids"],data["observation_group_ids"],data["group_ids"]):
            stored = dict(held_geometry=fold["held_geometry"],partitions={role:dict(episodes=[ids[i] for i in index],n=len(index),
                observations=len(set(data["observation_group_ids"][index])),proposals=len(set(data["group_ids"][index])))
                for role,index in fold["indices"].items()},models={})
            test = fold["indices"]["test"]
            for name in audit.MODEL_NAMES:
                probability = np.full(4,.5) if name=="training_prevalence" else np.array([.2,.7,.4,.8])
                stored["models"][name] = dict(metrics=model.outcome_metrics(data["y"][test],probability),calibration=loo.calibration(data["y"][test],probability))
                for i,p in zip(test,probability,strict=True):writer.writerow([fold["held_geometry"],name,ids[i],int(data["y"][i]),p])
            report["folds"].append(stored)
    report["predictions_sha256"] = audit.sha(root/"predictions.csv")[7:]
    audit.write(root/"report.json",report)
    return root,data,ids


def test_archived_prediction_metrics_replay_without_claiming_available_weights(archived):
    root,data,ids = archived
    with audit.guarded() as counts:result = audit.replay_archived_loo(root,data,ids)
    assert result["status"] == "VERIFIED_PERSISTED_PREDICTIONS_ONLY"
    assert result["attempts"] == 12 and len(result["folds"]) == 3
    assert not result["coefficients_available"] and result["coefficient_reproduction"].startswith("NOT_RUN")
    assert counts["optimization_attempts"] == counts["integration_api_attempts"] == 0


@pytest.mark.parametrize("change", ["csv_bytes","membership","label","metric","snapshot"])
def test_archived_changed_payload_membership_label_metric_or_source_refused(archived,change):
    root,data,ids = archived
    report = audit.read(root/"report.json")
    if change == "csv_bytes":
        with (root/"predictions.csv").open("a") as handle:handle.write("changed\n")
    elif change == "membership":
        report["folds"][0]["partitions"]["test"]["episodes"][0] = "foreign"
    elif change == "label":
        # Preserve total class counts so row-level mismatch is the refusal.
        data["y"][[0,1]] = data["y"][[1,0]]
    elif change == "metric":
        report["folds"][0]["models"]["commands_only"]["metrics"]["failure_roc_auc"] = .123
    else:
        (root/"source/retrospective_holdout.py").write_text("changed")
    (root/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):audit.replay_archived_loo(root,data,ids)


def test_metrics_comparison_preserves_null_single_class_quantities():
    audit.compare_metrics(dict(auc=None,n=3,bins=[.1]),dict(auc=None,n=3,bins=[.1+1e-14]))
    with pytest.raises(ValueError):audit.compare_metrics(dict(auc=.5),dict(auc=None))


def test_refit_cannot_start_without_exact_declaration_and_commit(tmp_path,monkeypatch):
    declaration = dict(project=str(tmp_path))
    path = tmp_path/"protocol.json";audit.write(path,declaration)
    monkeypatch.setattr(audit,"refit_declaration",lambda project:declaration)
    def uncommitted(*args):raise ValueError("manufactured uncommitted declaration")
    monkeypatch.setattr(audit,"require_committed",uncommitted)
    monkeypatch.setattr(loo,"run",lambda *a:pytest.fail("refit before commit"))
    with pytest.raises(ValueError,match="uncommitted"):
        audit.run_refit(path,tmp_path/"never")
    assert not (tmp_path/"never").exists()
