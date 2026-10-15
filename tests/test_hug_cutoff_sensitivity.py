"""Manufactured strict recordings/models; no actual fitting, physics or HUG."""
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
sys.path.insert(0,str(Path(__file__).parent))
import test_hug_reserve_evaluation as support
from test_hug_reserve_evaluation import artifacts

SPEC=importlib.util.spec_from_file_location("cutoff_audit",Path(__file__).parents[1]/"scripts/audit_hug_cutoff_sensitivity.py")
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)
collector=audit.collector


@pytest.fixture(autouse=True)
def forbid_real_work(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail("real physics/fitting/HUG prohibited in manufactured cutoff tests")
    for name in ("mj_step","mj_step1","mj_step2"):monkeypatch.setattr(mujoco,name,forbidden)
    from grasp_failure_prediction.prediction import recorded_training
    monkeypatch.setattr(recorded_training,"fit_logistic",forbidden)
    monkeypatch.setattr(recorded_training,"_fit_gru",forbidden)


@pytest.fixture
def study(artifacts,monkeypatch):
    from grasp_failure_prediction.evaluation.prediction_recording import AttemptMetadata,PredictionRecorder,write_incomplete_attempt
    from grasp_failure_prediction.prediction.recorded_dataset import export_recorded_dataset,PrefixSpec
    monkeypatch.setattr(audit,"frozen",support.reserve)
    a=artifacts;project=a["project"]
    own=project/audit.OWN_SOURCE;own.parent.mkdir(parents=True);own.write_text("# manufactured bound audit source\n")
    cohort=support.reserve.expected_reserve_cohort(project)
    names=a["contract"]["joint_names"];limits=np.tile([-1.,1.],(24,1));h="sha256:"+"b"*64
    raw=a["inputs"]/"runs/manufactured_study"
    roots=[];attempts=[]
    for i,name in enumerate(("test_late_close","test_all_close","train","validation","early_failure","mapping_refusal")):
        root=raw/name/"prediction_recording";roots.append(root)
        hashes={key:h for key in ("code","configuration","inputs","model","assets","native_pose","mapped_pose")}
        hashes["label_rule"]=cohort["label_rule_sha256"]
        metadata=dict(attempt_id=name,hand_id=cohort["hand_id"],execution_protocol_id=cohort["execution_protocol_id"],
            object_id="manufactured_object",geometry_group_id=h,observation_group_id=h,proposal_group_id="sha256:"+str(i)*64,
            mapping=dict(status="ok"),execution_status="completed",provenance_hashes=hashes)
        if name=="mapping_refusal":
            metadata.update(mapping=dict(status="unreachable_pregrasp",code="manufactured",message="manufactured static refusal"),execution_status="not_executed")
            write_incomplete_attempt(root,AttemptMetadata(**metadata))
        else:
            recorder=PredictionRecorder(names,limits)
            for t in range(1,164):
                time=t*.04;close_start=2.08 if name=="test_late_close" else 1.2
                phase="approach" if time<close_start-1e-9 else "close_fingers" if time<=2.96+1e-9 else "lift"
                recorder.append(time_s=time,action_interval_start_s=(t-1)*.04,phase=phase,
                    hand_qpos=np.full(24,(i+time)/100),hand_qvel=np.full(24,.2),requested_target_rad=np.full(24,.3),applied_target_rad=np.full(24,.3))
            success=name in {"test_late_close","train"}
            first=2.8 if name=="early_failure" else 3.4 if name=="test_all_close" else None
            metadata["outcome"]=dict(success=success,failure_type=None if success else "failed_acquisition",
                label_rule_id=cohort["label_rule_id"],first_failure_time_s=first)
            recorder.write(root,AttemptMetadata(**metadata))
        attempts.append(dict(attempt_id=name,recording_path=str(root)))
    dataset=project/"manufactured_original_prefixes"
    manifest=export_recorded_dataset(roots,dataset,prefix_spec=PrefixSpec(**{k:a["contract"][k] for k in PrefixSpec.model_fields}))
    study_path=project/"manufactured_study_summary.json"
    support.write(study_path,dict(condition="study",status="COMPLETED_BOUNDED_QUOTA",preparation_sha256=collector.sha(a["main"]),
        quota=6,attempts=attempts,raw_artifact_bindings=support.reserve._files(raw)))
    report=json.loads(a["report"].read_text())
    del report["source_bindings"]["input_files"][str(a["manifest"])]
    a["manifest"]=dataset/"manifest.json"
    report["source_bindings"]["input_files"][str(dataset/"dataset.npz")]=collector.sha(dataset/"dataset.npz")
    ids=dict(train=["train"],validation=["validation"],test=["test_late_close","test_all_close"])
    report["folds"][0]["partitions"]={role:dict(episode_ids=values,n=len(values)) for role,values in ids.items()}
    for model in report["folds"][0]["models"]:
        directory=a["report"].parent/model["model_directory"]
        metadata=json.loads((directory/"model.json").read_text())
        metadata.update(training_episode_ids_sha256=collector.digest(ids["train"]),validation_episode_ids_sha256=collector.digest(ids["validation"]))
        (directory/"model.json").write_text(json.dumps(metadata))
        (directory/"predictions.json").write_text(json.dumps(dict(episode_ids=ids["test"],y_success=[1,0],p_success=[.5,.5],warning_threshold=.5)))
        model.update(model_metadata_sha256=collector.sha(directory/"model.json"),predictions_sha256=collector.sha(directory/"predictions.json"))
    a["report"].write_text(json.dumps(report));support.refreeze_manufactured_training(a)
    a.update(study=study_path,raw=raw,dataset=dataset,output=project/"cutoff_evidence",roots=roots)
    return a


def prepare(study):
    a=study
    return audit.prepare(a["project"],a["main"],a["training_protocol"],a["report"],a["study"],a["output"])


def test_prepare_does_not_extract_export_or_predict(study,monkeypatch):
    from grasp_failure_prediction.prediction import recorded_dataset,recorded_training
    def forbidden(*args,**kwargs):pytest.fail("preparation extracted/exported/predicted")
    monkeypatch.setattr(recorded_dataset,"_prefix",forbidden)
    monkeypatch.setattr(recorded_dataset,"export_recorded_dataset",forbidden)
    monkeypatch.setattr(recorded_training,"predict_saved_model",forbidden)
    p=prepare(study)
    assert p["cutoffs_s"]==[2.96,2.76,2.56,2.36] and len(p["models"]["bundles"])==15
    assert p["original_folds"]["manufactured_fold"]["test"]==["test_late_close","test_all_close"]
    assert not (study["output"]/"prefixes").exists()
    audit.freeze(study["output"])


@pytest.mark.parametrize("change",["grid","threshold","primary","fit","model_subset"])
def test_frozen_policy_cannot_select_cutoff_or_model(study,change):
    prepare(study);path=study["output"]/"protocol.draft.json";p=json.loads(path.read_text())
    if change=="grid":p["cutoffs_s"].pop()
    elif change=="threshold":p["warning_threshold"]=.4
    elif change=="primary":p["primary_model"]="small_gru"
    elif change=="fit":p["fit_allowed"]=True
    else:p["models"]["bundles"].pop()
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):audit.freeze(study["output"])


@pytest.mark.parametrize("change",["drop_null_attempt","source_inventory","original_dataset"])
def test_edited_draft_cannot_change_original_denominators_or_inputs(study,change):
    prepare(study);path=study["output"]/"protocol.draft.json";p=json.loads(path.read_text())
    if change=="drop_null_attempt":p["attempts"].pop()
    elif change=="source_inventory":p["source_bindings"].pop(next(iter(p["source_bindings"])))
    else:p["original_dataset"]=str(study["raw"])
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match="inventory changed"):audit.freeze(study["output"])


def test_raw_tamper_refuses_before_export(study):
    prepare(study);audit.freeze(study["output"])
    (study["roots"][0]/"attempt.json").write_text("changed")
    with pytest.raises(ValueError,match="frozen source/input changed"):audit.run(study["output"])
    assert not (study["output"]/"started.json").exists()


def test_missing_window_excluded_but_never_new_eligible_test_or_training_rows(study,monkeypatch):
    from grasp_failure_prediction.prediction import recorded_training
    prepare(study);audit.freeze(study["output"]);calls=[]
    def predict(directory,X):calls.append((str(directory),X.copy()));return np.full(len(X),.5)
    monkeypatch.setattr(recorded_training,"predict_saved_model",predict)
    result=audit.run(study["output"])
    assert result["status"]=="COMPLETED_DIAGNOSTIC" and result["baseline_exact_arrays"]
    assert len(calls)==15*4
    assert [item["included"] for item in result["cutoffs"]]==[4,4,4,4]
    # Early failure becomes causally eligible earlier, but was not in any original model test set.
    assert any(row["attempt_id"]=="early_failure" and row["status"]=="INCLUDED" for row in result["cutoffs"][1]["audit_rows"])
    assert any(row["attempt_id"]=="test_late_close" and row["status"]=="UNAVAILABLE" for row in result["cutoffs"][1]["audit_rows"])
    for model in result["models"]:
        assert model["common_episode_ids"]==["test_all_close"]
        assert model["cutoffs"][0]["episode_ids"]==["test_late_close","test_all_close"]
        assert all(row["episode_ids"]==["test_all_close"] for row in model["cutoffs"][1:])
        assert all(row["common_metrics"]["episodes"]==1 for row in model["cutoffs"])
        assert model["cutoffs"][1]["known_first_failure_proxy_lead"]["lead_s"]==pytest.approx([.64])
    assert [len(x) for _,x in calls[:4]]==[2,1,1,1]
    with pytest.raises(FileExistsError):audit.run(study["output"])


@pytest.mark.parametrize("fault",["alignment","early_failure_alignment","unexpected_exporter","baseline_bytes","prediction"])
def test_unexpected_input_export_or_prediction_fault_hard_stops(study,monkeypatch,fault):
    from grasp_failure_prediction.prediction import recorded_dataset,recorded_training
    prepare(study);audit.freeze(study["output"]);calls=[]
    def predict(*args):calls.append(1);return np.full(len(args[1]),.5)
    monkeypatch.setattr(recorded_training,"predict_saved_model",predict)
    if fault=="alignment":
        original=recorded_dataset._prefix
        def bad(*args):raise ValueError("prediction history does not match the declared sample interval")
        monkeypatch.setattr(recorded_dataset,"_prefix",bad)
    elif fault=="early_failure_alignment":
        from grasp_failure_prediction.evaluation import prediction_recording
        original=prediction_recording.load_attempt
        def bad(path):
            attempt,payload,arrays=original(path)
            if attempt.attempt_id=="early_failure":arrays["time_s"][1]+=.01
            return attempt,payload,arrays
        monkeypatch.setattr(prediction_recording,"load_attempt",bad)
    elif fault=="unexpected_exporter":
        def bad(*args,**kwargs):raise RuntimeError("manufactured unexpected exporter error")
        monkeypatch.setattr(recorded_dataset,"export_recorded_dataset",bad)
    elif fault=="baseline_bytes":
        original=audit._export_cutoff
        def bad(*args):
            info,data=original(*args)
            if args[1]==2.96:data.X[0,0]+=1
            return info,data
        monkeypatch.setattr(audit,"_export_cutoff",bad)
    else:
        def bad(*args):calls.append(1);raise RuntimeError("manufactured prediction error")
        monkeypatch.setattr(recorded_training,"predict_saved_model",bad)
    result=audit.run(study["output"])
    assert result["status"]=="HARD_STOP"
    assert len(calls)==(1 if fault=="prediction" else 0)
    with pytest.raises(FileExistsError):audit.run(study["output"])


def test_zero_early_windows_leave_grid_undefined_without_replacement(study,monkeypatch):
    from grasp_failure_prediction.prediction import recorded_dataset,recorded_training
    prepare(study);audit.freeze(study["output"])
    original=recorded_dataset._prefix;calls=[]
    def missing(arrays,spec):
        if spec.prediction_time_s<2.96:raise ValueError("insufficient samples before prediction cutoff")
        return original(arrays,spec)
    monkeypatch.setattr(recorded_dataset,"_prefix",missing)
    monkeypatch.setattr(recorded_training,"predict_saved_model",lambda directory,X:calls.append(len(X)) or np.full(len(X),.5))
    result=audit.run(study["output"])
    assert result["status"]=="COMPLETED_DIAGNOSTIC" and len(result["cutoffs"])==4
    assert all(row["status"]=="UNDEFINED_NO_ELIGIBLE_PREFIXES" for row in result["cutoffs"][1:])
    assert calls==[2]*15 and all(model["common_episode_ids"]==[] for model in result["models"])
    assert all(cell["common_metrics"] is None for model in result["models"] for cell in model["cutoffs"])
