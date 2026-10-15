"""Manufactured reserve artifacts only; no real inference, fitting or physics."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
SPEC = importlib.util.spec_from_file_location("reserve_evaluation", Path(__file__).parents[1] / "scripts/evaluate_hug_reserve_models.py")
reserve = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(reserve)
collector = reserve.collector


@pytest.fixture(autouse=True)
def forbid_physics_and_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real physics or fitting forbidden in reserve preparation tests")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)
    from grasp_failure_prediction.prediction import recorded_training as training
    monkeypatch.setattr(training, "fit_logistic", forbidden)
    monkeypatch.setattr(training, "_fit_gru", forbidden)


def write(path, value):
    collector.write(path, value)
    return path


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    from grasp_failure_prediction.prediction.recorded_dataset import PrefixSpec, _input_contract
    project = tmp_path / "source"; project.mkdir()
    inputs = tmp_path / "runtime"; (inputs / "runs").mkdir(parents=True)
    code = project / "code.py"; code.write_text("# manufactured source only\n")
    trainer = project / "src/grasp_failure_prediction/prediction/recorded_training.py"
    trainer.parent.mkdir(parents=True); trainer.write_text("# manufactured trainer declaration\n")
    fitted_sources = ["src/grasp_failure_prediction/" + name for name in (
        "prediction/recorded_training.py", "prediction/model.py", "prediction/recorded_dataset.py",
        "evaluation/prediction_recording.py", "evaluation/scenario_sampler.py")]
    for name in fitted_sources[1:] + ["src/grasp_failure_prediction/evaluation/scoring.py"]:
        path = project / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# manufactured dependency only\n")
    frozen_blobs = {}
    monkeypatch.setattr(reserve, "_fit_commit_blob_sha", lambda project, commit, relative: frozen_blobs[relative])
    monkeypatch.setattr(collector, "SOURCE_FILES", ("code.py",))
    monkeypatch.setattr(reserve, "EXTRA_SOURCES", ("code.py", "src/grasp_failure_prediction/prediction/recorded_training.py"))
    monkeypatch.setattr(reserve, "_require_commit", lambda paths: "manufactured-no-git-commit")
    prefix = dict(protocol_id="hug_grasp_approach_close_prefix_v1", prediction_time_s=2.96, history_steps=20, control_timestep_s=.04)
    contract = _input_contract(PrefixSpec(**prefix), [f"j{i}" for i in range(24)])
    main = dict(schema_version=collector.SCHEMA, project=str(project), inputs_project=str(inputs),
        source_bindings={str(code): collector.sha(code)}, input_bindings={}, asset_bindings={}, dependency_dirs=[],
        prefix=prefix, hug_root=str(tmp_path / "hug"), checkpoint=str(tmp_path / "weights.bin"))
    main_path = write(tmp_path / "main/preparation.json", main)
    state_hash = "sha256:" + "c" * 64
    write(main_path.parent / "generation.json", dict(status="COMPLETED_BOUNDED_GENERATION",
        preparation_sha256=collector.sha(main_path), loaded_model_state_sha256=state_hash))
    models = tmp_path / "models"
    schedule = dict(schema_version="hug_scenario_schedule_v1", scenarios=[dict(object_id="development_object", geometry_group_id="development_geometry")])
    schedule_path = write(tmp_path / "development_schedule.json", schedule)
    items = []
    for kind in reserve.MODEL_KINDS:
        for seed in (0, 1, 2):
            directory = models / kind / str(seed); directory.mkdir(parents=True)
            (directory / "model.npz").write_bytes(b"manufactured numeric model fixture")
            (directory / "normalizer.npz").write_bytes(b"manufactured normalizer fixture")
            selection = dict(declared=True, no_reserve_data=True)
            metadata = dict(schema_version="hug_recorded_predictor_v1", kind=kind, seed=seed, input_contract=contract,
                selection=selection, selection_sha256=collector.digest(selection),
                artifacts={name: collector.sha(directory / name) for name in ("model.npz", "normalizer.npz")})
            write(directory / "model.json", metadata)
            write(directory / "predictions.json", dict(manufactured_development=True))
            items.append(dict(name=kind, seed=seed, status="completed", model_directory=str(directory.relative_to(models)),
                model_metadata_sha256=collector.sha(directory / "model.json"), selection_sha256=collector.digest(selection),
                predictions_sha256=collector.sha(directory / "predictions.json")))
    manifest_path = write(tmp_path / "development_manifest.json", dict(
        schema_version="hug_recorded_prefix_dataset_v1", input_contract=contract,
        cohort=reserve.expected_reserve_cohort(project)))
    report_path = write(models / "report.json", dict(schema_version="hug_recorded_model_comparison_v1", status="completed",
        specification=dict(warning_threshold=.5), input_contract=contract,
        source_bindings=dict(input_files={str(schedule_path): collector.sha(schedule_path),
            str(manifest_path): collector.sha(manifest_path)}, schedule_sha256=collector.digest(schedule),
            dataset_manifest_sha256=collector.sha(manifest_path)),
        training_source_sha256=collector.sha(trainer), folds=[dict(fold_id="manufactured_fold", models=items)]))
    training_protocol = project / "research_artifacts/manufactured_training_protocol.json"
    state = dict(project=project, report=report_path, training_protocol=training_protocol,
                 manifest=manifest_path, fitted_sources=fitted_sources, frozen_blobs=frozen_blobs)
    refreeze_manufactured_training(state)
    qualifications = {}
    for name in reserve.RESERVES:
        object_id = "hug_" + name; geometry = collector.digest(dict(geometry=name))
        asset_root = inputs / "runs/assets" / name; asset_root.mkdir(parents=True)
        (asset_root / "object.xml").write_text('<mujoco><asset/></mujoco>')
        case = dict(object=dict(id=object_id, geometry=dict(mjcf_path=str((asset_root / "object.xml").relative_to(inputs)))))
        case_path = write(tmp_path / name / ("case_" + name + ".json"), case)
        proto = dict(schema_version="hug_reserve_asset_qualification_v1", runtime_project=str(inputs), asset_root=str(asset_root.parent),
            cases={case_path.name: collector.sha(case_path)},
            objects=[dict(object_id=object_id, geometry_content_hash=geometry,
                          geometry_files_sha256={"object.xml": collector.sha(asset_root / "object.xml")})])
        proto_path = write(case_path.parent / "protocol.json", proto)
        capture_root = inputs / "runs" / ("qualified_" + name)
        captures = []
        for camera in ("front_left", "front_right"):
            capture_id = name + "_" + camera
            capture = capture_root / capture_id
            write(capture / "scene_contract.json", dict(object_id=object_id, geometry_content_hash=geometry))
            write(capture / "capture_report.json", dict(accepted_stationary=True))
            captures.append(dict(object_id=object_id, capture_id=capture_id, camera_id=camera, status="PASS_STATIONARY_CAPTURE", outcome=None,
                capture_report_sha256=collector.sha(capture / "capture_report.json"),
                capture_report=dict(scene_contract_sha256=collector.sha(capture / "scene_contract.json"))))
        write(capture_root / "capture_qualification.json", dict(protocol_sha256=collector.sha(proto_path),
            freeze_commit=reserve.COFFEE_FREEZE if name == "coffee_cup" else "HEAD",
            grasp_inference_calls=0, grasp_labels=0, model_fits=0, captures=captures))
        qualifications[name] = (proto_path, capture_root)
    return dict(**state, inputs=inputs, main=main_path, qualifications=qualifications,
                output=tmp_path / "reserve_evidence", contract=contract, state_hash=state_hash)


def refreeze_manufactured_training(state):
    """Create synthetic immutable-byte receipts; never commit or fit real models."""
    project, report_path = state["project"], state["report"]
    report = json.loads(report_path.read_text())
    manifest = state["manifest"]
    report["source_bindings"]["input_files"][str(manifest)] = collector.sha(manifest)
    report["source_bindings"]["dataset_manifest_sha256"] = collector.sha(manifest)
    declaration = dict(schema_version="hug_recorded_training_protocol_v1", requires_commit_before_fit=True,
        specification=report["specification"], input_contract=report["input_contract"], source_bindings=report["source_bindings"],
        source_files={relative: collector.sha(project / relative) for relative in state["fitted_sources"]})
    state["training_protocol"].parent.mkdir(parents=True, exist_ok=True)
    state["training_protocol"].write_text(json.dumps(declaration))
    report.update(protocol_sha256=collector.sha(state["training_protocol"]),
        frozen_protocol_receipt=dict(commit="a" * 40, protocol_and_bound_sources_clean_at_HEAD=True))
    report_path.write_text(json.dumps(report))
    state["frozen_blobs"].update(declaration["source_files"])
    state["frozen_blobs"][state["training_protocol"].relative_to(project).as_posix()] = collector.sha(state["training_protocol"])


def prepare(artifacts):
    a = artifacts
    return reserve.prepare(a["project"], a["main"], a["report"], a["qualifications"], a["output"],
                           training_protocol=a["training_protocol"])


def test_preparation_hashes_all_bundles_and_exact_unused_capture_plan(artifacts, monkeypatch):
    from grasp_failure_prediction.prediction import recorded_training
    monkeypatch.setattr(recorded_training, "predict_saved_model", lambda *a: pytest.fail("preparation predicted reserve"))
    protocol = prepare(artifacts)
    assert len(protocol["models"]["bundles"]) == 15
    assert sum(row["role"] == "primary" for row in protocol["models"]["bundles"]) == 3
    assert len(protocol["generation_slots"]) == 8
    assert len({row["capture_id"] for row in protocol["generation_slots"]}) == 4
    assert {row["generation_seed"] for row in protocol["generation_slots"]} == {9000, 9001}
    assert protocol["warning_threshold"] == .5 and not protocol["fit_allowed"]
    assert not Path(protocol["raw_output"]).exists()
    assert protocol["objects"][0]["qualification_freeze_is_concrete"] is False
    reserve.freeze(artifacts["output"], "protocol")
    assert reserve.read_protocol(artifacts["output"])["models"] == protocol["models"]


def test_runtime_bootstraps_bound_project_before_registry_cohort_import(artifacts, monkeypatch):
    prepare(artifacts)
    state = {"bootstrapped": False}
    bootstrap = collector.bootstrap
    cohort = reserve.verify_training_cohort
    def initialize(protocol):
        state["bootstrapped"] = True
        return bootstrap(protocol)
    def verify(*args):
        assert state["bootstrapped"], "registry cohort imported before bound dependency bootstrap"
        return cohort(*args)
    monkeypatch.setattr(collector, "bootstrap", initialize)
    monkeypatch.setattr(reserve, "verify_training_cohort", verify)
    reserve.freeze(artifacts["output"], "protocol")
    state["bootstrapped"] = False
    reserve.read_protocol(artifacts["output"])


def test_zero_complete_models_blocks_without_side_effects(artifacts):
    report = json.loads(artifacts["report"].read_text())
    for item in report["folds"][0]["models"]:
        item.update(status="skipped", reason="single_class_train")
    artifacts["report"].write_text(json.dumps(report))
    with pytest.raises(reserve.ReserveBlocked, match="zero complete"):
        prepare(artifacts)
    assert not artifacts["output"].exists()


@pytest.mark.parametrize("relative", ["src/grasp_failure_prediction/prediction/model.py",
    "src/grasp_failure_prediction/prediction/recorded_dataset.py",
    "src/grasp_failure_prediction/evaluation/scenario_sampler.py"])
def test_unchanged_trainer_cannot_admit_changed_fitted_dependency(artifacts, relative):
    trainer = artifacts["project"] / "src/grasp_failure_prediction/prediction/recorded_training.py"
    original = collector.sha(trainer)
    (artifacts["project"] / relative).write_text("# changed predictor dependency after manufactured fitting\n")
    with pytest.raises(ValueError, match="frozen fitted source closure changed"):
        prepare(artifacts)
    assert collector.sha(trainer) == original and not artifacts["output"].exists()


def test_fitting_protocol_must_match_immutable_git_receipt(artifacts):
    relative = artifacts["training_protocol"].relative_to(artifacts["project"]).as_posix()
    artifacts["frozen_blobs"][relative] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="immutable fitting commit"):
        prepare(artifacts)
    assert not artifacts["output"].exists()


@pytest.mark.parametrize("field", ["hand_id", "execution_protocol_id", "label_rule_id", "label_rule_sha256"])
def test_same_prefix_from_different_physical_cohort_is_refused(artifacts, field):
    manifest = json.loads(artifacts["manifest"].read_text())
    manifest["cohort"][field] = "manufactured_different_cohort"
    artifacts["manifest"].write_text(json.dumps(manifest))
    # This is a fully hash-bound alternative synthetic fitting cohort, not a file-tamper test.
    refreeze_manufactured_training(artifacts)
    assert manifest["input_contract"] == artifacts["contract"]
    with pytest.raises(ValueError, match="cohort hand/controller/label rule differs"):
        prepare(artifacts)
    assert not artifacts["output"].exists()


def test_fitted_dependency_is_rechecked_before_fresh_inference(artifacts):
    prepare(artifacts); reserve.freeze(artifacts["output"], "protocol")
    (artifacts["project"] / "src/grasp_failure_prediction/prediction/model.py").write_text("# changed after reserve freeze\n")
    with pytest.raises(ValueError, match="frozen fitted source closure changed"):
        reserve.read_protocol(artifacts["output"])


@pytest.mark.parametrize("change", ["missing_kind", "model_hash", "threshold"])
def test_no_model_subset_or_thawed_bundle_can_be_selected(artifacts, change):
    report = json.loads(artifacts["report"].read_text())
    if change == "missing_kind": report["folds"][0]["models"].pop()
    elif change == "threshold": report["specification"]["warning_threshold"] = .4
    else:
        item = report["folds"][0]["models"][0]
        (artifacts["report"].parent / item["model_directory"] / "model.npz").write_bytes(b"changed")
    artifacts["report"].write_text(json.dumps(report))
    with pytest.raises(ValueError): prepare(artifacts)


def test_bear_or_failed_capture_cannot_replace_fixed_reserves(artifacts):
    bad = dict(artifacts["qualifications"]); bad["bear_figure"] = bad.pop("coffee_cup")
    with pytest.raises(ValueError, match="exactly bell_pepper"):
        reserve.prepare(artifacts["project"], artifacts["main"], artifacts["report"], bad, artifacts["output"],
                        training_protocol=artifacts["training_protocol"])
    root = artifacts["qualifications"]["bell_pepper"][1]
    qualification = json.loads((root / "capture_qualification.json").read_text())
    qualification["captures"][0]["status"] = "CAPTURE_REJECTED"
    (root / "capture_qualification.json").write_text(json.dumps(qualification))
    with pytest.raises(ValueError, match="exactly two qualified"):
        prepare(artifacts)


def test_committed_protocol_cannot_silently_change_model_or_capture_bytes(artifacts):
    protocol = prepare(artifacts); reserve.freeze(artifacts["output"], "protocol")
    capture = Path(protocol["generation_slots"][0]["observation_path"])
    (capture / "capture_report.json").write_text("changed")
    with pytest.raises(ValueError, match="frozen source/input changed"):
        reserve.read_protocol(artifacts["output"])


def test_edited_draft_cannot_drop_diagnostics_or_change_declared_seeds(artifacts):
    prepare(artifacts)
    path=artifacts["output"]/"protocol.draft.json"
    original=json.loads(path.read_text())
    altered=json.loads(path.read_text());altered["models"]["bundles"].pop()
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match="model inventory"):
        reserve.freeze(artifacts["output"],"protocol")
    original["generation_slots"][0]["generation_seed"]=9002
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError,match="fixed8/32"):
        reserve.freeze(artifacts["output"],"protocol")


def proposal_rows(protocol):
    return [dict(**slot, status="MAPPED", mapping_status="ok", proposal_group_id=collector.digest(slot),
                 geometry_group_id=collector.digest(dict(object=slot["object_id"])),
                 observation_group_id=collector.digest(dict(capture=slot["capture_id"])))
            for slot in protocol["generation_slots"]]


def test_reserve_schedule_has32_fixed_variants_and_no_training_folds(artifacts):
    protocol = prepare(artifacts)
    schedule = reserve.reserve_schedule(proposal_rows(protocol))
    assert len(schedule["scenarios"]) == 32 and "folds" not in schedule
    assert len({row["scenario_id"] for row in schedule["scenarios"]}) == 32
    for slot in protocol["generation_slots"]:
        selected = [row for row in schedule["scenarios"] if row["generation_slot_id"] == slot["slot_id"]]
        assert {(row["mass_multiplier"], row["friction_coefficient"]) for row in selected} == {(.75,.5),(.75,1.),(1.25,.5),(1.25,1.)}
        assert all(row["sensor_noise_std_rad"] == row["table_offset_m"] == 0 for row in selected)


def test_missing_authorization_blocks_fresh_inference(artifacts, monkeypatch):
    prepare(artifacts); reserve.freeze(artifacts["output"], "protocol")
    bad = write(artifacts["output"].parent / "bad_auth.json", dict(authorized=False))
    with pytest.raises(ValueError, match="authorization"):
        reserve.generate(artifacts["output"], bad)
    assert not Path(reserve.read_protocol(artifacts["output"])["raw_output"]).exists()


def test_class_degenerate_ranking_is_null_with_true_warning_counts():
    metrics = reserve._metrics(np.array([0,0,0], dtype=np.uint8), np.array([.9,.1,.2]))
    assert metrics["failure_roc_auc"] is metrics["failure_average_precision"] is None
    assert metrics["failure_confusion"]["true_positive"] == 2
    assert metrics["failure_confusion"]["false_negative"] == 1
    assert metrics["success_false_alarm_rate"] is None
    assert metrics["ranking_status"] == "skipped_single_class"


def fake_generation(artifacts, monkeypatch, *, refusal_seed=None, fault_seed=None):
    from grasp_failure_prediction.proposals import generate as generator
    calls=[]
    def binding(request, project):
        return dict(geometry_group_id=collector.digest(dict(geometry_content_hash=request.object.geometry_content_hash)),
                    observation_group_id=collector.digest(dict(capture=str(request.observation_path))))
    def generate(request_path, directory, **kwargs):
        request=json.loads(Path(request_path).read_text());calls.append((request,kwargs))
        native=Path(directory)/"sample_0000/native";native.mkdir(parents=True)
        (native/"proposal.pkl").write_bytes(("manufactured_"+str(request["seed"])).encode())
        write(native/"inference_report.json",dict(loaded_model_state_sha256=artifacts["state_hash"]))
        if request["seed"]==fault_seed:raise RuntimeError("manufactured inference fault")
        if request["seed"]==refusal_seed:
            write(native/"mapping_record.json",dict(mapping_status="refused_guard"))
            raise ValueError("manufactured static refusal")
        path=native.parent/"proposal.json";write(path,dict(manufactured=True))
        return SimpleNamespace(proposals=[SimpleNamespace(proposal_path=Path("sample_0000/proposal.json"))])
    def row(protocol,slot,path,validation):
        request=generator.ProposalRequest.model_validate(calls[-1][0])
        bound=binding(request,artifacts["inputs"])
        return dict(**slot,**bound,status="MAPPED",mapping_status="ok",outcome=None,
            proposal_path=str(path),proposal_group_id=collector.digest(slot),proposal_hash="sha256:"+"d"*64)
    monkeypatch.setattr(generator,"generate_proposals",generate)
    monkeypatch.setattr(generator,"_expected_input_binding",binding)
    monkeypatch.setattr(collector,"_validate_fresh",lambda *a:dict(static_input_gate="PASS"))
    monkeypatch.setattr(collector,"_proposal_row",row)
    return calls


def authorize(artifacts,stage,quota,extra=None):
    path=artifacts["output"].parent/(stage+"_auth.json")
    write(path,dict(schema_version="hug_reserve_authorization_v1",authorized=True,stage=stage,quota=quota,
        bindings={str(artifacts["output"] / "protocol.json"):collector.sha(artifacts["output"] / "protocol.json"),**(extra or {})}))
    return path


def test_fresh_generation_forwards_declared_runtime_and_retains_refusals(artifacts,monkeypatch):
    protocol=prepare(artifacts);reserve.freeze(artifacts["output"],"protocol")
    calls=fake_generation(artifacts,monkeypatch,refusal_seed=9001)
    authorization=authorize(artifacts,"generate",8)
    result=reserve.generate(artifacts["output"],authorization)
    assert result["status"]=="COMPLETED_BOUNDED_GENERATION" and len(calls)==8
    assert result["counts"]=={"MAPPED":4,"MAPPING_REFUSED":4}
    assert all(kwargs["runtime_dependency_dirs"]==protocol["dependency_dirs"] for _,kwargs in calls)
    assert all(row["outcome"] is None for row in result["proposals"])
    assert len(result["schedule"]["scenarios"])==32
    reserve.freeze(artifacts["output"],"generation")
    with pytest.raises(FileExistsError):reserve.generate(artifacts["output"],authorization)


def test_unexpected_inference_error_accounts_all_slots_without_replacement(artifacts,monkeypatch):
    prepare(artifacts);reserve.freeze(artifacts["output"],"protocol")
    calls=fake_generation(artifacts,monkeypatch,fault_seed=9001)
    result=reserve.generate(artifacts["output"],authorize(artifacts,"generate",8))
    assert result["status"]=="HARD_STOP" and len(calls)==2
    assert [row["status"] for row in result["proposals"]]==["MAPPED","ERROR"]+["NOT_RUN"]*6
    assert "schedule" not in result
    with pytest.raises(ValueError,match="cannot freeze"):
        reserve.freeze(artifacts["output"],"generation")


def test_task_fault_preserves32_null_attempt_denominators_and_never_predicts(artifacts,monkeypatch):
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    prepare(artifacts);reserve.freeze(artifacts["output"],"protocol")
    fake_generation(artifacts,monkeypatch)
    reserve.generate(artifacts["output"],authorize(artifacts,"generate",8))
    reserve.freeze(artifacts["output"],"generation")
    calls=[]
    def fault(protocol,proposal,scenario,directory,filter_record):
        calls.append(scenario)
        row=collector._null_attempt(protocol,proposal,scenario,directory,mapping="setup_error",code="manufactured_fault",status="ERROR")
        row["error"]=dict(type="RuntimeError",message="manufactured task fault")
        return row
    monkeypatch.setattr(collector,"_collect_attempt",fault)
    gen=artifacts["output"]/"generation.json"
    result=reserve.collect(artifacts["output"],authorize(artifacts,"collect",32,{str(gen):collector.sha(gen)}))
    assert result["status"]=="HARD_STOP" and len(calls)==1 and len(result["attempts"])==32
    assert [row["status"] for row in result["attempts"]]==["ERROR"]+["NOT_RUN"]*31
    assert all(load_attempt(row["recording_path"])[0].outcome is None for row in result["attempts"])
    with pytest.raises(ValueError,match="completed predeclared"):
        reserve.evaluate(artifacts["output"])


@pytest.mark.parametrize("prediction_fault",[False,True])
def test_evaluation_predicts_each_frozen_bundle_once_on_same_eligible_X(artifacts, monkeypatch,prediction_fault):
    from grasp_failure_prediction.evaluation.prediction_recording import AttemptMetadata, PredictionRecorder
    from grasp_failure_prediction.prediction.recorded_dataset import export_recorded_dataset, PrefixSpec
    from grasp_failure_prediction.prediction import recorded_training
    protocol = prepare(artifacts); output = artifacts["output"]; reserve.freeze(output, "protocol")
    proposals = proposal_rows(protocol); schedule = reserve.reserve_schedule(proposals)
    generation = dict(status="COMPLETED_BOUNDED_GENERATION", protocol_sha256=collector.sha(output / "protocol.json"),
                      proposals=proposals, schedule=schedule)
    write(output / "generation.json", generation)
    roots, attempts = [], []
    h = "sha256:" + "a" * 64
    for i, scenario in enumerate(schedule["scenarios"]):
        root = Path(protocol["raw_output"]) / "tasks" / scenario["attempt_id"]
        recorder = PredictionRecorder([f"j{j}" for j in range(24)], np.tile([-1.,1.], (24,1)))
        for t in range(1, 75):
            recorder.append(time_s=t*.04, action_interval_start_s=(t-1)*.04, phase="close_fingers",
                hand_qpos=np.full(24,.2), hand_qvel=np.full(24,.3), requested_target_rad=np.full(24,.4), applied_target_rad=np.full(24,.4))
        success = bool(i%2)
        recorder.write(root, AttemptMetadata(attempt_id=scenario["attempt_id"], hand_id="shadow_hand_right",
            execution_protocol_id="hug_grasp_approach_v1", object_id=scenario["object_id"], geometry_group_id=scenario["geometry_group_id"],
            observation_group_id=scenario["observation_group_id"], proposal_group_id=scenario["proposal_group_id"],
            mapping=dict(status="ok"), execution_status="completed",
            outcome=dict(success=success, failure_type=None if success else "failed_acquisition", label_rule_id=protocol["expected_cohort"]["label_rule_id"], first_failure_time_s=None),
            provenance_hashes={**{name:h for name in ("code","configuration","inputs","model","assets","native_pose","mapped_pose","label_rule")},
                               "scenario":collector.digest(scenario), "label_rule":protocol["expected_cohort"]["label_rule_sha256"]}))
        roots.append(root); attempts.append(dict(attempt_id=scenario["attempt_id"]))
    dataset = output / "reserve_prefixes"
    export_recorded_dataset(roots, dataset, prefix_spec=PrefixSpec(**protocol["prefix"]))
    write(output / "collection.json", dict(status="COMPLETED_BOUNDED_QUOTA", attempts=attempts,
        protocol_sha256=collector.sha(output / "protocol.json"), generation_sha256=collector.sha(output / "generation.json"),
        raw_artifact_bindings=reserve._files(Path(protocol["raw_output"])),
        dataset=dict(directory=str(dataset), bindings=reserve._files(dataset))))
    calls=[]
    def predict(directory, X):
        calls.append((directory, X.shape, X.tobytes()))
        if prediction_fault and len(calls)==2:raise RuntimeError("manufactured predictor fault")
        return np.full(len(X), .5)
    monkeypatch.setattr(recorded_training, "predict_saved_model", predict)
    report = reserve.evaluate(output)
    assert len(report["models"])==len(protocol["models"]["bundles"])==15
    if prediction_fault:
        assert report["status"]=="HARD_STOP" and len(calls)==2
        assert [row["status"] for row in report["models"]]==["COMPLETED","ERROR"]+["NOT_RUN"]*13
    else:
        assert report["status"] == "COMPLETED_FROZEN_MODEL_EVALUATION" and len(calls)==15
        assert all(row["metrics"]["episodes"] == 32 for row in report["models"])
    assert len({(shape, values) for _,shape,values in calls}) == 1
    assert report["counts"]["attempted"] == report["counts"]["included"] == 32
    with pytest.raises(FileExistsError): reserve.evaluate(output)
