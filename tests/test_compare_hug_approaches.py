"""Mocked paired-controller orchestration; real physics is always forbidden."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import compare_hug_approaches as paired


@pytest.fixture(autouse=True)
def block_physics(monkeypatch):
    def forbidden(*a,**k):pytest.fail("real physics forbidden in paired comparison tests")
    for name in ("mj_step","mj_step1","mj_step2"):monkeypatch.setattr(mujoco,name,forbidden)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    source=tmp_path/"collection";source.mkdir();inputs=tmp_path/"inputs";inputs.mkdir()
    code=tmp_path/"synthetic_code.py";code.write_text("# manufactured source\n")
    dummy=tmp_path/"synthetic_native_input.bin";dummy.write_bytes(b"synthetic only, never unpickled")
    protocol=dict(project=str(tmp_path),inputs_project=str(inputs),raw_output=str(inputs/"runs/fresh_v2"),
        source_bindings={str(code):paired.sha(code)},asset_bindings={str(dummy):paired.sha(dummy)},
        numerical_policy=dict(guard_every_physics_step=True,time_increment_atol_s=1e-12))
    paired.write(source/"preparation.json",protocol)
    proposals=[]
    for oi,object_id in enumerate(paired.collection.CATEGORIES):
        for ci in range(2):
            for seed in paired.collection.GENERATION_SEEDS:
                slot=f"proposal_{len(proposals):04d}"
                proposals.append(dict(slot_id=slot,object_id=object_id,capture_id=f"capture_{oi}_{ci}",generation_seed=seed,
                    mapping_status="ok",geometry_group_id=paired.digest(object_id),observation_group_id=paired.digest([object_id,ci]),
                    proposal_group_id=paired.digest(slot),native_sha256=paired.sha(dummy),proposal_hash=paired.digest(dict(slot=slot))))
    generated=dict(status="COMPLETED_BOUNDED_GENERATION",proposals=proposals,artifact_bindings={str(dummy):paired.sha(dummy)})
    paired.write(source/"generation.json",generated)
    monkeypatch.setattr(paired.collection,"read_protocol",lambda output:protocol)
    output=tmp_path/"evidence"
    value=paired.prepare(source,output);paired.freeze(output)
    auth=tmp_path/"authorization.json"
    paired.write(auth,dict(schema_version="hug_paired_approach_authorization_v1",authorized=True,
        protocol_sha256=paired.sha(output/"protocol.json"),quota=16))
    return SimpleNamespace(source=source,inputs=inputs,output=output,value=value,protocol=protocol,
        generated=generated,auth=auth,tmp=tmp_path)


def test_prepare_all_eight_pairs_before_outcomes_no_raw_side_effect(prepared):
    value=prepared.value;schedule=value["schedule"]
    assert len(schedule)==16 and len({r["pair_id"] for r in schedule})==8
    assert [r["condition"] for r in schedule]==list(paired.CONDITIONS)*8
    for i in range(0,16,2):
        a,b=schedule[i:i+2]
        assert a["native_sha256"]==b["native_sha256"] and a["canonical_proposal_hash"]==b["canonical_proposal_hash"]
        assert a["physical_outcome"] is None and b["physical_outcome"] is None
    assert not Path(value["raw_output"]).exists()
    assert value["quota"]["maximum_total_steps"]==52160


def test_prepare_refuses_missing_or_selected_mug_subset(prepared):
    proposals=prepared.generated["proposals"]
    with pytest.raises(ValueError,match="all eight"):paired.paired_schedule(proposals[:-1])
    changed=[dict(r) for r in proposals]
    changed[-1]["generation_seed"]=5000
    with pytest.raises(ValueError,match="complete"):paired.paired_schedule(changed)


def test_frozen_source_or_schedule_mutation_rejected(prepared):
    changed=json.loads(json.dumps(prepared.value));changed["schedule"]=changed["schedule"][:-1]
    with pytest.raises(ValueError,match="schedule"):paired.verify_protocol(changed)
    source=Path(next(iter(prepared.protocol["source_bindings"])))
    source.write_text("changed")
    with pytest.raises(ValueError,match="frozen source/input"):paired.verify_protocol(prepared.value)


def test_exact_authorization_required_before_any_attempt(prepared,monkeypatch):
    wrong=json.loads(prepared.auth.read_text());wrong["quota"]=15
    path=prepared.tmp/"wrong.json";paired.write(path,wrong)
    monkeypatch.setattr(paired,"_attempt",lambda *a:pytest.fail("unauthorized execution"))
    with pytest.raises(ValueError,match="authorization"):paired.run(prepared.output,path)
    assert not Path(prepared.value["raw_output"]).exists()


def test_error_keeps_all_remaining_rows_not_run_and_physical_labels_null(prepared,monkeypatch):
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    calls=[]
    def error(value,protocol,proposal,row,path):
        calls.append(row)
        result=paired._null(value,row,path,status="ERROR",mapping="setup_error",code="manufactured_fault")
        result["error"]=dict(type="RuntimeError",message="manufactured fault")
        return result
    monkeypatch.setattr(paired,"_attempt",error)
    result=paired.run(prepared.output,prepared.auth)
    assert result["status"]=="HARD_STOP" and len(calls)==1
    assert [r["status"] for r in result["schedule"]]==["ERROR"]+["NOT_RUN"]*15
    assert result["physics_steps"]==0 and result["analysis"]["paired_completed"]==0
    for row in result["schedule"]:
        metadata,_,arrays=load_attempt(row["recording_path"])
        assert metadata.outcome is None and metadata.execution_status=="not_executed" and arrays is None
        assert metadata.execution_protocol_id==paired.PROTOCOL_IDS[row["condition"]]
    with pytest.raises(FileExistsError):paired.run(prepared.output,prepared.auth)


def test_denominators_keep_planner_refusal_separate_from_physical_collision(prepared):
    rows=[dict(r,status="NOT_RUN",execution_status="not_executed") for r in prepared.value["schedule"]]
    rows[0].update(status="COMPLETED",execution_status="completed",physical_outcome=dict(success=False,failure_type="approach_collision"),score=dict(acquired_object=False))
    rows[1].update(status="PLANNER_REFUSED")
    rows[2].update(status="COMPLETED",execution_status="completed",physical_outcome=dict(success=True,failure_type=None),score=dict(acquired_object=True))
    rows[3].update(status="COMPLETED",execution_status="completed",physical_outcome=dict(success=True,failure_type=None),score=dict(acquired_object=True))
    result=paired.summarize(rows);a,b=(result["conditions"][c] for c in paired.CONDITIONS)
    assert a["planned"]==8 and a["completed"]==2 and a["approach_collision"]==1
    assert a["physical_collision_rate"]==dict(numerator=1,denominator=2,value=.5)
    assert b["planner_refused"]==1 and b["completed"]==1 and b["physical_success_rate"]["value"]==1.
    assert b["planned_population_success_rate"]==dict(numerator=1,denominator=8,value=.125)
    assert result["paired_completed"]==1 and result["pairs"][0]["success_difference"] is None


def test_finger_identity_allows_only_named_wrist_changes():
    pose=SimpleNamespace(joint_names=("WRJ1","WRJ2","FFJ1"),qpos=np.array([0.,0.,.2]))
    adjusted=SimpleNamespace(joint_names=pose.joint_names,qpos=np.array([.3,-.2,.2]))
    paired._finger_identity(pose,adjusted)
    adjusted.qpos[-1]+=.0001
    with pytest.raises(ValueError,match="finger"):paired._finger_identity(pose,adjusted)


@pytest.fixture
def fake_attempt(prepared,monkeypatch):
    from grasp_failure_prediction.evaluation.prediction_recording import PredictionRecorder
    from grasp_failure_prediction.evaluation.registry import load_protocol_registry,load_environment_registry
    from grasp_failure_prediction.evaluation.approach_execution import grasp_approach_protocol
    from grasp_failure_prediction.evaluation.runner import ExecutionTrace,RunnerStep,ExecutionState
    from grasp_failure_prediction.evaluation import recorded_runner,approach_execution
    import run_hug_saved_recording_pilot as pilot
    names=list(json.loads((Path(__file__).parent/"fixtures/provider_saved_cube.json").read_text())["joint_targets_rad"])
    base=load_protocol_registry().resolve("fixed_grasp_lift_v2")
    pose=SimpleNamespace(joint_names=tuple(names),qpos=np.full(24,.1))
    plan_record=dict(feasible=True,candidate="manufactured")
    plan=SimpleNamespace(feasible=True,adjusted_pose=pose,as_record=lambda:plan_record)
    budgets=[]
    @contextmanager
    def budget(maximum):
        b=dict(total=0);budgets.append(b);yield b
    class Runner:
        def __init__(self,protocol,*args,**kwargs):
            self.protocol=protocol;self.physics_timestep_s=.002;self.control_timestep_s=.04
            self.model=SimpleNamespace();self.data=SimpleNamespace();self.approach_plan=plan
            self.clearance_configuration=object();self.approach_configuration=object()
            self.prediction_recorder=PredictionRecorder(names,np.tile([-1.,1.],(24,1)))
        def execute(self,pose,**kwargs):
            steps=[]
            for i in range(163):
                phase=ExecutionState.CLOSE_FINGERS if i<74 else ExecutionState.HOLD
                time=round((i+1)*.04,9);budgets[-1]["total"]+=20
                self.prediction_recorder.append(time_s=time,action_interval_start_s=round(i*.04,9),phase=phase.value,
                    hand_qpos=np.full(24,.1),hand_qvel=np.zeros(24),requested_target_rad=np.full(24,2.),applied_target_rad=np.ones(24),diagnostics={})
                steps.append(RunnerStep(i,time,phase,np.zeros(3),np.full(24,.1),np.array([0.,0.,.03]),1,True,False,True))
            return ExecutionTrace(tuple(steps))
    environment=load_environment_registry().resolve("adroit_shadow_tabletop_v2")
    context=SimpleNamespace(resolved=SimpleNamespace(environment=environment,execution_protocol=base),geometry=None,verify_bindings=lambda:None)
    monkeypatch.setattr(paired.collection,"_case_and_context",lambda *a:(context,pose,np.array([0.,.1,.05]),np.array([1.,0.,0.,0.])))
    monkeypatch.setattr(paired.collection,"_new_runner",lambda *a:Runner(grasp_approach_protocol(base)))
    monkeypatch.setattr(recorded_runner,"RecordedActuatedShadowRunner",Runner)
    monkeypatch.setattr(paired,"_nominal_reset",lambda *a:dict(mass=.18,friction="unchanged"))
    monkeypatch.setattr(approach_execution,"plan_grasp_approach",lambda *a,**k:plan)
    monkeypatch.setattr(paired.collection,"physics_budget",budget)
    monkeypatch.setattr(mujoco,"mj_saveModel",lambda model,path:Path(path).write_bytes(b"synthetic model"))
    monkeypatch.setattr(pilot,"_save_partial",lambda path,recorder,data:paired.write(path/"partial.json",dict(samples=recorder.sample_count)))
    return SimpleNamespace(prepared=prepared,Runner=Runner,budgets=budgets,plan=plan)


def test_paired_mocked_run_records16_separate_protocol_outcomes(fake_attempt):
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    f=fake_attempt;p=f.prepared
    result=paired.run(p.output,p.auth)
    assert result["status"]=="COMPLETED_PAIRED_QUOTA" and result["physics_steps"]==52160
    assert result["analysis"]["paired_completed"]==8
    for condition in paired.CONDITIONS:
        count=result["analysis"]["conditions"][condition]
        assert count["planned"]==8 and count["mapped"]==8 and count["completed"]==8
    for row in result["schedule"]:
        metadata,_,arrays=load_attempt(row["recording_path"])
        assert metadata.execution_protocol_id==paired.PROTOCOL_IDS[row["condition"]]
        assert metadata.outcome is not None and arrays["hand_qvel"].shape==(163,24)


def test_preplanned_refusal_executes_no_new_condition_and_no_fake_failure(fake_attempt):
    f=fake_attempt;f.plan.feasible=False
    result=paired.run(f.prepared.output,f.prepared.auth)
    assert result["status"]=="COMPLETED_PAIRED_QUOTA" and result["physics_steps"]==26080
    new=result["analysis"]["conditions"]["grasp_derived_v1"]
    assert new["planner_refused"]==8 and new["completed"]==0 and new["execution_attempted"]==0
    assert new["physical_success_rate"]["value"] is None
    assert result["analysis"]["paired_completed"]==0


def test_late_comparison_publication_failure_withdraws_completed_marker(fake_attempt,monkeypatch):
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    f=fake_attempt;p=f.prepared;original=paired.write
    def fail_after_publish(path,value):
        original(path,value)
        if Path(path).name=="result.json":raise OSError("manufactured late publication failure")
    monkeypatch.setattr(paired,"write",fail_after_publish)
    result=paired.run(p.output,p.auth)
    assert result["status"]=="HARD_STOP" and result["physics_steps"]==3260
    assert [r["status"] for r in result["schedule"]]==["ERROR"]+["NOT_RUN"]*15
    row=result["schedule"][0];path=Path(row["recording_path"]).parent
    assert not (path/"result.json").exists() and (path/"quarantined_result.json").exists()
    metadata,_,_=load_attempt(row["recording_path"])
    assert metadata.execution_status=="runtime_error" and metadata.outcome is None
    assert list(path.rglob("attempt.json"))==[path/"prediction_recording/attempt.json"]
