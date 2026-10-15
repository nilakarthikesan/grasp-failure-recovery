"""Manufactured orchestration tests: all real physics APIs are blocked."""
import importlib.util
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


SPEC=importlib.util.spec_from_file_location("hug_build_collection",Path(__file__).parents[1]/"scripts/run_hug_build_collection.py")
driver=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(driver)


@pytest.fixture(autouse=True)
def block_physics(monkeypatch):
    def blocked(*args,**kwargs):pytest.fail("real physics forbidden in collection-driver tests")
    for name in ("mj_step","mj_step1","mj_step2"):monkeypatch.setattr(mujoco,name,blocked)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    source=tmp_path/"source";source.mkdir();inputs=tmp_path/"inputs";inputs.mkdir()
    output=tmp_path/"evidence";hug=tmp_path/"hug"
    dep=tmp_path/"deps";dep.mkdir()
    declaration=source/"research_artifacts/hug_native_mano_comparison_2026-10-15_v1/measurement/protocol.json"
    driver.write(declaration,{"dependency_dirs":[str(dep)]})
    checkpoint=tmp_path/"checkpoint.bin";checkpoint.write_bytes(b"synthetic checkpoint not executed")
    for name in ("src/models/mano.py","assets/mano_models/models/MANO_RIGHT.pkl"):
        path=hug/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"synthetic-only")
    urdf=source/"third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf"
    urdf.parent.mkdir(parents=True);urdf.write_text('<robot name="synthetic"/>')
    code=source/"code.py";code.write_text("# synthetic source binding\n")
    monkeypatch.setattr(driver,"SOURCE_FILES",("code.py",))
    entries=[]
    for index in range(32):
        object_id=list(driver.CATEGORIES)[index//8]
        case=dict(schema_version="eval_case_v1",case_id=f"old_{index}",environment=dict(id="adroit_shadow_tabletop_v2"),
            execution_protocol=dict(id="fixed_grasp_lift_v2"),embodiment=dict(id="shadow_hand_right"),
            object=dict(id=object_id,mass_kg=.18,friction_profile_id="nominal_v1",geometry=None),
            initial_condition=dict(id="old_capture",object_position_m=[0,0,.03],object_orientation_xyzw=[0,0,0,1]),
            grasp=dict(source="hug",id="old_grasp",prediction_path="old/proposal.pkl",observation_path="old/capture",inference_seed=0),
            motion_profile_id="nominal_lift_v1",seed=42)
        path=inputs/f"old_{index}.json";driver.write(path,{"case":case})
        entries.append(dict(object_id=object_id,resolved_case_path=str(path)))
    inventory=tmp_path/"inventory.json";driver.write(inventory,{"modern_cases":entries})
    protocol=driver.prepare(source,inputs,output,inventory,hug_root=hug,checkpoint=checkpoint)
    driver.freeze(output,"preparation")
    auth=tmp_path/"generate_authorization.json"
    driver.write(auth,dict(schema_version="hug_build_authorization_v1",stage="generate",authorized=True,
        bindings={str(output/"preparation.json"):driver.sha(output/"preparation.json")},quota=32))
    # Runtime task imports must use the real project while input bindings stay synthetic.
    monkeypatch.setattr(driver,"bootstrap",lambda protocol:None)
    return SimpleNamespace(source=source,inputs=inputs,output=output,protocol=protocol,auth=auth,tmp=tmp_path)


def test_prepare_fixed_quota_new_inputs_and_no_raw_side_effects(prepared):
    p=prepared.protocol
    assert len(p["generation_slots"])==32
    assert len({r["capture_id"] for r in p["generation_slots"]})==8
    assert {r["generation_seed"] for r in p["generation_slots"]}=={3000,3001,3002,3003}
    assert p["quotas"]["pilot_attempts"]==20 and p["quotas"]["study_attempts"]==128
    assert p["study_approach_modes"]==["grasp_derived"]
    assert not Path(p["raw_output"]).exists()
    assert not (prepared.output/"generation.json").exists()
    assert not any("stored_shadow_command" in r for r in p["generation_slots"])


def test_preparation_is_immutable_and_detects_bound_source_mutation(prepared):
    with pytest.raises(FileExistsError):driver.freeze(prepared.output,"preparation")
    (prepared.source/"code.py").write_text("changed")
    with pytest.raises(ValueError,match="frozen source/input changed"):driver.read_protocol(prepared.output)


def test_explicit_authorization_requires_exact_stage_quota_and_binding(prepared):
    for field,value in (("quota",31),("stage","collect_pilot"),("authorized",False),("bindings",{})):
        auth=json.loads(prepared.auth.read_text());auth[field]=value
        path=prepared.tmp/(field+"_bad.json");driver.write(path,auth)
        with pytest.raises(ValueError,match="authorization"):
            driver.authorized(prepared.output,path,"generate",quota=32)


def _install_fake_generation(prepared,monkeypatch,*,refusal_seed=None,error_seed=None):
    from grasp_failure_prediction.integrations import sim_observations
    from grasp_failure_prediction.proposals import generate
    calls=dict(capture=[],generation=[])
    def capture(case_path,output_dir,**kwargs):
        case=json.loads(Path(case_path).read_text());out=Path(output_dir)
        out.mkdir(parents=True)
        driver.write(out/"scene_contract.json",dict(object_id=case["object"]["id"],geometry_content_hash=driver.digest(case["object"]["id"]),
            mass_kg=.18,object_position_m=[0,0,.03],object_orientation_xyzw=[0,0,0,1]))
        driver.write(out/"observation_manifest.json",dict(capture=case["case_id"]))
        calls["capture"].append((case,kwargs))
    def infer(request_path,output_dir,**kwargs):
        assert kwargs["runtime_dependency_dirs"]==prepared.protocol["dependency_dirs"]
        request=json.loads(Path(request_path).read_text());calls["generation"].append(request)
        out=Path(output_dir);out.mkdir(parents=True)
        driver.write(out/"sample_0000/native/inference_report.json",dict(loaded_model_state_sha256="sha256:"+"a"*64))
        if request["seed"] in (refusal_seed,error_seed):
            native=out/"sample_0000/native"
            (native/"proposal.pkl").write_bytes(b"synthetic-native")
            if request["seed"]==refusal_seed:
                driver.write(native/"mapping_record.json",dict(mapping_status="refused_guard",outcome=None))
            raise ValueError("manufactured generation failure")
        path=out/"sample_0000/proposal.json";driver.write(path,{"slot":request["request_id"]})
        return SimpleNamespace(proposals=[SimpleNamespace(proposal_path=Path("sample_0000/proposal.json"))])
    def proposal(protocol,slot,path,validation):
        binding=generate._expected_input_binding(generate.ProposalRequest.model_validate(calls["generation"][-1]),prepared.inputs)
        return dict(**slot,status="MAPPED",mapping_status="ok",outcome=None,proposal_path=str(path),
            geometry_group_id=binding["geometry_group_id"],observation_group_id=binding["observation_group_id"],
            proposal_group_id=driver.digest(slot),proposal_hash=driver.digest(dict(proposal=slot)),native_sha256="sha256:"+"a"*64,mapping_validation=validation)
    monkeypatch.setattr(sim_observations,"capture_scene",capture)
    monkeypatch.setattr(generate,"generate_proposals",infer)
    monkeypatch.setattr(driver,"_validate_fresh",lambda *a:dict(static_input_gate="PASS",physics_steps=0))
    monkeypatch.setattr(driver,"_proposal_row",proposal)
    return calls


def test_generate_uses_eight_new_captures_and32fresh_seeds_and_freezes_folds(prepared,monkeypatch):
    calls=_install_fake_generation(prepared,monkeypatch)
    report=driver.generate(prepared.output,prepared.auth)
    assert report["status"]=="COMPLETED_BOUNDED_GENERATION"
    assert len(calls["capture"])==8 and len(calls["generation"])==32
    assert report["capture_physics_steps"]==0
    assert all(r["outcome"] is None for r in report["proposals"])
    for condition,count in (("pilot",20),("study",128)):
        study=report["conditions"][condition]
        assert len(study["schedule"]["scenarios"])==count
        assert len(study["attempt_binding"]["attempt_to_scenario"])==count
        assert all(row["approach_mode"]=="grasp_derived" for row in study["schedule"]["scenarios"])
        assert study["folds"]["frozen_before_outcomes"] is True
    driver.freeze(prepared.output,"generation")
    with pytest.raises(FileExistsError):driver.generate(prepared.output,prepared.auth)


def test_mapping_refusals_retained_with_null_outcome_and_no_replacement(prepared,monkeypatch):
    calls=_install_fake_generation(prepared,monkeypatch,refusal_seed=3001)
    report=driver.generate(prepared.output,prepared.auth)
    refused=[r for r in report["proposals"] if r["status"]=="MAPPING_REFUSED"]
    assert len(refused)==8 and len(report["proposals"])==32
    assert len(calls["generation"])==32
    assert report["status"]=="COMPLETED_BOUNDED_GENERATION"
    assert all(r["outcome"] is None and r["mapping_status"]=="refused_guard" for r in refused)
    slots={r["generation_slot_id"] for r in report["conditions"]["study"]["schedule"]["scenarios"]}
    assert slots & {r["slot_id"] for r in refused}


def test_unexpected_generation_error_hardstops_and_remaining_not_run(prepared,monkeypatch):
    calls=_install_fake_generation(prepared,monkeypatch,error_seed=3001)
    report=driver.generate(prepared.output,prepared.auth)
    assert report["status"]=="HARD_STOP" and len(calls["generation"])==2
    assert [r["status"] for r in report["proposals"]]==["MAPPED","ERROR"]+["NOT_RUN"]*30
    assert "conditions" not in report
    assert list(Path(prepared.protocol["raw_output"]).rglob("unexpected_error_traceback.txt"))
    with pytest.raises(ValueError,match="incomplete/error"):driver.freeze(prepared.output,"generation")


def test_numerical_budget_restores_apis_and_rejects_split_calls():
    original=mujoco.mj_step
    with driver.physics_budget(0) as budget:
        with pytest.raises(RuntimeError,match="split-step"):mujoco.mj_step1(None,None)
        with pytest.raises(RuntimeError,match="budget"):mujoco.mj_step(None,None)
    assert budget["total"]==0 and mujoco.mj_step is original


def test_loaded_state_mismatch_hardstops_instead_of_pooling_different_models(prepared,monkeypatch):
    calls=_install_fake_generation(prepared,monkeypatch)
    counter=[]
    def state(path):
        counter.append(path)
        return "sha256:"+("a" if len(counter)==1 else "b")*64
    monkeypatch.setattr(driver,"_loaded_state_hash",state)
    report=driver.generate(prepared.output,prepared.auth)
    assert report["status"]=="HARD_STOP" and len(calls["generation"])==2
    assert report["proposals"][1]["status"]=="ERROR"
    assert all(row["status"]=="NOT_RUN" for row in report["proposals"][2:])


@pytest.fixture
def generated(prepared,monkeypatch):
    _install_fake_generation(prepared,monkeypatch)
    report=driver.generate(prepared.output,prepared.auth)
    driver.freeze(prepared.output,"generation")
    prepared.generated=report
    auth=prepared.tmp/"pilot_authorization.json"
    driver.write(auth,dict(schema_version="hug_build_authorization_v1",stage="collect_pilot",authorized=True,
        bindings={str(prepared.output/"preparation.json"):driver.sha(prepared.output/"preparation.json"),
                  str(prepared.output/"generation.json"):driver.sha(prepared.output/"generation.json")},quota=20))
    prepared.pilot_auth=auth
    # These manufactured controller/recording tests exercise post-admission
    # behavior. The real fail-closed boundary is tested without this mock in
    # test_hug_fidelity_admission.py; synthetic rows are not fidelity evidence.
    monkeypatch.setattr(driver,"_require_collection_fidelity",lambda generated:None)
    return prepared


def test_null_refused_and_not_run_rows_preserve_schema_and_no_physical_labels(generated):
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    scenario=generated.generated["conditions"]["pilot"]["schedule"]["scenarios"][0]
    proposal=next(p for p in generated.generated["proposals"] if p["slot_id"]==scenario["generation_slot_id"])
    for i,mapping in enumerate(("refused_guard","unreachable_pregrasp","setup_error")):
        path=generated.tmp/f"null_{i}"
        row=driver._null_attempt(generated.protocol,proposal,scenario,path,mapping=mapping,code="unit_refusal",status="NOT_RUN")
        metadata,_,arrays=load_attempt(row["recording_path"])
        assert metadata.outcome is None and metadata.execution_status=="not_executed" and arrays is None
        assert metadata.mapping.code=="unit_refusal" and metadata.mapping.message


def test_filter_error_writes20not_run_rows_without_task_execution(generated,monkeypatch):
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    monkeypatch.setattr(driver,"_run_filters",lambda *a:dict(error=dict(type="RuntimeError",message="manufactured filter fault"),
        proposals={},physics_steps=1,artifact_bindings={}))
    monkeypatch.setattr(driver,"_collect_attempt",lambda *a:pytest.fail("task ran after filter hardstop"))
    result=driver.collect(generated.output,generated.pilot_auth,"pilot")
    assert result["status"]=="HARD_STOP" and result["counts"]=={"NOT_RUN":20}
    assert len(result["attempts"])==20 and result["physics_steps"]==0
    assert result["dataset"]["counts"]["included"]==0
    for row in result["attempts"]:
        metadata,_,_=load_attempt(row["recording_path"])
        assert metadata.outcome is None and metadata.mapping.status.value=="setup_error"


def test_study_requires_completed_pilot_and_separate_authorization(generated):
    with pytest.raises(ValueError,match="bounded pilot"):
        driver.collect(generated.output,generated.pilot_auth,"study")


@pytest.fixture
def fake_task(generated,monkeypatch):
    from grasp_failure_prediction.evaluation.prediction_recording import PredictionRecorder
    from grasp_failure_prediction.evaluation.registry import load_protocol_registry
    from grasp_failure_prediction.evaluation.approach_execution import grasp_approach_protocol
    from grasp_failure_prediction.evaluation.runner import ExecutionTrace,RunnerStep,ExecutionState
    import run_hug_saved_recording_pilot as pilot
    names=list(json.loads((Path(__file__).parent/"fixtures/provider_saved_cube.json").read_text())["joint_targets_rad"])
    budgets=[]
    @contextmanager
    def budget(maximum):
        b=dict(total=0,this_attempt=0);budgets.append(b)
        yield b
    class Runner:
        def __init__(self):
            self.protocol=grasp_approach_protocol(load_protocol_registry().resolve("fixed_grasp_lift_v2"))
            self.control_timestep_s=.04;self.physics_timestep_s=.002
            self.model=SimpleNamespace();self.data=SimpleNamespace(time=0.,qpos=np.zeros(38),qvel=np.zeros(36),qacc=np.zeros(36),ctrl=np.zeros(24))
            self.approach_plan=SimpleNamespace(as_record=lambda:dict(feasible=True,selected_candidate_id="synthetic"))
            self.prediction_recorder=PredictionRecorder(names,np.tile([-1.,1.],(24,1)))
        def execute(self,pose,**kwargs):
            steps=[]
            for i in range(163):
                phase=ExecutionState.CLOSE_FINGERS if i<74 else ExecutionState.HOLD
                time=round((i+1)*.04,9);self.data.time=time;budgets[-1]["total"]+=20
                self.prediction_recorder.append(time_s=time,action_interval_start_s=round(i*.04,9),phase=phase.value,
                    hand_qpos=np.full(24,.2),hand_qvel=np.full(24,.3),requested_target_rad=np.full(24,2.),applied_target_rad=np.ones(24),
                    diagnostics=dict(full_qpos=np.zeros(38),full_qvel=np.zeros(36)))
                step=RunnerStep(i,time,phase,np.zeros(3),np.full(24,.2),np.array([0,0,.03]),1,True,False,True)
                steps.append(step)
                if kwargs.get("step_callback"):kwargs["step_callback"](step)
            return ExecutionTrace(tuple(steps))
    runner=Runner()
    context=SimpleNamespace(verify_bindings=lambda:None)
    monkeypatch.setattr(driver,"_case_and_context",lambda *a:(context,object(),np.zeros(3),np.array([1,0,0,0])))
    monkeypatch.setattr(driver,"_new_runner",lambda *a:runner)
    monkeypatch.setattr(driver,"_apply_scenario",lambda r,c,s,p:(p,dict(manufactured=True,scenario=s)))
    monkeypatch.setattr(driver,"physics_budget",budget)
    monkeypatch.setattr(mujoco,"mj_saveModel",lambda model,path:Path(path).write_bytes(b"synthetic-compiled-model"))
    monkeypatch.setattr(pilot,"_save_partial",lambda directory,recorder,data:driver.write(directory/"partial_saved.json",dict(samples=recorder.sample_count)))
    scenario=generated.generated["conditions"]["pilot"]["schedule"]["scenarios"][0]
    proposal=next(p for p in generated.generated["proposals"] if p["slot_id"]==scenario["generation_slot_id"])
    return SimpleNamespace(root=generated,runner=runner,runner_type=Runner,scenario=scenario,proposal=proposal,budgets=budgets)


def test_direct_task_records_scenario_hashes_and_clipped_commands(fake_task):
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    f=fake_task;path=f.root.tmp/"task"
    row=driver._collect_attempt(f.root.protocol,f.proposal,f.scenario,path,dict(filter_status="rejected"))
    assert row["status"]=="COMPLETED" and row["physics_steps"]==3260 and row["filter_status"]=="rejected"
    meta,_,arrays=load_attempt(row["recording_path"])
    assert meta.execution_protocol_id=="hug_grasp_approach_v1" and meta.outcome.success is False
    assert meta.provenance_hashes["scenario"]==driver.digest(f.scenario)
    assert {"scenario_realization","model","approach_plan","label_rule"}<=set(meta.provenance_hashes)
    np.testing.assert_array_equal(arrays["applied_target_rad"],1.)
    assert (path/"result.json").exists()


def test_task_error_preserves_partial_null_label_and_stops_next_rows(fake_task,monkeypatch):
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    f=fake_task
    def fail(*args,**kwargs):
        f.budgets[-1]["total"]=1
        raise RuntimeError("manufactured first-step fault")
    monkeypatch.setattr(f.runner,"execute",fail)
    path=f.root.tmp/"failed_task"
    row=driver._collect_attempt(f.root.protocol,f.proposal,f.scenario,path,{})
    meta,_,arrays=load_attempt(row["recording_path"])
    assert row["status"]=="ERROR" and row["physics_steps"]==1
    assert meta.execution_status=="runtime_error" and meta.mapping.status.value=="ok" and meta.outcome is None and arrays is None
    assert (path/"partial_saved.json").exists() and (path/"exception_traceback.txt").exists()


def test_noisy_qpos_is_observation_only_and_deterministic():
    from grasp_failure_prediction.evaluation.prediction_recording import PredictionRecorder
    def record(seed,std):
        clean=PredictionRecorder(["a"],np.array([[-1.,1.]]));wrapper=driver._NoisyObservations(clean,std,seed)
        wrapper.append(time_s=.04,action_interval_start_s=0.,phase="close_fingers",hand_qpos=np.array([.2]),hand_qvel=np.array([.3]),
            requested_target_rad=np.array([2.]),applied_target_rad=np.array([1.]),diagnostics={})
        return wrapper
    a,b=record(10,.002),record(10,.002)
    np.testing.assert_array_equal(a.arrays()["hand_qpos"],b.arrays()["hand_qpos"])
    assert a.arrays()["hand_qpos"][0,0]!=.2
    np.testing.assert_array_equal(a.arrays()["applied_target_rad"],1.)
    np.testing.assert_array_equal(a.arrays()["hand_qvel"],.3)
    np.testing.assert_array_equal(a._diagnostics[0]["clean_hand_qpos_rad"],.2)
    np.testing.assert_array_equal(record(10,0.).arrays()["hand_qpos"],.2)


def test_setup_error_before_runner_is_null_and_unexecuted(fake_task,monkeypatch):
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    f=fake_task
    def fail(*a):raise ValueError("manufactured nominal capture binding failure")
    monkeypatch.setattr(driver,"_case_and_context",fail)
    row=driver._collect_attempt(f.root.protocol,f.proposal,f.scenario,f.root.tmp/"setup_fault",{})
    meta,_,_=load_attempt(row["recording_path"])
    assert row["status"]=="ERROR" and row["physics_steps"]==0
    assert meta.mapping.status.value=="setup_error" and meta.mapping.message
    assert meta.execution_status=="not_executed" and meta.outcome is None


def test_unreachable_pregrasp_is_expected_refusal_not_physical_failure(fake_task,monkeypatch):
    from grasp_failure_prediction.evaluation.approach_execution import ApproachPlanningRejected
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    f=fake_task
    plan=SimpleNamespace(rejection_reason="manufactured static refusal",as_record=lambda:dict(feasible=False))
    f.runner.approach_plan=plan
    def refused(*a,**k):raise ApproachPlanningRejected(plan)
    monkeypatch.setattr(f.runner,"execute",refused)
    row=driver._collect_attempt(f.root.protocol,f.proposal,f.scenario,f.root.tmp/"approach_refusal",{})
    meta,_,arrays=load_attempt(row["recording_path"])
    assert row["status"]=="APPROACH_REFUSED" and row["physics_steps"]==0
    assert meta.mapping.status.value=="unreachable_pregrasp" and meta.outcome is None and arrays is None


def test_first_task_runtime_fault_preserves_remaining19not_run(generated,monkeypatch):
    monkeypatch.setattr(driver,"_run_filters",lambda *a:dict(error=None,proposals={},physics_steps=0,artifact_bindings={}))
    calls=[]
    def fault(protocol,proposal,scenario,directory,filter_record):
        calls.append(scenario)
        row=driver._null_attempt(protocol,proposal,scenario,directory,mapping="setup_error",code="manufactured_fault",status="ERROR")
        row["error"]=dict(type="RuntimeError",message="manufactured task fault")
        return row
    monkeypatch.setattr(driver,"_collect_attempt",fault)
    report=driver.collect(generated.output,generated.pilot_auth,"pilot")
    assert report["status"]=="HARD_STOP" and len(calls)==1
    assert report["counts"]=={"ERROR":1,"NOT_RUN":19}
    assert all(row["outcome"] is None for row in report["attempts"])


def test_filter_close_only_stops_before_lift_and_keeps_refusals(fake_task,monkeypatch):
    from grasp_failure_prediction.proposals import physics_filter
    f=fake_task
    proposals=[dict(row) for row in f.root.generated["proposals"]]
    for row in proposals:
        if row["slot_id"]!=f.proposal["slot_id"]:row["mapping_status"]="refused_guard"
    monkeypatch.setattr(physics_filter.PhysicsFilterGeometry,"from_shadow_model",lambda model:SimpleNamespace())
    states=[]
    def filter_state(model,data,geometry,**kwargs):
        states.append(data.time)
        return SimpleNamespace(filter_status="rejected",as_record=lambda:dict(filter_status="rejected",task_outcome_label=None,predictor_feature_eligible=False))
    monkeypatch.setattr(physics_filter,"filter_closed_grasp",filter_state)
    report=driver._run_filters(f.root.protocol,dict(proposals=proposals),f.root.tmp/"filters")
    assert states==[2.96] and report["physics_steps"]==1480
    assert report["error"] is None
    assert report["proposals"][f.proposal["slot_id"]]["filter_status"]=="rejected"
    assert sum(row["filter_status"]=="NOT_RUN" for row in report["proposals"].values())==31
    assert all(row.get("task_outcome_label") is None for row in report["proposals"].values())


def test_scenario_mutations_bind_mass_inertia_both_contact_frictions_and_translation(monkeypatch):
    from run_hug_saved_recording_pilot import expected_object_inertia
    mass=.18;_,tensor=expected_object_inertia(None,mass)
    model=SimpleNamespace(ngeom=3,geom_bodyid=np.array([0,1,2]),geom_contype=np.ones(3,int),geom_conaffinity=np.ones(3,int),
        pair_geom1=np.array([],int),pair_geom2=np.array([],int),geom_friction=np.array([[1.,.005,.0001],[1.,.01,.001],[1.,.02,.002]]),
        geom_pos=np.array([[0.,0.,-.025],[0.,0.,0.],[0.,0.,0.]]),body_mass=np.array([0.,mass]),
        body_inertia=np.vstack([np.zeros(3),np.diag(tensor)]),body_ipos=np.zeros((2,3)),body_iquat=np.array([[1.,0,0,0],[1.,0,0,0]]))
    data=SimpleNamespace(qpos=np.zeros(7),geom_xpos=model.geom_pos.copy())
    class Runner:
        _table_geom_id=0;_object_body_id=1;_object_qpos_address=0;_hand_body_ids={2};object_collision_geom_ids={1};physics_timestep_s=.002
        def __init__(self):self.model=model;self.data=data
        def reset(self,**values):
            model.body_mass[1]=values["object_mass_kg"]
            model.body_inertia[1]=np.diag(tensor)*(values["object_mass_kg"]/mass)
            data.qpos[:3]=values["object_position_m"];data.qpos[3:]=values["object_orientation_wxyz"]
    context=SimpleNamespace(case=SimpleNamespace(object=SimpleNamespace(mass_kg=mass),
        initial_condition=SimpleNamespace(object_position_m=[0.,0.,.03],object_orientation_xyzw=[0.,0.,0.,1.])),
        geometry=None,scene_binding={"scene_contract_sha256":"sha256:"+"1"*64})
    monkeypatch.setattr(mujoco,"mj_forward",lambda m,d:d.geom_xpos.__setitem__(slice(None),m.geom_pos))
    original_secondary=model.geom_friction[:,1:].copy()
    palm,report=driver._apply_scenario(Runner(),context,dict(table_offset_m=.02,mass_multiplier=1.25,
        friction_coefficient=.5,sensor_noise_seed=10),np.array([0.,.1,.05]))
    assert report["object_mass_kg"]==pytest.approx(.225)
    np.testing.assert_allclose(palm,[0.,.1,.07])
    np.testing.assert_allclose(data.qpos[:3],[0.,0.,.05])
    assert model.geom_pos[0,2]==pytest.approx(-.005)
    np.testing.assert_array_equal(model.geom_friction[[1,2],0],[.5,.5])
    assert model.geom_friction[0,0]==1.
    np.testing.assert_array_equal(model.geom_friction[:,1:],original_secondary)


def test_full_manufactured20pilot_exports_and_loads_frozen_training_adapter(fake_task,monkeypatch):
    from grasp_failure_prediction.prediction.recorded_training import load_training_data
    f=fake_task
    monkeypatch.setattr(driver,"_new_runner",lambda *a:f.runner_type())
    filter_rows={p["slot_id"]:dict(filter_status="accepted" if i%2 else "rejected")
                 for i,p in enumerate(f.root.generated["proposals"])}
    monkeypatch.setattr(driver,"_run_filters",lambda *a:dict(error=None,proposals=filter_rows,physics_steps=0,artifact_bindings={}))
    report=driver.collect(f.root.output,f.root.pilot_auth,"pilot")
    assert report["status"]=="COMPLETED_BOUNDED_QUOTA" and report["counts"]=={"COMPLETED":20}
    assert report["physics_steps"]==65200 and report["dataset"]["counts"]["included"]==20
    assert {row["filter_status"] for row in report["attempts"]}=={"accepted","rejected"}
    data=load_training_data(f.root.output/"pilot_prefixes",f.root.output/"pilot_schedule.json",
        f.root.output/"pilot_folds.json",f.root.output/"pilot_attempt_scenario_binding.json")
    assert data.dataset.X.shape==(20,1440) and data.dataset.y.sum()==0
    assert len(data.folds)==4
    assert "filter_status" not in data.dataset.manifest["input_contract"]["fields"]
    with pytest.raises(FileExistsError):driver.collect(f.root.output,f.root.pilot_auth,"pilot")


@pytest.mark.parametrize("corrupt",["joint_targets","object_palm","world_palm"])
def test_canonical_command_must_match_fresh_mapping_record(corrupt):
    from grasp_failure_prediction.proposals.execution import _matrix
    pose=SimpleNamespace(translation_m=[.01,.02,.03],quaternion_xyzw=[0.,0.,0.,1.])
    proposal=SimpleNamespace(joint_targets_rad={"a":.1,"b":.2},object_to_palm=pose)
    capture=dict(object_position_m=[.1,0.,.03],object_orientation_xyzw=[0.,0.,0.,1.])
    obj=_matrix(pose.translation_m,pose.quaternion_xyzw)
    world=_matrix(capture["object_position_m"],capture["object_orientation_xyzw"])@obj
    mapping=dict(joint_names=["b","a"],qpos_rad=[.2,.1],object_to_palm_transform=obj.tolist(),world_palm_transform=world.tolist())
    names,q,actual=driver._canonical_mapping_binding(proposal,mapping,capture)
    assert names==("b","a");np.testing.assert_array_equal(q,[.2,.1]);np.testing.assert_array_equal(actual,world)
    if corrupt=="joint_targets":proposal.joint_targets_rad["a"]+=1e-12
    elif corrupt=="object_palm":mapping["object_to_palm_transform"][0][3]+=.001
    else:mapping["world_palm_transform"][0][3]+=.001
    with pytest.raises(ValueError,match="canonical"):
        driver._canonical_mapping_binding(proposal,mapping,capture)


def test_late_result_publication_error_withdraws_completed_marker(fake_task,monkeypatch):
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    f=fake_task;original=driver.write
    def fail_after_write(path,value):
        original(path,value)
        if Path(path).name=="result.json":raise OSError("manufactured late result flush failure")
    monkeypatch.setattr(driver,"write",fail_after_write)
    path=f.root.tmp/"late_fault"
    row=driver._collect_attempt(f.root.protocol,f.proposal,f.scenario,path,{})
    assert row["status"]=="ERROR" and row["outcome"] is None
    assert not (path/"result.json").exists() and not (path/"task_score.json").exists()
    assert (path/"quarantined_result.json").exists()
    meta,_,_=load_attempt(row["recording_path"])
    assert meta.execution_status=="runtime_error" and meta.outcome is None
    assert list(path.rglob("attempt.json"))==[path/"prediction_recording/attempt.json"]
