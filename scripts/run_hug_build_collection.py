#!/usr/bin/env python3
"""Prepare, generate and collect a finite NEW HUG study; never rewrite history.

Prepare only reads/hashes files. Generation requires a frozen preparation and
authorization; it captures new images, performs fresh HUG inference, validates
each returned mapping statically, then writes a separate pre-outcome schedule.
Collection needs another frozen authorization. No command infers permission.
Private native payloads, scenes and full-state traces stay under ignored runs/.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from xml.etree import ElementTree as ET

# Preserve the base numerical runtime before appending existing external deps.
import numpy as np
import mujoco


SCHEMA = "hug_build_collection_v1"
SOURCE_INDICES = (0, 8, 16, 24)
GENERATION_SEEDS = (3000, 3001, 3002, 3003)
CAMERAS = ((.25, -.35, .35), (-.25, -.35, .35))
CATEGORIES = {"object01": "cube", "hug_apple": "apple", "hug_milk_box": "milk_carton", "hug_mug": "mug"}
SOURCE_FILES = (
    "scripts/run_hug_build_collection.py", "scripts/run_hug_saved_recording_pilot.py",
    "scripts/infer_sim_observation.py",
    "src/grasp_failure_prediction/integrations/sim_observations.py",
    "src/grasp_failure_prediction/integrations/observations.py",
    "src/grasp_failure_prediction/integrations/hug.py",
    "src/grasp_failure_prediction/integrations/hug_frames.py",
    "src/grasp_failure_prediction/integrations/hug_projection.py",
    "src/grasp_failure_prediction/integrations/hug_preprocessing.py",
    "src/grasp_failure_prediction/integrations/hug_rng.py",
    "src/grasp_failure_prediction/evaluation/runner.py",
    "src/grasp_failure_prediction/evaluation/recorded_runner.py",
    "src/grasp_failure_prediction/evaluation/prediction_recording.py",
    "src/grasp_failure_prediction/evaluation/approach_execution.py",
    "src/grasp_failure_prediction/evaluation/table_clearance.py",
    "src/grasp_failure_prediction/evaluation/mapping_validation.py",
    "src/grasp_failure_prediction/evaluation/pose_validation.py",
    "src/grasp_failure_prediction/evaluation/retargeting.py",
    "src/grasp_failure_prediction/evaluation/scoring.py",
    "src/grasp_failure_prediction/evaluation/scene_contracts.py",
    "src/grasp_failure_prediction/evaluation/object_assets.py",
    "src/grasp_failure_prediction/evaluation/case_runner.py",
    "src/grasp_failure_prediction/evaluation/schema.py",
    "src/grasp_failure_prediction/evaluation/registry.py",
    "src/grasp_failure_prediction/evaluation/scenario_sampler.py",
    "src/grasp_failure_prediction/evaluation/configs/environments.yaml",
    "src/grasp_failure_prediction/evaluation/configs/execution_protocols.yaml",
    "src/grasp_failure_prediction/proposals/providers.py",
    "src/grasp_failure_prediction/proposals/generate.py",
    "src/grasp_failure_prediction/proposals/translation.py",
    "src/grasp_failure_prediction/proposals/artifacts.py",
    "src/grasp_failure_prediction/proposals/schema.py",
    "src/grasp_failure_prediction/proposals/configs/providers.yaml",
    "src/grasp_failure_prediction/proposals/configs/translations.yaml",
    "src/grasp_failure_prediction/proposals/physics_filter.py",
    "src/grasp_failure_prediction/proposals/execution.py",
    "src/grasp_failure_prediction/proposals/prediction_recording.py",
    "src/grasp_failure_prediction/proposals/registry.py",
    "src/grasp_failure_prediction/proposals/preflight.py",
    "src/grasp_failure_prediction/proposals/configs/preflight.yaml",
    "src/grasp_failure_prediction/proposals/configs/physics_filter.yaml",
    "src/grasp_failure_prediction/prediction/recorded_dataset.py",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__)


def bootstrap(protocol):
    sys.path.insert(0, str(Path(protocol["project"]) / "src"))
    sys.path.insert(0, str(Path(protocol["project"]) / "scripts"))
    for directory in protocol["dependency_dirs"]:
        if directory not in sys.path:
            sys.path.append(directory)


def verify(bindings):
    for filename, expected in bindings.items():
        if sha(filename) != expected:
            raise ValueError(f"frozen source/input changed: {filename}")


def generation_slots(objects):
    return [dict(slot_id=f"proposal_{i:04d}", object_id=obj["object_id"],
                 source_inventory_index=obj["source_inventory_index"], capture_id=f"capture_{oi:02d}_{ci:02d}",
                 capture_seed=2000 + 2 * oi + ci, camera_position_m=list(camera), generation_seed=seed)
            for i, (oi, obj, ci, camera, seed) in enumerate(
                (oi, obj, ci, camera, seed) for oi, obj in enumerate(objects)
                for ci, camera in enumerate(CAMERAS) for seed in GENERATION_SEEDS)]


def _geometry_paths(project, reference):
    if reference is None:
        return []
    mjcf = project / reference["mjcf_path"]
    tree = ET.fromstring(mjcf.read_bytes())
    compiler = tree.find("compiler")
    meshdir = "" if compiler is None else compiler.get("meshdir", "")
    return [mjcf] + [(mjcf.parent / meshdir / mesh.attrib["file"]).resolve()
                     for mesh in tree.findall("./asset/mesh")]


def prepare(project, inputs_project, output, inventory, *, hug_root, checkpoint,
            raw_output=None, attempts_per_object=32, scenario_seed=20261015):
    """Declare finite fresh generation and physical budgets without importing task code."""
    project, inputs_project, output = map(lambda p: Path(p).resolve(), (project, inputs_project, output))
    hug_root, checkpoint, inventory = map(lambda p: Path(p).resolve(), (hug_root, checkpoint, inventory))
    if output.exists():
        raise FileExistsError("new portable evidence directory required")
    if type(attempts_per_object) is not int or attempts_per_object not in (32, 64):
        raise ValueError("initial study quota must be128 or256 attempts (32 or64 per object)")
    if type(scenario_seed) is not int or scenario_seed < 0:
        raise ValueError("nonnegative scenario seed required")
    raw = Path(raw_output or inputs_project / "runs" / output.name).resolve()
    if not raw.is_relative_to(inputs_project / "runs") or raw.exists() or raw == output or output in raw.parents:
        raise ValueError("fresh private raw directory must be inside inputs_project/runs, separate from evidence")
    dependency = project / "research_artifacts/hug_native_mano_comparison_2026-10-15_v1/measurement/protocol.json"
    dependencies = json.loads(dependency.read_text())["dependency_dirs"]
    if not dependencies or any(not Path(p).is_dir() for p in dependencies):
        raise FileNotFoundError("existing frozen dependency directories unavailable")
    source = {str(project / name): sha(project / name) for name in SOURCE_FILES}
    input_files = {str(p): sha(p) for p in (inventory, checkpoint, dependency)}
    for name in ("src/models/mano.py", "assets/mano_models/models/MANO_RIGHT.pkl"):
        p = hug_root / name
        input_files[str(p)] = sha(p)
    # Native inference source is private/external; bind bytes without copying it.
    input_files.update({str(p):sha(p) for p in sorted((hug_root/"src").rglob("*.py"))})
    for directory in dependencies:
        package=Path(directory)/"dex_retargeting"
        input_files.update({str(p):sha(p) for p in sorted(package.rglob("*"))
                            if p.is_file() and p.suffix in {".py",".yaml",".yml"}})
    urdf = project / "third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf"
    assets = {str(p.resolve()): sha(p) for p in [urdf] + [(urdf.parent / m.attrib["filename"]).resolve()
             for m in ET.fromstring(urdf.read_bytes()).findall(".//mesh")]}
    entries = json.loads(inventory.read_text())["modern_cases"]
    if len(entries) != 32:
        raise ValueError("source inventory must be the explicitly selected original32")
    objects = []
    for index in SOURCE_INDICES:
        item = entries[index]
        resolved_path = Path(item["resolved_case_path"])
        old = json.loads(resolved_path.read_text())
        case = old["case"]
        if (case["execution_protocol"]["id"] != "fixed_grasp_lift_v2"
                or case["embodiment"]["id"] != "shadow_hand_right"):
            raise ValueError("source hand/base controller differs")
        input_files[str(resolved_path)] = sha(resolved_path)
        geometry_paths = _geometry_paths(inputs_project, case["object"].get("geometry"))
        assets.update({str(p): sha(p) for p in geometry_paths})
        objects.append(dict(object_id=item["object_id"], category_id=CATEGORIES[item["object_id"]],
            source_inventory_index=index, source_case_path=str(resolved_path), source_case=case))
    protocol = dict(schema_version=SCHEMA, status="PREPARED_NOT_AUTHORIZED", project=str(project),
        inputs_project=str(inputs_project), raw_output=str(raw), hug_root=str(hug_root), checkpoint=str(checkpoint),
        source_bindings=source, input_bindings=input_files, asset_bindings=assets,
        dependency_dirs=dependencies, runtime=runtime(), objects=objects, generation_slots=generation_slots(objects),
        generation_seeds=list(GENERATION_SEEDS), camera_positions_m=[list(p) for p in CAMERAS],
        quotas=dict(captures=8, native_proposals=32, pilot_attempts=20,
            study_attempts=4*attempts_per_object, study_attempts_per_object=attempts_per_object,
            maximum_capture_steps=8*2000, maximum_task_steps_per_attempt=3260,
            maximum_filter_axis_steps_per_proposal=6000,maximum_filter_close_steps=1480,
            maximum_filter_steps_per_proposal=7480,maximum_filter_total_steps=32*7480),
        scenario_seed=scenario_seed, study_approach_modes=["grasp_derived"],
        scenario_axes=dict(mass_multipliers=[.75,1.25],friction_coefficients=[.5,1.],table_offsets_m=[0.,.02],
            approach_modes=["grasp_derived"],sensor_noise_std_rad=[0.,.002]),
        prefix=dict(protocol_id="hug_grasp_approach_close_prefix_v1",prediction_time_s=2.96,history_steps=20,control_timestep_s=.04),
        numerical_policy=dict(time_increment_atol_s=1e-12, all_warning_counters_zero=True,
            guard_every_physics_step=True,finite_fields=["qpos","qvel","qacc","ctrl"],split_step_apis_forbidden=True),
        sensor_policy="independent seeded Gaussian qpos observations only; exact controls and raw qpos kept separately; no command noise",
        filter_policy="one nominal closed-state six-axis screen per mapped proposal before task rollouts; retain and execute all scheduled accepted/rejected proposals; error hard-stop",
        capture_scenario_policy="validate untouched nominal capture first; declare mass/inertia/friction/table/object translation separately; do not rewrite native/camera contract",
        table_offset_scope="translate table,object and palm by the declared Z offset; nuisance coverage, largely translation-equivalent under the ideal carrier",
        hard_stop="unexpected input/numerical/runtime/recording failure; no retries/replacements/repair; remaining rows NOT_RUN/outcome null",
        stage_order=["freeze preparation","authorized generate","freeze generated identity/schedule/folds","authorized20 pilot","inspect pilot integrity","separately authorized bounded study"],
        historical_data_policy="source assets/cases only; no saved commands or old rollouts enter new collection; no historical files are modified")
    output.mkdir(parents=True)
    write(output/"preparation.draft.json",protocol)
    return protocol


def read_protocol(output):
    protocol=json.loads((Path(output)/"preparation.json").read_text())
    if protocol.get("schema_version")!=SCHEMA or protocol.get("runtime")!=runtime():
        raise ValueError("unsupported declaration or different runtime")
    if protocol["generation_slots"]!=generation_slots(protocol["objects"]):
        raise ValueError("exact32 generation plan changed")
    verify({**protocol["source_bindings"],**protocol["input_bindings"],**protocol["asset_bindings"]})
    bootstrap(protocol)
    return protocol


def freeze(output, stage):
    output=Path(output)
    source=output/("preparation.draft.json" if stage=="preparation" else "generation.draft.json")
    value=json.loads(source.read_text())
    if stage=="preparation":
        verify({**value["source_bindings"],**value["input_bindings"],**value["asset_bindings"]})
    else:
        protocol=read_protocol(output)
        if value["preparation_sha256"]!=sha(output/"preparation.json"):
            raise ValueError("generation bound to another preparation")
        verify(value["artifact_bindings"])
        if value["status"]!="COMPLETED_BOUNDED_GENERATION" or len(value["proposals"])!=32:
            raise ValueError("incomplete/error generation cannot be frozen for collection")
    write(output/(stage+".json"),value)
    return sha(output/(stage+".json"))


def authorized(output, authorization, stage, *, extra_bindings=None, quota=None):
    receipt=json.loads(Path(authorization).read_text())
    expected={str((Path(output)/"preparation.json").resolve()):sha(Path(output)/"preparation.json"),**(extra_bindings or {})}
    if (receipt.get("schema_version")!="hug_build_authorization_v1" or receipt.get("stage")!=stage
            or receipt.get("authorized") is not True or receipt.get("bindings")!=expected
            or receipt.get("quota")!=quota):
        raise ValueError("explicit inspected authorization does not match this frozen stage/quota")
    verify(expected)
    return receipt


@contextmanager
def physics_budget(maximum_steps):
    """Sequential global per-substep guard, including capture and filter copies."""
    if type(maximum_steps) is not int or maximum_steps<0:
        raise ValueError("nonnegative finite step cap required")
    from run_hug_saved_recording_pilot import guarded_physics_step
    original={name:getattr(mujoco,name) for name in ("mj_step","mj_step1","mj_step2")}
    budget=dict(total=0,this_attempt=0)
    cap=dict(maximum_total_physics_steps=maximum_steps,physics_steps_per_attempt=maximum_steps)
    def step(model,data,*args,**kwargs):
        if args or kwargs:
            raise ValueError("multi-step integration bypass is forbidden")
        guarded_physics_step(model,data,original["mj_step"],budget,cap,time_tolerance_s=1e-12)
    def split(*args,**kwargs):
        raise RuntimeError("split-step APIs forbidden by collection protocol")
    mujoco.mj_step=step;mujoco.mj_step1=split;mujoco.mj_step2=split
    try:
        yield budget
    finally:
        for name,value in original.items():setattr(mujoco,name,value)


def _capture_case(obj,slot):
    # The raw native/camera capture is nominal. Scenario variation is later.
    case=json.loads(json.dumps(obj["source_case"]))
    case.update(case_id=slot["capture_id"],seed=slot["capture_seed"])
    case["grasp"].update(id="not_generated",prediction_path=None,observation_path=None,inference_seed=3000)
    return case


def _validate_fresh(protocol, proposal_path):
    """No solve or physics: inspect exactly the returned mapping/native files."""
    from PIL import Image
    from grasp_failure_prediction.evaluation import mapping_validation as mv
    from grasp_failure_prediction.evaluation.retargeting import RetargetedHandPose,camera_landmarks_to_wrist,reference_vectors,default_dex_urdf_root
    from grasp_failure_prediction.integrations.hug import load_hug_prediction
    from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
    from grasp_failure_prediction.proposals.artifacts import load_proposal
    native=Path(proposal_path).parent/"native"
    mapping=json.loads((native/"mapping_record.json").read_text())
    if mapping.get("native_proposal_sha256")!=sha(native/"proposal.pkl"):
        raise ValueError("mapping record binds another native proposal")
    canonical=load_proposal(proposal_path,verify_native_artifacts=True) if Path(proposal_path).is_file() else None
    if canonical is None and mapping.get("mapping_status") not in {"refused_guard","refused_invalid_retarget"}:
        raise ValueError("returned mapping is missing its canonical execution command")
    with mv.static_only() as counts:
        grasp=load_hug_prediction(native/"proposal.pkl")
        local=camera_landmarks_to_wrist(grasp.landmarks_3d,grasp.T_camera_wrist)@MANO_TO_OPERATOR_RIGHT.T
        world_palm=None
        if canonical is not None:
            names,q,world=_canonical_mapping_binding(canonical,mapping,json.loads((native/"scene_contract.json").read_text()))
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,world[:3,:3].reshape(9));world_palm=(world[:3,3],quat)
        else:
            names=tuple(mapping.get("joint_names",[]));q=np.asarray(mapping.get("qpos_rad",[]),float)
        pose=RetargetedHandPose(tuple(names),q,local,
            reference_vectors(local,mv.HUMAN_TASK_INDICES),grasp.T_camera_wrist)
        model=mujoco.MjModel.from_binary_path(str(native/"scene.mjb"));data=mujoco.MjData(model)
        with np.load(native/"scene_state.npz",allow_pickle=False) as state:
            data.qpos[:]=state["qpos"];data.qvel[:]=state["qvel"]
        mujoco.mj_forward(model,data)
        with Image.open(native/"rgb.png") as im:size=im.size
        report=mv.validate_hug_mapping(grasp,pose,urdf_root=default_dex_urdf_root(),
            T_world_camera=np.load(native/"T_world_camera.npy",allow_pickle=False),
            camera_intrinsics=np.load(native/"intrinsics.npy",allow_pickle=False),image_size=size,
            world_palm=world_palm,camera_model=model,camera_data=data,
            native_reconstructor=mv.make_mano_reconstructor(Path(protocol["hug_root"])),
            mapping_record=mapping,mapping_kind="fresh")
    if counts["physics_step_attempts"]:
        raise RuntimeError("static mapping validation attempted physics")
    report["static_api_counts"]=counts
    report["canonical_execution_command_binding"]="PASS" if canonical is not None else "NOT_APPLICABLE_MAPPING_REFUSED_NO_COMMAND"
    return report


def _canonical_mapping_binding(proposal,mapping,capture):
    """Validate the actual canonical command/placement, not a parallel payload."""
    from grasp_failure_prediction.proposals.execution import _matrix
    from grasp_failure_prediction.evaluation.mapping_validation import TOLERANCES,_rigid
    names=tuple(mapping["joint_names"])
    if len(names)!=len(set(names)) or set(names)!=set(proposal.joint_targets_rad):
        raise ValueError("canonical command joint names differ from mapping record")
    q=np.asarray([proposal.joint_targets_rad[name] for name in names],float)
    if not np.array_equal(q,np.asarray(mapping["qpos_rad"],float)):
        raise ValueError("canonical command joint targets differ from mapping record")
    object_palm=_matrix(proposal.object_to_palm.translation_m,proposal.object_to_palm.quaternion_xyzw)
    world=_matrix(capture["object_position_m"],capture["object_orientation_xyzw"])@object_palm
    for actual,key in ((object_palm,"object_to_palm_transform"),(world,"world_palm_transform")):
        recorded=_rigid(mapping[key])
        if (not np.allclose(actual[:3,3],recorded[:3,3],rtol=0,atol=TOLERANCES["compiled_fk_m"])
                or not np.allclose(actual[:3,:3],recorded[:3,:3],rtol=0,atol=TOLERANCES["rotation_matrix"])):
            raise ValueError("canonical palm transform differs from mapping record")
    return names,q,world


def _proposal_row(protocol,slot,path,validation):
    from grasp_failure_prediction.proposals.artifacts import load_proposal,proposal_content_hash
    proposal=load_proposal(path,verify_native_artifacts=True)
    native=Path(path).parent/"native/proposal.pkl"
    return dict(**slot,status="MAPPED" if validation["static_input_gate"]=="PASS" else "REFUSED_STATIC_GATE",
        mapping_status="ok" if validation["static_input_gate"]=="PASS" else "refused_guard",outcome=None,
        proposal_path=str(Path(path).resolve()),proposal_hash=proposal_content_hash(proposal),
        native_sha256=sha(native),geometry_group_id=proposal.input_binding.geometry_group_id,
        observation_group_id=proposal.input_binding.observation_group_id,
        proposal_group_id=digest(dict(native_pose_sha256=sha(native),observation_group_id=proposal.input_binding.observation_group_id,
                                      geometry_group_id=proposal.input_binding.geometry_group_id)),
        mapping_validation=validation)


def _loaded_state_hash(directory):
    report=json.loads((Path(directory)/"sample_0000/native/inference_report.json").read_text())
    value=report.get("loaded_model_state_sha256")
    if not isinstance(value,str) or len(value)!=71 or not value.startswith("sha256:"):
        raise ValueError("fresh inference must bind the complete loaded model state, including external encoder weights")
    int(value[7:],16)
    return value


def _freeze_schedules(protocol,proposals):
    from grasp_failure_prediction.evaluation.scenario_sampler import ScenarioAxes,sample_scenarios,freeze_category_folds
    objects=[]
    for obj in protocol["objects"]:
        rows=[r for r in proposals if r["object_id"]==obj["object_id"]]
        observations=sorted({r["observation_group_id"] for r in rows})
        geometries={r["geometry_group_id"] for r in rows}
        if len(observations)!=2 or len(geometries)!=1:
            raise ValueError("all fresh capture identities, including refused proposals, must remain resolvable")
        objects.append(dict(object_id=obj["object_id"],category_id=obj["category_id"],geometry_group_id=geometries.pop(),
                            observation_group_ids=observations))
    by_input={(r["object_id"],r["observation_group_id"],r["generation_seed"]):r for r in proposals}
    studies={}
    axes=ScenarioAxes(**{key:tuple(value) for key,value in protocol["scenario_axes"].items()})
    for condition,quota,seed in (("pilot",5,protocol["scenario_seed"]),("study",protocol["quotas"]["study_attempts_per_object"],protocol["scenario_seed"]+1)):
        schedule=sample_scenarios(objects,seed=seed,attempts_per_object=quota,generation_seeds=GENERATION_SEEDS,axes=axes)
        for i,row in enumerate(schedule["scenarios"]):
            proposal=by_input[(row["object_id"],row["observation_group_id"],row["generation_seed"])]
            row.update(proposal_group_id=proposal["proposal_group_id"],generation_slot_id=proposal["slot_id"],
                       attempt_id=f"{condition}_attempt_{i:04d}")
        # IDs bind actual proposal identities too, before any physical outcome.
        for row in schedule["scenarios"]:
            row["scenario_id"]="scenario_"+digest({k:v for k,v in row.items() if k!="scenario_id"})[7:23]
        schedule["counts"]["proposal_families"]=len({r["proposal_group_id"] for r in schedule["scenarios"]})
        studies[condition]=dict(schedule=schedule,folds=freeze_category_folds(schedule),
            attempt_binding=dict(schema_version="hug_attempt_scenario_binding_v1",schedule_sha256=digest(schedule),
                attempt_to_scenario={r["attempt_id"]:r["scenario_id"] for r in schedule["scenarios"]}))
    # JSON is the frozen interchange format; canonicalize tuples from axes now.
    return json.loads(json.dumps(studies,allow_nan=False))


def generate(output,authorization):
    output=Path(output).resolve();protocol=read_protocol(output)
    authorized(output,authorization,"generate",quota=protocol["quotas"]["native_proposals"])
    raw=Path(protocol["raw_output"])
    if raw.exists() or (output/"generation.draft.json").exists():
        raise FileExistsError("generation cannot be retried or overwrite prior raw outputs")
    raw.mkdir(parents=True)
    from grasp_failure_prediction.integrations.sim_observations import capture_scene
    from grasp_failure_prediction.proposals.generate import generate_proposals,_expected_input_binding
    from grasp_failure_prediction.proposals.schema import ProposalRequest
    objects={o["object_id"]:o for o in protocol["objects"]}
    captures={};rows=[];hard_error=None;physics_steps=0;loaded_state=None;started=time.monotonic()
    for slot in protocol["generation_slots"]:
        row=dict(**slot,status="NOT_RUN",mapping_status="setup_error",outcome=None)
        if hard_error:
            rows.append(row);continue
        directory=raw/"generation"/slot["slot_id"]
        try:
            capture_id=slot["capture_id"]
            if capture_id not in captures:
                case_path=raw/"capture_cases"/(capture_id+".json")
                write(case_path,_capture_case(objects[slot["object_id"]],slot))
                capture_path=raw/"observations"/capture_id
                with physics_budget(2000) as budget:
                    try:
                        capture_scene(case_path,capture_path,project_root=protocol["inputs_project"],camera_position_m=slot["camera_position_m"])
                    finally:physics_steps+=budget["total"]
                captures[capture_id]=str(capture_path)
            capture=Path(captures[capture_id]);contract=json.loads((capture/"scene_contract.json").read_text())
            request=ProposalRequest.model_validate(dict(request_id=slot["slot_id"],provider=dict(id="hug_mano_shadow_v1"),
                object=dict(id=slot["object_id"],geometry_content_hash=contract["geometry_content_hash"]),
                observation_path=str(capture.relative_to(protocol["inputs_project"])),seed=slot["generation_seed"],sample_count=1))
            request_path=raw/"requests"/(slot["slot_id"]+".json");write(request_path,request.model_dump(mode="json"))
            binding=_expected_input_binding(request,Path(protocol["inputs_project"]))
            row.update(geometry_group_id=binding["geometry_group_id"],observation_group_id=binding["observation_group_id"],
                       proposal_group_id=digest(dict(generation_slot=slot["slot_id"],input_binding=binding)))
            try:
                with physics_budget(0):
                    manifest=generate_proposals(request_path,directory,project_root=protocol["inputs_project"],
                        hug_root=protocol["hug_root"],checkpoint=protocol["checkpoint"],
                        runtime_dependency_dirs=protocol["dependency_dirs"])
            except Exception:
                mapping_path=directory/"sample_0000/native/mapping_record.json"
                mapping=json.loads(mapping_path.read_text()) if mapping_path.is_file() else None
                if mapping and mapping.get("mapping_status") in {"refused_invalid_retarget","refused_guard"}:
                    state_hash=_loaded_state_hash(directory)
                    if loaded_state is not None and state_hash!=loaded_state:raise RuntimeError("loaded model state changed across fresh generation")
                    loaded_state=state_hash;row["loaded_model_state_sha256"]=state_hash
                    validation=_validate_fresh(protocol,directory/"sample_0000/proposal.json")
                    write(directory/"mapping_validation.json",validation);row["mapping_validation"]=validation
                    row.update(status="MAPPING_REFUSED",mapping_status=mapping["mapping_status"],mapping_record_path=str(mapping_path))
                    native=mapping_path.parent/"proposal.pkl"
                    row.update(native_sha256=sha(native),proposal_group_id=digest(dict(native_pose_sha256=sha(native),
                        observation_group_id=row["observation_group_id"],geometry_group_id=row["geometry_group_id"])))
                    (directory/"mapping_refusal_traceback.txt").write_text(traceback.format_exc())
                    rows.append(row);continue
                raise
            path=directory/manifest.proposals[0].proposal_path
            validation=_validate_fresh(protocol,path)
            write(directory/"mapping_validation.json",validation)
            row=_proposal_row(protocol,slot,path,validation)
            state_hash=_loaded_state_hash(directory)
            if loaded_state is not None and state_hash!=loaded_state:raise RuntimeError("loaded model state changed across fresh generation")
            loaded_state=state_hash;row["loaded_model_state_sha256"]=state_hash
            verify({**protocol["source_bindings"],**protocol["input_bindings"],**protocol["asset_bindings"]})
        except Exception as error:
            directory.mkdir(parents=True,exist_ok=True)
            (directory/"unexpected_error_traceback.txt").write_text(traceback.format_exc())
            row.update(status="ERROR",error=dict(type=type(error).__name__,message=str(error)))
            hard_error=row["error"]
        rows.append(row)
        print(json.dumps(dict(stage="generate",slot_id=slot["slot_id"],status=row["status"])),flush=True)
    generated=dict(schema_version="hug_build_generation_v1",preparation_sha256=sha(output/"preparation.json"),
        status="HARD_STOP" if hard_error else "COMPLETED_BOUNDED_GENERATION",proposals=rows,captures=captures,
        counts={status:sum(r["status"]==status for r in rows) for status in sorted({r["status"] for r in rows})},
        elapsed_wall_s=time.monotonic()-started,capture_physics_steps=physics_steps,error=hard_error,
        loaded_model_state_sha256=loaded_state,
        loaded_model_state_policy="all32native generations, including mapping refusals, must report exactly the same complete loaded state hash",
        artifact_bindings={str(p):sha(p) for p in sorted(raw.rglob("*")) if p.is_file()})
    if not hard_error:
        try:
            generated["conditions"]=_freeze_schedules(protocol,rows)
        except Exception as error:
            (raw/"schedule_error_traceback.txt").write_text(traceback.format_exc())
            generated.update(status="HARD_STOP",error=dict(type=type(error).__name__,message=str(error)))
            generated["artifact_bindings"][str(raw/"schedule_error_traceback.txt")]=sha(raw/"schedule_error_traceback.txt")
    write(output/"generation.draft.json",generated)
    return generated


def collect(output,authorization,condition):
    output=Path(output).resolve();protocol=read_protocol(output)
    generation_path=output/"generation.json";generated=json.loads(generation_path.read_text())
    if generated.get("status")!="COMPLETED_BOUNDED_GENERATION":
        raise ValueError("collection requires completed frozen fresh generation")
    verify(generated["artifact_bindings"])
    if generated["conditions"]!=_freeze_schedules(protocol,generated["proposals"]):
        raise ValueError("generated schedule/folds changed")
    quota=protocol["quotas"]["pilot_attempts" if condition=="pilot" else "study_attempts"]
    extra={str(generation_path):sha(generation_path)}
    if condition=="study":
        pilot=output/"pilot_summary.json"
        if not pilot.exists() or json.loads(pilot.read_text()).get("status")!="COMPLETED_BOUNDED_QUOTA":
            raise ValueError("bounded pilot must complete without integrity errors before a separately authorized study")
        verify(json.loads(pilot.read_text())["raw_artifact_bindings"])
        extra[str(pilot)]=sha(pilot)
    authorized(output,authorization,"collect_"+condition,extra_bindings=extra,quota=quota)
    raw=Path(protocol["raw_output"])/condition
    if raw.exists() or (output/(condition+"_summary.json")).exists():
        raise FileExistsError("collection cannot retry, overwrite or replace attempts")
    raw.mkdir(parents=True)
    if condition=="pilot":
        filters=_run_filters(protocol,generated,raw/"filters")
    else:
        filters=json.loads((output/"pilot_summary.json").read_text())["filters"]
        verify(filters["artifact_bindings"])
    rows=[];hard_error=filters.get("error");total_steps=0;started=time.monotonic()
    study=generated["conditions"][condition]
    schedule=study["schedule"]
    if len(schedule["scenarios"])!=quota:
        raise ValueError("frozen condition quota differs")
    proposals={r["slot_id"]:r for r in generated["proposals"]}
    for scenario in schedule["scenarios"]:
        proposal=proposals[scenario["generation_slot_id"]]
        attempt_dir=raw/"attempts"/scenario["attempt_id"]
        if hard_error:
            row=_null_attempt(protocol,proposal,scenario,attempt_dir,mapping="setup_error",code="not_run_hard_stop",status="NOT_RUN")
        elif proposal["mapping_status"]!="ok":
            row=_null_attempt(protocol,proposal,scenario,attempt_dir,mapping=proposal["mapping_status"],code="fresh_mapping_refused",status="MAPPING_REFUSED")
        else:
            row=_collect_attempt(protocol,proposal,scenario,attempt_dir,
                                 filters["proposals"].get(proposal["slot_id"],{}))
            total_steps+=row["physics_steps"]
            if row["status"]=="ERROR":hard_error=row["error"]
        rows.append(row)
        print(json.dumps(dict(stage="collect",condition=condition,attempt_id=row["attempt_id"],status=row["status"])),flush=True)
    if total_steps>quota*protocol["quotas"]["maximum_task_steps_per_attempt"]:
        raise RuntimeError("global task step cap exceeded")
    report=dict(schema_version="hug_build_collection_result_v1",condition=condition,
        status="HARD_STOP" if hard_error else "COMPLETED_BOUNDED_QUOTA",error=hard_error,
        preparation_sha256=sha(output/"preparation.json"),generation_sha256=sha(generation_path),
        schedule_sha256=digest(schedule),quota=quota,attempts=rows,filters=filters,
        physics_steps=total_steps,elapsed_wall_s=time.monotonic()-started,
        counts={status:sum(r["status"]==status for r in rows) for status in sorted({r["status"] for r in rows})})
    try:
        verify({**protocol["source_bindings"],**protocol["input_bindings"],**protocol["asset_bindings"],**generated["artifact_bindings"]})
        from grasp_failure_prediction.prediction.recorded_dataset import export_recorded_dataset,PrefixSpec
        roots=[Path(r["recording_path"]) for r in rows]
        # All-zero/one-class or ineligible cohorts stay as measured; never repair quotas.
        eligible=sum(r["status"]=="COMPLETED" and r["outcome"]["failure_type"]!="approach_collision" for r in rows)
        if eligible:
            report["dataset"]=export_recorded_dataset(roots,output/(condition+"_prefixes"),prefix_spec=PrefixSpec(**protocol["prefix"]))
        else:
            report["dataset"]=dict(status="UNDEFINED_NO_ELIGIBLE_PREFIXES",counts=dict(attempted=len(rows),included=0))
        write(output/(condition+"_schedule.json"),schedule)
        write(output/(condition+"_folds.json"),study["folds"])
        write(output/(condition+"_attempt_scenario_binding.json"),study["attempt_binding"])
    except Exception as error:
        (raw/"postprocessing_error_traceback.txt").write_text(traceback.format_exc())
        report.update(status="HARD_STOP",error=dict(type=type(error).__name__,message=str(error)))
    report["raw_artifact_bindings"]={str(p):sha(p) for p in sorted(raw.rglob("*")) if p.is_file()}
    write(output/(condition+"_summary.json"),report)
    return report


def _case_and_context(protocol,proposal_row,work):
    """Validate the nominal capture before any declared scenario intervention."""
    from grasp_failure_prediction.proposals.artifacts import load_proposal,proposal_content_hash
    from grasp_failure_prediction.proposals.schema import EvaluationCaseV2
    from grasp_failure_prediction.proposals.registry import load_provider_registry
    from grasp_failure_prediction.proposals.prediction_recording import HugPredictionRecording
    from grasp_failure_prediction.proposals.execution import _validate_pose,_world_palm
    from grasp_failure_prediction.evaluation.case_runner import _resolve_object_geometry
    from grasp_failure_prediction.evaluation.registry import resolve_case
    path=Path(proposal_row["proposal_path"]);proposal=load_proposal(path,verify_native_artifacts=True)
    if proposal_content_hash(proposal)!=proposal_row["proposal_hash"]:
        raise ValueError("fresh frozen proposal content changed")
    native=path.parent/"native";capture=json.loads((native/"scene_contract.json").read_text())
    obj=next(o for o in protocol["objects"] if o["object_id"]==proposal_row["object_id"])
    case=json.loads(json.dumps(obj["source_case"]))
    case.update(schema_version="eval_case_v2",case_id=proposal_row["slot_id"])
    case["object"]["mass_kg"]=capture["mass_kg"]
    case["initial_condition"]=dict(id="nominal_fresh_capture",object_position_m=capture["object_position_m"],
                                    object_orientation_xyzw=capture["object_orientation_xyzw"])
    case["grasp"]=dict(id=proposal.proposal_id,proposal_path=str(path.relative_to(protocol["inputs_project"])),expected_proposal_hash=proposal_content_hash(proposal))
    case=EvaluationCaseV2.model_validate(case);resolved=resolve_case(case)
    case_path=work/"nominal_case.json";write(case_path,case.model_dump(mode="json"))
    context=HugPredictionRecording(case_path,work/"validation_only",Path(protocol["inputs_project"]))
    geometry=_resolve_object_geometry(case,Path(protocol["inputs_project"]))
    context.bind(case=case,resolved=resolved,geometry=geometry,proposal=proposal,proposal_path=path,
        provider_spec=load_provider_registry().resolve(proposal.provider.id,proposal.provider.config_hash))
    context.validate_inputs();context.verify_bindings()
    pose,_=_validate_pose(proposal)
    palm,quaternion=_world_palm(case,proposal)
    return context,pose,palm,quaternion


def _new_runner(protocol,context):
    from grasp_failure_prediction.evaluation.approach_execution import grasp_approach_protocol,GraspApproachRecordedRunner
    from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
    simulator=context.resolved.environment.simulator
    return GraspApproachRecordedRunner(grasp_approach_protocol(context.resolved.execution_protocol),default_dex_urdf_root(),
        physics_timestep_s=simulator.physics_timestep_s,control_timestep_s=simulator.control_timestep_s,object_geometry=context.geometry)


def _apply_scenario(runner,context,scenario,palm):
    """Apply interventions to a new model only; capture/native files stay unchanged."""
    case=context.case;offset=scenario["table_offset_m"]
    position=np.asarray(case.initial_condition.object_position_m)+[0,0,offset]
    runner.reset(seed=scenario["sensor_noise_seed"],object_mass_kg=case.object.mass_kg*scenario["mass_multiplier"],
        object_position_m=position,object_orientation_wxyz=np.asarray(case.initial_condition.object_orientation_xyzw)[[3,0,1,2]])
    model=runner.model;table=runner._table_geom_id
    model.geom_pos[table,2]+=offset
    objects=sorted(runner.object_collision_geom_ids)
    hand=[i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in runner._hand_body_ids and (model.geom_contype[i] or model.geom_conaffinity[i])]
    if not objects or not hand:raise ValueError("declared friction intervention needs hand and object collision geoms")
    if any(int(a) in objects or int(b) in objects for a,b in zip(model.pair_geom1,model.pair_geom2)):
        raise ValueError("explicit object contact-pair friction needs a separately declared intervention")
    before=model.geom_friction.copy();ids=sorted(set(objects+hand))
    model.geom_friction[ids,0]=scenario["friction_coefficient"]
    if not np.array_equal(model.geom_friction[:,1:],before[:,1:]):raise RuntimeError("secondary friction components changed")
    mujoco.mj_forward(model,runner.data)
    obj=runner._object_body_id
    actual=dict(scenario=scenario,nominal_capture_sha256=context.scene_binding["scene_contract_sha256"],
        object_mass_kg=float(model.body_mass[obj]),object_inertia_diagonal_kg_m2=model.body_inertia[obj].tolist(),
        object_inertial_frame_quaternion_wxyz=model.body_iquat[obj].tolist(),object_inertial_position_m=model.body_ipos[obj].tolist(),
        object_position_m=runner.data.qpos[runner._object_qpos_address:runner._object_qpos_address+3].tolist(),
        table_geom_position_m=model.geom_pos[table].tolist(),table_world_position_m=runner.data.geom_xpos[table].tolist(),
        hand_collision_geom_ids=hand,object_collision_geom_ids=objects,
        friction_before=before[ids].tolist(),friction_after=model.geom_friction[ids].tolist(),friction_geom_ids=ids,
        table_friction=model.geom_friction[table].tolist(),friction_scope="sliding coefficient on both hand/object; table coefficients unchanged",
        table_offset_scope="joint translation nuisance under ideal carrier",contact_diagnostic_lag_s=runner.physics_timestep_s)
    expected=case.object.mass_kg*scenario["mass_multiplier"]
    if not np.isclose(actual["object_mass_kg"],expected,rtol=0,atol=1e-12):raise RuntimeError("scenario mass intervention mismatch")
    if not np.all(model.geom_friction[ids,0]==scenario["friction_coefficient"]):raise RuntimeError("scenario friction intervention mismatch")
    from run_hug_saved_recording_pilot import expected_object_inertia,_quat_matrix
    expected_center,expected_tensor=expected_object_inertia(context.geometry,expected)
    rotation=_quat_matrix(model.body_iquat[obj]);tensor=rotation@np.diag(model.body_inertia[obj])@rotation.T
    actual["object_inertia_tensor_kg_m2"]=tensor.tolist()
    if not np.allclose(tensor,expected_tensor,rtol=0,atol=1e-12) or not np.allclose(model.body_ipos[obj],expected_center,rtol=0,atol=1e-6):
        raise RuntimeError("scenario object inertia does not match the unchanged geometry with declared mass scaling")
    return np.asarray(palm)+[0,0,offset],actual


class _NoisyObservations:
    """Qpos measurement noise only. The wrapped strict recorder owns artifacts."""
    def __init__(self,recorder,std,seed):
        self.recorder=recorder;self.std=std;self.rng=np.random.default_rng(seed)
    def append(self,**values):
        clean=np.asarray(values["hand_qpos"]).copy()
        noise=np.zeros_like(clean) if self.std==0 else self.rng.normal(0,self.std,size=clean.shape)
        values["hand_qpos"]=clean+noise
        values["diagnostics"]={**values.get("diagnostics",{}),"clean_hand_qpos_rad":clean,"qpos_observation_noise_rad":noise}
        return self.recorder.append(**values)
    def __getattr__(self,name):return getattr(self.recorder,name)


def _base_hashes(protocol,proposal,scenario):
    return dict(code=digest(protocol["source_bindings"]),configuration=digest(dict(preparation=protocol,scenario=scenario)),
        inputs=digest(dict(proposal=proposal,scenario=scenario)),assets=digest(protocol["asset_bindings"]),
        scenario=digest(scenario),native_pose=proposal.get("native_sha256",digest(proposal)),
        mapped_pose=proposal.get("proposal_hash",digest(proposal)))


def _metadata(protocol,proposal,scenario,*,mapping="ok",execution="not_executed",outcome=None,code=None,hashes=None):
    from grasp_failure_prediction.evaluation.prediction_recording import AttemptMetadata
    return AttemptMetadata(attempt_id=scenario["attempt_id"],hand_id="shadow_hand_right",execution_protocol_id="hug_grasp_approach_v1",
        object_id=scenario["object_id"],geometry_group_id=scenario["geometry_group_id"],
        observation_group_id=scenario["observation_group_id"],proposal_group_id=scenario["proposal_group_id"],
        mapping=dict(status=mapping,code=code,message=None if mapping=="ok" else (code or "declared unexecuted mapping/setup refusal")),execution_status=execution,outcome=outcome,
        provenance_hashes=hashes or _base_hashes(protocol,proposal,scenario))


def _null_attempt(protocol,proposal,scenario,directory,*,mapping,code,status):
    from grasp_failure_prediction.evaluation.prediction_recording import write_incomplete_attempt
    path=directory/"prediction_recording"
    write_incomplete_attempt(path,_metadata(protocol,proposal,scenario,mapping=mapping,code=code))
    return dict(attempt_id=scenario["attempt_id"],scenario_id=scenario["scenario_id"],status=status,mapping_status=mapping,
        outcome=None,physics_steps=0,recording_path=str(path),attempt_sha256=sha(path/"attempt.json"))


def _collect_attempt(protocol,proposal,scenario,directory,filter_record):
    from grasp_failure_prediction.evaluation import scoring
    from grasp_failure_prediction.evaluation.approach_execution import ApproachPlanningRejected
    from grasp_failure_prediction.evaluation.prediction_recording import write_incomplete_attempt,load_attempt
    from run_hug_saved_recording_pilot import _save_partial
    directory.mkdir(parents=True)
    runner=None;budget=dict(total=0);executed=False;hashes=_base_hashes(protocol,proposal,scenario)
    try:
        context,pose,palm,quaternion=_case_and_context(protocol,proposal,directory)
        runner=_new_runner(protocol,context)
        palm,actual=_apply_scenario(runner,context,scenario,palm)
        write(directory/"scenario_realization.json",actual);hashes["scenario_realization"]=sha(directory/"scenario_realization.json")
        runner.prediction_recorder=_NoisyObservations(runner.prediction_recorder,scenario["sensor_noise_std_rad"],scenario["sensor_noise_seed"])
        hashes["label_rule"]=digest(dict(scorer=sha(Path(scoring.__file__)),success=runner.protocol.success.model_dump(mode="json"),
            control_timestep_s=runner.control_timestep_s))
        hashes["execution_configuration"]=digest(runner.protocol.model_dump(mode="json"))
        context.verify_bindings()
        mujoco.mj_saveModel(runner.model,str(directory/"scene.mjb"));hashes["model"]=sha(directory/"scene.mjb")
        with physics_budget(protocol["quotas"]["maximum_task_steps_per_attempt"]) as budget:
            trace=runner.execute(pose,grasp_palm_position_m=palm,grasp_palm_quaternion_wxyz=quaternion)
        executed=True
        if budget["total"]!=3260 or len(trace.steps)!=163:raise RuntimeError("completed task differs from exact declared163endpoints/3260steps")
        outcome=asdict(scoring.score_trace(trace,runner.protocol,control_timestep_s=runner.control_timestep_s))
        failure=outcome["failure_type"];failure=None if failure is None else failure.value
        first=next((s.time_s-runner.physics_timestep_s for s in trace.steps if s.hand_table_contact and s.state.value in {"move_to_pregrasp","approach"}),None)
        physical=dict(success=outcome["success"],failure_type=failure,label_rule_id="hug_grasp_approach_task_outcome_v1",
            first_failure_time_s=first if failure=="approach_collision" else None)
        plan=runner.approach_plan.as_record();write(directory/"approach_plan.json",plan);hashes["approach_plan"]=sha(directory/"approach_plan.json")
        context.verify_bindings();verify({**protocol["source_bindings"],**protocol["input_bindings"],**protocol["asset_bindings"]})
        pending=directory/"_pending_recording"
        runner.prediction_recorder.write(pending,_metadata(protocol,proposal,scenario,execution="completed",outcome=physical,hashes=hashes))
        load_attempt(pending)
        pending.rename(directory/"prediction_recording")
        write(directory/"task_score.json",outcome)
        row=dict(attempt_id=scenario["attempt_id"],scenario_id=scenario["scenario_id"],status="COMPLETED",mapping_status="ok",
            outcome=physical,physical_score=outcome,physics_steps=budget["total"],recording_path=str(directory/"prediction_recording"),
            attempt_sha256=sha(directory/"prediction_recording/attempt.json"),scenario_realization_sha256=hashes["scenario_realization"],filter_status=filter_record.get("filter_status"))
        write(directory/"result.json",row)
        return row
    except Exception as error:
        (directory/"exception_traceback.txt").write_text(traceback.format_exc())
        for name in ("result.json","task_score.json"):
            path=directory/name
            if path.exists():path.rename(directory/("quarantined_"+name))
        for name in ("_pending_recording","prediction_recording"):
            pending=directory/name
            if pending.exists():
                pending.rename(directory/("quarantined"+name))
        for p in directory.rglob("attempt.json"):p.rename(p.with_name("quarantined_attempt.json"))
        if runner is not None:
            try:_save_partial(directory,runner.prediction_recorder,runner.data)
            except Exception as partial_error:write(directory/"partial_save_error.json",dict(type=type(partial_error).__name__,message=str(partial_error)))
            if runner.approach_plan is not None and not (directory/"approach_plan.json").exists():write(directory/"approach_plan.json",runner.approach_plan.as_record())
        if isinstance(error,ApproachPlanningRejected):
            return _null_attempt(protocol,proposal,scenario,directory,mapping="unreachable_pregrasp",code=error.code,status="APPROACH_REFUSED")
        # Attempted physics, including a failed first substep, belongs to the execution denominator.
        execution="runtime_error" if executed or budget["total"] else "not_executed"
        mapping="ok" if execution=="runtime_error" else "setup_error"
        metadata=_metadata(protocol,proposal,scenario,mapping=mapping,execution=execution,
                           code=None if mapping=="ok" else type(error).__name__,hashes=hashes)
        write_incomplete_attempt(directory/"prediction_recording",metadata)
        return dict(attempt_id=scenario["attempt_id"],scenario_id=scenario["scenario_id"],status="ERROR",mapping_status=mapping,
            outcome=None,error=dict(type=type(error).__name__,message=str(error)),physics_steps=budget["total"],
            recording_path=str(directory/"prediction_recording"),attempt_sha256=sha(directory/"prediction_recording/attempt.json"))


class _ClosedStateReached(Exception):pass


def _run_filters(protocol,generated,directory):
    from grasp_failure_prediction.proposals.physics_filter import filter_closed_grasp,PhysicsFilterGeometry,load_physics_filter_spec
    from grasp_failure_prediction.evaluation.approach_execution import ApproachPlanningRejected
    from run_hug_saved_recording_pilot import _save_partial
    results={};error=None;steps=0
    spec=load_physics_filter_spec()
    for proposal in generated["proposals"]:
        path=directory/proposal["slot_id"];runner=None;budget=dict(total=0)
        row=dict(filter_status="NOT_RUN",task_outcome_label=None,predictor_feature_eligible=False)
        if error or proposal["mapping_status"]!="ok":
            row["reason"]="hard_stop" if error else "mapping_refused"
        else:
            try:
                path.mkdir(parents=True)
                context,pose,palm,quaternion=_case_and_context(protocol,proposal,path)
                runner=_new_runner(protocol,context)
                nominal=dict(mass_multiplier=1.,friction_coefficient=1.,table_offset_m=0.,sensor_noise_seed=0)
                palm,realization=_apply_scenario(runner,context,nominal,palm)
                write(path/"nominal_realization.json",realization)
                def stop_at_closed(step):
                    if step.time_s>=protocol["prefix"]["prediction_time_s"]-1e-9:
                        if step.state.value!="close_fingers" or abs(step.time_s-2.96)>1e-9:raise RuntimeError("closed-state cutoff differs")
                        raise _ClosedStateReached()
                with physics_budget(protocol["quotas"]["maximum_filter_steps_per_proposal"]) as budget:
                    try:
                        runner.execute(pose,grasp_palm_position_m=palm,grasp_palm_quaternion_wxyz=quaternion,step_callback=stop_at_closed)
                    except _ClosedStateReached:pass
                    else:raise RuntimeError("close-only filter acquisition did not stop before lift")
                    if budget["total"]!=protocol["quotas"]["maximum_filter_close_steps"]:raise RuntimeError("closed-state physics budget mismatch")
                    _save_partial(path,runner.prediction_recorder,runner.data)
                    mujoco.mj_saveModel(runner.model,str(path/"scene.mjb"))
                    geometry=PhysicsFilterGeometry.from_shadow_model(runner.model)
                    result=filter_closed_grasp(runner.model,runner.data,geometry,configuration=spec,
                        provenance=dict(native_sha256=proposal["native_sha256"],mapped_proposal_sha256=proposal["proposal_hash"],
                            model_sha256=sha(path/"scene.mjb"),scope="nominal close-only source; scenario physics differs; not task outcome"))
                row=result.as_record();row["close_and_filter_physics_steps"]=budget["total"]
                context.verify_bindings()
                if result.filter_status=="error":raise RuntimeError("six-axis filter numerical/runtime error")
            except ApproachPlanningRejected as refusal:
                row.update(filter_status="NOT_RUN",reason="unreachable_pregrasp",approach_plan=refusal.plan.as_record())
            except Exception as fault:
                path.mkdir(parents=True,exist_ok=True);(path/"exception_traceback.txt").write_text(traceback.format_exc())
                row.update(filter_status="error",error=dict(type=type(fault).__name__,message=str(fault)))
                error=row["error"]
                if runner is not None:
                    try:_save_partial(path,runner.prediction_recorder,runner.data)
                    except Exception:pass
        steps+=budget["total"]
        path.mkdir(parents=True,exist_ok=True);write(path/"filter_result.json",row)
        results[proposal["slot_id"]]=row
        print(json.dumps(dict(stage="filter",slot_id=proposal["slot_id"],status=row["filter_status"])),flush=True)
    if steps>protocol["quotas"]["maximum_filter_total_steps"]:raise RuntimeError("filter total step cap exceeded")
    return dict(proposals=results,error=error,physics_steps=steps,configuration=asdict(spec),
        artifact_bindings={str(p):sha(p) for p in directory.rglob("*") if p.is_file()})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--inputs-project",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--hug-root",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--raw-output",type=Path);p.add_argument("--attempts-per-object",type=int,choices=(32,64),default=32)
    p.add_argument("--scenario-seed",type=int,default=20261015)
    p=sub.add_parser("freeze");p.add_argument("--output",type=Path,required=True);p.add_argument("--stage",choices=("preparation","generation"),required=True)
    for command in ("generate","collect"):
        p=sub.add_parser(command);p.add_argument("--output",type=Path,required=True);p.add_argument("--authorization",type=Path,required=True)
        if command=="collect":p.add_argument("--condition",choices=("pilot","study"),required=True)
    args=parser.parse_args(argv)
    if args.command=="prepare":
        value=prepare(args.project,args.inputs_project,args.output,args.inventory,hug_root=args.hug_root,checkpoint=args.checkpoint,
            raw_output=args.raw_output,attempts_per_object=args.attempts_per_object,scenario_seed=args.scenario_seed)
        print(json.dumps(dict(status=value["status"],quotas=value["quotas"],preparation_draft=str(args.output/"preparation.draft.json"))))
    elif args.command=="freeze":print(json.dumps(dict(sha256=freeze(args.output,args.stage))))
    elif args.command=="generate":print(json.dumps(dict(status=generate(args.output,args.authorization)["status"])))
    else:collect(args.output,args.authorization,args.condition)


if __name__=="__main__":main()
