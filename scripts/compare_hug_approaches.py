#!/usr/bin/env python3
"""Prepare a frozen paired mug controller comparison; run only with authorization.

All eight freshly generated mug proposals are paired in fixed order, including
mapping refusals. The new condition changes wrist redundancy and approach by
its named static policy. This does not isolate approach direction as one factor.
No native generation, retarget solve, training or outcome-guided repair occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

import numpy as np
import mujoco

import run_hug_build_collection as collection


SCHEMA="hug_paired_mug_approaches_v1"
CONDITIONS=("vertical_v2","grasp_derived_v1")
PROTOCOL_IDS={"vertical_v2":"fixed_grasp_lift_v2","grasp_derived_v1":"hug_grasp_approach_v1"}
sha,digest,write=collection.sha,collection.digest,collection.write


def paired_schedule(proposals):
    mugs=[row for row in proposals if row["object_id"]=="hug_mug"]
    if len(mugs)!=8 or len({row["slot_id"] for row in mugs})!=8:
        raise ValueError("comparison requires all eight fresh mug proposal slots")
    if len({r["capture_id"] for r in mugs})!=2 or {r["generation_seed"] for r in mugs}!=set(collection.GENERATION_SEEDS):
        raise ValueError("the complete two-capture/four-seed mug cohort is required")
    return [dict(attempt_id=f"mug_pair_{pair:02d}_{condition}",pair_id=f"mug_pair_{pair:02d}",condition=condition,
        source_slot_id=row["slot_id"],source_mapping_status=row["mapping_status"],object_id="hug_mug",
        geometry_group_id=row["geometry_group_id"],observation_group_id=row["observation_group_id"],
        proposal_group_id=row["proposal_group_id"],native_sha256=row["native_sha256"],
        canonical_proposal_hash=row.get("proposal_hash"),physical_outcome=None)
        for pair,row in enumerate(sorted(mugs,key=lambda r:r["slot_id"])) for condition in CONDITIONS]


def prepare(collection_output,output,*,raw_output=None):
    source=Path(collection_output).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError("new comparison evidence directory required")
    protocol=collection.read_protocol(source)
    generation_path=source/"generation.json";generated=json.loads(generation_path.read_text())
    if generated.get("status")!="COMPLETED_BOUNDED_GENERATION":raise ValueError("completed frozen fresh generation required")
    collection.verify(generated["artifact_bindings"])
    schedule=paired_schedule(generated["proposals"])
    raw=Path(raw_output or Path(protocol["raw_output"])/"mug_comparison_v1").resolve()
    if raw.exists() or not raw.is_relative_to(Path(protocol["inputs_project"])/"runs") or output==raw or output in raw.parents:
        raise ValueError("new ignored raw directory required")
    source_files={**protocol["source_bindings"],str(Path(__file__).resolve()):sha(__file__)}
    doc=Path(protocol["project"])/"docs/HUG_APPROACH_COMPARISON_V1.md"
    if doc.is_file():source_files[str(doc)]=sha(doc)
    value=dict(schema_version=SCHEMA,status="PREPARED_NOT_AUTHORIZED",collection_output=str(source),raw_output=str(raw),
        source_bindings=source_files,input_bindings={str(source/"preparation.json"):sha(source/"preparation.json"),
            str(generation_path):sha(generation_path),**generated["artifact_bindings"]},
        schedule=schedule,quota=dict(proposals=8,pairs=8,maximum_attempts=16,maximum_steps_per_attempt=3260,maximum_total_steps=52160),
        runtime=collection.runtime(),conditions=dict(vertical_v2="unchanged RecordedActuatedShadowRunner/fixed_grasp_lift_v2; world-negative-Z",
            grasp_derived_v1="GraspApproachRecordedRunner/hug_grasp_approach_v1; static wrist redundancy and signed local-axis policy"),
        nominal_policy="exact fresh captured object mass/pose, imported hand/object/table friction and table placement; zero command and sensor noise",
        immutable_targets="same native hash, canonical joint targets and intended world palm in each pair; only named new condition may adjust WRJ1/WRJ2",
        static_planner="first feasible candidate in frozen policy; no task outcome or predictor used; check before physics and require same plan at execution",
        numerical_policy=protocol["numerical_policy"],
        labels="existing task scorer unchanged; mapping/planner/runtime refusals retain physical_outcome null",
        denominator_policy="planned8/condition; processed,mapped,execution-attempted,completed and successes reported separately; physical rates use completed labels only",
        hard_stop="first unexpected numerical/input/runtime/recording failure; all remaining rows NOT_RUN, no retry/replacement",
        scope="paired descriptive comparison on reused fresh development proposals; controller condition changes wrist and direction; no held-out performance claim")
    output.mkdir(parents=True);write(output/"protocol.draft.json",value)
    return value


def verify_protocol(value):
    if value.get("schema_version")!=SCHEMA or value.get("runtime")!=collection.runtime():raise ValueError("comparison protocol/runtime mismatch")
    if value.get("quota")!=dict(proposals=8,pairs=8,maximum_attempts=16,maximum_steps_per_attempt=3260,maximum_total_steps=52160):
        raise ValueError("exact paired quota changed")
    collection.verify({**value["source_bindings"],**value["input_bindings"]})
    root=Path(value["collection_output"])
    protocol=collection.read_protocol(root);generated=json.loads((root/"generation.json").read_text())
    if value["schedule"]!=paired_schedule(generated["proposals"]):raise ValueError("paired schedule changed")
    return protocol,generated


def freeze(output):
    output=Path(output);value=json.loads((output/"protocol.draft.json").read_text())
    verify_protocol(value);write(output/"protocol.json",value)
    return sha(output/"protocol.json")


def _counts(rows,condition):
    rows=[row for row in rows if row["condition"]==condition]
    completed=[r for r in rows if r.get("physical_outcome") is not None]
    successes=sum(r["physical_outcome"]["success"] for r in completed)
    acquired=sum(r["score"]["acquired_object"] for r in completed)
    collisions=sum(r["physical_outcome"]["failure_type"]=="approach_collision" for r in completed)
    def rate(n,d):return dict(numerator=n,denominator=d,value=None if d==0 else n/d)
    mapped=sum(r["source_mapping_status"]=="ok" for r in rows)
    return dict(planned=len(rows),processed=sum(r["status"]!="NOT_RUN" for r in rows),mapped=mapped,
        planner_refused=sum(r["status"]=="PLANNER_REFUSED" for r in rows),
        execution_attempted=sum(r["execution_status"] in {"completed","runtime_error"} for r in rows),completed=len(completed),
        runtime_errors=sum(r["execution_status"]=="runtime_error" for r in rows),
        approach_collision=collisions,acquired=acquired,successes=successes,
        physical_collision_rate=rate(collisions,len(completed)),physical_acquisition_rate=rate(acquired,len(completed)),
        physical_success_rate=rate(successes,len(completed)),planned_population_success_rate=rate(successes,len(rows)),mapped_success_rate=rate(successes,mapped))


def summarize(rows):
    pairs=[]
    for pair_id in sorted({r["pair_id"] for r in rows}):
        pair={r["condition"]:r for r in rows if r["pair_id"]==pair_id}
        a,b=(pair[c] for c in CONDITIONS)
        both=all(r.get("physical_outcome") is not None for r in (a,b))
        pairs.append(dict(pair_id=pair_id,source_slot_id=a["source_slot_id"],both_completed=both,
            statuses={c:pair[c]["status"] for c in CONDITIONS},
            success_difference=None if not both else int(b["physical_outcome"]["success"])-int(a["physical_outcome"]["success"]),
            collision_difference=None if not both else int(b["physical_outcome"]["failure_type"]=="approach_collision")-int(a["physical_outcome"]["failure_type"]=="approach_collision")))
    return dict(conditions={c:_counts(rows,c) for c in CONDITIONS},pairs=pairs,
        paired_completed=sum(p["both_completed"] for p in pairs),
        interpretation="null refusals/errors are not physical failures; wrist redundancy and approach both change in the named condition")


def _metadata(value,row,*,mapping="ok",execution="not_executed",outcome=None,code=None,hashes=None):
    from grasp_failure_prediction.evaluation.prediction_recording import AttemptMetadata
    provenance=dict(code=digest(value["source_bindings"]),configuration=digest(value),inputs=digest(row))
    provenance.update(hashes or {})
    return AttemptMetadata(attempt_id=row["attempt_id"],hand_id="shadow_hand_right",execution_protocol_id=PROTOCOL_IDS[row["condition"]],
        object_id=row["object_id"],geometry_group_id=row["geometry_group_id"],observation_group_id=row["observation_group_id"],
        proposal_group_id=row["proposal_group_id"],mapping=dict(status=mapping,code=code,message=None if mapping=="ok" else code),
        execution_status=execution,outcome=outcome,provenance_hashes=provenance)


def _null(value,row,path,*,status,mapping,code):
    from grasp_failure_prediction.evaluation.prediction_recording import write_incomplete_attempt
    write_incomplete_attempt(path/"prediction_recording",_metadata(value,row,mapping=mapping,code=code))
    return dict(row,status=status,execution_status="not_executed",physical_outcome=None,physics_steps=0,
        recording_path=str(path/"prediction_recording"),attempt_sha256=sha(path/"prediction_recording/attempt.json"))


def _nominal_reset(runner,context):
    case=context.case
    runner.reset(seed=case.seed,object_mass_kg=case.object.mass_kg,
        object_position_m=np.asarray(case.initial_condition.object_position_m),
        object_orientation_wxyz=np.asarray(case.initial_condition.object_orientation_xyzw)[[3,0,1,2]])
    captured=mujoco.MjModel.from_binary_path(str(context.proposal_path.parent/"native/scene.mjb"))
    obj=runner._object_body_id
    captured_obj=mujoco.mj_name2id(captured,mujoco.mjtObj.mjOBJ_BODY,"object")
    if captured_obj<0 or captured.ngeom!=runner.model.ngeom:raise ValueError("captured nominal geometry contract differs")
    if (not np.array_equal(runner.model.geom_friction,captured.geom_friction)
            or not np.array_equal(runner.model.geom_pos[runner._table_geom_id],captured.geom_pos[runner._table_geom_id])
            or abs(float(runner.model.body_mass[obj])-float(captured.body_mass[captured_obj]))>1e-12
            or not np.allclose(runner.model.body_inertia[obj],captured.body_inertia[captured_obj],rtol=0,atol=1e-12)):
        raise ValueError("comparison must preserve captured nominal mass, inertia, friction and table")
    return dict(object_mass_kg=float(runner.model.body_mass[obj]),object_inertia_diagonal=runner.model.body_inertia[obj].tolist(),
        all_geom_friction=runner.model.geom_friction.tolist(),table_geom_position_m=runner.model.geom_pos[runner._table_geom_id].tolist(),
        object_position_m=list(case.initial_condition.object_position_m),object_orientation_xyzw=list(case.initial_condition.object_orientation_xyzw),
        sensor_noise_std_rad=0.,command_noise_rad=0.,capture_model_sha256=sha(context.proposal_path.parent/"native/scene.mjb"))


def _finger_identity(original,adjusted):
    before=dict(zip(original.joint_names,original.qpos,strict=True));after=dict(zip(adjusted.joint_names,adjusted.qpos,strict=True))
    if set(before)!=set(after) or any(before[name]!=after[name] for name in before if name not in {"WRJ1","WRJ2"}):
        raise ValueError("new controller altered original finger targets")


def _attempt(value,protocol,proposal,row,path):
    from grasp_failure_prediction.evaluation import scoring
    from grasp_failure_prediction.evaluation.approach_execution import plan_grasp_approach
    from grasp_failure_prediction.evaluation.recorded_runner import RecordedActuatedShadowRunner
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt,write_incomplete_attempt
    from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
    from run_hug_saved_recording_pilot import _save_partial
    path.mkdir(parents=True);runner=None;budget=dict(total=0);execution_started=False
    hashes=dict(native_pose=row["native_sha256"],mapped_pose=row["canonical_proposal_hash"],assets=digest(protocol["asset_bindings"]))
    try:
        context,pose,palm,quaternion=collection._case_and_context(protocol,proposal,path)
        if row["condition"]=="grasp_derived_v1":runner=collection._new_runner(protocol,context)
        else:
            sim=context.resolved.environment.simulator
            runner=RecordedActuatedShadowRunner(context.resolved.execution_protocol,default_dex_urdf_root(),
                physics_timestep_s=sim.physics_timestep_s,control_timestep_s=sim.control_timestep_s,object_geometry=context.geometry)
        realization=_nominal_reset(runner,context)
        realization.update(intended_world_palm_position_m=palm.tolist(),intended_world_palm_quaternion_wxyz=quaternion.tolist(),
            canonical_joint_targets_rad=dict(zip(pose.joint_names,pose.qpos.tolist(),strict=True)))
        write(path/"nominal_realization.json",realization);hashes["realization"]=sha(path/"nominal_realization.json")
        plan_record=None
        if row["condition"]=="grasp_derived_v1":
            plan=plan_grasp_approach(runner,pose,palm,quaternion,clearance=runner.clearance_configuration,approach=runner.approach_configuration)
            plan_record=plan.as_record();write(path/"approach_plan.json",plan_record)
            if not plan.feasible:return _null(value,row,path,status="PLANNER_REFUSED",mapping="unreachable_pregrasp",code="frozen_static_planner_refusal")
            _finger_identity(pose,plan.adjusted_pose)
            hashes["approach_plan"]=sha(path/"approach_plan.json")
        hashes["label_rule"]=digest(dict(scorer=sha(Path(scoring.__file__)),success=runner.protocol.success.model_dump(mode="json"),control_timestep_s=runner.control_timestep_s))
        hashes["execution_configuration"]=digest(runner.protocol.model_dump(mode="json"))
        mujoco.mj_saveModel(runner.model,str(path/"scene.mjb"));hashes["model"]=sha(path/"scene.mjb")
        context.verify_bindings();collection.verify({**value["source_bindings"],**value["input_bindings"]})
        with collection.physics_budget(value["quota"]["maximum_steps_per_attempt"]) as budget:
            execution_started=True
            kwargs=dict(grasp_palm_position_m=palm,grasp_palm_quaternion_wxyz=quaternion)
            if row["condition"]=="vertical_v2":kwargs["approach_direction_world"]=np.array([0.,0.,-1.])
            trace=runner.execute(pose,**kwargs)
        if len(trace.steps)!=163 or budget["total"]!=3260:raise RuntimeError("exact completed task endpoint/step quota differs")
        if plan_record is not None and runner.approach_plan.as_record()!=plan_record:raise RuntimeError("static new-condition plan changed at execution")
        score=asdict(scoring.score_trace(trace,runner.protocol,control_timestep_s=runner.control_timestep_s))
        failure=score["failure_type"];failure=None if failure is None else failure.value
        first=next((s.time_s-runner.physics_timestep_s for s in trace.steps if s.hand_table_contact and s.state.value in {"move_to_pregrasp","approach"}),None)
        outcome=dict(success=score["success"],failure_type=failure,label_rule_id=PROTOCOL_IDS[row["condition"]]+"_task_outcome_v1",
            first_failure_time_s=first if failure=="approach_collision" else None)
        context.verify_bindings();collection.verify({**value["source_bindings"],**value["input_bindings"]})
        pending=path/"_pending_recording"
        runner.prediction_recorder.write(pending,_metadata(value,row,execution="completed",outcome=outcome,hashes=hashes))
        load_attempt(pending);pending.rename(path/"prediction_recording")
        result=dict(row,status="COMPLETED",execution_status="completed",physical_outcome=outcome,score=score,physics_steps=budget["total"],
            recording_path=str(path/"prediction_recording"),attempt_sha256=sha(path/"prediction_recording/attempt.json"),
            nominal_realization_sha256=hashes["realization"])
        write(path/"result.json",result)
        return result
    except Exception as error:
        (path/"exception_traceback.txt").write_text(traceback.format_exc())
        for name in ("result.json","_pending_recording","prediction_recording"):
            old=path/name
            if old.exists():old.rename(path/("quarantined_"+name))
        for old in path.rglob("attempt.json"):old.rename(old.with_name("quarantined_attempt.json"))
        if runner is not None:
            try:_save_partial(path,runner.prediction_recorder,runner.data)
            except Exception as partial:write(path/"partial_snapshot_error.json",dict(type=type(partial).__name__,message=str(partial)))
        execution="runtime_error" if execution_started else "not_executed";mapping="ok" if execution_started else "setup_error"
        write_incomplete_attempt(path/"prediction_recording",_metadata(value,row,mapping=mapping,execution=execution,
            code=None if execution_started else type(error).__name__,hashes=hashes))
        return dict(row,status="ERROR",execution_status=execution,physical_outcome=None,error=dict(type=type(error).__name__,message=str(error)),
            physics_steps=budget["total"],recording_path=str(path/"prediction_recording"),attempt_sha256=sha(path/"prediction_recording/attempt.json"))


def run(output,authorization):
    output=Path(output).resolve();value=json.loads((output/"protocol.json").read_text())
    protocol,generated=verify_protocol(value)
    auth=json.loads(Path(authorization).read_text())
    if (auth.get("schema_version")!="hug_paired_approach_authorization_v1" or auth.get("authorized") is not True
            or auth.get("protocol_sha256")!=sha(output/"protocol.json") or auth.get("quota")!=16):
        raise ValueError("explicit comparison authorization must bind frozen protocol and exact16 attempts")
    raw=Path(value["raw_output"])
    if raw.exists() or (output/"summary.json").exists():raise FileExistsError("comparison cannot retry or overwrite")
    raw.mkdir(parents=True);rows=[];error=None;steps=0;started=time.monotonic()
    proposals={r["slot_id"]:r for r in generated["proposals"]}
    for row in value["schedule"]:
        path=raw/row["attempt_id"]
        if error:result=_null(value,row,path,status="NOT_RUN",mapping="setup_error",code="not_run_hard_stop")
        elif row["source_mapping_status"]!="ok":result=_null(value,row,path,status="MAPPING_REFUSED",mapping=row["source_mapping_status"],code="fresh_mapping_refused")
        else:result=_attempt(value,protocol,proposals[row["source_slot_id"]],row,path)
        rows.append(result);steps+=result["physics_steps"]
        if result["status"]=="ERROR":error=result["error"]
        print(json.dumps(dict(attempt_id=row["attempt_id"],condition=row["condition"],status=result["status"])),flush=True)
    report=dict(schema_version=SCHEMA,status="HARD_STOP" if error else "COMPLETED_PAIRED_QUOTA",error=error,
        protocol_sha256=sha(output/"protocol.json"),schedule=rows,physics_steps=steps,elapsed_wall_s=time.monotonic()-started,
        analysis=summarize(rows))
    try:
        verify_protocol(value)
        if steps>value["quota"]["maximum_total_steps"]:raise RuntimeError("comparison total step cap exceeded")
    except Exception as fault:
        (raw/"final_verification_traceback.txt").write_text(traceback.format_exc())
        report.update(status="HARD_STOP",error=dict(type=type(fault).__name__,message=str(fault)))
    report["raw_artifact_bindings"]={str(p):sha(p) for p in raw.rglob("*") if p.is_file()}
    write(output/"summary.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare");p.add_argument("--collection-output",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--raw-output",type=Path)
    p=sub.add_parser("freeze");p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("run");p.add_argument("--output",type=Path,required=True);p.add_argument("--authorization",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=="prepare":print(json.dumps(dict(status=prepare(args.collection_output,args.output,raw_output=args.raw_output)["status"])))
    elif args.command=="freeze":print(json.dumps(dict(sha256=freeze(args.output))))
    else:print(json.dumps(dict(status=run(args.output,args.authorization)["status"])))


if __name__=="__main__":main()
