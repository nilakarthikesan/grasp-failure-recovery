#!/usr/bin/env python3
"""Prepare and run an immutable post-training reserve evaluation; never fit.

Only bell_pepper and coffee_cup's already qualified two captures are admitted.
Every COMPLETE fold/seed bundle is bound before fresh reserve inference/labels.
summary_logistic is primary; all other fixed bundles are diagnostics, never a
reserve-selected winner. Preparation performs no inference or integration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

import numpy as np
import mujoco

import run_hug_build_collection as collector


SCHEMA = "hug_reserve_evaluation_protocol_v1"
RESERVES = ("bell_pepper", "coffee_cup")
SEEDS = (9000, 9001)
PRIMARY = "summary_logistic"
COFFEE_FREEZE = "464f08fd77a90419d17a75a420ae80fd5b16895c"
MODEL_KINDS = ("training_prevalence", "commands_only_logistic", "summary_logistic",
               "full_history_logistic", "small_gru")
EXTRA_SOURCES = ("scripts/evaluate_hug_reserve_models.py",
                 "src/grasp_failure_prediction/prediction/recorded_training.py",
                 "src/grasp_failure_prediction/prediction/model.py")


class ReserveBlocked(ValueError):
    """No prospective inference/labels may start without eligible frozen models."""


def _json(path):
    return json.loads(Path(path).read_text())


def _files(root):
    return {str(path.resolve()): collector.sha(path) for path in sorted(Path(root).rglob("*")) if path.is_file()}


def _require_commit(paths):
    paths = [Path(p).resolve() for p in paths]
    root = Path(subprocess.check_output(["git", "-C", str(paths[0].parent), "rev-parse", "--show-toplevel"],
                                        text=True, stderr=subprocess.PIPE).strip()).resolve()
    try:
        relative = [str(p.relative_to(root)) for p in paths]
        subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", "--", *relative],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *relative],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (ValueError, subprocess.CalledProcessError) as error:
        raise ValueError("commit the exact reserve declaration/ledger and bound source before running") from error
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _fit_commit_blob_sha(project, commit, relative):
    """Read immutable fitting bytes, without changing checkout or Git state."""
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=project,
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import hashlib
    return "sha256:" + hashlib.sha256(result.stdout).hexdigest()


def verify_fitted_source_closure(project, report, training_protocol):
    """Bind ALL source dependencies declared at fitting, not current substitutes."""
    if training_protocol is None:
        raise ValueError("the exact committed training protocol is required for reserve admission")
    project, path = Path(project).resolve(), Path(training_protocol).resolve()
    declaration = _json(path)
    commit = report.get("frozen_protocol_receipt", {}).get("commit", "")
    if (not re.fullmatch(r"[0-9a-f]{40}", commit)
            or declaration.get("schema_version") != "hug_recorded_training_protocol_v1"
            or collector.sha(path) != report.get("protocol_sha256")
            or declaration.get("specification") != report["specification"]
            or declaration.get("source_bindings") != report["source_bindings"]
            or declaration.get("input_contract") != report["input_contract"]
            or declaration.get("requires_commit_before_fit") is not True
            or report.get("frozen_protocol_receipt", {}).get("protocol_and_bound_sources_clean_at_HEAD") is not True
            or not declaration.get("source_files")):
        raise ValueError("training protocol/receipt does not match the frozen fitted report")
    if not path.is_relative_to(project):
        raise ValueError("provide the original training protocol path inside its source repository")
    protocol_relative = path.relative_to(project).as_posix()
    if _fit_commit_blob_sha(project, commit, protocol_relative) != report["protocol_sha256"]:
        raise ValueError("training protocol differs from its immutable fitting commit")
    for relative, expected in declaration["source_files"].items():
        source = (project / relative).resolve()
        if (Path(relative).is_absolute() or not source.is_relative_to(project)
                or collector.sha(source) != expected
                or _fit_commit_blob_sha(project, commit, relative) != expected):
            raise ValueError(f"frozen fitted source closure changed: {relative}")
    return dict(commit=commit, protocol_path=str(path), protocol_sha256=report["protocol_sha256"],
                source_files=declaration["source_files"])


def expected_reserve_cohort(project):
    from grasp_failure_prediction.evaluation.registry import load_protocol_registry, load_environment_registry
    base = load_protocol_registry().resolve("fixed_grasp_lift_v2")
    environment = load_environment_registry().resolve("adroit_shadow_tabletop_v2")
    return dict(hand_id="shadow_hand_right", execution_protocol_id="hug_grasp_approach_v1",
                label_rule_id="hug_grasp_approach_task_outcome_v1",
                label_rule_sha256=collector.digest(dict(scorer=collector.sha(Path(project) / "src/grasp_failure_prediction/evaluation/scoring.py"),
                    success=base.success.model_dump(mode="json"),control_timestep_s=environment.simulator.control_timestep_s)))


def verify_training_cohort(project, report):
    """The same feature shape cannot substitute for a physical target contract."""
    bindings = report["source_bindings"].get("input_files", {})
    collector.verify(bindings)
    manifests = [(Path(path), _json(path)) for path in bindings if Path(path).suffix == ".json"
                 and _json(path).get("schema_version") == "hug_recorded_prefix_dataset_v1"]
    expected = expected_reserve_cohort(project)
    if (len(manifests) != 1 or collector.sha(manifests[0][0]) != report["source_bindings"].get("dataset_manifest_sha256")
            or manifests[0][1].get("cohort") != expected
            or manifests[0][1].get("input_contract") != report["input_contract"]):
        raise ValueError("frozen training cohort hand/controller/label rule differs from reserve target")
    return expected


def complete_model_bundles(report_path):
    """Include every completed bundle, without inspecting comparative scores."""
    path = Path(report_path).resolve()
    report = _json(path)
    if (report.get("schema_version") != "hug_recorded_model_comparison_v1"
            or report.get("status") not in {"completed", "completed_with_skips", "completed_with_model_errors"}):
        raise ReserveBlocked("BLOCKED: a completed recorded_training report is required")
    if report["specification"]["warning_threshold"] != .5:
        raise ValueError("reserve threshold is fixed at0.5")
    contract, bindings, bundles, omitted, seen = report["input_contract"], {str(path): collector.sha(path)}, [], [], set()
    for fold in report["folds"]:
        expected = {(name, seed) for name in MODEL_KINDS for seed in (0, 1, 2)}
        if {(item["name"], item["seed"]) for item in fold["models"]} != expected:
            raise ValueError("every training fold must account for all five kinds and three seeds")
        for item in fold["models"]:
            identity = (fold["fold_id"], item["name"], item["seed"])
            if identity in seen or item["name"] not in MODEL_KINDS or item["seed"] not in (0, 1, 2):
                raise ValueError("unknown or repeated model kind/fold/seed")
            seen.add(identity)
            if item["status"] != "completed":
                omitted.append(dict(fold_id=identity[0], kind=identity[1], seed=identity[2], status=item["status"], reason=item.get("reason")))
                continue
            directory = (path.parent / item["model_directory"]).resolve()
            if not directory.is_relative_to(path.parent):
                raise ValueError("model bundle escapes its report directory")
            metadata_path = directory / "model.json"
            if collector.sha(metadata_path) != item["model_metadata_sha256"]:
                raise ValueError("completed model metadata hash mismatch")
            metadata = _json(metadata_path)
            if (metadata.get("schema_version") != "hug_recorded_predictor_v1" or metadata["kind"] != identity[1]
                    or metadata["seed"] != identity[2] or metadata["input_contract"] != contract
                    or metadata["selection_sha256"] != item["selection_sha256"]
                    or collector.digest(metadata["selection"]) != item["selection_sha256"]
                    or set(metadata["artifacts"]) != {"model.npz", "normalizer.npz"}):
                raise ValueError("completed model selection/input contract differs from training report")
            current = {str(metadata_path): item["model_metadata_sha256"],
                       str(directory / "predictions.json"): item["predictions_sha256"],
                       **{str(directory / name): value for name, value in metadata["artifacts"].items()}}
            collector.verify(current)
            bindings.update(current)
            bundles.append(dict(bundle_id=f"bundle_{len(bundles):04d}", fold_id=identity[0], kind=identity[1], seed=identity[2],
                                directory=str(directory), role="primary" if identity[1] == PRIMARY else "diagnostic"))
    if not bundles:
        raise ReserveBlocked("BLOCKED: zero complete frozen model bundles; no reserve inference or task labels")
    if not any(bundle["kind"] == PRIMARY for bundle in bundles):
        raise ReserveBlocked("BLOCKED: predeclared primary summary_logistic has no complete bundle")
    return report, dict(report_path=str(path), bundles=bundles, omitted=omitted, bindings=bindings, input_contract=contract)


def qualified_captures(name, protocol_path, capture_root, inputs_project):
    if name not in RESERVES:
        raise ValueError("only predeclared bell_pepper and coffee_cup reserves; bear stays excluded")
    protocol_path, capture_root = Path(protocol_path).resolve(), Path(capture_root).resolve()
    protocol, report = _json(protocol_path), _json(capture_root / "capture_qualification.json")
    if (protocol.get("schema_version") != "hug_reserve_asset_qualification_v1"
            or report["protocol_sha256"] != collector.sha(protocol_path)
            or any(report.get(key) != 0 for key in ("grasp_inference_calls", "grasp_labels", "model_fits"))
            or Path(protocol["runtime_project"]).resolve() != inputs_project):
        raise ValueError("reserve qualification does not bind unused captures in the expected runtime project")
    if name == "coffee_cup" and report["freeze_commit"] != COFFEE_FREEZE:
        raise ValueError("coffee reserve differs from its concrete464f08f qualification freeze")
    object_id = "hug_" + name
    captures = [row for row in report["captures"] if row["object_id"] == object_id]
    if (len(captures) != 2 or {row["camera_id"] for row in captures} != {"front_left", "front_right"}
            or any(row["status"] != "PASS_STATIONARY_CAPTURE" or row.get("outcome") is not None for row in captures)):
        raise ValueError("each admitted reserve needs exactly two qualified captures with null grasp labels")
    case_file = "case_" + name + ".json"
    case_path = protocol_path.parent / case_file
    if collector.sha(case_path) != protocol["cases"][case_file]:
        raise ValueError("qualified reserve case hash mismatch")
    asset = next(row for row in protocol["objects"] if row["object_id"] == object_id)
    asset_root = Path(protocol["asset_root"]) / name
    asset_bindings = {str((asset_root / relative).resolve()): value
                      for relative, value in asset["geometry_files_sha256"].items()}
    if "download_manifest_sha256" in asset:
        asset_bindings[str(asset_root / "download_manifest.json")] = asset["download_manifest_sha256"]
    collector.verify(asset_bindings)
    bindings = {str(protocol_path): collector.sha(protocol_path), str(case_path): collector.sha(case_path),
                str(capture_root / "capture_qualification.json"): collector.sha(capture_root / "capture_qualification.json")}
    slots = []
    for row in sorted(captures, key=lambda item: item["capture_id"]):
        capture = capture_root / row["capture_id"]
        if not capture.is_relative_to(inputs_project / "runs"):
            raise ValueError("reserve raw capture must remain in the private runtime runs directory")
        if (collector.sha(capture / "capture_report.json") != row["capture_report_sha256"]
                or (capture / "proposal.pkl").exists() or (capture / "inference_report.json").exists()):
            raise ValueError("reserve capture changed or already contains a grasp inference")
        contract = _json(capture / "scene_contract.json")
        if (contract["object_id"] != object_id or contract["geometry_content_hash"] != asset["geometry_content_hash"]
                or collector.sha(capture / "scene_contract.json") != row["capture_report"]["scene_contract_sha256"]):
            raise ValueError("reserve capture scene does not match qualified geometry")
        bindings.update(_files(capture))
        slots.append(dict(capture_id=row["capture_id"], observation_path=str(capture),
                          object_id=object_id, camera_id=row["camera_id"]))
    obj = dict(object_id=object_id, category_id="reserve_" + name, source_case=_json(case_path),
               geometry_content_hash=asset["geometry_content_hash"], qualification_freeze=report["freeze_commit"],
               qualification_freeze_is_concrete=report["freeze_commit"] != "HEAD", asset_bindings=asset_bindings)
    return obj, slots, bindings


def prepare(project, main_preparation, model_report, qualifications, output, *, training_protocol=None, raw_output=None):
    """Read/hash only; actual model bundles must already exist before declaration."""
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("fresh reserve evidence directory required")
    report, models = complete_model_bundles(model_report)  # blocks before any side effect.
    main_path = Path(main_preparation).resolve()
    main, main_generation = _json(main_path), _json(main_path.parent / "generation.json")
    if (main.get("schema_version") != collector.SCHEMA
            or main_generation.get("status") != "COMPLETED_BOUNDED_GENERATION"
            or main_generation["preparation_sha256"] != collector.sha(main_path)):
        raise ValueError("reserve must reuse a completed, frozen main-study generator/controller declaration")
    collector.verify({**main["source_bindings"], **main["input_bindings"], **main["asset_bindings"]})
    collector.bootstrap(main)
    trainer_path = project / "src/grasp_failure_prediction/prediction/recorded_training.py"
    if report.get("training_source_sha256") != collector.sha(trainer_path):
        raise ValueError("current predictor implementation differs from the frozen training source")
    fitted_sources = verify_fitted_source_closure(project, report, training_protocol)
    cohort = verify_training_cohort(project, report)
    inputs = Path(main["inputs_project"]).resolve()
    if set(qualifications) != set(RESERVES):
        raise ValueError("declare exactly bell_pepper and coffee_cup qualification pairs")
    source = {str(project / name): collector.sha(project / name) for name in (*collector.SOURCE_FILES, *EXTRA_SOURCES)}
    bindings = {str(main_path): collector.sha(main_path), str(main_path.parent / "generation.json"): collector.sha(main_path.parent / "generation.json"),
                str(Path(training_protocol).resolve()): collector.sha(training_protocol)}
    objects, captures, assets = [], [], dict(main["asset_bindings"])
    for name in RESERVES:
        obj, slots, bound = qualified_captures(name, *qualifications[name], inputs)
        objects.append(obj); captures.extend(slots); bindings.update(bound)
        assets.update(obj["asset_bindings"])
        bindings.update({str(path): collector.sha(path) for path in collector._geometry_paths(inputs, obj["source_case"]["object"]["geometry"])})
    # Confirm these geometries/objects were absent from ALL fitting/tuning/test data.
    training_bindings = report["source_bindings"].get("input_files", {})
    if not training_bindings:
        raise ValueError("training report lacks its file-backed source schedule")
    collector.verify(training_bindings)
    schedules = [_json(path) for path in training_bindings if Path(path).suffix == ".json"
                 and _json(path).get("schema_version") == "hug_scenario_schedule_v1"]
    if len(schedules) != 1 or collector.digest(schedules[0]) != report["source_bindings"]["schedule_sha256"]:
        raise ValueError("training schedule not uniquely hash-bound")
    old_objects = {row["object_id"] for row in schedules[0]["scenarios"]}
    old_geometries = {row["geometry_group_id"] for row in schedules[0]["scenarios"]}
    if any(obj["object_id"] in old_objects or collector.digest({"geometry_content_hash": obj["geometry_content_hash"]}) in old_geometries for obj in objects):
        raise ValueError("reserve object/geometry overlaps model development data")
    slots = [dict(**capture, slot_id=f"reserve_proposal_{i:04d}", generation_seed=seed)
             for i, (capture, seed) in enumerate((capture, seed) for capture in captures for seed in SEEDS)]
    raw = Path(raw_output or inputs / "runs" / output.name).resolve()
    if raw.exists() or not raw.is_relative_to(inputs / "runs") or raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("fresh reserve raw output must be separate under private runs/")
    prefix = main["prefix"]
    contract = models["input_contract"]
    if any(contract[key] != prefix[key] for key in prefix):
        raise ValueError("trained input contract differs from the fixed collector prefix")
    declaration = dict(schema_version=SCHEMA, status="PREPARED_NOT_AUTHORIZED", project=str(project), inputs_project=str(inputs),
        main_generation_path=str(main_path.parent / "generation.json"),
        fitted_source_closure=fitted_sources, expected_cohort=cohort,
        raw_output=str(raw), source_bindings=source, input_bindings={**main["input_bindings"], **bindings, **training_bindings},
        asset_bindings=assets, dependency_dirs=main["dependency_dirs"], runtime=collector.runtime(),
        hug_root=main["hug_root"], checkpoint=main["checkpoint"], objects=objects, generation_slots=slots,
        generation_seeds=list(SEEDS), expected_loaded_model_state_sha256=main_generation["loaded_model_state_sha256"],
        models=models, primary_model=PRIMARY, warning_threshold=.5, calibration_bins=10,
        quotas=dict(native_proposals=8, task_attempts=32, maximum_task_steps_per_attempt=3260, maximum_total_task_steps=104320),
        prefix=prefix, axes=dict(mass_multipliers=[.75, 1.25], friction_coefficients=[.5, 1.], table_offsets_m=[0.], sensor_noise_std_rad=[0.]),
        fit_allowed=False, threshold_selection_allowed=False, reserve_winner_selection_allowed=False,
        filter_policy="not run; retain all mapped proposals and all declared static refusals",
        retry_policy="no retries/replacements/repairs; bear remains excluded; unexpected fault hard-stops remaining slots/attempts",
        scope="two previously unused qualified geometries/captures; no broad category-generalization claim")
    output.mkdir(parents=True)
    collector.write(output / "protocol.draft.json", declaration)
    return declaration


def _verify_protocol(protocol):
    if (protocol.get("schema_version") != SCHEMA or protocol.get("warning_threshold") != .5
            or protocol.get("primary_model") != PRIMARY or protocol.get("fit_allowed") is not False
            or protocol.get("threshold_selection_allowed") is not False
            or protocol.get("reserve_winner_selection_allowed") is not False
            or protocol.get("runtime") != collector.runtime()):
        raise ValueError("reserve declaration schema/runtime/fixed model policy changed")
    if not protocol["models"]["bundles"]:
        raise ReserveBlocked("BLOCKED: no complete model bundles")
    collector.verify({**protocol["source_bindings"], **protocol["input_bindings"], **protocol["asset_bindings"], **protocol["models"]["bindings"]})
    collector.bootstrap(protocol)  # Before registry/Pydantic imports from the bound project.
    fitted_report, expected_models = complete_model_bundles(protocol["models"]["report_path"])
    if protocol["models"] != expected_models:
        raise ValueError("reserve model inventory differs from every complete frozen bundle")
    if (verify_fitted_source_closure(protocol["project"], fitted_report, protocol["fitted_source_closure"]["protocol_path"])
            != protocol["fitted_source_closure"] or verify_training_cohort(protocol["project"], fitted_report) != protocol["expected_cohort"]):
        raise ValueError("reserve fitted source/physical target cohort changed")
    slots = protocol["generation_slots"]
    if (len(slots) != 8 or len({row["slot_id"] for row in slots}) != 8
            or {row["object_id"] for row in slots} != {"hug_" + name for name in RESERVES}
            or len({row["capture_id"] for row in slots}) != 4
            or any({row["generation_seed"] for row in slots if row["capture_id"] == capture} != set(SEEDS)
                   for capture in {row["capture_id"] for row in slots})
            or protocol["quotas"] != dict(native_proposals=8, task_attempts=32, maximum_task_steps_per_attempt=3260, maximum_total_task_steps=104320)
            or protocol["axes"] != dict(mass_multipliers=[.75,1.25], friction_coefficients=[.5,1.], table_offsets_m=[0.],sensor_noise_std_rad=[0.])):
        raise ValueError("reserve fixed8/32 generation/scenario budget changed")
    if protocol["expected_loaded_model_state_sha256"] != _json(protocol["main_generation_path"])["loaded_model_state_sha256"]:
        raise ValueError("reserve generator identity differs from main frozen generation")


def read_protocol(output, *, require_commit=True):
    path = Path(output).resolve() / "protocol.json"
    protocol = _json(path); _verify_protocol(protocol)
    if require_commit:
        _require_commit([path, *protocol["source_bindings"]])
    return protocol


def freeze(output, stage):
    output = Path(output).resolve()
    if stage == "protocol":
        value = _json(output / "protocol.draft.json"); _verify_protocol(value)
    else:
        protocol = read_protocol(output)
        value = _json(output / "generation.draft.json")
        if value["status"] != "COMPLETED_BOUNDED_GENERATION" or value["protocol_sha256"] != collector.sha(output / "protocol.json"):
            raise ValueError("cannot freeze incomplete/foreign reserve generation")
        collector.verify(value["artifact_bindings"])
        if value["schedule"] != reserve_schedule(value["proposals"]):
            raise ValueError("reserve schedule differs from fixed full-factorial declaration")
    collector.write(output / (stage + ".json"), value)
    return collector.sha(output / (stage + ".json"))


def _authorized(output, path, stage, quota, extra=None):
    declaration = _json(path)
    expected = {str((Path(output) / "protocol.json").resolve()): collector.sha(Path(output) / "protocol.json"), **(extra or {})}
    if (declaration.get("schema_version") != "hug_reserve_authorization_v1" or declaration.get("authorized") is not True
            or declaration.get("stage") != stage or declaration.get("quota") != quota or declaration.get("bindings") != expected):
        raise ValueError("explicit reserve authorization does not match exact stage/protocol/quota")
    collector.verify(expected)


def reserve_schedule(proposals):
    if len(proposals) != 8:
        raise ValueError("exactly eight reserve generation rows required")
    rows = []
    for proposal in proposals:
        for mass in (.75, 1.25):
            for friction in (.5, 1.):
                row = dict(attempt_id=f"reserve_attempt_{len(rows):04d}", generation_slot_id=proposal["slot_id"],
                    object_id=proposal["object_id"], geometry_group_id=proposal["geometry_group_id"],
                    observation_group_id=proposal["observation_group_id"], proposal_group_id=proposal["proposal_group_id"],
                    generation_seed=proposal["generation_seed"], mass_multiplier=mass, friction_coefficient=friction,
                    table_offset_m=0., sensor_noise_std_rad=0., sensor_noise_seed=0, approach_mode="grasp_derived")
                rows.append(dict(scenario_id="reserve_scenario_" + collector.digest(row)[7:23], **row))
    return dict(schema_version="hug_reserve_scenario_schedule_v1", scenarios=rows, outcome_selection=False,
                counts=dict(attempts=32, objects=2, observations=len({r["observation_group_id"] for r in rows}),
                            proposal_families=len({r["proposal_group_id"] for r in rows})))


def generate(output, authorization):
    output = Path(output).resolve(); protocol = read_protocol(output)
    _authorized(output, authorization, "generate", 8)
    raw = Path(protocol["raw_output"])
    if raw.exists() or (output / "generation.draft.json").exists():
        raise FileExistsError("reserve inference cannot retry/overwrite an earlier run")
    raw.mkdir(parents=True)
    from grasp_failure_prediction.proposals.generate import generate_proposals, _expected_input_binding
    from grasp_failure_prediction.proposals.schema import ProposalRequest
    rows, error = [], None
    for slot in protocol["generation_slots"]:
        row = dict(**slot, status="NOT_RUN", mapping_status="setup_error", outcome=None)
        if error:
            rows.append(row); continue
        directory = raw / "generation" / slot["slot_id"]
        try:
            capture = Path(slot["observation_path"]); contract = _json(capture / "scene_contract.json")
            request = ProposalRequest.model_validate(dict(request_id=slot["slot_id"], provider=dict(id="hug_mano_shadow_v1"),
                object=dict(id=slot["object_id"], geometry_content_hash=contract["geometry_content_hash"]),
                observation_path=str(capture.relative_to(protocol["inputs_project"])), seed=slot["generation_seed"], sample_count=1))
            request_path = raw / "requests" / (slot["slot_id"] + ".json"); collector.write(request_path, request.model_dump(mode="json"))
            binding = _expected_input_binding(request, Path(protocol["inputs_project"]))
            row.update(geometry_group_id=binding["geometry_group_id"], observation_group_id=binding["observation_group_id"])
            try:
                with collector.physics_budget(0):
                    manifest = generate_proposals(request_path, directory, project_root=protocol["inputs_project"],
                                                  hug_root=protocol["hug_root"], checkpoint=protocol["checkpoint"],
                                                  runtime_dependency_dirs=protocol["dependency_dirs"])
            except Exception:
                record_path = directory / "sample_0000/native/mapping_record.json"
                mapping = _json(record_path) if record_path.is_file() else {}
                if mapping.get("mapping_status") not in {"refused_guard", "refused_invalid_retarget"}:
                    raise
                native = record_path.parent / "proposal.pkl"
                row.update(status="MAPPING_REFUSED", mapping_status=mapping["mapping_status"], native_sha256=collector.sha(native),
                    proposal_group_id=collector.digest(dict(native_pose_sha256=collector.sha(native),
                        observation_group_id=row["observation_group_id"], geometry_group_id=row["geometry_group_id"])))
                validation = collector._validate_fresh(protocol, record_path.parent.parent / "proposal.json")
            else:
                proposal_path = directory / manifest.proposals[0].proposal_path
                validation = collector._validate_fresh(protocol, proposal_path)
                row = collector._proposal_row(protocol, slot, proposal_path, validation)
            state = collector._loaded_state_hash(directory)
            if state != protocol["expected_loaded_model_state_sha256"]:
                raise ValueError("reserve inference loaded a different generator model state")
            row.update(loaded_model_state_sha256=state, mapping_validation=validation)
            collector.write(directory / "mapping_validation.json", validation)
            _verify_protocol(protocol)
        except Exception as fault:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "exception_traceback.txt").write_text(traceback.format_exc())
            row.update(status="ERROR", error=dict(type=type(fault).__name__, message=str(fault)))
            error = row["error"]
        rows.append(row)
    result = dict(schema_version="hug_reserve_generation_v1", status="HARD_STOP" if error else "COMPLETED_BOUNDED_GENERATION",
        protocol_sha256=collector.sha(output / "protocol.json"), proposals=rows, error=error, artifact_bindings=_files(raw),
        counts={status: sum(row["status"] == status for row in rows) for status in sorted({r["status"] for r in rows})})
    if not error:
        result["schedule"] = reserve_schedule(rows)
    collector.write(output / "generation.draft.json", result)
    return result


def collect(output, authorization):
    output = Path(output).resolve(); protocol = read_protocol(output)
    generation_path = output / "generation.json"; generation = _json(generation_path)
    _require_commit([generation_path])
    if (generation["status"] != "COMPLETED_BOUNDED_GENERATION"
            or generation["protocol_sha256"] != collector.sha(output / "protocol.json")
            or generation["schedule"] != reserve_schedule(generation["proposals"])):
        raise ValueError("reserve labels require exact complete frozen generation and schedule")
    collector.verify(generation["artifact_bindings"])
    _authorized(output, authorization, "collect", 32, {str(generation_path): collector.sha(generation_path)})
    raw = Path(protocol["raw_output"]) / "tasks"
    if raw.exists() or (output / "collection.json").exists():
        raise FileExistsError("reserve task collection cannot retry/overwrite")
    raw.mkdir(parents=True)
    proposals = {row["slot_id"]: row for row in generation["proposals"]}
    rows, error, steps = [], None, 0
    for scenario in generation["schedule"]["scenarios"]:
        proposal = proposals[scenario["generation_slot_id"]]
        directory = raw / scenario["attempt_id"]
        if error:
            row = collector._null_attempt(protocol, proposal, scenario, directory, mapping="setup_error", code="not_run_hard_stop", status="NOT_RUN")
        elif proposal["mapping_status"] != "ok":
            row = collector._null_attempt(protocol, proposal, scenario, directory, mapping=proposal["mapping_status"], code="reserve_mapping_refused", status="MAPPING_REFUSED")
        else:
            row = collector._collect_attempt(protocol, proposal, scenario, directory,
                                             dict(filter_status="NOT_RUN_RESERVE_UNFILTERED_PREDECLARED"))
            steps += row["physics_steps"]
            if row["status"] == "ERROR": error = row["error"]
        rows.append(row)
    if steps > protocol["quotas"]["maximum_total_task_steps"] or len(rows) != 32:
        raise RuntimeError("reserve task quota/accounting exceeded")
    ledger = dict(schema_version="hug_reserve_collection_v1", status="HARD_STOP" if error else "COMPLETED_BOUNDED_QUOTA",
                  protocol_sha256=collector.sha(output / "protocol.json"), generation_sha256=collector.sha(generation_path),
                  attempts=rows, physics_steps=steps, error=error)
    from grasp_failure_prediction.prediction.recorded_dataset import export_recorded_dataset, PrefixSpec
    try:
        eligible = any(row["status"] == "COMPLETED" and row["outcome"]["failure_type"] != "approach_collision" for row in rows)
        if eligible:
            dataset = output / "reserve_prefixes"
            manifest = export_recorded_dataset([row["recording_path"] for row in rows], dataset, prefix_spec=PrefixSpec(**protocol["prefix"]))
            if (manifest["input_contract"] != protocol["models"]["input_contract"]
                    or manifest.get("cohort") != protocol["expected_cohort"]):
                raise ValueError("reserve input/physical target contract differs from all frozen models")
            ledger["dataset"] = dict(schema_version="hug_reserve_dataset_v1", directory=str(dataset),
                bindings=_files(dataset), counts=manifest["counts"], schedule_sha256=collector.digest(generation["schedule"]))
        else:
            ledger["dataset"] = dict(schema_version="hug_reserve_dataset_v1", status="NO_ELIGIBLE_PREFIXES",
                                      counts=dict(attempted=32, included=0))
        _verify_protocol(protocol); collector.verify(generation["artifact_bindings"])
    except Exception as fault:
        ledger.update(status="HARD_STOP", error=dict(type=type(fault).__name__, message=str(fault)))
        (raw / "postprocessing_traceback.txt").write_text(traceback.format_exc())
    ledger["raw_artifact_bindings"] = _files(raw)
    collector.write(output / "collection.json", ledger)
    return ledger


def _metrics(y, probability):
    from grasp_failure_prediction.prediction.model import outcome_metrics
    result = outcome_metrics(y, probability)
    if len(np.unique(y)) != 2:
        result.update(failure_roc_auc=None, failure_average_precision=None, ranking_status="skipped_single_class")
    else:
        result["ranking_status"] = "measured"
    bins = []
    membership = np.minimum((probability * 10).astype(int), 9)
    for i in range(10):
        selected = membership == i
        if selected.any():
            bins.append(dict(lower=i / 10, upper=(i + 1) / 10, n=int(selected.sum()),
                mean_p_success=float(probability[selected].mean()), observed_success_fraction=float(y[selected].mean())))
    result["calibration"] = dict(bins=bins, expected_calibration_error=sum(
        row["n"] * abs(row["mean_p_success"] - row["observed_success_fraction"]) for row in bins) / len(y))
    return result


def evaluate(output):
    """Exactly one public prediction call per bound bundle over identical rows."""
    output = Path(output).resolve(); protocol = read_protocol(output)
    collection_path = output / "collection.json"; ledger = _json(collection_path)
    generation_path = output / "generation.json"; generation = _json(generation_path)
    _require_commit([collection_path, generation_path])
    if (ledger["status"] != "COMPLETED_BOUNDED_QUOTA" or len(ledger["attempts"]) != 32
            or ledger["protocol_sha256"] != collector.sha(output / "protocol.json")
            or ledger["generation_sha256"] != collector.sha(generation_path)
            or generation["protocol_sha256"] != collector.sha(output / "protocol.json")
            or generation["schedule"] != reserve_schedule(generation["proposals"])):
        raise ValueError("only the completed predeclared32-attempt reserve can be evaluated")
    collector.verify(ledger["raw_artifact_bindings"])
    result_path = output / "evaluation.json"
    if result_path.exists() or (output / "evaluation_started.json").exists():
        raise FileExistsError("reserve evaluation cannot overwrite or select a later result")
    report = dict(schema_version="hug_reserve_model_evaluation_v1", primary_model=PRIMARY, warning_threshold=.5,
        collection_sha256=collector.sha(collection_path), protocol_sha256=collector.sha(output / "protocol.json"),
        reserve_winner_selection=False, fitting_calls=0, models=[])
    dataset_info = ledger["dataset"]
    if dataset_info.get("status") == "NO_ELIGIBLE_PREFIXES":
        report.update(status="NO_ELIGIBLE_PREFIXES", counts=dataset_info["counts"],
                      models=[dict(**bundle,status="SKIPPED",reason="no_eligible_prefixes",metrics=None)
                              for bundle in protocol["models"]["bundles"]])
    else:
        collector.verify(dataset_info["bindings"])
        from grasp_failure_prediction.prediction.recorded_dataset import load_recorded_dataset
        from grasp_failure_prediction.prediction.recorded_training import predict_saved_model
        dataset = load_recorded_dataset(dataset_info["directory"])
        if (dataset.manifest["input_contract"] != protocol["models"]["input_contract"]
                or dataset.manifest.get("cohort") != protocol["expected_cohort"]):
            raise ValueError("reserve input/physical target contract changed")
        scenarios = {row["attempt_id"]: row for row in generation["schedule"]["scenarios"]}
        audit_rows = dataset.manifest["episodes"] + dataset.manifest["excluded"]
        if {row["attempt_id"] for row in audit_rows} != set(scenarios):
            raise ValueError("reserve dataset does not account for every scheduled attempt")
        for row in dataset.manifest["episodes"]:
            scenario = scenarios[row["attempt_id"]]
            if (row["provenance_hashes"].get("scenario") != collector.digest(scenario)
                    or any(row[field] != scenario[other] for field, other in (
                        ("object_id", "object_id"), ("geometry_group_id", "geometry_group_id"),
                        ("observation_group_id", "observation_group_id"), ("group_id", "proposal_group_id")))):
                raise ValueError("reserve prefix scenario/group binding differs from frozen schedule")
        eligible_ids = dataset.episode_ids.tolist()
        objects = np.asarray([row["object_id"] for row in dataset.manifest["episodes"]])
        report.update(status="COMPLETED_FROZEN_MODEL_EVALUATION", counts=dataset.manifest["counts"],
                      eligible_episode_ids_sha256=collector.digest(eligible_ids))
        collector.write(output / "evaluation_started.json", dict(protocol_sha256=report["protocol_sha256"],
                        collection_sha256=report["collection_sha256"], model_bundles=len(protocol["models"]["bundles"])))
        error = None
        for bundle in protocol["models"]["bundles"]:
            row = dict(**bundle, status="NOT_RUN", metrics=None)
            report["models"].append(row)
            if error:
                continue
            try:
                probability = np.asarray(predict_saved_model(bundle["directory"], dataset.X), dtype=np.float64)
                metrics = _metrics(dataset.y, probability)
                prediction = dict(episode_ids=eligible_ids, p_success=probability.tolist(), warning_threshold=.5)
                destination = output / "predictions" / (bundle["bundle_id"] + ".json")
                collector.write(destination, prediction)
                row.update(status="COMPLETED",metrics=metrics,predictions_sha256=collector.sha(destination),
                    by_object={name: _metrics(dataset.y[objects == name], probability[objects == name]) for name in sorted(set(objects))})
            except Exception as fault:
                error = dict(type=type(fault).__name__, message=str(fault))
                row.update(status="ERROR", error=error)
                report.update(status="HARD_STOP", error=error)
    _verify_protocol(protocol)
    collector.write(result_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("project", "main-preparation", "model-report", "training-protocol", "bell-protocol", "bell-captures", "coffee-protocol", "coffee-captures", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--raw-output", type=Path)
    p = commands.add_parser("freeze"); p.add_argument("--output", type=Path, required=True); p.add_argument("--stage", choices=("protocol", "generation"), required=True)
    for name in ("generate", "collect", "evaluate"):
        p = commands.add_parser(name); p.add_argument("--output", type=Path, required=True)
        if name != "evaluate": p.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            value = prepare(args.project, args.main_preparation, args.model_report,
                dict(bell_pepper=(args.bell_protocol, args.bell_captures), coffee_cup=(args.coffee_protocol, args.coffee_captures)),
                args.output, training_protocol=args.training_protocol, raw_output=args.raw_output)
        elif args.command == "freeze": value = dict(sha256=freeze(args.output, args.stage))
        elif args.command == "generate": value = generate(args.output, args.authorization)
        elif args.command == "collect": value = collect(args.output, args.authorization)
        else: value = evaluate(args.output)
    except ReserveBlocked as blocked:
        print(json.dumps(dict(status="BLOCKED", reason=str(blocked)))); raise SystemExit(2)
    print(json.dumps(dict(status=value.get("status", "frozen"))))


if __name__ == "__main__": main()
