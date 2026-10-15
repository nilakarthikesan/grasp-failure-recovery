"""Explicit offline frame derivative of a validated, retained UniDex export.

This companion is deliberately outside the native provider's fresh-bundle API.
It never calls a model, changes joints, clips a target or retries sampling.
"""
from __future__ import annotations

import importlib.util
import importlib.abc
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import sys

import mujoco
import numpy as np

from grasp_failure_prediction.evaluation.schema import canonical_json, canonical_sha256
from grasp_failure_prediction.proposals import unidexgrasp as native_adapter
from grasp_failure_prediction.proposals.artifacts import file_sha256, load_request, load_proposal, write_proposal
from grasp_failure_prediction.proposals.dexgraspnet import validate_fk_translation
from grasp_failure_prediction.proposals.providers import _binding
from grasp_failure_prediction.proposals.registry import load_provider_registry
from grasp_failure_prediction.proposals.schema import GenerationRecord, GraspProposal, NativeArtifactReference, ProposalObjectBinding, ProviderProvenance, ProviderValidation
from grasp_failure_prediction.proposals.translation import TranslationError, translate_native_pose

SCHEMA = "unidexgrasp_derived_frame_bundle_v1"
CLASSIFICATION = "offline_deterministic_frame_reconstruction"
ALGORITHM = "unidexgrasp_saved_canonical_to_input_cpu_float64_v1"
POSITION_TOLERANCE_M = 1e-6
MATRIX_TOLERANCE = 2e-5
TIP_GATE_M = .002
INTERMEDIATE_SCHEMA = "unidexgrasp_sampling_intermediates_v1"


@contextmanager
def static_boundary():
    """Allow FK only; block integration and any already-loaded model call path."""
    restored = []
    def forbidden(*args, **kwargs):
        raise RuntimeError("Offline frame reconstruction prohibits physics and model calls")
    class DenyExecutionImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "tensorflow", "jax", "jaxlib"} or fullname == "mujoco.rollout":
                raise RuntimeError("Offline frame reconstruction prohibits execution imports: " + fullname)
            return None
    import_guard = DenyExecutionImports()
    sys.meta_path.insert(0, import_guard)
    for name in ("mj_step", "mj_step1", "mj_step2"):
        restored.append((mujoco, name, getattr(mujoco, name)))
        setattr(mujoco, name, forbidden)
    torch = sys.modules.get("torch")
    if torch is not None:
        for name in ("__call__", "_call_impl"):
            restored.append((torch.nn.Module, name, getattr(torch.nn.Module, name)))
            setattr(torch.nn.Module, name, forbidden)
    rollout = sys.modules.get("mujoco.rollout")
    if rollout is not None:
        if hasattr(rollout, "rollout"):
            restored.append((rollout, "rollout", rollout.rollout))
            rollout.rollout = forbidden
        if hasattr(rollout, "Rollout") and hasattr(rollout.Rollout, "rollout"):
            restored.append((rollout.Rollout, "rollout", rollout.Rollout.rollout))
            rollout.Rollout.rollout = forbidden
    try:
        yield
    finally:
        sys.meta_path.remove(import_guard)
        for owner, name, original in reversed(restored):
            setattr(owner, name, original)


def write_json(path, value):
    Path(path).write_bytes(canonical_json(value) + b"\n")


def validate_arrays(original_poses, intermediates, seeds):
    """Reject changed rows, coercions and joint bits before any output write."""
    seeds = np.asarray(seeds)
    if seeds.dtype != np.uint32 or seeds.ndim != 1 or not len(seeds):
        raise ValueError("Expected nonempty explicit uint32 seed vector")
    count = len(seeds)
    shapes = {"sampled_rotation": (count, 3, 3), "canon_translation": (count, 3),
              "hand_qpos": (count, 22), "seeds": (count,)}
    if set(intermediates) != set(shapes):
        raise ValueError("Retained intermediate field set differs")
    for name, shape in shapes.items():
        value = intermediates[name]
        dtype = np.uint32 if name == "seeds" else np.float32
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError("Retained intermediate dtype, shape or finite check failed: " + name)
    if intermediates["seeds"].tobytes() != seeds.tobytes():
        raise ValueError("Retained seed order differs; no selection or resequencing is permitted")
    if original_poses.shape != (count, 28) or original_poses.dtype != np.float32 or not np.isfinite(original_poses).all():
        raise ValueError("Original native poses must be finite float32 [N,28]")
    if original_poses[:, 6:].tobytes() != intermediates["hand_qpos"].tobytes():
        raise ValueError("Original poses and retained q bits differ")


def reconstruct_arrays(original_poses, intermediates, seeds):
    """Apply the existing declared inverse rigid transform in CPU float64."""
    validate_arrays(original_poses, intermediates, seeds)
    result, rows = [], []
    for index, seed in enumerate(seeds):
        rotation = intermediates["sampled_rotation"][index].astype(np.float64)
        translation = intermediates["canon_translation"][index].astype(np.float64)
        joints = intermediates["hand_qpos"][index]
        pose = native_adapter.compose_unidexgrasp_prediction(translation, rotation, joints)
        if pose.dtype != np.float64 or pose[6:].tobytes() != joints.astype(np.float64).tobytes():
            raise ValueError("Reconstruction changed promoted joint bits, including signed zero")
        expected_translation = np.einsum("ji,j->i", rotation, translation)
        encoded_rotation = native_adapter._axis_angle_matrix(pose[3:6])
        position_error = float(np.max(np.abs(pose[:3] - expected_translation)))
        matrix_error = float(np.max(np.abs(encoded_rotation - rotation.T)))
        if position_error > POSITION_TOLERANCE_M or matrix_error > MATRIX_TOLERANCE:
            raise ValueError("Derived frame fails unchanged canonical numeric criteria")
        original_rotation = native_adapter._axis_angle_matrix(original_poses[index, 3:6])
        rows.append({"native_row": index, "seed": int(seed), "original_q_float32_bits_preserved": True,
            "derived_q_exact_float64_promotion_including_signed_zero": True,
            "translation_max_absolute_error_m": position_error,
            "rotation_matrix_max_absolute_error": matrix_error,
            "numeric_consistency_pass": True,
            "raw_vs_derived_translation_l2_m": float(np.linalg.norm(pose[:3] - original_poses[index, :3])),
            "raw_vs_derived_rotation_matrix_max_absolute_delta": float(np.max(np.abs(encoded_rotation - original_rotation)))})
        result.append(pose)
    return {"hand_pose": np.asarray(result, dtype=np.float64),
            "hand_qpos_float32": intermediates["hand_qpos"].copy(), "seeds": seeds.copy()}, rows


def guarded_files(paths):
    result = {}
    for path in paths:
        path = Path(path).absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError("Evidence and implementation must be regular files: " + str(path))
        result[str(path)] = file_sha256(path)
    return result


def verify_unchanged(guards):
    for path, digest in guards.items():
        if file_sha256(path) != digest:
            raise ValueError("Original evidence or reused code changed: " + path)


def copy_inventory(directory, records, guards):
    directory.mkdir(parents=True, exist_ok=False)
    inventory = []
    for role, original in records:
        original = Path(original).absolute()
        digest = guards[str(original)]
        target = directory / (role + "_" + original.name)
        shutil.copyfile(original, target)
        if file_sha256(target) != digest:
            raise ValueError("Evidence copy hash differs")
        inventory.append({"role": role, "path": str(target.relative_to(directory.parent)),
                          "original_path": str(original), "sha256": digest})
    return inventory


def load_worker(project_root):
    """Import the existing validator, never its CLI or inference entry point."""
    path = project_root / "scripts/infer_unidexgrasp_proposals.py"
    spec = importlib.util.spec_from_file_location("unidexgrasp_unchanged_transport_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_evidence(*, project_root, bundle, request_file, input_manifest, transport_request, transport_native):
    project_root = Path(project_root).resolve()
    if Path(native_adapter.__file__).resolve() != project_root / "src/grasp_failure_prediction/proposals/unidexgrasp.py":
        raise ValueError("Loaded project package differs from explicit project root")
    request = load_request(request_file)
    if request.provider.id != "unidexgrasp_shadow_v1" or request.observation_path is not None or request.provider_inputs:
        raise ValueError("Only the declared UniDex input channel is supported")
    manifest, report, worker_poses, worker_output, report_file, source = native_adapter._load_export(bundle, request)
    if manifest.pose_field != "hand_pose" or report["tta_applied"] is not False:
        raise ValueError("This derivative supports only untouched pre-TTA hand_pose")
    raw, _, _, _ = native_adapter.read_unidexgrasp_input(input_manifest, request)
    for path, expected in [(input_manifest, report["input_manifest_sha256"]),
                           (request_file, json.loads(transport_request.read_text())["request_file"]["sha256"])]:
        native_adapter._verify_file(path, expected, "original physical input/request")
    if raw["object_to_input"] != manifest.object_to_input.model_dump(mode="json") or raw["object_scale"] != manifest.object_scale or raw["point_cloud_sha256"] != report["point_cloud_sha256"]:
        raise ValueError("Physical input differs from original bundle")
    transport = json.loads(transport_request.read_text())
    response_file = transport_native / "transport_response.json"
    response = json.loads(response_file.read_text())
    if transport.get("schema_version") != "unidexgrasp_native_transport_request_v2" or transport.get("mode") != "fresh_inference":
        raise ValueError("Requires the original retained v2 inference transport")
    if transport["request_binding"] != request.model_dump(mode="json"):
        raise ValueError("Original transport/request binding differs")
    worker = load_worker(project_root)
    original_poses = worker._validate_transport_response(response, transport, file_sha256(transport_request), transport_native)
    if transport.get("conditioning") != raw or transport["input_manifest"]["sha256"] != file_sha256(input_manifest):
        raise ValueError("Original transport conditioning differs from physical input")
    for name in ("source_repository", "source_commit", "input_manifest_sha256", "point_cloud_sha256",
                 "source_hand_mjcf_sha256", "checkpoint_sha256", "config_sha256", "checkpoint_set_id",
                 "sample_count", "seeds", "sampling_variant", "tta_applied"):
        if name not in report or report[name] != response[name]:
            raise ValueError("Original bundle report differs from original transport response: " + name)
    if transport.get("source_repository") != response["source_repository"] or transport.get("source_commit") != response["source_commit"]:
        raise ValueError("Original transport source identity differs")
    for name in ("object_id", "geometry_content_hash", "object_scale", "object_to_input"):
        if report[name] != raw[name]:
            raise ValueError("Original physical report field differs: " + name)
    native_binding = report.get("native_transport")
    if native_binding != {"request_sha256": file_sha256(transport_request), "response_sha256": file_sha256(response_file), "exporter_sha256": transport["exporter_sha256"]}:
        raise ValueError("Original bundle/transport lineage differs")
    for name, expected in [("transport_request.json", file_sha256(transport_request)), ("transport_response.json", file_sha256(response_file))]:
        native_adapter._verify_file(bundle.parent / name, expected, "original bundle transport")
    with np.load(worker_output, allow_pickle=False) as arrays:
        if set(arrays.files) != {"hand_pose"}:
            raise ValueError("Original worker archive has undeclared fields")
        worker_raw = arrays["hand_pose"]
    if worker_raw.dtype != np.float32 or worker_raw.shape != original_poses.shape or worker_raw.tobytes() != original_poses.tobytes():
        raise ValueError("Original worker poses are not exact float32 original transport rows")
    retained = report.get("retained_intermediates")
    if retained != response["intermediates"] or retained.get("schema_version") != INTERMEDIATE_SCHEMA:
        raise ValueError("Original bundle/transport retained-intermediate bindings differ")
    worker._validate_retained_intermediates(retained, transport["request_binding"], bundle.parent)
    intermediate_file = bundle.parent / "sampling_intermediates.npz"
    with np.load(intermediate_file, allow_pickle=False) as z:
        intermediates = {name: z[name].copy() for name in z.files}
    seeds = np.asarray(report["seeds"], dtype=np.uint32)
    validate_arrays(original_poses, intermediates, seeds)
    cloud = native_adapter._child(input_manifest.parent, raw["point_cloud_path"], "point cloud")
    records = [("original_native_manifest", bundle), ("original_inference_report", report_file),
        ("original_worker_output", worker_output), ("original_sampling_intermediates", intermediate_file),
        ("original_source_hand", source), ("original_bundle_transport_request", bundle.parent / "transport_request.json"),
        ("original_bundle_transport_response", bundle.parent / "transport_response.json"),
        ("original_transport_request", transport_request), ("original_transport_response", response_file),
        ("original_transport_output", transport_native / "model_output.npz"),
        ("original_transport_intermediates", transport_native / "sampling_intermediates.npz"),
        ("original_proposal_request", request_file), ("original_input_manifest", input_manifest), ("original_point_cloud", cloud)]
    return request, manifest, report, source, original_poses, intermediates, seeds, records


def source_limit_violations(model, values):
    violations = {}
    for index, value in enumerate(values):
        if model.jnt_limited[index] and (value < model.jnt_range[index, 0] - 1e-9 or value > model.jnt_range[index, 1] + 1e-9):
            violations[native_adapter.SOURCE_JOINT_NAMES[index]] = {"value_rad": float(value),
                "minimum_rad": float(model.jnt_range[index, 0]), "maximum_rad": float(model.jnt_range[index, 1])}
    return violations


def run_reconstruction(**kwargs):
    """Public static boundary, including when called without the CLI wrapper."""
    with static_boundary():
        return _run_reconstruction(**kwargs)


def _run_reconstruction(*, project_root, bundle, request_file, input_manifest, transport_request, transport_native, output, companion_paths=()):
    """Validate originals, retain every row, and export passing derived targets."""
    from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
    from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
    project_root = Path(project_root).resolve()
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError("Derivative output must be a new directory")
    if any(output.resolve().is_relative_to(Path(root).resolve()) for root in (bundle.parent, transport_native)):
        raise ValueError("Derivative output must be outside original bundle and transport directories")
    values = validate_evidence(project_root=project_root, bundle=bundle, request_file=request_file,
        input_manifest=input_manifest, transport_request=transport_request, transport_native=transport_native)
    request, manifest, original_report, source, poses, intermediates, seeds, records = values
    code_files = [Path(__file__), *map(Path, companion_paths),
        project_root / "scripts/infer_unidexgrasp_proposals.py",
        project_root / "cloud/unidexgrasp/native_exporter.py",
        project_root / "third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf"]
    code_files.extend(project_root / ("src/grasp_failure_prediction/" + name) for name in (
        "proposals/unidexgrasp.py", "proposals/dexgraspnet.py", "proposals/translation.py", "proposals/providers.py",
        "proposals/schema.py", "proposals/artifacts.py", "proposals/registry.py", "evaluation/pose_validation.py",
        "evaluation/retargeting.py", "proposals/configs/providers.yaml", "proposals/configs/translations.yaml",
        "proposals/configs/dexgraspnet_shadow_mapping_v1.yaml"))
    code_files = list(dict.fromkeys(code_files))
    guards = guarded_files([path for _, path in records] + code_files)
    derived, numeric_rows = reconstruct_arrays(poses, intermediates, seeds)
    spec = load_provider_registry().resolve(request.provider.id, request.provider.expected_config_hash)
    if spec.translation_strategy_id != native_adapter.TRANSLATION_STRATEGY:
        raise ValueError("Registered translation strategy differs")
    validator = ShadowPoseValidator(default_dex_urdf_root())
    pending = []
    with tempfile.TemporaryDirectory(prefix="unidex-derived-frame-static-") as temporary:
        source_model = native_adapter._prepare_source(source, Path(temporary) / "source.xml", manifest.source_hand_mjcf_sha256)
        for index, pose in enumerate(derived["hand_pose"]):
            row = dict(numeric_rows[index])
            row["source_joint_limit_violations"] = source_limit_violations(source_model, pose[6:])
            row["source_legal"] = not bool(row["source_joint_limit_violations"])
            qlf = float(pose[6 + native_adapter.SOURCE_JOINT_NAMES.index("robot0:LFJ4")])
            row["source_LFJ4_rad"] = qlf
            row["global_little_tip_bound_applicability"] = "COVERED_EXPLICIT_SOURCE_INTERVAL" if 0 <= qlf <= .785 else "NOT_COVERED_OUTSIDE_SOURCE_LFJ4_INTERVAL"
            try:
                native = native_adapter.decode_unidexgrasp_hand_pose(pose, object_to_input=manifest.object_to_input,
                    source_hand_mjcf=source, source_hand_sha256=manifest.source_hand_mjcf_sha256,
                    kinematics_output=Path(temporary) / ("row_%04d.xml" % index))
                translated = translate_native_pose(native, spec.translation_strategy_id)
                native_adapter._validate_target_joint_limits(translated.joint_targets_rad, validator)
                validation = validate_fk_translation(native, translated, maximum_error_m=TIP_GATE_M)
                row.update(status="PRODUCTION_PROPOSAL_PRESENT", target_legal=True,
                           scoped_mapping_accepted=row["source_legal"], production_fk_validation=validation)
                pending.append((row, translated, validation))
            except TranslationError as exc:
                if exc.code not in {"shadow_joint_limit_violation", "forward_kinematics_residual_exceeded"}:
                    raise
                row.update(status="ADAPTER_REJECTED", exception_type=type(exc).__name__, reason_code=exc.code,
                           reason=str(exc), target_legal=False if exc.code == "shadow_joint_limit_violation" else None,
                           scoped_mapping_accepted=False)
                pending.append((row, None, None))
    verify_unchanged(guards)
    output.mkdir(parents=True, exist_ok=False)
    evidence = copy_inventory(output / "original_evidence", records, guards)
    implementation = copy_inventory(output / "implementation", [("code_%02d" % i, path) for i, path in enumerate(code_files)], guards)
    derived_file = output / "derived_output.npz"
    np.savez_compressed(derived_file, **derived)
    lineage = {"schema_version": "unidexgrasp_frame_derivation_lineage_v1", "classification": CLASSIFICATION,
        "algorithm_id": ALGORITHM, "original_evidence": evidence, "implementation": implementation,
        "derived_output": {"path": derived_file.name, "sha256": file_sha256(derived_file)},
        "seeds": [int(seed) for seed in seeds], "joint_values_changed": False,
        "new_model_inference_executed": False, "new_model_forward_calls": 0, "new_sampling_calls": 0,
        "physics_steps": 0, "tta_applied": False, "derived_surface_status": "NOT_MEASURED_UNRESOLVED"}
    lineage_file = output / "derivation_lineage.json"
    write_json(lineage_file, lineage)
    all_artifacts = [(record["role"], output / record["path"]) for record in evidence + implementation]
    all_artifacts += [("derived_frame_output", derived_file), ("derived_frame_lineage", lineage_file)]
    rows, proposal_map = [], {}
    for row, translated, validation in pending:
        index = row["native_row"]
        if translated is not None:
            sample = output / "proposals" / ("raw_%04d" % index)
            native_dir = sample / "native"
            native_dir.mkdir(parents=True)
            artifacts = []
            for role, original in all_artifacts:
                relative = original.relative_to(output)
                target = native_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                if file_sha256(target) != file_sha256(original):
                    raise ValueError("Proposal evidence copy changed")
                artifacts.append(NativeArtifactReference(role=role, path=Path("native") / relative, sha256=file_sha256(target)))
            proposal = GraspProposal(proposal_id=f"{request.request_id}_derived_row_{index:04d}",
                provider=ProviderProvenance(id=spec.provider_id, config_hash=canonical_sha256(spec),
                    code_revision=native_adapter.SOURCE_COMMIT, adapter_code_revision=file_sha256(Path(__file__)),
                    checkpoint_hash=canonical_sha256(original_report["checkpoint_sha256"])),
                generation=GenerationRecord(seed=int(seeds[index]), sample_index=index),
                object=ProposalObjectBinding(id=request.object.id, geometry_content_hash=request.object.geometry_content_hash),
                object_to_palm=translated.object_to_palm, joint_targets_rad=translated.joint_targets_rad,
                compatible_execution_protocols=spec.compatible_execution_protocols,
                input_binding=_binding(request, input_manifest_hash=file_sha256(input_manifest), scene_contract_hash=None),
                provider_validation=ProviderValidation(status="passed", metrics=validation),
                provider_metrics={"classification": CLASSIFICATION, "algorithm_id": ALGORITHM,
                    "derived_bundle_schema": SCHEMA, "derived_raw_row": index,
                    "original_native_inference_classification": original_report["classification"],
                    "new_model_inference_executed": False, "joint_values_changed": False,
                    "sampling_variant": original_report["sampling_variant"], "tta_applied": False,
                    "original_native_manifest_sha256": file_sha256(bundle),
                    "original_inference_report_sha256": file_sha256(bundle.parent / manifest.report_path),
                    "original_sampling_intermediates_sha256": original_report["retained_intermediates"]["sha256"],
                    "derived_output_sha256": file_sha256(derived_file), "derivation_lineage_sha256": file_sha256(lineage_file),
                    "translation_strategy_id": spec.translation_strategy_id,
                    "actual_source_hand_sha256": manifest.source_hand_mjcf_sha256,
                    "source_legal": row["source_legal"], "scoped_mapping_accepted": row["scoped_mapping_accepted"],
                    "derived_surface_status": "NOT_MEASURED_UNRESOLVED"}, native_artifacts=tuple(artifacts))
            proposal_file = sample / "proposal.json"
            write_proposal(proposal_file, proposal)
            load_proposal(proposal_file, verify_native_artifacts=True)
            row.update(path=str(proposal_file), sha256=file_sha256(proposal_file), deep_artifact_verification="PASS")
        row["raw_row"] = index
        rows.append(row)
        proposal_map[str(index)] = dict(row)
    verify_unchanged(guards)
    accepted = sum(row["status"] == "PRODUCTION_PROPOSAL_PRESENT" for row in rows)
    accounting = {"schema_version": "unidexgrasp_derived_frame_import_accounting_v1", "classification": CLASSIFICATION,
        "planned_native_rows": len(rows), "native_rows": len(rows), "all_native_rows_accounted": True,
        "production_accepted": accepted, "production_rejected": len(rows) - accepted,
        "scoped_mapping_accepted_jointly_legal": sum(row["scoped_mapping_accepted"] for row in rows), "rows": rows}
    write_json(output / "production_import_accounting.json", accounting)
    write_json(output / "proposal_map.json", proposal_map)
    report = {**lineage,
        "schema_version": "unidexgrasp_frame_reconstruction_report_v1",
        "original_native_canonical_audit_status": "PRESERVED_SEPARATE_NOT_RECLASSIFIED",
        "algorithm": {"input_dtype": "original_float32_exact_promotion", "compute_dtype": "float64_cpu",
            "rotation": "sampled_rotation.T", "translation": "sampled_rotation.T @ canon_translation",
            "rotation_encoding": "unchanged compose_unidexgrasp_prediction MuJoCo quaternion to axis angle",
            "joint_rule": "exact promotion, no clipping, no fitting, no selection"},
        "numeric_position_consistency_tolerance_m": POSITION_TOLERANCE_M,
        "numeric_matrix_consistency_tolerance": MATRIX_TOLERANCE, "historical_maximum_named_tip_gate_m": TIP_GATE_M,
        "sample_count": len(rows), "production_accepted": accepted, "production_rejected": len(rows) - accepted,
        "scoped_mapping_accepted_jointly_legal": accounting["scoped_mapping_accepted_jointly_legal"], "rows": rows,
        "original_files_and_reused_implementation_unchanged": True,
        "source_legality_is_additional_reporting_only_current_production_gate_unchanged": True}
    report_file = output / "reconstruction_report.json"
    write_json(report_file, report)
    artifacts = {role: {"path": name, "sha256": file_sha256(output / name)} for role, name in (
        ("derived_output", "derived_output.npz"), ("reconstruction_report", "reconstruction_report.json"),
        ("import_accounting", "production_import_accounting.json"), ("proposal_map", "proposal_map.json"))}
    result = {"schema_version": SCHEMA, "classification": CLASSIFICATION, "algorithm_id": ALGORITHM,
        "artifacts": artifacts, "original_evidence": evidence, "implementation": implementation,
        "source_hand_mjcf_sha256": manifest.source_hand_mjcf_sha256,
        "object_to_input": manifest.object_to_input.model_dump(mode="json"), "object_scale": manifest.object_scale,
        "object_id": manifest.object_id, "geometry_content_hash": manifest.geometry_content_hash,
        "seeds": [int(seed) for seed in seeds], "joint_names": list(manifest.joint_names), "sample_count": len(rows),
        "pose_field": "hand_pose", "pose_frame": "input_point_cloud", "translation_units": "m",
        "rotation_representation": "axis_angle", "joint_units": "rad", "derived_surface_status": "NOT_MEASURED_UNRESOLVED"}
    write_json(output / "derived_manifest.json", result)
    verify_unchanged(guards)
    return result
