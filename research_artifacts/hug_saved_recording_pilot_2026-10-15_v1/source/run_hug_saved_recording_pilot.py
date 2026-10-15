#!/usr/bin/env python3
"""Prepare/freeze/run a bounded saved-command HUG-to-Shadow recording pilot.

Preparation and tests never advance physics. Run additionally requires a root
authorization receipt bound to the frozen protocol and static/camera reports.
No inference, retarget optimization, grasp repair, or historical writes occur.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


SCHEMA = "hug_saved_recording_pilot_v1"
UNIQUE_INDICES = tuple(range(0, 32, 2))
REPEAT_INDICES = (0, 8, 16, 24)
QUOTA = {"attempts": 20, "unique_saved_commands": 16, "deterministic_repeats": 4,
         "objects": 4, "observations": 8, "control_endpoints_per_attempt": 163,
         "physics_steps_per_attempt": 3260, "maximum_total_physics_steps": 65200}
SOURCE_FILES = (
    "scripts/run_hug_saved_recording_pilot.py",
    "src/grasp_failure_prediction/evaluation/runner.py",
    "src/grasp_failure_prediction/evaluation/recorded_runner.py",
    "src/grasp_failure_prediction/evaluation/prediction_recording.py",
    "src/grasp_failure_prediction/evaluation/scoring.py",
    "src/grasp_failure_prediction/evaluation/pose_validation.py",
    "src/grasp_failure_prediction/evaluation/retargeting.py",
    "src/grasp_failure_prediction/evaluation/object_assets.py",
    "src/grasp_failure_prediction/evaluation/scene_contracts.py",
    "src/grasp_failure_prediction/evaluation/case_runner.py",
    "src/grasp_failure_prediction/evaluation/schema.py",
    "src/grasp_failure_prediction/evaluation/registry.py",
    "src/grasp_failure_prediction/evaluation/configs/environments.yaml",
    "src/grasp_failure_prediction/evaluation/configs/execution_protocols.yaml",
    "src/grasp_failure_prediction/integrations/hug.py",
    "src/grasp_failure_prediction/integrations/hug_frames.py",
    "src/grasp_failure_prediction/prediction/recorded_dataset.py",
    "src/grasp_failure_prediction/prediction/model.py",
)


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def digest_json(value) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tagged(value: str) -> str:
    return value if value.startswith("sha256:") else "sha256:" + value


def schedule() -> list[dict]:
    indices = UNIQUE_INDICES + REPEAT_INDICES
    return [{"attempt_id": f"attempt_{order:04d}", "source_inventory_index": index,
             "repeat_of": None if order < 16 else f"attempt_{UNIQUE_INDICES.index(index):04d}"}
            for order, index in enumerate(indices)]


def read_command(case: dict) -> dict:
    ref = case["stored_shadow_command"]
    if sha(ref["path"]) != tagged(ref["sha256"]):
        raise ValueError("saved command hash differs from inventory")
    with np.load(ref["path"], allow_pickle=False) as data:
        row = ref["row_index"]
        names = data["joint_names"].astype(str).tolist()
        values = {"qpos_rad": data["nominal_target_rad"][row].copy(),
                  "root_position_m": data["root_target_position_m"][row].copy(),
                  "root_quaternion_wxyz": data["root_target_quaternion_wxyz"][row].copy()}
        if "applied_noise_rad" not in data or not np.array_equal(data["applied_noise_rad"][row], np.zeros(24)):
            raise ValueError("selected saved endpoint must have exactly zero applied noise")
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError("expected the 24 unique named saved Shadow joints")
    for key, shape in (("qpos_rad", (24,)), ("root_position_m", (3,)), ("root_quaternion_wxyz", (4,))):
        if values[key].shape != shape or not np.isfinite(values[key]).all():
            raise ValueError("saved command contains invalid numerical values")
    if abs(np.linalg.norm(values["root_quaternion_wxyz"]) - 1.) > 1e-10:
        raise ValueError("saved root quaternion is not unit length")
    return {"joint_names": names, **{key: value.tolist() for key, value in values.items()}}


def _xml_asset_paths(mjcf: Path) -> list[Path]:
    tree = ET.fromstring(mjcf.read_bytes())
    compiler = tree.find("compiler")
    meshdir = "" if compiler is None else compiler.get("meshdir", "")
    return [mjcf] + [(mjcf.parent / meshdir / item.attrib["file"]).resolve()
                     for item in tree.findall("./asset/mesh")]


def prepare(project: Path, inputs_project: Path, output: Path, inventory: Path,
            *, raw_output: Path | None = None) -> dict:
    """Bind the predetermined quota and source bytes; no model compilation/steps."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("pilot evidence directory must be fresh")
    project, inputs_project = project.resolve(), inputs_project.resolve()
    raw_output = (raw_output or project / "runs/hug_saved_recording_pilot_2026-10-15_v1").resolve()
    if raw_output.exists():
        raise FileExistsError("raw pilot directory must be new")
    if raw_output == output.resolve() or output.resolve() in raw_output.parents:
        raise ValueError("raw full-state artifacts must be separate from portable evidence")
    inventory = inventory.resolve()
    source = json.loads(inventory.read_text())["modern_cases"]
    if len(source) != 32:
        raise ValueError("this declared pilot requires the original32 inventory")
    source_bindings = {str((project / path).resolve()): sha(project / path) for path in SOURCE_FILES}
    urdf = project / "third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf"
    hand_assets = [urdf] + [(urdf.parent / mesh.attrib["filename"]).resolve()
                            for mesh in ET.fromstring(urdf.read_bytes()).findall(".//mesh")]
    source_bindings.update({str(path): sha(path) for path in sorted(set(hand_assets))})
    inputs = {str(inventory): sha(inventory)}
    dependency_protocol = project / "research_artifacts/hug_native_mano_comparison_2026-10-15_v1/measurement/protocol.json"
    dependency_dirs = json.loads(dependency_protocol.read_text())["dependency_dirs"]
    if not dependency_dirs or not all(Path(path).is_dir() for path in dependency_dirs):
        raise ValueError("declared existing dependency directories are unavailable")
    inputs[str(dependency_protocol)] = sha(dependency_protocol)
    models_dir = project / "research_artifacts/task_outcome_baseline_v1"
    model_bindings = {str(models_dir / name): sha(models_dir / name) for name in (
        "report.json", "proprioception_and_commands.npz", "commands_only.npz")}
    inputs.update(model_bindings)
    frozen_cases = {}
    for index in UNIQUE_INDICES:
        case = source[index]
        paths = []
        for key in ("native", "stored_shadow_command", "scene_mjb"):
            ref = case[key]
            if sha(ref["path"]) != tagged(ref["sha256"]):
                raise ValueError(f"inventory binding mismatch: {index}/{key}")
            paths.append(Path(ref["path"]))
        for name, ref in case["frame_files"].items():
            if sha(ref["path"]) != tagged(ref["sha256"]):
                raise ValueError(f"inventory frame binding mismatch: {index}/{name}")
            paths.append(Path(ref["path"]))
        paths.extend(Path(case[key]) for key in ("collection_metadata_path", "resolved_case_path"))
        resolved = json.loads(Path(case["resolved_case_path"]).read_text())
        if (resolved["case"]["execution_protocol"]["id"] != "fixed_grasp_lift_v2"
                or resolved["case"]["embodiment"]["id"] != "shadow_hand_right"):
            raise ValueError("pilot cannot change study hand/controller")
        observation = inputs_project / resolved["case"]["grasp"]["observation_path"]
        paths.extend(path for path in observation.iterdir() if path.is_file())
        paths.extend(path for path in Path(case["native"]["path"]).parent.iterdir()
                     if path.is_file() and path.name != "proposal.pkl")
        geometry = resolved["case"]["object"]["geometry"]
        object_assets = [] if geometry is None else _xml_asset_paths(inputs_project / geometry["mjcf_path"])
        paths.extend(object_assets)
        local_bindings = {str(path.resolve()): sha(path) for path in sorted(set(paths))}
        recorded = json.loads(Path(case["collection_metadata_path"]).read_text())
        inputs.update(local_bindings)
        frozen_cases[str(index)] = {
            "object_id": case["object_id"], "observation_group_id": case["observation_group_id"],
            "geometry_group_id": recorded["geometry_group_id"], "proposal_group_id": recorded["group_id"],
            "native_sha256": tagged(case["native"]["sha256"]),
            "command": read_command(case), "input_bindings": local_bindings,
            "asset_sha256": digest_json({str(p.resolve()): sha(p) for p in sorted(set(hand_assets + object_assets))}),
        }
    objects = {case["object_id"] for case in frozen_cases.values()}
    observations = {case["observation_group_id"] for case in frozen_cases.values()}
    if len(objects) != 4 or len(observations) != 8:
        raise ValueError("predeclared four-object/eight-observation coverage changed")
    protocol = {
        "schema_version": SCHEMA, "status": "PREDECLARED_NOT_AUTHORIZED",
        "project": str(project), "inputs_project": str(inputs_project), "inventory": str(inventory),
        "raw_output": str(raw_output), "saved_models_dir": str(models_dir), "saved_model_bindings": model_bindings,
        "dependency_dirs": dependency_dirs, "dependency_protocol": str(dependency_protocol),
        "dependency_policy": "base NumPy/MuJoCo imported first; append existing declared dependency dirs, no installation",
        "source_bindings": source_bindings, "input_bindings": inputs,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__},
        "schedule": schedule(), "cases": frozen_cases,
        "quota": dict(QUOTA),
        "execution_protocol_id": "fixed_grasp_lift_v2", "hand_id": "shadow_hand_right",
        "command_noise_rad": 0., "sensor_noise": 0., "approach_direction_world": [0., 0., -1.],
        "tolerances": {"source_compiled_joint_limits_rad": 1e-12, "root_palm_roundtrip": 1e-10,
                       "scene_position_m": 1e-6, "scene_orientation": 1e-6, "object_mass_kg": 1e-12,
                       "object_inertia_tensor": 1e-12, "determinism_trace_abs": 1e-10,
                       "physics_time_increment_abs_s": 1e-12,
                       "determinism_outcome": "all scored outcome fields exact"},
        "prefix": {"protocol_id": "hug_saved_zero_noise_close_prefix_v1", "prediction_time_s": 2.96,
                   "history_steps": 20, "control_timestep_s": .04},
        "tracking_gate": "report only; no physical tracking-error acceptance threshold",
        "mapping_gate": "existing fixed_grasp_lift_v2 0.01m mean fingertip engineering guard; literal source limits",
        "hard_stop": "first unexpected input/numerical/runtime/recording/determinism failure; remainder NOT_RUN",
        "numerical_gate": "before/after each physics step: full qpos/qvel/qacc/ctrl finite, no MuJoCo warnings, expected time increment",
        "scope": "saved HUG-to-Shadow replay outcomes in current source-accurate simulator; not new mapping or native-hand validation",
        "model_smoke_scope": "unchanged saved weights on reused development inputs/current source; pipeline compatibility only, no prospective eval or improved baseline",
        "forbidden": ["new HUG inference", "new retarget solve", "native dynamic hand", "outcome-guided repair",
                      "gate loosening", "historical overwrite", "training", "unbounded collection"],
    }
    output.mkdir(parents=True, exist_ok=True)
    write(output / "protocol.draft.json", protocol)
    return protocol


def verify_bindings(protocol: dict) -> None:
    if (protocol.get("schema_version") != SCHEMA or protocol.get("schedule") != schedule()
            or protocol.get("quota") != QUOTA or protocol.get("hand_id") != "shadow_hand_right"
            or protocol.get("execution_protocol_id") != "fixed_grasp_lift_v2"
            or protocol.get("command_noise_rad") != 0. or protocol.get("sensor_noise") != 0.
            or protocol.get("approach_direction_world") != [0., 0., -1.]):
        raise ValueError("pilot schema or exact bounded schedule differs from declaration")
    if protocol.get("runtime") != {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__}:
        raise ValueError("pilot runtime differs from declared versions")
    for path, expected in {**protocol["source_bindings"], **protocol["input_bindings"]}.items():
        if sha(path) != expected:
            raise ValueError(f"bound input/source changed: {path}")


def freeze(output: Path) -> str:
    path = output / "protocol.draft.json"
    protocol = json.loads(path.read_text())
    verify_bindings(protocol)
    frozen = output / "protocol.json"
    if frozen.exists():
        raise FileExistsError("pilot protocol is already frozen")
    frozen.write_bytes(path.read_bytes())
    return sha(frozen)


def verify_authorization(protocol_path: Path, receipt: dict) -> None:
    if (receipt.get("schema_version") != "hug_saved_recording_pilot_authorization_v1"
            or receipt.get("authorized") is not True
            or receipt.get("protocol_sha256") != sha(protocol_path)
            or receipt.get("accepted_saved_replay_profile") is not True
            or receipt.get("camera_checks_pass") is not True):
        raise ValueError("root authorization must bind frozen protocol and accepted static/camera gates")
    for name in ("static_profile_report", "camera_report"):
        reference = receipt.get(name, {})
        if not reference.get("path") or sha(reference["path"]) != reference.get("sha256"):
            raise ValueError(f"authorization does not bind {name}")


def phase_tracking(arrays: dict) -> dict:
    error = arrays["hand_qpos"] - arrays["applied_target_rad"]
    result = {}
    for phase in dict.fromkeys(arrays["phase"].tolist()):
        values = error[arrays["phase"] == phase]
        result[phase] = {"samples": len(values), "joint_values": values.size,
                         "rmse_rad": float(np.sqrt(np.mean(values ** 2))),
                         "mean_abs_rad": float(np.mean(abs(values))),
                         "p95_abs_rad": float(np.quantile(abs(values), .95)),
                         "max_abs_rad": float(np.max(abs(values)))}
    return result


def compare_replay(first: dict, second: dict, tolerance: float) -> dict:
    if first["outcome"] != second["outcome"]:
        raise ValueError("deterministic repeat changed exact scored outcome")
    errors = {}
    for group, fields in (("observables", ("time_s", "hand_qpos", "hand_qvel", "requested_target_rad", "applied_target_rad")),
                          ("diagnostics", ("full_qpos", "full_qvel"))):
        for field in fields:
            a, b = first[group][field], second[group][field]
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("deterministic repeat trace shapes/values differ")
            error = float(np.max(abs(a - b)))
            if error > tolerance:
                raise ValueError(f"deterministic repeat exceeded tolerance: {field}={error}")
            errors[field] = error
    for field in ("joint_names", "phase"):
        if not np.array_equal(first["observables"][field], second["observables"][field]):
            raise ValueError(f"deterministic repeat changed {field}")
    return {"passed": True, "categorical_and_scored_outcome_exact": True, "maximum_absolute_errors": errors}


def model_smoke(bundle, protocol: dict) -> dict:
    """Run immutable saved weights after exact robot/history contract checks."""
    from grasp_failure_prediction.prediction.model import MODEL_SCHEMA, load_model, outcome_metrics, summarize_prefix
    models_dir = Path(protocol["saved_models_dir"])
    for path, expected in protocol["saved_model_bindings"].items():
        if sha(path) != expected:
            raise ValueError("saved model weights/report changed")
    report = json.loads((models_dir / "report.json").read_text())
    old, new = report["input_contract"], bundle.manifest["input_contract"]
    if (report.get("model_schema") != MODEL_SCHEMA
            or report.get("summary_order") != ["last", "mean", "population_std", "last_minus_first"]
            or old.get("fields") != ["hand_qpos", "hand_qvel", "commanded_target_rad"]
            or new.get("fields") != ["hand_qpos", "hand_qvel", "applied_target_rad"]
            or any(old[key] != new[key] for key in (
                "joint_names", "history_steps", "prediction_time_s", "control_timestep_s", "features", "flatten_order", "dtype"))
            or len(new["joint_names"]) != 24 or new["history_steps"] != 20 or bundle.X.shape[1] != 1440):
        raise ValueError("recorded prefix differs from immutable saved model contract")
    results = {}
    for name in ("proprioception_and_commands", "commands_only"):
        info = report["models"][name]
        model_path = models_dir / info["model_file"]
        if model_path.name != name + ".npz" or sha(model_path) != tagged(info["model_sha256"]):
            raise ValueError("saved weight hash differs from its original model report")
        features = summarize_prefix(bundle.X, commands_only=name == "commands_only")
        model = load_model(model_path)
        scores = model.predict_success(features)
        results[name] = {"model_sha256": sha(model_path), "feature_shape": list(features.shape),
                         "p_success": scores.tolist(), "smoke_metrics": outcome_metrics(bundle.y, scores)}
    return {"scope": protocol["model_smoke_scope"], "model_training_performed": False,
            "weights_report_sha256": sha(models_dir / "report.json"), "models": results}


@dataclass(frozen=True)
class SavedCommandPose:
    joint_names: tuple[str, ...]
    qpos: np.ndarray


class AttemptStopped(RuntimeError):
    def __init__(self, mapping_status: str, code: str, message: str):
        self.mapping_status, self.code = mapping_status, code
        super().__init__(message)


def _quat_matrix(quaternion) -> np.ndarray:
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=np.float64))
    return result.reshape(3, 3)


def expected_object_inertia(geometry, mass: float) -> tuple[np.ndarray, np.ndarray]:
    if geometry is None:
        return np.zeros(3), np.eye(3) * mass / 3 * (2 * .025 ** 2)
    inertial = ET.fromstring(geometry.source_xml).find("./worldbody/body/inertial")
    if any(name in inertial.attrib for name in ("euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("pilot only admits the frozen assets' literal inertial frames")
    if "fullinertia" in inertial.attrib:
        xx, yy, zz, xy, xz, yz = map(float, inertial.attrib["fullinertia"].split())
        tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    else:
        tensor = np.diag(np.fromstring(inertial.attrib["diaginertia"], sep=" "))
    rotation = _quat_matrix(np.fromstring(inertial.get("quat", "1 0 0 0"), sep=" "))
    return (np.fromstring(inertial.get("pos", "0 0 0"), sep=" "),
            rotation @ tensor @ rotation.T * mass / float(inertial.attrib["mass"]))


def _save_partial(attempt_dir: Path, recorder=None, live_data=None) -> dict:
    """Best-effort raw debugging only; never admit a partial physical label."""
    files = {}
    (attempt_dir / "failure_traceback.txt").write_text(traceback.format_exc())
    files["failure_traceback.txt"] = sha(attempt_dir / "failure_traceback.txt")
    if recorder is not None and recorder.sample_count:
        np.savez_compressed(attempt_dir / "partial_observables.npz", **recorder.arrays())
        files["partial_observables.npz"] = sha(attempt_dir / "partial_observables.npz")
        rows = recorder._diagnostics
        if rows and rows[0]:
            np.savez_compressed(attempt_dir / "partial_diagnostics.npz",
                                **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
            files["partial_diagnostics.npz"] = sha(attempt_dir / "partial_diagnostics.npz")
    if live_data is not None:
        np.savez_compressed(attempt_dir / "failed_live_state.npz", time_s=live_data.time,
                            qpos=live_data.qpos, qvel=live_data.qvel, qacc=live_data.qacc, ctrl=live_data.ctrl,
                            warning_counts=_warning_counts(live_data))
        files["failed_live_state.npz"] = sha(attempt_dir / "failed_live_state.npz")
    return files


def _warning_counts(data) -> np.ndarray:
    return np.asarray([data.warning[index].number for index in range(int(mujoco.mjtWarning.mjNWARNING))])


def guarded_physics_step(model, data, original_step, budget: dict, quota: dict,
                         *, time_tolerance_s: float) -> None:
    """Stop immediately, including MuJoCo's finite-state automatic resets."""
    if (budget["total"] >= quota["maximum_total_physics_steps"]
            or budget["this_attempt"] >= quota["physics_steps_per_attempt"]):
        raise RuntimeError("declared physics-step budget exhausted")
    before_warnings = _warning_counts(data).copy()
    if before_warnings.any():
        raise RuntimeError("MuJoCo warning already present before physics step")
    if not all(np.isfinite(getattr(data, field)).all() for field in ("qpos", "qvel", "qacc", "ctrl")):
        raise RuntimeError("nonfinite full state before physics step")
    before = float(data.time)
    if not np.isfinite(before):
        raise RuntimeError("nonfinite simulator time before physics step")
    budget["total"] += 1
    budget["this_attempt"] += 1
    original_step(model, data)
    warnings = _warning_counts(data)
    if np.any(warnings != before_warnings):
        names = [name for name, value in mujoco.mjtWarning.__members__.items()
                 if name != "mjNWARNING" and warnings[int(value)]]
        raise RuntimeError("MuJoCo warning during physics step: " + ",".join(names))
    if not all(np.isfinite(getattr(data, field)).all() for field in ("qpos", "qvel", "qacc", "ctrl")):
        raise RuntimeError("nonfinite full state after physics step")
    if not np.isfinite(data.time) or not np.isclose(float(data.time) - before, float(model.opt.timestep),
                                                  rtol=0, atol=time_tolerance_s):
        raise RuntimeError("unexpected simulator time increment or automatic reset")


def _preflight(protocol: dict, source: dict, frozen: dict):
    from grasp_failure_prediction.evaluation.case_runner import _resolve_object_geometry, _source_scene_binding
    from grasp_failure_prediction.evaluation.pose_validation import load_shadow_model, _link_positions
    from grasp_failure_prediction.evaluation.registry import resolve_case
    from grasp_failure_prediction.evaluation.schema import EvaluationCase
    from grasp_failure_prediction.integrations.hug import load_hug_prediction

    if read_command(source) != frozen["command"]:
        raise AttemptStopped("refused_guard", "saved_command_changed", "literal frozen saved command changed")
    history = json.loads(Path(source["resolved_case_path"]).read_text())
    case = EvaluationCase.model_validate(history["case"])
    resolved = resolve_case(case)
    if (resolved.environment.model_dump(mode="json") != history["environment"]
            or resolved.execution_protocol.model_dump(mode="json") != history["execution_protocol"]
            or resolved.environment_config_hash != history["environment_config_hash"]
            or resolved.execution_protocol_config_hash != history["execution_protocol_config_hash"]):
        raise AttemptStopped("setup_error", "registry_changed", "current registry differs from saved fixed protocol")
    geometry = _resolve_object_geometry(case, Path(protocol["inputs_project"]))
    native_path = Path(source["native"]["path"])
    binding = _source_scene_binding(case, resolved, Path(protocol["inputs_project"]), native_path, geometry)
    if binding["scene_contract_status"] != "verified":
        raise AttemptStopped("setup_error", "unverified_scene", "saved capture requires its verified scene contract")
    camera = np.load(source["frame_files"]["T_world_camera.npy"]["path"], allow_pickle=False)
    tol = protocol["tolerances"]
    if (camera.shape != (4, 4) or not np.isfinite(camera).all()
            or not np.allclose(camera[3], [0., 0., 0., 1.], rtol=0, atol=tol["scene_orientation"])
            or not np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), rtol=0, atol=tol["scene_orientation"])
            or abs(np.linalg.det(camera[:3, :3]) - 1.) > tol["scene_orientation"]):
        raise AttemptStopped("setup_error", "camera_contract", "saved camera is not a finite rigid transform")
    command = frozen["command"]
    urdf_root = Path(protocol["project"]) / "third_party/dex-urdf/robots/hands"
    urdf = urdf_root / "shadow_hand/shadow_hand_right.urdf"
    source_joints = {joint.attrib["name"]: joint for joint in ET.fromstring(urdf.read_bytes()).findall("joint")
                     if joint.attrib["type"] == "revolute"}
    names, q = command["joint_names"], np.asarray(command["qpos_rad"])
    if set(names) != set(source_joints):
        raise AttemptStopped("refused_invalid_retarget", "source_joint_set", "saved command joint names differ from source")
    hand = load_shadow_model(urdf_root, floating_root=True)
    data = mujoco.MjData(hand)
    for name, target in zip(names, q, strict=True):
        joint = mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower, upper = (float(source_joints[name].find("limit").get(key)) for key in ("lower", "upper"))
        if not lower <= target <= upper:
            raise AttemptStopped("refused_invalid_retarget", "literal_source_joint_limit", f"saved {name} violates source limits")
        if not np.allclose(hand.jnt_range[joint], [lower, upper], rtol=0, atol=tol["source_compiled_joint_limits_rad"]):
            raise AttemptStopped("setup_error", "compiled_joint_limit", "compiled/source joint limits disagree")
        data.qpos[hand.jnt_qposadr[joint]] = target
    root = mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")
    address = int(hand.jnt_qposadr[root])
    data.qpos[address:address + 7] = np.r_[command["root_position_m"], command["root_quaternion_wxyz"]]
    mujoco.mj_forward(hand, data)
    palm_id = mujoco.mj_name2id(hand, mujoco.mjtObj.mjOBJ_BODY, "palm")
    palm_position = data.xpos[palm_id].copy()
    palm_quaternion = np.empty(4)
    mujoco.mju_mat2Quat(palm_quaternion, data.xmat[palm_id])
    native = load_hug_prediction(native_path)
    reference_tips = native.landmarks_3d[[4, 8, 12, 16, 20]] @ camera[:3, :3].T + camera[:3, 3]
    tip_errors = np.linalg.norm(_link_positions(hand, data, urdf)[:5] - reference_tips, axis=1)
    if float(tip_errors.mean()) > resolved.execution_protocol.parameters.maximum_mean_fingertip_error_m:
        raise AttemptStopped("refused_guard", "fingertip_engineering_guard", "saved command fails unchanged mean fingertip guard")
    return resolved, geometry, SavedCommandPose(tuple(names), q), palm_position, palm_quaternion, {
        "mean_fingertip_error_m": float(tip_errors.mean()), "maximum_fingertip_error_m": float(tip_errors.max()),
        "world_palm_position_m": palm_position.tolist(), "world_palm_quaternion_wxyz": palm_quaternion.tolist(),
        "scene_contract_status": "verified", "new_retarget_solves": 0,
    }


def execute_attempt(protocol: dict, source: dict, frozen: dict, attempt_dir: Path, step_budget: dict) -> dict:
    from grasp_failure_prediction.evaluation.recorded_runner import RecordedActuatedShadowRunner
    from grasp_failure_prediction.evaluation.scoring import score_trace

    resolved, geometry, pose, palm_position, palm_quaternion, validation = _preflight(protocol, source, frozen)
    spec = resolved.environment.simulator
    runner = RecordedActuatedShadowRunner(
        resolved.execution_protocol, Path(protocol["project"]) / "third_party/dex-urdf/robots/hands",
        physics_timestep_s=spec.physics_timestep_s, control_timestep_s=spec.control_timestep_s,
        object_geometry=geometry,
    )
    command = frozen["command"]
    named_q = dict(zip(pose.joint_names, pose.qpos, strict=True))
    ordered_q = np.asarray([named_q[name] for name in runner._hand_joint_names])
    root_position, root_quaternion = runner._root_pose_for_palm(palm_position, palm_quaternion, ordered_q)
    tolerance = protocol["tolerances"]["root_palm_roundtrip"]
    if (np.max(abs(root_position - command["root_position_m"])) > tolerance
            or np.max(abs(_quat_matrix(root_quaternion) - _quat_matrix(command["root_quaternion_wxyz"]))) > tolerance):
        raise AttemptStopped("setup_error", "saved_root_roundtrip", "current palm target does not reconstruct saved root")
    case = resolved.case
    runner.reset(seed=case.seed, object_mass_kg=case.object.mass_kg,
                 object_position_m=np.asarray(case.initial_condition.object_position_m),
                 object_orientation_wxyz=np.asarray(case.initial_condition.object_orientation_xyzw)[[3, 0, 1, 2]])
    obj = runner._object_body_id
    center, expected_tensor = expected_object_inertia(geometry, case.object.mass_kg)
    actual_rotation = _quat_matrix(runner.model.body_iquat[obj])
    actual_tensor = actual_rotation @ np.diag(runner.model.body_inertia[obj]) @ actual_rotation.T
    if (abs(runner.model.body_mass[obj] - case.object.mass_kg) > protocol["tolerances"]["object_mass_kg"]
            or np.max(abs(actual_tensor - expected_tensor)) > protocol["tolerances"]["object_inertia_tensor"]
            or np.max(abs(runner.model.body_ipos[obj] - center)) > protocol["tolerances"]["scene_position_m"]):
        raise AttemptStopped("setup_error", "object_inertia_contract", "current object mass/inertia does not reproduce source tensor")
    validation.update(object_mass_kg=float(runner.model.body_mass[obj]),
                      object_inertia_tensor_max_error=float(np.max(abs(actual_tensor - expected_tensor))))
    model_path = attempt_dir / "scene.mjb"
    mujoco.mj_saveModel(runner.model, str(model_path))
    original_step = mujoco.mj_step
    step_budget["this_attempt"] = 0
    def bounded_step(model, data):
        return guarded_physics_step(model, data, original_step, step_budget, protocol["quota"],
                                    time_tolerance_s=protocol["tolerances"]["physics_time_increment_abs_s"])
    mujoco.mj_step = bounded_step
    try:
        trace = runner.execute(pose, grasp_palm_position_m=palm_position,
                               grasp_palm_quaternion_wxyz=palm_quaternion,
                               approach_direction_world=np.asarray(protocol["approach_direction_world"]))
    except Exception:
        _save_partial(attempt_dir, runner.prediction_recorder, runner.data)
        raise
    finally:
        mujoco.mj_step = original_step
    if len(trace.steps) != protocol["quota"]["control_endpoints_per_attempt"]:
        raise RuntimeError("unexpected completed control-endpoint count")
    outcome = asdict(score_trace(trace, resolved.execution_protocol, control_timestep_s=spec.control_timestep_s))
    outcome["failure_type"] = None if outcome["failure_type"] is None else outcome["failure_type"].value
    arrays = runner.prediction_recorder.arrays()
    diagnostic = {key: np.stack([row[key] for row in runner.prediction_recorder._diagnostics])
                  for key in runner.prediction_recorder._diagnostics[0]}
    first_failure = next((step.time_s - spec.physics_timestep_s for step in trace.steps
                          if step.hand_table_contact and step.state.value in {"move_to_pregrasp", "approach"}), None)
    return {"recorder": runner.prediction_recorder, "observables": arrays, "diagnostics": diagnostic,
            "outcome": outcome, "first_failure_time_s": first_failure if outcome["failure_type"] == "approach_collision" else None,
            "validation": validation, "model_sha256": sha(model_path)}


def _metadata(protocol, protocol_sha, entry, source, frozen, *, mapping="ok", code=None, message=None,
              execution_status, outcome=None, model_sha=None):
    from grasp_failure_prediction.evaluation.prediction_recording import AttemptMetadata
    hashes = {
        "code": digest_json(protocol["source_bindings"]), "configuration": protocol_sha,
        "inputs": digest_json(frozen["input_bindings"]), "native_pose": frozen["native_sha256"],
        "mapped_pose": digest_json(frozen["command"]), "assets": frozen["asset_sha256"],
        "label_rule": digest_json({"scorer": protocol["source_bindings"][str(Path(protocol["project"]) / "src/grasp_failure_prediction/evaluation/scoring.py")],
                                   "success": {"minimum_lift_m": .14, "required_hold_s": 2., "maximum_object_drop_m": .02},
                                   "control_timestep_s": .04}),
    }
    if model_sha is not None:
        hashes["model"] = model_sha
    return AttemptMetadata(
        attempt_id=entry["attempt_id"], hand_id="shadow_hand_right", execution_protocol_id="fixed_grasp_lift_v2",
        object_id=source["object_id"], geometry_group_id=frozen["geometry_group_id"],
        observation_group_id=source["observation_group_id"], proposal_group_id=frozen["proposal_group_id"],
        mapping={"status": mapping, "code": code, "message": message}, execution_status=execution_status,
        outcome=outcome, provenance_hashes=hashes,
    )


def run(output: Path, authorization_path: Path, *, executor=execute_attempt) -> dict:
    """Run the exact authorized plan; every unexpected failure stops the queue."""
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    verify_authorization(protocol_path, json.loads(authorization_path.read_text()))
    # The established base NumPy/MuJoCo versions are already imported above.
    # Append the original validated dependency bundle only afterward, avoiding
    # its alternate NumPy build while supplying Pydantic/YAML/Pinocchio.
    sys.path.insert(0, str(Path(protocol["project"]) / "src"))
    for directory in protocol.get("dependency_dirs", []):
        if directory not in sys.path:
            sys.path.append(directory)
    from grasp_failure_prediction.evaluation.prediction_recording import write_incomplete_attempt
    from grasp_failure_prediction.prediction.recorded_dataset import PrefixSpec, export_recorded_dataset, load_recorded_dataset
    raw_output = Path(protocol["raw_output"])
    if raw_output.exists() or (output / "summary.json").exists():
        raise FileExistsError("never overwrite pilot measurements")
    # Failsafe: the allowed dynamic runner never requires this optimizer.
    from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter
    old_retarget = ShadowHandRetargeter.retarget
    def no_solve(*args, **kwargs):
        raise RuntimeError("saved pilot forbids new retarget solves")
    ShadowHandRetargeter.retarget = no_solve
    protocol_sha = sha(protocol_path)
    summary = {"schema_version": SCHEMA, "protocol_sha256": protocol_sha,
               "authorization_sha256": sha(authorization_path), "status": "RUNNING",
               "schedule": [{**entry, "status": "NOT_RUN", "outcome": None} for entry in protocol["schedule"]],
               "physics_steps": 0, "new_retarget_solves": 0, "new_inference_calls": 0,
               "tracking_scope": "joint error to final clipped interval target; report only"}
    completed, record_dirs, budget = {}, [], {"total": 0, "this_attempt": 0}
    write(output / "summary.json", summary)
    try:
        verify_bindings(protocol)
        inventory = json.loads(Path(protocol["inventory"]).read_text())["modern_cases"]
    except Exception as error:
        summary.update(status="HARD_STOPPED", hard_stop_reason=str(error), all_inputs_sources_still_bound=False)
        summary["counts"] = {"planned": 20, "completed": 0, "not_run": 20, "incomplete": 0,
                             "success": 0, "physical_failure": 0, "approach_collision": 0}
        summary["export"] = {"status": "NO_COMPLETED_PHYSICAL_ATTEMPTS"}
        write(output / "summary.json", summary)
        ShadowHandRetargeter.retarget = old_retarget
        return summary
    (raw_output / "attempts").mkdir(parents=True)
    (output / "cases").mkdir()
    try:
        for index, entry in enumerate(protocol["schedule"]):
            source = inventory[entry["source_inventory_index"]]
            frozen = protocol["cases"][str(entry["source_inventory_index"]) ]
            attempt_dir = raw_output / "attempts" / entry["attempt_id"]
            attempt_dir.mkdir()
            row = summary["schedule"][index]
            steps_before, result = budget["total"], None
            try:
                if sha(protocol_path) != protocol_sha:
                    raise RuntimeError("frozen pilot protocol changed during execution")
                verify_bindings(protocol)
                result = executor(protocol, source, frozen, attempt_dir, budget)
                determinism = None
                if entry["repeat_of"] is not None:
                    determinism = compare_replay(completed[entry["repeat_of"]], result,
                                                 protocol["tolerances"]["determinism_trace_abs"])
                physical = {"success": bool(result["outcome"]["success"]),
                            "failure_type": result["outcome"]["failure_type"],
                            "label_rule_id": "fixed_grasp_lift_v2_task_outcome_v1",
                            "first_failure_time_s": result["first_failure_time_s"]}
                metadata = _metadata(protocol, protocol_sha, entry, source, frozen,
                                     execution_status="completed", outcome=physical, model_sha=result["model_sha256"])
                record = attempt_dir / "recording"
                result["recorder"].write(record, metadata)
                write(output / "cases" / (entry["attempt_id"] + "_scored_outcome.json"), result["outcome"])
                write(output / "cases" / (entry["attempt_id"] + "_static_validation.json"), result["validation"])
                row.update(status="COMPLETED", outcome=physical, recording=str(record),
                           raw_record_sha256=sha(record / "attempt.json"),
                           raw_artifacts_sha256=json.loads((record / "attempt.json").read_text())["artifacts_sha256"],
                           tracking_by_phase=phase_tracking(result["observables"]), determinism=determinism)
                completed[entry["attempt_id"]] = result
                record_dirs.append(record)
            except Exception as error:
                partial = _save_partial(attempt_dir, None if result is None else result.get("recorder"))
                # execute_attempt may have retained an earlier partial state.
                for path in attempt_dir.iterdir():
                    if path.is_file():
                        partial[path.name] = sha(path)
                mapping = (error.mapping_status if isinstance(error, AttemptStopped) else
                           "setup_error" if budget["total"] == steps_before and result is None else "ok")
                code = error.code if isinstance(error, AttemptStopped) else type(error).__name__
                execution = "not_executed" if mapping != "ok" else "runtime_error"
                metadata = _metadata(protocol, protocol_sha, entry, source, frozen, mapping=mapping,
                                     code=code if mapping != "ok" else None,
                                     message=str(error) if mapping != "ok" else None,
                                     execution_status=execution)
                record = attempt_dir / "incomplete"
                write_incomplete_attempt(record, metadata)
                record_dirs.append(record)
                row.update(status="MAPPING_OR_SETUP_REFUSED" if mapping != "ok" else "RUNTIME_OR_RECORDING_ERROR",
                           outcome=None, recording=str(record), raw_record_sha256=sha(record / "attempt.json"),
                           raw_partial_artifacts_sha256=partial,
                           error={"code": code, "message": str(error)})
                summary["status"] = "HARD_STOPPED"
                summary["hard_stop_attempt"] = entry["attempt_id"]
                break
            finally:
                summary["physics_steps"] = budget["total"]
                write(output / "summary.json", summary)
        else:
            summary["status"] = "COMPLETED_BOUNDED_QUOTA"
        try:
            verify_bindings(protocol)
            summary["all_inputs_sources_still_bound"] = True
        except Exception as error:
            summary.update(status="HARD_STOPPED", hard_stop_reason=str(error), all_inputs_sources_still_bound=False)
    finally:
        ShadowHandRetargeter.retarget = old_retarget
    write(output / "attempts_manifest.json", [str(path) for path in record_dirs])
    outcomes = [row["outcome"] for row in summary["schedule"] if row["outcome"] is not None]
    summary["counts"] = {"planned": 20, "completed": len(outcomes),
                         "not_run": sum(row["status"] == "NOT_RUN" for row in summary["schedule"]),
                         "incomplete": len(record_dirs) - len(outcomes),
                         "success": sum(row["success"] for row in outcomes),
                         "physical_failure": sum(not row["success"] for row in outcomes),
                         "approach_collision": sum(row["failure_type"] == "approach_collision" for row in outcomes)}
    if record_dirs and outcomes:
        try:
            manifest = export_recorded_dataset(record_dirs, output / "derived_prefixes",
                                               prefix_spec=PrefixSpec.model_validate(protocol["prefix"]))
            bundle = load_recorded_dataset(output / "derived_prefixes")
            smoke = model_smoke(bundle, protocol)
            write(output / "model_input_smoke.json", smoke)
            summary["export"] = {"status": "EXPORTED", "counts": manifest["counts"],
                                 "prefix_shape": list(bundle.X.shape),
                                 "model_smoke_path": "model_input_smoke.json",
                                 "model_training_performed": False}
        except Exception as error:
            if isinstance(error, ValueError) and str(error) == "no completed cutoff-eligible physical attempts to export":
                summary["export"] = {"status": "NO_ELIGIBLE_PREFIXES", "reason": str(error)}
            else:
                summary["status"] = "HARD_STOPPED"
                (raw_output / "postprocessing_traceback.txt").write_text(traceback.format_exc())
                summary["export"] = {"status": "ERROR", "reason": str(error), "error_type": type(error).__name__,
                                     "traceback_sha256": sha(raw_output / "postprocessing_traceback.txt")}
    else:
        summary["export"] = {"status": "NO_COMPLETED_PHYSICAL_ATTEMPTS"}
    write(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--project", type=Path, required=True)
    prep.add_argument("--inputs-project", type=Path, required=True)
    prep.add_argument("--inventory", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--raw-output", type=Path)
    for action in ("freeze", "run"):
        item = sub.add_parser(action)
        item.add_argument("--output", type=Path, required=True)
        if action == "run":
            item.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.project, args.inputs_project, args.output, args.inventory, raw_output=args.raw_output)
        print(json.dumps({"status": result["status"], "quota": result["quota"],
                          "draft_sha256": sha(args.output / "protocol.draft.json")}))
    elif args.action == "freeze":
        print(json.dumps({"frozen_protocol_sha256": freeze(args.output), "execution_authorized": False}))
    else:
        result = run(args.output, args.authorization)
        print(json.dumps({"status": result["status"], "counts": result["counts"], "physics_steps": result["physics_steps"]}))


if __name__ == "__main__":
    main()
