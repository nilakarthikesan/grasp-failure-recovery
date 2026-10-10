"""Load, validate, and execute one portable HUG evaluation case."""

from __future__ import annotations

import argparse
import importlib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import mujoco
import numpy as np
import yaml

from grasp_failure_prediction.integrations.hug import load_hug_prediction
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT

from .artifacts import write_evaluation_artifacts
from .object_assets import load_object_geometry
from .pose_validation import ShadowPoseValidator, _body_id, format_validation
from .registry import RegistryError, ResolvedCase, resolve_case
from .recording import EvaluationVideoRecorder
from .retargeting import RetargetingError, ShadowHandRetargeter, default_dex_urdf_root
from .runner import AdroitShadowRunner, RunnerStep
from .schema import EvaluationCase, EvaluationResult, canonical_sha256
from .scoring import score_trace
from .scene_contracts import validate_scene_contract


SUPPORTED_OBJECTS = {"object01"}
SUPPORTED_FRICTION_PROFILES = {"nominal_v1"}
SUPPORTED_MOTION_PROFILES = {"nominal_lift_v1"}
SHADOW_PALM_CENTER_OFFSET_LOCAL_M = np.array([0.0, 0.0, 0.045])


def load_case(path: str | Path) -> EvaluationCase:
    case_path = Path(path)
    with case_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("evaluation case must contain a YAML/JSON mapping")
    return EvaluationCase.model_validate(payload)


def _validate_case_configuration(case: EvaluationCase) -> None:
    if case.object.geometry is None and case.object.id not in SUPPORTED_OBJECTS:
        raise RegistryError(f"unknown object ID: {case.object.id}")
    if case.object.friction_profile_id not in SUPPORTED_FRICTION_PROFILES:
        raise RegistryError(
            f"unknown friction profile ID: {case.object.friction_profile_id}"
        )
    if case.motion_profile_id not in SUPPORTED_MOTION_PROFILES:
        raise RegistryError(f"unknown motion profile ID: {case.motion_profile_id}")


def _resolve_object_geometry(case: EvaluationCase, project: Path):
    reference = case.object.geometry
    if reference is None:
        return None
    return load_object_geometry(project / reference.mjcf_path,
                                expected_content_hash=reference.expected_content_hash)


def _geometry_content_hash(geometry) -> str:
    if geometry is not None:
        return geometry.content_hash
    return canonical_sha256({"kind": "legacy_cube_v1", "half_size_m": [.025, .025, .025],
                             "base_mass_kg": .18, "friction": [1., .005, .0001]})


def _source_scene_binding(case, resolved, project, proposal, geometry) -> dict:
    """Verify captured inputs and fresh inference, preserving explicit legacy status."""
    digest = lambda path: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    camera = proposal.parent / "T_world_camera.npy"
    camera_hash = digest(camera) if camera.is_file() else None
    observation = project / case.grasp.observation_path if case.grasp.observation_path is not None else None
    hashes = ({path.name: digest(path) for path in sorted(observation.iterdir())
               if path.is_file() and path.name in {
                   "rgb.png", "depth.png", "depth_m.npy", "depth.npy", "object_mask.png",
                   "intrinsics.npy", "T_world_camera.npy"}}
              if observation is not None and observation.is_dir() else {})
    sidecar = proposal.parent / "scene_contract.json"
    result = {"camera_sha256": camera_hash, "observation_files_sha256": hashes,
              "scene_contract_status": "legacy_unverified", "scene_contract_sha256": None,
              "scene_contract": None, "observation_manifest_sha256": None,
              "observation_selection_uv": None, "observation_source": None,
              "inference_report_sha256": None, "hug_inference_provenance": None}
    if not sidecar.is_file():
        if observation is not None and (observation / "scene_contract.json").is_file():
            raise ValueError("captured observation requires its matching proposal scene contract")
        if geometry is not None:
            raise ValueError("explicit object geometry requires a captured scene contract")
        return result
    if observation is None or not observation.is_dir():
        raise ValueError("verified scene requires referenced initial observation files")
    contract = json.loads(sidecar.read_text())
    validate_scene_contract(
        contract, object_id=case.object.id, geometry_content_hash=_geometry_content_hash(geometry),
        mass_kg=case.object.mass_kg, object_position_m=case.initial_condition.object_position_m,
        object_orientation_xyzw=case.initial_condition.object_orientation_xyzw,
        environment_config_hash=resolved.environment_config_hash, camera_sha256=camera_hash,
        observation_files_sha256=hashes,
    )
    captured_sidecar = observation / "scene_contract.json"
    if not captured_sidecar.is_file() or digest(captured_sidecar) != digest(sidecar):
        raise ValueError("proposal scene contract must match the captured observation contract")
    captured_manifest = observation / "observation_manifest.json"
    copied_manifest = proposal.parent / "observation_manifest.json"
    if (not captured_manifest.is_file() or not copied_manifest.is_file()
            or digest(captured_manifest) != digest(copied_manifest)):
        raise ValueError("proposal observation manifest must match the captured observation manifest")
    manifest_hash = digest(captured_manifest)
    manifest = json.loads(captured_manifest.read_text())
    if not isinstance(manifest, dict) or "selection_uv" not in manifest or not manifest.get("source"):
        raise ValueError("captured observation manifest requires selection and source provenance")
    inference_path = proposal.parent / "inference_report.json"
    if not inference_path.is_file():
        raise ValueError("verified scene requires a fresh HUG inference report")
    inference = json.loads(inference_path.read_text())
    if (not isinstance(inference, dict) or inference.get("real_hug_inference_passed") is not True
            or inference.get("proposal_sha256") != digest(proposal).removeprefix("sha256:")
            or inference.get("scene_contract_sha256") != digest(sidecar).removeprefix("sha256:")):
        raise ValueError("HUG inference report does not bind this proposal to its captured scene")
    if type(inference.get("seed")) is not int or inference["seed"] != case.grasp.inference_seed:
        raise ValueError("HUG inference seed must match the declared inference_seed as an integer")
    if inference.get("manifest_sha256") != manifest_hash.removeprefix("sha256:"):
        raise ValueError("HUG inference manifest does not match the referenced observation")
    copied_depth = proposal.parent / "depth.png"
    if (inference.get("hug_depth_png_sha256") != hashes["depth.png"].removeprefix("sha256:")
            or not copied_depth.is_file() or digest(copied_depth) != hashes["depth.png"]):
        raise ValueError("HUG inference depth PNG does not match the referenced observation")
    names = {"rgb_path":"rgb.png", "depth_m_path":"depth_m.npy",
             "intrinsics_path":"intrinsics.npy", "mask_path":"object_mask.png"}
    if any(inference.get("observation_hashes",{}).get(key) != hashes[name].removeprefix("sha256:")
           for key, name in names.items()):
        raise ValueError("HUG inference inputs do not match the referenced observation")
    result.update(scene_contract_status="verified", scene_contract_sha256=digest(sidecar),
                  scene_contract=contract, observation_manifest_sha256=manifest_hash,
                  observation_selection_uv=manifest["selection_uv"], observation_source=manifest["source"],
                  inference_report_sha256=digest(inference_path),
                  hug_inference_provenance={key: inference.get(key) for key in (
                      "seed", "sampling_steps", "device", "dtype", "generation_rng",
                      "hug_code_commit", "checkpoint_sha256", "inference_script_sha256",
                      "manifest_sha256", "hug_depth_png_sha256", "prepared_sample_sha256",
                      "input_tensor_hash",
                  )})
    return result


def _saved_prediction_path(case: EvaluationCase, project_root: Path) -> Path:
    if case.grasp.prediction_path is None:
        raise ValueError(
            "saved-proposal mode requires grasp.prediction_path in the evaluation case"
        )
    prediction = project_root / case.grasp.prediction_path
    if not prediction.is_file():
        raise FileNotFoundError(f"HUG prediction does not exist: {prediction}")
    return prediction


def _runner_class(resolved: ResolvedCase) -> type[AdroitShadowRunner]:
    module_name, separator, class_name = resolved.environment.runner.partition(":")
    if not separator:
        raise RegistryError("environment runner must use module:Class syntax")
    runner_class = getattr(importlib.import_module(module_name), class_name)
    if not issubclass(runner_class, AdroitShadowRunner):
        raise RegistryError("registered environment runner has the wrong interface")
    return runner_class


def _table_parallel_world_wrist(
    object_position_world_m: np.ndarray,
    palm_height_above_object_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place the Shadow palm horizontally and directly above the object."""

    # The Shadow palm plane is local x-z. Rotating +90 degrees around local x
    # maps that plane to world x-y with the grasping side facing the object.
    half_sqrt_two = np.sqrt(0.5)
    quaternion_wxyz = np.array([half_sqrt_two, half_sqrt_two, 0.0, 0.0])
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quaternion_wxyz)
    visual_palm_center = np.asarray(object_position_world_m, dtype=np.float64).copy()
    visual_palm_center[2] += palm_height_above_object_m
    palm_body_origin = visual_palm_center - (
        rotation.reshape(3, 3) @ SHADOW_PALM_CENTER_OFFSET_LOCAL_M
    )
    return palm_body_origin, quaternion_wxyz


def _hug_world_palm_pose(
    grasp,
    pose,
    validator: ShadowPoseValidator,
    observation_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose HUG's world wrist pose with Dex's corrected hand frame."""

    camera_path = observation_dir / "T_world_camera.npy"
    if not camera_path.is_file():
        raise FileNotFoundError(f"observation has no world camera transform: {camera_path}")
    T_world_camera = np.asarray(np.load(camera_path), dtype=np.float64)
    if T_world_camera.shape != (4, 4) or not np.all(np.isfinite(T_world_camera)):
        raise ValueError("T_world_camera must be a finite 4x4 transform")
    T_world_mano = T_world_camera @ grasp.T_camera_wrist
    world_operator_rotation = T_world_mano[:3, :3] @ MANO_TO_OPERATOR_RIGHT.T

    validator.set_pose(pose)
    base_to_palm_rotation = validator.data.xmat[
        _body_id(validator.model, "palm")
    ].reshape(3, 3)
    world_palm_rotation = world_operator_rotation @ base_to_palm_rotation
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, world_palm_rotation.reshape(9))
    # HUG's wrist origin is used as the Shadow palm target. This fixed origin
    # correspondence is versioned with the environment and can be calibrated
    # independently without changing the predicted grasp.
    return T_world_mano[:3, 3].copy(), quaternion


def _ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")


def _run_case_inference(
    case: EvaluationCase,
    project: Path,
    output: Path,
    *,
    hug_root: Path,
    checkpoint: Path,
) -> Path:
    """Run HUG for one case in an isolated copy of its observation."""

    if case.grasp.observation_path is None:
        raise ValueError(
            "--inference requires grasp.observation_path in the evaluation case"
        )
    observation = project / case.grasp.observation_path
    if not observation.is_dir():
        raise FileNotFoundError(f"HUG observation directory does not exist: {observation}")
    if not hug_root.is_dir():
        raise FileNotFoundError(f"HUG checkout does not exist: {hug_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HUG checkpoint does not exist: {checkpoint}")

    inference_dir = output / "hug_inference"
    shutil.copytree(observation, inference_dir)
    # A captured observation may also contain an older proposal from a prior
    # run. The isolated case must always generate a fresh proposal for its seed.
    stale_proposal = inference_dir / "proposal.pkl"
    if stale_proposal.exists():
        stale_proposal.unlink()
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[3] / "scripts" / "infer_sim_observation.py"),
        "--hug-root",
        str(hug_root),
        "--observation",
        str(inference_dir),
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(case.grasp.inference_seed),
    ]
    subprocess.run(command, cwd=project, check=True)
    proposal = inference_dir / "proposal.pkl"
    if not proposal.is_file():
        raise RuntimeError(f"HUG inference did not produce a proposal: {proposal}")
    return proposal


def _execute(
    runner: AdroitShadowRunner,
    pose,
    palm_position: np.ndarray,
    palm_quaternion: np.ndarray,
    approach_direction: np.ndarray,
    viewer: bool,
    video_path: Path | None,
):
    recorder = (
        EvaluationVideoRecorder(runner, video_path)
        if video_path is not None
        else None
    )

    def record(step: RunnerStep) -> None:
        if recorder is not None:
            recorder.capture(step)

    if not viewer:
        try:
            return runner.execute(
                pose,
                grasp_palm_position_m=palm_position,
                grasp_palm_quaternion_wxyz=palm_quaternion,
                approach_direction_world=approach_direction,
                step_callback=record,
            )
        finally:
            if recorder is not None:
                recorder.close()

    import mujoco.viewer

    with mujoco.viewer.launch_passive(runner.model, runner.data) as window:
        def sync(step: RunnerStep) -> None:
            record(step)
            window.sync()
            time.sleep(runner.control_timestep_s)

        try:
            trace = runner.execute(
                pose,
                grasp_palm_position_m=palm_position,
                grasp_palm_quaternion_wxyz=palm_quaternion,
                approach_direction_world=approach_direction,
                step_callback=sync,
            )
            window.sync()
            return trace
        finally:
            if recorder is not None:
                recorder.close()


def run_case(
    case_path: str | Path,
    output_dir: str | Path,
    *,
    viewer: bool = False,
    record_video: bool = True,
    inference: bool = True,
    hug_root: str | Path | None = None,
    checkpoint: str | Path | None = None,
    project_root: str | Path | None = None,
) -> EvaluationResult:
    project = Path(project_root) if project_root is not None else Path.cwd()
    output = Path(output_dir)
    _ensure_empty_output(output)
    case = load_case(case_path)
    resolved = resolve_case(case)
    _validate_case_configuration(case)
    object_geometry = _resolve_object_geometry(case, project)
    if inference:
        if hug_root is None or checkpoint is None:
            raise ValueError("--inference requires --hug-root and --checkpoint")
        prediction_path = _run_case_inference(
            case,
            project,
            output,
            hug_root=Path(hug_root).resolve(),
            checkpoint=Path(checkpoint).resolve(),
        )
    else:
        prediction_path = _saved_prediction_path(case, project)
        output.mkdir(parents=True, exist_ok=True)

    _source_scene_binding(case, resolved, project, prediction_path, object_geometry)

    grasp = load_hug_prediction(prediction_path)
    urdf_root = default_dex_urdf_root()
    retargeter = ShadowHandRetargeter(urdf_root)
    pose = retargeter.retarget(grasp)
    pose_validator = ShadowPoseValidator(urdf_root)
    validation = pose_validator.validate(pose)
    maximum_error = (
        resolved.execution_protocol.parameters.maximum_mean_fingertip_error_m
    )
    # Saved-proposal mode is an explicit simulator-debug escape hatch and may
    # intentionally replay synthetic or malformed poses. Normal inference runs
    # must pass the versioned alignment gate before MuJoCo advances.
    if inference and validation.mean_fingertip_error_m > maximum_error:
        raise RetargetingError(
            "mean fingertip alignment error exceeds protocol limit: "
            f"{validation.mean_fingertip_error_m:.6f} m > {maximum_error:.6f} m"
        )

    runner_type = _runner_class(resolved)
    runner = runner_type(
        resolved.execution_protocol,
        urdf_root,
        physics_timestep_s=resolved.environment.simulator.physics_timestep_s,
        control_timestep_s=resolved.environment.simulator.control_timestep_s,
        object_geometry=object_geometry,
    )
    initial = case.initial_condition
    xyzw = np.asarray(initial.object_orientation_xyzw, dtype=np.float64)
    object_quaternion_wxyz = xyzw[[3, 0, 1, 2]]
    runner.reset(
        seed=case.seed,
        object_mass_kg=case.object.mass_kg,
        object_position_m=np.asarray(initial.object_position_m, dtype=np.float64),
        object_orientation_wxyz=object_quaternion_wxyz,
    )
    world_camera_path = prediction_path.parent / "T_world_camera.npy"
    if world_camera_path.is_file():
        palm_position, palm_quaternion = _hug_world_palm_pose(
            grasp, pose, pose_validator, prediction_path.parent
        )
        approach_direction = np.array([0.0, 0.0, -1.0])
    else:
        palm_position, palm_quaternion = _table_parallel_world_wrist(
            np.asarray(initial.object_position_m, dtype=np.float64),
            resolved.execution_protocol.parameters.palm_height_above_object_m,
        )
        approach_direction = np.array([0.0, 0.0, -1.0])
    trace = _execute(
        runner,
        pose,
        palm_position,
        palm_quaternion,
        approach_direction,
        viewer=viewer,
        video_path=(output / "rollout.mp4" if record_video else None),
    )
    outcome = score_trace(
        trace,
        resolved.execution_protocol,
        control_timestep_s=runner.control_timestep_s,
    )
    result = write_evaluation_artifacts(
        output,
        resolved=resolved,
        trace=trace,
        pose_validation=validation,
        outcome=outcome,
        final_qpos=runner.data.qpos.copy(),
        final_qvel=runner.data.qvel.copy(),
        video=(Path("rollout.mp4") if record_video else None),
        hug_proposal=(Path("hug_inference/proposal.pkl") if inference else None),
        inference_report=(
            Path("hug_inference/inference_report.json")
            if inference
            else None
        ),
    )
    print(format_validation(validation))
    print(
        json.dumps(
            {
                "case_id": result.case_id,
                "success": result.success,
                "failure_type": result.failure_type,
                "output": str(output),
            },
            indent=2,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", help="evaluation case YAML or JSON")
    parser.add_argument("--output", required=True, help="new or empty output directory")
    parser.add_argument("--viewer", action="store_true", help="play execution live")
    parser.add_argument(
        "--no-video",
        dest="record_video",
        action="store_false",
        help="disable rollout.mp4 recording for simulator debugging",
    )
    parser.set_defaults(record_video=True)
    parser.add_argument(
        "--no-inference",
        dest="inference",
        action="store_false",
        help="skip HUG and execute the saved grasp.prediction_path",
    )
    parser.set_defaults(inference=True)
    parser.add_argument("--hug-root", help="external pinned HUG checkout")
    parser.add_argument("--checkpoint", help="external HUG checkpoint")
    parser.add_argument(
        "--project-root",
        default=".",
        help="root used to resolve repository-relative grasp paths",
    )
    args = parser.parse_args()
    run_case(
        args.case,
        args.output,
        viewer=args.viewer,
        record_video=args.record_video,
        inference=args.inference,
        hug_root=args.hug_root,
        checkpoint=args.checkpoint,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()
