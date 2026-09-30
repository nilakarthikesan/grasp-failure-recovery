"""Load, validate, and execute one portable HUG evaluation case."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import yaml

from grasp_failure_prediction.integrations.hug import load_hug_prediction

from .artifacts import write_evaluation_artifacts
from .pose_validation import ShadowPoseValidator, format_validation
from .registry import RegistryError, ResolvedCase, resolve_case
from .retargeting import ShadowHandRetargeter, default_dex_urdf_root
from .runner import AdroitShadowRunner, RunnerStep
from .schema import EvaluationCase, EvaluationResult
from .scoring import score_trace


SUPPORTED_OBJECTS = {"object01"}
SUPPORTED_FRICTION_PROFILES = {"nominal_v1"}
SUPPORTED_MOTION_PROFILES = {"nominal_lift_v1"}


def load_case(path: str | Path) -> EvaluationCase:
    case_path = Path(path)
    with case_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("evaluation case must contain a YAML/JSON mapping")
    return EvaluationCase.model_validate(payload)


def _validate_case_assets(case: EvaluationCase, project_root: Path) -> Path:
    if case.object.id not in SUPPORTED_OBJECTS:
        raise RegistryError(f"unknown object ID: {case.object.id}")
    if case.object.friction_profile_id not in SUPPORTED_FRICTION_PROFILES:
        raise RegistryError(
            f"unknown friction profile ID: {case.object.friction_profile_id}"
        )
    if case.motion_profile_id not in SUPPORTED_MOTION_PROFILES:
        raise RegistryError(f"unknown motion profile ID: {case.motion_profile_id}")
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


def _camera_to_world_wrist(
    runner: AdroitShadowRunner, T_camera_wrist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    camera_id = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    if camera_id < 0:
        raise RegistryError("registered environment has no front camera")
    mujoco.mj_forward(runner.model, runner.data)
    world_from_mujoco_camera = runner.data.cam_xmat[camera_id].reshape(3, 3)
    # MuJoCo cameras use +x right, +y up, -z forward. HUG uses OpenCV camera
    # coordinates: +x right, +y down, +z forward.
    mujoco_camera_from_opencv = np.diag([1.0, -1.0, -1.0])
    world_from_opencv = world_from_mujoco_camera @ mujoco_camera_from_opencv
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = world_from_opencv
    world_from_camera[:3, 3] = runner.data.cam_xpos[camera_id]
    world_from_wrist = world_from_camera @ np.asarray(T_camera_wrist)
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, world_from_wrist[:3, :3].reshape(9))
    return world_from_wrist[:3, 3].copy(), quaternion


def _ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")


def _execute(
    runner: AdroitShadowRunner,
    pose,
    palm_position: np.ndarray,
    palm_quaternion: np.ndarray,
    viewer: bool,
):
    if not viewer:
        return runner.execute(
            pose,
            grasp_palm_position_m=palm_position,
            grasp_palm_quaternion_wxyz=palm_quaternion,
        )

    import mujoco.viewer

    with mujoco.viewer.launch_passive(runner.model, runner.data) as window:
        def sync(_: RunnerStep) -> None:
            window.sync()
            time.sleep(runner.control_timestep_s)

        trace = runner.execute(
            pose,
            grasp_palm_position_m=palm_position,
            grasp_palm_quaternion_wxyz=palm_quaternion,
            step_callback=sync,
        )
        window.sync()
        return trace


def run_case(
    case_path: str | Path,
    output_dir: str | Path,
    *,
    viewer: bool = False,
    project_root: str | Path | None = None,
) -> EvaluationResult:
    project = Path(project_root) if project_root is not None else Path.cwd()
    output = Path(output_dir)
    _ensure_empty_output(output)
    case = load_case(case_path)
    resolved = resolve_case(case)
    prediction_path = _validate_case_assets(case, project)

    grasp = load_hug_prediction(prediction_path)
    urdf_root = default_dex_urdf_root()
    retargeter = ShadowHandRetargeter(urdf_root)
    pose = retargeter.retarget(grasp)
    pose_validator = ShadowPoseValidator(urdf_root)
    validation = pose_validator.validate(pose)

    runner_type = _runner_class(resolved)
    runner = runner_type(
        resolved.execution_protocol,
        urdf_root,
        physics_timestep_s=resolved.environment.simulator.physics_timestep_s,
        control_timestep_s=resolved.environment.simulator.control_timestep_s,
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
    palm_position, palm_quaternion = _camera_to_world_wrist(
        runner, grasp.T_camera_wrist
    )
    trace = _execute(
        runner, pose, palm_position, palm_quaternion, viewer=viewer
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
        "--project-root",
        default=".",
        help="root used to resolve repository-relative grasp paths",
    )
    args = parser.parse_args()
    run_case(
        args.case,
        args.output,
        viewer=args.viewer,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()
