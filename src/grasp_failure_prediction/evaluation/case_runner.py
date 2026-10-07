"""Load, validate, and execute one portable HUG evaluation case."""

from __future__ import annotations

import argparse
import importlib
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
SHADOW_PALM_CENTER_OFFSET_LOCAL_M = np.array([0.0, 0.0, 0.045])


def load_case(path: str | Path) -> EvaluationCase:
    case_path = Path(path)
    with case_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("evaluation case must contain a YAML/JSON mapping")
    return EvaluationCase.model_validate(payload)


def _validate_case_configuration(case: EvaluationCase) -> None:
    if case.object.id not in SUPPORTED_OBJECTS:
        raise RegistryError(f"unknown object ID: {case.object.id}")
    if case.object.friction_profile_id not in SUPPORTED_FRICTION_PROFILES:
        raise RegistryError(
            f"unknown friction profile ID: {case.object.friction_profile_id}"
        )
    if case.motion_profile_id not in SUPPORTED_MOTION_PROFILES:
        raise RegistryError(f"unknown motion profile ID: {case.motion_profile_id}")


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
        str(project / "scripts" / "infer_sim_observation.py"),
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
):
    if not viewer:
        return runner.execute(
            pose,
            grasp_palm_position_m=palm_position,
            grasp_palm_quaternion_wxyz=palm_quaternion,
            approach_direction_world=approach_direction,
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
            approach_direction_world=approach_direction,
            step_callback=sync,
        )
        window.sync()
        return trace


def run_case(
    case_path: str | Path,
    output_dir: str | Path,
    *,
    viewer: bool = False,
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
        inference=args.inference,
        hug_root=args.hug_root,
        checkpoint=args.checkpoint,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()
