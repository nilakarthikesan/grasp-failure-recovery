"""Small, reproducible pilot of bounded disturbances to saved HUG grasps.

This collects final success/failure labels, not a trained predictor or a dataset
split. Initial RGB-D inputs are referenced; no per-frame vision is collected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
import pinocchio

from grasp_failure_prediction.integrations.hug import load_hug_prediction

from .artifacts import write_evaluation_artifacts
from .case_runner import (
    _hug_world_palm_pose, _runner_class, _saved_prediction_path,
    _validate_case_configuration, load_case,
)
from .pose_validation import PoseValidation, ShadowPoseValidator
from .registry import ResolvedCase, resolve_case
from .retargeting import RetargetedHandPose, ShadowHandRetargeter, default_dex_urdf_root
from .runner import ActuatedShadowRunner, ExecutionState
from .rollout_events import annotate_rollout_events
from .schema import canonical_sha256
from .scoring import score_trace


NOISY_STATES = frozenset({
    ExecutionState.CLOSE_FINGERS, ExecutionState.LIFT,
    ExecutionState.HOLD, ExecutionState.SCORE,
})


def _validate_noise(amplitude_rad: float, seed: int) -> None:
    if not np.isfinite(amplitude_rad) or amplitude_rad < 0.0:
        raise ValueError("noise amplitude must be finite and nonnegative")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("noise seed must be an integer in [0, 2**32)")


class NoisyShadowRunner(ActuatedShadowRunner):
    """Same fixed controller with one bounded finger-target bias per episode."""

    def __init__(self, *args, noise_amplitude_rad: float = 0.0,
                 noise_seed: int = 9000, **kwargs) -> None:
        _validate_noise(noise_amplitude_rad, noise_seed)
        super().__init__(*args, **kwargs)
        self.noise_amplitude_rad = float(noise_amplitude_rad)
        self.noise_seed = int(noise_seed)
        self._joint_ids = np.asarray([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self._hand_joint_names
        ])
        self._hand_dof_addresses = self.model.jnt_dofadr[self._joint_ids]
        self.action_history: list[dict[str, np.ndarray | float]] = []
        self._draw_bias()

    def _draw_bias(self) -> None:
        rng = np.random.default_rng(self.noise_seed)
        self.episode_bias_rad = rng.uniform(
            -1.0, 1.0, len(self._hand_joint_names)
        ) * self.noise_amplitude_rad
        for index, name in enumerate(self._hand_joint_names):
            if name.startswith("WRJ"):
                self.episode_bias_rad[index] = 0.0

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self.action_history.clear()
        self._draw_bias()

    def _noisy_command(self, state: ExecutionState, nominal: np.ndarray):
        requested = (self.episode_bias_rad.copy() if state in NOISY_STATES
                     else np.zeros(len(self._hand_joint_names)))
        # Preserve every original command bit when noise is disabled.
        command = np.asarray(nominal, dtype=np.float64).copy()
        if self.noise_amplitude_rad and state in NOISY_STATES:
            limits = self.model.jnt_range[self._joint_ids]
            command = np.clip(command + requested, limits[:, 0], limits[:, 1])
        return command, requested

    def _advance(self, state, root_position, root_quaternion, hand_qpos) -> None:
        nominal = np.asarray(hand_qpos, dtype=np.float64).copy()
        command, requested = self._noisy_command(state, nominal)
        previous = (self._last_control.copy() if self._initialized else command.copy())
        interval_start = float(self.data.time)
        super()._advance(state, root_position, root_quaternion, command)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        integration_state = np.empty(mujoco.mj_stateSize(self.model, state_spec))
        mujoco.mj_getState(self.model, self.data, integration_state, state_spec)
        self.action_history.append({
            "action_interval_start_s": interval_start,
            "time_s": float(self.data.time),
            "diagnostic_time_s": float(self.data.time) - self.physics_timestep_s,
            "nominal_target_rad": nominal,
            "previous_command_rad": previous,
            "commanded_target_rad": command.copy(),
            "requested_noise_rad": requested,
            "applied_noise_rad": command - nominal,
            "root_target_position_m": np.asarray(root_position).copy(),
            "root_target_quaternion_wxyz": np.asarray(root_quaternion).copy(),
            "full_qpos": self.data.qpos.copy(),
            "full_qvel": self.data.qvel.copy(),
            "hand_qvel": self.data.qvel[self._hand_dof_addresses].copy(),
            "hand_actuator_force": self.data.actuator_force[self._actuator_ids].copy(),
            "integration_state": integration_state,
        })

    def write_action_artifacts(self, output: Path) -> None:
        np.savez_compressed(
            output / "actions.npz",
            joint_names=np.asarray(self._hand_joint_names),
            joint_limits_rad=self.model.jnt_range[self._joint_ids],
            episode_bias_rad=self.episode_bias_rad,
            physics_timestep_s=self.physics_timestep_s,
            control_timestep_s=self.control_timestep_s,
            **{key: np.stack([row[key] for row in self.action_history])
               for key in self.action_history[0]},
        )
        # XML aids inspection; binary includes meshes for self-contained state replay.
        mujoco.mj_saveLastXML(str(output / "scene.xml"), self.model)
        mujoco.mj_saveModel(self.model, str(output / "scene.mjb"))


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                    + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    files = (
        "noisy_rollouts.py", "runner.py", "retargeting.py", "scoring.py",
        "registry.py", "case_runner.py", "artifacts.py", "schema.py", "pose_validation.py",
        "rollout_events.py",
        "../integrations/hug.py", "../integrations/hug_frames.py",
    )
    return {name: _sha256(directory / name) for name in files}


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "mujoco": mujoco.__version__, "pinocchio": pinocchio.__version__}
    for package in ("dex-retargeting",):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    # A PYTHONPATH-selected Pinocchio module can differ from installed metadata.
    try:
        versions["pin_distribution_metadata"] = importlib.metadata.version("pin")
    except importlib.metadata.PackageNotFoundError:
        versions["pin_distribution_metadata"] = "unknown"
    return versions


def _validate_camera_transform(path: Path) -> None:
    transform = np.load(path, allow_pickle=False)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("world camera transform must be a finite 4x4 rigid transform")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-5)
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5)):
        raise ValueError("world camera transform must have a proper rigid rotation and homogeneous row")


@dataclass(frozen=True)
class _PreparedCase:
    resolved: ResolvedCase
    pose: RetargetedHandPose
    validation: PoseValidation
    palm_position: np.ndarray
    palm_quaternion: np.ndarray
    source: dict


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("manifest case paths must be project-relative without '..'")
    return path


def _prepare_case(case_path: Path, project: Path) -> _PreparedCase:
    case = load_case(case_path)
    resolved = resolve_case(case)
    _validate_case_configuration(case)
    if not issubclass(_runner_class(resolved), ActuatedShadowRunner):
        raise ValueError("noisy collection requires the force-limited actuated runner")
    proposal = _saved_prediction_path(case, project)
    camera = proposal.parent / "T_world_camera.npy"
    if not camera.is_file():
        raise FileNotFoundError(f"trusted replay requires world camera transform: {camera}")
    _validate_camera_transform(camera)
    observation = (project / case.grasp.observation_path
                   if case.grasp.observation_path is not None else None)
    if observation is not None and not observation.is_dir():
        raise FileNotFoundError(f"initial observation directory does not exist: {observation}")
    root = default_dex_urdf_root()
    grasp = load_hug_prediction(proposal)
    pose = ShadowHandRetargeter(root).retarget(grasp)
    validator = ShadowPoseValidator(root)
    validation = validator.validate(pose)
    threshold = resolved.execution_protocol.parameters.maximum_mean_fingertip_error_m
    if validation.mean_fingertip_error_m > threshold:
        raise ValueError(f"baseline fingertip alignment {validation.mean_fingertip_error_m:.6f} m "
                         f"exceeds {threshold:.6f} m gate")
    palm_position, palm_quaternion = _hug_world_palm_pose(
        grasp, pose, validator, proposal.parent
    )
    proposal_hash, camera_hash = _sha256(proposal), _sha256(camera)
    observation_hashes = ({
        str(path.relative_to(observation)): _sha256(path)
        for path in sorted(observation.iterdir()) if path.is_file()
        and path.name in {"rgb.png", "depth.png", "depth_m.npy", "depth.npy",
                          "object_mask.png", "intrinsics.npy", "T_world_camera.npy"}
    } if observation else {})
    group = {
        "object_id": case.object.id, "observation_files_sha256": observation_hashes,
        "proposal_sha256": proposal_hash, "camera_sha256": camera_hash,
    }
    observation_group = {key: value for key, value in group.items() if key != "proposal_sha256"}
    source = dict(group, group_id=canonical_sha256(group),
                  observation_group_id=canonical_sha256(observation_group),
                  proposal_path=str(case.grasp.prediction_path),
                  observation_path=str(case.grasp.observation_path) if observation else None)
    return _PreparedCase(resolved, pose, validation, palm_position, palm_quaternion, source)


def _validate_execution(runner: NoisyShadowRunner, trace) -> None:
    times = np.asarray([step.time_s for step in trace.steps])
    if not len(times) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise RuntimeError("simulation trace times must be finite and strictly increasing")
    for step in trace.steps:
        if not all(np.all(np.isfinite(value)) for value in (
            step.hand_qpos, step.palm_position_m, step.object_position_m,
            step.maximum_contact_normal_force_n, step.maximum_actuator_force_fraction,
        )):
            raise RuntimeError("simulation trace contains nonfinite state")
    for row in runner.action_history:
        if not all(np.all(np.isfinite(value)) for value in row.values()):
            raise RuntimeError("simulation action history contains nonfinite state or commands")
    for warning in (mujoco.mjtWarning.mjWARN_BADQPOS,
                    mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC):
        if runner.data.warning[warning].number:
            raise RuntimeError(f"MuJoCo numerical instability warning: {warning.name}")


def _run_episode(prepared: _PreparedCase, output: Path, entry: dict) -> dict:
    resolved = prepared.resolved
    runner = NoisyShadowRunner(
        resolved.execution_protocol, default_dex_urdf_root(),
        physics_timestep_s=resolved.environment.simulator.physics_timestep_s,
        control_timestep_s=resolved.environment.simulator.control_timestep_s,
        noise_amplitude_rad=entry["noise_amplitude_rad"], noise_seed=entry["noise_seed"],
    )
    case = resolved.case
    initial = case.initial_condition
    runner.reset(
        seed=case.seed, object_mass_kg=case.object.mass_kg,
        object_position_m=np.asarray(initial.object_position_m),
        object_orientation_wxyz=np.asarray(initial.object_orientation_xyzw)[[3, 0, 1, 2]],
    )
    trace = runner.execute(
        prepared.pose, grasp_palm_position_m=prepared.palm_position,
        grasp_palm_quaternion_wxyz=prepared.palm_quaternion,
    )
    _validate_execution(runner, trace)
    outcome = score_trace(trace, resolved.execution_protocol,
                          control_timestep_s=runner.control_timestep_s)
    output.mkdir(parents=True, exist_ok=True)
    runner.write_action_artifacts(output)
    events = annotate_rollout_events(
        trace, resolved.execution_protocol,
        control_timestep_s=runner.control_timestep_s, outcome=outcome,
    )
    diagnostic_timing = {
        "state_time": "joint/object qpos and qvel use control-interval endpoints",
        "diagnostic_lag_s": float(runner.physics_timestep_s),
        "diagnostic_fields": ["palm_position_m", "contact diagnostics", "actuator/contact forces"],
        "reason": "mj_step evaluates diagnostics before its final physics integration substep",
        "lift_proxy": "combines endpoint height with preceding physics-step contact diagnostics",
    }
    events["observations"]["diagnostic_timing"] = diagnostic_timing
    _write_json(output / "events.json", events)
    eligible_actions = [row for row, step in zip(runner.action_history, trace.steps, strict=True)
                        if step.state in NOISY_STATES]
    noise_application = {
        "eligible_phases": sorted(state.value for state in NOISY_STATES),
        "eligible_start_s": float(eligible_actions[0]["action_interval_start_s"]),
        "eligible_end_s": float(eligible_actions[-1]["time_s"]),
        "nonzero_requested_bias": bool(np.any(runner.episode_bias_rad)),
        "timing": "bias affects target endpoints; actuator commands interpolate over each control interval",
    }
    metadata = dict(entry, **prepared.source, status="completed", label=int(outcome.success),
                    failure_type=outcome.failure_type.value if outcome.failure_type else None,
                    source_file_sha256=_source_hashes(),
                    runtime_versions=_runtime_versions(),
                    schema_version="noisy_rollout_pilot_v2",
                    noise_mode="uniform_episode_bias",
                    noise_application=noise_application,
                    diagnostic_timing=diagnostic_timing,
                    event_annotations={"path": "events.json", "schema_version": events["schema_version"]},
                    action_alignment="commands act over [action_interval_start_s, time_s]; "
                    "each physics substep interpolates previous_command_rad to commanded_target_rad",
                    state_replay="scene.mjb plus integration_state using mjSTATE_INTEGRATION",
                    vision="referenced initial RGB-D only; no per-frame RGB-D",
                    contacts="privileged simulator diagnostics; sensor availability not assumed",
                    privileged_replay_only_fields=["full_qpos", "full_qvel", "integration_state"],
                    proposed_predictor_input_fields={
                        "trajectory.npz": ["time_s", "state", "hand_qpos"],
                        "actions.npz": ["hand_qvel", "commanded_target_rad",
                                        "root_target_position_m", "root_target_quaternion_wxyz"],
                    },
                    training_input_policy="a future training loader must explicitly whitelist "
                    "observable inputs; full simulator state, outcome labels, noise seeds, and "
                    "hidden object conditions are not predictor features",
                    label_scope="final episode outcome plus sampled lift/retention and height-loss proxies; "
                    "events.json is supervision only, not physical slip onset or predictor input")
    _write_json(output / "collection_metadata.json", metadata)
    # Publish the viewer's completed result only after every extra artifact exists.
    write_evaluation_artifacts(
        output, resolved=resolved, trace=trace, pose_validation=prepared.validation,
        outcome=outcome, final_qpos=runner.data.qpos.copy(), final_qvel=runner.data.qvel.copy(),
    )
    return metadata


def collect_noisy_rollouts(
    manifest_path: str | Path, output_dir: str | Path, *,
    project_root: str | Path | None = None,
    noise_amplitudes_rad=(0.0, 0.05, 0.15), repetitions: int = 2, seed: int = 9000,
) -> dict:
    """Freeze a small collection plan, run it, and index completed and error episodes."""
    amplitudes = tuple(float(value) for value in noise_amplitudes_rad)
    if not amplitudes or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("provide noise amplitudes and a positive integer repetition count")
    for amplitude in amplitudes:
        _validate_noise(amplitude, seed)
    project = Path(project_root or Path.cwd()).resolve()
    manifest = Path(manifest_path)
    paths = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise ValueError("manifest must be a nonempty JSON list of relative case paths")
    relative_paths = [_relative_path(path) for path in paths]
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("manifest case paths must be unique")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    prepared, preflight_errors, plan = {}, {}, []
    for case_index, path in enumerate(relative_paths):
        try:
            prepared[case_index] = _prepare_case(project / path, project)
        except Exception as exc:
            preflight_errors[case_index] = {"error_type": type(exc).__name__, "error": str(exc)}
        for repetition in range(repetitions):
            # Paired amplitude comparisons use exactly the same random direction.
            noise_seed = int(np.random.SeedSequence([seed, case_index, repetition])
                             .generate_state(1)[0])
            for amplitude in amplitudes:
                plan.append({"episode_id": f"episode_{len(plan):06d}",
                             "case_index": case_index, "case_path": str(path),
                             "repetition": repetition, "noise_seed": noise_seed,
                             "noise_amplitude_rad": amplitude,
                             "group_id": (prepared[case_index].source["group_id"]
                                          if case_index in prepared else None),
                             "observation_group_id": (prepared[case_index].source["observation_group_id"]
                                                      if case_index in prepared else None),
                             "object_id": (prepared[case_index].source["object_id"]
                                           if case_index in prepared else None)})
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", {
        "schema_version": "noisy_rollout_plan_v1", "manifest_sha256": _sha256(manifest),
        "project_root": str(project), "master_seed": int(seed), "episodes": plan,
        "source_file_sha256": _source_hashes(),
        "runtime_versions": _runtime_versions(),
        "preflight_errors": preflight_errors,
        "split_policy": "no splits generated; keep all proposal variants in group_id together; "
        "use observation_group_id for scene holdouts and object_id for object holdouts",
        "limitations": "pilot only; one cube does not measure generalization",
    })
    summary = {"planned": len(plan), "completed": 0, "successes": 0, "failures": 0, "errors": 0}
    with (output / "index.jsonl").open("w", encoding="utf-8") as index:
        for entry in plan:
            episode_output = output / entry["episode_id"]
            try:
                if entry["case_index"] in preflight_errors:
                    row = dict(entry, status="validation_failed", label=None,
                               **preflight_errors[entry["case_index"]])
                else:
                    row = _run_episode(prepared[entry["case_index"]], episode_output, entry)
            except Exception as exc:
                # A failed episode must not appear completed in the existing viewer.
                (episode_output / "result.json").unlink(missing_ok=True)
                row = dict(entry, status="runtime_failed", label=None,
                           error_type=type(exc).__name__, error=str(exc))
            if row["status"] == "completed":
                summary["completed"] += 1
                summary["successes" if row["label"] else "failures"] += 1
            else:
                summary["errors"] += 1
                episode_output.mkdir(parents=True, exist_ok=True)
                _write_json(episode_output / "collection_metadata.json", row)
            index.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            index.flush()
            _write_json(output / "summary.json", summary)
            print(f'{entry["episode_id"]}: {row["status"]} label={row["label"]}', flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON list of project-relative saved-proposal cases")
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--noise-amplitudes", type=float, nargs="+", default=[0.0, 0.05, 0.15])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9000)
    args = parser.parse_args()
    summary = collect_noisy_rollouts(args.manifest, args.output, project_root=args.project_root,
                                    noise_amplitudes_rad=args.noise_amplitudes,
                                    repetitions=args.repetitions, seed=args.seed)
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
