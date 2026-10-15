"""Opt-in wrist clearance planning for the existing force-limited controller.

This is a different execution protocol. Historical v2 rollouts and the frozen
predictor evaluation remain unchanged; preflight rejection has no binary label.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from pydantic import Field

from .artifacts import write_evaluation_artifacts
from .noisy_rollouts import (
    NoisyShadowRunner, _prepare_case, _runtime_versions, _sha256,
    _source_hashes, _validate_execution, _write_json,
)
from .registry import ExecutionProtocolSpec, ResolvedCase
from .retargeting import default_dex_urdf_root
from .rollout_events import annotate_rollout_events
from .schema import RegistryReference, canonical_sha256
from .scoring import score_trace
from .table_clearance import (
    TableClearanceConfiguration, _lowest_world_z, plan_table_clearance,
)


class TableSafeProtocolSpec(ExecutionProtocolSpec):
    execution_protocol_id: Literal["fixed_grasp_lift_v3_table_clearance"]
    controller: Literal["palm_preserving_wrist_clearance_v1"]
    base_protocol_id: str
    base_protocol_config_hash: str
    clearance_configuration: dict[str, float | int | bool]
    runtime_table_margin_m: float = Field(default=0.001, gt=0, le=0.010)


def table_safe_protocol(base: ExecutionProtocolSpec, configuration=None):
    if base.execution_protocol_id != "fixed_grasp_lift_v2":
        raise ValueError("table clearance execution requires fixed_grasp_lift_v2 input")
    config = configuration or TableClearanceConfiguration()
    return TableSafeProtocolSpec.model_validate(dict(
        base.model_dump(mode="json"),
        execution_protocol_id="fixed_grasp_lift_v3_table_clearance",
        controller="palm_preserving_wrist_clearance_v1",
        base_protocol_id=base.execution_protocol_id,
        base_protocol_config_hash=canonical_sha256(base),
        clearance_configuration=asdict(config),
    ))


class ClearanceRejected(ValueError):
    """The saved proposal cannot be executed under the clearance policy."""


class MotionClearanceRejected(ClearanceRejected):
    """Actual simulated geometry entered the runtime table clearance margin."""


class TableSafeNoisyShadowRunner(NoisyShadowRunner):
    """Adjust wrist/carrier placement, then use unchanged dynamics and noise."""

    def __init__(self, protocol: TableSafeProtocolSpec, *args, **kwargs):
        if not isinstance(protocol, TableSafeProtocolSpec):
            raise TypeError("use table_safe_protocol to version the planning configuration")
        self.clearance_configuration = TableClearanceConfiguration(
            **protocol.clearance_configuration
        )
        self.clearance_plan = None
        self.runtime_guard_event = None
        super().__init__(protocol, *args, **kwargs)
        self._clearance_scratch = mujoco.MjData(self.model)
        self._clearance_geoms = tuple(
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) in self._hand_body_ids
            and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])
        )

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.clearance_plan = None
        self.runtime_guard_event = None

    def _guard_table_clearance(self, state):
        # Kinematics in scratch data reads the actual endpoint, including
        # compliant carrier motion and fingers deflected by object contact.
        # It leaves the live solver and its lagged contact diagnostics untouched.
        scratch = self._clearance_scratch
        scratch.qpos[:] = self.data.qpos
        mujoco.mj_kinematics(self.model, scratch)
        table_top = (scratch.geom_xpos[self._table_geom_id, 2]
                     + self.model.geom_size[self._table_geom_id, 2])
        clearance, geom = min(
            (_lowest_world_z(self.model, scratch, g) - table_top, g)
            for g in self._clearance_geoms
        )
        if clearance < self.protocol.runtime_table_margin_m:
            body = int(self.model.geom_bodyid[geom])
            self.runtime_guard_event = dict(
                time_s=float(self.data.time), phase=state.value,
                geom_id=int(geom), body=self.model.body(body).name,
                actual_table_clearance_m=float(clearance),
                required_table_clearance_m=self.protocol.runtime_table_margin_m,
                scope="actual active hand geometry above the infinite table top plane",
            )
            raise MotionClearanceRejected(
                f"runtime table margin reached by {self.model.body(body).name} "
                f"at {self.data.time:.3f} s ({clearance * 1000:.3f} mm clearance)"
            )

    def _advance(self, state, root_position, root_quaternion, hand_qpos):
        # Same interpolation and mj_step order as the original actuated runner;
        # the only addition is a non-mutating guard after each physics substep.
        nominal = np.asarray(hand_qpos, dtype=np.float64).copy()
        command, requested = self._noisy_command(state, nominal)
        previous = self._last_control.copy() if self._initialized else command.copy()
        interval_start = float(self.data.time)
        target_root = np.r_[root_position, root_quaternion]
        root = self._root_qpos_address
        if not self._initialized:
            self.data.qpos[root:root + 7] = target_root
            self.data.qpos[self._hand_qpos_addresses] = command
            self.data.qvel[:] = 0.0
            self.data.mocap_pos[0] = root_position
            self.data.mocap_quat[0] = root_quaternion
            self.data.ctrl[self._actuator_ids] = command
            mujoco.mj_forward(self.model, self.data)
            self._last_root = target_root.copy()
            self._last_control = command.copy()
            self._initialized = True
            self._guard_table_clearance(state)
        start = self.data.qpos.copy()
        start[root:root + 7] = self._last_root
        goal = start.copy()
        goal[root:root + 7] = target_root
        velocity = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, velocity, self.control_timestep_s, start, goal)
        for substep in range(1, self.physics_steps_per_control + 1):
            alpha = substep / self.physics_steps_per_control
            root_command = start.copy()
            mujoco.mj_integratePos(self.model, root_command, velocity, alpha * self.control_timestep_s)
            self.data.mocap_pos[0] = root_command[root:root + 3]
            self.data.mocap_quat[0] = root_command[root + 3:root + 7]
            self.data.ctrl[self._actuator_ids] = ((1.0 - alpha) * self._last_control + alpha * command)
            mujoco.mj_step(self.model, self.data)
            self._guard_table_clearance(state)
        self._last_root = target_root.copy()
        self._last_control = command.copy()
        self._record(state)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.empty(mujoco.mj_stateSize(self.model, state_spec))
        mujoco.mj_getState(self.model, self.data, integration, state_spec)
        self.action_history.append(dict(
            action_interval_start_s=interval_start, time_s=float(self.data.time),
            diagnostic_time_s=float(self.data.time) - self.physics_timestep_s,
            nominal_target_rad=nominal, previous_command_rad=previous,
            commanded_target_rad=command.copy(), requested_noise_rad=requested,
            applied_noise_rad=command - nominal,
            root_target_position_m=np.asarray(root_position).copy(),
            root_target_quaternion_wxyz=np.asarray(root_quaternion).copy(),
            full_qpos=self.data.qpos.copy(), full_qvel=self.data.qvel.copy(),
            hand_qvel=self.data.qvel[self._hand_dof_addresses].copy(),
            hand_actuator_force=self.data.actuator_force[self._actuator_ids].copy(),
            integration_state=integration,
        ))

    def execute(self, pose, *, grasp_palm_position_m,
                grasp_palm_quaternion_wxyz, **kwargs):
        # The table-plane certificate assumes the original vertical descent.
        direction = np.asarray(kwargs.get("approach_direction_world", [0, 0, -1]), dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("approach direction must be a finite 3-vector")
        norm = float(np.linalg.norm(direction))
        if norm <= 0 or not np.allclose(direction / norm, [0, 0, -1], rtol=0, atol=1e-12):
            raise ValueError("table clearance execution currently supports vertical descent only")
        self.clearance_plan = plan_table_clearance(
            self, pose, grasp_palm_position_m, grasp_palm_quaternion_wxyz,
            configuration=self.clearance_configuration,
        )
        if not self.clearance_plan.feasible:
            raise ClearanceRejected(self.clearance_plan.rejection_reason)
        # Keep the original WORLD palm pose. Recomputing it from HUG with the
        # new wrist angles would rotate the intended grasp and defeat invariance.
        return super().execute(
            self.clearance_plan.adjusted_pose,
            grasp_palm_position_m=grasp_palm_position_m,
            grasp_palm_quaternion_wxyz=grasp_palm_quaternion_wxyz,
            **kwargs,
        )


def _version_resolved(resolved: ResolvedCase, configuration):
    protocol = table_safe_protocol(resolved.execution_protocol, configuration)
    config_hash = canonical_sha256(protocol)
    case = resolved.case.model_copy(update={
        "execution_protocol": RegistryReference(
            id=protocol.execution_protocol_id, expected_config_hash=config_hash
        )
    })
    return replace(resolved, case=case, execution_protocol=protocol,
                   execution_protocol_config_hash=config_hash)


def execute_table_safe_case(case_path, output_dir, *, project_root=None,
                            noise_amplitude_rad=0.0, noise_seed=9000,
                            configuration=None):
    """Validate and run one saved v2 proposal with separately versioned planning."""
    project = Path(project_root or Path.cwd()).resolve()
    case_path = Path(case_path)
    if not case_path.is_absolute():
        case_path = project / case_path
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be new or empty: {output}")
    prepared = _prepare_case(case_path, project)
    resolved = _version_resolved(prepared.resolved, configuration)
    runner = TableSafeNoisyShadowRunner(
        resolved.execution_protocol, default_dex_urdf_root(),
        physics_timestep_s=resolved.environment.simulator.physics_timestep_s,
        control_timestep_s=resolved.environment.simulator.control_timestep_s,
        noise_amplitude_rad=noise_amplitude_rad, noise_seed=noise_seed,
        object_geometry=prepared.object_geometry,
    )
    case = resolved.case
    runner.reset(
        seed=case.seed, object_mass_kg=case.object.mass_kg,
        object_position_m=np.asarray(case.initial_condition.object_position_m),
        object_orientation_wxyz=np.asarray(case.initial_condition.object_orientation_xyzw)[[3, 0, 1, 2]],
    )
    source_hashes = _source_hashes()
    source_hashes.update({name: _sha256(Path(__file__).parent / name)
                          for name in ("table_safe_execution.py", "table_clearance.py")})
    metadata = dict(
        prepared.source, schema_version="table_safe_rollout_v1",
        case_path=str(case_path), noise_amplitude_rad=float(noise_amplitude_rad),
        original_case_sha256=_sha256(case_path),
        original_resolved_case=prepared.resolved.case.model_dump(mode="json"),
        resolved_environment=resolved.environment.model_dump(mode="json"),
        environment_config_hash=resolved.environment_config_hash,
        noise_seed=int(noise_seed), noise_mode="uniform_episode_bias",
        execution_protocol=resolved.execution_protocol.model_dump(mode="json"),
        execution_protocol_config_hash=resolved.execution_protocol_config_hash,
        source_file_sha256=source_hashes, runtime_versions=_runtime_versions(),
        intended_world_palm_position_m=prepared.palm_position.tolist(),
        intended_world_palm_quaternion_wxyz=prepared.palm_quaternion.tolist(),
        purpose="small controller validation; not a frozen-model evaluation or training dataset",
        vision="referenced initial RGB-D only; no per-frame RGB-D",
        label_scope="unchanged final grasp/lift/hold scorer; clearance rejection has no label",
        static_scope="nominal pregrasp/approach geometry; dynamic tracking and noisy closure require validation",
        proximal_object_measure_scope="nominal approach endpoint only; no general swept-object certificate",
    )
    try:
        trace = runner.execute(
            prepared.pose, grasp_palm_position_m=prepared.palm_position,
            grasp_palm_quaternion_wxyz=prepared.palm_quaternion,
        )
    except ClearanceRejected as exc:
        output.mkdir(parents=True, exist_ok=True)
        status = ("runtime_clearance_rejected" if isinstance(exc, MotionClearanceRejected)
                  else "clearance_rejected")
        metadata.update(status=status, label=None, error=str(exc))
        metadata["clearance_plan"] = runner.clearance_plan.as_record()
        if runner.runtime_guard_event is not None:
            metadata["runtime_guard_event"] = runner.runtime_guard_event
            np.savez_compressed(output / "runtime_guard_state.npz",
                                qpos=runner.data.qpos.copy(), qvel=runner.data.qvel.copy(),
                                time_s=float(runner.data.time))
        _write_json(output / "collection_metadata.json", metadata)
        return metadata
    _validate_execution(runner, trace)
    outcome = score_trace(trace, resolved.execution_protocol,
                          control_timestep_s=runner.control_timestep_s)
    output.mkdir(parents=True, exist_ok=True)
    runner.write_action_artifacts(output)
    events = annotate_rollout_events(trace, resolved.execution_protocol,
                                    control_timestep_s=runner.control_timestep_s,
                                    outcome=outcome)
    _write_json(output / "events.json", events)
    metadata.update(status="completed", label=int(outcome.success),
                    failure_type=outcome.failure_type.value if outcome.failure_type else None,
                    clearance_plan=runner.clearance_plan.as_record(),
                    runtime_guard=f"{runner.protocol.runtime_table_margin_m * 1000:g} mm table plane "
                    f"margin checked at every {runner.physics_timestep_s * 1000:g} ms physics endpoint; "
                    "margin violations stop execution and have no task label",
                    sampled_hand_table_contact=any(s.hand_table_contact for s in trace.steps),
                    sampled_approach_table_contact=any(
                        s.hand_table_contact for s in trace.steps
                        if s.state.value in {"reset", "load_object", "retarget", "move_to_pregrasp", "approach"}),
                    state_replay="scene.mjb plus actions.npz integration_state (mjSTATE_INTEGRATION)",
                    retargeting_metric_scope="original finger-shape gate; compensated world palm and finger geometry unchanged",
                    privileged_replay_only_fields=["full_qpos", "full_qvel", "integration_state"])
    _write_json(output / "collection_metadata.json", metadata)
    write_evaluation_artifacts(
        output, resolved=resolved, trace=trace, pose_validation=prepared.validation,
        outcome=outcome, final_qpos=runner.data.qpos.copy(), final_qvel=runner.data.qvel.copy(),
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="original saved-proposal v2 case YAML")
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--noise-amplitude", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=9000)
    parser.add_argument("--clearance-m", type=float, default=0.010)
    args = parser.parse_args()
    record = execute_table_safe_case(
        args.case, args.output, project_root=args.project_root,
        noise_amplitude_rad=args.noise_amplitude, noise_seed=args.noise_seed,
        configuration=TableClearanceConfiguration(clearance_m=args.clearance_m),
    )
    print(f'{record["status"]}: label={record["label"]}')
    if record["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
