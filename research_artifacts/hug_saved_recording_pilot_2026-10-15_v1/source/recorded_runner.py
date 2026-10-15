"""Opt-in robot-observable recorder bridge for the fixed actuated Shadow runner.

This subclass does not change commands, controller, scoring or historical
artifacts. Using it for collection still requires a declared mapping/collection
protocol. Importing the bridge does not run physics.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .prediction_recording import PredictionRecorder
from .runner import ActuatedShadowRunner


class RecordedActuatedShadowRunner(ActuatedShadowRunner):
    """Record measured joint states and actual endpoint servo controls.

    Full qpos/qvel use the endpoint time_s. Contact/force diagnostics come from
    MuJoCo's final force evaluation, at diagnostic_contact_time_s = t - dt.
    Neither diagnostic clock nor diagnostic values enter predictor features.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        joint_ids = np.asarray([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self._hand_joint_names
        ])
        if (np.any(joint_ids < 0) or not np.all(self.model.actuator_ctrllimited[self._actuator_ids])
                or self.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)):
            raise ValueError("recording requires named joints with bounded position servos")
        self._recorded_dof_addresses = self.model.jnt_dofadr[joint_ids].copy()
        self._recorded_target_limits = self.model.actuator_ctrlrange[self._actuator_ids].copy()
        self.prediction_recorder = PredictionRecorder(self._hand_joint_names, self._recorded_target_limits)

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self.prediction_recorder = PredictionRecorder(self._hand_joint_names, self._recorded_target_limits)

    def _advance(self, state, root_position, root_quaternion, hand_qpos) -> None:
        requested = np.asarray(hand_qpos, dtype=np.float64).copy()
        interval_start = float(self.data.time)
        super()._advance(state, root_position, root_quaternion, hand_qpos)
        self._capture_prediction_step(state, requested, interval_start)

    def _capture_prediction_step(self, state, requested, interval_start) -> None:
        """Capture an already completed interval; callable in static bridge tests."""
        actual_control = self.data.ctrl[self._actuator_ids].copy()
        # data.ctrl retains the requested value; MuJoCo applies ctrllimited
        # clipping internally. Persist its effective final value explicitly.
        if not np.array_equal(actual_control, requested):
            raise ValueError("endpoint servo controls differ from requested targets")
        applied = np.clip(actual_control, self._recorded_target_limits[:, 0],
                          self._recorded_target_limits[:, 1])
        step = self._trace[-1]
        if step.time_s != float(self.data.time) or step.state is not state:
            raise ValueError("runner trace and measured state endpoints are not aligned")
        self.prediction_recorder.append(
            time_s=float(self.data.time), action_interval_start_s=interval_start,
            phase=state.value, hand_qpos=self.data.qpos[self._hand_qpos_addresses],
            hand_qvel=self.data.qvel[self._recorded_dof_addresses],
            requested_target_rad=requested, applied_target_rad=applied,
            diagnostics={
                "diagnostic_contact_time_s": np.asarray(float(self.data.time) - self.physics_timestep_s),
                "full_qpos": self.data.qpos, "full_qvel": self.data.qvel,
                "contact_count": np.asarray(step.contact_count),
                "hand_object_contact": np.asarray(int(step.hand_object_contact)),
                "hand_table_contact": np.asarray(int(step.hand_table_contact)),
                "object_table_contact": np.asarray(int(step.object_table_contact)),
                "maximum_contact_normal_force_n": np.asarray(step.maximum_contact_normal_force_n),
            },
        )
