"""A scripted privileged demonstrator for the weighted-container task.

This is a *teacher*, not the learned policy: it is allowed to use privileged
state (the container pose and the target region) because its only job is to
generate successful demonstrations. The learned ACT policy in
``train_act`` never sees privileged information.

The controller is a simple phase machine over OSC pose deltas
``[dx, dy, dz, drx, dry, drz, gripper]`` (rotation held at zero, orientation
kept fixed). Gripper convention: ``+1`` closes, ``-1`` opens (robosuite
default two-finger gripper).
"""

from __future__ import annotations

import enum

import numpy as np


class _Stage(enum.IntEnum):
    APPROACH = 0   # move above the container
    DESCEND = 1    # lower onto the container
    CLOSE = 2      # close the gripper
    LIFT = 3       # raise to transport height
    TRANSPORT = 4  # move horizontally to the target
    LOWER = 5      # lower onto the table at the target
    RELEASE = 6    # open the gripper
    DONE = 7


class ScriptedDemonstrator:
    """Deterministic pick / lift / transport / place teacher."""

    def __init__(
        self,
        table_top: float,
        cube_half_z: float,
        transport_height: float = 0.15,
        pos_gain: float = 10.0,
        xy_tol: float = 0.008,
        z_tol: float = 0.006,
        close_hold_steps: int = 8,
    ) -> None:
        self.table_top = table_top
        self.cube_half_z = cube_half_z
        self.transport_height = transport_height
        self.pos_gain = pos_gain
        self.xy_tol = xy_tol
        self.z_tol = z_tol
        self.close_hold_steps = close_hold_steps
        self.reset()

    def reset(self) -> None:
        self._stage = _Stage.APPROACH
        self._close_counter = 0
        self._release_counter = 0

    @property
    def stage(self) -> _Stage:
        return self._stage

    def _move(self, eef_pos, goal, gripper_cmd) -> np.ndarray:
        delta = np.clip((np.asarray(goal) - np.asarray(eef_pos)) * self.pos_gain, -1.0, 1.0)
        return np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, gripper_cmd])

    def act(self, eef_pos: np.ndarray, cube_pos: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        """Return an OSC pose-delta + gripper action for the current step."""
        eef_pos = np.asarray(eef_pos, dtype=float)
        cube_pos = np.asarray(cube_pos, dtype=float)
        above = np.array([cube_pos[0], cube_pos[1], self.table_top + self.transport_height])
        grasp = np.array([cube_pos[0], cube_pos[1], cube_pos[2]])
        transport_goal = np.array([target_xy[0], target_xy[1], self.table_top + self.transport_height])
        place = np.array([target_xy[0], target_xy[1], self.table_top + self.cube_half_z + 0.005])

        if self._stage == _Stage.APPROACH:
            action = self._move(eef_pos, above, -1.0)
            if np.linalg.norm(eef_pos[:2] - above[:2]) < self.xy_tol and abs(eef_pos[2] - above[2]) < 0.03:
                self._stage = _Stage.DESCEND
            return action

        if self._stage == _Stage.DESCEND:
            action = self._move(eef_pos, grasp, -1.0)
            if abs(eef_pos[2] - grasp[2]) < self.z_tol + self.cube_half_z:
                self._stage = _Stage.CLOSE
            return action

        if self._stage == _Stage.CLOSE:
            self._close_counter += 1
            if self._close_counter >= self.close_hold_steps:
                self._stage = _Stage.LIFT
            return self._move(eef_pos, grasp, 1.0)

        if self._stage == _Stage.LIFT:
            action = self._move(eef_pos, above, 1.0)
            if eef_pos[2] >= above[2] - 0.02:
                self._stage = _Stage.TRANSPORT
            return action

        if self._stage == _Stage.TRANSPORT:
            action = self._move(eef_pos, transport_goal, 1.0)
            if np.linalg.norm(eef_pos[:2] - transport_goal[:2]) < self.xy_tol:
                self._stage = _Stage.LOWER
            return action

        if self._stage == _Stage.LOWER:
            action = self._move(eef_pos, place, 1.0)
            if abs(eef_pos[2] - place[2]) < self.z_tol + self.cube_half_z:
                self._stage = _Stage.RELEASE
            return action

        if self._stage == _Stage.RELEASE:
            self._release_counter += 1
            if self._release_counter >= self.close_hold_steps:
                self._stage = _Stage.DONE
            return self._move(eef_pos, place, -1.0)

        # DONE: hold and stay open.
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
