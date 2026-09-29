"""Weighted-container pick / lift / transport / place task.

A thin wrapper around robosuite's single-object ``Lift`` environment (Panda +
two-finger gripper on MuJoCo). The wrapper adds everything the Part I contract
needs on top of the base task:

* a rigid **container** (the cube) whose mass, inertia, and friction can be
  overridden as *privileged metadata* (never a policy input);
* a **target placement region** offset from the container's start, so every
  episode requires a real transport;
* per-step **phase** labels, **success/failure** detection, and separate
  ``collision`` / ``excess_force`` / ``controller_fault`` flags;
* the policy-visible **observation contract** (front + wrist RGB, joint state,
  end-effector pose, gripper state) cleanly separated from privileged truth;
* a restorable **state snapshot** each step for Part III branching.

Rendering is offscreen; on macOS set ``MUJOCO_GL=cgl``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from grasp_failure_prediction.part1 import snapshot as snap
from grasp_failure_prediction.part1.config import (
    ObservationContract,
    Part1Config,
    Phase,
    default_config,
)
from grasp_failure_prediction.part1.record import StepRecord


class WeightedContainerTask:
    """Part I grasp-and-transport task over a mass/friction-tunable container."""

    def __init__(
        self,
        config: Optional[Part1Config] = None,
        mass_kg: Optional[float] = None,
        friction: Optional[np.ndarray] = None,
        transport_distance: float = 0.15,
        render_images: bool = True,
        capture_snapshots: bool = True,
    ) -> None:
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        self.config = config or default_config()
        self.contract = self.config.observation
        self.render_images = render_images
        self.capture_snapshots = capture_snapshots
        self.transport_distance = transport_distance

        # Physics metadata (labels, not policy inputs).
        self.object_mass = float(
            mass_kg if mass_kg is not None else self.config.physics.nominal_mass_kg
        )
        self.object_friction = np.array(
            friction
            if friction is not None
            else [
                self.config.physics.nominal_sliding_friction,
                self.config.physics.nominal_torsional_friction,
                self.config.physics.nominal_rolling_friction,
            ],
            dtype=np.float64,
        )

        controller = load_composite_controller_config(robot=self.config.task.robot)
        self._env = suite.make(
            env_name="Lift",
            robots=self.config.task.robot,
            controller_configs=controller,
            has_renderer=False,
            has_offscreen_renderer=render_images,
            use_camera_obs=render_images,
            camera_names=list(self.config.task.camera_names) if render_images else None,
            camera_heights=self.config.task.camera_height,
            camera_widths=self.config.task.camera_width,
            control_freq=self.config.task.control_freq,
            horizon=self.config.task.horizon,
            ignore_done=True,
        )
        self._cube = self._env.cube
        self._table_top = float(self._env.table_offset[2])
        # robosuite rebuilds ``sim`` on every reset(), so model/data handles and
        # the (size-randomized) geom are rebound after each reset via _bind_sim.
        self._bind_sim()

        # Episode state.
        self._step_index = 0
        self._seed: Optional[int] = None
        self._start_xy = np.zeros(2)
        self._target_xy = np.zeros(2)
        self._grasped_once = False
        self._lifted_once = False
        self._phase = Phase.REACH
        self._terminal = False

    # ------------------------------------------------------------------ props
    @property
    def action_dim(self) -> int:
        return self._env.action_dim

    @property
    def dt(self) -> float:
        return 1.0 / self.config.task.control_freq

    @property
    def target_xy(self) -> np.ndarray:
        return self._target_xy.copy()

    @property
    def table_top(self) -> float:
        return self._table_top

    @property
    def cube_half_z(self) -> float:
        return self._cube_half_z

    def container_position(self) -> np.ndarray:
        """Current container xyz (privileged; for teachers/analysis, not policy)."""
        try:
            raw = self._env._get_observations(force_update=True)
        except (AttributeError, TypeError):
            raw = self._env._get_observations()
        return np.array(raw["cube_pos"])

    # --------------------------------------------------------------- bind sim
    def _bind_sim(self) -> None:
        """(Re)bind model/data handles and container ids to the current sim."""
        self._cube = self._env.cube
        self._model = self._env.sim.model
        self._data = self._env.sim.data
        self._body_id = self._model.body_name2id(self._cube.root_body)
        self._geom_ids = [
            self._model.geom_name2id(g) for g in self._cube.contact_geoms
        ]
        self._cube_joint = self._cube.joints[0]
        self._cube_half_z = float(self.config.physics.nominal_half_size)

    # ---------------------------------------------------------------- physics
    def _apply_physics_overrides(self) -> None:
        """Override container mass/inertia/friction (privileged metadata).

        Writes go to the *raw* mjModel so the robosuite binding wrapper, the
        physics step, and the state-snapshot reader all see the same values.
        """
        import mujoco

        m = getattr(self._model, "_model", self._model)
        d = getattr(self._data, "_data", self._data)
        half = float(self.config.physics.nominal_half_size)

        # Pin the container to a fixed size so it is identical every episode.
        for gid in self._geom_ids:
            m.geom_size[gid][:3] = [half, half, half]
            m.geom_friction[gid] = self.object_friction
        # Fixed mass and a matching solid-box inertia (Ixx = 2/3 m h^2 for a cube).
        m.body_mass[self._body_id] = self.object_mass
        box_inertia = (2.0 / 3.0) * self.object_mass * (half ** 2)
        m.body_inertia[self._body_id] = np.array([box_inertia] * 3)

        # Reseat the container on the table (its spawn z used the old half-size).
        addr = self._model.get_joint_qpos_addr(self._cube_joint)
        start = addr[0] if isinstance(addr, tuple) else addr
        d.qpos[start + 2] = self._table_top + half
        d.qvel[
            self._model.get_joint_qvel_addr(self._cube_joint)[0]
            if isinstance(self._model.get_joint_qvel_addr(self._cube_joint), tuple)
            else self._model.get_joint_qvel_addr(self._cube_joint)
        ] = 0.0
        mujoco.mj_forward(m, d)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        raw = self._env.reset()
        # sim was rebuilt; rebind handles before applying overrides.
        self._bind_sim()
        self._apply_physics_overrides()
        # Refresh observation after applying overrides / recompute.
        try:
            raw = self._env._get_observations(force_update=True)
        except (AttributeError, TypeError):
            pass

        self._step_index = 0
        self._grasped_once = False
        self._lifted_once = False
        self._phase = Phase.REACH
        self._terminal = False
        self._start_xy = np.array(raw["cube_pos"][:2])
        # Target is a fixed transport offset from the start (guarantees motion).
        self._target_xy = self._start_xy + np.array([self.transport_distance, 0.0])
        return self._policy_obs(raw)

    # -------------------------------------------------------------- obs split
    def _policy_obs(self, raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.render_images:
            front = raw["frontview_image"].astype(np.uint8)
            wrist = raw["robot0_eye_in_hand_image"].astype(np.uint8)
        else:
            shape = (self.config.task.camera_height, self.config.task.camera_width, 3)
            front = np.zeros(shape, dtype=np.uint8)
            wrist = np.zeros(shape, dtype=np.uint8)
        obs = {
            "rgb_front": front,
            "rgb_wrist": wrist,
            "joint_pos": raw["robot0_joint_pos"].astype(np.float32),
            "joint_vel": raw["robot0_joint_vel"].astype(np.float32),
            "eef_pose": np.concatenate(
                [raw["robot0_eef_pos"], raw["robot0_eef_quat"]]
            ).astype(np.float32),
            "gripper_state": raw["robot0_gripper_qpos"].astype(np.float32),
        }
        # Guard: the policy observation must contain no privileged key.
        self.contract.assert_no_leak(obs.keys())
        return obs

    def _privileged(self, raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        n_contacts, max_force, total_force, mean_dist = self._contact_summary()
        return {
            "object_pose": np.concatenate(
                [raw["cube_pos"], raw["cube_quat"]]
            ).astype(np.float64),
            "object_mass": np.array([self.object_mass], dtype=np.float64),
            "object_friction": self.object_friction.astype(np.float64),
            "contacts": np.array(
                [n_contacts, max_force, total_force, mean_dist], dtype=np.float64
            ),
        }

    # --------------------------------------------------------------- contacts
    def _contact_summary(self) -> Tuple[float, float, float, float]:
        """Summarize container contacts (privileged): count, max/total force, mean dist."""
        import mujoco

        m = getattr(self._model, "_model", self._model)
        d = getattr(self._data, "_data", self._data)
        geom_set = set(self._geom_ids)
        forces = []
        dists = []
        buf = np.zeros(6, dtype=np.float64)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in geom_set or c.geom2 in geom_set:
                mujoco.mj_contactForce(m, d, i, buf)
                forces.append(abs(float(buf[0])))  # normal component
                dists.append(float(c.dist))
        if not forces:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(len(forces)),
            float(np.max(forces)),
            float(np.sum(forces)),
            float(np.mean(dists)),
        )

    # ---------------------------------------------------------------- helpers
    def _is_grasped(self) -> bool:
        return bool(
            self._env._check_grasp(
                gripper=self._env.robots[0].gripper,
                object_geoms=self._cube.contact_geoms,
            )
        )

    def _cube_pos(self, raw: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array(raw["cube_pos"])

    def _lift_ok(self, cube_pos: np.ndarray) -> bool:
        return (cube_pos[2] - self._table_top) >= self.config.task.lift_height

    def _xy_to_target(self, cube_pos: np.ndarray) -> float:
        return float(np.linalg.norm(cube_pos[:2] - self._target_xy))

    def _object_lost(self, cube_pos: np.ndarray) -> bool:
        # Fell off / below the table, or left the workspace horizontally.
        if cube_pos[2] < self._table_top - 0.05:
            return True
        if np.linalg.norm(cube_pos[:2] - self._start_xy) > 0.5:
            return True
        return False

    def _update_phase(self, grasped: bool, lifted: bool, xy_dist: float) -> None:
        if self._terminal:
            self._phase = Phase.DONE
            return
        tol = self.config.task.place_tolerance
        if grasped:
            self._grasped_once = True
        if lifted:
            self._lifted_once = True
        # Forward-only phase progression.
        candidate = self._phase
        if not self._grasped_once:
            candidate = Phase.REACH
        elif self._grasped_once and not self._lifted_once:
            candidate = Phase.GRASP
        elif lifted and xy_dist > tol:
            candidate = Phase.TRANSPORT if self._phase >= Phase.TRANSPORT else Phase.LIFT
        elif self._lifted_once and xy_dist <= tol:
            candidate = Phase.PLACE
        self._phase = Phase(max(int(self._phase), int(candidate)))

    # ------------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], StepRecord]:
        controller_fault = not np.all(np.isfinite(action))
        if controller_fault:
            action = np.nan_to_num(action)
        raw, _reward, _done, _info = self._env.step(action)
        self._step_index += 1

        cube_pos = self._cube_pos(raw)
        grasped = self._is_grasped()
        lifted = self._lift_ok(cube_pos)
        xy_dist = self._xy_to_target(cube_pos)
        _, max_force, _, _ = self._contact_summary()

        excess_force = max_force > self.config.task.excess_force_newtons
        lost = self._object_lost(cube_pos)

        # Success: at target, resting on the table, and released.
        resting = abs(cube_pos[2] - (self._table_top + self._cube_half_z)) < 0.02
        at_target = xy_dist <= self.config.task.place_tolerance
        success = bool(
            self._lifted_once and at_target and resting and not grasped
        )
        # Failure: object loss, or horizon reached without success.
        horizon_reached = self._step_index >= self.config.task.horizon
        failure = bool(lost or (horizon_reached and not success))

        if success or failure:
            self._terminal = True
        self._update_phase(grasped, lifted, xy_dist)

        obs = self._policy_obs(raw)
        privileged = self._privileged(raw)
        snapshot = None
        if self.capture_snapshots:
            snapshot = snap.capture(
                self._env.sim,
                tracked_body_ids=[self._body_id],
                tracked_geom_ids=self._geom_ids,
                extra={"step_index": self._step_index, "seed": self._seed or -1},
            )
        record = StepRecord(
            t=self._step_index * self.dt,
            step_index=self._step_index,
            phase=int(self._phase),
            obs=obs,
            action=np.asarray(action, dtype=np.float32),
            privileged=privileged,
            success=success,
            failure=failure,
            object_lost=bool(lost),
            collision=self._robot_table_collision(),
            excess_force=bool(excess_force),
            controller_fault=bool(controller_fault),
            snapshot=snapshot,
        )
        return obs, record

    def _gripper_contact_geoms(self):
        """Return gripper contact geom names, handling dict or single gripper."""
        gripper = self._env.robots[0].gripper
        if isinstance(gripper, dict):
            geoms = []
            for g in gripper.values():
                geoms.extend(getattr(g, "contact_geoms", []))
            return set(geoms)
        return set(getattr(gripper, "contact_geoms", []))

    def _robot_table_collision(self) -> bool:
        """Best-effort: a non-gripper robot geom contacting the table."""
        m = getattr(self._model, "_model", self._model)
        d = getattr(self._data, "_data", self._data)
        gripper_geoms = self._gripper_contact_geoms()
        for i in range(d.ncon):
            c = d.contact[i]
            names = {self._geom_name(g) for g in (c.geom1, c.geom2)}
            robot_geom = any(n and n.startswith("robot0_") for n in names)
            table_geom = any(n and "table" in n for n in names)
            gripper = any(n in gripper_geoms for n in names)
            if robot_geom and table_geom and not gripper:
                return True
        return False

    def _geom_name(self, gid: int):
        try:
            return self._model.geom_id2name(gid)
        except Exception:  # noqa: BLE001
            return None

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> snap.SimSnapshot:
        return snap.capture(
            self._env.sim,
            tracked_body_ids=[self._body_id],
            tracked_geom_ids=self._geom_ids,
        )

    def restore(self, snapshot: snap.SimSnapshot) -> None:
        snap.restore(self._env.sim, snapshot)

    def close(self) -> None:
        self._env.close()
