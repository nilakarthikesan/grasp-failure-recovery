"""Deterministic fixed-protocol runner for the official Shadow Hand model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tempfile
from typing import Callable
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from .pose_validation import _body_id, load_shadow_model
from .registry import ExecutionProtocolSpec
from .retargeting import RetargetedHandPose, RetargetingError, reorder_for_mujoco


class ExecutionState(str, Enum):
    RESET = "reset"
    LOAD_OBJECT = "load_object"
    RETARGET = "retarget"
    MOVE_TO_PREGRASP = "move_to_pregrasp"
    APPROACH = "approach"
    CLOSE_FINGERS = "close_fingers"
    LIFT = "lift"
    HOLD = "hold"
    SCORE = "score"


@dataclass(frozen=True)
class RunnerStep:
    index: int
    time_s: float
    state: ExecutionState
    palm_position_m: np.ndarray
    hand_qpos: np.ndarray
    object_position_m: np.ndarray
    contact_count: int
    hand_object_contact: bool
    hand_table_contact: bool
    object_table_contact: bool


@dataclass(frozen=True)
class ExecutionTrace:
    steps: tuple[RunnerStep, ...]

    @property
    def states(self) -> tuple[ExecutionState, ...]:
        result: list[ExecutionState] = []
        for step in self.steps:
            if not result or result[-1] is not step.state:
                result.append(step.state)
        return tuple(result)


def _temporary_mjcf(base_model: mujoco.MjModel) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
    handle.close()
    path = Path(handle.name)
    mujoco.mj_saveLastXML(str(path), base_model)
    return path


def build_tabletop_model(
    urdf_root: str | Path,
    *,
    physics_timestep_s: float = 0.002,
) -> mujoco.MjModel:
    """Add a table, a free rigid object, lighting, and a camera to Shadow URDF."""

    base = load_shadow_model(urdf_root, floating_root=True)
    temporary = _temporary_mjcf(base)
    try:
        tree = ET.parse(temporary)
        root = tree.getroot()
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("timestep", str(physics_timestep_s))
        option.set("gravity", "0 0 -9.81")

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise RetargetingError("compiled Shadow model has no worldbody")
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": "key_light",
                "pos": "0 -0.5 1.2",
                "dir": "0 0.4 -1",
                "directional": "true",
            },
        )
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": "front",
                "pos": "0 -0.7 0.55",
                "quat": "0.9238795 0.3826834 0 0",
            },
        )
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "table",
                "type": "box",
                "size": "0.4 0.4 0.025",
                "pos": "0 0 -0.025",
                "rgba": "0.55 0.45 0.35 1",
                "friction": "1.0 0.005 0.0001",
            },
        )
        object_body = ET.SubElement(
            worldbody, "body", {"name": "object", "pos": "0 0 0.03"}
        )
        ET.SubElement(object_body, "freejoint", {"name": "object_free"})
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": "object_geom",
                "type": "box",
                "size": "0.025 0.025 0.025",
                "mass": "0.18",
                "rgba": "0.15 0.45 0.8 1",
                "friction": "1.0 0.005 0.0001",
            },
        )
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        return mujoco.MjModel.from_xml_path(str(temporary))
    finally:
        temporary.unlink(missing_ok=True)


def _joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RetargetingError(f"MuJoCo model is missing joint {name!r}")
    return int(model.jnt_qposadr[joint_id])


def _quat_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quaternion


class AdroitShadowRunner:
    """Run the reviewed fixed grasp/lift sequence without a learned policy."""

    def __init__(
        self,
        protocol: ExecutionProtocolSpec,
        urdf_root: str | Path,
        *,
        physics_timestep_s: float = 0.002,
        control_timestep_s: float = 0.04,
    ) -> None:
        ratio = control_timestep_s / physics_timestep_s
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("control timestep must be an integer number of physics steps")
        self.protocol = protocol
        configured_duration = sum(
            (
                protocol.parameters.approach_duration_s,
                protocol.parameters.close_duration_s,
                protocol.parameters.lift_duration_s,
                protocol.parameters.hold_duration_s,
            )
        )
        if configured_duration > protocol.parameters.total_timeout_s:
            raise ValueError("execution protocol phases exceed total_timeout_s")
        self.physics_timestep_s = physics_timestep_s
        self.control_timestep_s = control_timestep_s
        self.physics_steps_per_control = int(round(ratio))
        self.model = build_tabletop_model(
            urdf_root, physics_timestep_s=physics_timestep_s
        )
        self.data = mujoco.MjData(self.model)
        self._root_qpos_address = _joint_qpos_address(self.model, "world_joint")
        self._object_qpos_address = _joint_qpos_address(self.model, "object_free")
        self._palm_body_id = _body_id(self.model, "palm")
        self._object_body_id = _body_id(self.model, "object")
        self._table_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "table"
        )
        self._object_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
        )
        forearm_body_id = _body_id(self.model, "forearm")
        self._hand_body_ids = {
            body_id
            for body_id in range(self.model.nbody)
            if self._is_descendant(body_id, forearm_body_id)
        }
        self._hand_joint_names = tuple(
            str(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index))
            for index in range(1, 25)
        )
        self._hand_qpos_addresses = np.asarray(
            [_joint_qpos_address(self.model, name) for name in self._hand_joint_names]
        )
        self._step_index = 0
        self._trace: list[RunnerStep] = []

    def reset(
        self,
        *,
        seed: int,
        object_mass_kg: float,
        object_position_m: np.ndarray,
        object_orientation_wxyz: np.ndarray,
    ) -> None:
        del seed  # MuJoCo dynamics here are deterministic; retained by the case record.
        mujoco.mj_resetData(self.model, self.data)
        base_mass = float(self.model.body_mass[self._object_body_id])
        scale = object_mass_kg / base_mass
        self.model.body_mass[self._object_body_id] *= scale
        self.model.body_inertia[self._object_body_id] *= scale
        address = self._object_qpos_address
        self.data.qpos[address : address + 3] = object_position_m
        self.data.qpos[address + 3 : address + 7] = object_orientation_wxyz
        mujoco.mj_forward(self.model, self.data)
        self._step_index = 0
        self._trace = []

    def _is_descendant(self, body_id: int, ancestor_id: int) -> bool:
        current = body_id
        while current > 0:
            if current == ancestor_id:
                return True
            current = int(self.model.body_parentid[current])
        return False

    def _contact_flags(self) -> tuple[bool, bool, bool]:
        hand_object = False
        hand_table = False
        object_table = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_pair = {int(contact.geom1), int(contact.geom2)}
            body_pair = {
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            }
            if self._object_geom_id in geom_pair and body_pair & self._hand_body_ids:
                hand_object = True
            if self._table_geom_id in geom_pair and body_pair & self._hand_body_ids:
                hand_table = True
            if geom_pair == {self._table_geom_id, self._object_geom_id}:
                object_table = True
        return hand_object, hand_table, object_table

    def _record(self, state: ExecutionState) -> None:
        object_address = self._object_qpos_address
        hand_object, hand_table, object_table = self._contact_flags()
        self._trace.append(
            RunnerStep(
                index=self._step_index,
                time_s=float(self.data.time),
                state=state,
                palm_position_m=self.data.xpos[self._palm_body_id].copy(),
                hand_qpos=self.data.qpos[self._hand_qpos_addresses].copy(),
                object_position_m=self.data.qpos[
                    object_address : object_address + 3
                ].copy(),
                contact_count=int(self.data.ncon),
                hand_object_contact=hand_object,
                hand_table_contact=hand_table,
                object_table_contact=object_table,
            )
        )
        self._step_index += 1

    def _set_kinematic_targets(
        self,
        root_position: np.ndarray,
        root_quaternion: np.ndarray,
        hand_qpos: np.ndarray,
    ) -> None:
        root = self._root_qpos_address
        self.data.qpos[root : root + 3] = root_position
        self.data.qpos[root + 3 : root + 7] = root_quaternion
        self.data.qpos[self._hand_qpos_addresses] = hand_qpos
        self.data.qvel[:30] = 0.0

    def _advance(
        self,
        state: ExecutionState,
        root_position: np.ndarray,
        root_quaternion: np.ndarray,
        hand_qpos: np.ndarray,
    ) -> None:
        for _ in range(self.physics_steps_per_control):
            self._set_kinematic_targets(root_position, root_quaternion, hand_qpos)
            mujoco.mj_step(self.model, self.data)
        self._set_kinematic_targets(root_position, root_quaternion, hand_qpos)
        mujoco.mj_forward(self.model, self.data)
        self._record(state)

    def _control_steps(self, duration_s: float) -> int:
        return max(1, int(round(duration_s / self.control_timestep_s)))

    def _root_pose_for_palm(
        self,
        palm_position: np.ndarray,
        palm_quaternion_wxyz: np.ndarray,
        wrist_qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self._hand_qpos_addresses] = wrist_qpos
        root = self._root_qpos_address
        self.data.qpos[root : root + 3] = 0.0
        self.data.qpos[root + 3 : root + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        root_to_palm_rotation = self.data.xmat[self._palm_body_id].reshape(3, 3).copy()
        root_to_palm_translation = self.data.xpos[self._palm_body_id].copy()
        target_rotation = _quat_to_matrix(palm_quaternion_wxyz)
        root_rotation = target_rotation @ root_to_palm_rotation.T
        root_position = palm_position - root_rotation @ root_to_palm_translation
        return root_position, _matrix_to_quat(root_rotation)

    def execute(
        self,
        pose: RetargetedHandPose,
        *,
        grasp_palm_position_m: np.ndarray,
        grasp_palm_quaternion_wxyz: np.ndarray,
        approach_direction_world: np.ndarray = np.array([0.0, 0.0, -1.0]),
        step_callback: Callable[[RunnerStep], None] | None = None,
    ) -> ExecutionTrace:
        params = self.protocol.parameters
        direction = np.asarray(approach_direction_world, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        target_hand = reorder_for_mujoco(
            pose.joint_names, pose.qpos, self._hand_joint_names
        )
        lower = self.model.jnt_range[1:25, 0]
        upper = self.model.jnt_range[1:25, 1]
        open_hand = np.clip(np.zeros(24), lower, upper)
        open_hand[:2] = target_hand[:2]
        closed_hand = open_hand + params.grip_command * (target_hand - open_hand)

        pregrasp_palm = (
            np.asarray(grasp_palm_position_m, dtype=np.float64)
            - direction * params.pregrasp_distance_m
        )
        pre_root, root_quaternion = self._root_pose_for_palm(
            pregrasp_palm, grasp_palm_quaternion_wxyz, open_hand
        )
        grasp_root, _ = self._root_pose_for_palm(
            np.asarray(grasp_palm_position_m, dtype=np.float64),
            grasp_palm_quaternion_wxyz,
            open_hand,
        )
        lift_root = grasp_root + np.array([0.0, 0.0, params.lift_height_m])

        for state in (
            ExecutionState.RESET,
            ExecutionState.LOAD_OBJECT,
            ExecutionState.RETARGET,
            ExecutionState.MOVE_TO_PREGRASP,
        ):
            self._advance(state, pre_root, root_quaternion, open_hand)
            if step_callback is not None:
                step_callback(self._trace[-1])

        for step in range(1, self._control_steps(params.approach_duration_s) + 1):
            alpha = step / self._control_steps(params.approach_duration_s)
            root_position = (1 - alpha) * pre_root + alpha * grasp_root
            self._advance(
                ExecutionState.APPROACH, root_position, root_quaternion, open_hand
            )
            if step_callback is not None:
                step_callback(self._trace[-1])

        for step in range(1, self._control_steps(params.close_duration_s) + 1):
            alpha = step / self._control_steps(params.close_duration_s)
            hand = (1 - alpha) * open_hand + alpha * closed_hand
            self._advance(
                ExecutionState.CLOSE_FINGERS, grasp_root, root_quaternion, hand
            )
            if step_callback is not None:
                step_callback(self._trace[-1])

        for step in range(1, self._control_steps(params.lift_duration_s) + 1):
            alpha = step / self._control_steps(params.lift_duration_s)
            root_position = (1 - alpha) * grasp_root + alpha * lift_root
            self._advance(
                ExecutionState.LIFT, root_position, root_quaternion, closed_hand
            )
            if step_callback is not None:
                step_callback(self._trace[-1])

        for _ in range(self._control_steps(params.hold_duration_s)):
            self._advance(
                ExecutionState.HOLD, lift_root, root_quaternion, closed_hand
            )
            if step_callback is not None:
                step_callback(self._trace[-1])
        self._advance(ExecutionState.SCORE, lift_root, root_quaternion, closed_hand)
        if step_callback is not None:
            step_callback(self._trace[-1])
        return ExecutionTrace(tuple(self._trace))
