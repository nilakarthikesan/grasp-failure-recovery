"""Deterministic wrist redundancy planning without changing a proposed grasp.

The two wrist joints sit upstream of the palm. Changing them and compensating
the free forearm pose preserves the world palm and every finger pose. This
planner checks nominal geometry before execution; it does not suppress contacts
or guarantee collision freedom under actuator tracking error or finger noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Any

import mujoco
import numpy as np

from .retargeting import RetargetedHandPose, RetargetingError, reorder_for_mujoco
from .runner import OPENING_JOINTS, force_close_targets


@dataclass(frozen=True)
class TableClearanceConfiguration:
    clearance_m: float = 0.010
    proximal_object_clearance_m: float = 0.001
    grid_points_per_joint: int = 13
    refinement_rounds: int = 3
    check_closing_targets: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.clearance_m) or self.clearance_m <= 0:
            raise ValueError("table clearance must be finite and positive")
        if (
            not np.isfinite(self.proximal_object_clearance_m)
            or self.proximal_object_clearance_m < 0
        ):
            raise ValueError("proximal object clearance must be finite and nonnegative")
        if (
            not isinstance(self.grid_points_per_joint, int)
            or isinstance(self.grid_points_per_joint, bool)
            or not 3 <= self.grid_points_per_joint <= 101
        ):
            raise ValueError("grid_points_per_joint must be an integer in [3, 101]")
        if (
            not isinstance(self.refinement_rounds, int)
            or isinstance(self.refinement_rounds, bool)
            or not 0 <= self.refinement_rounds <= 8
        ):
            raise ValueError("refinement_rounds must be an integer in [0, 8]")


@dataclass(frozen=True)
class TableClearancePlan:
    feasible: bool
    original_wrist_rad: tuple[float, float]
    adjusted_wrist_rad: tuple[float, float] | None
    original_table_clearance_m: float
    minimum_table_clearance_m: float | None
    original_proximal_object_distance_m: float
    minimum_proximal_object_distance_m: float | None
    evaluated_candidates: int
    rejection_reason: str | None
    configuration: TableClearanceConfiguration
    adjusted_pose: RetargetedHandPose | None
    root_position_m: np.ndarray | None
    root_quaternion_wxyz: np.ndarray | None

    def as_record(self) -> dict[str, Any]:
        """JSON-compatible provenance without copying the proposal payload."""

        return {
            "schema_version": "table_clearance_plan_v1",
            "feasible": self.feasible,
            "wrist_joint_names": ["WRJ2", "WRJ1"],
            "original_wrist_rad": list(self.original_wrist_rad),
            "adjusted_wrist_rad": (
                None if self.adjusted_wrist_rad is None else list(self.adjusted_wrist_rad)
            ),
            "original_table_clearance_m": self.original_table_clearance_m,
            "minimum_table_clearance_m": self.minimum_table_clearance_m,
            "original_proximal_object_distance_m": self.original_proximal_object_distance_m,
            "minimum_proximal_object_distance_m": self.minimum_proximal_object_distance_m,
            "evaluated_candidates": self.evaluated_candidates,
            "rejection_reason": self.rejection_reason,
            "configuration": asdict(self.configuration),
            "root_position_m": (
                None if self.root_position_m is None else self.root_position_m.tolist()
            ),
            "root_quaternion_wxyz": (
                None
                if self.root_quaternion_wxyz is None
                else self.root_quaternion_wxyz.tolist()
            ),
            "search_objective": "smallest squared wrist change among evaluated feasible candidates",
            "table_measure": "minimum support above the table top plane, including all active hand geoms",
        }


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quaternion)
    return result.reshape(3, 3)


def _descendant(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def _lowest_world_z(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> float:
    """Exact support along world Z for the compiled hand's convex geometry."""

    kind = int(model.geom_type[geom])
    size = model.geom_size[geom]
    row = data.geom_xmat[geom].reshape(3, 3)[2]
    center = float(data.geom_xpos[geom, 2])
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        return center + float(np.min(model.mesh_vert[start : start + count] @ row))
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        support = np.abs(row) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        support = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        support = size[0] + abs(row[2]) * size[1]
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        support = size[0] * np.linalg.norm(row[:2]) + abs(row[2]) * size[1]
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        support = np.linalg.norm(row * size)
    else:
        raise RetargetingError(f"unsupported active hand geometry type: {kind}")
    return center - float(support)


def plan_table_clearance(
    runner: Any,
    pose: RetargetedHandPose,
    palm_position_m: np.ndarray,
    palm_quaternion_wxyz: np.ndarray,
    *,
    configuration: TableClearanceConfiguration | None = None,
) -> TableClearancePlan:
    """Preserve the grasp and choose a nearby wrist posture above the table.

    Only WRJ2 and WRJ1 change. The 22 finger targets and requested world palm
    pose are preserved. All operations use fresh scratch data; neither runner
    data nor model arrays are modified. Object geometry is taken from the
    runner's current state, so reset/restore the scene before calling.

    The infinite top plane is a conservative check for the finite horizontal
    table: it also rejects geometry below the top outside the table footprint.
    A vertical descending approach has its lowest nominal geometry at this
    endpoint. Other approach paths must be checked separately by the caller.
    """

    config = configuration or TableClearanceConfiguration()
    model = runner.model
    position = np.asarray(palm_position_m, dtype=np.float64)
    quaternion = np.asarray(palm_quaternion_wxyz, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("palm position must be a finite 3-vector")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("palm quaternion must be a finite 4-vector")
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6, rtol=0):
        raise ValueError("palm quaternion must have unit norm")
    names = tuple(runner._hand_joint_names)
    if (
        len(names) != 24
        or len(set(names)) != 24
        or len(pose.joint_names) != 24
        or len(set(pose.joint_names)) != 24
        or set(pose.joint_names) != set(names)
        or not {"WRJ2", "WRJ1"}.issubset(names)
    ):
        raise RetargetingError("planner requires the exact 24 named Shadow joints")
    target = reorder_for_mujoco(pose.joint_names, pose.qpos, names)
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    )
    if np.any(joint_ids < 0) or np.any(
        model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE
    ):
        raise RetargetingError("all 24 hand joints must exist as named hinge joints")
    addresses = model.jnt_qposadr[joint_ids]
    limits = model.jnt_range[joint_ids]
    if (
        not np.all(np.isfinite(target))
        # Historical XML scene builders rounded serialized joint ranges;
        # current live-spec builders preserve them. Keep the original fingers
        # intact within that rounding tolerance. Commands still use the exact
        # same physical-limit clipping as the original runner.
        or np.any(target < limits[:, 0] - 1e-6)
        or np.any(target > limits[:, 1] + 1e-6)
    ):
        raise RetargetingError("input pose must be finite and inside physical joint limits")
    wrist_indices = np.asarray([names.index("WRJ2"), names.index("WRJ1")])
    original = target[wrist_indices].copy()
    wrist_limits = limits[wrist_indices]
    if np.any(original < wrist_limits[:, 0]) or np.any(original > wrist_limits[:, 1]):
        raise RetargetingError("input wrist pose must be inside physical joint limits")
    root = int(runner._root_qpos_address)
    palm = int(runner._palm_body_id)
    table = int(runner._table_geom_id)
    if int(model.geom_type[table]) != mujoco.mjtGeom.mjGEOM_BOX:
        raise RetargetingError("planner requires the registered horizontal box table")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = runner.data.qpos
    mujoco.mj_forward(model, scratch)
    table_rotation = scratch.geom_xmat[table].reshape(3, 3)
    if not np.allclose(table_rotation, np.eye(3), atol=1e-10, rtol=0):
        raise RetargetingError("planner requires an axis-aligned table")
    table_top = float(scratch.geom_xpos[table, 2] + model.geom_size[table, 2])
    hand_geoms = tuple(
        int(geom)
        for geom in range(model.ngeom)
        if int(model.geom_bodyid[geom]) in runner._hand_body_ids
        and (model.geom_contype[geom] or model.geom_conaffinity[geom])
    )
    proximal_geoms = tuple(
        geom for geom in hand_geoms
        if not _descendant(model, int(model.geom_bodyid[geom]), palm)
    )
    object_geoms = tuple(sorted(runner.object_collision_geom_ids))
    if not hand_geoms or not proximal_geoms or not object_geoms:
        raise RetargetingError("planner needs active hand, proximal and object geometry")
    desired_rotation = _rotation(quaternion)
    evaluated: dict[tuple[float, float], tuple[float, float, np.ndarray, np.ndarray]] = {}

    def evaluate(wrist: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        key = tuple(float(value) for value in wrist)
        if key in evaluated:
            return evaluated[key]
        candidate = target.copy()
        candidate[wrist_indices] = wrist
        scratch.qpos[addresses] = candidate
        scratch.qpos[root : root + 7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_forward(model, scratch)
        relative_rotation = scratch.xmat[palm].reshape(3, 3).copy()
        relative_position = scratch.xpos[palm].copy()
        root_rotation = desired_rotation @ relative_rotation.T
        root_position = position - root_rotation @ relative_position
        root_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(root_quaternion, root_rotation.reshape(9))
        scratch.qpos[root : root + 7] = np.r_[root_position, root_quaternion]
        near_open = candidate.copy()
        for index, name in enumerate(names):
            if name in OPENING_JOINTS:
                near_open[index] -= 0.2
        near_open = np.clip(near_open, limits[:, 0], limits[:, 1])
        states = [near_open]
        if config.check_closing_targets:
            params = runner.protocol.parameters
            grasp = near_open + params.grip_command * (candidate - near_open)
            states.extend([
                grasp,
                force_close_targets(names, grasp, limits, params.force_close_delta_rad),
            ])
        clearance = float("inf")
        proximal_distance = float("inf")
        for state in states:
            scratch.qpos[addresses] = state
            mujoco.mj_forward(model, scratch)
            clearance = min(
                clearance,
                min(_lowest_world_z(model, scratch, geom) for geom in hand_geoms) - table_top,
            )
            proximal_distance = min(
                proximal_distance,
                min(
                    float(mujoco.mj_geomDistance(model, scratch, hand, obj, 10.0, None))
                    for hand in proximal_geoms for obj in object_geoms
                ),
            )
        result = (clearance, proximal_distance, root_position, root_quaternion)
        evaluated[key] = result
        return result

    def valid(result: tuple[float, float, np.ndarray, np.ndarray]) -> bool:
        return (
            result[0] >= config.clearance_m
            and result[1] >= config.proximal_object_clearance_m
        )

    original_result = evaluate(original)
    best: np.ndarray | None = original.copy() if valid(original_result) else None

    def ordered(candidates: list[np.ndarray]) -> list[np.ndarray]:
        unique = {tuple(float(value) for value in candidate) for candidate in candidates}
        return [np.asarray(value) for value in sorted(
            unique,
            key=lambda value: (float(np.sum((np.asarray(value) - original) ** 2)), value),
        )]

    if best is None:
        axes = [
            np.unique(np.r_[np.linspace(low, high, config.grid_points_per_joint), value])
            for (low, high), value in zip(wrist_limits, original, strict=True)
        ]
        for candidate in ordered([np.asarray(value) for value in product(*axes)]):
            if valid(evaluate(candidate)):
                best = candidate
                break
        if best is not None:
            spacing = (wrist_limits[:, 1] - wrist_limits[:, 0]) / (
                config.grid_points_per_joint - 1
            )
            for _ in range(config.refinement_rounds):
                axes = [
                    np.clip(value + np.linspace(-step, step, 9), low, high)
                    for value, step, (low, high) in zip(best, spacing, wrist_limits, strict=True)
                ]
                candidates = ordered([best, *[np.asarray(value) for value in product(*axes)]])
                for candidate in candidates:
                    if valid(evaluate(candidate)):
                        best = candidate
                        break
                spacing /= 4.0

    if best is None:
        return TableClearancePlan(
            False, tuple(original), None, original_result[0], None,
            original_result[1], None, len(evaluated),
            "no wrist posture satisfied table and proximal object clearance on the bounded search grid",
            config, None, None, None,
        )
    result = evaluate(best)
    pose_qpos = np.asarray(pose.qpos, dtype=np.float64).copy()
    for name, value in zip(("WRJ2", "WRJ1"), best, strict=True):
        pose_qpos[pose.joint_names.index(name)] = value
    return TableClearancePlan(
        True, tuple(original), tuple(best), original_result[0], result[0],
        original_result[1], result[1], len(evaluated), None, config,
        replace(pose, qpos=pose_qpos), result[2].copy(), result[3].copy(),
    )
