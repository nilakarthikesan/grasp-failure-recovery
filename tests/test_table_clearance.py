"""Geometry regressions for wrist redundancy, independent of ignored runs."""

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import (
    RetargetedHandPose,
    RetargetingError,
    default_dex_urdf_root,
    reorder_for_mujoco,
)
from grasp_failure_prediction.evaluation.runner import ActuatedShadowRunner
from grasp_failure_prediction.evaluation.table_clearance import (
    TableClearanceConfiguration,
    _lowest_world_z,
    plan_table_clearance,
)


@pytest.fixture
def setup():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/shadow_hug_cube_target.json").read_text()
    )
    runner = ActuatedShadowRunner(
        load_protocol_registry().resolve("fixed_grasp_lift_v2"),
        default_dex_urdf_root(),
    )
    runner.reset(
        seed=1,
        object_mass_kg=0.18,
        object_position_m=np.array([1.0, 1.0, 1.0]),
        object_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    pose = RetargetedHandPose(
        tuple(fixture["joint_names"]), np.asarray(fixture["joints"]),
        np.zeros((21, 3)), np.zeros((10, 3)), np.eye(4),
    )
    # A low sideways grasp has a valid palm/finger location but its redundant
    # forearm posture penetrates the table. Use the real compiled Shadow URDF.
    position = np.array([0.0, 0.0, 0.13])
    quaternion = np.array([np.cos(np.pi / 6), 0.0, np.sin(np.pi / 6), 0.0])
    return runner, pose, position, quaternion


def geometry_at_pose(runner, pose, position, quaternion):
    """Independent FK construction; deliberately avoid the planner's helpers."""

    model = runner.model
    data = mujoco.MjData(model)
    data.qpos[:] = runner.data.qpos
    data.qpos[runner._hand_qpos_addresses] = reorder_for_mujoco(
        pose.joint_names, pose.qpos, runner._hand_joint_names
    )
    root = runner._root_qpos_address
    data.qpos[root : root + 7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    root_rotation = rotation.reshape(3, 3) @ data.xmat[runner._palm_body_id].reshape(3, 3).T
    root_position = position - root_rotation @ data.xpos[runner._palm_body_id]
    root_quaternion = np.empty(4)
    mujoco.mju_mat2Quat(root_quaternion, root_rotation.reshape(9))
    data.qpos[root : root + 7] = np.r_[root_position, root_quaternion]
    mujoco.mj_forward(model, data)
    return data


def test_clears_forearm_without_changing_world_grasp_or_finger_targets(setup):
    runner, pose, position, quaternion = setup
    original = geometry_at_pose(runner, pose, position, quaternion)
    plan = plan_table_clearance(runner, pose, position, quaternion)
    assert plan.feasible
    assert plan.original_table_clearance_m < 0
    assert plan.minimum_table_clearance_m >= 0.010
    assert plan.adjusted_wrist_rad != plan.original_wrist_rad
    corrected = geometry_at_pose(runner, plan.adjusted_pose, position, quaternion)
    for name in ("palm", "ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal"):
        body = runner.model.body(name).id
        np.testing.assert_allclose(corrected.xpos[body], original.xpos[body], atol=1e-13, rtol=0)
        np.testing.assert_allclose(corrected.xmat[body], original.xmat[body], atol=1e-13, rtol=0)
    for index, name in enumerate(pose.joint_names):
        if name not in {"WRJ2", "WRJ1"}:
            assert plan.adjusted_pose.qpos[index] == pose.qpos[index]
    for name in ("WRJ2", "WRJ1"):
        joint = runner.model.joint(name).id
        value = plan.adjusted_pose.qpos[pose.joint_names.index(name)]
        assert runner.model.jnt_range[joint, 0] <= value <= runner.model.jnt_range[joint, 1]
    np.testing.assert_allclose(corrected.qpos[:3], plan.root_position_m, atol=1e-13, rtol=0)
    assert plan.as_record()["schema_version"] == "table_clearance_plan_v1"
    json.dumps(plan.as_record(), allow_nan=False)


def test_keeps_an_already_clear_posture_exactly(setup):
    runner, pose, position, quaternion = setup
    position[2] = 0.5
    plan = plan_table_clearance(runner, pose, position, quaternion)
    assert plan.feasible
    assert plan.evaluated_candidates == 1
    assert plan.adjusted_wrist_rad == plan.original_wrist_rad
    np.testing.assert_array_equal(plan.adjusted_pose.qpos, pose.qpos)


def test_planning_is_deterministic_and_does_not_mutate_runner_or_model(setup):
    runner, pose, position, quaternion = setup
    data_fields = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "xpos", "geom_xpos")
    model_fields = ("body_mass", "body_inertia", "geom_contype", "geom_conaffinity", "mesh_vert")
    data_before = {name: getattr(runner.data, name).copy() for name in data_fields}
    model_before = {name: getattr(runner.model, name).copy() for name in model_fields}
    pose_before = pose.qpos.copy()
    first = plan_table_clearance(runner, pose, position, quaternion)
    second = plan_table_clearance(runner, pose, position, quaternion)
    assert first.as_record() == second.as_record()
    for name in data_fields:
        np.testing.assert_array_equal(getattr(runner.data, name), data_before[name])
    for name in model_fields:
        np.testing.assert_array_equal(getattr(runner.model, name), model_before[name])
    np.testing.assert_array_equal(pose.qpos, pose_before)


def test_rejects_unreachable_clearance_without_suppressing_contact(setup):
    runner, pose, position, quaternion = setup
    position[2] = -0.1
    plan = plan_table_clearance(runner, pose, position, quaternion)
    assert not plan.feasible
    assert plan.adjusted_pose is None
    assert plan.root_position_m is None
    assert plan.minimum_table_clearance_m is None
    assert plan.rejection_reason
    assert np.any(runner.model.geom_contype)
    assert np.any(runner.model.geom_conaffinity)


def test_search_avoids_proximal_collision_with_object(setup):
    runner, pose, position, quaternion = setup
    position[2] = 0.5
    original = geometry_at_pose(runner, pose, position, quaternion)
    forearm = runner.model.body("forearm").id
    geom = int(np.flatnonzero(runner.model.geom_bodyid == forearm)[0])
    address = runner._object_qpos_address
    runner.data.qpos[address : address + 3] = original.geom_xpos[geom]
    mujoco.mj_forward(runner.model, runner.data)
    plan = plan_table_clearance(runner, pose, position, quaternion)
    assert plan.feasible
    assert plan.original_proximal_object_distance_m < 0.001
    assert plan.minimum_proximal_object_distance_m >= 0.001
    assert plan.adjusted_wrist_rad != plan.original_wrist_rad


def test_named_mapping_accepts_permuted_pose_and_rejects_duplicates(setup):
    runner, pose, position, quaternion = setup
    plan = plan_table_clearance(runner, pose, position, quaternion)
    permutation = np.arange(24)[::-1]
    permuted = replace(
        pose, joint_names=tuple(pose.joint_names[i] for i in permutation),
        qpos=pose.qpos[permutation],
    )
    alternate = plan_table_clearance(runner, permuted, position, quaternion)
    assert alternate.as_record() == plan.as_record()
    broken = replace(pose, joint_names=(pose.joint_names[1], *pose.joint_names[1:]))
    with pytest.raises(RetargetingError, match="exact 24"):
        plan_table_clearance(runner, broken, position, quaternion)


def test_optional_closing_checks_and_support_match_geometry(setup):
    runner, pose, position, quaternion = setup
    position[2] = 0.5
    plan = plan_table_clearance(
        runner, pose, position, quaternion,
        configuration=TableClearanceConfiguration(check_closing_targets=True),
    )
    assert plan.feasible
    assert plan.configuration.check_closing_targets
    data = geometry_at_pose(runner, pose, position, quaternion)
    # The table's axis aligned box support must give the known zero top and
    # negative bottom; this exercises primitive support independently of mesh.
    table = runner._table_geom_id
    assert _lowest_world_z(runner.model, data, table) == pytest.approx(-0.05)


@pytest.mark.parametrize("kwargs", [
    {"clearance_m": 0}, {"clearance_m": float("nan")},
    {"proximal_object_clearance_m": -0.1}, {"grid_points_per_joint": 2},
    {"refinement_rounds": -1}, {"refinement_rounds": True},
])
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TableClearanceConfiguration(**kwargs)


def test_rejects_nonunit_palm_rotation_and_out_of_range_pose(setup):
    runner, pose, position, quaternion = setup
    with pytest.raises(ValueError, match="unit norm"):
        plan_table_clearance(runner, pose, position, quaternion * 2)
    invalid = pose.qpos.copy()
    invalid[0] = 2
    with pytest.raises(RetargetingError, match="physical joint limits"):
        plan_table_clearance(runner, replace(pose, qpos=invalid), position, quaternion)
