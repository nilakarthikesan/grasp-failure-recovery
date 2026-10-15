from dataclasses import replace

import mujoco
import numpy as np
import pytest

pytest.importorskip("scipy")

from grasp_failure_prediction.evaluation.native_mano import NativeManoGeometry, NativeManoStaticModel


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


@pytest.fixture
def geometry():
    parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
    joints = np.zeros((16, 3))
    for finger in range(5):
        for segment in range(3):
            joints[1 + 3 * finger + segment] = [(finger - 2) * .018, .02 + segment * .024, .001 * finger]
    tip_indices = np.array([15, 3, 6, 12, 9])
    tips = joints[tip_indices] + [0, .021, .003]
    cube = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    vertices, owners = [], []
    for j in range(16):
        count = 49 if j < 10 else 48  # Exactly 778 source vertex slots.
        for k in range(count):
            offsets = cube[k % 8] * ([.041, .018, .006] if j == 0 else [.004, .007, .003])
            vertices.append(joints[j] + offsets)
            owners.append(j)
    weights = np.eye(16)[owners]
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return NativeManoGeometry(joints, parents, np.array(vertices), faces, weights, tips, tip_indices)


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native MANO diagnostics must not integrate physics")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)


def independent_fk(geometry, translation, root, local):
    positions = np.zeros((16, 3))
    frames = np.zeros((16, 3, 3))
    positions[0], frames[0] = translation, root
    for j in range(1, 16):
        parent = geometry.parents[j]
        delta = geometry.neutral_joints[j] - geometry.neutral_joints[parent]
        positions[j] = positions[parent] + frames[parent] @ delta
        frames[j] = frames[parent] @ local[j - 1]
    tips = np.array([positions[j] + frames[j] @ (geometry.neutral_tips[t] - geometry.neutral_joints[j])
                     for t, j in enumerate(geometry.tip_joint_indices)])
    ends = []
    for j in range(1, 16):
        children = np.flatnonzero(geometry.parents == j)
        target = (geometry.neutral_joints[children[0]] if len(children) else
                  geometry.neutral_tips[np.flatnonzero(geometry.tip_joint_indices == j)[0]])
        ends.append([positions[j], positions[j] + frames[j] @ (target - geometry.neutral_joints[j])])
    return positions, tips, np.array(ends)


def test_nontrivial_hierarchy_and_compiled_geometries_match_independent_fk(geometry):
    hand = NativeManoStaticModel(geometry)
    translation = np.array([.19, -.12, .33])
    root = rotation([2, -1, 3], .73)
    local = np.array([rotation([1 + j % 3, (-1)**j, .4], (j - 7) * .09) for j in range(15)])
    hand.set_pose(translation, root, local)
    actual = hand.readback()
    joints, tips, ends = independent_fk(geometry, translation, root, local)
    np.testing.assert_allclose(actual.joint_positions_world, joints, atol=2e-15, rtol=0)
    np.testing.assert_allclose(actual.tip_positions_world, tips, atol=2e-15, rtol=0)
    np.testing.assert_allclose(actual.capsule_endpoints_world, ends, atol=2e-15, rtol=0)
    np.testing.assert_allclose(actual.wrist_rotation_world, root, atol=2e-15, rtol=0)
    np.testing.assert_array_equal(actual.capsule_joint_indices, np.arange(1, 16))
    owners = geometry.skinning_weights.argmax(axis=1)
    expected_radius = []
    for j, (start, end) in enumerate(ends, 1):
        # Independent neutral-axis calculation, no model size input.
        children = np.flatnonzero(geometry.parents == j)
        destination = (geometry.neutral_joints[children[0]] if len(children) else
                       geometry.neutral_tips[np.flatnonzero(geometry.tip_joint_indices == j)[0]])
        direction = destination - geometry.neutral_joints[j]
        direction /= np.linalg.norm(direction)
        offsets = geometry.neutral_vertices[owners == j] - geometry.neutral_joints[j]
        squared_distances = [np.dot(np.cross(offset, direction), np.cross(offset, direction)) for offset in offsets]
        expected_radius.append(np.sqrt(sum(squared_distances) / len(squared_distances)))
    np.testing.assert_allclose(actual.capsule_radii, expected_radius, atol=1e-15, rtol=0)
    # Compare convex-hull support over independent non-axis-aligned directions.
    source_palm = geometry.neutral_vertices[owners == 0] @ root.T + translation
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 3, 2], [4, -3, -2]], float)
    for direction in directions:
        direction /= np.linalg.norm(direction)
        assert np.max(actual.palm_vertices_world @ direction) == pytest.approx(np.max(source_palm @ direction), abs=1e-8)
        assert np.min(actual.palm_vertices_world @ direction) == pytest.approx(np.min(source_palm @ direction), abs=1e-8)
    assert hand.data.time == 0
    assert hand.model.nu == 0
    assert hand.model.nq == 67 and hand.model.nv == 51
    assert not hand.model.geom_contype.any() and not hand.model.geom_conaffinity.any()
    assert np.all(hand.data.qvel == 0)


def test_repeated_pose_is_independent_of_previous_pose(geometry):
    hand = NativeManoStaticModel(geometry)
    identity = np.repeat(np.eye(3)[None], 15, axis=0)
    hand.set_pose([.1, .2, .3], np.eye(3), identity)
    first = hand.readback()
    hand.set_pose([-.2, .1, .4], rotation([1, 2, 3], .9), np.array([rotation([1, 0, 0], .3)] * 15))
    hand.set_pose([.1, .2, .3], np.eye(3), identity)
    second = hand.readback()
    for field in first.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    second.joint_positions_world[:] = 100
    assert np.max(hand.readback().joint_positions_world) < 1


def test_source_geometry_is_copied_and_readonly(geometry):
    source = geometry.neutral_joints.copy()
    copied = replace(geometry, neutral_joints=source)
    source[:] = 99
    assert copied.neutral_joints[0, 0] == 0
    with pytest.raises(ValueError):
        copied.neutral_joints[0, 0] = 2


@pytest.mark.parametrize("field,value", [
    ("neutral_joints", np.zeros((15, 3))),
    ("neutral_vertices", np.zeros((777, 3))),
    ("parents", np.array([-1, 2, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])),
    ("parents", np.array([-1., 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])),
    ("tip_joint_indices", np.array([15, 3, 6, 12, 12])),
    ("skinning_weights", np.zeros((778, 16))),
    ("neutral_tips", np.full((5, 3), np.nan)),
    ("faces", np.array([[0, 1, 778]])),
    ("faces", np.array([[0, 1, 1]])),
    ("faces", np.array([[0.0, 1.0, 2.0]])),
])
def test_malformed_source_is_rejected(geometry, field, value):
    with pytest.raises(ValueError):
        replace(geometry, **{field: value})


def test_degenerate_palm_and_bone_are_rejected(geometry):
    vertices = geometry.neutral_vertices.copy()
    vertices[geometry.skinning_weights.argmax(axis=1) == 0, 2] = 0
    with pytest.raises(ValueError):
        NativeManoStaticModel(replace(geometry, neutral_vertices=vertices, faces=np.array([[49, 50, 51]])))
    tips = geometry.neutral_tips.copy()
    tips[0] = geometry.neutral_joints[geometry.tip_joint_indices[0]]
    with pytest.raises(ValueError, match="positive length"):
        NativeManoStaticModel(replace(geometry, neutral_tips=tips))


@pytest.mark.parametrize("position,root,local", [
    ([0, 0, np.inf], np.eye(3), np.repeat(np.eye(3)[None], 15, axis=0)),
    ([0, 0, 0], np.diag([1, 1, -1]), np.repeat(np.eye(3)[None], 15, axis=0)),
    ([0, 0, 0], np.eye(3) * 1.001, np.repeat(np.eye(3)[None], 15, axis=0)),
    ([0, 0, 0], np.eye(3), np.repeat(np.eye(3)[None], 14, axis=0)),
    ([0, 0, 0], np.eye(3), np.full((15, 3, 3), np.nan)),
])
def test_invalid_pose_does_not_mutate_state(geometry, position, root, local):
    hand = NativeManoStaticModel(geometry)
    before = hand.data.qpos.copy()
    with pytest.raises(ValueError):
        hand.set_pose(position, root, local)
    np.testing.assert_array_equal(hand.data.qpos, before)


def test_externally_advanced_state_fails_closed(geometry):
    hand = NativeManoStaticModel(geometry)
    hand.data.time = .01
    with pytest.raises(RuntimeError, match="advanced"):
        hand.readback()
    with pytest.raises(RuntimeError, match="advanced"):
        hand.set_pose(np.zeros(3), np.eye(3), np.repeat(np.eye(3)[None], 15, axis=0))


def test_float32_rotations_are_explicitly_normalized_and_measured(geometry):
    hand = NativeManoStaticModel(geometry)
    root = rotation([2, -1, 3], .73).astype(np.float32)
    local = np.array([rotation([1 + j % 3, (-1)**j, .4], (j - 7) * .09) for j in range(15)], dtype=np.float32)
    hand.set_pose([.1, .2, .3], root, local)
    diagnostics = hand.pose_diagnostics
    assert 0 < diagnostics["input_rotation_orthogonality_max_abs"] < 1e-6
    assert 0 < diagnostics["rotation_projection_max_abs"] < 1e-6
    assert diagnostics["rotation_quaternion_roundtrip_max_abs"] < 1e-14
    actual = hand.readback().wrist_rotation_world
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-14, rtol=0)
    np.testing.assert_allclose(actual, root, atol=1e-6, rtol=0)
    before = hand.data.qpos.copy()
    with pytest.raises(ValueError, match="orthonormal"):
        hand.set_pose([.1, .2, .3], root * (1 + 1e-6), local)
    np.testing.assert_array_equal(hand.data.qpos, before)
