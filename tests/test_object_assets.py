from __future__ import annotations

import os
from pathlib import Path
import shutil

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation.object_assets import load_object_geometry
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
from grasp_failure_prediction.evaluation.runner import ActuatedShadowRunner, AdroitShadowRunner
from grasp_failure_prediction.evaluation.schema import canonical_sha256


def write_box_mesh(path, center, halfsize):
    vertices = [np.asarray(center) + np.asarray(halfsize) * sign for sign in (
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    )]
    faces = ((1, 3, 2), (1, 4, 3), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8))
    path.write_text("\n".join("v " + " ".join(str(float(x)) for x in vertex) for vertex in vertices)
                    + "\n" + "\n".join("f " + " ".join(str(x) for x in face) for face in faces) + "\n")


@pytest.fixture
def asset(tmp_path):
    root = tmp_path / "asset"
    root.mkdir()
    write_box_mesh(root / "left.obj", [-0.03, 0, 0], [0.02, 0.02, 0.025])
    write_box_mesh(root / "right.obj", [0.03, 0, 0], [0.02, 0.02, 0.025])
    write_box_mesh(root / "visual.obj", [0, 0, 0], [0.05, 0.02, 0.025])
    path = root / "object.xml"
    path.write_text('''<mujoco model="two_piece_object">
      <compiler meshdir="." angle="radian"/>
      <asset>
        <mesh name="visible" file="visual.obj"/>
        <mesh name="left" file="left.obj"/>
        <mesh name="right" file="right.obj"/>
      </asset>
      <worldbody><body name="source_object" pos="0.2 0.1 0.05" quat="0.965925826 0 0 0.258819045">
        <inertial pos="0.003 -0.001 0.002" mass="0.2"
          fullinertia="0.00025 0.00035 0.00030 0.00001 -0.00002 0.000005"/>
        <freejoint/>
        <geom type="mesh" mesh="visible" contype="0" conaffinity="0" group="2"/>
        <geom type="mesh" mesh="left" pos="0.002 0.001 0.001" quat="0.965925826 0 0 0.258819045"
          friction="1 0.005 0.0001" group="3"/>
        <geom type="mesh" mesh="right" friction="1 0.005 0.0001" group="3"/>
      </body></worldbody>
    </mujoco>''')
    return path


def make_runner(kind, geometry=None):
    protocol = "fixed_grasp_lift_v2" if kind is ActuatedShadowRunner else "fixed_grasp_lift_v1"
    return kind(load_protocol_registry().resolve(protocol), default_dex_urdf_root(), object_geometry=geometry)


def test_geometry_identity_covers_mjcf_and_meshes_and_is_location_independent(asset, tmp_path):
    first = load_object_geometry(asset)
    assert set(first.files_sha256) == {"object.xml", "visual.obj", "left.obj", "right.obj"}
    assert first.content_hash == canonical_sha256(first.files_sha256)
    assert first.native_mass_kg == 0.2
    assert first.source_model_name == "two_piece_object"
    assert first.collision_geom_indices == (1, 2)
    shutil.copytree(asset.parent, tmp_path / "copy")
    second = load_object_geometry(tmp_path / "copy/object.xml", expected_content_hash=first.content_hash)
    assert second.content_hash == first.content_hash
    mesh = asset.parent / "right.obj"
    mesh.write_text(mesh.read_text() + "# different source bytes\n")
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_object_geometry(asset, expected_content_hash=first.content_hash)


@pytest.mark.parametrize("kind", [AdroitShadowRunner, ActuatedShadowRunner], ids=["kinematic", "actuated"])
def test_native_compound_geometry_and_inertial_frames_survive_import(asset, kind):
    geometry = load_object_geometry(asset)
    native = mujoco.MjModel.from_xml_path(str(asset))
    runner = make_runner(kind, geometry)
    body = runner._object_body_id
    assert len(runner.object_geom_ids) == 3
    assert len(runner.object_collision_geom_ids) == 2
    assert runner.object_render_geom_ids == runner.object_geom_ids
    assert runner._object_geom_id in runner.object_collision_geom_ids
    assert mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision_001") in runner.object_collision_geom_ids
    for field in ("body_mass", "body_inertia", "body_ipos"):
        np.testing.assert_allclose(getattr(runner.model, field)[body], getattr(native, field)[1],
                                   atol=1e-12, rtol=2e-6)
    # The existing actuation conversion serializes a compiled model through
    # MuJoCo XML, which rounds quaternions to about six significant figures.
    np.testing.assert_allclose(runner.model.body_iquat[body], native.body_iquat[1], atol=1e-6)
    ids = sorted(runner.object_geom_ids)
    for field in ("geom_pos", "geom_quat", "geom_friction", "geom_contype", "geom_conaffinity", "geom_group"):
        tolerance = 1e-6 if field == "geom_quat" else 1e-7 if field == "geom_pos" else 1e-12
        np.testing.assert_allclose(getattr(runner.model, field)[ids], getattr(native, field), atol=tolerance)
    # Off-center mesh frames remain off-center; no compiled mesh transform is applied twice.
    assert runner.model.geom_pos[ids[1], 0] < -0.01
    assert runner.model.geom_pos[ids[2], 0] > 0.01
    lower, upper = np.asarray(geometry.local_collision_bounds_m)
    assert lower[0] < -0.04 and upper[0] == pytest.approx(0.05, abs=1e-8)


def test_second_collision_part_contributes_object_table_contact(asset):
    runner = make_runner(AdroitShadowRunner, load_object_geometry(asset))
    runner.reset(seed=0, object_mass_kg=0.2, object_position_m=np.array([0, 0, 0.024]),
                 object_orientation_wxyz=np.array([1.0, 0, 0, 0]))
    # Isolate a physical table contact from the second piece, the old single-geom
    # implementation would miss it. Park the hand away from both pieces.
    runner.data.qpos[runner._root_qpos_address : runner._root_qpos_address + 3] = [0, 0, 1]
    first = runner._object_geom_id
    second = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision_001")
    runner.model.geom_contype[first] = runner.model.geom_conaffinity[first] = 0
    mujoco.mj_forward(runner.model, runner.data)
    pairs = [{int(contact.geom1), int(contact.geom2)} for contact in runner.data.contact]
    assert {second, runner._table_geom_id} in pairs
    assert all(first not in pair for pair in pairs)
    hand_object, _, object_table, _, _ = runner._contact_metrics()
    assert object_table and not hand_object


@pytest.mark.parametrize(("before", "after"), [
    ('mass="0.2"', 'mass="nan"'),
    ('fullinertia="0.00025 0.00035 0.00030 0.00001 -0.00002 0.000005"',
     'fullinertia="-0.001 0.00035 0.00030 0 0 0"'),
    ('pos="0.002 0.001 0.001"', 'pos="nan 0 0"'),
    ('file="left.obj"', 'file="../left.obj"'),
    ('angle="radian"', 'angle="degree"'),
    ('<freejoint/>', '<freejoint/><freejoint/>'),
    ('</worldbody>', '<body name="extra"/></worldbody>'),
    ('file="right.obj"', 'file="right.obj" scale="1 inf 1"'),
    ('<asset>', '<asset><texture name="unsupported"/>'),
])
def test_invalid_physical_frames_and_asset_closures_are_rejected(asset, before, after):
    asset.write_text(asset.read_text().replace(before, after))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_object_geometry(asset)


def test_changed_asset_after_resolution_cannot_silently_enter_scene(asset):
    geometry = load_object_geometry(asset)
    mesh = asset.parent / "left.obj"
    mesh.write_text(mesh.read_text() + "# changed after preflight\n")
    with pytest.raises(ValueError, match="changed after resolution"):
        make_runner(AdroitShadowRunner, geometry)


@pytest.mark.parametrize("kind", [AdroitShadowRunner, ActuatedShadowRunner], ids=["kinematic", "actuated"])
def test_default_cube_keeps_legacy_mass_geometry_and_free_fall(kind):
    runner = make_runner(kind)
    geom = runner._object_geom_id
    assert runner.object_geometry is None
    assert runner.object_geom_ids == runner.object_collision_geom_ids == frozenset({geom})
    assert runner.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_array_equal(runner.model.geom_size[geom], [0.025, 0.025, 0.025])
    assert runner.model.body_mass[runner._object_body_id] == 0.18
    np.testing.assert_allclose(runner.model.body_inertia[runner._object_body_id], [0.000075] * 3)
    runner.reset(seed=0, object_mass_kg=0.18, object_position_m=np.array([0, 0, 0.15]),
                 object_orientation_wxyz=np.array([1.0, 0, 0, 0]))
    root = runner._root_qpos_address
    runner.data.qpos[root : root + 3] = [0, 0, 1]
    if runner.model.nmocap:
        runner.data.mocap_pos[0] = [0, 0, 1]
    mujoco.mj_forward(runner.model, runner.data)
    dt = runner.physics_timestep_s
    for count in range(1, 11):
        mujoco.mj_step(runner.model, runner.data)
        expected_height = 0.15 - 9.81 * dt**2 * count * (count + 1) / 2
        assert runner.data.qpos[runner._object_qpos_address + 2] == pytest.approx(expected_height, abs=1e-14)


@pytest.mark.skipif(not os.environ.get("GRASP_HUG_APPLE_ASSETS"), reason="optional reviewed HUG apple assets unavailable")
def test_reviewed_hug_apple_preserves_native_compound_geometry():
    root = Path(os.environ["GRASP_HUG_APPLE_ASSETS"])
    path = root / "object.xml" if root.is_dir() else root
    geometry = load_object_geometry(path)
    runner = make_runner(ActuatedShadowRunner, geometry)
    assert geometry.source_model_name == "apple"
    assert geometry.native_mass_kg == pytest.approx(0.2304)
    assert len(runner.object_geom_ids) == 3 and len(runner.object_collision_geom_ids) == 2
    lower, upper = np.asarray(geometry.local_collision_bounds_m)
    np.testing.assert_allclose(upper - lower, [0.08109483, 0.08461011, 0.09232164], atol=2e-8)
    np.testing.assert_allclose(runner.model.body_mass[runner._object_body_id], 0.2304)
