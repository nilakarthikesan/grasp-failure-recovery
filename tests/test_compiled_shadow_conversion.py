"""Static regressions: building a scene must not change the declared hand.

These tests compile models and evaluate kinematics only. Physics entry points
are blocked even if a runner is accidentally invoked by a future change.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation.pose_validation import load_shadow_model
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
from grasp_failure_prediction.evaluation.runner import (
    ActuatedShadowRunner, AdroitShadowRunner, build_tabletop_model,
)
from grasp_failure_prediction.evaluation.object_assets import load_object_geometry
from test_object_assets import asset  # shared static compound-mesh fixture


@pytest.fixture(autouse=True)
def prohibit_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("This conversion test must not advance physics")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)
    for cls in (AdroitShadowRunner, ActuatedShadowRunner):
        monkeypatch.setattr(cls, "execute", forbidden)
        monkeypatch.setattr(cls, "_advance", forbidden)


@pytest.fixture
def hand_root():
    root = Path(default_dex_urdf_root())
    if not (root / "shadow_hand/shadow_hand_right.urdf").is_file():
        pytest.skip("Pinned Shadow URDF assets are not installed")
    return root


def runner_for(kind, root, geometry=None):
    protocol = "fixed_grasp_lift_v2" if kind is ActuatedShadowRunner else "fixed_grasp_lift_v1"
    return kind(load_protocol_registry().resolve(protocol), root, object_geometry=geometry)


@pytest.mark.parametrize("kind", [AdroitShadowRunner, ActuatedShadowRunner])
def test_declared_hand_survives_scene_and_actuation_without_rounding(hand_root, kind):
    source = load_shadow_model(hand_root, floating_root=True)
    model = runner_for(kind, hand_root).model
    for bid in range(1, source.nbody):
        name = source.body(bid).name
        dest = model.body(name).id
        for field in ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia"):
            np.testing.assert_array_equal(getattr(model, field)[dest], getattr(source, field)[bid])
        a = np.flatnonzero(source.geom_bodyid == bid)
        b = np.flatnonzero(model.geom_bodyid == dest)
        for field in ("geom_pos", "geom_quat", "geom_size", "geom_type", "geom_friction",
                      "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp"):
            np.testing.assert_array_equal(getattr(model, field)[b], getattr(source, field)[a])
    for jid in range(source.njnt):
        dest = model.joint(source.joint(jid).name).id
        for field in ("jnt_pos", "jnt_axis", "jnt_range", "jnt_type", "jnt_limited"):
            np.testing.assert_array_equal(getattr(model, field)[dest], getattr(source, field)[jid])
    for mid in range(source.nmesh):
        dest = model.mesh(source.mesh(mid).name).id
        for field in ("mesh_pos", "mesh_quat", "mesh_scale"):
            np.testing.assert_array_equal(getattr(model, field)[dest], getattr(source, field)[mid])
        a, n = source.mesh_vertadr[mid], source.mesh_vertnum[mid]
        b = model.mesh_vertadr[dest]
        np.testing.assert_array_equal(model.mesh_vert[b:b+n], source.mesh_vert[a:a+n])


def test_servo_ranges_match_validator_and_controller_parameters_survive(hand_root):
    source = load_shadow_model(hand_root, floating_root=True)
    model = runner_for(ActuatedShadowRunner, hand_root).model
    assert model.nu == 24 and model.neq == 1 and model.nmocap == 1
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.noslip_iterations == 10
    assert model.eq_obj1id[0] == model.body("arm_target").id
    assert model.eq_obj2id[0] == model.body("forearm").id
    np.testing.assert_array_equal(model.eq_solref[0], [.01, 1.])
    for aid in range(model.nu):
        name = model.actuator(aid).name.removeprefix("servo_")
        sid = source.joint(name).id
        np.testing.assert_array_equal(model.actuator_ctrlrange[aid], source.jnt_range[sid])
        kp = 50. if name.startswith("WR") else 10.
        assert model.actuator_gainprm[aid, 0] == kp
        np.testing.assert_array_equal(model.actuator_biasprm[aid, :3], [0., -kp, -.3])
        effort = abs(source.jnt_actfrcrange[sid, 1]) if source.jnt_actfrclimited[sid] else 1.
        if not name.startswith("WR"):
            effort = min(effort, 1.)
        np.testing.assert_array_equal(model.actuator_forcerange[aid], [-effort, effort])
        assert model.actuator_ctrllimited[aid] and model.actuator_forcelimited[aid]


@pytest.mark.parametrize("kind", [AdroitShadowRunner, ActuatedShadowRunner])
def test_native_object_mesh_and_inertia_precision_survives_scene(asset, hand_root, kind):
    geometry = load_object_geometry(asset)
    native = mujoco.MjModel.from_xml_path(str(asset))
    runner = runner_for(kind, hand_root, geometry)
    model, bid = runner.model, runner._object_body_id
    for field in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat"):
        np.testing.assert_allclose(getattr(model, field)[bid], getattr(native, field)[1], rtol=0, atol=2e-15)
    gids = sorted(runner.object_geom_ids)
    for field in ("geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref", "geom_solimp"):
        np.testing.assert_allclose(getattr(model, field)[gids], getattr(native, field), rtol=0, atol=2e-15)


@pytest.mark.parametrize("kind", [AdroitShadowRunner, ActuatedShadowRunner])
def test_nonzero_wrist_and_rotated_palm_placement_do_not_advance_time(hand_root, kind):
    runner = runner_for(kind, hand_root)
    q = np.array([np.mean(runner.model.joint(n).range) for n in runner._hand_joint_names])
    q[0], q[1] = .11, -.17
    quaternion = np.array([.6, -.2, .3, .4]); quaternion /= np.linalg.norm(quaternion)
    position = np.array([.117, -.083, .261])
    root_pos, root_quat = runner._root_pose_for_palm(position, quaternion, q)
    a = runner._root_qpos_address
    runner.data.qpos[a:a+7] = np.r_[root_pos, root_quat]
    before = runner.data.qpos.copy()
    mujoco.mj_kinematics(runner.model, runner.data)
    expected = np.empty(9); mujoco.mju_quat2Mat(expected, quaternion)
    np.testing.assert_allclose(runner.data.xpos[runner._palm_body_id], position, rtol=0, atol=1e-14)
    np.testing.assert_allclose(runner.data.xmat[runner._palm_body_id], expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(runner.data.qpos, before)
    np.testing.assert_array_equal(runner.data.qvel, 0.)
    assert runner.data.time == 0


def test_colliders_use_each_declared_mesh_file_even_when_basenames_collide(hand_root):
    """An independent MJCF import avoids URDF's basename-only mesh aliasing."""
    from xml.etree import ElementTree as ET
    source = Path(hand_root) / "shadow_hand/shadow_hand_right.urdf"
    xml = ET.parse(source).getroot()
    actual = load_shadow_model(hand_root, floating_root=True)
    resolved = {}
    for link_name in ("ffdistal", "mfdistal", "rfdistal", "lfdistal"):
        collision = xml.find(f"link[@name='{link_name}']/collision")
        mesh = collision.find("geometry/mesh")
        path = (source.parent / mesh.attrib["filename"]).resolve()
        ref_xml = ET.Element("mujoco")
        assets = ET.SubElement(ref_xml, "asset")
        ET.SubElement(assets, "mesh", name="literal_file", file=str(path),
                      scale=mesh.get("scale", "1 1 1"))
        world = ET.SubElement(ref_xml, "worldbody")
        ET.SubElement(world, "geom", type="mesh", mesh="literal_file")
        reference = mujoco.MjModel.from_xml_string(ET.tostring(ref_xml, encoding="unicode"))
        gids = np.flatnonzero(actual.geom_bodyid == actual.body(link_name).id)
        assert len(gids) == 1
        mid = actual.geom_dataid[gids[0]]
        a, n = actual.mesh_vertadr[mid], actual.mesh_vertnum[mid]
        f, nf = actual.mesh_faceadr[mid], actual.mesh_facenum[mid]
        np.testing.assert_array_equal(actual.mesh_vert[a:a+n], reference.mesh_vert)
        np.testing.assert_array_equal(actual.mesh_face[f:f+nf], reference.mesh_face)
        np.testing.assert_array_equal(actual.mesh_scale[mid], reference.mesh_scale[0])
        resolved[link_name] = int(mid)
    assert resolved["ffdistal"] != resolved["mfdistal"]
    assert resolved["mfdistal"] == resolved["rfdistal"] == resolved["lfdistal"]


def test_legal_joint_boundary_commands_are_not_clipped_by_compilation(hand_root):
    from grasp_failure_prediction.evaluation.runner import force_close_targets
    source = load_shadow_model(hand_root, floating_root=True)
    runner = runner_for(ActuatedShadowRunner, hand_root)
    names = runner._hand_joint_names
    expected = np.array([source.joint(n).range for n in names])
    compiled = np.array([runner.model.joint(n).range for n in names])
    for side in (0, 1):
        command = force_close_targets(names, expected[:, side], compiled, 0.)
        np.testing.assert_array_equal(command, expected[:, side])
        runner.data.qpos[runner._hand_qpos_addresses] = command
        mujoco.mj_kinematics(runner.model, runner.data)
        np.testing.assert_array_equal(runner.data.qpos[runner._hand_qpos_addresses], expected[:, side])
        assert runner.data.time == 0


def test_actuation_is_independent_of_global_xml_parser_state(hand_root, monkeypatch):
    from grasp_failure_prediction.evaluation.runner import _build_tabletop_spec, add_shadow_actuation
    def legacy_export_forbidden(*args, **kwargs):
        raise AssertionError("Live scene construction must not serialize compiled XML")
    monkeypatch.setattr(mujoco, "mj_saveLastXML", legacy_export_forbidden)
    spec = _build_tabletop_spec(hand_root)
    original = spec.compile()
    mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="unrelated"><geom size=".1"/></body></worldbody></mujoco>')
    actuated = add_shadow_actuation(spec)
    assert mujoco.mj_name2id(actuated, mujoco.mjtObj.mjOBJ_BODY, "unrelated") == -1
    assert actuated.nu == 24 and spec.compile().nu == 0
    np.testing.assert_array_equal(actuated.jnt_range, original.jnt_range)
    np.testing.assert_array_equal(actuated.geom_pos, original.geom_pos)
    with pytest.raises(TypeError, match="live tabletop MjSpec"):
        add_shadow_actuation(original)


def test_binary_scene_roundtrip_preserves_runtime_model_without_xml(hand_root, tmp_path, monkeypatch):
    runner = runner_for(ActuatedShadowRunner, hand_root)
    # Runtime mass changes must be exported from MjModel, not a stale MjSpec.
    bid = runner._object_body_id
    runner.model.body_mass[bid] *= 1.5
    runner.model.body_inertia[bid] *= 1.5
    def forbidden(*args, **kwargs):
        raise AssertionError("Binary export must not consult global XML state")
    monkeypatch.setattr(mujoco, "mj_saveLastXML", forbidden)
    path = tmp_path / "scene.mjb"
    mujoco.mj_saveModel(runner.model, str(path))
    restored = mujoco.MjModel.from_binary_path(str(path))
    for field in ("body_pos", "body_quat", "body_mass", "body_inertia", "jnt_range",
                  "geom_pos", "geom_quat", "geom_size", "mesh_vert", "mesh_face",
                  "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm", "eq_data"):
        np.testing.assert_array_equal(getattr(restored, field), getattr(runner.model, field))


def test_retained_actuated_spec_supports_precise_static_scene_edits(hand_root):
    runner = runner_for(ActuatedShadowRunner, hand_root)
    query = runner._scene_spec.copy()
    query.worldbody.add_geom(name="static_probe", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                             size=[.001, 0., 0.], pos=[1., 1., 1.], contype=0, conaffinity=0)
    model = query.compile()
    assert model.nu == 24 and model.neq == 1
    np.testing.assert_array_equal(model.jnt_range, runner.model.jnt_range)
    np.testing.assert_array_equal(model.body_mass, runner.model.body_mass)
    np.testing.assert_array_equal(model.body_inertia, runner.model.body_inertia)
    assert runner._scene_spec.compile().ngeom == runner.model.ngeom


def test_renderer_loader_supports_exact_mjb_and_historical_xml(tmp_path):
    from grasp_failure_prediction.evaluation.render_rollout import _load_scene_model
    xml = tmp_path / "historical.xml"
    xml.write_text('<mujoco><worldbody><geom type="sphere" size=".012345678901234"/></worldbody></mujoco>')
    model = _load_scene_model(xml)
    binary = tmp_path / "current.mjb"
    mujoco.mj_saveModel(model, str(binary))
    np.testing.assert_array_equal(_load_scene_model(binary).geom_size, model.geom_size)
