"""Static and manufactured tests only: real mj_step is never called."""

from dataclasses import replace
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.proposals import physics_filter as pf


@pytest.fixture(autouse=True)
def forbid_dynamics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real physics integration is forbidden in these tests")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(mujoco, "mj_step1", forbidden)
    monkeypatch.setattr(mujoco, "mj_step2", forbidden)


@pytest.fixture
def scene():
    # Model compilation and forward kinematics only; no dynamic integration.
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.002"/>
      <worldbody><geom name="table" type="plane" size="2 2 .1" pos="0 0 -2"/>
      <body name="carrier" mocap="true" pos="0 0 1"/>
      <body name="forearm" pos="0 0 1"><freejoint/>
        <geom name="arm" type="sphere" size=".02"/>
        <body name="palm"><geom name="palm_geom" type="sphere" size=".02"/>
          <body name="ffdistal" pos=".1 0 0"><joint name="finger"/><geom name="ff" type="sphere" size=".01"/></body>
          <body name="thdistal" pos="-.1 0 0"><geom name="th" type="sphere" size=".01"/></body>
          <body name="mfdistal" pos="0 .1 0"><geom name="mf" type="sphere" size=".01"/></body>
          <body name="rfdistal" pos="0 -.1 0"><geom name="rf" type="sphere" size=".01"/></body>
          <body name="lfdistal" pos="0 -.2 0"><geom name="lf" type="sphere" size=".01"/></body>
        </body></body>
      <body name="object" pos="0 0 .3"><freejoint/><geom name="obj" type="sphere" size=".03" mass=".2"/></body>
      </worldbody><equality><weld body1="forearm" body2="carrier"/></equality>
      <actuator><position joint="finger" kp="1"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    data.time = 7.25
    data.ctrl[:] = .137
    data.mocap_pos[:] = [0.1, 0.2, 1.3]
    data.mocap_quat[:] = [1, 0, 0, 0]
    data.qacc_warmstart[:] = .7
    data.qfrc_applied[0] = .03  # hand force is preserved, not the free object.
    mujoco.mj_forward(model, data)
    return model, data, pf.PhysicsFilterGeometry.from_shadow_model(model)


@pytest.fixture
def manufactured(monkeypatch):
    calls = []
    def step(model, data):
        calls.append({"gravity": model.opt.gravity.copy(), "time": float(data.time),
                      "ctrl": data.ctrl.copy(), "mocap_pos": data.mocap_pos.copy(),
                      "mocap_quat": data.mocap_quat.copy(), "state": pf._integration_state(model, data),
                      "table_masks": (model.geom_contype[0], model.geom_conaffinity[0]),
                      "data": data, "model": model})
        data.time += model.opt.timestep
    def sample(model, data, geometry, force_floor):
        return pf._Sample(np.zeros(3), np.eye(3), frozenset({"thumb", "index"}), 0., 0.)
    monkeypatch.setattr(mujoco, "mj_step", step)
    monkeypatch.setattr(pf, "_sample", sample)
    return calls


def short_spec(**kwargs):
    return replace(pf.PhysicsFilterSpec(), filter_id="test_physics_filter_v1",
                   hold_s=.020, contact_window_s=.010, **kwargs)


def test_registry_exact_defaults_and_serializable():
    assert pf.load_physics_filter_spec() == pf.PhysicsFilterSpec()
    json.dumps(pf.PhysicsFilterResult("accepted", {}, {}).as_record(), allow_nan=False)


@pytest.mark.parametrize("change", [dict(gravity_m_s2=float("nan")), dict(hold_s=0),
    dict(minimum_distal_fingers=True), dict(minimum_distal_fingers=6),
    dict(maximum_total_steps=0), dict(maximum_wall_s=float("inf")),
    dict(minimum_supported_fraction=1.1), dict(maximum_unsupported_s=-1),
    dict(disable_declared_table_collisions=1), dict(minimum_table_clearance_m=-.1)])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        replace(pf.PhysicsFilterSpec(), **change)


def test_all_axes_restore_exact_state_without_source_mutation(scene, manufactured):
    model, data, geometry = scene
    before = pf._integration_state(model, data)
    gravity, qacc = model.opt.gravity.copy(), data.qacc.copy()
    result = pf.filter_closed_grasp(model, data, geometry, configuration=short_spec(),
                                    provenance={"command_sha256": "fixed-command"})
    assert result.filter_status == "accepted"
    assert len(manufactured) == 60
    assert len({id(manufactured[i]["data"]) for i in range(0, 60, 10)}) == 6
    assert len({id(manufactured[i]["model"]) for i in range(0, 60, 10)}) == 6
    for i, (_, axis) in enumerate(pf.AXES):
        call = manufactured[i * 10]
        np.testing.assert_array_equal(call["gravity"], np.asarray(axis) * 9.81)
        np.testing.assert_array_equal(call["state"], before)
        np.testing.assert_array_equal(call["ctrl"], data.ctrl)
        np.testing.assert_array_equal(call["mocap_pos"], data.mocap_pos)
        np.testing.assert_array_equal(call["mocap_quat"], data.mocap_quat)
        assert call["time"] == 7.25
        assert call["table_masks"] == (0, 0)
    np.testing.assert_array_equal(before, pf._integration_state(model, data))
    np.testing.assert_array_equal(gravity, model.opt.gravity)
    np.testing.assert_array_equal(qacc, data.qacc)
    assert model.geom_contype[0] == 1
    assert result.provenance["source_state_and_gravity_unchanged"]
    assert result.provenance["task_outcome_label"] is None
    assert not result.provenance["predictor_feature_eligible"]
    assert all(a["initial_state_sha256"] == result.provenance["initial_state_sha256"] for a in result.axes)
    json.dumps(result.as_record(), allow_nan=False)


@pytest.mark.parametrize("fault,match", [("warning", "warning"), ("nan_qacc", "qacc"),
    ("stall", "time"), ("reset", "time"), ("exception", "engine failed")])
def test_numerical_fault_is_error_and_remaining_axes_not_run(scene, manufactured, monkeypatch, fault, match):
    model, data, geometry = scene
    def step(m, d):
        if fault == "warning":
            d.warning.number[0] += 1
            d.qpos[:] = np.nan_to_num(d.qpos)  # finite auto-reset cannot hide warning.
            d.time += m.opt.timestep
        elif fault == "nan_qacc":
            d.qacc[0] = np.nan
            d.time += m.opt.timestep
        elif fault == "reset":
            d.time = 0
        elif fault == "exception":
            raise ValueError("engine failed")
    monkeypatch.setattr(mujoco, "mj_step", step)
    result = pf.filter_closed_grasp(model, data, geometry, configuration=short_spec())
    assert result.filter_status == "error"
    assert result.axes[0]["steps"] == 1
    assert match in result.axes[0]["error"]
    assert [a["status"] for a in result.axes] == ["error"] + ["not_run"] * 5
    assert result.provenance["attempted_steps"] == 1
    assert result.provenance["source_state_and_gravity_unchanged"]


def test_hard_quotas_checked_before_any_step(scene, manufactured):
    for spec in (short_spec(maximum_steps_per_axis=9), short_spec(maximum_total_steps=59),
                 replace(short_spec(), hold_s=.021)):
        with pytest.raises(ValueError):
            pf.filter_closed_grasp(*scene, configuration=spec)
    assert manufactured == []


def test_wall_guard_stops_before_step(scene, manufactured, monkeypatch):
    ticks = iter([0., 2., 3.])
    monkeypatch.setattr(pf.time, "monotonic", lambda: next(ticks))
    result = pf.filter_closed_grasp(*scene, configuration=short_spec(maximum_wall_s=1.))
    assert result.filter_status == "error"
    assert result.provenance["attempted_steps"] == 0
    assert manufactured == []


def test_measured_rejection_thresholds_and_final_endpoint(scene, manufactured, monkeypatch):
    initial_time = scene[1].time
    def sample(model, data, geometry, force_floor):
        end = data.time - initial_time > .019
        return pf._Sample(np.array([.021 if end else 0., 0, 0]), np.eye(3),
                          frozenset({"palm", "thumb"}), .006, .02)
    monkeypatch.setattr(pf, "_sample", sample)
    result = pf.filter_closed_grasp(*scene, configuration=short_spec(maximum_unsupported_s=.01))
    assert result.filter_status == "rejected"
    codes = {c["code"] for c in result.rejection_codes}
    assert "GRAVITY_FAIL_pos_x" in codes
    assert {"INSUFFICIENT_CONTACT", "CONTACT_DROPOUT", "PENETRATION_EXCEEDED", "EXTERNAL_OBJECT_SUPPORT"} <= codes
    assert all(a["steps"] == 10 for a in result.axes)
    gravity_code = next(c for c in result.rejection_codes if c["code"] == "GRAVITY_FAIL_pos_x")
    assert gravity_code["measured"] == .021 and gravity_code["threshold"] == .020
    assert result.axes[0]["region_contact_fraction"]["palm"] == 1.


def test_relative_palm_frame_is_invariant_to_common_rigid_motion(scene):
    model, data, geometry = scene
    first = pf._sample(model, data, geometry, .01)
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    shift = np.array([2., -1., .4])
    for body in (geometry.object_body_id, geometry.reference_body_id):
        data.xpos[body] = rotation @ data.xpos[body] + shift
        data.xmat[body] = (rotation @ data.xmat[body].reshape(3, 3)).ravel()
    second = pf._sample(model, data, geometry, .01)
    np.testing.assert_allclose(first.relative_position, second.relative_position, atol=1e-15, rtol=0)
    np.testing.assert_allclose(first.relative_rotation, second.relative_rotation, atol=1e-15, rtol=0)


def test_contact_support_uses_active_force_and_distinct_regions(monkeypatch):
    geometry = pf.PhysicsFilterGeometry(1, 2, (0,), (1, 2, 3, 4),
                                        {"thumb": (1, 2), "index": (3,)}, (4,))
    contacts = [SimpleNamespace(geom1=0, geom2=other, dist=-.002, efc_address=active)
                for other, active in ((1, 0), (2, 1), (3, -1), (4, 2), (5, 3))]
    data = SimpleNamespace(ncon=len(contacts), contact=contacts, xpos=np.zeros((3, 3)),
                           xmat=np.tile(np.eye(3).ravel(), (3, 1)))
    def force(model, data, index, result):
        result[0] = 0.01 if index == 3 else .1  # exact floor does not count.
    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    sample = pf._sample(None, data, geometry, .01)
    assert sample.supported_regions == frozenset({"thumb"})
    assert sample.penetration_m == .002
    assert sample.external_support_n == .1


def test_persistence_is_per_window_and_dropout_crosses_window_boundary():
    good, bad = frozenset({"thumb", "index"}), frozenset({"palm", "thumb"})
    windows, longest = pf._window_records([good] * 9 + [bad] * 3 + [good] * 8, .002, 10, 2)
    assert [w["supported_fraction"] for w in windows] == [.9, .8]
    assert longest == .006
    assert windows[0]["minimum_distal_finger_count"] == 1
    assert windows[1]["maximum_distal_finger_count"] == 2
    assert windows[0]["region_contact_fraction"]["palm"] == .1
    assert windows[1]["region_contact_fraction"]["thumb"] == 1.


def test_external_object_forces_and_gravity_compensation_refused(scene, manufactured):
    model, data, geometry = scene
    data.xfrc_applied[geometry.object_body_id, 0] = .1
    with pytest.raises(ValueError, match="external applied forces"):
        pf.filter_closed_grasp(model, data, geometry, configuration=short_spec())
    data.xfrc_applied[:] = 0
    model.body_gravcomp[geometry.object_body_id] = 1
    with pytest.raises(ValueError, match="gravity compensation"):
        pf.filter_closed_grasp(model, data, geometry, configuration=short_spec())
    assert manufactured == []


def test_optional_table_plane_gate_is_static_and_copy_only(scene, manufactured):
    model, data, geometry = scene
    geometry = replace(geometry, table_top_z_m=2.)
    result = pf.filter_closed_grasp(model, data, geometry,
                                    configuration=short_spec(minimum_table_clearance_m=.01))
    assert result.filter_status == "rejected"
    assert any(c["code"] == "TABLE_COLLISION_PREDICTED" for c in result.rejection_codes)
    assert model.geom_contype[0] == 1


def test_support_geometries_cannot_be_double_counted(scene, manufactured):
    model, data, geometry = scene
    fingers = dict(geometry.distal_finger_geom_ids)
    fingers["index"] = fingers["thumb"]
    with pytest.raises(ValueError, match="multiple support regions"):
        pf.filter_closed_grasp(model, data, replace(geometry, distal_finger_geom_ids=fingers),
                               configuration=short_spec())
    assert manufactured == []


def test_exact_physical_thresholds_are_inclusive(scene, manufactured, monkeypatch):
    def sample(model, data, geometry, force_floor):
        # Initial penetration at the exact cap passes; constant relative pose
        # has zero displacement. External support at the force floor passes.
        return pf._Sample(np.zeros(3), np.eye(3), frozenset({"thumb", "index"}), .005, .01)
    monkeypatch.setattr(pf, "_sample", sample)
    result = pf.filter_closed_grasp(*scene, configuration=short_spec())
    assert result.filter_status == "accepted"


def test_partial_object_geom_declaration_is_refused(scene, manufactured):
    model, data, geometry = scene
    with pytest.raises(ValueError, match="must be declared"):
        pf.filter_closed_grasp(model, data, replace(geometry, object_geom_ids=()), configuration=short_spec())
    assert manufactured == []


def test_missing_optional_table_plane_is_refused_before_steps(scene, manufactured):
    with pytest.raises(ValueError, match="table plane gate"):
        pf.filter_closed_grasp(*scene, configuration=short_spec(minimum_table_clearance_m=.01))
    assert manufactured == []


def test_post_step_forward_warning_is_not_hidden(scene, manufactured, monkeypatch):
    original_forward = mujoco.mj_forward
    seen = 0
    def forward(model, data):
        nonlocal seen
        original_forward(model, data)
        seen += 1
        if seen == 2:  # initial forward succeeds, first endpoint forward warns.
            data.warning.number[0] += 1
    monkeypatch.setattr(mujoco, "mj_forward", forward)
    result = pf.filter_closed_grasp(*scene, configuration=short_spec())
    assert result.filter_status == "error"
    assert result.provenance["attempted_steps"] == 1
    assert "warning" in result.axes[0]["error"]
