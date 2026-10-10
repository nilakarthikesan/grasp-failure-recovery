from __future__ import annotations

from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
from grasp_failure_prediction.evaluation.runner import ActuatedShadowRunner, AdroitShadowRunner


@pytest.fixture(params=[AdroitShadowRunner, ActuatedShadowRunner], ids=["kinematic", "actuated"])
def runner(request):
    protocol_id = "fixed_grasp_lift_v2" if request.param is ActuatedShadowRunner else "fixed_grasp_lift_v1"
    return request.param(load_protocol_registry().resolve(protocol_id), default_dex_urdf_root())


def reset(runner, mass, *, position=None, quaternion=None):
    runner.reset(
        seed=9000,
        object_mass_kg=mass,
        object_position_m=np.array([0.0, 0.0, 0.03]) if position is None else position,
        object_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]) if quaternion is None else quaternion,
    )


def fresh_mass_model(runner, path, mass):
    """Compile the same scene with mass/inertia specified before compilation."""
    mujoco.mj_saveLastXML(str(path), runner.model)
    tree = ET.parse(path)
    geom = tree.getroot().find(".//body[@name='object']/geom[@name='object_geom']")
    assert geom is not None
    # The cube's inertia is inferred independently from its geometry and mass.
    geom.set("mass", str(mass))
    tree.write(path, encoding="utf-8")
    return mujoco.MjModel.from_xml_path(str(path))


def test_mass_reset_matches_fresh_compilation_and_preserves_requested_state(runner, tmp_path):
    mass = 0.36
    expected = fresh_mass_model(runner, tmp_path / "mass_matched.xml", mass)
    position = np.array([0.07, -0.04, 0.12])
    quaternion = np.array([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)])
    reset(runner, mass, position=position, quaternion=quaternion)

    body = runner._object_body_id
    for field in ("body_mass", "body_inertia", "body_invweight0", "body_subtreemass"):
        np.testing.assert_allclose(getattr(runner.model, field)[body], getattr(expected, field)[body])
    joint = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    dof = int(runner.model.jnt_dofadr[joint])
    for field in ("dof_M0", "dof_invweight0"):
        np.testing.assert_allclose(
            getattr(runner.model, field)[dof : dof + 6], getattr(expected, field)[dof : dof + 6]
        )
    address = runner._object_qpos_address
    expected_qpos = runner.model.qpos0.copy()
    expected_qpos[address : address + 3] = position
    expected_qpos[address + 3 : address + 7] = quaternion
    np.testing.assert_allclose(runner.data.qpos, expected_qpos, atol=1e-15, rtol=0.0)
    np.testing.assert_array_equal(runner.data.qvel, np.zeros(runner.model.nv))
    assert runner.data.time == 0.0


def test_default_mass_reset_preserves_original_dynamics(runner):
    """The existing 180 g pilot remains bit-identical to its original reset."""
    expected_data = mujoco.MjData(runner.model)
    mujoco.mj_resetData(runner.model, expected_data)
    address = runner._object_qpos_address
    expected_data.qpos[address : address + 3] = [0.0, 0.0, 0.03]
    expected_data.qpos[address + 3 : address + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(runner.model, expected_data)
    reset(runner, 0.18)
    for _ in range(4):
        for field in ("qpos", "qvel", "qacc", "qM", "qfrc_bias"):
            np.testing.assert_array_equal(getattr(runner.data, field), getattr(expected_data, field))
        assert runner.data.time == expected_data.time
        mujoco.mj_step(runner.model, runner.data)
        mujoco.mj_step(runner.model, expected_data)


def test_repeated_mass_resets_do_not_accumulate_inertia_scaling(runner):
    body = runner._object_body_id
    initial_mass = float(runner.model.body_mass[body])
    initial_inertia = runner.model.body_inertia[body].copy()
    initial_inverse_weight = runner.model.body_invweight0[body].copy()
    for mass in (0.36, 0.09, 0.27, 0.18, 0.18):
        reset(runner, mass)
        assert runner.model.body_mass[body] == pytest.approx(mass)
        assert runner.model.body_subtreemass[body] == pytest.approx(mass)
        np.testing.assert_allclose(runner.model.body_inertia[body], initial_inertia * mass / initial_mass)
        np.testing.assert_allclose(runner.model.body_invweight0[body], initial_inverse_weight * initial_mass / mass)
