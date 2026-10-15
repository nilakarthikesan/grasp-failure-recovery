"""Manufactured contact-law checks; no assets, model, contact search or steps."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("mano_point_contact_law_v1", Path(__file__).with_name("contact_law.py"))
law = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = law
SPEC.loader.exec_module(law)


def args():
    jh = np.zeros((3, 51)); jh[:, :3] = np.eye(3)
    return dict(signed_gap_m=-.001, object_outward_normal_world=np.array([0., 0., 1.]),
                common_point_world_m=np.array([.2, -.3, .4]),
                object_wrench_origin_world_m=np.array([-.1, .2, .3]),
                hand_point_jacobian=jh, object_point_jacobian=np.eye(3),
                hand_generalized_velocity=np.zeros(51), object_generalized_velocity=np.zeros(3),
                parameters=law.ContactParameters(1000., 2., .5, .001))


@pytest.mark.parametrize("gap", [0., .001, 1.])
@pytest.mark.parametrize("vz", [-1., 0., 1.])
def test_no_force_without_penetration(gap, vz):
    a = args(); a['signed_gap_m'] = gap; a['hand_generalized_velocity'][2] = vz
    r = law.evaluate_contact(**a)
    assert not r['hand_force_world_n'].any()
    assert r['elastic_energy_j'] == 0
    assert r['power_identity_residual_w'] == 0


@pytest.mark.parametrize("vz", [-2., -.1, 0., .1, .5, 1., 2.])
@pytest.mark.parametrize("vx", [-1., 0., 1.])
def test_unilateral_dissipative_and_virtual_work(vz, vx):
    a = args(); a['hand_generalized_velocity'][:3] = [vx, .2, vz]
    r = law.evaluate_contact(**a)
    assert r['normal_force_n'] >= 0
    assert r['normal_dissipation_power_w'] <= 1e-14
    assert r['friction_power_w'] <= 1e-14
    assert abs(r['power_identity_residual_w']) < 1e-14
    assert r['mechanical_power_w']+r['elastic_energy_rate_w'] <= 1e-14
    ft = r['hand_force_world_n'][:2]
    assert np.linalg.norm(ft) <= .5*r['normal_force_n']+1e-14
    np.testing.assert_array_equal(r['hand_force_world_n'], -r['object_force_world_n'])
    expected = r['hand_force_world_n'] @ r['relative_velocity_world_m_per_s']
    assert abs(r['mechanical_power_w']-expected) < 1e-14


def test_elastic_force_is_negative_potential_gradient():
    a = args(); a['parameters'] = law.ContactParameters(1000., 0., 0., .001)
    h = 1e-8
    a['signed_gap_m'] = -.001+h; ep = law.evaluate_contact(**a)['elastic_energy_j']
    a['signed_gap_m'] = -.001-h; em = law.evaluate_contact(**a)['elastic_energy_j']
    a['signed_gap_m'] = -.001; r = law.evaluate_contact(**a)
    assert abs(-(ep-em)/(2*h)-r['normal_force_n']) < 1e-10


def test_nontrivial_jacobians_object_frame_and_common_point_wrench():
    a = args(); rng = np.random.default_rng(139)
    a['hand_point_jacobian'] = rng.normal(size=(3,51))
    # Actual six-dof object point Jacobian in an explicit world tangent.
    p = a['common_point_world_m']-a['object_wrench_origin_world_m']
    skew = np.array([[0., -p[2], p[1]], [p[2], 0., -p[0]], [-p[1], p[0], 0.]])
    a['object_point_jacobian'] = np.column_stack((np.eye(3), -skew))
    a['hand_generalized_velocity'] = rng.normal(size=51)
    a['object_generalized_velocity'] = rng.normal(size=6)
    a['object_outward_normal_world'] = np.array([.6, 0., .8])
    r = law.evaluate_contact(**a)
    assert r['normal_force_n'] > 0
    np.testing.assert_allclose(r['object_generalized_load'], r['object_wrench_force_then_torque'], atol=1e-14)
    f = r['hand_force_world_n']; point = a['common_point_world_m']
    # Shift object torque from its origin to world origin before adding.
    total_torque = np.cross(point, f)+r['object_wrench_force_then_torque'][3:]+np.cross(a['object_wrench_origin_world_m'], -f)
    np.testing.assert_allclose(total_torque, 0., atol=1e-14)
    assert abs(r['power_identity_residual_w']) < 1e-12


@pytest.mark.parametrize("field,value", [
    ('signed_gap_m', float('nan')), ('signed_gap_m', True),
    ('object_outward_normal_world', np.array([0., 0., 2.])),
    ('object_outward_normal_world', np.array([0., 0., 1.], dtype=np.float32)),
    ('hand_point_jacobian', np.zeros((3,67))),
    ('hand_generalized_velocity', np.zeros(67)),
    ('object_point_jacobian', np.zeros((3,0))),
    ('common_point_world_m', np.array([0., np.inf, 0.]))])
def test_refuse_ambiguous_or_invalid_inputs(field, value):
    a = args(); a[field] = value
    with pytest.raises(ValueError): law.evaluate_contact(**a)


@pytest.mark.parametrize("values", [(-1., 2., .5, .001), (1., -1., .5, .001),
    (1., 1., -.5, .001), (1., 1., .5, 0.), (1., 1., float('nan'), .001),
    (True, 1., .5, .001)])
def test_refuse_invalid_parameters(values):
    with pytest.raises(ValueError): law.ContactParameters(*values)


def test_kernel_cannot_admit_collection_or_training():
    r = law.evaluate_contact(**args())
    assert not r['native_hand_qualified']
    assert not r['collection_allowed'] and not r['training_allowed']


def test_finite_inputs_cannot_emit_overflowing_forces_or_energy():
    a = args(); a['parameters'] = law.ContactParameters(1e308, 2., .5, .001)
    a['signed_gap_m'] = -2.
    with pytest.raises(ValueError, match='finite numerical domain'): law.evaluate_contact(**a)
