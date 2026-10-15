"""Explicit compliant point-contact law, separate from stock MuJoCo flex.

This pure numerical kernel neither finds contacts nor integrates a simulator.
The caller must provide a reliable signed gap and an outward object normal,
a single common world-space application point, and the matching hand and
object point Jacobians. It is not a native-hand admission certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import math
import numpy as np

LAW_SCHEMA = "mano_compliant_common_point_kelvin_voigt_coulomb_v1"


@dataclass(frozen=True)
class ContactParameters:
    stiffness_n_per_m: float
    damping_ns_per_m: float
    friction: float
    friction_speed_m_per_s: float

    def __post_init__(self):
        values = (self.stiffness_n_per_m, self.damping_ns_per_m,
                  self.friction, self.friction_speed_m_per_s)
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float))
               or not math.isfinite(x) for x in values):
            raise ValueError("Finite explicit real contact parameters required")
        if (self.stiffness_n_per_m <= 0 or self.damping_ns_per_m < 0
                or self.friction < 0 or self.friction_speed_m_per_s <= 0):
            raise ValueError("Positive stiffness/speed and nonnegative damping/friction required")


def _array(value, shape, name):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float64
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError("Explicit finite float64 " + name + " required")
    return value


def _evaluate_contact(*, signed_gap_m, object_outward_normal_world,
                     common_point_world_m, object_wrench_origin_world_m,
                     hand_point_jacobian, object_point_jacobian,
                     hand_generalized_velocity, object_generalized_velocity,
                     parameters):
    """Return force, equal/opposite generalized loads, and a power identity.

    Gap is positive outside the object. The normal points out of the object
    toward the hand, and v_rel = v_hand(point)-v_object(the SAME point).
    For gap < 0: delta=-gap, Fn=max(0,k*delta-c*vn). Otherwise Fn=0.
    Ft=-mu*Fn*vt/sqrt(|vt|^2+speed^2). No attraction, no force at gap>=0.

    E=.5*k*delta^2 and E_dot=-k*delta*vn. Thus
    E_dot+F.v_rel=(Fn-k*delta)*vn+Ft.vt <= 0, including the clamped
    separating branch. Contact-point/normal changes require a qualified
    geometric gap derivative; this identity alone does not establish it.
    Jacobians are inputs, never inferred from nq, and velocities have the
    same tangent-coordinate convention as their corresponding Jacobians.
    """
    if not isinstance(parameters, ContactParameters):
        raise ValueError("Versioned explicit ContactParameters required")
    if (isinstance(signed_gap_m, (bool, np.bool_))
            or not isinstance(signed_gap_m, (int, float))
            or not math.isfinite(signed_gap_m)):
        raise ValueError("Finite signed gap required")
    normal = _array(object_outward_normal_world, (3,), "outward normal")
    if abs(float(normal @ normal)-1) > 1e-12:
        raise ValueError("Literal unit normal required; no normalization or orientation repair")
    point = _array(common_point_world_m, (3,), "common world point")
    origin = _array(object_wrench_origin_world_m, (3,), "object wrench origin")
    jh = _array(hand_point_jacobian, (3, 51), "named hand point Jacobian")
    if not isinstance(object_point_jacobian, np.ndarray) or object_point_jacobian.ndim != 2:
        raise ValueError("Explicit object point Jacobian required")
    no = object_point_jacobian.shape[1]
    if no < 1:
        raise ValueError("Object Jacobian must have at least one coordinate")
    jo = _array(object_point_jacobian, (3, no), "object point Jacobian")
    vh = _array(hand_generalized_velocity, (51,), "named hand velocity")
    vo = _array(object_generalized_velocity, (no,), "object velocity")
    relative_velocity = jh @ vh - jo @ vo
    vn = float(normal @ relative_velocity)
    vt = relative_velocity-vn*normal
    penetration = max(0., -float(signed_gap_m))
    elastic = parameters.stiffness_n_per_m*penetration
    magnitude = (max(0., elastic-parameters.damping_ns_per_m*vn)
                 if signed_gap_m < 0 else 0.)
    tangent_force = (-parameters.friction*magnitude*vt
                     / math.sqrt(float(vt @ vt)+parameters.friction_speed_m_per_s**2))
    hand_force = magnitude*normal+tangent_force
    object_force = -hand_force
    hand_tau = jh.T @ hand_force
    object_tau = jo.T @ object_force
    energy_rate = -elastic*vn
    normal_dissipation_power = (magnitude-elastic)*vn
    friction_power = float(tangent_force @ vt)
    mechanical_power = float(hand_tau @ vh + object_tau @ vo)
    return dict(schema=LAW_SCHEMA, signed_gap_m=float(signed_gap_m),
                penetration_m=penetration, normal_force_n=magnitude,
                hand_force_world_n=hand_force, object_force_world_n=object_force,
                common_point_world_m=point.copy(),
                object_wrench_force_then_torque=np.concatenate((object_force,
                    np.cross(point-origin, object_force))),
                hand_generalized_load=hand_tau, object_generalized_load=object_tau,
                relative_velocity_world_m_per_s=relative_velocity,
                elastic_energy_j=.5*parameters.stiffness_n_per_m*penetration**2,
                elastic_energy_rate_w=energy_rate,
                mechanical_power_w=mechanical_power,
                normal_dissipation_power_w=normal_dissipation_power,
                friction_power_w=friction_power,
                power_identity_residual_w=energy_rate+mechanical_power
                    -normal_dissipation_power-friction_power,
                native_hand_qualified=False, collection_allowed=False,
                training_allowed=False)


@wraps(_evaluate_contact)
def evaluate_contact(**kwargs):
    """Refuse overflow as well as nonfinite inputs; no nonfinite loads escape."""
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            result = _evaluate_contact(**kwargs)
            for key, value in result.items():
                if isinstance(value, (np.ndarray, float, int)) and not np.isfinite(value).all():
                    raise ValueError('Nonfinite contact output: '+key)
        return result
    except (FloatingPointError, OverflowError) as error:
        raise ValueError('Contact arithmetic exceeded finite numerical domain') from error
