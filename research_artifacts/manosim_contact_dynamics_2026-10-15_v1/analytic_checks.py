"""Independent analytic checks for explicitly scoped, manufactured mechanics."""
from __future__ import annotations
import numpy as np

MASS = .2
RADIUS_M = .01
GRAVITY_M_S2 = 9.81
MU = .5
IMPEDANCE = .9
STIFFNESS_S2 = 10000.
DAMPING_S = 200.
LIMITS = dict(jacobian=1e-10, virtual_work_W=1e-10,
    discrete_freefall_m=1e-10, discrete_freefall_m_s=1e-10,
    support_force_relative=.01, support_penetration_m=5e-6, settled_velocity_m_s=1e-4,
    regularized_creep_relative=.05, regularized_creep_floor_m_s=1e-5,
    sliding_acceleration_relative=.05, friction_cone_relative=1e-8,
    momentum_kg_m_s=1e-9, angular_momentum_kg_m2_s=1e-10, action_reaction_N=1e-9, angular_action_reaction_N_m=1e-10,
    force_balance_N=1e-8, discrete_work_J=1e-9,
    cone_primal_relative=1e-8, cone_dual_relative=1e-7, cone_complementarity_relative=1e-7,
    impact_energy_gain_fraction=.01,
    convergence_position_m=1e-4, convergence_velocity_m_s=1e-3)


def finite(value, shape=None):
    a = np.asarray(value, dtype=float)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError('Finite analytic operands with declared shape required')
    return a


def rest_penetration(gravity=GRAVITY_M_S2, impedance=IMPEDANCE, stiffness=STIFFNESS_S2):
    """Constant impedance/direct-solref scalar normal equilibrium, metres.

    d*a_ref+(1-d)*a_unconstrained=0, k=stiffness/d,
    a_ref=k*delta: delta=g*(1-d)/stiffness. Requires exact scalar inverse mass.
    """
    if not np.isfinite([gravity,impedance,stiffness]).all() or not 0 < impedance < 1 or stiffness <= 0 or gravity < 0:
        raise ValueError('Positive physical scalar reference parameters required')
    return gravity*(1-impedance)/stiffness


def regularized_creep(force_N, mass=MASS, impedance=IMPEDANCE, damping=DAMPING_S):
    """Interior friction cone, scalar translation: v=(1-d)*F/(m*damping).

    This is a soft-friction creep reference, never an exact-stick assertion.
    """
    if not np.isfinite([force_N,mass,impedance,damping]).all() or mass <= 0 or damping <= 0 or not 0 < impedance < 1:
        raise ValueError('Positive scalar reference parameters required')
    return (1-impedance)*force_N/(mass*damping)


def euler_freefall(z0, v0, h, steps, gravity=GRAVITY_M_S2):
    if not np.isfinite([z0,v0,h,gravity]).all() or h<=0 or type(steps) is not int or steps<0:
        raise ValueError('Finite initial conditions and exact positive step interval required')
    return z0 + v0*h*steps - .5*gravity*h*h*steps*(steps+1), v0-gravity*h*steps


def quaternion_matrix(value):
    w,x,y,z=finite(value,(4,))
    if abs(np.linalg.norm(value)-1)>1e-8:raise ValueError('Raw unit wxyz quaternion required')
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def point_velocity_jacobian(rotation, lever):
    """Free-joint world translation/local angular DOFs -> world point velocity."""
    r, p = finite(rotation, (3, 3)), finite(lever, (3,))
    return np.column_stack([np.eye(3), np.column_stack([np.cross(r[:, k], p) for k in range(3)])])


def contact_jacobian(frame, positions, rotations, dof_addresses, contact_position, nv):
    """Contact frame rows times (body1 point velocity - body0 point velocity)."""
    basis, point = finite(frame, (3, 3)), finite(contact_position, (3,))
    result = np.zeros((3, nv))
    for sign, position, rotation, address in zip((-1, 1), positions, rotations, dof_addresses, strict=True):
        if address is not None:
            result[:, address:address+6] += sign*basis@point_velocity_jacobian(rotation, point-finite(position, (3,)))
    return result


def virtual_work(jacobian, velocity, contact_force):
    j = finite(jacobian); v = finite(velocity, (j.shape[1],)); f = finite(contact_force, (j.shape[0],))
    return float(f@(j@v)), float((j.T@f)@v)


def cone_residual(force, gradient, mu):
    f, g = finite(force), finite(gradient)
    if f.shape != g.shape or f.shape not in ((1,), (3,)) or mu < 0:
        raise ValueError('One normal or three elliptic contact coordinates required')
    tangential = np.linalg.norm(f[1:]) if len(f) == 3 else 0.
    dual_tangent = np.linalg.norm(g[1:]) if len(g) == 3 else 0.
    return dict(primal_N=float(max(0., -f[0], tangential-mu*f[0])),
        dual_m_s2=float(max(0., mu*dual_tangent-g[0])),
        complementarity_N_m_s2=float(abs(f@g)),
        force_scale_N=float(1+np.linalg.norm(f)), gradient_scale_m_s2=float(1+np.linalg.norm(g)))


def discrete_force_work(mass, v0, v1, total_force, h):
    before, after, force = finite(v0), finite(v1), finite(total_force)
    if mass <= 0 or h <= 0 or before.shape != after.shape or force.shape != before.shape:
        raise ValueError('Aligned constant-mass translation operands required')
    force_error = mass*(after-before)/h-force
    energy_change = .5*mass*(after@after-before@before)
    midpoint_work = h*force@(.5*(before+after))
    return float(np.max(np.abs(force_error))), float(abs(energy_change-midpoint_work))


def action_reaction(force, point, origins):
    f, p = finite(force, (3,)), finite(point, (3,))
    # Torques and orbital angular momentum use the SAME application point.
    torques = [np.cross(p-finite(origin, (3,)), sign*f)
               for sign, origin in zip((-1, 1), origins, strict=True)]
    total_torque = sum(torques) + sum(np.cross(finite(origin), sign*f)
                    for sign, origin in zip((-1, 1), origins, strict=True))
    return float(np.linalg.norm(-f+f)), float(np.linalg.norm(total_torque))
