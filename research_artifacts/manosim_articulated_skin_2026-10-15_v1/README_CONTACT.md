# A separate articulated contact model, not a stock-flex force correction

Status: implemented pure numerical kernel with manufactured controls. No
native contact search, engine callback, integration or hand calibration is
provided by this file. Collection and training admission remain false.

The completed fixed-pose surface reference preserves all original MANO skin
triangles. Attaching that surface to one palm does not articulate fingers.
The earlier MuJoCo 3.4.0 multi-owner primitive also fails the material-point
derivative check: normalized inverse-distance owner weights are not the
derivative of MANO's blended skin and pose correctives. Changing only vertex
positions would therefore leave incorrect forces or velocities.

The velocity/Jacobian/force-transpose relationship and the distinction between
quaternion position coordinates and tangent velocities follow the pinned
[MuJoCo 3.4.0 computation reference](https://mujoco.readthedocs.io/en/3.4.0/computation/#general-framework).
The versioned primitive failures above are repository measurements, not claims
made by that documentation.

The selected next implementation uses the complete continuous MANO skin
function x(q), its 51-dimensional local-tangent Jacobian J, and generalized
contact loads J transpose times force. It is a separately versioned compliant
contact model; it must pass its own integration qualification. The stock
MuJoCo rigid-contact fixtures in the adjacent dynamics artifact do not qualify
this law or a future native-hand callback.

`contact_law.py` takes an explicit signed gap, reliable outward object normal,
one common world point, matching hand/object point Jacobians and generalized
velocities, and immutable parameters. It never guesses a normal, repairs an
input quaternion, or substitutes 67 position coordinates for 51 velocities.
The same world point is used for both reactions so tangential forces do not
introduce an artificial force couple. A future object adapter must provide
the actual engine point Jacobian rather than treating its local free-joint
angular velocity as a world angular velocity.

For penetration delta=max(0,-gap), the stored energy is k*delta^2/2. When
gap<0, normal force is max(0,k*delta-c*v_normal); otherwise it is zero.
Friction is -mu*F_normal*v_tangent/sqrt(|v_tangent|^2+v_regularization^2).
The clamp prevents tensile contact. With a geometrically correct gap rate,
the derivative of stored energy plus mechanical contact power equals
(F_normal-k*delta)*v_normal + F_friction dot v_tangent, which is nonpositive.
This is a pointwise energy identity, not proof that a contact finder gives
the correct gap derivative or that a discretized integrator is stable.

Manufactured controls cover contact/no-contact branches, separating clamp,
normal and friction dissipation, the elastic potential gradient, Coulomb
force bound, generalized virtual work, reaction force/torque at a common
point, and explicit refusals for invalid dimensions, precision and normals.
Their numerical parameters are test fixtures, not measured hand properties.

Before native integration: qualify continuous skin and its MuJoCo tangent
binding, reliable oriented contact geometry, this law's controlled analytic
integration fixtures, callback force replacement, timestep convergence and
energy/momentum accounting. Real-world fidelity additionally needs physical
calibration and comparison with real observations. No existing result proves
those future steps.
