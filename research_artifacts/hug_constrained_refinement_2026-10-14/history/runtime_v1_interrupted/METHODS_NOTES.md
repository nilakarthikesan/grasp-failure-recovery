# HUG constrained refinement: method notes

This method note describes the implementation, not a completed mapping result.
Results must be populated from the complete six-case report and independent
six-coarse/three-fine audit before publication.

## Source and target

The source is the original saved HUG MANO hand in its recorded object/world
frame. The target is the compiled Shadow Hand at simulation time zero. The
original HUG proposals, source meshes, object geometry and existing mapper are
held fixed. The candidate changes 24 original mapped wrist/finger targets
jointly, using a local SLSQP solve. It does not establish a global optimum or
that an exact human-to-robot mapping exists.

The native wrist and target palm POSITION stay aligned. Palm ORIENTATION follows
the existing q-dependent wrist convention. A compensating carrier translation
preserves that position; carrier rotation and hand/object scale are unchanged.
The final actual geometry is rebuilt outside optimization caches and checked
against separate source-URDF forward kinematics.

## Fingertip constraints

For each of the five fingertips, let e_i(q) be the 3D target-to-native positional
error. The nonlinear constraint is ||e_i(q)||² <= ||e_i(q_original)||². A separate
computed mean constraint requires mean_i ||e_i(q)|| <= 10 mm. Thus the solve
cannot trade a worse thumb for a better average. The starting errors are
approximation errors, not acceptable scientific fidelity tolerances.

Final engineering checks report exact nonworsening and nonworsening with an
explicit 10 nm arithmetic allowance separately. The 10 mm computed mean guard
is not enlarged. All source-URDF joint limits and the common original±0.35 rad
trust domain apply. Middle landmarks have no corresponding hard constraint;
their measured errors must be reported, including regressions.

## Surface and direction fitting

Fixed object-side surface quadrature fits unsigned anatomical distances and
closest-point displacement vectors against the native hand. It preserves the
source proposal's poor relationships as targets. It does not reward extra
contact, successful lifting, reduced penetration or all-facing directions.

For reliable native/original comparisons within the shared 5 mm band, the
direction target is a_native = -n_native_hand · n_object. The candidate score
uses the actual current hand outward normal. Negative native scores remain
negative targets. This score compares normal compatibility; it does not force
every component of the 3D normal to agree and does not measure contact forces.

The original shared eligibility is frozen. Each available anatomical part gets
64 area-quantile bins and mass 1/6. Repeated query bins retain quadrature mass
and are not independent samples. Missing parts retain missing coverage; their
mass is not redistributed. Original source coverage has no palm sites in any
case, no thumb sites for apple and no index sites for milk. Other anatomical
regions still appear in the independent full-grid audit.

For each fixed direction site, the candidate is checked again for proximity,
anatomical identity, nearest-feature ties, normal validity, query inside the
hand, buried target interfaces and closest points inside/on the visible object.
A newly unavailable candidate comparison keeps residual 2; it is not dropped
or assigned zero weight. These validity changes and their denominators remain
visible. Engineering invalid-feature penalties are not scientific direction
certificates.

## Candidate selection and independent acceptance

Settings and selection policy are frozen before the saved-grasp fits. Only
major solver iterates and the final raw result can be selected; finite
difference and line-search probes cannot. The lowest-objective feasible
improvement is retained, with earliest-major-iterate tie preference. If no
candidate qualifies, the exact original target/root is retained. Raw final
refusal, selected source/iteration, solver convergence and budget exhaustion
are recorded separately. Feasible selection does not certify convergence.

The final independent audit retains the original query cells, anatomical
definitions, 1/2/5 mm bands, uncertainty allowances, normal eligibility and
strict acceptance rule. Lower fitting error and fingertip guard passage are
not strict mapping acceptance. Definite lost/added area or reliable sampled
direction reversals refute strict preservation. Missing continuous normal
certificates and ambiguous distance cells retain UNRESOLVED status. No
successful lift is used to establish geometric fidelity.

These are development cases from existing saved proposals and one MANO shape.
They do not establish unseen-object generalization, native physical-contact
equivalence, native failure-label preservation or predictor adequacy. No HUG
inference, integrated physics step, eval episode, new task label or model
training is authorized or performed in this phase.
