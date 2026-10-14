# HUG fingertip-constrained, direction-aware refinement

14 October 2026. **The new experimental code is implemented; the six real
saved-grasp fits and independent mapping verdict are pending.** The default
mapper remains unchanged. The last completed staged correction still failed
strict preservation in all six cases and three finer repeats; see
[the previous result](HUG_SURFACE_REFINEMENT.md).

This is joint-constrained method V1, using canonical **runtime V2.1**. Those
runtime names are separate from the earlier staged surface V2.1/interpolation
V3.1 method. Only completed results from the canonical run will determine its
measured geometry result.

## What the new code changes

The [isolated API](../src/grasp_failure_prediction/evaluation/constrained_surface_refinement.py)
jointly fits all 24 original Shadow wrist/finger targets with a local SLSQP
solve. Source-URDF bounds and a common original±0.35 rad trust domain apply.
The original HUG proposal, object geometry, hand scales, carrier rotation and
existing mapper are fixed. This does not certify a global optimum or existence
of an exact human-to-Shadow configuration.

For each of the five fingertips, the native-relative position error must not
exceed that fingertip's **original mapping error**. A separate mean error guard
remains strictly ≤10 mm. Thus improving the average cannot conceal a worse
thumb. Final checks report exact nonworsening and nonworsening with a declared
10 nm arithmetic allowance separately. This allowance is not a scientific
contact-fidelity tolerance, and the 10 mm mean guard is not enlarged. Starting
errors remain approximation errors; the constraints do not force exact native
tip positions. Middle landmarks have no corresponding hard constraint and
their errors, including regressions, will be reported.

The MANO wrist and Shadow palm POSITION stay aligned. Palm ORIENTATION follows
the existing q-dependent wrist convention. Carrier translation only compensates
that wrist articulation. Final geometry is recomputed outside optimizer caches
and checked against independent source-URDF forward kinematics.

## What surface direction means here

The fixed objective retains native anatomical unsigned distances and closest
point vectors. Direction fitting adds the actual native object-facing score
a = −n_hand · n_object. Native negative/away-facing scores remain targets too.
This compares normal compatibility; it does not constrain every component of
the 3D normal vector or establish contact-force equivalence. Poor source poses
are retained rather than rewarded for extra contact or successful lifting.

Reliable native/original comparisons in the shared 5 mm band define fixed
direction sites. Each available part has 64 area-quantile bins and mass 1/6.
Repeated bins retain quadrature mass; they are not independent samples.
Missing-part mass is not redistributed. The original stratum has no palm sites
in any case, no thumb sites for apple and no index sites for milk. Distance and
closest-point fitting still cover all six anatomical parts, and the independent
full-grid audit covers regions outside the fitting stratum.

Actual candidate normals, proximity, anatomical identity, nearest-feature ties,
inside/buried states and visible-object membership are checked at every fixed
site. A newly unavailable comparison retains residual 2 and an explicit reason;
it is not dropped or reweighted to zero. The source-bound helper reproduces the
previous original paired eligibility counts and areas in all six cases. This
checks its implementation, not the candidate mapping result.

## Selection, tests and pending acceptance

Common settings and selection were frozen before the first constrained fits:
120 iterations, eps 1e-5 rad, ftol 1e-9 and robust f_scale 0.2. Only major solver
iterates and the final raw result can be selected. The lowest-objective feasible
improvement is retained, with an earlier major iterate preferred on a tie. If
none qualifies, the exact original target/root is retained. Line-search and
finite-difference probes are not candidates. Raw refusal, selected iteration,
convergence and budget exhaustion are reported separately. A feasible selection
does not certify convergence or scientific mapping acceptance.

[The 50 new manufactured tests](../tests/test_constrained_surface_refinement.py)
pass. Together with the previous surface/guard tests, **120 installed API tests
pass**. They check per-tip constraints, the average-error loophole, native-away
targets, invalid-site penalties, major/raw selection, exact fallback and
defensive callback handling. [Frozen implementation evidence](../research_artifacts/hug_constrained_refinement_2026-10-14/implementation/manifest.json)
records versions and exact source hashes. The installed API and experiment
solver are byte-identical; these tests are not real-grasp sign-off.

The ongoing canonical runtime uses six independent singleton processes, two
active Numba query threads and one BLAS/OpenMP thread per process. Pool capacity
four permits the preserved geometry library to import before the active mask is
set to two. Exact field/reference parity was checked on six original poses and
three fixed static perturbations. An earlier capacity-two launch stopped at
import before any case or solve. Earlier incomplete searches and an unexecuted
faster-helper preparation remain separate history, not scientific results.

After all six fits finish, independent source preflight will recompute every
original/final fingertip error, physical/trust bounds, root convention and
recorded candidate selection. The unchanged six-coarse/three-fine audit then
uses the original 1/2/5 mm bands, query cells, anatomical definitions, normal
eligibility, numerical allowances and strict acceptance rule.

Lower fitting error or passing fingertip guards cannot establish strict
preservation. Definite lost/added area and reliable sampled direction reversals
remain counterexamples. Continuous-normal evidence and ambiguous distance cells
remain UNRESOLVED where the old audit cannot certify them. Physical contact,
native failure-label preservation, independent unseen-object validation and
predictor adequacy remain unresolved or untested. No new HUG inference, physics
rollout, eval episode/label, predictor fit, manuscript edit or other-generator
advance occurs in this phase.
