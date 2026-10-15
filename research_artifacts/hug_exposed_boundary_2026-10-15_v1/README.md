# HUG native exposed-boundary diagnosis — 15 October 2026 v1

Completed static measurements on **all 32 existing saved poses**, retaining the
unchanged `native_mano_static_capsule_hull_v1` geometry. This is development
diagnosis, not a collision-fidelity acceptance or qualification for execution.
No inference, retargeting, integration, outcomes, labels, collection or training
was performed by this audit. The run counted 65 static forwards and zero calls
to guarded integration/refitting APIs. No paid compute was used.

`protocol.json` and all source snapshots were frozen before measurement in
commit `5fe44be`. Protocol SHA-256:
`4b0e841bfd91fa5583fe73ebdaffbcd1997cc9fd9c9bc3ce970ba6d4325ab461`.
All frozen input/source bindings were verified before and after measurement.
The original comparison's measured-driver binding resolves to its documented
archived original bytes; its later serialization repair is not substituted
silently. Existing Shadow measurements are hash-linked, not replayed or refit.

## Measured findings

At 2,048 probes per case, there are 65,536 object-surface geometric queries,
not 65,536 independent grasps. All 32 saved meshes reconstruct exactly from
their saved MANO parameters and pass closed oriented-edge/volume controls.
These topology checks **do not certify absence of self-intersections**.

- The native collider union contains 4,762 probes. Relative to resolved saved
  skin winding occupancy, it adds 1,461 penetration flags and loses 426. Three
  skin occupancy queries remain unknown because of the frozen 1 micrometre
  surface-side exclusion. These are geometric occupancy differences, not new
  physical failure labels.
- The nearest individual component projection is buried inside another
  collider at 1,931 probes. These projections are explicitly excluded as
  exposed-union distance claims. For queries outside the whole union, nearest
  component distances/projections are exact to the represented primitives and
  are checked exposed. Interior distance is an interval.
- The largest sampled penetration has a **10.615 mm lower bound and a
  12.122 mm upper bound** on the maximum over all probes. The two maxima need
  not occur at the same point. The largest pointwise interval width is
  3.648 mm. The maximum upper bound occurs in saved mug case 29.
- Of 9,109 probes within 5 mm of the skin, 3,526 have eligible local normals in
  both representations, and 73 have a nearest-patch normal angle above 90
  degrees. **5,583 normal queries remain unknown.** These angles compare each
  representation's nearest patch; they do not certify an anatomical
  correspondence or reproduce the historical orientation acceptance test.
- At the original literal 512 probes per case, the corresponding counts are
  366 added/129 lost penetration flags, 463 buried component projections, and
  19 angles above 90 degrees among 899 eligible normal comparisons. The
  changes survive using the original probes.

The sampling study uses a fixed nested 128/512/2,048 scheme, independent of hand
errors. Maximum penetration upper bounds increase from **8.592 to 11.129 to
12.122 mm**. Thus a coarse finite maximum would miss deeper discrepancies.
This is density sensitivity, not a confidence interval or a bound over the
continuous object surface. The new whole-union near-band counts also differ
in meaning from the earlier sum over six anatomical component distances; do
not compare those denominators or replace historical failures with this study.

At 2,048 probes per case, added/lost penetration counts are cube 525/166,
apple 347/114, milk box 225/70 and mug 364/76. The cube's archived object ID is
`object01`. All four objects exhibit changes; no easy-case filtering was used.

## Method and remaining uncertainty

The compiled capsule endpoints/radii and convex palm are read from the
unchanged static MANO model. For exterior union queries, the nearest component
projection is an exposed boundary point. A smooth unique component supplies
its outward normal; edge, joint/tangent and interior cases do not supply a
unique exposed normal here.

For interior queries, the maximum depth in any containing component is a
lower bound on distance to the union exterior. Six independently checked
exterior axis witnesses supply an upper bound; 36 bisections refine each
witness. A segment may pass through other/disconnected components, so these
are valid exterior witnesses, not a claim of the first ray exit or the optimal
exit. A near-band assignment stays unknown when its interval straddles the
band. No minimum unsigned component distance is used as union penetration.

Skin occupancy uses oriented solid-angle winding, with a globally declared
volume orientation and explicit 0/1 eligibility. A closest skin face supplies a
normal only in its interior and when offsets by plus/minus 1 micrometre give
outside/inside winding witnesses. This establishes local orientation at that
patch, not a globally injective physical skin. The 1 micrometre and 1 nanometre
values are numerical eligibility controls, **not allowed anatomical errors**.

Object probes use the same captured object representation as the prior audit:
visual triangles where available, collision geometry otherwise (cube). Their
geometry/metadata and all source cases are bound, but no visual-to-physical
object discrepancy is certified. Finite queries can miss narrow features.
No global Hausdorff, continuous contact preservation, physical contact, mass,
inertia, actuation, controller or outcome claim is established.

The tested capsule/convex-palm collision approximation therefore still needs
work before native execution qualification. The next bounded geometry study
should retain these witnesses/unknowns, instrument the responsible components,
and compare a prospectively fixed, more faithful source-derived decomposition
under the same saved inputs. Choosing an error allowance merely to pass these
development cases would not supply an independent physical rationale. The
current Shadow default and every historical criterion/result are unchanged.

## Contents and reproduction

- `source/`: exact protocol-bound module, driver, model and manufactured tests.
- `protocol.json`: frozen source/input bindings and measurement definitions.
- `cases/`: all 32 immutable scalar case reports, including all three densities.
- `results.json`: runtime, counters, report hashes and density aggregates.
- `run.log`: complete successful measurement log.
- `test.log`: 16 passing manufactured controls; pure static geometry.
- `verification.json`: source/test/protocol receipt and known scope.
- `manifest.json`: archive-relative SHA-256 values, excluding itself.

The tests cover exact capsule and convex distances, overlap/buried-interface
counterexamples, lower/upper bounds, normal ambiguity refusal, winding,
reversed/open topology, local two-sided witnesses, and rigid covariance. Before
protocol freeze, test setup required the existing external dependency path and
canonical source path, and a manufactured zero comparison was given a
1e-14 numerical assertion tolerance. No measured geometry was changed and no
cohort run failed or was selectively repeated.

Use the existing runtime and licensed/private inputs; the archive does not
redistribute MANO or source proposal geometry. Create a new output directory;
never overwrite this archive:

```sh
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
  scripts/audit_hug_exposed_boundary.py prepare \
  --project /Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction \
  --output /absolute/path/to/a/new/versioned/audit
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
  scripts/audit_hug_exposed_boundary.py run \
  --output /absolute/path/to/a/new/versioned/audit
```

Base NumPy/MuJoCo load before external dependency paths, as in the driver.
Inputs and dependencies are external historical paths; verify they are still
available before attempting a new reproduction. Current source must match its
frozen bindings, or use the archived source snapshot in a separately declared
reproduction protocol.
