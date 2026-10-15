# HUG objective diagnostic (2026-10-14)

Read-only diagnosis of the completed original constrained attempt. A frozen 60-call stencil used the literal original V2.1 callback on saved cases 8/20/28; no optimizer, generated grasp, physics step, outcome label or eval data.

## What the recorded old fit actually did

All six first recorded major callbacks move many joints to their 0.35 rad correction limit. This large local-SQP departure worsens nonlinear tip errors; intermediate local-SQP steps are not promised to be feasible. The best maximum tip increase over recorded majors is still above the declared 10 nm final allowance in every case. This is evidence of a failed local procedure, not a proof that no feasible mapping correction exists.

- Case 8: 12 joints at 0.35 rad on first callback; 43/94 callbacks improve scalar cost; zero eligible improving callbacks. Smallest recorded maximum tip increase 69.880548 µm.
- Case 12: 11 joints at 0.35 rad on first callback; 23/87 callbacks improve scalar cost; zero eligible improving callbacks. Smallest recorded maximum tip increase 197.151931 µm.
- Case 16: 17 joints at 0.35 rad on first callback; 52/117 callbacks improve scalar cost; zero eligible improving callbacks. Smallest recorded maximum tip increase 226.701458 µm.
- Case 20: 14 joints at 0.35 rad on first callback; 108/118 callbacks improve scalar cost; zero eligible improving callbacks. Smallest recorded maximum tip increase 0.049410 µm.
- Case 24: 14 joints at 0.35 rad on first callback; 12/116 callbacks improve scalar cost; zero eligible improving callbacks. Smallest recorded maximum tip increase 59.262828 µm.
- Case 28: 13 joints at 0.35 rad on first callback; 5/117 callbacks improve scalar cost; zero eligible improving callbacks. Smallest recorded maximum tip increase 4.124901 µm.

## Tiny objective probes

For WRJ1, FFJ1 and THJ1 at each representative q0, symmetric 1e-6 and 1e-7 rad steps produced no normal invalidity/part/geometry/tie identity change; slopes converge. Positive 1e-5 rad steps also retain valid sites. The two negative 1e-5 rad MugB changes described below are exceptions. This narrow probe distinguishes a measured nearby objective jump from proof that this jump caused all six solver failures; other coordinates and later feature switches remain untested.

- Case 8: facing cost 0.01206557/7.56703342 (0.159%) at q0; raw final facing cost 0.16377271 with 31 invalid bins. Central slopes at 1e-7 rad: WRJ1 -29.35274687, FFJ1 -0.47004545, THJ1 -2.80810939cost/rad.
- Case 20: facing cost 0.02914575/12.44964735 (0.234%) at q0; raw final facing cost 0.41567365 with 76 invalid bins. Central slopes at 1e-7 rad: WRJ1 -24.72894788, FFJ1 -0.03551400, THJ1 -2.81302293cost/rad.
- Case 28: facing cost 0.04065645/8.41083154 (0.483%) at q0; raw final facing cost 0.30578178 with 54 invalid bins. Central slopes at 1e-7 rad: WRJ1 -63.37621033, FFJ1 -0.84006229, THJ1 -0.89269239cost/rad.
- Case 28 WRJ1 -1e-05 rad invalidates thumb bin 283 (mapped_mesh_competing_facet); facing cost jumps 0.00481645. Native score 0.34065833; original mapped score 0.61138317; unavailable candidate score 0.61138853 is correctly replaced by the fixed 2.0 residual before weighting.
- Case 28 THJ1 -1e-05 rad invalidates thumb bin 283 (mapped_mesh_competing_facet); facing cost jumps 0.00481199. Native score 0.34065833; original mapped score 0.61138317; unavailable candidate score 0.61137930 is correctly replaced by the fixed 2.0 residual before weighting.

## Implication for the next development attempt

Preserve strict per-tip eligibility and the independent mapping audit. Use a much smaller feasibility-aware development step/line-search instead of assuming that local SQP trial poses remain feasible; independently recompute all accepted poses. The objective stencil does not justify weakening the constraints or retuning away bad native grasp proposals.

The original objective retains fixed invalid-site penalties. The actual penalty is piecewise and jumps when eligibility changes; this is both a mathematical property of the definition and measured in the two negative MugB perturbations. Raw endpoints contain further invalid bins, but those endpoint observations do not identify causal transitions along the solver trajectory.

Protocol SHA256: `940d7647062a86df3ae4e35296bb2aea94058f67c544641f8730a7e7c8c9ee43`. Geometry report SHA256: `080b4f8425770979a9f6073d6f77eefefb0686799d5ac046b531b0edc0e5f6a7`. All 60 full geometry calls were static; integration attempts 0, deliberate guard selftests 9. Source hashes remained unchanged.
