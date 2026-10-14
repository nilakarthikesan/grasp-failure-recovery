"""Save implemented code evidence without declaring pending fits successful."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
HERE = Path(__file__).parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
protected = json.loads((HERE / 'protected_hashes.json').read_text())
for p, digest in protected.items():
    assert sha(p) == digest, p

bundle = ROOT / 'research_artifacts/hug_constrained_refinement_2026-10-14'
implementation = bundle / 'implementation'
assert not bundle.exists(), bundle
implementation.mkdir(parents=True)
names = ['constrained_refinement.py', 'test_constrained_surface_refinement.py',
         'run_manufactured_checks.py', 'manufactured_protocol.json',
         'manufactured_verification.json', 'SOLVER_API.md',
         'api_installation_verification.json', 'installed_api_test_verification.json']
for name in names:
    (implementation / name).write_bytes((HERE / name).read_bytes())
manifest = {'schema': 'hug_constrained_implementation_checkpoint_manifest_v1',
            'scope': 'Implemented isolated API and manufactured verification. Saved-grasp fits and independent mapping audits remain pending.',
            'files_sha256': {name: sha(implementation / name) for name in names},
            'new_eval_data': False, 'new_inference': False, 'physics_steps': 0}
(implementation / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(bundle / 'README.md').write_text('''# HUG constrained refinement evidence

Implementation checkpoint, 14 October 2026. The isolated API and 50 new
manufactured tests are implemented; the installed API plus the existing
surface-refinement/guard tests pass all 120 checks. These are mathematical/API
checks, not certification of saved-grasp geometry.

The canonical joint-constrained experiment uses the six existing seed-0 HUG
apple/milk-box/mug proposals under runtime V2.1, protocol e42b74b787a8a1a5b5a661596466fe66b24c14eea39db25ad039fda2a462b8ff.
Its complete fits and unchanged independent six-coarse/three-fine audit are
**PENDING** at this checkpoint. No result from an incomplete search is accepted.

[Implementation evidence](implementation/manifest.json) binds the frozen API,
tests, settings and manufactured verification. The experimental API is identical
to the frozen solver used by the running experiment. The default mapper and
grasp-eval do not invoke it. No new generator inference, physical evaluation
episode, task label, predictor training or other-generator mapping occurs.

The completed experiment and independent reports will be added separately.
Historical protocols and incomplete runtime preparations will retain their
status; no previous scientific result is rewritten or pooled.
''')

doc = ROOT / 'docs/HUG_CONSTRAINED_REFINEMENT.md'
assert not doc.exists(), doc
doc.write_text('''# HUG fingertip-constrained, direction-aware refinement

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
''')

entries = {
    'docs/RESEARCH_DECISIONS.md': '''

## 14 October 2026 — Joint HUG fingertip/direction fitting implemented; real results pending

**Authorization and previous plan:** The user asks to implement fingertip
constraints and actual native-relative surface-direction fitting on existing
HUG grasps, then rerun mapping checks. New eval data and other-generator work
remain deferred. The previous staged/interpolated surface correction worsened
tip/middle errors and failed all nine strict audits.

**Implemented evidence:** The new isolated joint SLSQP API enforces each
fingertip's original-error bound plus the existing strict 10 mm mean guard;
final engineering checks name a 10 nm per-tip arithmetic allowance separately.
Actual native facing scores, including away states, are fixed targets. Newly
invalid fixed sites retain nonzero penalties and reasons. Source limits,
original±0.35 rad trust bounds, original-q fallback, best feasible major/raw
selection and nonconvergence bookkeeping are explicit. Fifty new manufactured
tests pass; the installed API and prior surface/guard tests pass all 120 checks.
The installed module is byte-identical to the frozen experiment solver.

**Runtime choice and status:** Canonical runtime V2.1 (e42b74b7…) runs all six
existing seed-0 apple/milk/mug cases from original q under the same method V1
settings. It uses two active Numba threads with pool four and BLAS/OpenMP one.
One/two-thread field/reference parity passes six baselines and three fixed
static poses. V1 incomplete searches and V2 import-only aborts remain separate;
the faster burial helper passed separate equality checks but its frozen V3
fitting preparation is unexecuted. Retain the active run rather than discard
over twenty minutes of progress for an uncertain whole-fit timing gain.

**Pending measured claims:** No completed six-case constrained result or new
independent mapping verdict exists at this checkpoint. Next finish all six
under fixed 120-iteration settings, recompute source/target preflight and apply
the unchanged six-coarse/three-fine rule. Do not promote from average fitting
improvement or engineering guard passage. Default mapper/controller, original
inputs, frozen predictor and manuscript remain unchanged. No inference,
integrated physics, eval episode/task label, prediction training or provider
advance occurs. [Implementation and scope](HUG_CONSTRAINED_REFINEMENT.md).
''',
    'docs/PAPER_PROGRESS.md': '''

## HUG constrained correction implementation, 14 October 2026 — results pending

[The joint constrained method](HUG_CONSTRAINED_REFINEMENT.md) is implemented in
an isolated API: five original-error fingertip constraints, strict mean guard,
actual native-relative direction targets and retained invalid-site penalties.
Fifty new manufactured tests pass; 120 installed new/prior API tests pass.
The six existing-grasp fits are still running under canonical runtime V2.1.
Independent source preflight and unchanged nine-resolution surface audit are
pending; no new geometry result or mapping sign-off is claimed yet.

Default production mapping, controller, original inputs, manuscript and frozen
predictor remain unchanged. No new generator inference, eval episode/label,
training or other-generator validation occurs. This is implementation progress
toward supporting mapping validity, not a completed multi-generator prediction
study or independent generalization result.
''',
    'docs/THREAD_HANDOFF.md': '''

## 14 October 2026 — Fingertip/direction code checkpoint; HUG fits still running

Latest authorization: implement fingertip constraints and surface direction
fitting on existing HUG grasps, rerun mapping checks, generate no new eval data
and do not move to another generator. [The new method](HUG_CONSTRAINED_REFINEMENT.md)
and isolated API are implemented. Fifty new manufactured tests and all 120
installed new/prior surface API tests pass; this is not geometry sign-off.

Canonical joint method V1 runtime V2.1 is running all six existing seed-0
apple/milk/mug grasps from original q, fixed 120-iteration budget. Protocol
e42b74b787a8a1a5b5a661596466fe66b24c14eea39db25ad039fda2a462b8ff is at
/private/tmp/hug_constrained_refinement_runtime_v2_1_2026-10-14/protocol.json.
Complete fits, source preflight and unchanged six-coarse/three-fine audit are
PENDING at this checkpoint. Do not treat live callback probes as selected poses.

The final independent wrapper is prepared in
/private/tmp/hug_constrained_validation_runtime_v2_1_2026-10-14, protocol
9f0e0833c80a5b36f0ad21f629ecb84cf399f0e2e7669436eb3b93e561faf8f5.
Preserve incomplete V1, import-aborted V2 and unexecuted faster-helper V3 history
separately. Do not restart the active fits for further performance work. The
default mapper/controller/model/manuscript and 114 protected source bytes remain
unchanged. Continue the authorized static checks; no new inference, physical
eval data, training or another-generator advance is authorized.
''',
}
for relative, entry in entries.items():
    path = ROOT / relative
    assert 'Joint HUG fingertip/direction fitting implemented; real results pending' not in path.read_text()
    path.write_text(path.read_text() + entry)
index = ROOT / 'docs/README.md'
original = index.read_text()
needle = '- [HUG surface-aware correction and independent validation](HUG_SURFACE_REFINEMENT.md)\n'
assert needle in original
index.write_text(original.replace(needle, needle + '- [HUG fingertip/direction constrained correction](HUG_CONSTRAINED_REFINEMENT.md)\n', 1))
for p, digest in protected.items():
    assert sha(p) == digest, p
print(json.dumps({'implementation_checkpoint_saved': True, 'mapping_verdict': 'PENDING',
                  'protected114_hashes_unchanged': True, 'new_eval_data': False}))
