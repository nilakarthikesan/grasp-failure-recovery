# HUG-only system build — 15 October 2026

The user assigns the manuscript outside this repository. Build work must not
edit `docs/paper` or maintain manuscript progress. Continue on the existing
`codex/project-work` checkout; preserve unrelated dirty work and old experiments.

## Mapping fidelity is the current prerequisite

The user's renewed decision on 15 October is to repair and validate transfer
before more task-data collection or predictor training. Prior numerical passes,
completed rollouts and trained models remain historical results; they do not
qualify the native-to-Shadow surface claim. New collection admission must refuse
before filters or task execution if native-surface fidelity is unqualified.
Static diagnosis remains available. Simulator gap experiments follow the mapping
verdict, rather than being evidence that a failed mapping works.

The independent-static solver profile fixes a measured objective/gradient and
state-reuse defect. It is an explicitly separate mapping strategy for validation,
not an accepted replacement for the legacy mapping. Surface fidelity still needs
its own verdict under the unchanged acceptance rule. The bounded static
check covers all native-definite near/far relationships in the saved development
grids, preserving previous passing regions as well as previously failed ones.

The complete measured fields now refute the previous two fitted candidates:
milk case 20 has 530→684 definite mismatches; mug case 28 has 3183→2874 but loses 2112
previously passing relationships. Both fail; overlapping tuple counts are not
independent contacts. The static current-scene 30D pose adapter passes 17 controls
including independent source FK, all-column Jacobians, saved-root replay and
object/state invariance. It supplies a qualified numerical interface for a new
bounded repair, not a fidelity result. Actual-operand replay traced the warnings
to checked halfspace operations: independent arithmetic agreed within 5.55e-17 m
with no predicate disagreements. The new kernel changes only those three products;
this does not certify all distance calculations. Task collection and
model training remain paused.

The protected v2 exchange pilot also completes without an accepted relationship
repair: milk still has 530 definite mismatches, and both proposed steps create a
new little-finger violation; mug's fingertip restoration loses one protected
thumb relationship and is refused, retaining 3183 original definite mismatches.
An independent saved-array receipt reproduces all seven checkpoint fields.
These are two repeatedly inspected development grasps. A separately versioned
repair now addresses joint restoration and counterexample activation; it does
not weaken the acceptance rule. Continuous fidelity remains unproved. The
legacy object visual-transform warning is a separate unresolved arithmetic
limitation; the halfspace replay only cleared its named products.

Scope distinction: these full-field replays and local repair pilots begin from
saved legacy translator poses. They do not measure the newly registered
independent-static translator's actual surface fields. Its production-entry
output needs a separate, frozen native-input surface audit; passing its numerical
controls does not fill that evidence gap.

V3 joint restoration and counterexample retention now meet the original tip
caps and retain all previously passing fields in both development cases.
Milk's definite failures change 530 to 529 through one newly unresolved tuple;
mug resolves four prior unknowns while all 3183 definite failures persist.
Neither is faithful-transfer admission. The user's MANOSim reference is the
next priority: its native HUG hand shape matches our source bytes and permits
direct MANO joint transfer. Static pose qualification and separate collision
approximation measurements are being built; simulator realism remains unproved.

The first native test now refuses before all forwards because the actual grasp
betas differ from the nominal shared shape file. The two inspected exports use
the same values, exactly consistent with BF16 rounding. A shape-matched hand
must be built from those literal parameters before posing or collider testing;
the canonical hand is not silently substituted. No native fidelity result or
new task data follows from that refusal.

## Beginning of the system: decision and evidence boundary

The executable chain is rendered RGB-D/calibration → pretrained HUG MANO output
→ wrist-local landmarks → named Shadow targets/object-relative palm → MuJoCo
controller. HUG inference runs locally from pinned source and weights; it does
not download a pre-labelled successful grasp for each scene.

The mapping uses the same official Shadow URDF in Dex Retargeting and MuJoCo.
This gives a checkable common robot geometry. Camera/wrist transforms remove the
global human-hand pose before fitting local vectors. Explicit axis rotation,
joint-name lookup, metric scale, literal joint bounds and independent XML versus
compiled forward kinematics prevent silent frame, ordering and asset errors.
Static proposals disable streaming smoothing and the packaged 1.2 vector scale;
the retained legacy solver is instantiated afresh per proposal. Twenty solver
calls and all raw/status/clipping records remain auditable. Positive solver
termination codes do not prove a global optimum or contact preservation.

The MANO wrist is anchored to the Shadow palm by an explicit convention. That is
an anatomical approximation, not a calibrated identity. The saved 32-case
numerical gate passes with mean fingertip discrepancy 6.428 mm and worst tip
17.518 mm; a 10 mm *mean* gate does not bound every finger by 10 mm. Another
40 fresh proposals (32 development, eight reserve) pass the numerical gate.
These checks establish consistent conversion within scope.

The separate frozen [surface acceptance](HUG_MAPPING_ACCEPTANCE.md) rule fails
same-part proximity/orientation preservation in six detailed development cases.
It is not superseded by a green numerical gate. Native physical-contact and
failure-label equivalence remain unproven. Current training labels describe
the mapped Shadow hand under its named controller, not the original MANO hand.

The simulator is a project task environment built on MuJoCo 3.3.7, not a newly
written physics engine. It compiles the official hand directly from a live
MjSpec, retaining fixed-link inertias and distinct collision assets without a
lossy XML round trip. Object imports preserve native mesh frames, scale, centre
of mass and inertia tensor; mass interventions scale inertia consistently.
Stationary captures, rendered-camera checks and source/input hashes bind
perception to the nominal object geometry/pose. Declared mass, friction and
table/pose interventions are recorded separately before each execution. These
are simulator camera-consistency checks, not real-camera calibration evidence.

The controller uses force-limited position servos and an ideal mocap/weld arm
carrier. It has a 2 ms integration step and 40 ms recording/control interval.
This isolates grasp execution and provides exact timing, but does not represent
a calibrated physical arm. Servo gains, friction/contact parameters and timestep
accuracy still need dedicated physical/sensitivity validation. The 213 completed
study tasks pass every-substep finite-state/warning/time guards; this demonstrates
bounded numerical stability in that population, not real-world fidelity.

The milk translation diagnostic exposes a static distance-query sensitivity:
identical relative geometry changes one sampled distance after a 2 cm global
translation. Four conservative planner refusals are retained. Consequently we
cannot certify translation invariance, continuous swept clearance or universal
reachability. Timestep convergence, calibrated friction/compliance, real-hand
tracking and native-versus-mapped physical outcomes remain separate open gates.

## Phase 1

Saved-input automation **PASS_WITH_CAVEATS** at `d6601d1`: all32 current static
gates pass,96static forwards,zero integration attempts. The report distinguishes
recomputed checks from hash-bound historical scene/contact/tracking evidence.
Every fresh proposal requires the same licensed reconstruction/calibration and
static acceptance API plus raw optimizer/status/clipping provenance. Contact
distortion remains an approximation, not a failed numerical conversion to hide.

## Phase 2 implementation

- Generic proposal execution has an explicit `hug_prediction_recording_v1`
  opt-in, with null outcomes for refusals/setup/runtime errors. Historical default
  behavior and recordings remain separate.
- `shadow_physics_filter_v1` branches six independent gravity trials from one
  pre-lift closed snapshot. It reports contact persistence, displacement and
  penetration codes. Both accepted and rejected proposals remain scheduled.
- `hug_grasp_approach_v1` is a new, named controller condition. It preserves the
  intended palm/finger grasp while selecting a wrist posture and an objectward
  palm-normal/finger-direction approach using sampled clearance checks. v1/v2
  remain unchanged; finite samples do not certify a continuous collision sweep.
- The seeded sampler serializes geometry/category/capture/proposal identities,
  mass, friction, approach, table height and qpos sensor-noise conditions. Random
  quotas require reporting actual axis coverage; they do not guarantee balance.
- Category folds are frozen from identities before outcomes. One geometry can
  never cross categories, and one proposal/capture family stays in one fold.

The fresh pipeline has now run under separately committed stage declarations.
All 32 new HUG proposals pass the static mapping gate across eight calibrated
captures. The pilot completes 16 physical tasks and retains four null planner
refusals. Independent review matches all 16 exported causal prefixes exactly to
their raw observables and verifies source, native, scenario and outcome bindings.

The frozen 256-slot study completes with 213 physical labels (126 successes,
87 failures) and 43 null planner refusals, without runtime errors. Its causal
dataset has 213 rows and 1,440 features per row, from 28 proposal families and
four geometries. These diversity counts must accompany the attempt count.
The nominal six-axis screen accepts 11 proposals, rejects 16 and cannot run on
five unreachable approaches; its status never supplies a task label or feature.

The suggested 2,000-attempt/six-object physical collection target is not met.
The core declared study has 256 slots over four objects; two additional reserve
objects yield no executable tasks. This is an engineering cohort with explicit
coverage limits, rather than a completed large training dataset.

The separate eight-pair mug comparison measures four collisions and no successes
in eight old-controller executions. The new controller executes three grasps
without approach collision, succeeds once, and refuses five. Its planned success
rate is 1/8; conditional physical success is 1/3. Wrist posture and direction
change together. This is limited engineering evidence, not universal acquisition.

Bell pepper and coffee cup each pass two stationary, calibrated capture gates.
Bear figure fails both unchanged stationary gates and stays excluded. After the
60 models were frozen, eight fresh bell/coffee grasps passed numerical mapping;
all 32 reserve task slots were refused by the static approach policy. Their
outcomes remain null and all 60 predictor evaluations are explicitly skipped
with `NO_ELIGIBLE_PREFIXES`. Prospective coverage/generalization is unestablished.
Pinned geometry manifests retain attribution, source URLs and byte hashes.

## Phase 3 and Phase 4

Failure-phase analysis and the strict recorded-dataset model adapter are
implemented and tested. Their fixed analysis/model rules were committed before
fresh task labels. The actual Phase 3 audit passes all four frozen category
folds. All 60 model/fold/seed slots complete under the fixed comparison; summary
logistic has mean held-object failure AUROC 0.594, with individual folds
0.255–0.811. This does not establish reliable transfer. The comparison verifies
a structured frozen declaration binding its exact
specification, sources, dataset, scenario schedule, fold membership and attempt
bindings. Class-degenerate folds and failed collection gates remain explicit.

A later read-only audit reproduces the four original saved-weight AUROCs and
archived 3,120-attempt historical category metrics without pooling controllers or
refitting. A fixed four-cutoff replay preserves original held-test identities:
the summary model's mean AUROC is 0.594 at 2.96 s and 0.500 at
2.76/2.56/2.36 s. These shifted-time diagnostics demonstrate no useful earlier
ranking in this comparison; they do not measure slip onset.

The separate failure-family audit completes its four declared summary-logistic
candidate fits for the eligible no-contact fold. Missing causes and the
single-class acquisition training role remain explicit skips. Whole-family
bootstrap intervals use 1,000 draws separately per existing category/model/seed;
they are conditional on the observed families, not confidence intervals over
unseen objects, captures, hardware or simulator configurations.

`docs/RESULTS_LOG.md` is the append-only measured result ledger. Every run needs
a finite quota, stop conditions, source/config/input hashes and a new output
directory. No historical dataset, controller, scorer or result is rewritten.
