# Native HUG / MANOSim prediction plan

Prepared 2026-10-16. System implementation only; no manuscript work.

## Measured status on 2026-10-16

The compensated native component profile passes 12 fixtures and six timestep
comparisons with independent saved-data verification. The two 135-slot object
pilot versions produce zero physical labels or eligible training prefixes.
V2 fixes the exact upstream camera-K cast check (135/135 pass); all 60 captures
still fail the unchanged depth/visual-mesh gate. Seven grouped catalog holdouts
and five distinct trainer folds are verified but have no admitted examples.
No scientific predictor has been fitted. See `docs/RESULTS_LOG.md` for frozen
protocols, independent receipt hashes, actual operation counts and limitations.

## Prediction question and admission

The first model estimates whether a grasp will fail after a fixed observation
cutoff before lift, conditional on a valid native execution reaching that cutoff.
It does not estimate HUG's original quality score or general hardware reliability.
The initial native domain is the authored MANOSim capsule hand with the explicitly
qualified hand-only gravity compensation controller. Old Shadow results and
component fixtures are different domains and cannot supply its task labels.

Software development and manufactured model tests are implemented. Scientific
fitting requires independently verified task recordings, unchanged source/data
identities, numerical-readiness results, and both outcome classes within each
train, validation and test role. An empty or single-class role is reported as
skipped. A larger number of calls does not solve invalid labels or missing classes.

The collection exporter intentionally has no task qualifications attached.
The training loader refuses scientific fitting until an independent native-task
qualification binds its exact hand, collider, controller, recording and label
contracts. A component-fixture PASS cannot substitute for this task qualification.

## Why these three precedents define variety

1. [HUG-Bench](https://arxiv.org/html/2606.17054v1#S5.SS1) provides a reproducible
   object coverage grid: cylindrical, spheroidal, prismatic, appendaged and
   amorphous shapes across the authors' small, medium and large bins. We use all
   30 official test objects, two in each cell, rather than infer a class from a
   filename or select objects after seeing outcomes. Two fixed original views
   and generator seeds 42/43 give 120 primary proposals. Actual wrist and finger
   differences are measured separately; changing a seed is not evidence of a
   meaningfully different grasp by itself.
2. [UniDexGrasp](https://arxiv.org/html/2303.00938#S4.SS1) distinguishes unfamiliar
   instances within familiar categories from unfamiliar categories, and measures
   pose diversity separately from grasp quality. This motivates separate whole
   object and whole category tests. Every capture, generator seed, repeat and
   scenario of an object stays in one role within a fold. HUG's objects being
   unseen to HUG does not make them unseen to our predictor; our own grouping is
   still required. Shared source scenes limit claims about scene generalization.
3. [Dex-Net 2.0](https://arxiv.org/html/1703.09312) explicitly models physical and
   observation uncertainty, including friction, pose and depth corruption. This
   motivates later controlled physics/sensing campaigns. Its analytic robustness
   labels are not substituted for our executed lift/hold outcomes. We also do
   not copy an image-wise split for an unseen-object prediction question.

The catalog freezes two reciprocal object holdouts and five category holdouts.
The existing trainer implements its own five whole-category folds with a whole
validation category. These are separate protocol families, with separate manifests;
the seven catalog folds do not imply seven fitted models or trainer compatibility.
The pilot reports counts and class support in all predeclared roles before fits.

## Staged collection

The nominal pilot fixes physics, command noise and sensor noise. It adds 15
matched 1 ms re-evaluations to the 120 primary 2 ms attempts, one per geometry/size
cell. These matched attempts diagnose numerical sensitivity and never become
additional independent training samples. Original scene meshes, mass, inertia,
camera transforms, proposal outputs and controller clocks remain attributable.

The pilot models one target object and a table. Other objects visible in source
images are not simulated; it cannot establish clutter generalization.
Source-coordinate, camera/mesh, actuator-chart and actual compiled-hand gates
precede integration. Fixed support stability, approach tracking, chart and
penetration guards quarantine invalid attempts with null outcomes. Runtime
errors and mapping refusals remain in coverage denominators.

After the pilot, use its measured gaps to choose the next independently frozen
campaign. The proposed axes are object geometry, authored size, generated hand
pose/view, approach trajectory, mass and effective friction, followed separately
by observation/command noise. They are not yet qualified ranges. Preserve object
mass/inertia consistency and inspect actual combined contact friction: changing
an object coefficient alone may not change hand-object friction under MuJoCo's
parameter-combination rules. Do not broaden ranges or numerical tolerances merely
to recover a desired success rate. Keep the original nominal campaign separate.

Before scaling, inspect usable counts by object, geometry/size cell and failure
phase, plus matched timestep agreement. More captures of one object increase
correlation, not the number of independent objects. If a category lacks valid
successes or future failures, collect justified additional scenarios within the
prospectively assigned roles. Never rebalance by moving held-out objects into train.

## Recording, labels and model comparison

The collector preserves raw native qpos, qvel, requested/effective commands,
contract hashes and separate clocks. Physics diagnostics supply supervision;
they are not predictor features. The fixed task approaches, settles, closes,
lifts 15 cm, and holds for 3 s. Success requires at least 10 cm COM rise at lift
end, at least 8 cm throughout hold, relative object shift at most 2 cm, and no
disallowed hand/table contact or object/table support during hold. These are
engineering task definitions, not thresholds inferred from outcomes. Failure
reports retain earliest cause, phase, time and later diagnostic events.

The implemented feature window is 20 control endpoints ending at 1.49 s, one
10 ms interval before the prospective 1.5 s lift. Each row has 169 channels:
67 native pose coordinates, 51 velocities and 51 effective commands. Quaternion
hemisphere sign is canonicalized without normalizing away invalid raw data.
The window contains no future state, contact oracle, object outcome, identifiers
or timestamps as features. Failures already observed by the cutoff are excluded
from this future-failure task and reported separately.

Compare prevalence, regularized logistic regression, a small MLP and a small GRU
using identical admitted examples and roles. Normalize using training data only;
choose hyperparameters using validation only. The implemented bounded comparison
uses three fixed training seeds and at most 50 neural epochs. Assess failure
AUROC, failure average precision, log loss and calibration alongside class counts
and exclusions. Training seeds are not independent object replications, and
accuracy alone is inadequate under imbalance. No scientific accuracy result is
available from the manufactured software tests.

Independent saved-data readback precedes any claimed task qualification or
scientific fitting. Results and changes are appended to `docs/RESULTS_LOG.md`
and `docs/RESEARCH_DECISIONS.md`; failed protocol versions remain distinct.
