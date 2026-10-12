# Can the original wrist settings fit the corrected mug grasps?

12 October 2026. This is a **static geometry check**, before executing a paired
wrist experiment. It generates no new grasp attempts, labels or predictions.
The model, controller source and existing recordings remain unchanged.

## The result in simple terms

The original training data contains eight different recorded wrist-command
pairs. We placed each of those eight pairs into each of the eight existing mug
proposals, while compensating the hand's carrier pose to keep the intended
world palm and all 22 finger targets unchanged.

**None of these 64 combinations satisfies the existing 10 mm table-clearance
requirement.** Their nominal approach clearances range from −65.211 mm to
+0.187 mm. Negative clearance means some active hand geometry extends below
the table's top plane. Even the highest candidate leaves only 0.187 mm above
that plane. Seven combinations also fail the 1 mm proximal-hand/object gate.

The eight previously corrected wrist settings, called B in the planned paired
experiment, reproduce their original nominal geometry certificates. They all
pass the same gates, with table clearance from 10.001 mm to 10.042 mm.
Their compensated carrier coordinates and distances agree with the stored v3
plans within an absolute tolerance of 1e−12.

This matters because we cannot simply reuse an actually observed training wrist
setting as the other side of a safe mug comparison. A distinct safe A setting
still needs to be selected, frozen and supported explicitly by the executor.
If A is also outside the training range, the pair tests sensitivity between
two unfamiliar settings. It does not establish a familiar versus unfamiliar
wrist contrast.

## What was held fixed and checked

The script reads the original 520 training rows only and extracts the two last
recorded wrist commands from the documented input layout. These commands were
saved as float32; it evaluates those exact recorded values, converted to
float64 for geometry. It does not fill in or search the continuous rectangle
between them.

For every candidate, it preserves the saved HUG proposal, scene, object mass
and geometry, intended world palm, and all 22 original finger targets. Carrier
compensation preserves nominal world palm and finger geometry both at the
target pose and in the existing near-open approach state. The numerical
invariance gate is 1e−10 for position coordinates and rotation-matrix entries.

The geometry gates match the existing v3 planner: the near-open approach
endpoint must be at least 10 mm above the horizontal table plane and the
proximal hand must be at least 1 mm from the object. The table plane is
infinite and conservative relative to the finite table footprint. These gates
do **not** certify finger closure, actuator tracking, contact forces or a new
dynamic grasp outcome.

No simulator time is advanced and no action history is generated. The script
allows only exported MuJoCo functions needed for model setup, forward
geometry, distances and rigid transforms. Model compilation uses temporary
MJCF files through the existing builder; it does not export a new scene or
recording. All other exported `mj_` and `mju_` routines are blocked during the
audit. The `mj_step` blocker is explicitly exercised. No predictor or training
routine is imported or called by this script.

## What this does and does not establish

The result is **0/64 feasible actually observed training pairs**, and **8/8
nominally feasible stored B poses**. It is not a proof that every possible
angle inside the continuous training range is infeasible. It is also not a
new dynamic safety test, a causal wrist comparison, a model improvement or an
answer to the later equal-budget data-diversity research question.

The current table-safe executor automatically plans a nearby feasible wrist
setting. Its command-line entry point has no frozen A/B manifest input. A
requested unsafe setting could therefore be replaced by the planner. The
paired experiment needs explicit exact-pose selection and rejection if the
executed plan differs from that selection, before any paired rollout.

## Evidence and reproduction

* [report.json](report.json) enumerates all eight training pairs, all 64
  candidate checks, all eight stored B checks, failed gates, carrier
  coordinates, invariance checks, runtime versions and source hashes.
* [manifest.json](manifest.json) hashes this package and its audited sources.
* [analyze_wrist_pose_feasibility.py](../../scripts/analyze_wrist_pose_feasibility.py)
  reproduces the finite audit and refuses to overwrite an output directory.

The report hashes the complete frozen paired protocol and records its
numerical case/model/geometry contract separately. The protocol links this
report by path only, so there is no circular file hash.
Existing model, input data, Python sources, robot assets, object assets and
the consumed registry configuration files and eight raw zero-noise B
recordings are hashed before and after the audit. These are read-only inputs
to this check. Resolved environment and base-controller configurations are
recorded for each case.

With this repository installed and the ignored runtime assets available:

```bash
python -B scripts/analyze_wrist_pose_feasibility.py \
  --project-root /path/to/checkout/with/existing/runs \
  --output /tmp/wrist_pose_feasibility_reproduction
```

For the configured local project runtime, run from the source checkout:

```bash
sim_source='/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction'
sim_assets='/Users/nilakarthikesan/grasp-failure-prediction'
export PYTHONPATH="$sim_source/src:$sim_assets/runs/hug_integration_check/hug-deps:$sim_assets/runs/hug_integration_check/deps:$sim_assets/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages"
export PYTHONDONTWRITEBYTECODE=1
"$sim_assets/.venv/bin/python" -B \
  "$sim_source/scripts/analyze_wrist_pose_feasibility.py" \
  --project-root "$sim_assets" \
  --output /tmp/wrist_pose_feasibility_reproduction
```

Use a new output path for each reproduction. The original full recordings and
dependencies remain in the runtime checkout; this package contains only the
static audit.
