# Noisy rollout collection: a reproducible first experiment

This step replays saved HUG grasp proposals with the existing fixed, open-loop
controller and records what happens when finger targets are disturbed. It
collects data for a future failure predictor; it does not train one. The
controller does not change its plan in response to the disturbance or outcome.
Its low-level, force-limited joint servos still respond to physical contact.

The draft collector requires the partner branch's actuated Shadow Hand runner,
`fixed_grasp_lift_v2` protocol, retargeting fixes, and evaluation artifacts. Those
dependencies and this collector must be integrated before the command is
available from `main`. Use the checkout containing the draft implementation.

## What one attempt does

1. Load a locally generated, trusted HUG proposal and its camera-to-world
   transform. Retarget the proposal to the right Shadow Hand and check the
   baseline fingertip alignment gate.
2. Run the same approach, close, lift, and hold sequence used by the existing
   actuated runner.
3. Add one bounded, per-episode bias to finger targets during `CLOSE_FINGERS`,
   `LIFT`, `HOLD`, and `SCORE`. Keep the wrist targets and earlier phases unchanged.
4. Save commands, measured motion, simulation diagnostics, final outcome, and
   conservative event annotations. Record execution errors separately.

The current scene contains the existing cube. This step does not introduce new
objects, varied friction, mass augmentation, a recovery controller, or a trained
prediction model.

## What the noise means

For each attempt, a local random generator draws one value between -1 and 1 for
each hand joint. Multiplying it by the selected amplitude gives the requested
angle offset. The 22 finger offsets remain fixed throughout the noisy phases;
the two wrist offsets are set to zero. This is `uniform_episode_bias`, one noise
type at different strengths, rather than a new random jitter at every frame.

The amplitude is a bound for each finger target, in radians:

- `0.05` rad means up to about 2.9 degrees in either direction.
- `0.15` rad means up to about 8.6 degrees in either direction.
- `0.6` rad means up to about 34.4 degrees in either direction. This is a stress
  setting, not an estimate of ordinary hardware noise.

For example, a nominal 42-degree target with a requested -8-degree offset becomes
a 34-degree commanded target if the joint limits permit it. If the sum exceeds
a joint limit, clipping reduces the offset. `requested_noise_rad` records what
we asked for; `applied_noise_rad` records `commanded_target_rad` minus
`nominal_target_rad` after clipping. Actual measured joint angles can differ
from both targets because the servos have finite force and encounter contact.
The disturbance is to the command, not a physical obstacle inserted into the
scene. A noisy attempt may still succeed.

The random direction is paired across amplitudes for the same case and
repetition. Changing the amplitude scales the same draw, making those attempts
easier to compare. The draw is derived from the master seed, case index, and
repetition; reordering the manifest changes the draw assignment. `plan.json`
freezes the schedule, seeds, and source hashes before execution. With zero
amplitude, the original commands are preserved exactly. Repeated zero-noise
attempts are identical under this deterministic controller, so they are not
independent new examples.

## Final labels and event annotations

`1` means the completed attempt meets the registered success rule; `0` means a
completed physical attempt does not. For `fixed_grasp_lift_v2`, success requires:

- Hand-object contact during closing, lifting, or holding.
- A peak lift of at least 14 cm relative to the first recorded object height.
- At least 2 seconds of consecutive `HOLD` observations at least 12 cm above
  that reference height: the 14 cm lift threshold minus a 2 cm allowance.
- No more than 2 cm of height loss from the episode's peak to its final height.
- No hand-table collision during pregrasp or approach and no protocol timeout.

These are the existing scorer's criteria. Its `acquired_object` field means
contact was observed; contact alone does not prove a secured grasp. The failure
category is saved alongside the binary label, including `no_contact`,
`failed_acquisition`, `drop`, or `slip_during_hold`. A category is the scorer's
classification, not proof of a specific contact mechanism.

Validation errors, numerical instability, and artifact-writing or runtime errors
have `label: null` and a separate error status. They must not become ordinary
failure examples labeled `0`.

`events.json` uses the separately versioned `rollout_events_v1` annotation
schema. It does not change the final scorer or binary label:

- `lifted_contact_proxy` requires the minimum lift plus hand-object contact and
  no object-table contact during `LIFT` or `HOLD`.
- `retained_lift_proxy` additionally requires consecutive qualifying `HOLD`
  samples totaling the required duration. It describes sampled retention, not
  continuous stability between samples.
- `acquisition_deadline` records the observed control interval from the last
  `LIFT` sample to the first `HOLD` sample. Failing to acquire by that deadline
  is different from a physical slip beginning at that instant.
- `observed_height_loss` records a sampled crossing below the stable-height
  threshold after the lift proxy, bounded by observed timestamps. Its physical
  mechanism remains unknown. A transient loss can occur in an eventually
  successful attempt and does not overwrite its binary label.

The collector records control-interval endpoints, normally 40 ms apart. A
sampled crossing is not an exact slip-onset time. Missing samples, incomplete
follow-up, and unsupported failure categories remain explicit rather than
being treated as confirmed event-free motion. No future-horizon training labels
or trajectory windows are created in this step.

## What we save and where

The caller chooses the output directory. Use a new or empty directory; the
collector refuses to overwrite a nonempty collection. Raw runs stay outside
versioned source files under the repository's ignored `runs/` directory.

At collection level:

- `plan.json`: frozen cases, amplitudes, repetitions, seeds, source groups, and
  provenance hashes.
- `index.jsonl`: one completed or error record per planned attempt.
- `summary.json`: running counts of completed successes, failures, and errors.

Each completed `episode_XXXXXX/` bundle contains:

- `actions.npz`: nominal, previous, and commanded joint targets; requested and
  applied noise; command-interval timestamps; joint limits; measured velocities
  and actuator forces; and simulation states for replay.
- `trajectory.npz`: phase, measured joint angles, palm and object
  positions, and simulator contact diagnostics.
- `events.json`: conservative sampled event proxies and deadline annotations.
- `result.json` and `resolved_case.json`: final outcome, metrics, and the exact
  resolved environment/protocol settings.
- `collection_metadata.json`: collection settings, input references, grouping,
  source hashes, effective runtime versions, and data-use limits. Pinocchio's
  loaded version and installed `pin` distribution metadata are recorded separately
  because a dependency directory selected through `PYTHONPATH` can differ.
- `scene.mjb`, `scene.xml`, and `final_state.npz`: scene and final state. The
  binary scene plus saved `mjSTATE_INTEGRATION` snapshots supports faithful
  state replay; the XML aids inspection.

Commands act over `[action_interval_start_s, time_s]`. The underlying controller
interpolates from the previous command to the new target at physics substeps;
the accompanying state is measured at the interval's end. These command/state
logs make the applied noise inspectable. The collector does not automatically
render a video; saved state supports a separate replay/viewing step.

Joint/object positions and velocities use interval-end timestamps. MuJoCo
evaluates body-position, contact, and force diagnostics before the final physics
integration substep, so those diagnostics are normally 2 ms earlier. The offset
is recorded in `diagnostic_time_s` and collection/event timing metadata. Lift
proxies combine endpoint height with those earlier contact diagnostics; they
do not establish perfectly synchronized contact sensing.

Only the initial RGB-D observation is referenced. There is no camera observation
at every rollout frame. Simulator object state, contacts, and full integration
state are useful supervision and replay diagnostics; they are not automatically
available robot sensors.

A future predictor loader must explicitly whitelist its observable inputs. The
initial baseline can consider measured hand angles/velocities and command
history available up to the prediction time. Final outcomes, event annotations,
future samples, noise seeds/amplitudes, hidden object conditions, and full
simulator state must not enter that predictor. Any extra sensing needs its own
availability justification.

## Grouping before any train/validation/test split

`group_id` identifies a source proposal family from content hashes. Keep all
amplitude and repetition variants of that proposal together in any later split.
`observation_group_id` identifies the shared object/observation/camera source
without the particular proposal. It supports the stricter requirement of keeping
multiple proposals from the same observation together.

All ten current cube proposals share one initial observation and one object.
They are ten proposal families, not ten independent objects or scenes. This
pilot can verify collection and labels; it cannot establish unseen-object or
unseen-observation predictor generalization. No data splits are generated here.

## Portable input and command example

Run from the source checkout with its evaluation dependencies and the pinned
hand assets initialized:

```sh
python -m pip install -e ".[eval,dev]"
git submodule update --init third_party/dex-urdf
```

Place a locally generated trusted proposal at `runs/example/source/proposal.pkl`
and its corresponding `T_world_camera.npy` beside it. Keep the referenced
initial RGB-D inputs in `runs/example/observation/`. The collector replays saved
proposals; it does not run HUG inference or supply checkpoints. See
[the simulation pilot](HUG_SIMULATION_PILOT.md) for the inference workflow.

Create `runs/example/cases/cube_seed0.yaml`:

```yaml
schema_version: eval_case_v1
case_id: cube_seed0
environment:
  id: adroit_shadow_tabletop_v2
embodiment:
  id: shadow_hand_right
execution_protocol:
  id: fixed_grasp_lift_v2
object:
  id: object01
  mass_kg: 0.18
  friction_profile_id: nominal_v1
grasp:
  source: hug
  id: hug_seed0
  prediction_path: runs/example/source/proposal.pkl
  observation_path: runs/example/observation
  inference_seed: 0
initial_condition:
  id: pose001
  object_position_m: [0.0, 0.0, 0.03]
  object_orientation_xyzw: [0.0, 0.0, 0.0, 1.0]
motion_profile_id: nominal_lift_v1
seed: 9000
```

Create `runs/example/cases.json` containing project-relative paths:

```json
["runs/example/cases/cube_seed0.yaml"]
```

Then collect three paired attempts:

```sh
python -m grasp_failure_prediction.evaluation.noisy_rollouts \
  runs/example/cases.json \
  --project-root . \
  --output runs/example/collection \
  --noise-amplitudes 0 0.15 0.6 \
  --repetitions 1 \
  --seed 9000
```

The equivalent installed entry point is `collect-noisy-rollouts`. Three strengths
do not guarantee three different outcomes. Inspect the generated index and
bundles before treating the data as a usable mixture of successes and failures.
Refer to [the evaluation manifest](EVALUATION_MANIFEST.md) for the case schema.

## Fresh verification results

Verified on 10 October 2026 using source commit
`21d0fd4fad81a6bea351bdfe15f0e9ad20dadbe0` and the source-file hashes saved in
`plan.json`. The local manifest is `runs/noisy_pilot_2026-10-09/cases.json`;
the final output is `runs/noisy_pilot_2026-10-10/verification_v2/`.
These raw files remain local and are not included in the PR.

Ten saved proposals were each replayed with amplitudes `0` and `0.6` rad,
one repetition and master seed `9000`:

- Zero noise: 10 successes, 0 failures.
- Strong noise: 3 successes, 7 failures.
- Total: 20 completed attempts, 13 successes, 7 failures, 0 errors.

All seven failures are `failed_acquisition`: they never met the 14 cm lift
threshold. Thirteen attempts have lifted/retained proxies; there are no observed
post-lift height-loss events in this check. This batch does not establish timed
slip-prediction data. It contains ten proposal groups but one observation group
and one cube.

All 20 trajectories and the previously existing action fields match the earlier
pilot bit for bit. This verifies reproducibility with the same proposals and
seeds; it does not add independent research examples. A preliminary run at
`runs/noisy_pilot_2026-10-10/verification/` was superseded because it logged
installed `pin` metadata in place of the loaded Pinocchio version. Use
`verification_v2` for the corrected provenance record.

`verification_report.json` confirms index/result/event-label agreement,
source hashes and commit, command/state timing, diagnostic offset, noise bounds,
phase and wrist exclusions, proposal/observation groups, and binary-scene
restoration of saved final integration states for every attempt.

Effective runtime: Python 3.12.13, NumPy 2.5.3, MuJoCo 3.3.7,
dex-retargeting 0.5.0 and loaded Pinocchio 3.8.0. Installed `pin` distribution
metadata reports 4.1.0; it is recorded separately from the loaded runtime.
The full local suite passed with 143 tests and 5 skips; after the final timing
and version-record changes, the affected collector/event tests also passed.

The next collection milestone is independent scene/object variation with valid
HUG inputs and an audit of the resulting failure types. Future-prediction
windows, dataset splits, and model training follow once that collection has
enough supported successes and failures.
