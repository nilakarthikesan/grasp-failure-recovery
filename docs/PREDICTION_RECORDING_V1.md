# HUG prediction recording v1

Implemented 15 October 2026 as an opt-in foundation. Historical
`noisy_rollout_pilot_v3`, its exported dataset, controller and model CLI remain
unchanged. Static validation is not a completed dynamic collection pilot.

`evaluation/prediction_recording.py` defines `hug_prediction_recording_v1`.
Every attempt has a mapping decision, an execution status and an optional
physical outcome. A mapping refusal must be unexecuted with `outcome: null`;
runtime errors also retain null outcomes. They remain in attempted/executed
denominators and never become failure examples. Completed records bind code,
configuration, inputs, model, assets, native pose, mapped pose and label rule.

`RecordedActuatedShadowRunner` in `evaluation/recorded_runner.py` is an explicit
subclass of the fixed actuated runner. It changes recording only. Each completed
control interval records times, phase, named measured positions/velocities,
requested targets and effective clipped final servo targets. These are interval
endpoint commands; interpolated physics-substep commands are not recorded. The
separate diagnostic artifact contains full state and contact values. Full joint
state uses endpoint time; contact diagnostics carry their own time at t−2 ms.
Diagnostics never enter the feature allowlist.

After an authorized run, construct `AttemptMetadata` with a `MappingDecision`
and `PhysicalOutcome`, then call
`runner.prediction_recorder.write(fresh_directory, metadata)`. Refusals use
`write_unexecuted_attempt`; runtime errors, including before the first sample,
use `write_incomplete_attempt` or the recorder's error write. Existing files are
never overwritten. The generic `grasp-eval` runner selection is not silently
replaced by this subclass.

## Export

Create an explicit `attempts.json` list of attempt directories relative to that
manifest and a versioned prefix declaration, for example:

```json
{"protocol_id":"declared_close_prefix_v1","prediction_time_s":2.96,
 "history_steps":20,"control_timestep_s":0.04}
```

Then use:

```sh
python -m grasp_failure_prediction.prediction.recorded_dataset \
  --attempts-manifest /path/to/attempts.json \
  --prefix-protocol /path/to/prefix.json --output /path/to/NEW_DATASET
```

The output contains `dataset.npz` and `manifest.json`; use
`load_recorded_dataset` to validate/read it. The matrix retains the existing
time × [position, velocity, applied target] × joint ordering accepted by
`summarize_prefix`. Hand/controller/label rules cannot mix. Requested targets,
mapping/filter metadata, diagnostics, contacts, object truth and identities are
excluded from features. Known pre-cutoff failures and approach collisions are
explicit exclusions from the new forecasting risk set. Null/refused/error
attempts remain visible in the manifest. Splits remain unassigned until a
separate grouped protocol is frozen; old scene IDs are not reused.

Forty-four new manufactured/stubbed controls pass; with 28 unchanged historical
dataset tests, 72 pass with integration APIs blocked. Tests cover clipping,
alignment, missing fields, timing, causal cutoff, feature leakage, null/refusal
labels, zero-sample runtime errors, mixed protocols, group bindings, hashes and
denominators. No live execution, 20-attempt round-trip, dynamics tracking or
collection acceptance is claimed.
