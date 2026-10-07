# Portable Evaluation Manifest

Every evaluation case must identify the reusable simulated world through a
versioned `environment_id`. The environment implementation creates and runs the
world; the ID tells an evaluation platform which registered implementation and
contract to load.

The initial registry entry is:

```text
adroit_shadow_tabletop_v1
    → HUG/Shadow-Hand MuJoCo runner
```

## Case record

```json
{
  "case_id": "object01_grasp003_mass018_seed9000",
  "environment_id": "adroit_shadow_tabletop_v1",
  "embodiment_id": "shadow_hand_right",
  "execution_protocol_id": "fixed_grasp_lift_v1",
  "object_id": "object01",
  "hug_grasp_id": "grasp_003",
  "mass_kg": 0.18,
  "friction_profile": "nominal_v1",
  "initial_condition_id": "pose_001",
  "motion_profile": "nominal_lift_v1",
  "seed": 9000
}
```

`case_id` must be unique within a manifest. All referenced IDs must resolve to
versioned records or assets; a runner must not infer an asset from a display
name.

## Execution protocol registry

The execution protocol is versioned independently from the environment. The
environment fixes the controller implementation and its input/output contract;
the protocol fixes the deterministic sequence and its resolved parameters.

The initial registry entry is:

```yaml
fixed_grasp_lift_v1:
  pregrasp_distance_m: 0.08
  palm_height_above_object_m: 0.04
  approach_duration_s: 1.0
  grip_force: 0.6
  lift_height_m: 0.15
  lift_duration_s: 1.5
  hold_duration_s: 2.0
```

It executes:

```text
reset
    → load object
    → retarget HUG grasp
    → move to pre-grasp
    → approach
    → close fingers
    → lift
    → hold
    → score success/failure
```

The first experiment uses this hardcoded deterministic protocol. It does not
train or invoke a manipulation policy. This isolates HUG grasp quality from
controller-learning failures. Changes to the sequence, parameter meanings, or
defaults require a new protocol ID.

## Environment contract

An `environment_id` fixes all of the following:

- simulator and scene;
- table geometry;
- arm and hand model;
- controller;
- physics timestep;
- observation and action contracts;
- cameras; and
- success and failure definitions.

Any incompatible change to one of these fields requires a new environment
version, such as `adroit_shadow_tabletop_v2`. The version is part of the ID; it
must not be supplied as an independent mutable field.

Individual cases may vary:

- object;
- HUG grasp;
- mass;
- starting condition;
- motion profile; and
- seed.

Friction is represented by a versioned profile and remains fixed to
`nominal_v1` for the initial experiment. The schema supports other profiles for
later studies.

## Reproducibility record

The resolved run record must add:

```json
{
  "environment_config_hash": "sha256:...",
  "execution_protocol_id": "fixed_grasp_lift_v1",
  "execution_protocol_config_hash": "sha256:...",
  "execution_protocol_parameters": {
    "pregrasp_distance_m": 0.08,
    "palm_height_above_object_m": 0.04,
    "approach_duration_s": 1.0,
    "grip_force": 0.6,
    "lift_height_m": 0.15,
    "lift_duration_s": 1.5,
    "hold_duration_s": 2.0
  },
  "mujoco_version": "3.3.7",
  "code_commit": "..."
}
```

`environment_config_hash` is the SHA-256 digest of the canonical resolved
environment configuration, including the registered scene, model, controller,
timestep, camera, and contract definitions. `code_commit` records the exact
repository commit used by the runner. These values describe the execution and
must be stored with its trajectory and outcome.

`execution_protocol_config_hash` is the SHA-256 digest of the canonical
resolved protocol entry. The ID, full resolved parameters, and hash are stored
with both the submitted case and result so a later change cannot alter the
meaning of an existing run.

## Registry and validation behavior

The evaluation platform maintains an explicit registry from `environment_id`
to runner implementation and expected configuration hash. Before simulation,
it must:

1. reject an unknown `environment_id`;
2. reject an unknown `execution_protocol_id`;
3. resolve both registered implementations and canonical configurations;
4. recompute and compare both configuration hashes;
5. validate all referenced object, grasp, starting-condition, friction, motion,
   and embodiment IDs; and
6. reject duplicate `case_id` values and invalid numeric ranges.

An unknown environment or protocol ID, or either configuration-hash mismatch,
is a hard validation failure. The platform must never substitute a default or
silently run a different setup.

The portable manifest contains case inputs. The resolved run record adds
software provenance, the resolved hash, timestamps, outcome, failure type, and
trajectory artifact references.

## One-case command contract

The `grasp` section supports both the normal inference path and a saved-proposal
debug path. `observation_path` identifies the validated RGB-D capture and
`inference_seed` controls HUG generation. `prediction_path` is optional and is
used only when inference is explicitly disabled.

```yaml
grasp:
  source: hug
  id: grasp_003
  observation_path: observations/object01
  inference_seed: 42
```

Normal evaluation uses `--hug-root ... --checkpoint ...` and runs inference by
default. `--no-inference` skips model loading and executes `prediction_path`
instead.

The executable interface is:

```bash
run-hug-eval-case \
  eval_cases/hug_case_001/case.yaml \
  --output runs/hug_case_001 \
  --hug-root ../hug \
  --checkpoint ../hug/checkpoints/hug_full.safetensors \
  --viewer
```

Before opening the viewer or advancing simulation, this command must resolve
and validate both `environment_id` and `execution_protocol_id`, verify both
configuration hashes, and write the resolved environment and protocol records
to the output directory.
