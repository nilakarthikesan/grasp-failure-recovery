# Portable Evaluation Manifest

Evaluation cases use versioned environment and execution protocol references.
The environment fixes the simulated scene and controller contract; the protocol
fixes the execution sequence and its parameters.

The current one-case command is implemented. Batch scheduling, duplicate case
checks across a collection, and broader asset registries remain extensions.

## Executable case schema

The current input uses nested references, as in
[`eval_cases/hug_case_001/case.yaml`](../eval_cases/hug_case_001/case.yaml):

```yaml
schema_version: eval_case_v1
case_id: object01_grasp003_mass018_seed9000
environment:
  id: adroit_shadow_tabletop_v1
embodiment:
  id: shadow_hand_right
execution_protocol:
  id: fixed_grasp_lift_v1
object:
  id: object01
  mass_kg: 0.18
  friction_profile_id: nominal_v1
grasp:
  source: hug
  id: grasp_003
  prediction_path: grasps/object01/grasp_003.pkl
initial_condition:
  id: pose_001
  object_position_m: [0.0, 0.0, 0.03]
  object_orientation_xyzw: [0.0, 0.0, 0.0, 1.0]
motion_profile_id: nominal_lift_v1
seed: 9000
```

Prediction paths must be repository-relative and may not contain `..`.
The case schema rejects undeclared fields, invalid IDs, nonpositive object
mass, invalid seed ranges, and zero-length orientation quaternions. Use trusted
prediction files because loading a pickle can execute code.

The current runner accepts `object01`, `nominal_v1` friction, and
`nominal_lift_v1` motion. Unknown values are rejected. The initial-condition ID
labels the explicit pose supplied in the case; it is not a separate pose-asset
registry lookup.

## Environment and protocol registries

`adroit_shadow_tabletop_v1` resolves to the HUG/Shadow Hand MuJoCo runner and its
configuration. It fixes the scene, model, controller, physics/control timing,
observation/action contract, cameras, and outcome definitions.

`fixed_grasp_lift_v1` defines this sequence:

```text
reset → load object → retarget → pre-grasp → approach → close → lift → hold → score
```

Its parameters are in
[`execution_protocols.yaml`](../src/grasp_failure_prediction/evaluation/configs/execution_protocols.yaml).
The current settings include:

- 0.08 m pre-grasp distance and 0.04 m palm height above the object.
- 1.0 s approach and 1.0 s closure.
- A normalized position command of 0.6.
- A commanded 0.15 m lift over 1.5 s and a 2.0 s hold.
- A 10.0 s timeout.
- A 0.14 m minimum measured lift, 2.0 s required hold, and 0.02 m maximum drop.

The grip command interpolates hand joint positions toward the retargeted pose.
It is not a calibrated force command. Wrist placement is table-parallel and
defined relative to the object. The registered runner prescribes joint and
wrist positions during physics stepping rather than modeling an actuator-limited
robot. Its outcome depends on the complete retargeting and execution method,
not only HUG's source prediction.

Incompatible changes to an environment or protocol require a new versioned ID.
Results from different controllers or protocol meanings must remain distinguishable.

## Configuration hashes and provenance

Environment and execution protocol configurations are resolved and hashed
before execution. Each reference may include an optional
`expected_config_hash: "sha256:..."` to pin the expected configuration. A
supplied mismatch is a validation error. An omitted expected hash permits the
registered configuration to be resolved; the resulting hash is still recorded.

Each resolved result includes:

- environment, embodiment, and execution protocol IDs;
- environment and protocol configuration hashes;
- resolved protocol parameters;
- MuJoCo version and code commit;
- retargeting metrics, outcome measurements, and artifact references.

A batch manifest should additionally enforce unique `case_id` values and
freeze its object, grasp, initial-condition, friction, and motion definitions.
The current interface executes one case at a time.

## Run one case

Initialize the `dex-urdf` submodule, install the evaluation extra, and supply
the trusted prediction referenced by the case. Use a new or empty directory:

```bash
run-hug-eval-case \
  eval_cases/hug_case_001/case.yaml \
  --output runs/hug_case_001
```

For live viewing on macOS:

```bash
MUJOCO_GL=glfw mjpython -m grasp_failure_prediction.evaluation.case_runner \
  eval_cases/hug_case_001/case.yaml \
  --output runs/hug_case_001_viewer \
  --viewer
```

The command validates the case and registries before advancing simulation.
After execution it writes `resolved_case.json`, `trajectory.npz`,
`final_state.npz`, and `result.json`. The final state contains `qpos` and
`qvel`; it does not capture all physics integration, environment, controller,
or RNG state needed for complete intervention replay.

The scorer's `acquired_object` flag records any hand-object contact during
manipulation. Lift, hold, final height, collision, and timeout checks determine
task success. A height shortfall labeled `failed_acquisition` is not necessarily
a drop. These coarse outcome labels need separate validation before use as
future-drop prediction targets.
