# Grasp evaluation command

`grasp-eval` is the user-facing command for a reproducible grasp experiment.
The experiment file declares proposal requests and their physical evaluation
conditions. A normal run no longer requires users to coordinate separate
proposal, case, batch, and viewer commands.

## Experiment definition

```yaml
schema_version: grasp_eval_experiment_v1
experiment_id: hug_cube_pilot_v1
output_root: runs/hug_cube_pilot_v1
record_video: true

jobs:
  - id: cube_hug
    request: requests/hug_cube.yaml
    evaluations:
      - case_id: cube_hug_000_nominal
        proposal_sample_index: 0
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
        initial_condition:
          id: centered_upright
          object_position_m: [0.0, 0.0, 0.03]
          object_orientation_xyzw: [0.0, 0.0, 0.0, 1.0]
        motion_profile_id: nominal_lift_v1
        seed: 9000
```

Paths are relative to the project root. Every case selects one sample from its
job's `proposal_request_v1`. Multiple cases may reference the same sample to
measure robustness under controlled mass, pose, or seed changes. Different
jobs may select different registered providers while keeping the environment,
controller, and scorer fixed.

An HTTP or process-backed provider is selected by adding these fields to an
otherwise complete job:

```yaml
provider_id: dexgrasp_anything_shadow_v1
endpoint: https://worker.example/v1/grasp-proposals
bearer_token_env: GRASP_WORKER_TOKEN
```

Secrets are read from the named environment variable and must not be written
to the experiment file.

## Commands

Validate configuration, registry compatibility, sample indexes, objects, and
embodiments without running inference or MuJoCo:

```bash
grasp-eval validate experiments/pilot.yaml
```

Run local HUG inference and every declared case:

```bash
grasp-eval run experiments/pilot.yaml \
  --hug-root ../hug \
  --checkpoint ../hug/checkpoints/hug_full.safetensors
```

Check progress or continue an interrupted run:

```bash
grasp-eval status experiments/pilot.yaml
grasp-eval resume experiments/pilot.yaml \
  --hug-root ../hug \
  --checkpoint ../hug/checkpoints/hug_full.safetensors
```

Resume skips cases with completed `result.json` artifacts. An incomplete
proposal or case directory is preserved with an `.incomplete-NN` suffix before
that stage is retried.

Open the experiment's local results viewer:

```bash
grasp-eval view experiments/pilot.yaml
```

Use `--output PATH` to override `output_root`, and `--no-video` for a quick
non-recording run.

## Output bundle

```text
runs/hug_cube_pilot_v1/
├── resolved_experiment.json
├── experiment_result.json
├── proposals/
│   └── cube_hug/
│       ├── manifest.json
│       └── sample_0000/proposal.json
├── cases/
│   └── cube_hug_000_nominal.json
└── runs/
    └── cube_hug_000_nominal/
        ├── result.json
        ├── resolved_case.json
        ├── proposal.json
        ├── trajectory.npz
        ├── final_state.npz
        └── rollout.mp4
```

The resolved experiment, proposal manifest, generated cases, results, and
content hashes make the run auditable. Large-scale Parquet/DuckDB indexing is a
separate dataset-export step; this command does not introduce a database or
change the existing simulation protocol.
