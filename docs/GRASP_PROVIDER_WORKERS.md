# External grasp-provider worker contract

`generate-grasp-proposals` supports isolated model environments without adding
their CUDA or framework dependencies to the MuJoCo environment. Provider IDs
and immutable configuration are registered in
`src/grasp_failure_prediction/proposals/configs/providers.yaml`.

## Command worker

The command receives absolute YAML-or-JSON request and output paths:

```text
WORKER --request /absolute/request.json --output /absolute/output
```

It must write exactly `sample_count` artifacts at
`sample_0000/proposal.json`, `sample_0001/proposal.json`, and so on. Each file
must validate as `grasp_proposal_v1` and must bind the requested provider ID,
provider configuration hash, object ID, geometry hash and embodiment. The core
command validates all returned artifacts before writing `manifest.json`.

Generic providers name every model input explicitly. For example:

```yaml
schema_version: proposal_request_v1
request_id: mug_dga_seed42
provider: {id: dexgrasp_anything_shadow_v1}
object: {id: mug_001, geometry_content_hash: "sha256:..."}
embodiment_id: shadow_hand_right
provider_inputs:
  - {role: object_mesh, path: objects/mug_001/model.obj}
  - {role: object_point_cloud, path: observations/mug_001/points.npy}
seed: 42
sample_count: 8
```

Generation fails before starting a worker when a registered required role is
missing.

## HTTP worker

The client sends one `POST` with JSON:

```json
{
  "request": {"schema_version": "proposal_request_v1"},
  "provider_spec": {"schema_version": "proposal_provider_spec_v1"},
  "provider_config_hash": "sha256:...",
  "expected_input_binding": {
    "geometry_group_id": "sha256:...",
    "observation_group_id": "sha256:...",
    "generation_request_group_id": "sha256:...",
    "input_manifest_hash": "sha256:...",
    "scene_contract_hash": null
  },
  "inputs": [
    {
      "role": "observation",
      "path": "rgb.png",
      "sha256": "sha256:...",
      "content_base64": "..."
    }
  ]
}
```

Input entries contain the files referenced by `observation_path`,
`provider_input_path`, and `provider_inputs`; the worker verifies every hash
before inference and copies `expected_input_binding` into its proposal. A bearer
token can be read from an environment variable with `--bearer-token-env`, so no
credential enters a manifest or Git history.

The endpoint responds with:

```json
{
  "proposals": [
    {"schema_version": "grasp_proposal_v1"}
  ]
}
```

The response count must equal `sample_count`. Provider-native assets may remain
on the worker. If the response lists native artifacts, portable execution does
not require them; `validate-grasp-proposal --verify-native-artifacts` enables a
deep audit when they are copied beside the canonical JSON.

## Execution boundary

The worker does not run MuJoCo and cannot select environment physics, controller
settings, scoring thresholds or result labels. It returns only an
object-relative palm transform, exact named Shadow joint targets, provenance,
input grouping and declared fixed-approach compatibility. `eval_case_v2` binds
the proposal content hash and the core runner resolves the environment and
execution protocol independently.
