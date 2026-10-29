# Interchangeable grasp-proposal provider plan

> **Current workflow, 28 October 2026:** Use `phase1-scene-preparation` for Phase 1 and `project-work` for ongoing work. Follow [the contributor workflow](WORKING_BRANCH.md); this dated document preserves earlier scope, branch identities and source paths.

**Status:** Core provider boundary implemented; provider-specific GPU integrations remain future work

**Date:** 12 October 2026

**Historical implementation branch:** `project-work`

## Objective

Introduce one explicit boundary between grasp generation and grasp execution so
HUG, saved DexGraspNet records, DexGrasp Anything, UniDexGrasp, and later
providers can feed the same existing Shadow Hand evaluation pipeline.

```text
provider-specific input
        ↓
interchangeable proposal provider
        ↓
versioned canonical Shadow Hand proposal
        ↓
existing MuJoCo validator, controller, scorer, recorder and viewer
        ↓
existing rollout dataset and failure-prediction pipeline
```

The proposal source changes. The registered MuJoCo environment, Shadow Hand,
execution protocol, physics settings, outcome definition, artifact recording,
and prediction inputs do not.

## Scope

This implementation should:

1. Define a safe, versioned canonical proposal artifact for the right Shadow
   Hand.
2. Define a provider interface and registry.
3. Wrap the existing HUG inference/retargeting path as the first provider.
4. Make evaluation consume the canonical artifact instead of importing HUG
   structures directly.
5. Preserve existing HUG commands and historical result interpretation.
6. Add a saved-record provider to prove that the boundary is genuinely
   provider-independent.
7. Specify, without yet implementing, the remote-worker contract for GPU-only
   providers.

The saved-record proof provider should be small and deterministic. It can import
a reviewed Shadow Hand pose fixture or a selected DexGraspNet record after its
license and format are approved. It must not add a large external dataset to Git.

## Non-goals

This change does not:

- alter `adroit_shadow_tabletop_v2` or create a new physics environment;
- alter the Shadow Hand MJCF, actuators, controller, timestep, cameras, scorer,
  table-clearance logic, or execution protocol;
- run or train DexGrasp Anything, UniDexGrasp, GenDexGrasp, or CEDex yet;
- introduce a generic cloud scheduler, credentials, or vendor-specific API;
- compare native execution policies;
- retrain the failure predictor;
- reinterpret or rewrite historical HUG results;
- claim that proposals from different inputs are statistically equivalent.

Provider-native execution policies are a separate future interface. The first
comparison isolates proposal quality by using the same fixed evaluator for all
providers.

## Current coupling to remove

The current source is explicitly HUG-shaped:

- `GraspCase.source` is `Literal["hug"]`.
- `case_runner.py` imports `load_hug_prediction`, converts MANO frames, invokes
  `scripts/infer_sim_observation.py`, assumes `proposal.pkl`, and validates a
  HUG-specific inference report.
- `_source_scene_binding` requires HUG-specific RGB-D hashes and inference
  fields.
- `noisy_rollouts.py` loads HUG predictions and duplicates HUG pose handling.
- result artifacts use names such as `hug_proposal` and
  `hug_inference_provenance`.
- trusted tests rely on pickle files produced by HUG.

The evaluation runner, scoring, recording, object-asset support, table-clearance
checks, dataset exporter, and prediction model are already largely independent
of the proposal generator and should stay that way.

## Architectural boundary

### Provider request

A provider request describes the source inputs without encoding a particular
model's Python objects:

```python
ProposalRequest(
    request_id="mug001_scene004",
    provider=ProviderReference(id="hug_mano_shadow_v1"),
    object=RequestObject(id="mug_001", geometry_content_hash="sha256:..."),
    observation_path=Path("observations/mug_001/scene_004"),  # provider-specific
    embodiment_id="shadow_hand_right",
    seed=42,
    sample_count=4,
)
```

Providers declare requirements such as `rgb`, `depth_m`, `intrinsics`,
`selection_uv`, `point_cloud`, or `mesh`. Missing requirements fail before a
worker starts.

### Canonical proposal artifact

Use JSON for the validated interchange artifact. Do not make arbitrary pickle a
provider boundary. Provider-native outputs may be retained as separately hashed
artifacts for audit, but evaluation reads only canonical JSON and explicitly
referenced numeric arrays.

Proposed `grasp_proposal_v1` shape:

```json
{
  "schema_version": "grasp_proposal_v1",
  "proposal_id": "mug001_scene004_dga_seed42_sample00",
  "provider": {
    "id": "dexgrasp_anything_shadow_v1",
    "config_hash": "sha256:...",
    "code_revision": "...",
    "checkpoint_hash": "sha256:..."
  },
  "generation": {
    "seed": 42,
    "sample_index": 0
  },
  "object": {
    "id": "mug_001",
    "geometry_content_hash": "sha256:..."
  },
  "embodiment_id": "shadow_hand_right",
  "palm_link": "palm",
  "pose_frame": "mujoco_object_body",
  "object_to_palm": {
    "translation_m": [0.0, 0.0, 0.0],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "joint_targets_rad": {
    "rh_FFJ1": 0.0
  },
  "input_binding": {
    "geometry_group_id": "sha256:...",
    "observation_group_id": "sha256:...",
    "generation_request_group_id": "sha256:...",
    "input_manifest_hash": "sha256:...",
    "scene_contract_hash": "sha256:..."
  },
  "native_artifacts": [
    {"role": "provider_output", "path": "native/proposal.bin", "sha256": "sha256:..."}
  ],
  "provider_metrics": {}
}
```

The exact Shadow joint set and names come from the registered MuJoCo Shadow model,
not from a provider-supplied anonymous vector. `T_object_palm` maps the `palm`
link into the MuJoCo object body frame, in meters with an `xyzw` quaternion. Validation rejects missing,
unknown, duplicated, non-finite, or out-of-range joint values.

The canonical palm pose should be expressed relative to the object. Evaluation
composes it with the case's validated initial object transform. This makes the
proposal independent of world placement while retaining an exact scene and
object binding. The HUG adapter must demonstrate numerical equivalence with the
current camera-to-world path before this choice is accepted.

### Provider interface

```python
class GraspProposalProvider(Protocol):
    def generate(
        self,
        request: ProposalRequest,
        output_directory: Path,
    ) -> list[Path]: ...  # paths to canonical proposal.json files
```

Providers are resolved through a versioned registry. Unknown IDs, configuration
hash mismatches, incompatible embodiments, and unsupported input types fail
closed.

Initial registry entries:

```yaml
hug_mano_shadow_v1:
  adapter: grasp_failure_prediction.proposals.hug:HugProposalProvider
  embodiment_id: shadow_hand_right
  execution: local_or_external
  required_inputs: [rgb, depth_m, intrinsics, object_mask, selection_uv]

shadow_saved_record_v1:
  adapter: grasp_failure_prediction.proposals.saved:SavedShadowProposalProvider
  embodiment_id: shadow_hand_right
  execution: local
  required_inputs: [object_geometry]
```

Future entries can identify external workers without importing their dependencies
into the evaluation environment:

```yaml
dexgrasp_anything_shadow_v1:
  embodiment_id: shadow_hand_right
  execution: external
  required_inputs: [object_mesh, object_point_cloud]
```

The core repository should validate an externally produced artifact in exactly
the same way as a locally generated one. It should not install every provider's
CUDA stack into one environment.

## Evaluation-case schema

Add a new version rather than silently changing the meaning of historical
`eval_case_v1` files.

Proposed case fragment:

```yaml
schema_version: eval_case_v2
grasp:
  id: mug001_scene004_sample00
  proposal_path: proposals/mug001/scene004/sample00/proposal.json
  expected_proposal_hash: sha256:...
```

A generation manifest may additionally reference provider inputs and request
fresh inference. Execution manifests should normally point to an immutable,
hashed proposal. Separating generation from execution enables one proposal to
be evaluated under multiple controlled conditions without rerunning inference.

Keep parsing and executing `eval_case_v1` through `run-hug-eval-case` as a permanent
compatibility path. New provider-neutral work writes `eval_case_v2`. Historical
case files and artifacts are never rewritten.

## Commands

Introduce provider-neutral commands while retaining current HUG wrappers:

```bash
generate-grasp-proposals REQUEST.yaml --output proposals/BATCH_ID
validate-grasp-proposal proposals/BATCH_ID/sample_0000/proposal.json
run-grasp-eval-case CASE.yaml --output runs/CASE_ID
run-grasp-eval-batch cases/*.yaml --output-root runs/BATCH_ID
```

`run-hug-eval-case` remains as a compatibility wrapper during migration. It
should eventually call the provider-neutral path rather than maintain a second
execution implementation.

GPU-only workers may initially be invoked manually in their isolated cloud
environments. Their completion contract is a directory containing canonical
proposal artifacts, native-output hashes, logs, and a resolved provider report.
No cloud credentials belong in the case or proposal schema.

## Provenance and grouping

Every proposal must bind:

- provider ID and canonical provider configuration hash;
- provider code revision;
- checkpoint hash when a checkpoint is used;
- generation seed and sample index;
- object ID and dependency-closure geometry hash;
- embodiment ID;
- exact input-manifest hash;
- optional scene-contract and observation hashes;
- canonical proposal content hash;
- hashes of retained native outputs.

The rollout metadata should replace HUG-only names with provider-neutral names:

```text
proposal_provider_id
proposal_provider_config_hash
proposal_checkpoint_hash
proposal_sha256
proposal_input_manifest_sha256
proposal_group_id
```

Keep HUG-specific details under a namespaced provider report rather than flattening
them into the general result schema.

Dataset grouping must continue to keep all executions derived from one proposal
in the same partition. Add provider ID as analysis metadata, not as a predictor
feature in the default experiment. A future source-conditioned predictor must be
separately declared.

## Implementation sequence

### Step 0 — Freeze the current behavior

- Record the current registered environment and protocol hashes.
- Select one trusted HUG proposal with a verified scene contract.
- Save expected canonical joint targets, palm pose, validation result, rollout
  outcome, and artifact hashes needed for equivalence tests.
- Do not regenerate or relabel existing published collections.

### Step 1 — Add canonical schemas

- Add `ProposalRequest`, `ProposalRequirements`, `ProviderReference`,
  `GraspProposal`, `InputBinding`, and native-artifact reference models.
- Use strict Pydantic models, canonical JSON, SHA-256 hashes, relative paths,
  finite-number checks, normalized quaternions, meters, radians, and explicit
  quaternion ordering.
- Add `eval_case_v2`; preserve `eval_case_v1` parsing.

### Step 2 — Add provider registry and artifact validation

- Add a versioned provider registry with deterministic config hashes.
- Resolve provider adapters by `module:Class`, as the environment registry does.
- Add a standalone validator that requires no provider runtime.
- Validate the complete named Shadow joint set against the embodiment model.
- Verify object, input, and provider hashes during portable execution. Checkpoint
  and native-artifact verification is an optional deep audit because those files
  may remain on the generation worker.

### Step 3 — Wrap HUG

- Move fresh HUG subprocess invocation, native pickle loading, MANO frame
  conversion, Dex Retargeting, and HUG-specific provenance checks behind
  `HugProposalProvider`.
- Emit canonical JSON plus an optional retained native pickle.
- Convert the verified HUG palm pose into the object frame.
- Prove that conversion back into the evaluation case produces the same world
  palm transform and joint targets within declared tolerances.

### Step 4 — Decouple execution

- Make the one-case runner load only the canonical proposal interface.
- Remove HUG imports and HUG-specific report parsing from the common execution
  path.
- Reuse the existing pose validator, runner, table-clearance logic, scorer,
  recorder, and artifact writer without changing their physical behavior.
- Rename new generic result fields while retaining compatibility readers for
  historical HUG result bundles.

### Step 5 — Decouple batch collection

- Run canonical cases through the provider-neutral batch command. The older
  `collect-noisy-rollouts` command remains the versioned HUG compatibility collector.
- Preserve proposal, observation and generation-request grouping for downstream splits.
- Make dataset and fresh-evaluation readers accept provider-neutral provenance.
- Confirm that provider metadata remains excluded from default model inputs.

### Step 6 — Prove interchangeability

- Add `SavedShadowProposalProvider`.
- Execute one HUG proposal and one saved Shadow proposal through the same common
  evaluator.
- Demonstrate that neither path changes the environment or execution protocol
  hashes.
- Do not call this a scientific comparison; it is an interface acceptance test.

### Step 7 — Document the external-worker contract

- Document required input and output directory layouts.
- Provide an example unresolved DexGrasp Anything registry entry and fixture
  artifact, clearly labeled as a contract example rather than completed
  inference.
- Defer actual cloud installation and provider-specific adapters to subsequent
  reviewed tasks.

## Tests

### Schema and security tests

- Reject absolute paths and `..` traversal.
- Reject unknown fields, non-finite values, malformed hashes, zero quaternions,
  wrong units, incomplete joints, extra joints, and wrong embodiments.
- Reject proposal/object geometry mismatches.
- Reject tampered native artifacts, input manifests, provider configs, and
  checkpoints.
- Never deserialize an untrusted pickle in the common evaluator.

### Provider tests

- Registry resolution and unknown-provider rejection.
- Required-input validation before provider invocation.
- Deterministic fixture provider output for a fixed seed.
- Multiple samples receive stable unique IDs and sample indices.
- External proposals validate without importing provider dependencies.

### HUG equivalence tests

For the frozen trusted case, compare old and new paths:

- named Shadow joint targets;
- world palm translation and quaternion;
- fingertip alignment metrics;
- pose-validation result;
- commanded action sequence;
- task outcome and failure category;
- trajectory values within an explicitly recorded tolerance.

Any difference must be explained and versioned. Passing only schema tests is not
sufficient.

### Environment invariance tests

- Environment ID and hash remain unchanged.
- Execution protocol ID and hash remain unchanged.
- Runner class, MuJoCo timestep, actuator configuration, cameras, scoring
  thresholds, and table guard remain unchanged.
- The evaluator dispatches identical canonical proposals identically regardless
  of provider metadata.

### Compatibility tests

- Existing `eval_case_v1` HUG fixtures still parse and execute through the
  compatibility path.
- Existing published results remain readable.
- Existing prediction datasets and model artifacts retain their hashes.

## Acceptance criteria

The infrastructure is complete when:

1. One fresh or saved HUG case produces a canonical proposal and executes through
   the provider-neutral runner.
2. The frozen HUG equivalence test passes.
3. A non-HUG saved Shadow proposal executes through the same runner.
4. Common evaluation code imports no HUG model structures.
5. Both runs resolve the same environment and execution protocol hashes.
6. Proposal provenance and tamper checks fail closed.
7. Provider-neutral one-case and deterministic batch execution preserve geometry,
   observation and generation-request grouping. The HUG-specific noisy collector
   remains a permanent compatibility path.
8. The full existing test suite passes, with any optional skips explained.
9. Documentation distinguishes implemented providers from planned providers.
10. No model-specific CUDA dependency is added to the core evaluation
    environment.


## Implemented verification

- A project-authored saved Shadow target generated canonical JSON and completed
  the unchanged MuJoCo runner; its `approach_collision` label is only a dispatch
  smoke result.
- A fresh captured cube observation ran through the pinned HUG checkpoint on CPU,
  produced a canonical proposal, passed the 24-joint validator and completed a
  successful generic evaluation.
- The generic and legacy HUG runners executed that same proposal from the same
  object pose with identical phase/contact arrays. Maximum numeric differences
  were 5.1e-13 m in palm position, 8.2e-12 rad in hand position and 3.5e-9 N in
  contact force.
- The package wheel contains the provider registry and all four new commands.
- Runtime captures, checkpoint data, MANO assets and generated proposals remain
  ignored and outside Git.

## Suggested implementation commits

Keep review units small even though work remains on one branch:

1. `docs: define interchangeable grasp proposal contract`
2. `feat(proposals): add canonical proposal schemas and registry`
3. `feat(proposals): adapt HUG to canonical proposal artifacts`
4. `refactor(eval): consume provider-neutral grasp proposals`
5. `refactor(eval): generalize rollout provenance and grouping`
6. `test(proposals): prove saved-provider interchangeability`
7. `docs(proposals): document external GPU worker contract`

Do not commit generated checkpoints, licensed datasets, MANO assets, cloud
credentials, or large provider-native outputs.

## Decisions applied in the implementation

1. Use `T_object_palm`, relative to the registered MuJoCo object body and `palm` link.
2. Validate the exact 24-joint set from the pinned Shadow URDF.
3. Retain `eval_case_v1` permanently for historical HUG execution and reading.
4. Use a small project-authored deterministic fixture for the saved provider.
5. Keep proposal generation and simulation execution as separate commands.
6. Exclude provider identity from the initial predictor inputs; retain it for grouping and analysis.
7. Declare fixed world-negative-Z approach compatibility in both provider registry and proposal.

## Follow-on work after this plan

After the interface is accepted and verified, add providers one at a time:

1. DexGraspNet saved-record import.
2. DexGrasp Anything external GPU inference.
3. CoToGrasp Shadow inference.
4. UniDexGrasp proposal inference.
5. GenDexGrasp or CEDex for cross-embodiment work.

Each provider task must include one-object visual validation, coordinate and
scale audits, a small physics smoke batch, provenance tests, and a clear license
record before scaling collection.
