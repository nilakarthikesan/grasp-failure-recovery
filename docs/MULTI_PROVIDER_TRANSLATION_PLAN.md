# Multi-Provider Grasp Translation Plan

> **Current workflow, 28 October 2026:** Use `phase1-scene-preparation` for Phase 1 and `project-work` for ongoing work. Follow [the contributor workflow](WORKING_BRANCH.md); this dated document preserves earlier scope, branch identities and source paths.

**Status:** Milestones 1 through 3 implemented locally; fresh GPU inference remains gated
**Date:** 12 October 2026; revised 13 October 2026
**Historical implementation branch:** `multi-provider-translation`

## Implementation audit — 13 October 2026

The repository and pinned `dex-urdf` submodule provide enough evidence to
implement the provider-independent boundary now. The implementation includes a
strict translation-strategy registry separate from the provider registry,
provider kind and license/provenance records, a project-authored named-Shadow
replay decoder, exact identity mapping for the pinned right-Shadow URDF, and the
existing HUG MANO path registered as an official Dex Retargeting strategy. The
mapping and retargeter configuration are content-addressed and resolution fails
closed for an unknown source model, joint set, strategy, or changed mapping
evidence. This registration does not change HUG's physical conversion.

The implemented geometry preparer resolves the existing dependency-closure
object identity and area-samples collision triangles using seeded PCG64 in the
MuJoCo object-body frame. It records meters, the identity frame transform,
sampling configuration, dependency hashes, generated-file hashes, and a
canonical manifest hash. Normals are omitted unless requested.

Pre-rollout structural, kinematic, and scene-distance rejection paths now emit
structured codes with a null task label. Resolved successful evaluations report
provider kind, translation strategy, source revision, and license statuses.
The environment, controller, scorer, execution protocol, and dependency extras
are unchanged.

DexGraspNet is registered as an enabled `dataset_replay` against the official
release pinned at `bd1b13d7248729af117e1d46aaa6266b147a3c7b`. The importer
verifies the bundled record, object mesh, and source-hand hashes before opening
the official object-array format. The mapping converts the documented free palm
pose and 22 source joints into the canonical palm pose and 24 named target
joints. A source-versus-target FK gate measured 0.429 mm mean and 1.722 mm
maximum fingertip residual on the first record, below the registered 2 mm bound.
The licensed CC BY-NC inputs remain in ignored local storage.

The first ten official mug records completed the local gate: seven physical
rollouts reported `approach_collision`, one reported `failed_acquisition`, and
two were preserved as unlabeled `shadow_joint_limit_violation` rejections. This
shows that the free-space DexGraspNet grasp distribution is not automatically
compatible with the fixed vertical tabletop approach. The adapter does not move,
clip, or optimize proposals to improve evaluation outcomes. DexGrasp Anything
retains only the preexisting generic external canonical-proposal contract; its
native GPU checkpoint integration is not implemented or validated.

Still blocked are fresh non-HUG checkpoint inference, GPU/container
characterization, and provider-native rendered comparison. Scene preflight currently
has structured joint and implausible-distance rejection; registered tolerances
for table penetration, object penetration, pre-grasp collision, and full
approach-path collision remain proposed work rather than implemented evidence.

## 1. Goal in plain language

We want several grasp-generation systems to propose grasps for the same object,
then test every proposal with the same Shadow Hand, MuJoCo scene, controller and
success rule.

Each grasp system describes its answer differently. The translation layer turns
each answer into the one representation understood by our evaluator:

```text
provider-specific grasp artifact
        |
        v
provider decoder (parsing only)
        |
        v
registered source-hand pose
        |
        v
reusable translation strategy
        |
        v
object-relative Shadow Hand palm pose
+ named Shadow Hand joint targets
        |
        v
existing MuJoCo evaluator
```

This work does not combine raw MANO landmarks, contact maps and anonymous robot
joint vectors into one table. It converts each of them into a physically
executable Shadow Hand proposal first. MuJoCo then produces comparable rollout
records. Training-data normalization remains a later, separate step.

The implementation must not become one hand-written geometric conversion per
provider. A provider-specific decoder reads the native file format. A reusable,
versioned translation strategy handles geometry for a declared source hand
model. Providers that use the same source Shadow model share that strategy.

## 2. Why this boundary matters

The scientific comparison is only meaningful when the execution conditions are
held fixed. For the initial study, every provider must use:

- the registered right Shadow Hand embodiment;
- the same MuJoCo environment and object asset;
- the same fixed approach, close, lift and hold protocol;
- the same physics settings and collision policy;
- the same success and failure definitions; and
- the same rollout and video recording format.

The grasp proposal is the experimental variable. Provider runtimes, native
formats and model dependencies remain outside the common evaluator.

## 3. Current implemented foundation

The `project-work` branch already contains the provider-neutral foundation.
This is completed infrastructure, not proposed work:

- `grasp_proposal_v1`, `proposal_request_v1`, `eval_case_v2` and
  `eval_result_v2` strict schemas;
- a versioned provider registry with canonical configuration hashes;
- an object-relative `T_object_palm` pose and named Shadow joint targets in
  radians;
- proposal, object geometry, request, observation and native-artifact hashes;
- a HUG provider that runs inference, loads MANO output, retargets it and emits
  a canonical proposal;
- a saved Shadow proposal provider for deterministic replay;
- local, external-command and HTTP provider execution contracts;
- provider-neutral one-case and batch evaluation;
- joint-limit, self-collision and provenance validation;
- compatibility with the historical HUG runner; and
- tests showing that the generic HUG path and the legacy HUG path produce the
  same outcome and numerically equivalent trajectories within serialization
  tolerance.

The existing smoke tests establish that the interface dispatches correctly.
They do not establish that any non-HUG provider produces better grasps.

## 4. Scope of the next implementation

The next work should add provider decoders and registered translation strategies
without changing the evaluation environment. The initial target set is:

| Order | Provider | Initial use | Native input | Canonicalization path |
|---:|---|---|---|---|
| 1 | DexGraspNet | Licensed saved-record replay | object mesh + stored grasp | Shadow pose mapping |
| 2 | DexGrasp Anything | Fresh cloud inference | mesh or point cloud | native Shadow pose mapping |
| 3 | CoToGrasp | Fresh cloud inference | point cloud/object assets | native Shadow pose mapping |
| 4 | UniDexGrasp | Proposal generation only | point cloud | native Shadow goal mapping |
| 5 | GenDexGrasp or CEDex | Later expansion | geometry/contact representation | provider-specific optimization/mapping |

HUG remains the reference provider. UniDexGrasp's learned execution policy is
outside this phase because it would change both the proposal generator and the
controller. DexGraspNet dataset replay must be labeled `dataset_replay`; it must
not be described as fresh model inference.

## 5. Canonical contracts

### 5.1 Provider input bundle

Simulation gives us enough information to prepare different inputs from one
registered object. A provider request may bind any of the following roles:

```text
object_mesh
object_point_cloud
rgb
depth_m
camera_intrinsics
object_mask
selection_uv
scene_contract
```

Every input is a relative path with a recorded content hash. Provider registry
entries declare their required roles. Generation fails before invoking the
provider when a required role is missing.

The input-preparation code must preserve:

- object identity and dependency-closure geometry hash;
- metric scale in meters;
- the transform between object, camera and world frames;
- point-cloud sampling method, count and seed;
- camera calibration and depth convention; and
- any mask or selected-pixel provenance.

Generated point clouds and renders are derived inputs. Their generation
configuration and hashes must be saved so two providers can be shown to have
received equivalent object geometry.

### 5.2 Canonical proposal

Every completed translation must emit the existing `GraspProposal` contract. The essential
physical fields are:

```json
{
  "schema_version": "grasp_proposal_v1",
  "proposal_id": "provider_object_sample_0000",
  "embodiment_id": "shadow_hand_right",
  "palm_link": "palm",
  "pose_frame": "mujoco_object_body",
  "object_to_palm": {
    "translation_m": [0.0, 0.0, 0.0],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "joint_targets_rad": {
    "rh_FFJ1": 0.0
  }
}
```

The full artifact also records provider configuration and code revision,
checkpoint hash, generation seed and sample index, object/input bindings,
provider validation metrics and hashes of retained native outputs.

The provider decoder owns only native parsing:

- native file and tensor layout;
- provider field names;
- provider-declared hand-model identity;
- native units, frames and rotation representation as metadata; and
- preservation of the native output for audit.

The registered translation strategy owns geometric conversion:

- units to meters and radians;
- coordinate-axis and handedness conversion;
- rotation representation and quaternion ordering;
- object scale application;
- palm-link definition;
- joint-name and joint-order conversion;
- coupled or missing joint handling;
- forward-kinematics verification; and
- translation-quality metrics.

The decoder or translator may not change the proposal to make an evaluation
succeed. Any repair, optimization or collision projection must be explicit,
versioned and recorded as part of the provider configuration.

### 5.3 Translation-strategy registry

The translation registry resolves an exact source-hand model into the canonical
right Shadow Hand. It is separate from the provider registry because multiple
providers may emit the same hand representation.

```yaml
dexgrasp_shadow_right_to_shadow_hand_right_v1:
  strategy: exact_kinematic_mapping
  source_hand_model_id: dexgrasp_shadow_right
  target_hand_model_id: shadow_hand_right
  source_asset_hash: sha256:...
  target_asset_hash: sha256:...
  source_root_link: palm
  target_root_link: palm
  joint_mapping_path: configs/translations/dexgrasp_shadow_right_v1.yaml
  maximum_fingertip_error_m: 0.0001

mano_right_to_shadow_hand_right_v1:
  strategy: official_retargeter
  source_hand_model_id: mano_right
  target_hand_model_id: shadow_hand_right
  retargeter_id: dex_retargeting
  retargeter_config_hash: sha256:...
  maximum_mean_fingertip_error_m: 0.01
```

Every constant must be traceable to a pinned source asset, robot description,
official preprocessing implementation or official configuration. The registry
must record the evidence location and its hash. No transform, sign, zero offset
or scale may be selected because it improves MuJoCo success.

The first supported strategies are deliberately narrow:

1. `exact_kinematic_mapping` for a documented Shadow Hand model; and
2. `official_retargeter` for MANO through the existing Dex Retargeting path.

A contact-map provider must first use its authors' released hand-fitting method
to produce a recognized source-hand pose. An unknown or undocumented source
hand fails closed. This project will not invent an optimizer to make it fit.

### 5.4 Provider classification

Add or retain an explicit kind in provider documentation and resolved reports:

```text
model           trained network produces a new proposal
optimizer       numerical search produces a new proposal
dataset_replay  a previously generated proposal is imported
```

Analyses must not silently pool these categories under the word "inference."

## 6. Runtime design

Provider generation and evaluation stay separate:

```text
provider runtime                    translation boundary                 core runtime
----------------                    --------------------                 ------------
load model/checkpoint               decode native artifact              validate JSON
prepare native tensors              identify hashed source hand         load Shadow model
run inference/optimization  ----->  resolve translation strategy -----> execute protocol
write native artifact               verify geometric residuals          record result
                                    write canonical proposal
```

Each external provider can use its own Linux NVIDIA container or environment.
This is one environment per provider installation, not one environment per
evaluation case. A batch generation job can emit many canonical proposals.
Those portable artifacts can then be evaluated by the existing runtime.

Provider packages must not be added to the core evaluation dependency group.
The boundary is canonical JSON plus hashed native artifacts. The project may
invoke a provider through the existing external-command or HTTP contract.

## 7. Validation stages

Validation should distinguish translation errors from grasp failures.

### Stage A: structural validation

Reject the proposal before loading MuJoCo when:

- required fields or Shadow joints are missing;
- unknown joints or unsupported embodiment IDs are present;
- numbers are non-finite;
- the quaternion is not normalized;
- units or pose frame are unsupported;
- hashes do not match; or
- provider, object or protocol bindings are stale.

### Stage B: kinematic validation

Load the registered hand model and reject when:

- any joint exceeds its declared limit;
- required coupled-joint semantics cannot be reproduced;
- the palm link does not match the canonical hand model; or
- self-contact exceeds the declared acceptance rule.

### Stage C: scene preflight

Before executing the fixed protocol, detect:

- table penetration;
- excessive object penetration;
- clearly inverted or implausibly distant palm placement;
- an invalid pre-grasp offset; and
- an approach path that begins in collision.

The exact tolerances must be registered and hashed. A rejected proposal receives
a null task label and a structured rejection code. It is not counted as a
physical grasp failure.

Suggested statuses:

```text
translation_failed
proposal_rejected
runtime_failed
completed_success
completed_failure
```

## 8. Provider integration template

Every new provider should follow the same sequence.

### 8.1 Audit the official release

Record before implementation:

- official source repository and pinned revision;
- code, weights, dataset and object-asset licenses;
- released checkpoint hashes;
- supported hand model and exact Shadow asset;
- expected object scale and coordinate frame;
- native joint order, limits and rotation convention;
- required operating system, Python, CUDA and PyTorch versions; and
- whether the release supports inference, optimization or replay only.

If the official runtime or checkpoint is unavailable, stop and classify the
provider as unsupported rather than inventing an approximate implementation.

### 8.2 Build one native-output fixture

Run the provider on one redistributable test object and save:

- the exact input bundle;
- command and resolved configuration;
- seed;
- native output;
- logs;
- checkpoint and source revision; and
- one rendered native visualization when available.

Large or licensed files remain outside Git. Tests use a minimal, legally
redistributable fixture or a project-authored synthetic record.

### 8.3 Implement the native decoder

Write a small provider-specific decoder that parses the native record into a
structured source-hand pose. It must declare the source hand-model ID, root
link, native joint names, units, pose frame and rotation representation. It may
not contain empirically selected wrist offsets, scale factors or joint signs.

### 8.4 Resolve or add a translation strategy

Reuse an existing strategy when another provider declares the same hashed
source hand model. Add a strategy only when the source model or official
retargeting path is new.

Keep frame conversion in small named transforms rather than one opaque matrix
expression. Exact joint signs, zero conventions and link transforms must come
from the pinned robot descriptions or official code. Tests independently
reconstruct the expected object-to-palm pose.

For a native Shadow provider, the conversion should conceptually be:

```text
native object and hand transforms
        -> undo provider object scaling
        -> convert provider axes to canonical axes
        -> resolve native palm link to canonical palm link
        -> map native joint vector to named Shadow joints
        -> emit T_object_palm and joint_targets_rad
```

For HUG, the existing path remains:

```text
camera-frame MANO wrist and landmarks
        -> camera/object frame conversion
        -> Dex Retargeting to Shadow joints
        -> canonical palm link correction
        -> emit T_object_palm and joint_targets_rad
```

Translation acceptance is based on geometry before task execution. It must not
use lift success, failure rate or any other MuJoCo outcome as a tuning signal.

For exact mappings, sample legal source joint configurations, run forward
kinematics in both source and target models and compare corresponding palm,
finger and fingertip positions. Residuals should be near numerical precision.
For official retargeters, record the declared landmark or fingertip residual
and enforce the registered acceptance threshold.

### 8.5 Validate visually and numerically

For one object and ten proposals:

1. render the canonical pose without executing it;
2. inspect scale, palm direction, fingertip placement and table clearance;
3. compare native and canonical fingertip/contact locations where available;
4. record joint-limit and collision metrics;
5. execute only accepted proposals; and
6. review videos and structured results.

Do not begin a large collection until this gate passes.

Visual review is a final sanity check for approximately ten proposals. It is
not the method used to discover or tune translation constants.

### 8.6 Prove evaluator invariance

The provider-integration acceptance test must confirm that adding the provider
does not change:

- environment ID or configuration hash;
- execution-protocol ID or configuration hash;
- MuJoCo timestep or solver settings;
- Shadow Hand asset, actuators or controller;
- approach, lift and hold sequence;
- camera configuration;
- success/failure thresholds; or
- trajectory and video schema.

## 9. Implementation milestones

### Milestone 1 — Review and harden the existing boundary

**Purpose:** Confirm that the implemented foundation is suitable for external
providers before adding one.

Tasks:

- review the current schemas, registry and external worker contracts;
- document the exact canonical Shadow joint inventory and palm-link semantics;
- split native decoding from reusable translation strategy resolution;
- register source/target hand assets and mapping-evidence hashes;
- enforce the no-outcome-tuning and fail-closed policies;
- ensure provider kind and license metadata have an explicit home;
- add missing rejection codes for structural, kinematic and scene failures;
- verify that provider metadata is not used by the current predictor; and
- run the focused proposal tests and the full existing test suite.

Acceptance:

- the current HUG equivalence evidence remains valid;
- a saved canonical proposal executes without importing HUG; and
- rejected proposals cannot enter the task-failure dataset.

### Milestone 2 — Add reusable geometry input preparation

**Purpose:** Give mesh/point-cloud providers equivalent, traceable inputs.

Tasks:

- export or resolve the registered object mesh and dependency hash;
- generate a seeded point cloud in object coordinates;
- store point normals only when a provider requires them;
- write an input manifest containing units, transforms and hashes;
- add deterministic tests for point sampling and frame conversion; and
- verify the object reconstructed from the exported points overlays the MuJoCo
  object at metric scale.

Acceptance:

- repeated preparation with the same seed is identical;
- changed geometry or sampling configuration changes the manifest hash; and
- the prepared geometry aligns with the simulated object within a declared
  tolerance.

### Milestone 3 — Add DexGraspNet replay

**Purpose:** Exercise a non-HUG translation path before introducing a GPU model
runtime.

Tasks:

- audit the selected records and their redistribution terms;
- map DexGraspNet object identity and scale to a registered project object;
- convert one saved Shadow pose into the canonical palm and joint definitions;
- retain the original dataset record and hash;
- classify the provider as `dataset_replay`; and
- run the ten-proposal visual and physics gate.

Acceptance:

- one licensed saved record reaches the common evaluator;
- the result names DexGraspNet replay as its source;
- coordinate and scale checks are documented; and
- no evaluator configuration changes.

### Milestone 4 — Add DexGrasp Anything inference

**Purpose:** Prove that a GPU-generated native Shadow proposal can cross the
portable boundary.

Tasks:

- create a pinned external runtime/container;
- obtain and hash the official checkpoint;
- implement the provider-side input reader and output exporter;
- map native Shadow joints and palm frame to the canonical model;
- expose generation through the external-command or HTTP contract;
- retain native output, resolved configuration and logs; and
- run one-object and ten-proposal acceptance.

Acceptance:

- fresh inference produces validated canonical proposals;
- rerunning a fixed seed is characterized as deterministic or explicitly
  nondeterministic;
- invalid native proposals are rejected with useful diagnostics; and
- accepted proposals execute in the unchanged MuJoCo evaluator.

### Milestone 5 — Add CoToGrasp

Repeat the Milestone 4 process using its official Shadow checkpoint and point
cloud contract. Do not reuse DexGrasp Anything frame assumptions. Acceptance
requires an independent coordinate, joint-order and license audit.

### Milestone 6 — Add UniDexGrasp proposal generation

Integrate only its grasp-goal generator. Do not invoke its learned execution
policy in the common comparison. If its output cannot be executed by the fixed
controller without changing proposal semantics, record that incompatibility
instead of modifying the evaluator.

### Milestone 7 — Small paired comparison

After at least two non-HUG sources pass their adapter gates:

- select a declared set of common objects;
- use matched object geometry, mass, initial pose and execution protocol;
- request a fixed number of proposals per provider and object;
- preserve provider-native rejected proposals in the denominator;
- execute accepted proposals with repeated simulation seeds only when the
  source of randomness is declared; and
- report generation rejection, task success and failure type separately.

This milestone tests whether the comparison protocol works. It is not yet the
large prediction-training collection.

## 10. Tests required for every provider

### Unit tests

- native joint order maps to the correct named canonical joints;
- meters, radians, axis directions and quaternion ordering are correct;
- object-to-palm transform matches an independently calculated fixture;
- missing, extra and out-of-range joints fail;
- malformed and non-finite transforms fail;
- input, checkpoint, provider and native-artifact hash mismatches fail; and
- a fixed native fixture produces stable canonical JSON.

### Integration tests

- provider execution emits the requested number of uniquely identified samples;
- external output validates without importing the provider's ML dependencies;
- accepted proposals load into the registered Shadow model;
- rejected proposals do not launch a task rollout;
- one accepted proposal reaches a complete result with trajectory and video;
- environment and protocol hashes match the HUG path; and
- batch execution preserves proposal and object grouping.

### Manual review gate

- view the hand above the correct object;
- confirm palm orientation and approach direction;
- confirm finger configuration resembles the provider's native visualization;
- inspect table/object penetrations;
- watch the approach, close, lift and hold video; and
- record approval or rejection with artifact paths.

## 11. Dataset and predictor boundary

No new statistical normalizer is needed to implement provider translation.
Once comparable rollouts exist, the training-data builder can combine them.

The default pooled dataset should contain provider ID as provenance and an
analysis/grouping column, not automatically as a predictor feature. Otherwise,
the predictor could learn a provider's average success rate instead of the
physical warning signals.

Before pooled training:

- keep all executions derived from one proposal in one split;
- keep object and observation families separated according to the declared
  generalization test;
- fit numeric scaling on the training partition only;
- preserve original physical units in stored source records;
- report performance and calibration per provider as well as pooled; and
- include an unseen-provider test only if enough providers and proposals exist.

Whether pooled multi-provider training improves prediction is a hypothesis.
The translation layer only makes that experiment possible.

## 12. Licensing and provenance gate

Before enabling a provider, create a short provider record covering:

```text
code license
checkpoint license
dataset license
object-asset license
allowed research/commercial use
required attribution
redistribution restrictions
source revision and download location
```

Do not commit licensed MANO assets, large checkpoints, restricted dataset
records, cloud credentials or generated experiment batches. Record their
content hashes and expected external locations.

## 13. Explicit non-goals

This plan does not:

- replace the Shadow Hand or add multiple evaluation embodiments;
- change the MuJoCo environment, controller or success rule;
- compare native learned execution policies;
- claim that dataset replay is model inference;
- train a new predictor before the comparison dataset is validated;
- treat adapter rejection as grasp failure;
- merge existing scientific protocol versions; or
- regenerate or relabel historical experiment results.

## 14. Main-agent execution order

The main agent should proceed in this order:

1. Review this plan against the implemented proposal framework.
2. Run and document the current focused/full test baseline.
3. Fill any schema or rejection-status gaps without changing physical behavior.
4. Implement deterministic object mesh/point-cloud input preparation.
5. Add DexGraspNet replay and pass the one-object gate.
6. Review the evidence before selecting the first GPU provider.
7. Add DexGrasp Anything and pass the same gate.
8. Add CoToGrasp, then UniDexGrasp, one provider at a time.
9. Run the small paired comparison.
10. Design the pooled prediction dataset only after the rollout records pass
    provenance, grouping and outcome-quality checks.

For this provider plan, each material milestone should be its own commit on `project-work`. Update
`docs/RESEARCH_DECISIONS.md` with implemented evidence, rejected assumptions and
the next check. Do not describe planned integrations as completed.

## 15. Definition of done

The translation-layer expansion is complete for the first study when:

1. HUG plus at least two independent non-HUG providers produce canonical
   Shadow proposals.
2. At least one non-HUG provider performs fresh checkpoint inference.
3. Every provider passes coordinate, scale, joint, collision and provenance
   tests.
4. Providers that declare the same hashed source hand reuse one translation
   strategy rather than duplicate geometric logic.
5. Every mapping constant has pinned source evidence; none was tuned against
   evaluation success.
6. Exact mappings pass randomized forward-kinematics equivalence tests, while
   retargeted mappings satisfy their registered residual limits.
7. All accepted proposals use the same unchanged evaluator and protocol.
8. Rejections, runtime errors and physical task failures are distinct.
9. Videos, trajectories and resolved configurations are attached to results.
10. Provider licenses and runtime requirements are documented.
11. A small paired comparison completes on common objects.
12. The full repository test suite passes.
13. The resulting records are sufficient to design a pooled prediction dataset
    without translating native provider outputs again.

## 16. Rough effort estimate

These are planning estimates, not measured completion times:

- boundary review and validation hardening: 2–4 working days;
- reusable geometry input preparation: 2–4 days;
- DexGraspNet replay adapter: 2–4 days;
- each fresh cloud inference provider: 5–10 days;
- paired pilot, artifact review and fixes: 3–5 days.

A careful HUG plus two fresh/replay non-HUG comparison is likely a multi-week
integration effort. Model inference time is unlikely to dominate; coordinate
verification, environment compatibility, licensing and artifact review are the
larger risks.

## 17. Reference implementations

- HUG: <https://github.com/KevinyWu/hug>
- DexGraspNet: <https://github.com/PKU-EPIC/DexGraspNet>
- DexGrasp Anything: <https://github.com/4DVLab/DexGrasp-Anything>
- CoToGrasp: <https://github.com/CEA-LIST/CoToGrasp>
- UniDexGrasp: <https://github.com/PKU-EPIC/UniDexGrasp>
- GenDexGrasp: <https://github.com/tengyu-liu/GenDexGrasp>
- CEDex: <https://github.com/GeorgeWuzy/CEDex-Grasp>

These links identify candidate official releases. Inclusion still requires the
provider audit and acceptance gate above.
