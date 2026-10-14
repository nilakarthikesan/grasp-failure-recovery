# ContactGen and UniDexGrasp proposal integration

Implemented 13 October 2026. These adapters and isolated inference workers feed
the existing `grasp_proposal_v1` boundary. No ContactGen or UniDexGrasp inference,
simulation acceptance check, or evaluation collection was run for this change.
Unit fixtures are synthetic implementation checks, not model-generated evidence.

## The four requested sources

- **HUG:** existing MANO inference and translation path.
- **DexGrasp Anything:** existing [pinned GPU-worker handoff](DEXGRASP_ANYTHING_CLOUD_HANDOFF.md);
  that separate implementation remains in progress.
- **ContactGen:** `contactgen_mano_shadow_v1`, refined human MANO grasp translated
  through the same Dex Retargeting configuration as HUG.
- **UniDexGrasp:** `unidexgrasp_shadow_v1`, grasp-generation output translated
  through the existing named Shadow mapping after source-kinematics verification.

DexGraspNet saved-record replay remains a separate dataset-replay provider. It
does not stand in for fresh inference from any of these models.

## Generation and execution remain separate

Each new provider supports two paths:

1. An isolated model worker accepts the existing `--request PATH --output DIRECTORY`
   command contract, performs inference, preserves its native output and emits
   canonical proposals.
2. A local adapter imports a completed native JSON bundle and translates it
   without invoking the generator. Its metadata explicitly identifies an import.

Both paths use a versioned JSON file in `provider_input_path`. A fresh worker's
input manifest describes its geometry and frames; a completed bundle contains
native model output. ContactGen accepts one input channel: `provider_input_path` or a sole
`provider_inputs` entry with role `provider_input`. UniDexGrasp requires
`provider_input_path` alone. Mixed observation/native/geometry channels are
rejected to avoid ambiguous input identity. The original request manifest remains the input grouping
identity even when a worker creates a separate native bundle. Proposal metadata
records the pinned upstream revision, model/checkpoint hashes, seed, sample index
and adapter revision. A source revision declaration in imported JSON is
provenance from the exporting worker, not independent proof of model inference.

The core evaluator receives exact named Shadow targets in radians and
`T_object_palm` in meters. The existing environment, controller, approach,
lift/hold thresholds and `fixed_grasp_lift_v2` execution protocol are preserved.
A valid translation does not establish tabletop clearance or grasp success.

## ContactGen: complete the solver before MANO translation

Official source:
[stevenlsw/contactgen](https://github.com/stevenlsw/contactgen/tree/1aea2b6ce498ec90c4af0dd4c5fbdd4a23db761a),
revision `1aea2b6ce498ec90c4af0dd4c5fbdd4a23db761a`.

The official [demo](https://github.com/stevenlsw/contactgen/blob/1aea2b6ce498ec90c4af0dd4c5fbdd4a23db761a/demo.py)
samples contact location, part and direction, then runs `optimize_pose`. The
released demo exports OBJ meshes rather than this project's proposal JSON.
Our worker `scripts/infer_contactgen_object.py` follows that inference/solver
path and additionally preserves optimized MANO parameters, 778 vertices and
16 bone frames. Contact-only outputs are rejected.

The worker accepts `contactgen_object_input_v1`: exact mesh path/hash, object
ID/geometry hash, `units: meters`, and a declared mesh-to-object rigid transform.
It retains the vertex-mean centering offset. Its native output is
`contactgen_native_bundle_v1`, with `generation_stage: refined_mano`, right-hand
PCA-26/nonflat-mean configuration and hashes for ContactGen, the hand-SDF model
and MANO asset. External model/asset paths are supplied to the worker; no weights
or licensed MANO assets are bundled here.

The native importer also checks the original mesh hash and recomputes its
vertex mean before translation; retain that exact mesh under the project root.
The adapter restores centering and mesh placement, constructs the standard
21 MANO landmarks from the 16 joint frames plus five mesh fingertips, and
reuses `mano_right_to_shadow_hand_right_v1`. The wrist frame includes the shaped
MANO root position; optimized translation alone is not the wrist origin.
ContactGen's modified MANO implementation returns metric coordinates; the
adapter does not apply the upstream Manopth millimeter conversion. Shadow palm
placement follows the existing HUG wrist-origin policy and operator rotation.
The registered mean fingertip residual limit is 10 mm.

The worker command takes `--project-root`, `--contactgen-root`, `--checkpoint`,
`--checkpoint-sha256`, `--hand-sdf-checkpoint-sha256` and `--mano-asset-sha256`.
Optional `--hand-sdf-checkpoint` and `--mano-root` override the source checkout's
asset locations. Hash arguments use `sha256:` plus 64 lowercase hexadecimal
digits and are checked before any checkpoint or MANO deserialization. The common
external-command dispatcher appends `--request` and `--output`.

The upstream installation needs its own PyTorch/CUDA, PyTorch3D and PointNet
extension environment. Its full solver also needs the external hand-SDF and
licensed MANO files. Repository-wide source and separate checkpoint terms were
not verified; assets remain external and their license state is recorded.

## UniDexGrasp: proposal generation, source frame and Shadow verification

Official source:
[PKU-EPIC/UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp/tree/36c9bfcf7cedc987dd15e2384c548e51d5aadc21),
revision `36c9bfcf7cedc987dd15e2384c548e51d5aadc21`.
The integration uses its `dexgrasp_generation` proposal models. Its learned
execution policy is separate from our fixed MuJoCo evaluator.

The worker `scripts/infer_unidexgrasp_proposals.py` loads externally supplied IPDF
and GraspGlow configurations/checkpoints and processes the explicitly declared
input points through the IPDF-plus-Glow proposal stage without test-time
adaptation (TTA). It uses the non-joint-training `glow_config`; checkpoints must
match that architecture. It does not run the upstream dataset evaluation loop.
Native imports can select `tta_hand_pose` only when their report declares TTA
and binds the additional contact-model checkpoint. This worker does not perform
TTA or establish any before/after comparison. Native
hand output uses translation, axis-angle rotation and 22 named `robot0:` finger
joints; the input-to-canonical rotation must be inverted when composing the
pose back into the input frame. The adapter then applies the declared
input-to-object transform and actual source palm root offset.

The fresh worker input is `unidexgrasp_object_input_v1`. It binds object ID and
geometry hash, physical object scale, `object_to_input`, a bundle-relative numeric
point-cloud path/hash, `point_cloud_units: m`, `point_cloud_frame: input_point_cloud`
and `plane_input` (unit normal plus metric offset). Points must already describe
the physical geometry in the declared frame and contain at least 1,024 XYZ rows.
The worker neither samples a new cloud nor rescales coordinates. Native output
is `unidexgrasp_native_bundle_v1`, with numeric NPZ, a hash-bound inference report
and the source hand MJCF. The report preserves original input and effective
model-configuration hashes.

Worker arguments are `--project-root`, `--unidexgrasp-root`,
`--rotation-checkpoint`, `--rotation-checkpoint-sha256`, `--pose-checkpoint`,
`--pose-checkpoint-sha256` and `--source-hand-mjcf-sha256`. Optional
`--source-hand-mjcf` overrides `dexgrasp_generation/data/mjcf/shadow_hand.xml`;
`--device` defaults to `cuda:0`. Asset hashes are verified before checkpoint
loading, and model imports must resolve inside the pinned upstream checkout.

The external UniDexGrasp Shadow model is not assumed identical to DexGraspNet's
source hand. Its effective palm-subtree kinematics must match the pinned
DexGraspNet reference before the existing joint/sign mapping is reused. The
actual UniDexGrasp asset hash and equivalence certificate remain in provenance.
The translated pose passes the existing 2 mm maximum fingertip FK residual
gate. Execution preflight enforces canonical joint limits before any rollout.
Rejections do not clip or repair a grasp.

These workers import the core package and require Python 3.10 or later, with
compatible builds of the upstream CUDA dependencies. The upstream guides use
older Python versions; those exact historical environments cannot directly run
these workers. Source/configuration/checkpoint assets stay in the isolated model environment;
no framework dependency is added to the core package. The official repository
licenses its work and dataset under CC BY-NC 4.0. Separate checkpoint terms
remain unverified and are recorded conservatively.

## Validation and the next gate

The focused suite passes **72 tests**, with the two existing rollout tests
deliberately excluded. A test-run guard prohibits all MuJoCo stepping APIs. Both
worker `--help` entry points succeed without loading their model runtimes. The
eight changed/new Python files parse; the frozen predictor hash is unchanged.

Implemented unit checks cover native contracts, coordinate conversion,
provenance and asset mismatches, malformed values, grouping, registration and
the external-command boundary. Existing provider configuration hashes remain
unchanged. No generated model proposal or rollout is research evidence from
this implementation.

Fresh worker execution and one visual/physics acceptance case from each source
are deferred until the user requests generation. That next check should retain
the original native outputs and verify the declared geometry/frame/kinematics
before any larger batch. The [predictor recording gap](THREAD_HANDOFF.md) also
remains: the generic recorder lacks the closing velocity and command histories
needed to score the historical frozen predictor.
