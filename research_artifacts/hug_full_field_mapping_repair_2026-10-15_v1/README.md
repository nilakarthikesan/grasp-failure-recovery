# Current-geometry full-field HUG mapping diagnostic v1

Prepared 15 October 2026. This is the safe diagnostic prerequisite to a new
constraint-exchange repair, not another execution of the historical 178-row
fitter. It changes no mapper, scientific threshold, original tip cap, native
input, controller, dataset or model. No new native HUG inference, retargeting
solve, physics, task label or training is included in this version.

The two fixed development cases are milk box B/index20 and mug B/index28. Each
uses two already saved joint commands: the original command under the current
native-frame placement and the latest saved experimental command/root. The
experimental commands remain literal saved roots; they are not silently
reanchored. The four poses are not four independent native grasps. Static-v2
fresh solves, if later authorized, require a separately frozen version.

The current registered `ActuatedShadowRunner` constructs the hand/object scene
through current repository source. Independent source-XML FK and the independent
native-frame composition check placement. Original individual fingertip caps
are read literally from their bound historical reports, never recomputed as new
caps. Tiny current-representation cap excesses remain refusals in the report.
All 45 active hand colliders are inventoried; distances refer to the literal
six anatomical component-surface sets. This does not substitute an exposed
boundary or a simulator-contact measurement for the unchanged rule.

Every query on each case's complete original grid is retained. All native
definite-near **and** definite-far relationships at 1/2/5mm enter the reproducible
constraint ledger. This includes originally passing relationships, not merely
the old capped failure examples. The complete historical definite-mismatch
inventory is reproduced from its bound arrays before measurement. Native and
target ambiguous cells stay explicit; becoming ambiguous does not repair a
native-definite relationship. Candidate reports expose loss of an originally
passing current relationship and newly introduced definite violations.

Each pose's NPZ stores every query, radius, area, native/historical/current
distance, native classification, status mask and complete constraint row index.
The JSON gives all per-part/band inherited area intervals. Nested bands,
anatomical parts and repeated grids overlap; tuple counts and area bounds are
not independent contacts or task outcomes. No area total across parts is used
as a score. Definite mismatches yield `FAIL`. Otherwise the final status is
`INCOMPLETE`: full orientation and continuous coverage are not evaluated here.
This diagnostic has no fidelity `PASS` path.

The mathematical distance and interval kernels are immutable source snapshots;
their historical scene constructors are never called. Current project sources
are separately hash-bound and snapshotted without modifying dirty files.
Existing assets remain read-only/hash-bound and are not redistributed. No old
MJB is loaded and no historical pose-loader overlay is installed.

Execution uses one subprocess per case with a hard 600-second limit, at most48
static forward calls per case, 256-point query chunks, exactly two cases/four
poses, and no retries or replacement. The first unexpected error or timeout
stops later cases. Partial files and logs remain. All three MuJoCo integration
APIs, retargeter calls and inference entry points are forbidden. The declared
maximum collider-point count is the complete cell count times45colliders
times2poses per case; actual queries can be lower because carrier components
are not substituted for the six anatomical parts.

The existing feasible path solver itself is reusable, but its historical Case
constructor builds historical geometry and recalculates caps in that scene.
Calling it unchanged would not qualify a current-geometry repair. Its status in
this version is therefore `BLOCKED_CURRENT_SCENE_30D_ADAPTER_NOT_QUALIFIED`.
A subsequent version must independently qualify a current-scene30D adapter,
its tip Jacobian and serialized root before solving. It can then use fixed
worst-row constraint exchange while preserving all activated constraints and
full-field baseline-passing relationships at checkpoints. No such solve is
claimed in v1.

Prepare only, before measurement:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
  research_artifacts/hug_full_field_mapping_repair_2026-10-15_v1/audit_full_field.py \
  prepare --protocol research_artifacts/hug_full_field_mapping_repair_2026-10-15_v1/protocol.json
```

After root review and committing the source, tests, snapshots and protocol:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
  research_artifacts/hug_full_field_mapping_repair_2026-10-15_v1/audit_full_field.py \
  run --protocol research_artifacts/hug_full_field_mapping_repair_2026-10-15_v1/protocol.json \
  --freeze-commit <exact-commit> --output <fresh-output-directory>
```

The runner requires exact HEAD plus clean tracked protocol/source/snapshots,
verifies all bound current sources and inputs before and after each result,
and writes the resolved40-character commit. The original runtime imports base
NumPy/MuJoCo before appending existing Pydantic/geometry dependency directories.
No environment installation is performed.

Manufactured verification:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
  scripts/check_hug_build.py tests/test_hug_full_field_mapping.py -q
```

The tests refuse forward/integration calls and use manufactured arrays only.
They cover literal boundary semantics, all-native-definite coverage, new and
protected relationship failures, ambiguity, full historical mismatch accounting,
invalid inputs, deterministic ledger order and inherited predicate parity.
Actual measured results belong in the fresh execution receipt.
