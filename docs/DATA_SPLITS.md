# Step 5: portable closing histories and grouped splits

The first task-outcome dataset keeps the existing labels: **1 = success,
0 = failure** under the frozen 14 cm lift/2 s hold/2 cm final-loss rule. Labels
and failure categories were not changed to improve class balance. The
[working paper](paper/working_draft.tex) and [evidence ledger](PAPER_PROGRESS.md)
explain the task cutoff and the difference from future-slip prediction.

## What is published

[`research_artifacts/object_scene_pilot_v1/`](../research_artifacts/object_scene_pilot_v1/)
contains a compact, derived dataset for all 1,040 completed attempts. The raw
MuJoCo models, full simulator states, HUG/MANO files, meshes and checkpoints
remain local. Each row contains the final twenty closing samples through
**2.96 s**, before the first lift sample at **3.00 s**:

- 24 measured hand joint positions, in radians;
- 24 measured joint velocities, in radians per second;
- 24 actual clipped commanded targets, in radians.

The time-major window is flattened to 1,440 columns. `manifest.json` records
the field/joint ordering, units, sample offsets, frozen label rule, source hashes
and provenance. `dataset.npz` stores `X` separately from `y` and source identities.
These identities, outcomes, contacts, object state, event annotations, injected
noise settings, and all future samples are excluded from the model matrix.
The file is loaded with `allow_pickle=False`.

## Assignment and scope

The declared scenes determine membership, without consulting their outcomes:

- **Training:** cube A and apple A, 520 attempts, 231 successes / 289 failures;
  eight proposal families and two observation families.
- **Validation:** cube B, 260 attempts, 155 successes / 105 failures;
  four proposal families and one observation family.
- **Test:** apple B, 260 attempts, 58 successes / 202 failures;
  four proposal families and one observation family.

Every complete trajectory, proposal and all of its noise variants stay in one
partition. Observation identities are also disjoint. Both geometries occur in
training; validation is cube-only and test is apple-only. This is a
**retrospective development holdout**, selected after the collection and its
outcomes were inspected. It is not an untouched official benchmark or evidence
of generalization to unseen geometries. Four observations are too few for a
broad population claim.

`splits.json` stores all memberships and group counts. The loader recomputes the
declared assignments and verifies payload hashes, label/count agreement,
provenance references, named column ordering and group consistency. Invalid
or unlabeled source attempts are listed as exclusions rather than assigned
failure labels; this collection has zero such exclusions.

## Reproduce

From the project source, exporting the original local collection into a new
directory uses:

```sh
python -m grasp_failure_prediction.prediction.dataset \
  --runs-root /path/to/collection_1040 \
  --output /path/to/new_prefix_dataset
```

The published derived files support training without the raw collection or
HUG runtime. The source exporter fixes the 20-step, 2.96 s contract and refuses
to overwrite nonempty output. It preserves the source label/scorer version and
records the raw action, trajectory, metadata, result and configuration hashes.
Changing any prediction target or prefix requires an explicitly new version.

## Verification

The actual export contains **1,040 × 1,440** float32 input values, 444 success
labels and 596 failure labels, 16 proposal groups, four observation groups and
two geometry groups. The portable files occupy about **6.55 MB**. The feature
payload SHA-256 is
`29f773f7faf03cff5e1c694ef475d02adfd07c239314b349f2228f86db9d3aad`.

The 28 focused exporter/loader tests cover causal future/privileged-data
mutations, exact column order, timestamps and phases, group conflicts,
label-independent assignments, source-label/scorer agreement, exclusions,
corrupted exports and overwrite protection. A fresh export of all 1,040
actual episodes passed the same validated loader.
