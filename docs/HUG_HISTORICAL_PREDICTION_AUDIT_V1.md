# Later historical prediction audit v1

This is a later independent acceptance audit. The new-data model comparison
already ran before this audit was requested. It cannot retroactively satisfy
the original build brief's requested chronological order. It preserves the
original datasets, models, historical readback and retrospective analysis.

Available portable inputs are `research_artifacts/object_scene_pilot_v1`
(1,040 rows, 444 successes / 596 failures, two geometries, four captures,
16 proposal families) and `research_artifacts/fresh_object_scene_eval_v1`
(2,080 rows, 659 successes / 1,421 failures, four geometries, eight captures,
32 proposal families). Their combination contains 3,120 inspected attempts,
1,103 successes / 2,017 failures, four geometries, 12 captures and 48 proposal
families. Attempts and proposal families are not independent environments.

Both historical sources retain the original Shadow `fixed_grasp_lift_v2`
controller and `noisy_rollout_pilot_v3` collection lineage. The fresh frozen
execution reference binds an original resolved-case hash. Collector/scorer
source signatures, input contracts, feature columns and label rules must
agree. Corrected wrist/approach data and new `hug_grasp_approach_v1` recordings
are excluded. No history is relabeled or mixed with the new controller.

The original saved logistic coefficients are in
`research_artifacts/task_outcome_baseline_v1/`. Their arrays, normalization,
report, training implementation and original dataset/split hashes remain
immutable. The existing `scripts/verify_historical_prediction.py` reproduces
the original test full/commands-only AUROCs 0.852253328781154 and
0.6419426425401161. The frozen full model's fresh familiar/new-geometry AUROCs
are 0.917318763408498 and 0.558765241945835. These forecast the final task
outcome from twenty closing samples through 2.96 seconds, not slip onset.
Historical pre-cutoff mug approach collisions remain included in those
historical frozen results; this audit does not change their risk set.

An existing retrospective leave-one-geometry-out analysis was frozen at
`7b41f14f85d82969b01c4ab573c9bb40d2f0e162` and its outputs preserved in
`research_artifacts/hug_retrospective_object_holdout_2026-10-15_v1`.
Its unchanged protocol is `docs/HUG_RETROSPECTIVE_EVALUATION_V1.md`. The full
model's held-out apple, milk and cube AUROCs were 0.823838, 0.834486 and
0.872925; mug is single-class and AUROC is undefined. These are retrospective
development results on inspected outcomes. They are separate from the
original frozen-model generalization results.

`scripts/audit_hug_historical_prediction.py audit` performs no fitting. It
reuses the strict historical loaders and saved-weight replay, compares every
persisted readback metric at absolute tolerance `1e-12`, reconstructs frozen
LOO memberships from source identities, and recomputes all archived LOO model
metrics and calibration from `predictions.csv`. It checks source/protocol/
prediction hashes and all test labels/memberships. LOO coefficients were not
saved in that archive: coefficient reproduction is explicitly `NOT_RUN`,
while archived prediction-metric reproduction is separately reported. It does
not rerun the old bootstrap. Those intervals resample proposal families nested
within only two/four held-out captures and remain conditional diagnostics.

The read-only command blocks all three MuJoCo step APIs and optimizer entry
points. It writes a fresh report with bound source/input hashes, exact source
cohort, counts, archived-fit commit and this later-audit chronology:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python scripts/audit_hug_historical_prediction.py audit --output research_artifacts/hug_historical_prediction_audit_2026-10-15_v1.json
```

Any optional new historical refit is a separate version, not a replacement for
the existing LOO archive. Preparation below performs no optimization. It binds
the exact old inputs, source, runtime, full episode memberships, fixed original
three-model/four-regularization comparison, validation-only selection, 0.5
warning threshold and existing 1,000-resample proposal-family diagnostic.

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python scripts/audit_hug_historical_prediction.py prepare-refit --output research_artifacts/hug_historical_refit_protocol_2026-10-15_v2.json
```

Commit that new declaration and bound sources before invoking `run-refit`.
The driver refuses changed/untracked/staged-only declarations or sources,
verifies inputs again, and records an exact freeze commit and optimization
call count. Existing historical files cannot be the new output.

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python scripts/audit_hug_historical_prediction.py run-refit --protocol research_artifacts/hug_historical_refit_protocol_2026-10-15_v2.json --output research_artifacts/hug_historical_refit_2026-10-15_v2
```

Historical leave-one-failure-mode-out is not implemented by this audit. The
existing retrospective protocol explicitly defers it pending a risk-set design
for failures already occurring before the cutoff. A documented skip remains
more accurate than labeling those records as future failures. This audit does
not establish broad object-category transfer, physical slip prediction, new
mapping fidelity or real-robot performance.
