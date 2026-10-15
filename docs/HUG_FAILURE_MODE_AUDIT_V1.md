# HUG recorded failure-family audit v1

Prepared 15 October 2026. This is a separate, label-defined audit of the new
recorded cohort. It does not revise the frozen five-model category comparison,
the historical v2 datasets, their splits, or their results. Implementation and
manufactured tests are prepared; real audit fitting and uncertainty calculation
require a separate committed declaration and execution receipt.

## Fixed analysis choice

Only `summary_logistic` is fitted for this audit. It uses the existing
`summarize_prefix` and logistic optimizer without changing them. The fixed L2
grid is `(0.001, 0.01, 0.1, 1.0)`, with at most 400 iterations per candidate.
Normalization is fitted on training rows only. Validation log loss selects a
converged candidate; exact ties choose stronger L2. The failure-warning threshold
is 0.5. There is one deterministic fit per candidate, without a seed sweep.
Each eligible failure fold therefore permits four candidate fits; six taxonomy
entries cap the entire audit at 24. Missing and ineligible folds reduce that
budget, with no replacement or outcome-guided repair.

This choice is intentionally narrower than the existing five-model comparison.
Its results cannot establish that the other models generalize to unseen failure
modes. Failure labels determine the held families, so this is a retrospective
binary task-outcome audit, not prospective unseen-mode detection.

## Family partitions and physical labels

The fixed physical taxonomy is `approach_collision`, `no_contact`,
`failed_acquisition`, `slip_during_hold`, `drop`, and `timeout`, matching the bound
recorder/scorer schema. For each cause, every proposal family with any attempt
showing that cause is assigned to test. Every eligible variant in those families,
including successes and other co-occurring failures, stays in test. No such
family can enter training or validation.

Among the remaining families, validation contains the first `ceil(0.2 * N)`
families sorted by canonical SHA-256 of the audit identity, seed 20261015, held
cause, and family ID. Every other family is training. The rule includes families
with only excluded attempts and performs no class balancing, resampling, or
repair. The declaration records exact family roles, row indices, attempt IDs,
success/failure counts, and exclusions for all three roles.

A missing cause produces `missing_failure_mode`. A cause seen only in causally
excluded prefixes produces `held_mode_has_no_eligible_prefix`. Empty or
single-class train, validation, or test partitions are explicitly skipped, in
that order. A physical failure at/before the prediction cutoff can hold its
family out but cannot become a training row. Mapping refusals, unexecuted rows,
and runtime errors have null physical outcomes and never invent a failure cause.
All planned attempts remain in the denominator inventory.

The strict new-recording loader and training adapter validate feature contracts,
labels, hashes, category/group identities, and attempt/scenario linkage. This
audit additionally requires explicit scheduled attempt IDs to agree exactly
with the bound bijective attempt/scenario map, including excluded rows. It
never reads historical dataset formats or pools cohorts.

## Frozen category uncertainty

The uncertainty calculation reads existing `predictions.json` probabilities,
never model weights or a prediction function. Admission requires the exact
recorded dataset/source bindings, input contract, original category partitions,
episode order, labels, warning threshold, prediction hashes, and all original
five-model by three-seed entries. Original skipped/error entries remain visible.

For each completed category/model/seed entry separately, the existing
`retrospective_holdout.group_uncertainty` estimator resamples whole proposal
families with replacement, retaining every row of each selected family and
retaining multiplicity when a family is selected twice. Its seed is 20261015 and
its count is 1000. It returns 2.5/97.5 percentiles for failure AUROC, success false
alarm rate, and failure recall, plus valid and undefined draw counts. Categories
with fewer than two families are explicitly skipped. Single-class ranking is
undefined; fixed-threshold counts/rates remain available where their class
denominators exist.

These intervals are conditional on each held category and its observed proposal
families. They do not express uncertainty across categories or simulator/source
conditions. Seed repetitions are never pooled, treated as independent datasets,
or resampled. No fitting, threshold tuning, model choice, or new inference occurs
in this calculation. The historical estimator is reused unchanged; historical
loaders, results, and protocols remain untouched.

## Declaration and execution

The module is `grasp_failure_prediction.prediction.failure_mode_audit`. It exposes
`prepare`, `run`, and `uncertainty` commands. Preparation writes a fresh JSON
declaration containing exact fold plans, complete frozen prediction inventory,
cohort and feature contracts, runtime versions, and source/input file hashes.
It performs zero fits and zero predictions. Both execution commands require the
declaration and every bound source file to be tracked and unchanged at Git HEAD.
Input/source/split identities are rechecked before each candidate fit and before
report completion. Unexpected failure hard-stops the audit; later eligible folds
become `NOT_RUN`. Output directories/files must be fresh.

For the current cohort, the intended inputs are:

- `research_artifacts/hug_build_collection_2026-10-15_v2/study_prefixes`
- `research_artifacts/hug_build_collection_2026-10-15_v2/study_schedule.json`
- `research_artifacts/hug_build_collection_2026-10-15_v2/study_folds.json`
- `research_artifacts/hug_build_collection_2026-10-15_v2/study_attempt_scenario_binding.json`
- `research_artifacts/hug_build_model_comparison_2026-10-15_v1/models/report.json`

Run from the canonical `codex/project-work` checkout. The original virtualenv
requires its existing Pydantic dependency directory after base NumPy/MuJoCo have
loaded; no installation or environment modification is needed. A shell helper
can supply that import order:

```sh
audit_python() {
  /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python -c '
import runpy, sys, numpy, mujoco
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
sys.path.append("/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps")
runpy.run_module("grasp_failure_prediction.prediction.failure_mode_audit", run_name="__main__")
' "$@"
}
audit_python \
  --dataset research_artifacts/hug_build_collection_2026-10-15_v2/study_prefixes \
  --schedule research_artifacts/hug_build_collection_2026-10-15_v2/study_schedule.json \
  --category-folds research_artifacts/hug_build_collection_2026-10-15_v2/study_folds.json \
  --attempt-binding research_artifacts/hug_build_collection_2026-10-15_v2/study_attempt_scenario_binding.json \
  --protocol research_artifacts/hug_failure_family_audit_2026-10-15_v1/protocol.json \
  prepare --category-report research_artifacts/hug_build_model_comparison_2026-10-15_v1/models/report.json
```

After review and committing that declaration and source, repeat the same common
arguments with `run --output <fresh-directory>` to fit the separate audit and
calculate frozen category uncertainty. Use `uncertainty --output <fresh.json>`
for uncertainty alone. The latter still performs zero fits and zero predictions.
Do not prepare a new split or alter the grid based on a first result.

Static verification uses manufactured recordings and frozen probabilities:

```sh
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python scripts/check_hug_build.py tests/test_failure_mode_audit.py -q
```

The suite blocks all physics integration APIs and real logistic fitting. Any
saved-model round trip uses a manufactured zero-weight model rather than an
optimizer. Actual cohort counts and metric values belong in a later execution
receipt, not this preparation document.
