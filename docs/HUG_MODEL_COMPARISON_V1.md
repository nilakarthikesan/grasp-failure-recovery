# HUG recorded model comparison v1 — preparation only

Prepared 15 October 2026. `prediction/recorded_training.py` implements a new
comparison for `hug_recorded_prefix_dataset_v1`. No real-data model has been
trained in this preparation. Software tests use manufactured recordings and
tiny synthetic fitting, including a one-epoch GRU test. Execution must follow
Phase 1 signoff, frozen Phase 2/3 protocols and actual new collection. Historical
datasets, retrospective results and frozen baseline weights remain unchanged.

## Inputs and collection binding

Use the strict `prediction.recorded_dataset.load_recorded_dataset` adapter. It
checks schema, dataset hash, joint ordering, recorded causal endpoint times,
physical labels, exclusion denominators and the exact feature allowlist. Inputs
are only measured joint positions, measured joint velocities and actually
applied clipped targets in a declared close-only history. Requested targets,
contact forces, object state/properties, future samples, diagnostics, labels,
mapping/filter values, category/object/capture/proposal identities and seeds
are excluded from feature arrays. Final task labels provide supervision only.
The target is final task outcome from a causal close prefix, not physical slip
onset. Failures at/before cutoff and approach collisions remain counted but
are excluded by the exporter before training.

Three additional collection artifacts are required:

- `hug_scenario_schedule_v1`, created before outcomes by `scenario_sampler.py`.
- `hug_category_folds_v1`, exactly equal to `freeze_category_folds(schedule)`.
  Each category is held out once; the next sorted category is validation and
  remaining categories train. All variants of a category stay together.
- `hug_attempt_scenario_binding_v1`, with `schedule_sha256` (canonical sorted
  JSON SHA-256, prefixed `sha256:`) and an `attempt_to_scenario` dictionary.
  It must bijectively cover **all** scheduled scenarios and all exporter audit
  rows, including refusals, runtime errors and early exclusions. Repeats must be
  explicitly scheduled with distinct scenario IDs before collection.

Every included recorder attempt must also contain
`provenance_hashes["scenario"] = canonical_sha256(its_full_schedule_row)`.
`canonical_sha256` is exported by the training module. The actual executed
scenario must be bound during collection, not guessed from names afterward.
Included object/geometry/observation/proposal identities must match the row.
Missing linkage is a hard adapter refusal; the existing saved-command pilot
cannot silently acquire the new category folds or scenario protocol.

The adapter verifies stable grouping bindings and no shared category, geometry,
object, observation or proposal family across train/validation/test in each
fold. It stores canonical and file hashes for schedule, folds, binding, dataset
and manifest. These checks enforce declared membership; they cannot prove that
a physical executor applied the declared scenario. Phase 3 execution evidence
must establish that separately. One object per category supports only that
particular held object, not broad category-wide generalization.

## Fixed comparison budget

Five model names share exactly the same eligible rows, folds and three training
seeds `(0, 1, 2)`:

1. `training_prevalence`: constant success probability from training labels.
2. `commands_only_logistic`: existing last/mean/population-SD/last-minus-first
   summaries of applied command histories, preserving joint ordering.
3. `summary_logistic`: the same existing four summaries of all three fields.
4. `full_history_logistic`: flattened time-major close history with all fields.
5. `small_gru`: optional CPU Torch, one GRU layer and a single linear output,
   using the last recurrent hidden state over the same full history.

Logistic regularization candidates are `(0.001, 0.01, 0.1, 1.0)`, with at most
400 L-BFGS iterations each. A candidate must converge. Selection minimizes
**validation log loss only**; exact ties choose stronger regularization.
The logistic solver and prevalence model are deterministic; their three seed
records are repetitions, not three independent experiments or independent data.

GRU candidates are `(hidden=16, lr=0.001)`, `(16, 0.003)`, `(32, 0.001)` and
`(32, 0.003)`, each fixed at 50 full-batch Adam epochs, weight decay 0.0001,
without dropout, epoch tuning or test-driven stopping. Each candidate is seeded
and uses deterministic Torch algorithms on CPU. Selection minimizes validation
log loss; exact ties choose the earlier declared candidate. Missing Torch or
GRU disabled before fitting is an explicit skip, without automatic dependency
installation or a substitute model. Candidate/runtime errors are reported,
never converted to completed results. Each model/seed has at most four tuning
candidates. Settings can be changed only in a separately named, pre-frozen
comparison protocol; `TrainingSpec` rejects a larger grid.

Logistic normalization fits training rows only, with the existing 1e-8 active
column rule. GRU normalization fits per channel using all training-prefix
timepoints only. Validation/test statistics do not enter normalization. If all
GRU training channels are constant, the network receives one declared zero
channel, permitting a bias-only fit without observing held-out variability.
No training or normalization refit occurs after model selection.

The fixed decision is a failure warning if `1-p_success >= 0.5`, including ties.
There is no threshold fitting, calibration fitting, or selection on test scores.
Calibration is evaluated using ten predeclared equal-width probability bins.

## Missing classes, artifacts and metrics

An empty train/validation/test role skips all models for that fold. Single-class
train or validation skips all fitting for the fold, explicitly recording the
reason and null metrics. A single-class **test** set still supports real warning
counts, log loss, Brier score and calibration. Its ranking status is
`skipped_single_class_test`; AUROC and average precision are null, and unavailable
class-conditioned rates are null. No artificial examples or fabricated scores
are created to fill a missing class.

Every completed model/seed saves `model.npz`, `normalizer.npz`, `model.json` and
`predictions.json` in a fresh output directory. GRU weights are numeric NumPy
arrays, without pickle. Prevalence saves an explicit empty normalizer because it
has no features. Metadata binds input contract, training/validation episode IDs,
candidate selection, selected-model and normalizer hashes; predictions have a
separate hash. Saved model readback must match direct predictions (1e-12 absolute
for logistic/prevalence; 1e-7 for GRU). The inference function accepts only the
feature matrix and saved model directory, and opens no label/identity sources.

The report saves all folds, all named models and all seeds, including skips and
errors. Completed results include failure AUROC/AP where defined, failure recall
and precision, successful-task false-warning rate, full failure confusion counts,
accuracy, balanced accuracy, log loss, Brier score, fixed-bin calibration and ECE.
Fold episode IDs and row-membership hashes make same-row comparisons reviewable.
No seed variability is described as a confidence interval over unseen objects.
Bootstrap/categorical uncertainty and deployment thresholds need a separate
declared analysis if later requested.

## Explicit execution entry point

After collection, call the module's `prepare-training-protocol` subcommand with
`--dataset`, `--schedule`, `--folds`, `--attempt-binding` and a fresh `--protocol`
JSON path. This writes `hug_recorded_training_protocol_v1` without fitting. It
binds the exact `TrainingSpec`, input contract, dataset/manifest/schedule/folds/
binding canonical and file hashes, in-memory feature/label/fold digests, training
dependency source hashes and Python/NumPy/SciPy/optional-Torch runtime versions.
The declaration must be committed along with all bound source files **before**
fitting. Run authorization remains a separate root-owned phase-order gate.

Then call the `train` subcommand with the same inputs/protocol plus a fresh
`--output`. If GRU is disabled, `--without-gru` must be passed at both preparation
and fitting; disagreement is refused. Before any fit or output creation, code
checks exact structured settings, hashes of current input files, unchanged
loaded arrays/fold memberships, source hashes and runtime. Git must show the
protocol and all bound source files tracked and identical to committed HEAD.
Untracked, staged-only or modified protocol/source files are refused. Other
unrelated dirty project files are preserved and do not block this check.

The same checks run after fitting; a changed bound input/source/protocol or Git
HEAD prevents publication of a completed report. Commit/source receipts and
all actual hashes are saved. Git chronological freezing establishes that the
declaration was committed before this fitting command; it cannot establish that
no one previously inspected outcomes. Category taxonomy, schedule and comparison
decisions must still be honestly predeclared by the phase process. Merely
importing this module fits nothing. No cloud provisioning or new physics
collection happens here.

`tests/test_recorded_training.py` blocks every physics stepping API and covers
strict adapter linkage, excluded-attempt accounting, identity/fold tampering,
scenario hash enforcement, budgets, train-only normalization, deterministic
seed repetitions, numeric model persistence/readback, explicit class-degenerate
handling, artifact tampering and tiny optional-GRU fitting. They also check
non-fitting protocol preparation, declaration overwrite refusal, spec/source/
input drift, in-memory mutation and uncommitted-protocol refusal before fitting.
Manufactured fitting tests replace only the chronological Git-commit receipt;
all structured/hash checks run unchanged. They create no Git commits. These are
software tests, not the Phase 4 scientific result.
