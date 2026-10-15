# Recorded-cohort cutoff sensitivity v1 — prepared diagnostic

Prepared 15 October 2026. This additive implementation is not a new fit, simulation
study or slip-onset model. No real alternate prefixes or predictions are produced
during preparation. Historical fixed-cutoff verification and retrospective LOO
sources/results remain unchanged; those older cohorts do not supply a multiple-
cutoff audit for the new strict recording format.

The grid is fixed before replay: **2.96, 2.76, 2.56 and 2.36 seconds**, in that
order. Each window retains exactly 20 samples, 24 named joints, three allowed
fields (measured position, measured velocity, actually applied target), and 0.04 s
spacing. There is no padding, interpolation, shorter history, replacement grid,
new label, fitting, normalization fitting, threshold tuning or cutoff selection.
The fixed warning is `1-p_success >= 0.5`, including ties. Summary logistic is
primary; every completed original bundle remains an individually reported model.

Earlier times deliberately change `protocol_id` and `prediction_time_s` while
preserving the numerical feature layout. These are **new shifted-time input
protocols**, not deployment-equivalent tests of the fitted 2.96 s contract. The
2.96 s export must reproduce every original X/y/episode/group array exactly
before any model call. Baseline saved predictions also reproduce the original
held-test predictions within the already declared readback tolerances (1e-7 for
GRU and 1e-12 otherwise). No earlier-time success can select a replacement
cutoff, model, fold, seed or threshold.

Preparation binds the completed study ledger and all raw artifact hashes,
original strict dataset/schedule/folds/input bindings, all completed model
bundles, the immutable fitting protocol/commit/source closure, current cohort,
source dependencies, runtime and original test IDs. Each model's training and
validation identity hashes and saved test prediction IDs must match its original
fold partitions; roles cannot overlap. Models never receive training/validation
rows, newly eligible episodes outside their original held test IDs, or another
fold's test episodes.

After the protocol is frozen and committed, replay loads each strict raw
recording. Mapping/null/early-failure exclusions remain in the audit denominator.
The strict exporter's bound `_prefix` checker identifies only the two known
availability causes: insufficient samples or a window outside causal closing.
Missing cutoff endpoints, time alignment, malformed values, invalid hashes,
identity/cohort mismatch and unexpected exporter errors hard-stop the audit.
No invalid row is converted into a convenient exclusion. The wrapper records all
original attempts and every reason; the inner strict export contains the
available completed physical rows only. If none exist, that grid point remains
explicitly undefined and no fabricated dataset or prediction is created.

At each cutoff, each frozen bundle predicts only its original eligible held-test
IDs, once. Reported common-ID metrics use the intersection of those original test
IDs available at every grid point and reuse the same probabilities; no second
prediction or fit occurs. This separates changing input timing from changing the
risk set. All-cutoff-empty intersections remain undefined. Counts and all-model/
all-grid NOT_RUN/ERROR rows are retained after faults. A started marker prevents
retry/overwrite selection of a favorable replay.

Lead distributions use only existing non-null `first_failure_time_s` values on
selected failures, minus the declared cutoff. They are explicitly **recorded
first-failure proxies**, not physical slip-onset truth. Missing failure times
remain missing; no terminal acquisition score is assigned an invented onset.
Per-cutoff distributions report known and missing proxy denominators. Threshold
metrics include actual warnings, missed failures, successful-task false warnings,
proper scores and fixed calibration bins. Single-class AUROC/AP remain null.
Fold/seed repetitions do not supply independent object-level evidence.

The original-venv CLI supports `prepare`, `freeze` and `run`. Preparation requires
`--project`, `--main-preparation`, `--training-protocol`, `--model-report`,
`--study-summary` and a fresh `--output`. Freeze writes `protocol.json`; root must
commit that declaration and bound source before `run --output ...`. Preparation
uses reads/hashes only. The explicit run is the only path to alternate strict
exports and fixed-model prediction. It has no physics, inference or fitting API.
Tests use manufactured recording/model artifacts with all physics and fit APIs
blocked. Their software checks establish no empirical cutoff-sensitivity result.

Verification receipt: 17 manufactured tests passed, independently repeated by a
peer, with zero integration/fitting calls. Direct original-venv CLI help passed.
A separate read-only check matched all 60 actual frozen bundles' original
training/validation identity hashes and original test prediction IDs across four
folds (test denominators 64/64/60/25). No alternate real prefix extraction,
export, prediction or cutoff result was produced. Independent review found no
remaining meaningful leakage, denominator or protocol-selection blocker.
