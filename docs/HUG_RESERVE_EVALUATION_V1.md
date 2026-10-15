# Frozen-model reserve evaluation v1 — prepared implementation

Prepared 15 October 2026. This is a separate prospective evaluation implemented
in `scripts/evaluate_hug_reserve_models.py`. No real reserve protocol, HUG
inference, task labels or model predictions were produced during implementation.
Tests use manufactured artifacts and prediction callbacks; all real physics and
fitting APIs are blocked. Main collection/training code and historical data are
not edited by this work.

## Fixed scope and model selection

The two reserve objects are `hug_bell_pepper` and `hug_coffee_cup`, each with the
two existing qualified stationary captures. Bear figure failed qualification
and is excluded; it is not retried or silently replaced. These are two new
geometry/capture identities, not evidence of broad category generalization.

Preparation requires an existing completed `recorded_training` report. It binds
**every completed** model bundle across all available folds/seeds from all five
named kinds: prevalence, commands-only logistic, summary logistic, full-history
logistic and optional small GRU. Skipped/error entries are preserved explicitly.
Each fold must account for every model/seed, preventing a hand-picked subset.
The primary model kind is fixed as `summary_logistic`; its available fold/seed
bundles are all reported individually. Other bundles are diagnostics. No
reserve-based winning fold, seed or model is selected, and no ensemble is fitted.

If zero completed bundles exist, or the declared primary has no completed bundle,
preparation is explicitly **BLOCKED before any inference or task labels**.
Model report, selection, metadata, numeric weights, normalizer and development
prediction hashes are checked and bound. Preparation also requires the original
`--training-protocol` path. Its declaration, specification, input contract and
input bindings must match the fitted report. The exact protocol blob and every
source dependency declared there must match both the immutable fitting commit
recorded in the report and the current source bytes. An unchanged trainer cannot
admit changed model, exporter, recorder or split logic with old weights. These
checks repeat before fresh generation, collection and prediction. The reserve's object/geometry identities
must be absent from the complete hash-bound development scenario schedule.

The training report's uniquely hash-bound dataset manifest must also declare the
exact physical target cohort: `shadow_hand_right`, `hug_grasp_approach_v1`,
`hug_grasp_approach_task_outcome_v1`, and the current scorer/success-rule/control-
timestep digest used by the collector. Feature shape alone cannot admit a model
trained under another hand, controller or scoring rule. The reserve export and
one-pass evaluation recheck that same cohort as well as the prefix contract.

The trained input contract and fixed failure-warning rule are unchanged:
warning if `1-p_success >= 0.5`, including ties. There is no refitting,
normalization fitting, calibration fitting, hyperparameter or threshold tuning.

## Qualification and generation declaration

Bell pepper uses the v1 qualification's two passing captures; the v1 overall
report also retains bear failures. The v1 report literally recorded its freeze
reference as `HEAD`. That unresolved historical reference is preserved and
identified as such, rather than retroactively inventing a concrete commit.
The exact qualification protocol/report, case, asset closure and capture bytes
are bound again in the new prospective reserve declaration.

Coffee cup requires its passing v2 qualification with the concrete freeze
`464f08fd77a90419d17a75a420ae80fd5b16895c`. Both objects require exactly two
`PASS_STATIONARY_CAPTURE` rows with null grasp outcomes and zero prior inference,
grasp-label and fitting calls in their qualification reports. Original captures
must have no proposal/inference output. All qualified raw captures remain
unchanged; the existing provider copies them into fresh per-proposal directories.

The finite native generation quota is **eight**: seeds `(9000, 9001)` for each
of the four captures. Capture frames are reused; no new capture/settling physics
is authorized by this protocol. Native HUG loaded-model state must exactly match
the complete state hash recorded in the frozen main-study generation, including
external encoder parameters. The declared dependency paths are forwarded
explicitly to the provider's inference subprocess. Generation is surrounded by
a zero-physics budget and uses the collector's current canonical command/native
mapping/frame validation. Mapping refusals remain in all generation/schedule
denominators, with null task outcomes and no replacement solve.

The reserve declaration binds the completed main preparation/generation, current
helper and predictor source closure, runtime, model bundles, qualification
inputs, reserve geometry assets and every source capture file. Freeze the
declaration and commit it and bound source files before any fresh reserve
inference. Runtime refuses untracked, staged-only or changed bound source and
declaration bytes. Other unrelated dirty project work is preserved.

## Exactly 32 task attempts

Each of eight native generation slots has the full four combinations of mass
multipliers `(0.75, 1.25)` and sliding friction `(0.5, 1.0)`. Table offset and
sensor noise are both zero. This is 16 scheduled attempts per reserve object,
32 total, with no task-outcome selection. Actual native proposal hashes replace
synthetic family identities; repeats/variants retain their proposal and capture
bindings. There are no training folds for this reserve dataset.

The separate generated ledger/schedule is frozen and committed before task
labels. An explicit authorization must bind exact protocol/generation hashes,
stage and quotas. Task execution reuses only the collector's qualified nominal
capture admission, declared physical scenario intervention, grasp-derived
approach, guarded runner and strict recorder helpers. It does not rewrite the
main study or route these rows through its training split logic.

All accepted and rejected mapped proposals execute their declared scenarios.
No six-gravity screen is run for reserve collection; this is predeclared and
recorded as `NOT_RUN_RESERVE_UNFILTERED_PREDECLARED`. Unmapped proposals and
unreachable pregrasps remain null-outcome refusals. Every task has a hard bound
of 3,260 physics steps, with a total bound of 104,320; successful completion
requires the same exact 163 control endpoints. The collector's 2 ms per-step
warning, qpos/qvel/qacc/ctrl finite and time-increment guards remain active.
Mass/inertia and both hand/object sliding friction are measured and recorded.

Unexpected setup/numerical/runtime/publication failures hard-stop later attempts
as `NOT_RUN`, retaining all 32 rows with null outcomes where no physical result
exists. There are no retries, command repairs or quota replacements. Completed
physical task outcomes and early exclusions remain independent of mapping,
filter and setup status. An incomplete collection cannot produce a reserve
performance report.

## Dataset and one-pass evaluation

`hug_reserve_collection_v1` and `hug_reserve_dataset_v1` are new reserve ledgers
and dataset wrappers in fresh directories. The wrapper contains a separate
strict `hug_recorded_prefix_dataset_v1` export so the trained feature contract
remains exactly compatible: named measured positions, measured velocities and
actually applied targets in the causal close-only prefix. No old dataset is
appended or pooled. Refusals, runtime errors and at/before-cutoff/approach failures
remain in denominators and are excluded from predictor rows by the exporter.

After the collection ledger is committed, the evaluator verifies protocol,
generation, raw artifacts, dataset hashes, all 32 scheduled identities, per-row
scenario hashes/groups and the exact trained input contract. It calls public
`predict_saved_model()` **once per frozen bundle on the same complete eligible
X matrix**. Per-object metrics are derived from those same probabilities without
additional prediction calls. A started marker prevents repeating a partial
evaluation to choose a more favorable result. Prediction failures preserve an
explicit error row and all later bundles as `NOT_RUN`, with an overall
`HARD_STOP` rather than silently omitting failed models. No fitting API is called.

All bundle predictions are saved with hashes. Reports contain physical and
included/excluded denominators, confusion/warning counts, failure recall,
successful-task false-warning rate, AUROC/AP where defined, log loss, Brier score,
and ten fixed equal-width calibration bins/ECE. On a single-class reserve subset,
ranking metrics are null and marked `skipped_single_class`; actual warning counts
and proper scoring rules remain measured. Zero eligible prefixes produces an
explicit `NO_ELIGIBLE_PREFIXES` report, not artificial examples or scores.

No bundle is selected based on reserve performance, and fold/seed repetitions
are not treated as independent held-out objects. Any later model change or
repair requires a new protocol and independently unused evaluation data.

## Entry points and verification

Commands are `prepare`, `freeze --stage protocol`, `generate`,
`freeze --stage generation`, `collect` and `evaluate`. Preparation requires the
main preparation path, real model report, original committed training protocol,
both qualification protocol/raw capture paths and fresh evidence/raw destinations. Freeze and commit the exact prospective
declaration before generation, the generated ledger before collection, and the
collection ledger before evaluation. Generate/collect authorization schema is
`hug_reserve_authorization_v1`; quotas are 8 and 32 respectively. Root owns those
commits and run authorizations. The prepare command performs only reads/hashes
and writes the new draft declaration; it cannot provision cloud resources.

`tests/test_hug_reserve_evaluation.py` uses only manufactured source/capture/model
artifacts. Git fitting blobs are represented by immutable manufactured byte
receipts; no real model fitting or Git commit occurs. It covers zero-model
blocking, all-bundle inclusion, full fitted-source closure and Git-receipt drift,
same-prefix/different-cohort refusal, hash and threshold drift, fixed reserve admission, exact 8/32 quotas, explicit runtime forwarding,
refusal retention, inference/task hard stops, null labels, class degeneracy,
same-X one-call-per-bundle evaluation and overwrite refusal. These tests establish
software contracts, not a reserve generalization result.

Verification on 15 October 2026: 26 manufactured tests passed with real physics
and fit APIs blocked. The direct original-venv CLI help succeeded. A separate
read-only check of the now-completed development report admitted all 60 frozen
bundles, matched all five fitting source files and the original protocol Git blob
at fitting commit `064bfb324ad9b39c96b87696a887e387fe962c75`, and matched the
physical cohort digest `sha256:806de09593722ec52ee3aba5c8b718c31bc05e18362a1da35ed43d9c9761b56c`.
That check did not prepare a reserve declaration or perform reserve inference,
physics, fitting or prediction. Runtime verifies bound bytes and bootstraps the
canonical source/dependency paths before importing registry/Pydantic modules;
a regression covers that ordering on freeze and protocol-read paths.
