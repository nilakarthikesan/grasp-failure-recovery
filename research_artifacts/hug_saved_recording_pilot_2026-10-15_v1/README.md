# Saved-command HUG recording pilot — 15 October 2026 v1

A bounded live validation of HUG-derived saved Shadow commands through the
current source-accurate simulator, versioned recorder, physical scorer, causal
exporter and unchanged saved predictors. This is a development pipeline check,
not new-input generalization or a training result. No fresh HUG inference or
retarget optimization occurred. Native MANO remains a static reference.

## Frozen procedure and engineering verdict

Protocol/source were frozen at `11cd155`; root inspected the saved32 scene gate,
all32 camera checks and current16-case static preflight before committing the
exact20 authorization at `6ccbd11`. The protocol's SHA-256 is
`baa34141b24dcc18bf6011f203726e452722b8ea1c3c67ec87ce6eeb58f665bf`.
No source/input binding changed, no criterion was loosened and no attempt was
replaced. Existing Shadow `fixed_grasp_lift_v2` executes every even index in the
original32 inventory, followed by repeats of0/8/16/24. Command/sensor noise is
zero. Original data remain unchanged and unpooled.

**Scoped PASS:** all20 attempts complete once, with65,200 physics steps, zero
refusals/errors/NOT_RUN. Each has163 control endpoints. Before/after every
substep, the driver checks copied MuJoCo warning counts, full-state/control
finiteness and time progress. Fixed step/attempt budgets are enforced. Tracking
is reported, not accepted against an invented physical-error threshold.

## Counts and tracking

-20 attempts =8 successes,7 failed acquisitions,5 approach collisions.
-Cube4/5 successes; apple0/5; milk4/5; mug0/5 (all approach collisions).
-The16 unique commands alone have6 successes,6 failed acquisitions and4 approach
 collisions. Four extra repeats are not independent situations.
-All4 repeats exactly match scored outcomes and recorded numeric/categorical
 observables/diagnostics. Maximum recorded state/target/time difference is0.
 This covers control endpoints and full-qpos/qvel diagnostics, not unrecorded
 physics-substep trajectories.
-Pooled joint tracking RMSE: approach0.009449, closing0.039463,
 lift0.059766 and hold0.057929 rad. These compare achieved positions with final
 effective clipped interval targets and are descriptive.

Mug contact diagnostics occur at0.438–0.998 s, before the2.96 s cutoff. All5
remain physical failures in the20-outcome denominator but are excluded from
forecasting inputs. The new portable dataset has15 ×1440 float32 features,
8 successes/7 failures,12 proposal groups,6 captures and3 geometries. It has no
assigned train/validation/test split. All repeated proposal variants must stay
in the same future partition.

## Model compatibility and independent checks

Both immutable saved models accept the exact joint ordering,20-step history,
units and time-major [qpos,qvel,applied-target] contract, summarized to288/96
features. Their scores are a **compatibility smoke check on reused development
inputs**, never a new benchmark or improved baseline. No model is trained,
selected or calibrated here. In particular, the tiny reused-prefix AUROC in
`model_input_smoke.json` must not be presented as a generalization improvement.

An independent scalar checker recomputes all20 task outcomes from raw recorded
full states/phase/contact diagnostics, reconstructs X exactly from allowlisted
closing channels, checks all4 repeats, and reproduces saved-weight probabilities
within1.12e-16 without any forward/integration calls. Its receipt is
`independent_scalar_receipt.json`; source is `independent_scalar_check.py`.
All454 frozen bindings and80 raw file hashes were checked. Historical comparison
finds0/16 success-label changes and0/16 failure-type changes on corresponding
original zero-noise attempts; this finite agreement is not general trajectory
or outcome equivalence under importer changes.

Final focused regression:246 passed,2 optional skips,2 explicit free-fall
exclusions, with real integration APIs blocked (zero calls). See
`final_static_regression.json` and its log. The live pilot's65,200 steps are a
separate, explicitly authorized operation.

## Files and reproduction

Raw20 model/observable/diagnostic/metadata bundles total about145 MB under the
new ignored `runs/hug_saved_recording_pilot_2026-10-15_v1/` directory. They are
hash-referenced, not redistributed. This archive contains scalar reports,
protocols, source snapshots, one portable predictor NPZ and derived manifests;
no compiled scenes, original native payloads, licensed MANO models or checkpoints.
Pre-bootstrap unexecuted drafts are retained as preparation history.

From the canonical checkout, inspect `scripts/run_hug_saved_recording_pilot.py
--help`. To reproduce, prepare/freeze a NEW evidence directory and NEW raw path:

```sh
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python  scripts/run_hug_saved_recording_pilot.py prepare  --project /Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction  --inputs-project /Users/nilakarthikesan/grasp-failure-prediction  --inventory research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json  --output /absolute/path/to/NEW_EVIDENCE --raw-output /absolute/path/to/NEW_RAW
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python  scripts/run_hug_saved_recording_pilot.py freeze --output /absolute/path/to/NEW_EVIDENCE
```

Then inspect/commit the new protocol and provide a new hash-bound authorization
before `run --output NEW_EVIDENCE --authorization NEW_AUTHORIZATION`. The current
authorization cannot be reused with a different protocol. Existing private
saved inputs/assets/runtime are required. Never overwrite this evidence or
resume/extend its completed quota. The driver loads base NumPy/MuJoCo before
appending its declared existing dependencies; no package installation is needed.
