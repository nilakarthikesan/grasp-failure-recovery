# Mug table clearance evidence

This bundle documents a controller correction, with 27 small development checks.
It contains no new model predictions or training dataset.

Read the [step-by-step guide](../../docs/MUG_TABLE_CLEARANCE.md) and watch
[the before/after replay](mug_before_after.mp4). The same mug proposal uses zero
noise on both sides. `examples/before_v2` failed with a forearm approach collision;
`examples/after_v3` completed the grasp, lift and hold.

The final [summary](summary.json) has 26 completed trials with no table contact:
22 successes, four failed acquisitions, and one separately recorded runtime
rejection. The [independent audit](final_dynamic_replay_audit.json) reproduces all
completed physics steps and checks that the rejected little finger was still
0.946 mm above the table. All 27 planned trials are accounted for.
The [rejection reproduction check](rejection_reproduction_check.json) verifies
all 20 rejection invariants independently.

`protocol.json` records the new wrist planner and 1 mm runtime guard.
`original_contact_trace.json` and `original_goal_clearance.json` record the
original eight mug proposals. `rejected_trial` preserves the unlabeled
diagnostic state. The two examples include commands, states and initial scene
provenance for inspection. Raw MuJoCo models and third-party assets remain in
local `runs`; this compact bundle is not a self-contained physics replay.

`manifest.json` records raw and exported file hashes. Only the copied example
result files' video references were adjusted for this bundle. Historical source
records are unchanged. Controller source was validated before committing;
its source hashes identify the exact correction applied over the recorded base
commit. The original scorer, model, normalization and alert threshold are preserved.

Run the existing viewer from the repository:

```bash
python -m grasp_failure_prediction.evaluation.result_viewer \
  --runs-root research_artifacts/mug_table_clearance_v1/examples \
  --port 8771 --no-open
```

Open [the local viewer](http://127.0.0.1:8771/) while that server is running.
