# Saved capsule fixture figures

This package reads a once-only capsule fixture report (explicitly identified v1, v2 or v3), its exact protocol, and creation-time artifact receipts. It imports no simulation, MANO, HUG, skin, FK or contact-query implementation. It does not replay the experiment or recalculate its qualification verdict. The actual experiment must finish and root must authorize saved-result rendering first.

`plot_saved_capsule.py` writes four PNG/PDF pairs and a manifest into a fresh output directory: tracking wrist error to the same-clock reference plus the recorded maximum sixteen-bone rotation error to the fixed saved target; measured downward gravity sag versus the declared `mg / 300 N/m` equilibrium and its residual; normal force and engine gap for active contacts only; and all recorded static gate outcomes with actual per-fixture status. The step-start force/frame timestamps and state endpoint timestamps stay distinct. Tail limits are shown only over the declared last0.1seconds. A plotted early ramp exceeding a tail limit does not alter the stored verdict.

All FAIL/HARD_STOP/NOT_RUN statuses are retained. Partial traces can be displayed only when their original creation-time artifact receipt exists, and are marked PARTIAL. Missing schedule receipts are reported as unknown rather than as successful or unexecuted fixtures. There are no grasp-success counts or task labels. Active-only contact gaps avoid presenting absent-contact zero placeholders as measured distances. The manifest binds source report/protocol/worker and all generated figures, and retains the experiment's scope flags.

After actual completion and explicit root authorization, use the canonical original Python, with no MuJoCo path bootstrap:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python research_artifacts/manosim_capsule_execution_figures_2026-10-15_v1/plot_saved_capsule.py --report research_artifacts/manosim_capsule_execution_2026-10-15_v1/run_v1/report.json --report-sha256 ACTUAL_REPORT_SHA --protocol research_artifacts/manosim_capsule_execution_2026-10-15_v1/protocol.json --output research_artifacts/manosim_capsule_execution_figures_2026-10-15_v1/render_v1
```

The plotter verifies the supplied report SHA, matching parent/worker/protocol/freeze identity, original write-time hashes, raw/portable worker equality, artifact containment and once-per-step clocks. It does not reverify the experiment's full historical dependency closure or replace its independent saved-array scientific readback. Neither the plotter nor the authored capsule experiment grants full MANO skin equivalence, human-oracle calibration, new collection or training admission.
