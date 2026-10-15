# Retrospective HUG object holdouts — 15 October 2026

New development analysis, frozen at commit `7b41f14` before fitting. Original
1,040 and fresh 2,080 datasets remain unchanged. Their input/label/scorer
contracts match. All corrected-controller data are excluded. Results concern
four fixed geometries, 12 captures and 48 proposal families, not 3,120 independent
environments. These outcomes were previously inspected. The new fits do not
change the original frozen-model results or qualify a fresh mapping pipeline.

## Reproduce

Use the command in `docs/HUG_RETROSPECTIVE_EVALUATION_V1.md` with a fresh output
directory. Run `PYTHONPATH=src python scripts/verify_historical_prediction.py
--output /path/to/new_readback.json` to re-score immutable historical models
without optimization. Source snapshots, payload/protocol hashes and runtime
versions are saved alongside the report. No new simulation was collected.

## Full-model held-geometry results

- Apple: 1,040 attempts, AUROC 0.823838; false warnings 30/210 successes,
  missed failures 351/830. Commands-only AUROC 0.656546.
- Milk carton: 520 attempts, AUROC 0.834486; false warnings 86/245,
  misses 36/275. Commands-only AUROC 0.658427.
- Cube: 1,040 attempts, AUROC 0.872925; false warnings 471/648,
  misses 15/392. Commands-only AUROC 0.764342.
- Mug: 520 failures, zero successes, AUROC and false-warning rate undefined;
  290 warnings and 230 misses. These approach collisions occurred before the
  prefix cutoff, so this is not advance-warning performance.

`report.json` contains all metrics, calibration bins and per-fold tuning choices;
`predictions.csv` and partition memberships allow independent verification.
An independent review recomputed all 12 AUROCs/confusion counts directly from
CSV and checked all geometry/capture/proposal memberships against manifests.

## Uncertainty limits

The bootstrap samples whole proposal families, but four proposals share each
capture. It does not account for that shared-capture dependence. The report's
historical field name `independent_groups` counts unique proposal families,
not independently sampled environments; do not interpret it as such. The
intervals are conditional development diagnostics, not confidence intervals
for new captures or objects. Held folds have only two or four captures. No
interval is used as a manuscript population claim. Single-class resamples
retain undefined AUROC/false-alarm quantities rather than manufacturing values.

## Historical read-back

`historical_readback.json` verifies saved weights and dataset hashes and
reproduces original test AUROC 0.852253/0.641943 and frozen fresh familiar/new
AUROC 0.917319/0.558765. No optimizer runs. New retrospective fitting is a
separate analysis, never an improved score for those original frozen models.
