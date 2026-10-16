# Native HUG / MANOSim evidence handoff

Status: 2026-10-16, 00:55 Eastern. Build evidence only; manuscript handled outside
this repository. No trained native predictor or hardware result is claimed.

## Completed and independently verified

- Source/native hand and camera mapping passes all 135 object-pilot slots.
- Bounded compensated physics: 12 fixtures and six timestep comparisons pass;
  worst gravity/settled-approach wrist errors approximately 1.26/3.65 micrometres.
  Simultaneous index/middle contact reactions remain positive throughout assessed
  tails. This is component evidence, not a universal physical-fidelity proof.
- Fixed-transform exploratory correspondence diagnostics over 60 original views:
  median silhouette recall 98.45%, median IoU 81.84%, median per-view absolute
  ray-depth residual 8.39 mm. Original object/camera poses and meshes are unchanged.
  Stereo/reconstruction error is not bounded by 1 mm depth encoding precision.
- The original all32-point admission rule falsely treated encoding/pixel error as
  total sensor/reconstruction error. V1/v2 refusals are preserved. V3 explicitly
  studies the original authored-asset simulation domain, retaining RGBD mismatch
  diagnostically. It does not establish calibrated physical scene correspondence.
- V3 verifies compiled visual/collision vertex-frame transfer within the existing
  5 micrometre numerical reconstruction gate, plus original mass, inertia, COM,
  camera and native hand identities.

## Actual object task pilot v3

Prospective denominator: 30 objects, five authored geometry categories by three
authored size bins, two objects per cell, two captures and two HUG seeds. There
are 120 primary and 15 matched timestep slots.

Actual execution: 45 model/data pairs, 283 static forwards, 270 resets, 89,915
integration steps and 437,953 contact-force reads. The transaction completed.

Primary outcomes: 13 completed failures (8 failed acquisition, 2 approach
collision, 2 no contact, 1 hold slip); 107 null quarantines (49 reset support
stability, 36 penetration, 22 preclose tracking). All 15 matched slots quarantine
(6 reset stability, 4 penetration, 5 preclose). A quarantine is not a training
negative. There are zero completed successes. Eleven valid pre-lift feature
prefixes remain after excluding failures already observed at the fixed cutoff.
All eleven are future failures, so no discriminative predictor comparison is valid.

The raw record remains simulation-conditional and uses the fixed task/controller
and operating envelope. Matched task numerical readiness is not established
because all matched attempts quarantined. No dataset-wide task qualification or
scientific training admission is asserted.

## Evidence locations

- `research_artifacts/manosim_native_object_pilot_2026-10-16_v3/protocol.json`
  SHA256 `7fb1f60855120139d8b1f3859361032beeb4b793b8d247ddb833838bb96a9480`;
  experimental freeze `40f34e3d3a2867a7ad9772e10a2bdde2c9c752b2`.
- V3 `run_v1/report.json`, SHA256
  `882ea983e92ad1d83106b036d7d946b21250aadab89399800ee0e7a85bf7d0ab`.
- `research_artifacts/manosim_native_pilot_cohort_audit_2026-10-16_v3/independent_readback.json`,
  SHA256 `d983e8d48b84d1962faaf1259c8a35ac89b0129b79351d4e2d0fb203a62e6f2b`.
- V3 `run_v1/independent_readback.json`, SHA256
  `f693ed8352c0fe5a06649d48f0ba35a68ec8d906a66033cdcb82ab63c25a0cb4`.
  This first saved-only reader receipt preserves NumPy BLAS warning history;
  the experiment itself did not rerun or change. A clean scalar/einsum-equivalent
  reader receipt is saved separately as `run_v1/independent_readback_v2.json`,
  SHA256 `0e7582bea4b689c5a49448c55600e2a699d7de74fb39cfc1d3eb4d679f3f78ca`.
  It verifies all 45 compiled scenes, 218 contact witnesses and the original
  bindings/artifacts without warnings. Source mesh transfer error is at most
  15.3 nm; maximum contact-Jacobian discrepancy is 4.16e-17.
- Full result/provenance history: `docs/RESULTS_LOG.md`.

## Next implementation

V4 is being prepared to settle the object passively before the task, preserve its
measured state, and re-express the HUG grasp using the measured rigid object delta
without changing local finger poses. A standoff approach will stop before contact;
translation and finger closure then proceed together. Existing numerical/contact
limits remain unchanged. These are prospective repairs, not measured improvements.
Training is authorized immediately after independently qualified execution and
two-class support in grouped training/validation/test roles; neither is available
from v3. The implemented candidates are prevalence, logistic regression, MLP and
GRU with 20 pre-lift pose/velocity/effective-command samples and no future features.
