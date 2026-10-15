# Existing-cell radial control at the new canonical palm anchors

15 October 2026, America/New_York. This separate static development study binds
the completed source-only canonical calibration and six-case candidate output
before computing any new-anchor radial margin. It changes no calibration,
object, hand surface, source proposal, threshold or production mapper.

## Measured result

Both saved Mug grasps have positive-area original native palm cells that remain
mapped FAR for EVERY SO(3) palm orientation at the new canonical fixed anchor,
conditional on the unchanged arithmetic budgets. Mug A has three positive
reported cell rows, with maximum strict margin **7.135419421 mm**. Mug B has two,
with maximum strict margin **10.091851734 mm**. These rows span two existing
sampling resolutions and can overlap; do not sum their areas or treat them as
independent physical grasps. Every one of the seven original rows, including
nonpositive margins, remains in the report.

The source-only calibration shifts the anchor by approximately 8.668706 mm from
the original native wrist anchor. Its exact C matrix remains byte-identical to
the first exported source-only transform; this control neither chooses nor
retunes it after observing these margins. The new anchor therefore does not
remove the strict same-part palm obstruction for these saved cases.

All six candidate source-gate refusals are retained. Although their numerical,
physical-bound, solver-status and mean-tip gates pass, 21 of 30 exact ORIGINAL
per-tip caps fail across the six cases; no full surface-audit eligibility is
granted. The radial calculation can inspect a refused candidate's declared
anchor as a diagnostic over all orientations. It does not approve that pose,
waive its caps or turn a mapping refusal into an outcome/failure label.

## What the certificate covers

The exact eight compiled Shadow palm boxes remain fixed in the palm body frame.
Every box-surface point lies within the all-corner maximum radius about that
origin, regardless of orientation. The full object-cell radius is subtracted
from its centroid-to-anchor distance. A native DEFINITE_NEAR positive-area cell
whose resulting radial shell gap exceeds its original band refutes strict
same-part proximity preservation for that ENTIRE cell at the new fixed anchor.
The primary bounds retain 100 nm shell padding, 100 nm inherited distance
allowance and 100 nm transform allowance separately. They are conditional
engineering numeric certificates, not formal directed-rounding proofs.

The result applies to these two saved development grasps, the new fixed
canonical anchors and the same literal individual box surfaces. It does not
cover arbitrary translations, changed hand geometry, other placement rules,
all HUG proposals, exterior-union equivalence, failure-label preservation or
predictor adequacy. A nonpositive radial result would be UNRESOLVED, not PASS.
The original old-anchor proofs and all original strict surface audits remain
separate immutable history.

## Frozen provenance and execution

Protocol SHA:
`e02f6ef5e4715968631aa7ec01f10d6d3afe0fb99991bafbda9cdaf71411df74`.
Script SHA:
`0008d8fa1258819e1aa8f43795a8ffda4957418e65fe4a2a43ad0b2ef048d3a8`.
The protocol binds 127 inputs, including exact calibration JSON SHA
`2c07c84ec8699e9e4214311a431401b0f19724070d27aeeb1220f6270bf2eed5`,
candidate SHA
`84496804c00267024af56f3364457943696db8481b9c1853bca2c93c6c824b9d`,
the unchanged source-C float64 hash, original cell/query/native geometry closure
and prior independent actual-box/native-cell audit. Candidate anchors are
independently recomputed from original saved camera/native wrist transforms and
C.translation before world-to-object conversion.

Root and the primary independent auditor reviewed the frozen source transform
and shell algebra before execution. Independent numerical verification of the
completed report is recorded separately. There are no new object-to-hand
geometry queries, inference, simulation steps, eval episodes, outcome labels,
training examples or production edits. Raw licensed assets and full query
caches are read only and are not copied into this bundle.
