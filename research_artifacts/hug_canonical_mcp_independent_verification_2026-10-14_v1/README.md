# Independent canonical MCP calibration and saved-pose verification

Executed once on 2026-10-15 after parent source review and an immutable
pre-execution closure of 135 input/source/implementation hashes. This is an
independent numerical reconstruction of the completed canonical calibration
diagnostic; it does not repeat its optimization or certify mapping acceptance.

The canonical frames and direction of C were independently reconstructed from
the serialized source-only neutral descriptors and source URDF MCP origins. C
maps neutral target-palm coordinates into source MANO wrist coordinates. Its
float64 bytes remain identical to the calibration exported before the primary
V2 joint-order abort. The original primary V1/V2 aborts and V3 correction are
preserved in the separate primary calibration archive.

The MANO neutral forward was deliberately not repeated. Its descriptor is
inherited from the primary frozen template forward, whose asset and source
bytes are bound. This result independently checks the frame algebra, XML
anatomical correspondences, source/compiled neutral FK and all six saved-case
native WORLD targets, source FK, floating root and q-dependent palm placement.
The fixed source base-to-forearm transform is reconstructed from the XML rigid
chain, so a compiler-fused fixed forearm body cannot be silently confused with
the captured floating forearm frame.

All six numerical reconstruction gates pass, under the primary frozen numeric
allowances. Every original complete saved scene hash and original fingertip cap
was independently recovered. Explicit name/value correspondence preserves the
old joint ordering; the five anatomical caps are not reordered or rebased.
Physical joint limits, raw optimizer margin, all 120 retained solver statuses
and finite costs, and the strict 10 mm mean fingertip guard pass. However only
9/30 original individual fingertip caps pass: 21 fail, and **0/6 candidates pass
the combined source gate**. All six refused records remain in the report.

Native scientific WORLD targets are reconstructed directly from their saved
camera coordinates and original camera-to-world matrix. C affects only target
hand placement/reference vectors. No second operator rotation is introduced,
and the nonzero C translation remains in the reference vectors. A separate
floating zero-root FK independently reconstructs the required captured-scene
root; the actual palm rotation remains q-dependent. The calibrated palm anchor
is independently reconstructed rather than passing an old wrist-to-palm gate
by disabling it.

No full-surface geometry audit was authorized or executed for these refused
candidates. The nine historical original-pose surface results are not reused
as measurements of these new canonical poses. The separate new-anchor radial
control and its independent verification refute both saved Mug fixed anchors
conditionally for any palm orientation; see the sibling artifact
`hug_calibrated_anchor_radial_independent_verification_2026-10-15_v1`.

This verifies numerical implementation and preserves explicit refusals; it does
not establish known anatomical ground truth, full contact/normal equivalence,
failure-label preservation, held-out generalization or a complete HUG mapping.
The anatomy proxy remains a proposed new mapping convention.

No native HUG proposal generation, MANO neutral rerun, optimizer replay,
surface query, physics step, evaluation episode/data, training or production
change occurred. Three deliberately refused integration-API self-tests are
separately counted; actual integration calls are zero.

The protocol retains original frozen absolute paths. Archive cross-bindings
provide verified byte-identical persisted locations without rewriting those
historical protocol bytes. The manifest covers every archived local file.
