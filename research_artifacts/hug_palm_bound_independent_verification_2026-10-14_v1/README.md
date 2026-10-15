# Independent Mug B palm-bound geometry checks

These are two separately frozen static geometry verifications of existing
reported wrist-box midpoints. They do not rerun the subdivision search, select
new object queries, generate proposals or eval episodes, integrate physics,
label outcomes, train a predictor, or change the mapper.

The local verification (`protocol.json`, `report.json`) checks all **149**
covered leaves of the original q0 ±0.35-radian/source-URDF wrist domain.
The separate full-wrist verification (`full_wrist_v2/`) checks all **171**
leaves across both **full source-URDF wrist-joint limits**. The independent
protocol version is V2; it verifies the separate bound study's V4 result.
Both retain the original root orientation and native wrist/palm position.

Both checks pass. Independently evaluated direct analytic-box unsigned
distances differ from the bound reports by at most 1.2491e-16 m locally and
1.4572e-16 m across full wrist limits. Minimum recomputed strict margins are
28.7562066537 μm and 0.706247065674 μm respectively, after the inherited
100-nm distance and 100-nm transform budgets. These remain **conditional
engineering arithmetic bounds**, not formally rounded floating-point proofs.

The independent implementation reconstructs the original complete scene and
matches its float64 qpos hash to the preceding actual source/FK preflight.
Native wrist-to-palm placement, source-URDF/captured-scene FK and palm rotation
pass their unchanged numerical checks. It enumerates every active,
collision-compatible palm geom: all eight are exact boxes rigidly attached to
palm, with exactly the two normalized WRJ hinge ancestors. Box-corner radii are
computed independently from compiled halfsizes/local positions/quaternions.
The two original object cells, their radii/areas and native anatomical-palm
distances are reconstructed independently; both retain native-near intervals
under the stronger 100-nm allowance.

At each recorded midpoint, only the two wrist targets and the unique root
translation required to hold the original palm anchor vary. Fingers, root
quaternion, object state/world pose and zero time are checked. Direct
world-to-box distances use no bound-study distance callback or `a.geom_query`.
Native facet distances use an independent plane-projection/three-segment
calculation. Actual physics-step attempts are zero; each run records three
guard self-tests.

The continuous conclusion also relies on the separate motion-bound derivation
and complete partition review; finite midpoint agreement alone is not a
continuous proof. This is the literal union of individual box surfaces,
including buried interfaces—not a contact-force, margin, exposed-boundary or
failure-label certificate. The result applies to this saved Mug B proposal,
the same anatomy labels/boxes and fixed root rotation/palm anchor. It does not
prove impossibility for arbitrary free-root placements, alternate wrist
conventions, other hand geometry or all HUG proposals. No full HUG mapping
acceptance follows. Mug A's separate partial cover remains UNRESOLVED.

See `array_and_scene_provenance.json` for source arrays/hashes and
`manifest.json` for archive integrity. Original bound V1–V4 history and arithmetic
reviews remain in the separate reachability evidence bundle; no historical
protocol/result is rewritten here.
