# HUG geometry/reference validation — 14 October 2026

This is supporting static mapping evidence, using existing saved HUG proposals.
It creates no generator inference, physics rollout, eval episode/label, training
sample or predictor fit. Original production code, mapper/controller/colliders,
predictor and manuscript remain unchanged. Diagnostic derivatives and alternative
retargeted configurations are not new generated grasps or evaluation records.

Read [HUG findings](../../docs/HUG_MAPPING_CONFIRMATION.md) for the bounded
conclusion and [measurement/proposed-acceptance framework](../../docs/HUG_STATIC_VALIDATION_PROTOCOL.md)
for implemented versus pending criteria. Numerical reconstruction/frame/joint
checks are supported in tested scope. Cross-hand contact and failure-label
equivalence remain approximate and unconfirmed; no scientific pass gate is set.

## Evidence phases

- [Object geometry causes](geometry/README.md): three objects with paired saved groups; source
  scale/frame/compiled agreement, pinned asset provenance, local internal gap
  witnesses and exterior hull approximation; no production geometry correction.
- [Reference point/FK comparison](reference/README.md): 32 saved development
  proposals, four objects/eight observation groups/one MANO shape. Current
  vectors regenerate exactly. Released Dex Retargeting 0.5.0 Shadow DexPilot-style
  and fixed metric/no-filter sensitivity show tip versus middle-link tradeoffs.
- [Reference metadata correction](reference/corrections/README.md): preserve all
  original bytes; mislabeled physical-limit tables were captured-scene ranges.
  Verify all 128 targets against actual source URDF/clipping, without new solves.
- [Common-query visual surfaces](surfaces/README.md): six fixed seed-0 noncube
  cases; three methods, three bands and finer first-group repeats. Current whole
  unsigned 2 mm IoU is higher in all six pairs, with distinct per-finger tradeoffs.
- [Signed hand/object profiles](signed_profiles/README.md): same six fixed cases
  and previously computed targets. Original native surfaces already penetrate.
  Vertex counts, area profiles, target approximations and carriers remain separate.

The 32/six inspected cases are development evidence, not independent validation.
Thousands of quadrature sites do not create independent grasps. Proximity,
occupancy, conditional normals and sampled signed depth are different geometric
measurements. No low-penetration/lift-success acceptance or poor-case exclusion
is used. Lower tip error or higher overlap does not certify contact equivalence.

## Frozen versions, reproduction and verification

Subfolder manifests bind selected publication files; the parent manifest also
binds the correction supplement and protected hashes. Exact full reference and
signed reports are losslessly compressed, with uncompressed hashes verified.
Unexecuted/incomplete protocols remain versioned and are not pooled with final
metrics. Scientific acceptance margins and independent validation remain proposed.

Scripts require preserved original local scene/native/licensed assets, installed
runtime and earlier immutable audit bundles. No raw MANO asset/checkpoint, model,
scene binary, scratch mesh/query array or runtime cache is copied into this bundle.
Local V4 query derivatives are hash-bound in the surface protocol and stay outside
Git. Reproduce only into new temporary output directories. Each phase README
states its dependencies; frozen protocol files are copied rather than regenerated.
`prepare_local_sources.py` can restore the exact compressed diagnostic report
inputs at their required temporary paths, rejecting existing differing bytes.
Do that before reproducing signed profiles if the old temporary sources are gone.
`verify_publication.py` checks bound publication files, exact compressed reports,
protected source/model/paper hashes and current original input bindings. It does
not run optimizers or MuJoCo. Numeric/analytic sanity checks and independent
reviews found no blocker; this is not a scientific fidelity certificate.

Next: object-facing normal compatibility and spatial sampling uncertainty,
then externally justify margins and freeze an independent object/observation
validation design. Contact/outcome equivalence needs separate authorized evidence.
HUG stays active before other providers, collection and predictor training.
