# HUG orientation, uncertainty and scoped acceptance — 14 October 2026

This closes the authorized static checks on existing HUG grasps. No new grasp
inference, retarget optimization, physics rollout, eval record/label, predictor
fit or production change occurred. Original source/runtime/licensed assets,
nominal targets, geometry, predictor and manuscript remain unchanged.

[Findings](../../docs/HUG_MAPPING_CONFIRMATION.md) and [acceptance definition](../../docs/HUG_MAPPING_ACCEPTANCE.md)
separate three claims: numerical conversion passes within checked scope; strict
same-part surface relationships fail on all six detailed development cases
under explicit conditional numerical assumptions; physical contact, failure
labels and independent predictor adequacy remain unconfirmed or untested.
A failed mapping relationship is not a physical task failure label.

## Contents

- [Native contract](native_contract/README.md): extend saved-parameter MANO FK
  to all 32 existing proposals. Every 778-vertex/21-landmark/wrist-rotation
  export and 1,552 capped face table reproduces exactly. This is not HUG grasp
  generation. Camera reprojection remains the representative original cube.
- [Sampling uncertainty](sampling/README.md): same six fixed noncube cases and
  three declared finer repeats. Exact child cells and 1-Lipschitz distances
  produce conditional area/ratio intervals. All six cases retain definite
  same-part losses/additions; 178 auditable witness cells bind frozen criteria.
- [Object-facing orientation](orientation/README.md): valid/tied/buried/inside,
  same/different-part, lost/new and unavailable denominators stay explicit.
  Reliable sampled facing-state reversals are independently replayed. No
  continuous changed-normal area or physical contact-force certificate follows.
- [Frozen acceptance rule](acceptance/acceptance_rule.json), [explanation](acceptance/README.md)
  and [application](acceptance/acceptance_application.json): freeze before new
  reports, without selecting a mean IoU/tip/angle gate to make current HUG pass.
  Strict scoped preservation uses PASS/FAIL/UNRESOLVED. Approximate morphology
  conversion usefulness and failure-predictor adequacy are separate claims.

## Interpretation and provenance

The six cases are two existing seed-0 groups per apple/milk-box/mug, one MANO
shape. Nine quadrature resolutions are not independent proposals. Prior cube,
32-pose FK and original collision-approximation evidence stay separate. Distance
bounds apply to literal fixed anatomical/collider surface sets; individual
collider boundaries can include buried interfaces. Native carrier references
are unavailable and excluded from the six-part acceptance gate. Whole-hand
proximity diagnostics disclose their larger mapped collider domain.

The assumed 1/10/100 nm numerical allowances are arithmetic sensitivity budgets,
not formal global floating-point certificates. Real-arithmetic cell radii cover
entire visual triangles, whereas normal signs/weights describe sampled sites;
distance Lipschitz continuity cannot certify nearest-feature angular continuity.
Closed/oriented topology is checked but exhaustive self-intersection exclusion
is unavailable. No rule requires all native grasps to be good or lift successfully.

Strict zero changed resolved relationships defines the stated preservation
claim itself; it is not a universal paper threshold for useful retargeting.
Any later approximate adequacy budget needs its own external rationale and
independent validation. No such budget is invented from development scores.
The existing mapper remains implemented and approximate, without native-contact
or failure-label equivalence certification. All inspected cases remain development.

Subfolder manifests bind selected executable scripts, figures, reports, tests
and preserved protocol histories. Orientation full reports are losslessly
compressed. Incomplete/pre-metric versions are not pooled with final results.
No raw licensed asset, checkpoint, scene binary, scratch array or runtime cache
is copied into Git. Reproduction uses original saved local assets/runtime and
hash-bound query derivatives, writing into fresh temporary output directories.
Run sibling sampling reproduction before orientation; its exact ephemeral
fields remain local and are independently rebound by final orientation reports.
Frozen protocols are copied, never regenerated to change the experiment.

`verify_publication.py` verifies manifest/report/rule bindings, all nine sibling
protocol/source/cache bindings, witness inequalities, exact native decoding and
protected/original input hashes. It uses no simulator, optimizer or generator.
`acceptance/evaluate_acceptance.py` deterministically applies the frozen rule
from published reports; reproduce it in a temporary copy to preserve evidence.
These checks are artifact/numerical verification, not a physical fidelity proof.
