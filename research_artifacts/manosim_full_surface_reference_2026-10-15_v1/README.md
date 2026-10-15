# Complete native posed-surface reference — source preparation only

This version preserves the exact saved HUG surface for development cases20 and28:
778 vertices,1552 original closed faces, original face order and fixed native
anatomical labels. It attaches every vertex to the single saved actual palm body,
`mano_body_0`. The local vertex is the inverse saved body transform applied to
the original saved vertex; there is no hull, triangle deletion, OBJ rounding,
object-query fitting, new MANO forward, HUG generation or retargeting.

The scope is locked finger articulation and global rigid root motion. This is
not an implementation of articulated MANO linear blend skinning or pose
correctives. Changing any saved finger quaternion invalidates the condition.
Old hand inertias and skeleton remain historical source parameters; this
reference does not physically validate them.

`surface_candidate.py` performs pure array/XML preparation, closed-manifold and
oriented-winding checks, literal shape identity, hash recording and compiled
array readback checks. It never imports MuJoCo. The XML uses float64 round-trip
decimal coordinates and all original face indices. Invalid/degenerate surfaces
are refused without flipping, simplifying or deleting faces. Corresponding
oriented face normals are recomputed from actual compiled triangles. The
1e-12m vertex-component and1e-10 normal-component limits are numerical readback
controls, not contact or scientific acceptance allowances. Maximum vertex L2
error also bounds the position discrepancy of corresponding barycentric
points within each fixed triangle at this pose; it does not cover articulation
or contact-search completeness.

`run_surface_reference.py` is a separately frozen static wrapper. Preparation
inherits the complete1879-input native audit and1914-input construction-ablation
closure, their final independent receipts, and the exact primitive qualification
protocol, runtime/source bindings, STARTED identity and output artifacts. It
requires the primitive radius helper to qualify **all** zero-radius controls.
The MuJoCo3.3.7 manual documents positive2D flex radii, so zero radius remains
explicitly undocumented even if those bounded controls pass. A NOT_QUALIFIED
receipt blocks this wrapper. Positive-radius flexes are also refused by this
wrapper: they need an actual rounded-triangle boundary evaluator, and must not
be silently measured as zero-thickness triangles.

After a committed protocol and separate run authorization, each child is limited
to600s, one new model, one data object, two static forwards (initial state and
exact saved named pose), two complete fields and `query_count*3104`
triangle-point tests in chunks of256. All three integration APIs and parametric
model/generation/training imports are blocked. The new XML only absolutizes the
old16 mesh paths, disables their collision bits and adds the full-face flex;
the original67qpos/51dof/51actuator skeleton remains. The old static adapter's
source/model hash condition is not bypassed. The new wrapper checks actual
names, types and addresses and writes the literal saved qpos itself.

The first field must reproduce the frozen native reference within1e-12m and
with exact masks before candidate construction. The second field uses actual
compiled flex vertices, unchanged saved native face labels, original query
cells/radii and frozen field predicates. Actual vertices, faces, owner IDs,
oriented normals, body transforms, named addresses/state, radius/margin/gap and
write-time file hashes are retained before candidate queries. Full object-query
orientation and continuous surface certification remain unproved. No mapping
PASS, production promotion, physical outcome or collection/training admission
can be issued by this diagnostic.

All derived geometry goes under the predeclared ignored
`runs/manosim_full_surface_reference_2026-10-15_v1/run_v1/raw` root. The tracked
`run_v1` stores scalar reports and immutable artifact receipts. Historical
inputs and reports are preserved. Partial failures retain original write-time
artifact rows; there are no retries or adaptive variants.

Manufactured controls only:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python scripts/check_hug_build.py --import-mode=importlib research_artifacts/manosim_full_surface_reference_2026-10-15_v1/test_surface_candidate.py research_artifacts/manosim_full_surface_reference_2026-10-15_v1/test_run_surface_reference.py -q
```

Prospective commands, **only after successful primitive qualification**:

```sh
python research_artifacts/manosim_full_surface_reference_2026-10-15_v1/run_surface_reference.py prepare --protocol research_artifacts/manosim_full_surface_reference_2026-10-15_v1/protocol.json --primitive-report research_artifacts/manosim_flex_primitive_qualification_2026-10-15_v1/run_v1/report.json
python research_artifacts/manosim_full_surface_reference_2026-10-15_v1/run_surface_reference.py run --protocol research_artifacts/manosim_full_surface_reference_2026-10-15_v1/protocol.json --freeze-commit FULL_COMMIT --output research_artifacts/manosim_full_surface_reference_2026-10-15_v1/run_v1
```

Primary implementation references are the pinned3.3.7
[flex XML definition](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#deformable-flex),
[vertex transform implementation](https://github.com/google-deepmind/mujoco/blob/f1d45bd5422c74beddfb0d1deb590a02583d21de/src/engine/engine_core_smooth.c#L428),
and [contact Jacobian implementation](https://github.com/google-deepmind/mujoco/blob/f1d45bd5422c74beddfb0d1deb590a02583d21de/src/engine/engine_core_constraint.c#L835).
The final reference must remain gated by measured primitive contact/Jacobian
controls, not only vertex geometry agreement.
