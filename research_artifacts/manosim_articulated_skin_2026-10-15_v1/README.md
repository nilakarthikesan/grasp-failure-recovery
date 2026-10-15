# Articulated MANO skin qualification

This is a separate qualification of source skinning, simulator position-state
increments, and instantaneous derivatives. It grants no collection/training
admission and does not turn the previous locked-palm skin into an articulated
engine collision surface.

The candidate retains all shape terms, all 135 pose-corrective features and
the literal 16-bone linear blend weights. Positions use three world slides
and sixteen local ball rotations. Derivatives use the corresponding 51
tangent coordinates, rather than differentiating 67 quaternion positions as
independent variables. The pinned source MANO implementation is an independent
oracle for every declared state; the original saved native vertices stay
unchanged and are checked separately at the baseline.

Per saved case20/28: baseline; all45 finger angular directions at ±2degrees;
a fixed coupled direction at ±0.5/2/5degrees; root axes at ±2degrees; and
world slides at ±2mm. All109 states, their unmodified original shape row,
source32/candidate32/continuous64 surfaces, actual engine-integrated qpos and
array hashes are saved. Full51-column JVPs and independent central differences
at the saved baseline test the derivative; a separately evaluated VJP tests
force-transpose/virtual-work consistency. This finite grid is not a proof
over all joint configurations.

Native work is bounded before execution by `skin_case.BUDGET`. It allows one
original model compilation and109 `mj_integratePos` utility calls per case.
No Data, simulator FK, forward pipeline, contact search, physics step, grasp
generation or training is allowed. Thus it can establish a skin function and
its correspondence with the simulator's local ball increment convention, but
cannot establish rendered/collision-surface articulation, actuator tracking or
physical contact response in a running MANOSim scene.

`README_SKIN.md` describes the complete skin function and precision profiles.
`README_CONTACT.md` describes the separately versioned force-law kernel and its
remaining qualification requirements. Stock rigid contact dynamics and saved
object-grid orientation have separate protocols in adjacent artifact folders;
none of their passes may be substituted for native dynamic-hand qualification.

Preparation reads/hashes the completed fixed-surface lineage and new source
only. Root then commits source, successful manufactured controls and protocol
on `codex/project-work`. Only that frozen commit may run; every source/input
hash and each child's creation-time output receipt is checked again before
final publication. Output paths are write-once; failures retain partial files
and leave later cases NOT_RUN. Prior failures and source-control lineage remain
distinct. Geometry-derived arrays stay under ignored `runs`, while scalar
receipts and operation counts are committed.

Commands with the original `.venv/bin/python`:

```
run_skin.py prepare
# Commit the prepared source/protocol first.
run_skin.py run --freeze-commit <full-commit-id>
```

The next implementation check after a successful result is the actual
state-to-surface engine bridge, followed by oriented contact search and the
new law's integration/force-response fixtures. Global mapping and simulator
admission remain closed until their separate requirements pass.
