# Current-scene full-field exchange repair v3

This is a separately frozen static development diagnostic, following the
rejected v2 candidates. It cannot issue native fidelity PASS, collect physical
outcomes, generate new HUG proposals or train predictors.

The two changes address observed optimizer defects without changing the frozen
acceptance rule, geometry, source tip caps or original native targets:

- Restore tip caps while jointly constraining all baseline-preserved rows within
  100 nm of their full-field boundary. That value selects search constraints;
  every row is still checked by the unchanged complete-field admission test.
- Retain every guard counterexample from rejected full checkpoints. Preserve
  already passing relationships, and prevent previously unresolved native-near
  or native-far relationships from becoming opposite-definite. Unknown intervals
  remain unknown; crossing into ambiguity does not count as relationship repair.

The runner binds source, runtime, native inputs, original development domains,
all original full-grid queries and saved orientation witnesses before any solve.
Cases 20 and 28 are repeatedly inspected development grasps. No independence,
unseen-object generalization or continuous orientation certificate is claimed.

Maximum per case: 600 seconds, 1500 static forwards, 1500 raw callbacks,
six complete field checkpoints and three exchange rounds. Joint restoration
uses at most eight iterations and 400 callbacks within those totals. There is
also a 512-row local-model ceiling, including the six tip/mean rows. An excess
retains the complete counterexample inventory and refuses the solve; it never
truncates the protected constraints. There is
one declared run; partial/error results are retained without retries or budget
extensions. All complete checkpoints are indexed at creation and checked again
before publication. Source/HEAD must stay fixed through independent readback.

The legacy object visual-reference transform is checked against nonoptimized
einsum and scalar arithmetic using the exact operands during the same once-only
run. This scoped numerical comparison does not change its geometry or provide
a surface-fidelity allowance. Original outputs and warnings remain in receipts.

`protocol.json` and the freeze commit are prerequisites for `run_repair.py run`.
Manufactured tests qualify code paths; the saved run and independent readback
provide actual development measurements. A returned local improvement, if any,
is not production mapping admission. Collection and training remain paused.
