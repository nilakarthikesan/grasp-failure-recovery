# HUG grasp-derived approach controller v1

Implemented entry points are in
`src/grasp_failure_prediction/evaluation/approach_execution.py`. This is a
separate, opt-in controller condition named `hug_grasp_approach_v1`; it is not
registered as the default and does not change historical v1/v2 protocols.
No saved-case dynamic qualification or new outcomes have run for this condition.

Create the protocol with `grasp_approach_protocol(base_v2)`, then construct
`GraspApproachRecordedRunner`. The protocol retains base v2 timing, force-limited
servos, grip/force-close targets, lift/hold task and scorer. Its hash additionally
binds the wrist-clearance and approach configurations. It requires the new
prediction recorder and writes the same measured state/applied-target channels.

Call `plan_grasp_approach(runner, pose, world_palm_position, world_palm_quaternion)`
after resetting/restoring the scene. This function uses only static forwards on
scratch data. It does not integrate, change the live state/model, load HUG,
optimize the native grasp or use task outcomes.

The declared planning policy is:

1. Reuse `plan_table_clearance` to choose wrist redundancy while preserving the
   intended world palm and all 22 finger targets. Retain the original and
   adjusted wrist commands. This adjustment is a controller condition, not a
   claim that the original native or nominal mapped pose was unchanged.
2. Enumerate approaches along the intended palm's world-space **+Y, −Y, +Z,
   −Z axes**, in that fixed order. Local Y is the palm-normal axis; local Z is
   the finger-direction axis. Every signed candidate remains in the report.
3. An approach direction points from pregrasp toward the final grasp. Require
   positive projection toward the current object body origin from the palm,
   above the declared 1e-12 m numerical epsilon. Opposite/tangential candidates
   are explicit `not_objectward` refusals. Repeated directions are refused as
   duplicates. No generated outcome or predictor score selects a direction.
4. Use the base 80 mm straight-line pregrasp distance and unchanged near-open
   hand rule. At **33 uniform samples including both endpoints**, measure all
   active hand geometry's support above the horizontal table and proximal
   forearm/wrist geometry's distance to the object collision geometries.
5. Retain the existing nominal **10 mm table margin and 1 mm proximal-object
   margin**. Select the first feasible candidate in the fixed order, while
   measuring every other eligible candidate to preserve denominators. No
   candidate is repaired based on its eventual physical outcome.
6. A fixed world-negative-Z fallback is **disabled by default**. Enabling it
   changes the protocol hash, appends one declared candidate, and subjects it
   to the same objectward, duplicate and sampled-clearance checks.

`GraspApproachPlan.as_record()` contains the wrist plan, each signed candidate,
eligibility/refusal reason, every sample's measurements, selected direction,
sampling interval and explicit null outcome. Missing feasible candidates raise
`ApproachPlanningRejected`, with mapping status `unreachable_pregrasp` and no
physical label. The controller does not start, and the recording stays empty.
Unexpected runtime failures are a separate orchestrator status; they must not
be turned into completed physical failure labels.

On an accepted plan, `GraspApproachRecordedRunner.execute` passes the adjusted
wrist pose, **original intended world palm** and selected direction to the
existing recorded actuated motion. It rejects an externally supplied direction
override. The native proposal and original nominal mapped command should stay
immutable and separately hash-bound by the caller, alongside the controller's
adjusted wrist/path record.

The samples are separated by at most 2.5 mm along the default 80 mm path.
These are finite nominal checks, not a continuous swept-volume certificate.
Finger/palm object engagement is not rejected by the proximal-object gate.
Compliant carrier tracking, finger deflection, post-approach closure and dynamic
contacts require a separately frozen bounded execution pilot. The recorder's
measured state must be compared with commanded targets; this planner alone does
not establish achieved tracking or physical grasp success.

Current validation consists of manufactured/static controls for protocol
identity, axis/sign conventions, fallback/refusal denominators, scratch-state
invariance, near-open path sampling, wrist/finger invariance and refusal before
execution. All real integration APIs are prohibited in those tests. A prospective
mug before/after pilot must retain the original fixed approach, this named
condition, every planner refusal and every attempted input in separate counts;
it has not run as part of this implementation.
