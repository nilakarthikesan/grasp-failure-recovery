# Frozen paired mug approach comparison v1

This prepared protocol compares the unchanged `fixed_grasp_lift_v2` vertical
controller with `hug_grasp_approach_v1` on all eight fresh mug proposals from a
completed, frozen `hug_build_collection_v1` generation. It performs no new HUG
inference or retarget solve. No physical comparison has been run by this work.

`scripts/compare_hug_approaches.py prepare` requires the actual frozen
`generation.json`, binds all input/source hashes, and declares exactly eight
pairs, at most sixteen task executions. Each proposal runs vertical first,
then the new condition. Mapping refusals retain both rows, with null physical
outcomes. Candidates are never selected using physical results.

Both conditions use the same immutable native/canonical command and intended
world palm, exact fresh captured object mass/pose, imported friction and table
placement, and zero command/sensor noise. The new condition can adjust only
WRJ1/WRJ2 under its static wrist policy; all22finger targets must match exactly.
The precomputed plan must match the plan used at execution. This is a comparison
of named controller conditions: wrist redundancy and approach direction change
together, so a difference cannot be attributed to direction alone.

The unchanged scorer supplies completed physical labels. Static planner
refusals are `unreachable_pregrasp`, never success or physical failure. Runtime
faults also have null outcomes. Raw models/observables/diagnostics stay in a new
ignored `runs/` subdirectory. The portable summary stores scalar outcomes,
provenance hashes and explicit denominators; it does not pool this comparison
with historical data or the varied-physics study.

Every physics step uses the collection driver's guard: finite full state and
controls, zero MuJoCo warning counters, expected2ms time increment within1e-12s,
and no split-step API. Each completed task requires163control endpoints and
3260physics steps; the total cap is52160steps. The first unexpected error stops
the run and leaves every remaining scheduled row `NOT_RUN`. There are no retries,
replacement grasps, outcome-guided repairs, or gate changes.

Each condition reports planned, processed, mapped, planner-refused,
execution-attempted and completed counts, plus acquisition, collision and success
numerators. Physical rates use completed labels. Planned-population and mapped
success rates retain their own denominators. A refusal never silently becomes a
negative physical training label. Paired differences are defined only when both
members completed; all other pairs remain visible.

Commands use the original project interpreter with the canonical checkout as
the working directory. Preparation/freeze are read/hash/static operations.

```sh
python scripts/compare_hug_approaches.py prepare --collection-output research_artifacts/hug_build_collection_2026-10-15_v2 --output research_artifacts/hug_paired_mug_approaches_2026-10-15_v1
python scripts/compare_hug_approaches.py freeze --output research_artifacts/hug_paired_mug_approaches_2026-10-15_v1
```

Execution additionally requires an inspected receipt with schema
`hug_paired_approach_authorization_v1`, `authorized: true`, the exact frozen
`protocol_sha256`, and `quota:16`. A prepared/frozen protocol is not authorization.
The descriptive result uses development proposals; it is not held-out predictor
evaluation or a physical fidelity certificate.
