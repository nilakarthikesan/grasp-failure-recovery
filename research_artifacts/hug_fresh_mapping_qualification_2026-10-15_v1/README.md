# Four-input HUG fresh-solve qualification — 15 October 2026 v1

This completed **static development qualification** compares both existing
named profiles on the original saved proposal indices **0, 8, 16 and 24**:
the first original pose per cube, apple, milk box and mug. Selection, tolerances,
source/input hashes and the finite solve schedule were frozen before measurement
in commit `9d48405`. No outcome or error filtering selected these inputs.

Protocol SHA-256:
`f0e2dab9ff4110d11a2ab96f7ea06beca901e224b3ec47f37fc2db2cd67e50f6`.
All bindings were verified before and after the run. It made 38 project
retargeting calls, comprising 760 existing inner refinement calls, and 24
static forwards. All three integration APIs were blocked: **zero physics**.
No native inference, new observation, outcome, collection or training ran.

## Gate verdict

Both profiles pass all four declared **fresh-instance** gates. Each repeated
fresh command is bit-exact, with complete observed settings/status/raw-clamping
evidence, literal source-URDF limits, the existing 10 mm mean fingertip guard,
native-frame placement, exact named readback, and independent XML-versus-
compiled forward kinematics. No solver refusal occurred; none was discarded.

`legacy_shadow_vector_v1` remains the current provider/default profile. The
translation path constructs a fresh instance per proposal. Its four fresh
commands here are also bit-exact to the historical saved commands. This is an
observed result, not a requirement imposed on fresh solves.

**Reusing a legacy instance fails the declared input-order gate on all four
inputs.** Maximum differences from each fresh reference are 0.069609,
0.192417, 0.115449 and 0.204672 rad for indices 0, 8, 16 and 24 respectively.
Fresh-instance reproducibility therefore must not be described as permission
to reuse a legacy solver across independent proposals.

`independent_static_shadow_vector_v2` is bit-exact across every fresh,
forward/reverse reused, and A–B–A comparison. It remains an **unpromoted
separate profile**: its commands differ from saved legacy targets by 0.120779,
0.238724, 0.156068 and 0.146206 rad respectively. V2 removes the temporal term
and resets per proposal; changed fitting results are intentional protocol
differences, not evidence that the old commands were decoded incorrectly.

Mean fingertip errors for legacy are 6.704, 5.845, 6.647 and 6.609 mm; v2
has 6.869, 5.913, 6.560 and 6.599 mm. Thus reproducibility does not imply
uniformly better fitting. The largest individual fingertip error is 16.384 mm,
reported without inventing an individual-error acceptance threshold. The
existing guard is explicitly a **mean** error of 10 mm and is not a physical
contact-fidelity guarantee.

The largest independent world-FK discrepancy is 1.527e-16 m. Native input
float-valued rotation roundtrip errors range from 8.703 to 28.550 nm. That
transpose/remove/reapply diagnostic concerns small source orthogonality defects;
it is distinct from the earlier exact reconstruction of saved MANO parameters,
vertices and landmarks. Its 1 micrometre numerical eligibility allowance does
not replace or weaken the earlier reconstruction result or anatomical criteria.

All 760 observed optimizer statuses equal 3. Positive status is reported as
defined by the installed solver, not a proof of global optimality. Raw final
outputs and physically clamped outputs are distinct in every call report;
maximum final clipping is **0.001000006773 rad**. Literal source limits are
checked on the returned commands with **zero tolerance**, rather than calling
raw optimizer-margin values legal commands.

## What this supports

The current legacy implementation can support **bounded fresh-per-proposal
development solves on these tested inputs**, with a new instance each time and
explicit provenance. Its evidence is stronger than saved-target replay alone,
but does not qualify arbitrary new native proposals or instance reuse. V2
supports the tested order-independent static solves, without automatic promotion.

The canonical provider still does not propagate every raw optimizer/status
field; this audit observes and archives them separately. A future accepted
fresh-output artifact must retain profile, configuration, initialization/reset
policy, solver statuses and raw-versus-clamped commands. It cannot inherit
that provenance merely because a library call succeeds.

This audit leaves the already accepted all-32 **saved-command scene replay**
claim intact. It checks a fixed/floating source-accurate hand and native-frame
placement; existing separate all-32 assembled scene evidence covers those saved
commands. It does not validate a new controller/object scene or dynamic servo
tracking for fresh outputs. A bounded recorder pilot using qualified saved
commands is a separate next stage under its own execution protocol.

Native MANO/Shadow anatomical/contact equality remains unproved, and historical
surface counterexamples are unchanged. Four inspected development inputs are
not independent new-input qualification or evidence of broad generator
reliability. There is no physical execution acceptance or outcome claim here.

## Methods and files

For each profile, the schedule is: four separate fresh references, four fresh
repeats, one reused forward order, one separately initialized reused reverse
order, and one new A–B–A instance with indices 0, 8, 0. Each project solve uses
the existing 20-refinement implementation. A transparent observational wrapper
records original SeqRetargeting returns and optimizer statuses, restores the
method on success/refusal, and does not change the solver objective, state
initialization or clamping.

Fixed and floating compiled task points are compared with the preserved
independent source-XML FK implementation. Native wrist/operator/palm composition
is independently computed with the earlier declared normalized quaternion
convention; the production palm helper and compiled free-root readback are
checked against it. Source limits are parsed independently from literal XML.
No production runner is instantiated or advanced by this audit.

Numerical gates were frozen prospectively: 1e-10 rad repeat tolerance (actual
passing comparisons are bit-exact), 1e-10 m FK/palm positions, 1e-10 rotation-
matrix difference, 1e-10 m scalar input-frame formula, 1e-6 m native float-
rotation roundtrip, zero-tolerance literal limits and the pre-existing 0.010 m
mean fingertip engineering guard. No thresholds changed after measurement.

- `protocol.json`: source/input/dependency bindings, gates, selection and schedule.
- `source/`: 12 premeasurement source/config/reference snapshots; no licensed
  MANO model, checkpoint or original hand skin geometry.
- `calls/`: 38 complete returned-command/raw-refinement/status reports,
  including every reused-order failure.
- `results.json`: per-profile/per-case gate verdicts, call hashes and counters.
- `run.log`: complete successful bounded run.
- `test.log`: 14 passing manufactured wrapper/gate/frame controls.
- `verification.json` and `manifest.json`: final integrity receipts.

From the canonical checkout, reproduce into a **new directory**:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
  scripts/qualify_hug_fresh_mapping.py prepare \
  --project /Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction \
  --output /absolute/path/to/new/qualification
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
  scripts/qualify_hug_fresh_mapping.py run \
  --output /absolute/path/to/new/qualification
```

Preparation imports the runtime configuration to locate/hash it but performs no
solve. The original NumPy/MuJoCo runtime loads before external dependencies.
Private saved HUG outputs, existing runtime packages and the pinned Shadow
assets are required; this archive does not redistribute them. Never overwrite
this versioned evidence or modify its frozen bindings to match later code.
