# Updated Design Review: Fixed HUG Grasp Execution

## Response to the earlier questions

We propose using HUG proposals with a fixed Shadow Hand execution protocol for
the first dataset. This better isolates grasp/physics variation for the failure
predictor than making ACT competence a prerequisite. It is a scoped research
choice, not evidence that HUG or Shadow Hand already outperforms Panda/ACT.

- **Failure definitions:** separate invalid proposals, acquisition/lift failures,
  and post-acquisition loss. Drops onto the table must count. Slip is a precursor.
- **Training approach:** defer ACT versus Diffusion Policy comparisons. Train the
  failure predictor on executions across a declared physical-condition grid,
  retaining held-out values/combinations for testing.
- **Metrics:** separate execution quality from future-loss prediction. Report
  event-level warning times, calibration, recall/false alarms, and per-shift uncertainty.
- **End-to-end design:** fixed execution → labeled histories → predictor → matched
  recovery experiments. The learned-policy baseline remains a separate extension.

## Questions we still need reviewers to challenge

1. Does Shadow Hand retargeting add too much complexity compared with a simpler
   gripper? What checks isolate retargeting/controller faults from grasp instability?
2. What observable criteria and persistence thresholds establish acquisition and loss?
   How should we handle table-supported objects and temporary contact loss?
3. Are lift height, hold duration, closure commands, and motion profiles appropriate?
   How should we calibrate contact force rather than assume a normalized command is force?
4. Which object/proposal/physics groups must be held out to support each claim?
5. What sampling grid yields enough successes and failures without cherry-picking?
6. Which predictor inputs are realistically measurable? Is simulation contact truth
   appropriately separated as a proxy ablation?
7. Which baselines, false-alarm budgets, confidence intervals, and warning horizons
   make prediction results useful and interpretable?
8. What state and repeated-action replay checks are required for later recovery trials?

See PART_I_TRAINING_SPEC.md for the revised proposal. The existing evaluate.py
scores Panda/ACT executions only; it is not a HUG evaluator. No new benchmark
results are claimed. Earlier questions are preserved below for comparison.

<details>
<summary>Earlier Panda/ACT review questions</summary>

# Part I Design Review — Questions for Reviewers

Thanks for taking a look. The **infrastructure** (simulator setup, the
grasp-and-transport environment, synchronized logging, exact state snapshots,
demonstration collection, LeRobot export, and ACT training) is already on
`main` and is meant to be fairly standard engineering.

This PR holds the parts that are **still open for debate** and that I most want
challenged:

1. the **definitions of success and failure**,
2. the **training approach**, and
3. **how results are calculated and communicated**.

Everything here is **simulation-only**. Please push back hard — the point is to
get these definitions right before building Parts II (failure prediction) and
III (recovery) on top of them.

## What's in this PR

- `docs/PART_I_TRAINING_SPEC.md` — the proposed contract: task phases,
  success/failure definitions, what the policy sees vs. what stays hidden,
  nominal physics, data splits, and the closed-loop metrics.
- `src/grasp_failure_prediction/part1/evaluate.py` — how closed-loop results are
  computed and reported (`EvalResult.summary`).

## 1. Definitions of failure and success

Current proposal (`docs/PART_I_TRAINING_SPEC.md` §3, implemented in
`environment.py` `step()` and `config.py`):

- **Success:** container lifted ≥ `lift_height` (4 cm), transported to within
  `place_tolerance` (5 cm) of a target region, resting on the table, and
  released, inside the time limit.
- **Failure:** dropped / left the workspace, or the time limit is reached
  without success.
- Separate flags, *not* merged into success/failure: `object_lost`,
  `collision`, `excess_force` (> 60 N), `controller_fault`.

Questions:

- Is "object loss" the right primary failure signal for Part I, or should we
  define failure by grasp quality / slip rather than a full drop?
- Are the thresholds (`lift_height`, `place_tolerance`, `excess_force_newtons`)
  reasonable, or arbitrary? What would you set them to and why?
- Should a horizon **timeout** count as a failure, or as a separate outcome (we
  currently separate it in the metrics but the env flags it as failure)?
- For Part II later, failure needs to be predictable *before* it happens. Does
  this definition give a meaningful "lead time," or is a drop too abrupt?

## 2. Training approach

Current proposal:

- Imitation learning with **ACT** as the first baseline (`train_act.py`),
  trained from a **scripted privileged demonstrator** (`scripted.py`) that is
  allowed to see the container pose; the learned policy is not.
- Single nominal mass/friction during Part I training; physical variation is
  reserved for the frozen-policy Part II evaluation.

Questions:

- Is a scripted teacher acceptable as the demonstration source, or should we
  prioritize teleoperated / human demos from the start?
- ACT vs. Diffusion Policy (or something else) as the first baseline — worth it?
- Is freezing a single nominal-physics policy and only varying physics at
  evaluation the right experimental design, or should training see some
  variation?

## 3. How results are calculated and communicated

Current metrics (`evaluate.py` `EvalResult.summary`, spec §9), reported from
held-out closed-loop rollouts (not training loss):

- `success_rate`, `object_loss_rate`, `timeout_rate`,
- `collision_rate`, `excess_force_rate`,
- per-phase reach rates (grasp / lift / place).

Questions:

- What would you add or drop? (e.g., time-to-grasp, placement error
  distribution, force profiles, sample efficiency / learning curves.)
- How should uncertainty be reported across seeds — mean only, or with
  confidence intervals / per-seed detail?
- For the eventual failure-prediction work, what should we start logging *now*
  so the metrics translate (e.g., calibration-oriented quantities)?
- Is separating `object_loss` from `timeout` the right call, or misleading?

## 4. End-to-end design

The broader three-part framing lives in `docs/SYSTEM_DESIGN.md` (on `main`).

Questions:

- Is the staged, freeze-then-build approach (learn skill → predict failure →
  recover) sound, or would you structure the program differently?
- Anything that will make Parts II/III painful that we should fix in the Part I
  contract now (logging fields, snapshot contents, interfaces)?

Please leave inline comments on the specific lines/definitions above — concrete
counter-proposals for thresholds, metrics, and definitions are exactly what I'm
after.

</details>
