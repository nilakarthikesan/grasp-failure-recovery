# Independent HUG translated-palm method review, 15 October 2026

This folder contains static review and independent reconstruction of the six existing HUG proposals at inventory indices 8, 12, 16, 20, 24 and 28. It contains no model inference, new simulation episodes, grasp outcomes, labels or training data. All MuJoCo reconstruction uses forward kinematics only; the integration APIs are guarded.

The user authorized per-grasp Shadow Hand palm translation with the original accuracy criteria retained. These engineering checks do not establish scientific surface acceptance. The nine unchanged surface-resolution checks are owned by the separate surface-audit record. The proposals are previously inspected development cases, not independent validation data.

## Evidence map

| File | Scope |
| --- | --- |
| `METHOD_REVIEW_NOTES.md` | Initial invariant and failure-mode checklist. |
| `manufactured_preflight_controls_v2.py` / `manufactured_preflight_report.json` | Sixteen independent synthetic controls on the final method modules. |
| `preexecution_review.json` | 147 source, configuration, hash and AST checks on the reviewed V1 draft. |
| `freeze_transition_review.json` | Scientific settings unchanged during V1 draft-to-freeze transition. |
| `fine_preexecution_review.json` | 140 pre-execution checks on the one declared fine-scale extension; shared solver, cache, staging and scientific helpers unchanged. |
| `fine_freeze_transition_review.json` | Fine draft-to-freeze changes restricted to authorization and freeze provenance. |
| `verification_protocol.json`, `verify_outputs.py`, `verification_report.json` | Frozen independent V1 actual source/action/FK reconstruction and all 762 full callback/guard replays. |
| `diagnostic_selection/policy.json` | Ranking policy frozen before final fine results and any actual ranking. |
| `diagnostic_selection/select_and_verify.py` | Deterministic diagnostic ranking, complete callback/guard arithmetic and independent chosen-pose reconstruction. |
| `diagnostic_selection/test_selection_policy.py`, `synthetic_selection_checks.json` | Six synthetic ranking-policy controls. |
| `diagnostic_selection/freeze_inputs.py` | Creates complete immutable input closure before ranking; no geometry or score calculation. |
| `diagnostic_selection/protocol.json` | Binds 1,256 immutable method/source/trace inputs before any actual ranking. |
| `diagnostic_selection/candidate_targets.json`, `selection_report.json` | Completed six diagnostic targets and 98,184 checks of source/comparability, all 1,110 callbacks/guards, chosen-pose FK and original caps. |
| `diagnostic_selection/compact_review_table.json`, `compact_review_table.md` | Requested per-case source/callback, shift, objective, fingertip errors, signed exact cap excess and guard refusals. |

## Completed V1 evidence

Independent V1 verification passed 52,371 checks across all six cases and all 762 full callback values. This includes historical cap equality, source actions/native world targets, exact source joint ranges, fixed original root orientation, q-dependent palm compensation, original object transforms, full 27-dimensional state-key consistency, literal zero-allowance fingertip caps, fixed normal target/mass/validity handling and fresh terminal guard evidence. Actual integration attempts: zero. Three deliberately blocked guard self-tests are separately counted.

All six V1 primary returned poses are exact original fallbacks. The actual visited pattern scales are 1 and 0.5; four sweeps and the declared smaller floor are upper-budget settings, not completed coverage. The separately frozen fine extension therefore covers four smaller translation seed scales. Finite local searches and conservative direction refusals cannot prove global infeasibility.

## Diagnostic interpretation

The primary mapper's returned poses and extra guard decisions remain untouched. If no method returns a nonfallback guarded pose for a case, the frozen policy selects its lowest-comparable-objective, fully scored, original-tip-feasible nonzero translated pose for an explicitly labelled diagnostic surface audit. A diagnostic may still be refused by the additional direction guard. It is never a production candidate or a successful mapping by virtue of selection.

The extra guard requires no increase in each available part's direction SSE and no invalidation of originally valid sites. This is an additional conservative selector, distinct from the unchanged scientific mapping criteria. The diagnostic does not remove or relax either: it retains the refusal and asks the separate full-resolution scientific audit what the translated geometry preserves or changes.

## Completed diagnostic review

The one fine extension completed all 56 predeclared seeds per case and retained the original fallback in all six cases. Its 348 callbacks plus V1's 762 callbacks give 1,110 complete callback values independently checked by the diagnostic selector. All six chosen diagnostic poses satisfy the literal historical five-tip caps and the unchanged 10 mm mean bound. Source-versus-scene task-point disagreement is below 7.23e-8 m; actual requested-anchor disagreement is below 3.18e-17 m. These are implementation consistency results, not evidence that every surface relationship is preserved.

All six policy-chosen diagnostic poses originate in V1 and are refused by the unchanged extra direction guard. The policy had no minimum shift threshold. Consequently, indices 8, 12 and 16 have only 3.143, 3.284 and 1.165 nanometers of recorded palm displacement: their fit changes are dominated by joint adjustments, and they must not be described as meaningful palm-position changes. Indices 20, 24 and 28 move the palm by 4.892, 3.716 and 3.771 mm respectively. No threshold or reranking was introduced after observing this result.

The diagnostic review made zero integration attempts. The imported step guards remain installed; unlike the separate V1 verification, this run did not perform guard self-tests (the counter is zero). Complete comparison checks and every retained guard refusal are in `selection_report.json`; descriptive per-case values are in `compact_review_table.json`.

Key immutable bindings:

- Selection policy: `103bf94793a6b41b6095ef3247098e2bb3b65749535e3ebacc6bfac9dc11c1c3`.
- Selection input protocol: `c30598a6de5e9acdb6e64f5185378f8cec8cfec07bbf15db76117dbfec558ca0`.
- Diagnostic targets: `793ceb55ba4184600d5e646480d3543ec3c7550eed2f76d0eee4ac498c51d601`.
- Selection report: `2f61fa2c0b97bf87a1282c8d55ff8a4c578941c4f836d3a47d9aec8c82c46022`.

## Retained reviewer corrections

`manufactured_preflight_controls.py` and `manufactured_preflight_v1_aborted.json` retain an initial reviewer-harness coverage mistake: a zero-sweep synthetic setup never exercised the norm Jacobian callback. The zero-cap refusal itself passed. The corrected harness uses one sweep and explicitly exercises the zero-norm convention. `manufactured_preflight_report_before_final_budget.json` retains controls against the earlier draft modules; the final-module report is separate.

The fine freeze-transition read initially omitted the two expected authorization/draft flags from its metadata allowlist and failed before writing an output. The corrected report records this history; no scientific configuration or runtime was changed.

## Archive guidance

Copy this folder with its structure intact. Primary source cohorts and complete callback traces are archived by the primary fitter; frozen protocols preserve original runtime absolute paths and content hashes. Do not rewrite historical protocol paths or combine scientific protocol versions. A later archive manifest can map relocated bytes to their original hashes. Omit Python bytecode if any incidental cache is present.

The final diagnostic targets have been handed to the separate surface auditor. That agent owns the unchanged nine surface-resolution checks and their scientific pass/fail/unresolved conclusions. This review does not claim surface acceptance or authorize production promotion.
