# Experimental joint constrained static solver

This new version solves all 24 original Shadow wrist/finger targets together
using local SLSQP. It does not certify a global optimum or the existence of a
faithful human-to-Shadow configuration. No old mapper, experimental API,
controller, assets, accepted protocol or result is modified.

The eventual isolated project path is
`grasp_failure_prediction.evaluation.constrained_surface_refinement`, with
`tests/test_constrained_surface_refinement.py`. Its API is:

```python
evaluate(q_readonly) -> ConstrainedEvaluation(residuals, fingertip_error_vectors_m)
evaluate_fingertips(q_readonly) -> fingertip_error_vectors_m  # optional FK only
solve_constrained_refinement(original_q, physical_bounds, evaluate, config,
                             evaluate_fingertips=evaluate_fingertips)
```

Residuals are a fixed nonempty 1D finite vector, supplied by the frozen driver.
The five native-relative fingertip error vectors have shape `(5, 3)` in metres.
The caller must use identical source geometry, frames, site order, weights,
normal eligibility policy and residual blocks for all q. The solver cannot
independently certify that semantic claim about external callbacks.

The initial reference is always the **original mapping**. Each joint is bounded
by actual URDF limits intersected with original q ±0.35 rad. Internal inequalities
are each original squared tip error minus candidate squared tip error, divided
by `(0.010 m)^2`, and `(0.010 m − mean tip error)/(0.010 m)`.

Final candidate selection allows only 10 nm additional per-tip error norm as a
declared engineering arithmetic assumption. It reports exact and allowance-based
per-tip results separately. The existing computed **mean guard remains strictly
≤0.010 m**, without an enlargement. This is not an acceptable native mapping
error budget. Solver `ftol=1e-9` is dimensionless stopping precision, and absolute
finite-difference `eps=1e-5` is in radians; neither is the 10 mm guard or 10 nm
feasibility allowance. Fixed `maxiter=120` and robust soft_l1 `f_scale=0.2` apply
to every pose. Source targets are refused when outside actual bounds, without
clipping them into feasibility.

The solver assesses every **major SLSQP iterate** and its final raw result.
Only feasible candidates with objective improvement exceeding `1e-12` relative
to the original may be selected. It returns the lowest frozen objective among
those candidates, preferring the earlier major iterate on an exact tie; if none
qualifies it returns the exact original q. Finite-difference and line-search
probes are never candidates, and no outside-optimizer alpha interpolation or
case-specific fitting occurs. A later refused final result does not erase an
earlier valid major-iterate improvement.

The report preserves raw-final refusals separately from the selected source,
every major-iterate feasibility/objective, final feasibility, solver status,
budget exhaustion and exact fallback. A feasible improvement may be retained
despite budget exhaustion; that lack of convergence remains explicit. Mapping
acceptance is always false in this API report. Independent geometry and the
unchanged scientific acceptance rule are still required.

Both callbacks receive read-only copied q arrays. Full and optional FK-only
evaluations use separate bounded 64-entry LRU caches. Their baseline and
overlapping fingertip vectors must agree within `1e-12 m`, a separately named
arithmetic allowance. Containers copy and freeze residual/error arrays;
changed shapes, nonfinite geometry and detectable callback mutation are refused.

`native_direction_residual` retains finite fixed native facing scores, including
away-facing targets, and positive normalized weights. A valid candidate gets
its actual score difference; an invalid feature retains residual 2. No site is
dropped. Cosine endpoints receive a `1e-12` dimensionless arithmetic allowance,
separate from sign eligibility and scientific preservation criteria. Native
eligibility is determined by the source-bound geometry driver; unavailable
native regions remain unavailable in the audit.

`run_manufactured_checks.py` verifies frozen settings/hashes and runs 50 tests
using manufactured mathematical callbacks. These test individual constraints
versus an aggregate mean loophole, direction improvement while retaining all
five tip limits, best eligible major selection, raw refusal, exact fallback,
strict mean checking, budget labels and defensive callback handling. Passing
these tests validates solver bookkeeping, not real grasp mapping fidelity.
No inference, physics, new eval records, labels or model training occurs.
