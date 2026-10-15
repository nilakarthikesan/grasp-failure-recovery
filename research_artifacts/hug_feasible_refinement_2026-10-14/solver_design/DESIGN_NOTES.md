# Feasible-only HUG correction design (2026-10-14)

This directory contains a new experimental method and manufactured mathematics
tests. It contains no actual HUG surface fit results, simulator outcomes, newly
generated proposals, evaluation episodes or predictor training examples.

`solve_feasible_refinement(initial_q, physical_bounds, evaluate,
evaluate_fingertips=..., config=..., predeclared_seed_qpos_rad=[old_raw])` starts
from the exact ORIGINAL 24-joint mapping. Physical bounds are intersected with
the original ±0.35 rad joint correction domain. The five caps come from the
full ORIGINAL mapping's five native-relative tip error vectors, using the
explicit `sqrt(einsum("ij,ij->i", errors, errors))` norm calculation. All five
computed norms must be at or below those original norms; computed mean must be
at or below 0.010 m. There is no extra feasibility allowance. The separate
1e-12 m full/FK callback agreement check does not relax either guard.

A proposal is checked with cheap time-zero FK before the expensive surface
objective is called. An infeasible proposal is projected toward itself by a
cheap local SLSQP quadratic objective, constrained by original tip caps
tightened inward by 1e-9 m (zero caps stay zero) and mean tightened by 1e-9 m.
The tightened target is a numerical search choice; final acceptance uses the
unrelaxed original caps. A refused projection may be followed by at most 16
individually checked anchor interpolants. Feasibility of their segments is
never assumed. Projection success/nonconvergence and every FK evaluation,
proposed/restored point, acceptance, cache event and budget stop are recorded.

Only an exactly FK-feasible point is sent to the full surface callback; the
full result must independently pass the same exact caps and agree with FK.
Only a lower frozen objective is accepted. A cached result cannot validate the
last state: fresh FK and full callbacks are mandatory. An inconsistent fresh
candidate is discarded, and the exact original is checked freshly again.
Invalid source or irreproducible original geometry raises rather than claiming
a validated fallback. This API cannot establish that the caller preserved its
native surfaces, quadrature weights, negative facing targets or invalid-site
penalties; the driver and independent audit must verify those facts.

For the first stage, use `maximum_pattern_sweeps=0`, one predeclared old raw
candidate, and `maximum_full_geometry_evaluations=4`. The ordinary successful
path costs three full calls: original, projected candidate, fresh candidate.
The fourth is reserved for fresh original fallback if final verification
refuses. A seed refusal costs two full calls. The seed is copied and bound
checked before callbacks. It is a declared warm start for this new method,
never a retroactively eligible iterate of the earlier method.

If explicitly enabled later, bounded deterministic pattern search tries a
cheap common inward tip direction, physical coordinate directions and norm
Jacobian tangent directions. Every new direction is restored and checked in
the same way. Dense full-surface finite-difference gradients are not computed.
The default full-call budget is 32, not a promise of convergence. Refusal,
budget exhaustion, an inward target narrower than the feasible set, and local
pattern stagnation establish no global infeasibility or accuracy certificate.

The 26 manufactured tests passed in 0.36 seconds. They cover curved feasible
restoration, seed-only improvement, strict zero caps, nonimproving fallback,
original mean infeasibility, source/trust bounds, deterministic behavior,
full/FK disagreement, changing residual shape, callback mutation after making
a read-only argument writable, strict budgets and fresh-final refusal with a
fresh-original fallback. Original simulator and native HUG geometry have not
been evaluated by this authoring task.
