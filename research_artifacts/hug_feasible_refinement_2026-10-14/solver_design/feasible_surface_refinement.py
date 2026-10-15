"""Experimental feasible-only pattern search on static saved HUG geometry.

This is a new development method, not a reinterpretation of the earlier SLSQP
results. Cheap FK restoration is permitted to inspect infeasible proposals.
The expensive frozen surface objective is evaluated only at points that pass
all five ORIGINAL native-relative fingertip caps and the strict 10 mm mean
guard in the computed arithmetic. No acceptance allowance is added to a cap.
Finite budgets and local pattern-search refusal imply no global infeasibility.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linprog, minimize

from grasp_failure_prediction.evaluation.constrained_surface_refinement import (
    ConstrainedEvaluation,
)


@dataclass(frozen=True)
class FeasibleRefinementConfig:
    joint_delta_limit_rad: float = 0.35
    maximum_mean_fingertip_error_m: float = 0.010
    constraint_scale_m: float = 0.010
    f_scale: float = 0.2
    minimum_objective_improvement: float = 1e-12
    initial_pattern_radius_rad: float = 0.025
    minimum_pattern_radius_rad: float = 0.0015625
    finite_difference_step_rad: float = 1e-5
    restoration_inward_target_m: float = 1e-9
    restoration_maxiter: int = 40
    restoration_ftol: float = 1e-12
    restoration_backtracks: int = 16
    maximum_pattern_sweeps: int = 6
    maximum_full_geometry_evaluations: int = 32
    maximum_FK_evaluations: int = 6000
    maximum_restorations: int = 96
    evaluation_cache_size: int = 128
    full_FK_agreement_allowance_m: float = 1e-12

    def __post_init__(self):
        for name in (
            "joint_delta_limit_rad", "maximum_mean_fingertip_error_m",
            "constraint_scale_m", "f_scale", "minimum_objective_improvement",
            "initial_pattern_radius_rad", "minimum_pattern_radius_rad",
            "finite_difference_step_rad", "restoration_inward_target_m",
            "restoration_ftol", "full_FK_agreement_allowance_m",
        ):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, float(value))
        for name in (
            "restoration_maxiter", "restoration_backtracks", "maximum_pattern_sweeps",
            "maximum_full_geometry_evaluations", "maximum_FK_evaluations",
            "maximum_restorations", "evaluation_cache_size",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "maximum_pattern_sweeps" else 1
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.maximum_full_geometry_evaluations < 3 or self.maximum_FK_evaluations < 3:
            raise ValueError("three calls must be available for baseline and terminal verification reserves")
        if self.minimum_pattern_radius_rad > self.initial_pattern_radius_rad:
            raise ValueError("minimum pattern radius exceeds initial radius")
        if self.restoration_inward_target_m >= self.maximum_mean_fingertip_error_m:
            raise ValueError("inward restoration target must be smaller than the mean guard")


class _BudgetStop(Exception):
    pass


def solve_feasible_refinement(initial_q, physical_bounds, evaluate, *, evaluate_fingertips,
                              config=FeasibleRefinementConfig(), predeclared_seed_qpos_rad=None):
    """Search only FK-feasible surface points; return exact q0 on refusal.

The caller must preserve frozen native geometry, quadrature, direction scores,
invalid-site penalties and placement. API guards cannot verify that semantic
claim. Full/FK tolerance checks callback consistency only; feasibility uses
literal norm comparisons with no extra tolerance. A positive inward target is
an optional stricter projection target, never an acceptance relaxation.
Every proposed/restored/accepted point and every callback/cache/budget event is
recorded. A new surface pose is eligible only after exact FK guards, agreement
with a full evaluation, lower frozen cost and a fresh final evaluation.
"""
    if not isinstance(config, FeasibleRefinementConfig):
        raise ValueError("FeasibleRefinementConfig required")
    if not callable(evaluate) or not callable(evaluate_fingertips):
        raise ValueError("full and cheap FK callbacks are both required")
    q0 = np.array(initial_q, float, copy=True)
    bounds = np.array(physical_bounds, float, copy=True)
    if q0.shape != (24,) or not np.isfinite(q0).all():
        raise ValueError("24 finite original joint targets required")
    if (bounds.shape != (24, 2) or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("finite ordered 24x2 source URDF bounds required")
    if np.any(q0 < bounds[:, 0]) or np.any(q0 > bounds[:, 1]):
        raise ValueError("original joint targets outside source URDF bounds")
    low = np.maximum(bounds[:, 0], q0 - config.joint_delta_limit_rad)
    high = np.minimum(bounds[:, 1], q0 + config.joint_delta_limit_rad)
    if np.any(low >= high):
        raise ValueError("empty correction domain")
    seeds = np.empty((0, 24)) if predeclared_seed_qpos_rad is None else np.array(predeclared_seed_qpos_rad, float, copy=True)
    if (seeds.ndim != 2 or seeds.shape[1:] != (24,) or not np.isfinite(seeds).all()
        or np.any(seeds < low) or np.any(seeds > high)):
        raise ValueError("predeclared seeds must be finite Nx24 targets inside original correction bounds")
    events, proposals, accepted = [], [], []
    full_cache, tip_cache = OrderedDict(), OrderedDict()
    counts = {"full": 0, "FK": 0, "full_cache_hits": 0, "FK_cache_hits": 0,
              "restorations": 0, "agreement_checks": 0}
    residual_size = None
    max_agreement = 0.0
    original_norms = None
    termination = "pattern_sweep_budget_reached"
    failure = None

    def event(kind, **fields):
        events.append({"sequence": len(events) + 1, "kind": kind, **fields})

    def validated_q(q):
        v = np.array(q, float, copy=True)
        if (v.shape != (24,) or not np.isfinite(v).all()
            or np.any(v < low) or np.any(v > high)):
            raise ValueError("invalid or out-of-bounds joint target")
        v.setflags(write=False)
        return v

    def stored(cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > config.evaluation_cache_size:
            cache.popitem(last=False)

    def tip_norms(errors):
        with np.errstate(over="raise", invalid="raise"):
            norms = np.sqrt(np.einsum("ij,ij->i", errors, errors))
        if not np.isfinite(norms).all():
            raise ValueError("numerically unrepresentable fingertip norms")
        return norms

    def exact_feasible(errors):
        norms = tip_norms(errors)
        return bool(np.all(norms <= original_norms)
                    and norms.mean() <= config.maximum_mean_fingertip_error_m)

    def assessment(errors):
        norms = tip_norms(errors)
        return {"tip_error_vectors_m": errors.tolist(), "tip_error_norm_m": norms.tolist(),
                "tip_error_increase_m": (norms-original_norms).tolist(),
                "individual_tip_nonworsening_pass": (norms <= original_norms).tolist(),
                "mean_fingertip_error_m": float(norms.mean()),
                "strict_existing_mean_guard_pass": bool(norms.mean() <= config.maximum_mean_fingertip_error_m),
                "exact_computed_engineering_feasible": exact_feasible(errors)}

    def agree(first, second):
        nonlocal max_agreement
        difference = float(np.max(np.abs(first-second)))
        counts["agreement_checks"] += 1
        max_agreement = max(max_agreement, difference)
        event("full_FK_agreement", maxabs_m=difference)
        if difference > config.full_FK_agreement_allowance_m:
            raise ValueError("full and FK-only callbacks disagree")

    def tips(q, *, fresh=False, terminal=False):
        v = validated_q(q)
        key = v.tobytes()
        if not fresh and key in tip_cache:
            counts["FK_cache_hits"] += 1
            tip_cache.move_to_end(key)
            event("FK_cache_hit", qpos_rad=v.tolist())
            return tip_cache[key]
        cap = config.maximum_FK_evaluations if terminal else config.maximum_FK_evaluations - 2
        if counts["FK"] >= cap:
            event("FK_budget_stop", count=counts["FK"], limit=cap)
            raise _BudgetStop("FK callback budget reached")
        counts["FK"] += 1
        event("FK_callback", fresh=fresh, terminal=terminal, qpos_rad=v.tolist())
        preserved_target = v.copy()
        errors = np.array(evaluate_fingertips(v), float, copy=True)
        if errors.shape != (5, 3) or not np.isfinite(errors).all():
            raise ValueError("five finite FK fingertip vectors required")
        if not np.array_equal(v, preserved_target):
            raise ValueError("FK callback modified joint target")
        tip_norms(errors)
        errors.setflags(write=False)
        if key in full_cache:
            agree(errors, full_cache[key][0].fingertip_error_vectors_m)
        stored(tip_cache, key, errors)
        return errors

    def full(q, *, fresh=False, terminal=False, allow_infeasible_baseline=False):
        nonlocal residual_size
        v = validated_q(q)
        key = v.tobytes()
        fk = tips(v)
        if original_norms is not None and not exact_feasible(fk) and not allow_infeasible_baseline:
            raise ValueError("full objective requested at FK-infeasible target")
        if not fresh and key in full_cache:
            counts["full_cache_hits"] += 1
            full_cache.move_to_end(key)
            event("full_cache_hit", qpos_rad=v.tolist())
            return full_cache[key]
        cap = config.maximum_full_geometry_evaluations if terminal else config.maximum_full_geometry_evaluations - 2
        if counts["full"] >= cap:
            event("full_budget_stop", count=counts["full"], limit=cap)
            raise _BudgetStop("full surface callback budget reached")
        counts["full"] += 1
        event("full_callback", fresh=fresh, terminal=terminal, qpos_rad=v.tolist())
        preserved_target = v.copy()
        supplied = evaluate(v)
        if not isinstance(supplied, ConstrainedEvaluation):
            raise ValueError("full callback must return ConstrainedEvaluation")
        value = ConstrainedEvaluation(supplied.residuals, supplied.fingertip_error_vectors_m)
        if not np.array_equal(v, preserved_target):
            raise ValueError("full callback modified joint target")
        if residual_size is None:
            residual_size = value.residuals.size
        elif value.residuals.size != residual_size:
            raise ValueError("frozen surface residual shape changed")
        agree(fk, value.fingertip_error_vectors_m)
        if original_norms is not None and not exact_feasible(value.fingertip_error_vectors_m) and not allow_infeasible_baseline:
            raise ValueError("full callback fails exact fingertip caps despite FK preflight")
        with np.errstate(over="raise", invalid="raise"):
            cost = float(config.f_scale*np.sum(np.hypot(config.f_scale, value.residuals)-config.f_scale))
        if not np.isfinite(cost):
            raise ValueError("numerically unrepresentable surface objective")
        stored(full_cache, key, (value, cost))
        return value, cost

    initial_errors = tips(q0)
    original_norms = tip_norms(initial_errors)
    original, original_cost = full(q0, allow_infeasible_baseline=True)
    # The actual full ORIGINAL mapping supplies the caps. FK callback agreement
    # is an independent consistency check, not a redefinition of that reference.
    original_norms = tip_norms(original.fingertip_error_vectors_m)
    baseline = assessment(original.fingertip_error_vectors_m)
    baseline["objective"] = original_cost
    current, current_cost = q0.copy(), original_cost

    def jacobian(q):
        # Bounded central differences of 5 norm constraints use FK only.
        base = tip_norms(tips(q))
        derivative = np.zeros((5, 24))
        for j in range(24):
            plus, minus = q.copy(), q.copy()
            plus[j] = min(high[j], q[j]+config.finite_difference_step_rad)
            minus[j] = max(low[j], q[j]-config.finite_difference_step_rad)
            span = plus[j]-minus[j]
            if span > 0:
                pn = tip_norms(tips(plus)) if plus[j] != q[j] else base
                mn = tip_norms(tips(minus)) if minus[j] != q[j] else base
                derivative[:, j] = (pn-mn)/span
        event("FK_constraint_jacobian", qpos_rad=q.tolist(), matrix=derivative.tolist())
        return derivative

    def directions(q, derivative, radius):
        result = []
        active = np.flatnonzero(original_norms > config.restoration_inward_target_m)
        # A cheap LP seeks a common inward norm direction. The mean derivative
        # is a linear combination of five norm rows; it needs no extra row.
        if active.size:
            rows = derivative[active]/config.constraint_scale_m
            nonzero = np.linalg.norm(rows, axis=1) > 1e-14
            rows = rows[nonzero]
            if len(rows):
                variable_bounds = list(zip(np.maximum(-1., (low-q)/radius), np.minimum(1., (high-q)/radius), strict=True)) + [(0., 1.)]
                lp = linprog(np.r_[np.zeros(24), -1.], A_ub=np.column_stack([rows, np.ones(len(rows))]),
                             b_ub=np.zeros(len(rows)), bounds=variable_bounds, method="highs")
                event("inward_direction_LP", success=bool(lp.success), status=int(lp.status),
                      inward_margin=float(lp.x[-1]) if lp.success else None)
                if lp.success and lp.x[-1] > 1e-12:
                    margin = float(lp.x[-1])*.5
                    minimum_norm = minimize(lambda x: float(x@x/2), lp.x[:24],
                        jac=lambda x: x, method="SLSQP", bounds=variable_bounds[:24],
                        constraints=[{"type": "ineq", "fun": lambda x: -rows@x-margin,
                                      "jac": lambda x: -rows}],
                        options={"maxiter": 40, "ftol": 1e-12, "disp": False})
                    direction = np.asarray(minimum_norm.x, float)
                    if (minimum_norm.success and np.all(rows@direction <= -margin)
                        and np.any(direction != 0)):
                        result.append(("common_tip_inward_minimum_norm", direction))
                    else:
                        result.append(("common_tip_inward_LP", lp.x[:24]))
        # Physical coordinate directions precede the SVD tangent basis to avoid
        # spending a small surface budget on many directions affecting only
        # manufactured or actual redundant coordinates. Ordering is fixed.
        for j in range(24):
            unit = np.eye(24)[j]
            result += [(f"joint_{j}_positive", unit), (f"joint_{j}_negative", -unit)]
        # Right singular vectors tangent to computed tip norms, then physical
        # coordinate directions as a declared deterministic fallback.
        _, singular, vectors = np.linalg.svd(derivative, full_matrices=True)
        tolerance = (max(derivative.shape)*np.finfo(float).eps*singular[0]) if singular.size and singular[0] else 0.
        rank = int((singular > tolerance).sum())
        event("norm_jacobian_rank", rank=rank, singular_values=singular.tolist())
        for j, direction in enumerate(vectors[rank:]):
            direction = direction.copy()
            if direction[np.argmax(np.abs(direction))] < 0:
                direction *= -1
            result.append((f"norm_tangent_{j}_positive", direction))
            result.append((f"norm_tangent_{j}_negative", -direction))
        return result

    def restore(proposal, anchor, entry):
        proposed_errors = tips(proposal)
        entry["proposal_assessment"] = assessment(proposed_errors)
        if exact_feasible(proposed_errors):
            entry["restoration"] = "proposal_already_exactly_feasible"
            return proposal.copy()
        if counts["restorations"] >= config.maximum_restorations:
            raise _BudgetStop("restoration budget reached")
        counts["restorations"] += 1
        # Positive caps are tightened by a fixed inward target. A zero cap
        # stays exactly zero. This can refuse a narrow but feasible set.
        inward_caps = np.maximum(0., original_norms-config.restoration_inward_target_m)
        inward_mean = config.maximum_mean_fingertip_error_m-config.restoration_inward_target_m
        if inward_mean <= 0:
            raise ValueError("inward target exceeds mean guard")
        def inequalities(q):
            norms = tip_norms(tips(q))
            return np.r_[(inward_caps-norms)/config.constraint_scale_m,
                         (inward_mean-norms.mean())/config.constraint_scale_m]
        def constraint_jac(q):
            derivative = jacobian(np.asarray(q, float))
            return np.vstack([-derivative/config.constraint_scale_m,
                              -derivative.mean(axis=0)/config.constraint_scale_m])
        scale = max(config.initial_pattern_radius_rad, 1e-12)
        result = minimize(lambda q: float(np.sum(((q-proposal)/scale)**2)/2), proposal.copy(),
                          jac=lambda q: (q-proposal)/scale**2, method="SLSQP",
                          bounds=list(zip(low, high, strict=True)),
                          constraints=[{"type": "ineq", "fun": inequalities, "jac": constraint_jac}],
                          options={"maxiter": config.restoration_maxiter, "ftol": config.restoration_ftol,
                                   "disp": False})
        candidate = np.asarray(result.x, float)
        entry["restoration"] = "cheap_FK_SLSQP_projection"
        entry["restoration_optimizer"] = {"success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "iterations": int(result.nit), "qpos_rad": candidate.tolist(),
            "tightened_tip_caps_m": inward_caps.tolist(), "tightened_mean_target_m": inward_mean}
        if candidate.shape != (24,) or not np.isfinite(candidate).all() or np.any(candidate < low) or np.any(candidate > high):
            entry["refusal"] = "projection_target_invalid"
            return None
        errors = tips(candidate)
        entry["restored_assessment"] = assessment(errors)
        if exact_feasible(errors):
            return candidate.copy()
        # Segments can be nonconvex. Every interpolant is checked independently;
        # no claim that interpolation preserves feasibility is made.
        entry["restoration_backtracking"] = []
        for step in range(1, config.restoration_backtracks+1):
            fraction = 0.5**step
            probe = anchor + fraction*(candidate-anchor)
            errors = tips(probe)
            check = assessment(errors)
            entry["restoration_backtracking"].append({"fraction": fraction, "qpos_rad": probe.tolist(), "assessment": check})
            if check["exact_computed_engineering_feasible"]:
                return probe
        entry["refusal"] = "no_exactly_feasible_projection_or_checked_backtrack"
        return None

    if not baseline["exact_computed_engineering_feasible"]:
        termination = "original_mean_guard_infeasible_no_search"
    else:
        radius = config.initial_pattern_radius_rad
        try:
            for seed_index, seed in enumerate(seeds):
                if counts["full"] >= config.maximum_full_geometry_evaluations-2:
                    raise _BudgetStop("full surface callback budget reached")
                entry = {"sequence": len(proposals)+1, "sweep": 0, "radius_rad": None,
                         "direction": "predeclared_seed", "seed_index": seed_index,
                         "anchor_qpos_rad": current.tolist(), "proposal_qpos_rad": seed.tolist(), "accepted": False}
                proposals.append(entry)
                candidate = restore(seed, current.copy(), entry)
                if candidate is None:
                    continue
                entry["candidate_qpos_rad"] = candidate.tolist()
                if np.array_equal(candidate, current):
                    entry["refusal"] = "projection_returned_anchor"
                    continue
                value, cost = full(candidate)
                entry["full_assessment"] = {**assessment(value.fingertip_error_vectors_m), "objective": cost}
                if cost < current_cost-config.minimum_objective_improvement:
                    current, current_cost = candidate.copy(), cost
                    entry["accepted"] = True
                    accepted.append({"proposal_sequence": entry["sequence"], "qpos_rad": current.tolist(), "objective": cost})
                else:
                    entry["refusal"] = "frozen_surface_objective_not_improved"
            if config.maximum_pattern_sweeps == 0:
                termination = "predeclared_seed_only_stage_complete_no_certificate"
            for sweep in range(1, config.maximum_pattern_sweeps+1):
                derivative = jacobian(current)
                improved = False
                anchor = current.copy()
                for name, direction in directions(anchor, derivative, radius):
                    if counts["full"] >= config.maximum_full_geometry_evaluations-2:
                        raise _BudgetStop("full surface callback budget reached")
                    proposal = np.clip(anchor + radius*direction, low, high)
                    entry = {"sequence": len(proposals)+1, "sweep": sweep, "radius_rad": radius,
                             "direction": name, "anchor_qpos_rad": anchor.tolist(), "proposal_qpos_rad": proposal.tolist(),
                             "accepted": False}
                    proposals.append(entry)
                    if np.array_equal(proposal, anchor):
                        entry["refusal"] = "zero_proposal_after_bounds"
                        continue
                    candidate = restore(proposal, anchor, entry)
                    if candidate is None:
                        continue
                    entry["candidate_qpos_rad"] = candidate.tolist()
                    if np.array_equal(candidate, anchor):
                        entry["refusal"] = "projection_returned_anchor"
                        continue
                    value, cost = full(candidate)
                    entry["full_assessment"] = {**assessment(value.fingertip_error_vectors_m), "objective": cost}
                    if cost < current_cost-config.minimum_objective_improvement:
                        current, current_cost = candidate.copy(), cost
                        entry["accepted"] = True
                        accepted.append({"proposal_sequence": entry["sequence"], "qpos_rad": current.tolist(), "objective": cost})
                        improved = True
                        break  # first improvement; recompute local FK geometry
                    entry["refusal"] = "frozen_surface_objective_not_improved"
                if not improved:
                    radius *= 0.5
                    event("pattern_radius_reduced", sweep=sweep, radius_rad=radius)
                    if radius < config.minimum_pattern_radius_rad:
                        termination = "pattern_radius_floor_reached_no_certificate"
                        break
        except _BudgetStop as exc:
            termination = str(exc)
        except (ValueError, FloatingPointError) as exc:
            failure = str(exc)
            termination = "geometry_or_callback_refusal"
            current, current_cost = q0.copy(), original_cost

    # Fresh terminal FK/full callbacks bypass caches and consume reserved calls.
    # Any final inconsistency forces q0 and another fresh original verification.
    terminal_value = None
    try:
        tips(current, fresh=True, terminal=True)
        terminal_value, terminal_cost = full(current, fresh=True, terminal=True,
                                             allow_infeasible_baseline=np.array_equal(current, q0))
        if not np.array_equal(current, q0) and not terminal_cost < original_cost-config.minimum_objective_improvement:
            raise ValueError("fresh final objective failed original-relative improvement")
        if abs(terminal_cost-current_cost) > 1e-10:
            raise ValueError("fresh final objective changed from cached objective")
    except (ValueError, FloatingPointError, _BudgetStop) as exc:
        failure = str(exc)
        termination = "fresh_final_verification_refusal"
        current, current_cost = q0.copy(), original_cost
        tips(current, fresh=True, terminal=True)
        terminal_value, terminal_cost = full(current, fresh=True, terminal=True, allow_infeasible_baseline=True)
        if abs(terminal_cost-original_cost) > 1e-10:
            raise ValueError("fresh original objective inconsistent after terminal refusal")
    selected = not np.array_equal(current, q0)
    final_assessment = {**assessment(terminal_value.fingertip_error_vectors_m), "objective": terminal_cost}
    report = {"schema": "hug_feasible_only_static_pattern_refinement_v1", "config": asdict(config),
              "original_qpos_rad": q0.tolist(), "returned_qpos_rad": current.tolist(),
              "predeclared_seed_qpos_rad": seeds.tolist(),
              "physical_bounds_rad": bounds.tolist(), "bounded_correction_domain_rad": np.column_stack([low, high]).tolist(),
              "baseline": baseline, "final": final_assessment, "final_assessment": final_assessment,
              "surface_candidate_returned": selected, "exact_original_fallback": not selected,
              "termination": termination, "callback_or_geometry_refusal": failure,
              "proposals": proposals, "accepted_candidates": accepted, "events": events, "counters": counts,
              "full_FK_agreement_maxabs_m": max_agreement, "final_callbacks_fresh_outside_cache": True,
              "feasibility_error_allowance_m": 0., "mapping_fidelity_acceptance": False,
              "optimizer_global_feasibility_or_convergence_claim": False,
              "scientific_acceptance_rule_changed": False, "physics_steps": 0, "new_eval_data": False}
    return current.copy(), report
