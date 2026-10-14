"""Experimental joint SLSQP refinement with native-relative fingertip guards.

This is a static development API, not a grasp generator or execution path.
The caller supplies a frozen native-relative surface/direction residual and
five fingertip error vectors from time-zero forward kinematics. SLSQP jointly
optimizes all 24 wrist/finger targets; it is a local nonlinear optimizer, not a
certificate of a global optimum or of feasible retargeting across morphologies.

Each tip's squared error must not exceed its ORIGINAL mapping's squared error,
and mean tip error must respect the existing STRICT 10 mm engineering guard.
The final per-tip comparison includes a reported 10 nm arithmetic allowance. Neither
that allowance, optimizer stopping tolerance nor a lower residual is scientific
mapping acceptance. Independent surface/uncertainty checks remain required.
Only SLSQP major iterates and its final result are eligible; finite-difference
probes are never selected. No post-hoc interpolant search, task-success reward
or proposal filtering occurs.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

DIRECTION_SCORE_ENDPOINT_ALLOWANCE = 1e-12


def _positive_scalar(value, name):
    if (isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a positive finite numeric scalar")


@dataclass(frozen=True)
class ConstrainedRefinementConfig:
    joint_delta_limit_rad: float = 0.35
    maximum_mean_fingertip_error_m: float = 0.010
    constraint_scale_m: float = 0.010
    feasibility_error_allowance_m: float = 1e-8
    minimum_objective_improvement: float = 1e-12
    f_scale: float = 0.2
    maxiter: int = 120
    ftol: float = 1e-9
    eps: float = 1e-5
    evaluation_cache_size: int = 64
    fingertip_callback_agreement_allowance_m: float = 1e-12

    def __post_init__(self):
        for name in ("joint_delta_limit_rad", "maximum_mean_fingertip_error_m",
                     "constraint_scale_m", "feasibility_error_allowance_m",
                     "minimum_objective_improvement", "f_scale", "ftol", "eps",
                     "fingertip_callback_agreement_allowance_m"):
            _positive_scalar(getattr(self, name), name)
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("maxiter", "evaluation_cache_size"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class ConstrainedEvaluation:
    residuals: np.ndarray
    fingertip_error_vectors_m: np.ndarray

    def __post_init__(self):
        residuals = np.array(self.residuals, dtype=float, copy=True)
        errors = np.array(self.fingertip_error_vectors_m, dtype=float, copy=True)
        if (residuals.ndim != 1 or residuals.size == 0
            or not np.isfinite(residuals).all()):
            raise ValueError("a nonempty finite residual vector is required")
        if errors.shape != (5, 3) or not np.isfinite(errors).all():
            raise ValueError("five finite native-relative 3D tip error vectors are required")
        residuals.setflags(write=False)
        errors.setflags(write=False)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "fingertip_error_vectors_m", errors)


def native_direction_residual(native_scores, candidate_scores, fixed_weights,
                              candidate_valid):
    """Retain fixed native direction targets and penalties for invalid sites.

    Scores are native-relative geometric compatibility values; an original
    away-facing score remains a negative target. Fixed positive weights must
    sum to one. A currently invalid feature contributes residual 2 instead of
    being dropped. Geometry eligibility and native-score validity are caller
    responsibilities; this helper implements no physical-contact criterion.
    Cosine endpoints allow only the declared 1e-12 dimensionless arithmetic
    allowance, distinct from directional sign eligibility or scientific gates.
    """
    native = np.array(native_scores, dtype=float, copy=True)
    candidate = np.array(candidate_scores, dtype=float, copy=True)
    weights = np.array(fixed_weights, dtype=float, copy=True)
    valid = np.array(candidate_valid, dtype=bool, copy=True)
    if (native.ndim != 1 or native.size == 0 or candidate.shape != native.shape
        or weights.shape != native.shape or valid.shape != native.shape
        or not np.isfinite(native).all() or not np.isfinite(weights).all()
        or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0)
        or not np.isfinite(candidate[valid]).all()
        or np.any(np.abs(native) > 1.0 + DIRECTION_SCORE_ENDPOINT_ALLOWANCE)
        or np.any(np.abs(candidate[valid]) > 1.0 + DIRECTION_SCORE_ENDPOINT_ALLOWANCE)):
        raise ValueError("fixed finite native directions and positive normalized weights required")
    residual = np.full(native.shape, 2.0)
    residual[valid] = candidate[valid] - native[valid]
    return np.sqrt(weights) * residual


def _tip_values(evaluation):
    with np.errstate(over="raise", invalid="raise"):
        squares = np.einsum("ij,ij->i", evaluation.fingertip_error_vectors_m,
                            evaluation.fingertip_error_vectors_m)
        norms = np.sqrt(squares)
    if not np.isfinite(squares).all() or not np.isfinite(norms).all():
        raise ValueError("tip error magnitudes must be numerically representable")
    return squares, norms


def _cost(evaluation, config):
    # Equivalent to the preceding frozen soft_l1 least-squares objective.
    # hypot avoids an unnecessary square overflow for a finite residual.
    with np.errstate(over="raise", invalid="raise"):
        value = config.f_scale * np.sum(
            np.hypot(config.f_scale, evaluation.residuals) - config.f_scale)
    if not np.isfinite(value):
        raise ValueError("surface objective must be numerically representable")
    return float(value)


def solve_constrained_refinement(
    initial_q: np.ndarray,
    physical_bounds: np.ndarray,
    evaluate: Callable[[np.ndarray], ConstrainedEvaluation],
    config: ConstrainedRefinementConfig = ConstrainedRefinementConfig(),
    *,
    evaluate_fingertips: Callable[[np.ndarray], np.ndarray] | None = None,
):
    """Return the best eligible major/final iterate or the exact original pose.

    ``initial_q`` is the ORIGINAL 24-joint mapped target, never an earlier
    corrected candidate. Each read-only callback q must be evaluated against
    the same native geometry, query sites, frame convention, weights and
    objective blocks. The pure API cannot verify that external semantic claim.

    Internal constraints use the unrelaxed original squared tip errors. Final
    arithmetic feasibility compares each norm against its original norm plus
    ``feasibility_error_allowance_m``. The existing mean guard remains strict,
    without an arithmetic enlargement. Both exact and allowance-dependent results are
    reported. No feasibility constraint is silently converted into a loss.

    SLSQP success is reported separately from candidate selection. A feasible
    lower-objective budget-exhausted candidate may be returned, with its lack
    of solver convergence explicit. It is never scientific mapping acceptance.
    Invalid source/config/baseline input raises ValueError. Invalid later
    geometry or optimizer refusal returns the unchanged original with a report.
    An optional FK-only ``evaluate_fingertips`` callback supplies the same 5x3
    error vectors for cheap constraint derivatives. Baseline and overlapping
    full/FK calls must agree within the separately reported arithmetic allowance.
    Eligible candidates are SLSQP major-iteration callbacks and its final raw
    result only. Select the lowest original frozen cost among feasible eligible
    candidates; finite-difference/line-search evaluations are not candidates.
    """
    if not isinstance(config, ConstrainedRefinementConfig):
        raise ValueError("ConstrainedRefinementConfig is required")
    if not callable(evaluate) or (evaluate_fingertips is not None
                                  and not callable(evaluate_fingertips)):
        raise ValueError("static geometry callbacks must be callable")
    q0 = np.array(initial_q, dtype=float, copy=True)
    limits = np.array(physical_bounds, dtype=float, copy=True)
    if q0.shape != (24,) or not np.isfinite(q0).all():
        raise ValueError("24 finite original wrist/finger joint targets are required")
    if (limits.shape != (24, 2) or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])):
        raise ValueError("finite ordered 24x2 actual source-URDF bounds are required")
    if np.any(q0 < limits[:, 0]) or np.any(q0 > limits[:, 1]):
        raise ValueError("original targets must be feasible inside actual physical bounds")
    lower = np.maximum(limits[:, 0], q0 - config.joint_delta_limit_rad)
    upper = np.minimum(limits[:, 1], q0 + config.joint_delta_limit_rad)
    if np.any(lower >= upper):
        raise ValueError("each joint needs a nonempty constrained correction domain")

    cache = OrderedDict()
    tip_cache = OrderedDict()
    calls = 0
    tip_calls = 0
    cache_hits = 0
    tip_cache_hits = 0
    callback_agreement_checks = 0
    callback_agreement_max_error_m = 0.0
    refused_evaluations = []
    residual_count = None

    def cache_tips(key, errors):
        tip_cache[key] = errors
        tip_cache.move_to_end(key)
        if len(tip_cache) > config.evaluation_cache_size:
            tip_cache.popitem(last=False)

    def check_tip_agreement(first, second):
        nonlocal callback_agreement_checks, callback_agreement_max_error_m
        error = float(np.max(np.abs(first - second)))
        callback_agreement_checks += 1
        callback_agreement_max_error_m = max(callback_agreement_max_error_m, error)
        if error > config.fingertip_callback_agreement_allowance_m:
            raise ValueError("full and FK-only fingertip callbacks disagree")

    def checked(q):
        nonlocal calls, cache_hits, residual_count
        candidate = np.array(q, dtype=float, copy=True)
        if candidate.shape != (24,) or not np.isfinite(candidate).all():
            raise ValueError("optimizer supplied an invalid joint target")
        if np.any(candidate < lower) or np.any(candidate > upper):
            raise ValueError("optimizer supplied a target outside physical/trust bounds")
        key = candidate.tobytes()
        if key in cache:
            cache_hits += 1
            cache.move_to_end(key)
            return cache[key]
        candidate.setflags(write=False)
        calls += 1
        try:
            supplied = evaluate(candidate)
            if not isinstance(supplied, ConstrainedEvaluation):
                raise ValueError("callback must return ConstrainedEvaluation")
            # Copy once more in case a callback deliberately made its cached
            # result arrays writable after constructing the data container.
            value = ConstrainedEvaluation(supplied.residuals,
                                          supplied.fingertip_error_vectors_m)
            if not np.array_equal(candidate, np.asarray(q)):
                raise ValueError("callback modified its joint argument")
            if residual_count is None:
                residual_count = value.residuals.size
            elif value.residuals.size != residual_count:
                raise ValueError("callback residual shape changed from its frozen baseline")
            _tip_values(value)
            _cost(value, config)
            if key in tip_cache:
                check_tip_agreement(value.fingertip_error_vectors_m, tip_cache[key])
        except (ValueError, FloatingPointError) as exc:
            refused_evaluations.append({"qpos_rad": candidate.tolist(),
                                        "reason": str(exc)})
            raise ValueError(f"static geometry evaluation refused: {exc}") from exc
        cache[key] = value
        cache_tips(key, value.fingertip_error_vectors_m)
        cache.move_to_end(key)
        if len(cache) > config.evaluation_cache_size:
            cache.popitem(last=False)
        return value

    def checked_tips(q, force_cheap=False):
        nonlocal tip_calls, tip_cache_hits
        if evaluate_fingertips is None:
            return checked(q).fingertip_error_vectors_m
        candidate = np.array(q, dtype=float, copy=True)
        if (candidate.shape != (24,) or not np.isfinite(candidate).all()
            or np.any(candidate < lower) or np.any(candidate > upper)):
            raise ValueError("invalid FK-only fingertip joint target")
        key = candidate.tobytes()
        if key in tip_cache and not force_cheap:
            tip_cache_hits += 1
            tip_cache.move_to_end(key)
            return tip_cache[key]
        candidate.setflags(write=False)
        tip_calls += 1
        try:
            supplied = np.array(evaluate_fingertips(candidate), dtype=float, copy=True)
            value = ConstrainedEvaluation(np.zeros(1), supplied)
            if not np.array_equal(candidate, np.asarray(q)):
                raise ValueError("FK-only callback modified its joint argument")
            _tip_values(value)
            if key in cache:
                check_tip_agreement(value.fingertip_error_vectors_m,
                                    cache[key].fingertip_error_vectors_m)
        except (ValueError, FloatingPointError) as exc:
            refused_evaluations.append({"callback": "FK-only fingertips",
                                        "qpos_rad": candidate.tolist(), "reason": str(exc)})
            raise ValueError(f"FK-only fingertip evaluation refused: {exc}") from exc
        cache_tips(key, value.fingertip_error_vectors_m)
        return value.fingertip_error_vectors_m

    original = checked(q0)
    if evaluate_fingertips is not None:
        checked_tips(q0, force_cheap=True)
    original_squares, original_norms = _tip_values(original)
    original_cost = _cost(original, config)

    def inequalities(q):
        value = ConstrainedEvaluation(np.zeros(1), checked_tips(q))
        squares, norms = _tip_values(value)
        return np.r_[
            (original_squares - squares) / config.constraint_scale_m**2,
            (config.maximum_mean_fingertip_error_m - norms.mean()) / config.constraint_scale_m,
        ]

    def assess(q, value):
        squares, norms = _tip_values(value)
        exact_tip = norms <= original_norms
        allowed_tip = norms <= original_norms + config.feasibility_error_allowance_m
        mean = float(norms.mean())
        exact_mean = mean <= config.maximum_mean_fingertip_error_m
        allowed_mean = exact_mean  # The existing computed mean guard is strict.
        bounds_pass = bool(np.all(q >= lower) and np.all(q <= upper))
        return {
            "objective": _cost(value, config),
            "tip_error_vectors_m": value.fingertip_error_vectors_m.tolist(),
            "tip_error_squared_m2": squares.tolist(),
            "tip_error_norm_m": norms.tolist(),
            "tip_error_increase_m": (norms - original_norms).tolist(),
            "mean_fingertip_error_m": mean,
            "physical_and_trust_bounds_pass": bounds_pass,
            "exact_individual_tip_nonworsening_pass": exact_tip.tolist(),
            "allowance_individual_tip_nonworsening_pass": allowed_tip.tolist(),
            "exact_existing_mean_guard_pass": bool(exact_mean),
            "allowance_existing_mean_guard_pass": bool(allowed_mean),
            "exact_engineering_feasible": bool(bounds_pass and exact_tip.all() and exact_mean),
            "allowance_engineering_feasible": bool(bounds_pass and allowed_tip.all() and allowed_mean),
            "normalized_constraint_values": np.r_[
                (original_squares - squares) / config.constraint_scale_m**2,
                (config.maximum_mean_fingertip_error_m - mean) / config.constraint_scale_m,
            ].tolist(),
        }

    baseline = assess(q0, original)
    major_iterations = []
    best_q = None
    best_assessment = None
    best_iteration = None

    def candidate_refusals(assessment):
        reasons = []
        if not all(assessment["allowance_individual_tip_nonworsening_pass"]):
            reasons.append("individual_native_relative_tip_constraint_violated")
        if not assessment["allowance_existing_mean_guard_pass"]:
            reasons.append("existing_mean_fingertip_guard_violated")
        if not assessment["objective"] < original_cost - config.minimum_objective_improvement:
            reasons.append("frozen_surface_objective_not_improved")
        return reasons

    def major_callback(q):
        nonlocal best_q, best_assessment, best_iteration
        iteration = len(major_iterations) + 1
        candidate = np.asarray(q, dtype=float)
        entry = {"major_iteration": iteration, "qpos_rad": None,
                 "assessment": None, "eligible_lower_objective_candidate": False,
                 "refusal_reasons": []}
        if candidate.shape != (24,) or not np.isfinite(candidate).all():
            entry["refusal_reasons"] = ["major_iterate_nonfinite_or_invalid_shape"]
        else:
            entry["qpos_rad"] = candidate.tolist()
            if np.any(candidate < lower) or np.any(candidate > upper):
                entry["refusal_reasons"] = ["major_iterate_outside_physical_or_trust_bounds"]
            else:
                try:
                    value = assess(candidate, checked(candidate))
                    entry["assessment"] = value
                    entry["refusal_reasons"] = candidate_refusals(value)
                    entry["eligible_lower_objective_candidate"] = not entry["refusal_reasons"]
                    if (not entry["refusal_reasons"] and
                        (best_assessment is None or value["objective"] < best_assessment["objective"])):
                        best_q = candidate.copy()
                        best_assessment = value
                        best_iteration = iteration
                except (ValueError, FloatingPointError) as exc:
                    entry["refusal_reasons"] = ["major_iterate_geometry_evaluation_refused"]
                    entry["geometry_refusal"] = str(exc)
        major_iterations.append(entry)

    solver = None
    optimizer_exception = None
    try:
        solver = minimize(
            lambda q: _cost(checked(q), config),
            q0.copy(), method="SLSQP", bounds=list(zip(lower, upper, strict=True)),
            constraints=[{"type": "ineq", "fun": inequalities}],
            callback=major_callback,
            options={"maxiter": config.maxiter, "ftol": config.ftol,
                     "eps": config.eps, "disp": False},
        )
    except (ValueError, FloatingPointError) as exc:
        optimizer_exception = str(exc)

    raw_q = None
    raw_assessment = None
    refusals = []
    if optimizer_exception is not None:
        refusals.append("optimizer_or_geometry_evaluation_refused")
    elif solver is not None:
        possible = np.asarray(solver.x, dtype=float)
        if possible.shape != (24,) or not np.isfinite(possible).all():
            refusals.append("raw_candidate_nonfinite_or_invalid_shape")
        else:
            raw_q = possible.copy()
            if np.any(raw_q < lower) or np.any(raw_q > upper):
                refusals.append("raw_candidate_outside_physical_or_trust_bounds")
            else:
                try:
                    raw_assessment = assess(raw_q, checked(raw_q))
                except (ValueError, FloatingPointError) as exc:
                    optimizer_exception = str(exc)
                    refusals.append("raw_candidate_geometry_evaluation_refused")
                if raw_assessment is not None:
                    refusals.extend(candidate_refusals(raw_assessment))
    raw_eligible = raw_q is not None and raw_assessment is not None and not refusals
    selected_source = "original_fallback"
    if best_q is not None:
        selected_source = "major_iteration"
    if (raw_eligible and
        (best_assessment is None or raw_assessment["objective"] < best_assessment["objective"])):
        best_q, best_assessment = raw_q.copy(), raw_assessment
        best_iteration = None
        selected_source = "raw_final"
    selected = best_q is not None
    returned = best_q.copy() if selected else q0.copy()
    final = best_assessment if selected else baseline
    solver_success = bool(solver.success) if solver is not None else False
    solver_status = int(solver.status) if solver is not None else None
    iterations = int(getattr(solver, "nit", 0)) if solver is not None else 0
    budget_exhausted = bool(solver is not None and not solver_success
                            and (solver_status == 9 or iterations >= config.maxiter))
    audit = {
        "schema": "hug_joint_constrained_static_refinement_v1",
        "config": asdict(config),
        "optimized_joint_count": 24,
        "initial_reference": "original mapping, not earlier corrected targets",
        "original_qpos_rad": q0.tolist(),
        "raw_candidate_qpos_rad": raw_q.tolist() if raw_q is not None else None,
        "returned_qpos_rad": returned.tolist(),
        "physical_bounds_rad": limits.tolist(),
        "bounded_correction_domain_rad": np.column_stack([lower, upper]).tolist(),
        "baseline": baseline,
        "raw_assessment": raw_assessment,
        "final_assessment": final,
        "surface_candidate_returned": bool(selected),
        "exact_original_fallback": not bool(selected),
        "selected_candidate_source": selected_source,
        "selected_major_iteration": best_iteration,
        "major_iteration_assessments": major_iterations,
        "raw_final_eligible_lower_objective_candidate": bool(raw_eligible),
        "mapping_fidelity_acceptance": False,
        "refusal_reasons": refusals,
        "raw_final_refusal_reasons": refusals.copy(),
        "selection_refusal_reasons": [] if selected else ["no_eligible_lower_objective_major_or_final_iterate"],
        "solver_success": solver_success,
        "solver_status": solver_status,
        "solver_message": str(solver.message) if solver is not None else optimizer_exception,
        "optimizer_exception": optimizer_exception,
        "iterations": iterations,
        "function_evaluations": int(getattr(solver, "nfev", 0)) if solver is not None else 0,
        "jacobian_evaluations": int(getattr(solver, "njev", 0)) if solver is not None else 0,
        "budget_exhausted": budget_exhausted,
        "candidate_returned_despite_solver_nonconvergence": bool(selected and not solver_success),
        "geometry_callback_evaluations": calls,
        "FK_only_fingertip_callback_supplied": evaluate_fingertips is not None,
        "FK_only_fingertip_callback_evaluations": tip_calls,
        "evaluation_cache_hits": cache_hits,
        "fingertip_cache_hits": tip_cache_hits,
        "full_FK_callback_agreement_checks": callback_agreement_checks,
        "full_FK_callback_agreement_max_error_m": callback_agreement_max_error_m,
        "refused_geometry_evaluations": refused_evaluations,
        "final_feasibility_is_conditional_on_engineering_arithmetic_allowance": True,
        "scientific_acceptance_rule_changed": False,
        "physics_steps": 0,
        "new_eval_data": False,
    }
    return returned, audit
