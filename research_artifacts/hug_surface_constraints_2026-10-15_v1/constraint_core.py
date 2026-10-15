"""Literal witness predicates and non-cancelling maximum violations.

All rows remain present. Solver quantities are search surrogates; final truth
uses the original arithmetic order and exact orientation eligibility.
"""
import numpy as np

DISTANCE_SCALE_M = .001
OPTIMIZER_INWARD_DISTANCE_M = 1e-9
OPTIMIZER_INWARD_SCORE = 1e-8


def proximity_constraints(distance_m, radius_m, allowance_m, band_m, native_near):
    d, r, eps, b = (np.asarray(x, float) for x in (distance_m, radius_m, allowance_m, band_m))
    near = np.asarray(native_near, bool)
    if not (d.shape == r.shape == eps.shape == b.shape == near.shape):
        raise ValueError('Every proximity source row needs one current distance')
    if not all(np.isfinite(x).all() for x in (d, r, eps, b)) or np.any(d < 0) or np.any(r < 0) or np.any(eps < 0):
        raise ValueError('Invalid proximity inputs')
    # Literal inherited cell_intervals operation order, not rearranged math.
    error = r + eps
    lower = np.maximum(0., d - error)
    upper = d + error
    margin_m = np.where(near, b - upper, lower - b)
    passed = np.where(near, upper <= b, lower > b)
    # Stricter solver target only. No such buffer enters final predicates.
    solver_margin = (margin_m - OPTIMIZER_INWARD_DISTANCE_M) / DISTANCE_SCALE_M
    return {'lower_m': lower, 'upper_m': upper, 'margin_m': margin_m,
            'passed': passed, 'solver_margin': solver_margin,
            'strict_far_boundary': (~near) & (lower == b)}


def orientation_constraints(score, distance_m, source_sign, source_band_m, deadzone,
                            query_boundary_tolerance_m, eligible=None):
    s, d, sign, b = (np.asarray(x, float) for x in (score, distance_m, source_sign, source_band_m))
    if not (s.shape == d.shape == sign.shape == b.shape) or not all(np.isfinite(x).all() for x in (s, d, sign, b)):
        raise ValueError('One finite score/distance is required for each fixed orientation row')
    if not np.isin(sign, (-1., 1.)).all():
        raise ValueError('Reliable source signs must preserve native facing AND away')
    sign_margin = sign * s - deadzone
    band_margin_m = b - d
    boundary_margin_m = d - query_boundary_tolerance_m
    raw_pass = (sign_margin > 0) & (d <= b) & (d > query_boundary_tolerance_m)
    # Three fixed rows per witness. Band is its original source band, not 5mm.
    surrogate = np.concatenate((sign_margin - OPTIMIZER_INWARD_SCORE,
        (band_margin_m - OPTIMIZER_INWARD_DISTANCE_M) / DISTANCE_SCALE_M,
        (boundary_margin_m - OPTIMIZER_INWARD_DISTANCE_M) / DISTANCE_SCALE_M))
    final = None if eligible is None else raw_pass & np.asarray(eligible, bool)
    return {'sign_margin': sign_margin, 'band_margin_m': band_margin_m,
            'query_boundary_margin_m': boundary_margin_m, 'raw_passed': raw_pass,
            'solver_margin': surrogate, 'passed': final}


def exact_tip_status(error_vectors_m, original_caps_m):
    e=np.asarray(error_vectors_m,float);caps=np.asarray(original_caps_m,float)
    if e.shape!=(5,3) or caps.shape!=(5,) or not np.isfinite(e).all() or not np.isfinite(caps).all():
        raise ValueError('Five exact fingertip vectors/caps required')
    norms=np.sqrt(np.einsum('ij,ij->i',e,e));mean=float(norms.mean())
    return {'norms_m':norms,'mean_m':mean,'individual_pass':norms<=caps,
            'mean_pass':mean<=.010,'passed':bool(np.all(norms<=caps) and mean<=.010)}


def diagnostic_rank(exact_eligibility, exact_margin, literal_surface_pass, state):
    """Frozen lexicographic diagnosis; invalidating witnesses never helps rank.

Only exact tip/domain-feasible states may be passed here. Every missing valid
orientation row is counted before the worst normalized raw violation. A
remaining constraint violation never receives credit from unrelated gains.
"""
    eligible=np.asarray(exact_eligibility,bool);margin=np.asarray(exact_margin,float)
    passed=np.asarray(literal_surface_pass,bool);x=np.asarray(state,float)
    if not np.isfinite(margin).all() or not np.isfinite(x).all():raise ValueError('Nonfinite rank input')
    return (int(not passed.all()),int((~eligible).sum()),float(max(0.,np.max(-margin,initial=0.))),
            int((~passed).sum()),float(np.dot(x,x)))
