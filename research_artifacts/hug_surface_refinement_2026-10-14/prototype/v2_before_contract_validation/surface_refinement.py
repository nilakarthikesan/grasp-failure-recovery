"""Bounded static, proposal-relative surface refinement; no grasp generator.

The evaluator supplied by the caller runs time-zero FK on actual colliders.
It returns unsigned part distances, closest points, facing scores and anchors.
All targets are native proposal relationships, including poor relationships.
No contact-quality reward or task success objective is present.
"""
from dataclasses import dataclass, asdict
from typing import Callable
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class RefinementConfig:
    near_band_m: float = 0.010
    near_sites: int = 192
    broad_sites: int = 64
    near_mass: float = 0.9
    distance_scale_m: float = 0.002
    closest_vector_weight: float = 1.0
    normal_weight: float = 0.0
    anchor_scale_m: float = 0.010
    anchor_weight: float = 0.25
    joint_delta_limit_rad: float = 0.35
    joint_weight: float = 0.05
    max_nfev: int = 60
    diff_step: float = 1e-4
    ftol: float = 1e-5
    xtol: float = 1e-5
    gtol: float = 1e-5
    loss: str = "soft_l1"
    f_scale: float = 0.2


def area_quantile_sites(mask: np.ndarray, areas: np.ndarray, count: int):
    """Deterministic quadrature over ordered cells; repeats carry their mass.

    Duplicate indices represent quantile bins hitting the same triangle. They
    are intentional, not independent observations; each bin has equal mass.
    This is an optimization quadrature, not an uncertainty certificate.
    """
    eligible = np.flatnonzero(mask)
    if not len(eligible):
        return np.empty(0, dtype=np.int64)
    total = float(areas[eligible].sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("positive finite query areas required")
    levels = (np.arange(count) + 0.5) * total / count
    chosen = np.searchsorted(np.cumsum(areas[eligible]), levels)
    return eligible[np.minimum(chosen, len(eligible)-1)]


def select_sites(native_distance, original_distance, areas, config):
    if native_distance.shape != original_distance.shape or areas.shape != native_distance.shape:
        raise ValueError("part distances and areas must share a vector shape")
    if not np.isfinite(native_distance).all() or not np.isfinite(original_distance).all():
        raise ValueError("finite exact part distances required")
    near = (native_distance <= config.near_band_m) | (original_distance <= config.near_band_m)
    ni = area_quantile_sites(near, areas, config.near_sites)
    bi = area_quantile_sites(np.ones(len(areas), bool), areas, config.broad_sites)
    if len(ni):
        weights = np.r_[np.full(len(ni), config.near_mass/len(ni)),
                        np.full(len(bi), (1-config.near_mass)/len(bi))]
        return np.r_[ni, bi], weights, {"near_bin_count":len(ni), "broad_bin_count":len(bi),
             "near_source_cell_count":int(near.sum()), "near_source_area_mm2":float(areas[near].sum())}
    return bi, np.full(len(bi), 1/len(bi)), {"near_bin_count":0, "broad_bin_count":len(bi),
             "near_source_cell_count":0, "near_source_area_mm2":0.}


def refine_part(
    initial_q: np.ndarray,
    physical_bounds: np.ndarray,
    native_distances: np.ndarray,
    native_closest_points: np.ndarray,
    native_facing: np.ndarray,
    normal_eligible: np.ndarray,
    query_weights: np.ndarray,
    original_anchor_positions: np.ndarray,
    evaluate: Callable,
    config: RefinementConfig = RefinementConfig(),
):
    """Solve one anatomical part, returning a proposal and honest diagnostics.

    ``evaluate(q)`` must return distances, closest points, facing and anchors
    at the preselected fixed sites. The caller freezes wrist/palm and other
    finger joints and certifies that part-joint sets do not overlap.
    ``accepted`` means lower frozen objective with numerical/trust checks,
    not mapping fidelity acceptance. Budget exhaustion is recorded explicitly.
    """
    q0 = np.asarray(initial_q, float).copy()
    limits = np.asarray(physical_bounds, float)
    if limits.shape != (len(q0), 2) or not np.isfinite(limits).all():
        raise ValueError("physical joint bounds must be finite Nx2")
    if np.any(q0 < limits[:,0]-1e-9) or np.any(q0 > limits[:,1]+1e-9):
        raise ValueError("initial targets exceed actual source URDF bounds")
    lo = np.maximum(limits[:,0], q0-config.joint_delta_limit_rad)
    hi = np.minimum(limits[:,1], q0+config.joint_delta_limit_rad)
    if np.any(lo >= hi):
        raise ValueError("empty bounded correction domain")
    native_distances = np.asarray(native_distances, float)
    native_closest_points = np.asarray(native_closest_points, float)
    native_facing = np.asarray(native_facing, float)
    normal_eligible = np.asarray(normal_eligible, bool)
    weights = np.asarray(query_weights, float)
    if native_distances.shape != weights.shape or np.any(weights <= 0) or not np.isclose(weights.sum(),1):
        raise ValueError("positive normalized fixed-site quadrature required")
    anchor0 = np.asarray(original_anchor_positions, float)
    sqrtw = np.sqrt(weights)
    nw = sqrtw[normal_eligible]
    if len(nw):
        nw = nw / np.linalg.norm(nw)
    calls = 0

    def components(q):
        nonlocal calls
        calls += 1
        distance, closest, facing, anchors = evaluate(q)
        if not np.isfinite(distance).all() or not np.isfinite(closest).all() or not np.isfinite(facing).all() or not np.isfinite(anchors).all():
            raise ValueError("nonfinite static geometry evaluation")
        return {
            "distance": sqrtw*(distance-native_distances)/config.distance_scale_m,
            "closest_vector": config.closest_vector_weight*sqrtw[:,None]*(closest-native_closest_points)/config.distance_scale_m/np.sqrt(3),
            "facing": config.normal_weight*nw*(facing[normal_eligible]-native_facing[normal_eligible]),
            "anchor": config.anchor_weight*(anchors-anchor0).ravel()/config.anchor_scale_m/np.sqrt(anchor0.size),
            "joint": config.joint_weight*(q-q0)/config.joint_delta_limit_rad/np.sqrt(len(q0)),
        }

    def residual(q):
        return np.concatenate([v.ravel() for v in components(q).values()])

    def cost(r):
        z=(r/config.f_scale)**2
        return float(config.f_scale**2*np.sum(np.sqrt(1+z)-1))

    before = components(q0)
    r0 = np.concatenate([v.ravel() for v in before.values()])
    solution = least_squares(residual, q0, bounds=(lo,hi), loss=config.loss,
        f_scale=config.f_scale, max_nfev=config.max_nfev, diff_step=config.diff_step,
        ftol=config.ftol, xtol=config.xtol, gtol=config.gtol)
    candidate = solution.x.copy()
    after = components(candidate)
    r1 = np.concatenate([v.ravel() for v in after.values()])
    good = (np.isfinite(candidate).all() and np.all(candidate >= lo-1e-9)
        and np.all(candidate <= hi+1e-9) and cost(r1) < cost(r0)-1e-12)
    returned = candidate if good else q0
    return returned, {
        "accepted_objective_improvement":bool(good),
        "does_not_mean_mapping_acceptance":True,
        "solver_success":bool(solution.success), "solver_status":int(solution.status),
        "solver_message":str(solution.message), "nfev":int(solution.nfev),
        "njev":int(solution.njev or 0), "geometry_evaluations_including_audit":calls,
        "before_robust_cost":cost(r0), "after_robust_cost":cost(r1),
        "before_component_sse":{k:float(np.dot(v.ravel(),v.ravel())) for k,v in before.items()},
        "after_component_sse":{k:float(np.dot(v.ravel(),v.ravel())) for k,v in after.items()},
        "maximum_abs_joint_delta_rad":float(np.max(np.abs(returned-q0))),
        "original_qpos_rad":q0.tolist(), "raw_candidate_qpos_rad":candidate.tolist(),
        "returned_qpos_rad":returned.tolist(), "active_trust_bound_count":int(np.sum(
             np.minimum(np.abs(candidate-lo),np.abs(candidate-hi)) < 1e-5)),
        "bounded_domain_rad":np.column_stack([lo,hi]).tolist(),
        "native_normal_eligible_site_count":int(normal_eligible.sum()),
    }
