"""Separate exact-path visual-membership acceleration, no geometry alteration."""
import numpy as np


class VisualMembershipAABB:
    """Skip only guaranteed outside/nonboundary sites; retain original otherwise.

    Triangle surfaces lie within their AABB, so distance to that AABB lower
    bounds surface distance. A conservative arithmetic guard makes borderline
    predicates use the old exact path. This is shortcut eligibility, not a new
    boundary/contact tolerance. Original output flags and diagnostic keys stay
    identical. ``last_audit`` records shortcut work separately.
    """

    def __init__(self, original):
        self.original = original
        self.last_audit = {}

    def __call__(self, points, tri, p):
        q = np.asarray(points, float)
        if not len(q):
            self.last_audit = {"point_count": 0, "shortcut_count": 0, "exact_count": 0}
            return self.original(points, tri, p)
        vertices = np.asarray(tri, float).reshape(-1, 3)
        lo, hi = vertices.min(0), vertices.max(0)
        within = np.all((q >= lo) & (q <= hi), axis=1)
        gap = np.maximum(np.maximum(lo - q, q - hi), 0.0)
        gap_norm = np.linalg.norm(gap, axis=1)
        # 64 float64 eps times the coordinate scale is a conservative fallback
        # shell for arithmetic predicates. The old exact threshold is unchanged.
        scale = np.maximum(1.0, np.maximum(np.abs(q).max(1), np.abs(vertices).max()))
        guard = 64 * np.finfo(float).eps * scale
        shortcut = gap_norm > p["object_boundary_tolerance_m"] + guard
        exact = ~shortcut
        kind = np.ones(len(q), np.int8)
        if exact.any():
            kind[exact], detail = self.original(q[exact], tri, p)
            minimum = detail["minimum_winding"]
            maximum = detail["maximum_winding"]
            if shortcut.any():
                minimum, maximum = min(minimum, 0.0), max(maximum, 0.0)
            half = detail["half_winding_warning_count"]
        else:
            minimum, maximum, half = 0.0, 0.0, 0
        diagnostic = {
            "point_count": len(q), "inside_count": int((kind < 0).sum()),
            "outside_count": int((kind > 0).sum()), "boundary_count": int((kind == 0).sum()),
            "minimum_winding": minimum, "maximum_winding": maximum,
            "half_winding_warning_count": half,
            "AABB_exact_outside_reject_count": int((~within).sum()),
        }
        self.last_audit = {"point_count": len(q), "shortcut_count": int(shortcut.sum()),
                           "exact_count": int(exact.sum()),
                           "predicate_arithmetic_guard_max_m": float(guard.max())}
        return kind, diagnostic
