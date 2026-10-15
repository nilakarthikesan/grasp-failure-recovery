"""Pooled-reference hard-row policy for the bounded SE(3) search.

This module is pure array logic. It does not evaluate geometry or change the
literal final scientific predicates. Existing inward search buffers stay on
both hard and soft rows. These stricter optimization targets do not relax any
final criterion and are not a convergence or feasibility certificate. The
V2 far-distance search margin stays unclamped (same feasible set for positive
bands). Exact tip/domain, surface pass and orientation-eligibility gates remain
required after optimization, including tie/burial/visual-membership conditions
that these raw guidance rows do not fully encode.
"""
import numpy as np

DISTANCE_SCALE_M = .001
DISTANCE_SEARCH_BUFFER_M = 1e-9
SIGN_SEARCH_BUFFER = 1e-8


def _bool_vector(value, name):
    out = np.asarray(value)
    if out.ndim != 1 or (out.size and out.dtype != np.dtype(bool)):
        raise ValueError(f'{name} must be a one-dimensional boolean vector')
    return out.astype(bool, copy=True)


def protected_surface_constraint_policy(reference_exact):
    """Return g = surface_margin + offset + slack_coefficients * epigraph.

    Protection is derived from this single fixed pooled reference only. Passed
    proximity rows protect one raw inequality, passed orientation rows protect
    sign/band/query-boundary inequalities, and eligible orientation rows protect
    nearest-part and exterior guidance. Failed/unprotected rows retain buffers
    and can receive slack. Hard rows retain the same inward search buffers to
    avoid deliberately targeting strict final boundaries. Returned arrays are
    detached from source arrays. A reference that cannot satisfy a buffered hard
    row is refused rather than silently changing its mask or buffer.
    """
    proximity = _bool_vector(reference_exact['proximity']['passed'], 'proximity.passed')
    orientation = _bool_vector(reference_exact['exact_orientation']['passed'], 'exact_orientation.passed')
    valid = _bool_vector(reference_exact['exact_orientation_fields']['valid'], 'exact_orientation_fields.valid')
    literal = _bool_vector(reference_exact['literal_surface_pass'], 'literal_surface_pass')
    p, o = len(proximity), len(orientation)
    if len(valid) != o or literal.shape != (p + o,):
        raise ValueError('Reference row dimensions disagree')
    if not np.array_equal(literal, np.r_[proximity, orientation]):
        raise ValueError('Literal reference masks disagree with component predicates')
    if np.any(orientation & ~valid):
        raise ValueError('A passed orientation witness must also be exactly eligible')
    surface = np.asarray(reference_exact['surface_margin'], dtype=float)
    if surface.shape != (p + 5 * o,) or not np.isfinite(surface).all():
        raise ValueError('Expected finite raw surface margins in P+5O order')
    hard = np.concatenate((proximity, orientation, orientation, orientation, valid, valid))
    slack = (~hard).astype(float)
    buffers = np.r_[np.full(p, DISTANCE_SEARCH_BUFFER_M / DISTANCE_SCALE_M),
                    np.full(o, SIGN_SEARCH_BUFFER),
                    np.full(4 * o, DISTANCE_SEARCH_BUFFER_M / DISTANCE_SCALE_M)]
    offset = np.zeros_like(surface)
    adjusted = surface + offset
    if np.any(adjusted[hard] < 0.):
        raise ValueError('Fixed reference violates a selected hard raw search inequality')
    labels = (['proximity'] * p + ['orientation_sign'] * o +
              ['orientation_source_band'] * o + ['orientation_query_boundary'] * o +
              ['orientation_nearest_part_guidance'] * o + ['orientation_exterior_guidance'] * o)
    reasons = (['preserve_reference_proximity_pass' if x else 'repair_reference_proximity_failure' for x in proximity] +
               ['preserve_reference_orientation_pass' if x else 'repair_reference_orientation_failure' for x in orientation] * 3 +
               ['preserve_reference_orientation_eligibility_guidance' if x else 'repair_reference_orientation_ineligibility' for x in valid] * 2)
    return {
        'schema': 'hug_pooled_reference_hard_surface_policy_v1',
        'reference_scope': 'single_frozen_pooled_pose',
        'proximity_count': p, 'orientation_count': o,
        'block_order': ['proximity', 'orientation_sign', 'orientation_source_band',
                        'orientation_query_boundary', 'orientation_nearest_part_guidance',
                        'orientation_exterior_guidance'],
        'hard_mask': hard, 'slack_coefficients': slack, 'offset': offset,
        'row_block_labels': labels, 'row_reasons': reasons,
        'hard_count': int(hard.sum()), 'soft_count': int((~hard).sum()),
        'reference_hard_adjusted_margin_min': float(adjusted[hard].min()) if hard.any() else None,
        'reference_adjusted_surface_margin': adjusted,
        'protected_literal_surface_rows': literal,
        'protected_orientation_eligible_rows': valid,
        'fixed_offset_removes_search_buffers_only': False,
        'existing_inward_search_buffers_retained_on_every_row': True,
        'search_buffer_per_raw_row': buffers,
        'far_search_margin_remains_unclamped': True,
        'floating_point_boundary_requires_literal_final_gate': True,
        'strict_far_sign_and_boundary_require_literal_final_gate': True,
        'guidance_does_not_fully_enforce_orientation_eligibility': True,
        'all_original_exact_gates_required': True,
    }
