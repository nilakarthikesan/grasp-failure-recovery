"""Add necessary eligibility guidance without changing exact scientific gates."""
import numpy as np
from protected_surface_policy import protected_surface_constraint_policy as old_policy

def protected_surface_constraint_policy(reference_exact):
    p=len(reference_exact['proximity']['passed']);o=len(reference_exact['exact_orientation']['passed'])
    sm=np.asarray(reference_exact['surface_margin'],float)
    if sm.shape!=(p+7*o,) or not np.isfinite(sm).all():raise ValueError('Expected P+7O finite guidance rows')
    old=dict(reference_exact);old['surface_margin']=sm[:p+5*o]
    policy=old_policy(old);valid=policy['protected_orientation_eligible_rows'];extra=np.r_[valid,valid]
    if np.any(sm[p+5*o:][extra]<0):raise ValueError('Pooled protected eligibility guidance violated')
    policy['schema']='hug_pooled_protection_with_visual_facet_guidance_v2'
    policy['hard_mask']=np.r_[policy['hard_mask'],extra]
    policy['slack_coefficients']=np.r_[policy['slack_coefficients'],(~extra).astype(float)]
    policy['offset']=np.r_[policy['offset'],np.zeros(2*o)]
    policy['row_block_labels']+=['closest_visual_exterior_guidance']*o+['mesh_conflicting_facet_guidance']*o
    policy['row_reasons']+=['preserve_reference_orientation_eligibility_guidance' if x else 'repair_reference_orientation_ineligibility' for x in valid]*2
    policy['block_order']+=['closest_visual_exterior_guidance','mesh_conflicting_facet_guidance']
    policy['reference_adjusted_surface_margin']=sm.copy()
    policy['hard_count']=int(policy['hard_mask'].sum());policy['soft_count']=int((~policy['hard_mask']).sum())
    policy['reference_hard_adjusted_margin_min']=float(sm[policy['hard_mask']].min()) if policy['hard_mask'].any() else None
    policy['search_buffer_per_raw_row']=np.r_[policy['search_buffer_per_raw_row'],np.zeros(2*o)]
    policy['extra_guidance_has_no_new_acceptance_allowance']=True
    return policy
