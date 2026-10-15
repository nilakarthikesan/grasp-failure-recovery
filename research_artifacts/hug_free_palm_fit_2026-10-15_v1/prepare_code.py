from pathlib import Path
old=Path('/private/tmp/hug_guarded_geometry_2026-10-14_v1')
out=Path(__file__).parent
s=(old/'feasible_surface_refinement.py').read_text()
s=s.replace('FeasibleRefinementConfig','FreePalmConfig').replace('solve_feasible_refinement','solve_free_palm')
s=s.replace('(24,','(27,').replace('range(24)','range(27)').replace('np.zeros(24)','np.zeros(27)').replace('np.eye(24)','np.eye(27)').replace('[:24]','[:27]').replace('(0, 24)','(0, 27)').replace('(5, 24)','(5, 27)')
s=s.replace('24 finite original joint targets required','27 finite original normalized state coordinates required').replace('24x2 source URDF bounds','27x2 normalized state bounds').replace('Nx24 targets','Nx27 states')
s=s.replace('low = np.maximum(bounds[:, 0], q0 - config.joint_delta_limit_rad)\n    high = np.minimum(bounds[:, 1], q0 + config.joint_delta_limit_rad)','low = bounds[:, 0].copy()\n    high = bounds[:, 1].copy()')
s=s.replace('candidate_guard=None):','candidate_guard=None, evaluate_fingertip_jacobian=None):')
s=s.replace('"restorations": 0, "agreement_checks": 0}', '"restorations": 0, "agreement_checks": 0, "analytic_Jacobians": 0}')
start=s.index('    def jacobian(q):')
end=s.index('    def directions(q, derivative, radius):',start)
s=s[:start]+'''    def jacobian(q):
        errors = tips(q)
        norms = tip_norms(errors)
        if not callable(evaluate_fingertip_jacobian):
            raise ValueError('analytic fingertip vector Jacobian callback required')
        counts['analytic_Jacobians'] += 1
        matrix = np.asarray(evaluate_fingertip_jacobian(validated_q(q)), float)
        if matrix.shape != (5, 3, 27) or not np.isfinite(matrix).all():
            raise ValueError('5x3x27 finite fingertip vector Jacobian required')
        derivative = np.einsum('ik,ikj->ij', errors, matrix)
        nonzero = norms > 0
        derivative[nonzero] /= norms[nonzero, None]
        derivative[~nonzero] = 0.
        event('analytic_FK_constraint_jacobian', state_scaled=q.tolist(), matrix=derivative.tolist(),
              zero_norm_rows=int((~nonzero).sum()))
        return derivative

'''+s[end:]
s=s.replace('for j in range(27):\n            unit = np.eye(27)[j]', 'for j in list(range(24,27))+list(range(24)):\n            unit = np.eye(27)[j]')
s=s.replace('_qpos_rad','_state_scaled').replace('physical_bounds_rad','physical_bounds_scaled').replace('bounded_correction_domain_rad','bounded_correction_domain_scaled').replace('radius_rad','radius_scaled').replace('qpos_rad','state_scaled')
s=s.replace('hug_feasible_only_static_pattern_refinement_v1','hug_free_palm_static_pattern_refinement_v1')
s='''"""27-coordinate static free-palm development solver; derived from frozen guarded solver.
Coordinates are normalized joint deltas and world translation, with bounds supplied by caller.
No joint/state changes are scientific acceptance. Exact old five tip caps remain mandatory.
"""\n'''+s[s.index('from collections'):]
(out/'free_palm_solver.py').write_text(s)
s=(old/'keyed_direction_evidence.py').read_text().replace('(24,)','(27,)').replace('24 finite q','27 finite normalized state').replace('qpos_rad','state_scaled').replace('exact_q_keyed_direction_evidence_v1','exact_27state_keyed_direction_evidence_v1')
(out/'keyed_direction_evidence.py').write_text(s)
