from pathlib import Path
out=Path(__file__).parent
p=out/'free_palm_kinematics.py';s=p.read_text();s=s.replace("    bounds=np.concatenate((np.column_stack(((lowq-q0)/JOINT_SCALE_RAD,(highq-q0)/JOINT_SCALE_RAD)),", """    qbounds=np.column_stack(((lowq-q0)/JOINT_SCALE_RAD,(highq-q0)/JOINT_SCALE_RAD))
    # Round normalized endpoints inward until decoded q remains inside literal bounds.
    # This only shrinks the search domain; no candidate clamp or acceptance allowance.
    endpoint_shrinks=[]
    for j in range(24):
        counts=[0,0]
        while q0[j]+JOINT_SCALE_RAD*qbounds[j,0] < lowq[j]:
            qbounds[j,0]=np.nextafter(qbounds[j,0],np.inf);counts[0]+=1
        while q0[j]+JOINT_SCALE_RAD*qbounds[j,1] > highq[j]:
            qbounds[j,1]=np.nextafter(qbounds[j,1],-np.inf);counts[1]+=1
        endpoint_shrinks.append(counts)
    bounds=np.concatenate((qbounds,""")
s=s.replace("'domain_padding_not_acceptance_allowance':True}","'domain_padding_not_acceptance_allowance':True,'normalized_joint_endpoint_inward_ulp_steps':endpoint_shrinks}")
p.write_text(s)
p=out/'run_free_palm.py';s=p.read_text();s=s.replace("        original_caps=np.sqrt(np.einsum('ij,ij->i',initial_errors,initial_errors))", """        original_caps=np.sqrt(np.einsum('ij,ij->i',initial_errors,initial_errors))
        cap_binding=p['prior_original_cap_bindings'][str(index)]
        assert a.sha(cap_binding['path'])==cap_binding['sha256']
        prior_cap_record=read(cap_binding['path'])
        assert prior_cap_record['source_inventory_index']==index and prior_cap_record['native_sha256']==case['native']['sha256']
        prior_caps=np.asarray(prior_cap_record['engineering_guard_independent_audit']['original_tip_caps_sqrt_einsum_m'])
        assert np.array_equal(original_caps,prior_caps), 'Original cap rebasing is forbidden'""")
s=s.replace("        assert np.array_equal(original_cap_norms,np.asarray(audit['baseline']['tip_error_norm_m']))", "        assert np.array_equal(original_cap_norms,np.asarray(audit['baseline']['tip_error_norm_m']))\n        assert np.array_equal(original_cap_norms,prior_caps)")
s=s.replace("'target_palm_world_rotation':desiredR.tolist()","'target_palm_world_rotation':d.xmat[palm].reshape(3,3).tolist(),'source_expected_palm_world_rotation':desiredR.tolist()")
s=s.replace("'authoritative_original_tip_caps_sqrt_einsum_m':original_cap_norms.tolist(),","'authoritative_original_tip_caps_sqrt_einsum_m':original_cap_norms.tolist(),\n            'prior_original_cap_binding':cap_binding,'prior_original_caps_reproduced_exactly':True,")
p.write_text(s)
p=out/'free_palm_solver.py';s=p.read_text();s=s.replace("        counts['analytic_Jacobians'] += 1", "        if counts['analytic_Jacobians'] >= config.maximum_FK_evaluations:\n            raise _BudgetStop('analytic Jacobian callback budget reached')\n        counts['analytic_Jacobians'] += 1")
s=s.replace("'joint_", "'coordinate_")
p.write_text(s)
