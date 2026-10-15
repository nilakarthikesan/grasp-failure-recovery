"""Independent arithmetic audit of the declared seed-only restoration trace."""
import numpy as np


def verify_recorded_selection(target, physical_ranges, protocol):
    audit=target['solver_audit']; original=np.asarray(target['original_qpos_rad'],float)
    bounds=np.asarray([physical_ranges[n] for n in target['joint_names']],float)
    low=np.maximum(bounds[:,0],original-protocol['maximum_joint_delta_rad'])
    high=np.minimum(bounds[:,1],original+protocol['maximum_joint_delta_rad'])
    assert audit['schema']=='hug_feasible_only_static_pattern_refinement_v1'
    assert audit['config']['maximum_pattern_sweeps']==0
    assert audit['config']['maximum_full_geometry_evaluations']==4
    assert audit['feasibility_error_allowance_m']==0
    assert audit['final_callbacks_fresh_outside_cache'] and not audit['mapping_fidelity_acceptance']
    assert np.array_equal(original,np.asarray(audit['original_qpos_rad']))
    assert np.array_equal(bounds,np.asarray(audit['physical_bounds_rad']))
    assert np.array_equal(np.column_stack([low,high]),np.asarray(audit['bounded_correction_domain_rad']))
    original_vectors=np.asarray(audit['baseline']['tip_error_vectors_m'],float)
    assert original_vectors.shape==(5,3) and np.isfinite(original_vectors).all()
    original_norms=np.sqrt(np.einsum('ij,ij->i',original_vectors,original_vectors))
    baseline_cost=float(audit['baseline']['objective'])
    assert np.isfinite(baseline_cost) and baseline_cost>=0
    mean_limit=protocol['existing_engineering_mean_tip_guard_m']
    improvement=protocol['minimum_objective_improvement']

    def checked_q(q):
        v=np.asarray(q,float)
        assert v.shape==(24,) and np.isfinite(v).all()
        assert np.all(v>=low) and np.all(v<=high)
        return v

    def check_assessment(row):
        errors=np.asarray(row['tip_error_vectors_m'],float)
        assert errors.shape==(5,3) and np.isfinite(errors).all()
        norm=np.sqrt(np.einsum('ij,ij->i',errors,errors))
        assert np.array_equal(norm,np.asarray(row['tip_error_norm_m']))
        assert np.array_equal(norm-original_norms,np.asarray(row['tip_error_increase_m']))
        caps=norm<=original_norms; mean=float(norm.mean()); feasible=bool(caps.all() and mean<=mean_limit)
        assert caps.tolist()==row['individual_tip_nonworsening_pass']
        assert mean==row['mean_fingertip_error_m']
        assert bool(mean<=mean_limit)==row['strict_existing_mean_guard_pass']
        assert feasible==row['exact_computed_engineering_feasible']
        return feasible

    assert check_assessment(audit['baseline'])
    seeds=np.asarray(audit['predeclared_seed_qpos_rad'],float)
    assert seeds.shape==(1,24)
    seed=checked_q(seeds[0]);assert np.array_equal(seed,np.asarray(target['predeclared_seed_qpos_rad']))
    assert np.array_equal(seed,np.asarray(target['raw_candidate_qpos_rad']))
    assert target['seed_source_cohort_sha256']==audit['predeclared_seed_source_sha256']
    proposals=audit['proposals'];accepted=audit['accepted_candidates']
    assert len(proposals)<=1 and len(accepted)<=1
    chosen=None
    for number,row in enumerate(proposals,1):
        assert row['sequence']==number and row['sweep']==0 and row['seed_index']==0
        assert row['direction']=='predeclared_seed' and row['radius_rad'] is None
        assert np.array_equal(np.asarray(row['anchor_qpos_rad']),original)
        assert np.array_equal(checked_q(row['proposal_qpos_rad']),seed)
        if 'proposal_assessment' in row:check_assessment(row['proposal_assessment'])
        if 'restored_assessment' in row:check_assessment(row['restored_assessment'])
        for backtrack in row.get('restoration_backtracking',[]):
            checked_q(backtrack['qpos_rad']);check_assessment(backtrack['assessment'])
        if 'full_assessment' not in row:
            assert not row['accepted'];continue
        candidate=checked_q(row['candidate_qpos_rad']);assessment=row['full_assessment']
        assert check_assessment(assessment)
        cost=float(assessment['objective']);assert np.isfinite(cost) and cost>=0
        improved=bool(cost<baseline_cost-improvement)
        assert improved==row['accepted']
        if improved:chosen=(candidate,cost,row['sequence'])
    if chosen is None:
        assert accepted==[]
    else:
        assert len(accepted)==1
        assert np.array_equal(np.asarray(accepted[0]['qpos_rad']),chosen[0])
        assert accepted[0]['objective']==chosen[1] and accepted[0]['proposal_sequence']==chosen[2]
    refused=audit['callback_or_geometry_refusal'] is not None
    if refused or chosen is None:
        expected=original; source='original_fallback'; chosen_sequence=None; cost=baseline_cost
    else:
        expected=chosen[0];source='predeclared_seed_after_FK_restore';cost=chosen[1];chosen_sequence=chosen[2]
    returned=checked_q(target['qpos_rad'])
    assert np.array_equal(returned,expected) and np.array_equal(returned,np.asarray(audit['returned_qpos_rad']))
    assert audit['selected_candidate_source']==source and audit['selected_major_iteration'] is None
    assert bool(source!='original_fallback')==audit['surface_candidate_returned']
    assert bool(source=='original_fallback')==audit['exact_original_fallback']
    final=audit['final_assessment'];assert final==audit['final'] and check_assessment(final)
    assert abs(final['objective']-cost)<1e-10
    assert abs(final['objective']-target['selected_fresh_diagnostics']['objective'])<1e-10
    events=audit['events'];counts=audit['counters']
    assert [e['sequence'] for e in events]==list(range(1,len(events)+1))
    full_events=[e for e in events if e['kind']=='full_callback']
    fk_events=[e for e in events if e['kind']=='FK_callback']
    assert len(full_events)==counts['full']<=audit['config']['maximum_full_geometry_evaluations']
    assert len(fk_events)==counts['FK']<=audit['config']['maximum_FK_evaluations']
    assert full_events[-1]['fresh'] and full_events[-1]['terminal']
    assert fk_events[-1]['fresh'] and fk_events[-1]['terminal']
    assert np.array_equal(np.asarray(full_events[-1]['qpos_rad']),returned)
    assert np.array_equal(np.asarray(fk_events[-1]['qpos_rad']),returned)
    assert target['full_callback_count_including_fresh_final']==counts['full']
    assert audit['full_FK_agreement_maxabs_m']<=audit['config']['full_FK_agreement_allowance_m']
    return {'recorded_seed_restoration_selection_verified':True,'selected_candidate_source':source,
        'selected_proposal_sequence':chosen_sequence,'recorded_seed_proposal_count':len(proposals),
        'historically_accepted_seed_count':len(accepted),'fresh_final_refusal_forces_original':refused,
        'exact_computed_all_five_caps_verified_in_recorded_assessments':True,
        'norm_arithmetic':'sqrt(einsum(ij,ij->i)) on independently recovered vectors',
        'feasibility_error_allowance_m':0.,'full_and_FK_terminal_events_fresh_verified':True,
        'objective_values_not_independently_recomputed_for_all_trace_poses':True,
        'not_scientific_mapping_acceptance':True}
