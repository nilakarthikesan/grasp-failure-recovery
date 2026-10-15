"""Read-only independent arithmetic of recorded guard/selection evidence.

No geometry callback or shared guard/solver implementation is imported here.
Complete recorded residuals support cost arithmetic, not a geometry replay.
"""
import hashlib
import numpy as np

def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def verify_recorded_selection(target, physical_ranges, protocol):
    audit=target['solver_audit']; evidence=target['keyed_direction_evidence']
    original=np.asarray(target['original_qpos_rad'],float)
    bounds=np.asarray([physical_ranges[n] for n in target['joint_names']],float)
    low=np.maximum(bounds[:,0],original-protocol['maximum_joint_delta_rad'])
    high=np.minimum(bounds[:,1],original+protocol['maximum_joint_delta_rad'])
    cfg=audit['config']; expected=protocol['candidate_solver_configuration']
    assert cfg==expected and cfg['maximum_pattern_sweeps']==2
    assert cfg['maximum_full_geometry_evaluations']==64
    assert audit['schema']=='hug_feasible_only_static_pattern_refinement_v1'
    assert audit['feasibility_error_allowance_m']==0
    assert audit['final_callbacks_fresh_outside_cache'] and not audit['mapping_fidelity_acceptance']
    assert not audit['optimizer_global_feasibility_or_convergence_claim']
    assert not audit['scientific_acceptance_rule_changed'] and audit['physics_steps']==0 and not audit['new_eval_data']
    assert np.array_equal(original,np.asarray(audit['original_qpos_rad']))
    assert np.array_equal(bounds,np.asarray(audit['physical_bounds_rad']))
    assert np.array_equal(np.column_stack([low,high]),np.asarray(audit['bounded_correction_domain_rad']))

    def checked_q(q):
        v=np.asarray(q,float)
        assert v.shape==(24,) and np.isfinite(v).all()
        assert np.all(v>=low) and np.all(v<=high)
        return v

    original_vectors=np.asarray(audit['baseline']['tip_error_vectors_m'],float)
    assert original_vectors.shape==(5,3) and np.isfinite(original_vectors).all()
    original_norms=np.sqrt(np.einsum('ij,ij->i',original_vectors,original_vectors))
    assert np.array_equal(original_vectors,np.asarray(target['baseline_fresh_diagnostics']['task_link_error_vectors_m'])[:5])
    mean_limit=protocol['existing_engineering_mean_tip_guard_m']
    improvement=protocol['minimum_objective_improvement']
    def check_assessment(row):
        errors=np.asarray(row['tip_error_vectors_m'],float)
        assert errors.shape==(5,3) and np.isfinite(errors).all()
        norm=np.sqrt(np.einsum('ij,ij->i',errors,errors))
        assert np.array_equal(norm,np.asarray(row['tip_error_norm_m']))
        assert np.array_equal(norm-original_norms,np.asarray(row['tip_error_increase_m']))
        caps=norm<=original_norms;mean=float(norm.mean());feasible=bool(caps.all() and mean<=mean_limit)
        assert caps.tolist()==row['individual_tip_nonworsening_pass']
        assert mean==row['mean_fingertip_error_m']
        assert bool(mean<=mean_limit)==row['strict_existing_mean_guard_pass']
        assert feasible==row['exact_computed_engineering_feasible']
        return feasible
    assert check_assessment(audit['baseline'])

    assert evidence['schema']=='exact_q_keyed_direction_evidence_v1'
    callbacks=evidence['callbacks'];assessments=evidence['guard_assessments']
    assert callbacks and [c['callback_sequence'] for c in callbacks]==list(range(1,len(callbacks)+1))
    assert evidence['callback_count']==len(callbacks)==audit['counters']['full']==target['full_callback_count_including_fresh_final']
    assert len(callbacks)<=cfg['maximum_full_geometry_evaluations']
    assert evidence['guard_reference_is_first_actual_q0_callback'] and not evidence['evidence_cache_eviction']
    assert not evidence['scientific_mapping_acceptance']
    assert np.array_equal(checked_q(callbacks[0]['qpos_rad']),original)
    reference=target['direction_reference_by_part']; baseline_parts=callbacks[0]['normal_diagnostics']
    assert set(reference)==set(baseline_parts)
    assert all(all(v['candidate_valid']) for v in baseline_parts.values() if v['available'])
    assert any(r['available'] for r in reference.values())
    keyed={};by_sequence={};costs={};guard_expected={}
    for callback in callbacks:
        seq=callback['callback_sequence'];q=checked_q(callback['qpos_rad']);key=q.tobytes()
        residual=np.asarray(callback['residuals'],float);tips=np.asarray(callback['fingertip_error_vectors_m'],float)
        assert residual.ndim==1 and residual.size and np.isfinite(residual).all()
        assert tips.shape==(5,3) and np.isfinite(tips).all()
        assert callback['q_float64_sha256']==_array_sha(q)
        assert callback['residuals_float64_sha256']==_array_sha(residual)
        assert callback['fingertips_float64_sha256']==_array_sha(tips)
        assert callback['complete_callback_value_and_facing_block_verified']
        assert callback['repeat_exact_q_callback']==(key in keyed)
        if key in keyed:
            previous=keyed[key]
            assert np.array_equal(residual,np.asarray(previous['residuals']))
            assert np.array_equal(tips,np.asarray(previous['fingertip_error_vectors_m']))
            assert callback['normal_diagnostics']==previous['normal_diagnostics']
        keyed[key]=callback;by_sequence[seq]=callback
        cost=float(cfg['f_scale']*np.sum(np.hypot(cfg['f_scale'],residual)-cfg['f_scale']))
        assert np.isfinite(cost) and cost>=0;costs[seq]=cost
        assert set(callback['normal_diagnostics'])==set(reference)
        facing=[];results={};reasons=[]
        for part,ref in reference.items():
            entry=callback['normal_diagnostics'][part];base=baseline_parts[part]
            assert entry['available'] is ref['available'] and base['available'] is ref['available']
            if not ref['available']:
                assert entry=={'available':False}
                results[part]={'accepted':True,'available':False,'fixed_site_count':0,'fixed_weight_mass':0.0,'preservation_status':'UNAVAILABLE'}
                continue
            native=np.asarray(ref['native_score'],float);weights=np.asarray(ref['weights'],float)
            scores=np.asarray(entry['candidate_score'],float);valid=np.asarray(entry['candidate_valid'])
            base_scores=np.asarray(base['candidate_score'],float);base_valid=np.asarray(base['candidate_valid'])
            assert native.ndim==1 and native.size and weights.shape==native.shape and np.isfinite(native).all()
            assert np.isfinite(weights).all() and np.all(weights>0) and np.isclose((weights*6).sum(),1.0)
            assert scores.shape==valid.shape==base_scores.shape==base_valid.shape==native.shape
            assert valid.dtype.kind==base_valid.dtype.kind=='b' and np.isfinite(scores).all()
            assert np.max(np.abs(native))<=1+1e-12 and np.max(np.abs(scores))<=1+1e-12
            assert np.array_equal(np.asarray(entry['native_score']),native) and np.array_equal(np.asarray(entry['weights']),weights)
            delta=np.full(native.shape,2.0);delta[valid]=scores[valid]-native[valid]
            base_delta=np.full(native.shape,2.0);base_delta[base_valid]=base_scores[base_valid]-native[base_valid]
            expected_residual=np.sqrt(weights*6)*delta/np.sqrt(6)
            assert np.array_equal(expected_residual,np.asarray(entry['residual']))
            facing.append(expected_residual)
            sse=float(np.dot(weights,delta*delta));base_sse=float(np.dot(weights,base_delta*base_delta))
            invalidated=base_valid&~valid;increased=sse>base_sse
            if invalidated.any():reasons.append(f'{part}:original_valid_sites_invalidated')
            if increased:reasons.append(f'{part}:direction_sse_increased')
            results[part]={'accepted':bool(not increased and not invalidated.any()),'available':True,'evidence_valid':True,
                'fixed_site_count':len(native),'fixed_weight_mass':float(weights.sum()),'baseline_direction_sse':base_sse,
                'candidate_direction_sse':sse,'direction_sse_change':sse-base_sse,'direction_sse_nonincrease':bool(not increased),
                'original_valid_site_count':int(base_valid.sum()),'original_valid_sites_invalidated_count':int(invalidated.sum()),
                'original_valid_sites_invalidated_indices':np.flatnonzero(invalidated).tolist(),'candidate_invalid_site_count':int((~valid).sum()),
                'candidate_invalid_fixed_mass':float(weights[~valid].sum()),'invalid_site_residual':2.0}
        combined=np.concatenate(facing);start=evidence['normal_block_start'];count=evidence['normal_block_count']
        assert len(combined)==count and start>=0 and start+count<=residual.size
        assert np.array_equal(residual[start:start+count],combined)
        guard_expected[seq]={'schema':'fixed_direction_component_guard_v1','accepted':not reasons,'refusal_reasons':reasons,
            'parts':results,'sse_increase_allowance':0.0,'invalid_site_residual':2.0,'weights_renormalized':False,
            'missing_part_mass_redistributed':False,'scientific_mapping_acceptance':False,
            'metadata_callback_sequence':seq,'q_float64_sha256':_array_sha(q),'residuals_float64_sha256':_array_sha(residual),
            'fingertips_float64_sha256':_array_sha(tips),'authoritative_exact_q_metadata':True,'complete_callback_value_agreement':True}
    assert evidence['distinct_exact_q_count']==len(keyed)
    for result in assessments:
        assert result==guard_expected[result['metadata_callback_sequence']]
    baseline_cost=float(audit['baseline']['objective'])
    assert baseline_cost==costs[1]==target['baseline_fresh_diagnostics']['objective']

    events=audit['events'];counts=audit['counters']
    assert [e['sequence'] for e in events]==list(range(1,len(events)+1))
    full_events=[e for e in events if e['kind']=='full_callback'];fk_events=[e for e in events if e['kind']=='FK_callback']
    guard_events=[e for e in events if e['kind']=='candidate_guard_assessment']
    agreement=[e for e in events if e['kind']=='full_FK_agreement']
    assert len(full_events)==counts['full']<=cfg['maximum_full_geometry_evaluations']
    assert len(fk_events)==counts['FK']<=cfg['maximum_FK_evaluations']
    assert len(guard_events)==counts['candidate_guard']
    assert len(agreement)==counts['agreement_checks']
    assert max(e['maxabs_m'] for e in agreement)==audit['full_FK_agreement_maxabs_m']<=cfg['full_FK_agreement_allowance_m']
    assert counts['restorations']<=cfg['maximum_restorations']
    assert target['tip_callback_count']==counts['FK']+1
    assert len(assessments)==len(guard_events)+1
    for event,callback in zip(full_events,callbacks,strict=True):
        assert np.array_equal(checked_q(event['qpos_rad']),checked_q(callback['qpos_rad']))
    for event,result in zip(guard_events,assessments[:-1],strict=True):
        assert event['assessment']==result==guard_expected[result['metadata_callback_sequence']]
        assert _array_sha(checked_q(event['qpos_rad']))==result['q_float64_sha256']

    seeds=np.asarray(audit['predeclared_seed_qpos_rad'],float)
    assert seeds.shape==(1,24)
    seed=checked_q(seeds[0]);assert np.array_equal(seed,np.asarray(target['predeclared_seed_qpos_rad']))
    assert np.array_equal(seed,np.asarray(target['raw_candidate_qpos_rad']))
    assert target['seed_source_cohort_sha256']==audit['predeclared_seed_source_sha256']
    proposals=audit['proposals'];accepted=audit['accepted_candidates'];chosen=[]
    current=original.copy();current_cost=baseline_cost
    max_sweep=0
    for number,row in enumerate(proposals,1):
        assert row['sequence']==number
        sweep=row['sweep'];assert 0<=sweep<=cfg['maximum_pattern_sweeps'] and sweep>=max_sweep;max_sweep=sweep
        assert np.array_equal(checked_q(row['anchor_qpos_rad']),current)
        checked_q(row['proposal_qpos_rad'])
        if sweep==0:
            assert number==1 and row['direction']=='predeclared_seed' and row['seed_index']==0 and row['radius_rad'] is None
            assert np.array_equal(checked_q(row['proposal_qpos_rad']),seed)
        else:
            assert row['radius_rad'] in [cfg['initial_pattern_radius_rad']*(.5**i) for i in range(cfg['maximum_pattern_sweeps'])]
        for key in ['proposal_assessment','restored_assessment']:
            if key in row:check_assessment(row[key])
        for backtrack in row.get('restoration_backtracking',[]):
            checked_q(backtrack['qpos_rad']);check_assessment(backtrack['assessment'])
        if 'full_assessment' not in row:
            assert not row['accepted'];continue
        q=checked_q(row['candidate_qpos_rad']);assessment=row['full_assessment'];assert check_assessment(assessment)
        guard=row['candidate_guard'];assert guard==guard_expected[guard['metadata_callback_sequence']]
        callback=by_sequence[guard['metadata_callback_sequence']]
        assert np.array_equal(q,np.asarray(callback['qpos_rad']))
        assert np.array_equal(np.asarray(assessment['tip_error_vectors_m']),np.asarray(callback['fingertip_error_vectors_m']))
        cost=float(assessment['objective']);assert cost==costs[guard['metadata_callback_sequence']]
        eligible=bool(guard['accepted'] and cost<current_cost-improvement)
        assert eligible==row['accepted']
        if not guard['accepted']:assert row['refusal']=='candidate_guard_refused'
        if eligible:
            current=q.copy();current_cost=cost
            chosen.append({'proposal_sequence':number,'qpos_rad':q.tolist(),'objective':cost})
    assert chosen==accepted
    refused=audit['callback_or_geometry_refusal'] is not None
    if refused:current=original;current_cost=baseline_cost
    returned=checked_q(target['qpos_rad']);assert np.array_equal(returned,current)
    assert np.array_equal(returned,np.asarray(audit['returned_qpos_rad']))
    selected=not np.array_equal(returned,original)
    assert selected==audit['surface_candidate_returned'] and (not selected)==audit['exact_original_fallback']
    source=('guarded_'+('predeclared_seed_after_FK_restore' if chosen[-1]['proposal_sequence']==1 else 'pattern_after_FK_restore')) if selected else 'original_fallback'
    assert audit['selected_candidate_source']==source and audit['selected_major_iteration'] is None
    final=audit['final_assessment'];assert final==audit['final'] and check_assessment(final)
    terminal=callbacks[-1];assert np.array_equal(returned,np.asarray(terminal['qpos_rad']))
    assert final['objective']==costs[len(callbacks)]==target['selected_fresh_diagnostics']['objective']
    assert abs(final['objective']-current_cost)<1e-10
    assert np.array_equal(np.asarray(final['tip_error_vectors_m']),np.asarray(terminal['fingertip_error_vectors_m']))
    assert np.array_equal(np.asarray(final['tip_error_vectors_m']),np.asarray(target['selected_fresh_diagnostics']['task_link_error_vectors_m'])[:5])
    assert target['terminal_direction_guard']==assessments[-1]==guard_expected[len(callbacks)]
    assert target['terminal_direction_guard']['accepted']
    assert evidence['fresh_terminal_callback_sequence']==evidence['last_guard_callback_sequence']==len(callbacks)
    assert full_events[-1]['fresh'] and full_events[-1]['terminal'] and fk_events[-1]['fresh'] and fk_events[-1]['terminal']
    assert np.array_equal(np.asarray(full_events[-1]['qpos_rad']),returned) and np.array_equal(np.asarray(fk_events[-1]['qpos_rad']),returned)
    if selected:
        assert final['objective']<baseline_cost-improvement
        assert guard_events[-1]['terminal'] and guard_events[-1]['assessment']['metadata_callback_sequence']==len(callbacks)
    return {'recorded_guarded_pattern_selection_verified':True,'selected_candidate_source':source,
        'selected_proposal_sequence':chosen[-1]['proposal_sequence'] if selected else None,'proposal_count':len(proposals),
        'historically_accepted_count':len(chosen),'exact_computed_all_five_caps_verified_in_recorded_assessments':True,
        'all_recorded_full_residual_objectives_recomputed':True,'all_guard_part_SSE_validity_and_metadata_bindings_recomputed':True,
        'all_full_FK_guard_budget_counts_verified':True,'full_and_FK_terminal_events_fresh_verified':True,
        'feasibility_error_allowance_m':0.,'objective_geometry_not_replayed_at_all_trace_poses':True,
        'original_fallback_does_not_mean_geometric_PASS':not selected,'not_scientific_mapping_acceptance':True}
