"""Read-only source/cohort/candidate audit; no model load or physics."""
from pathlib import Path
import hashlib,json,math
import xml.etree.ElementTree as ET
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def main():
    p=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json');c=read(OUT/'candidate_targets.json')
    assert c['schema']=='hug_global_constrained_surface_candidate_v1' and c['protocol_sha256']==ph
    assert c['runtime_protocol_version']==p['runtime_protocol_version']==2
    assert c['runtime_amendment']==p['runtime_amendment']==1
    assert c['numba_pool_capacity']==p['numba_pool_capacity']==4
    assert len(c['cohort_provenance'])==6 and c['numba_threads_per_process']==2
    assert len(c['cases'])==len({r['native_sha256'] for r in c['cases']})==6 and c['complete_original_six_case_coverage']
    assert c['physics_steps']==0 and c['guards']['actual']==0 and c['guards']['selftests']==18
    assert c['new_eval_data'] is False and c['new_generator_proposal'] is False and c['training'] is False
    for cohort in c['cohort_provenance']:
        r=read(cohort['path']);assert sha(cohort['path'])==cohort['sha256'] and r['protocol_sha256']==ph
        assert [x['source_inventory_index'] for x in r['cases']]==cohort['cohort_case_indices']
        assert r['guards']['actual']==0 and r['guards']['selftests']==3
    hashes={**read(OUT/'protected_hashes.json'),**p['source_sha256'],**p['implementation_sha256'],
        **p['query_cache_sha256'],**p['full_cache_sha256']}
    for r in c['cases']:hashes.update(r['input_sha256'])
    for path,h in hashes.items():assert sha(path)==h,path
    urdf=next(path for path in p['source_sha256'] if path.endswith('/shadow_hand_right.urdf'))
    ranges={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))
        for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
    selected=0;fallback=0;budgets=0
    for r in c['cases']:
        q0=r['original_qpos_rad'];q=r['qpos_rad'];audit=r['solver_audit']
        assert len(q0)==len(q)==len(r['joint_names'])==24 and q==audit['returned_qpos_rad']
        for name,original,final in zip(r['joint_names'],q0,q,strict=True):
            assert math.isfinite(original) and math.isfinite(final)
            low,high=ranges[name];assert low<=original<=high and low<=final<=high
            assert abs(final-original)<=p['solver_configuration']['joint_delta_limit_rad']+1e-14
        base=r['baseline_fresh_diagnostics'];final=r['selected_fresh_diagnostics'];cap=p['solver_configuration']['feasibility_error_allowance_m']
        original_errors=base['task_link_error_norm_m'];final_errors=final['task_link_error_norm_m']
        assert len(original_errors)==len(final_errors)==10
        assert all(value<=reference+cap for value,reference in zip(final_errors[:5],original_errors[:5]))
        assert final['mean_fingertip_error_m']<=p['solver_configuration']['maximum_mean_fingertip_error_m']
        assert r['time_s']==0 and r['object_state_and_world_pose_unchanged'] and r['final_state_freshly_recomputed_outside_solver_cache']
        assert abs(final['objective']-audit['final_assessment']['objective'])<1e-10
        if audit['surface_candidate_returned']:
            selected+=1
            assert final['objective']<base['objective']-p['solver_configuration']['minimum_objective_improvement']
            assert audit['selected_candidate_source'] in ('major_iteration','raw_final')
            if audit['selected_candidate_source']=='major_iteration':
                row=next(x for x in audit['major_iteration_assessments'] if x['major_iteration']==audit['selected_major_iteration'])
                assert q==row['qpos_rad'] and row['eligible_lower_objective_candidate']
            else:assert q==audit['raw_candidate_qpos_rad']
        else:
            fallback+=1
            assert q==q0 and r['root_target_position_m']==r['original_root_target_position_m']
            assert r['root_target_quaternion_wxyz']==r['original_root_target_quaternion_wxyz']
        budgets+=int(audit['budget_exhausted'])
    report={'schema':'hug_global_constrained_records_verification_v1','candidate_sha256':sha(OUT/'candidate_targets.json'),
        'protocol_sha256':ph,'complete_case_count':6,'unique_source_hashes_checked':len(hashes),
        'protected_source_count':len(read(OUT/'protected_hashes.json')),'source_and_protocol_hashes_preserved':True,
        'selected_surface_candidate_count':selected,'exact_original_fallback_count':fallback,'solver_budget_exhaustion_count':budgets,
        'all_actual_source_joint_bounds_and_tip_caps_checked':True,'all_final_states_fresh_outside_solver_cache':True,
        'science_acceptance_not_decided_by_this_verifier':True,'new_eval_data':False,'physics_steps':0}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
