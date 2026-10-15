"""Post-hoc read-only trace arithmetic; no geometry or acceptance changes."""
from pathlib import Path
import collections,hashlib,json
import numpy as np
OUT=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=OUT/'candidate_targets.json';report=json.loads(p.read_text());rows=[]
    for r in report['cases']:
        a=r['solver_audit'];events=a['events'];counts=collections.Counter(e['kind'] for e in events)
        q0=np.asarray(a['original_qpos_rad']);bounds=np.asarray(a['bounded_correction_domain_rad']);step=a['config']['finite_difference_step_rad']
        jac_callback_indices=set();jac_cache_indices=set();jac_request_count=0
        for i,e in enumerate(events):
            if e['kind']!='FK_constraint_jacobian':continue
            q=np.asarray(e['qpos_rad']);requested=[q]
            for j in range(24):
                plus=q.copy();minus=q.copy();plus[j]=min(bounds[j,1],q[j]+step);minus[j]=max(bounds[j,0],q[j]-step)
                if plus[j]>minus[j]:
                    if plus[j]!=q[j]:requested.append(plus)
                    if minus[j]!=q[j]:requested.append(minus)
            preceding=[n for n in range(i) if events[n]['kind'] in ('FK_callback','FK_cache_hit')][-len(requested):]
            assert len(preceding)==len(requested)
            for n,target in zip(preceding,requested,strict=True):
                assert np.array_equal(np.asarray(events[n]['qpos_rad']),target),(r['source_inventory_index'],i,n)
                if events[n]['kind']=='FK_callback':jac_callback_indices.add(n)
                else:jac_cache_indices.add(n)
            jac_request_count+=len(requested)
        reasons=collections.Counter();guard_reasons=collections.Counter();project_status=collections.Counter()
        for e in a['proposals']:
            if e.get('accepted'):reasons['accepted']+=1
            elif e.get('refusal'):reasons[e['refusal']]+=1
            elif 'full_assessment' not in e:reasons['incomplete_proposal_at_budget_stop']+=1
            else:reasons['unclassified_completed_proposal']+=1
            guard_reasons.update(e.get('candidate_guard',{}).get('refusal_reasons',[]))
            opt=e.get('restoration_optimizer')
            if opt:project_status[(str(opt['success']),str(opt['status']))]+=1
        rows.append({'index':r['source_inventory_index'],'object_id':r['object_id'],'source_case':r['source_case'],
            'selected_new_pose':a['surface_candidate_returned'],'exact_q0_returned':bool(np.array_equal(q0,np.asarray(r['qpos_rad']))),
            'termination':a['termination'],'FK_callbacks':counts['FK_callback'],'full_callbacks':counts['full_callback'],
            'completed_FK_norm_jacobians':counts['FK_constraint_jacobian'],'completed_Jacobian_tip_requests':jac_request_count,
            'completed_Jacobian_FK_callbacks_lower_bound':len(jac_callback_indices),
            'completed_Jacobian_FK_fraction_lower_bound':len(jac_callback_indices)/counts['FK_callback'],
            'completed_Jacobian_FK_cache_hits':len(jac_cache_indices),
            'restorations':a['counters']['restorations'],'proposal_outcome_counts':dict(reasons),
            'per_part_guard_refusal_reason_counts':dict(guard_reasons),
            'completed_projection_optimizer_status_counts':[{'success':s=='True','status':int(t),'count':v} for (s,t),v in project_status.items()],
            'terminal_direction_guard_pass':r['terminal_direction_guard']['accepted'],
            'no_mapping_acceptance_claim':True})
    result={'schema':'hug_guarded_geometry_budget_trace_summary_v1','posthoc_trace_arithmetic_only':True,'candidate_sha256':sha(p),
        'protocol_sha256':report['protocol_sha256'],'source_script_sha256':sha(__file__),'cases':rows,
        'completed_Jacobian_attribution_verified_by_exact_ordered_q_requests':True,
        'Jacobian_callback_count_is_lower_bound_because_final_budget_interrupted_Jacobian_may_not_emit_completion_event':True,
        'physics_steps':0,'new_eval_data':False,'scientific_mapping_acceptance':False}
    with (OUT/'budget_trace_summary.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
