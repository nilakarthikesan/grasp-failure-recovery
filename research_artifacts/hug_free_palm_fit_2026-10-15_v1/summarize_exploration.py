"""Post-selection arithmetic only; all attempts come from frozen primary traces."""
from pathlib import Path
from collections import Counter
import hashlib,json,gzip
import numpy as np
OUT=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(states):
    x=np.asarray(states,float).reshape((-1,27));shifts=x[:,24:]*.005 if len(x) else np.zeros((0,3));norms=np.linalg.norm(shifts,axis=1)*1000
    return {'count':len(x),'nonzero_translation_count':int((norms>0).sum()),
        'nonzero_palm_shift_norm_min_mm':float(norms[norms>0].min()) if np.any(norms>0) else None,
        'palm_shift_norm_max_mm':float(norms.max()) if len(norms) else None,
        'world_displacement_component_min_mm':(shifts.min(0)*1000).tolist() if len(shifts) else None,
        'world_displacement_component_max_mm':(shifts.max(0)*1000).tolist() if len(shifts) else None}

def main():
    records=[];inputs={}
    for indices in ((8,16,24),(12,20,28)):
        path=OUT/('candidate_cohort_'+'_'.join(map(str,indices))+'.json');inputs[str(path)]=sha(path)
        cohort=json.loads(path.read_text())
        for c in cohort['cases']:
            a=c['solver_audit'];proposals=a['proposals'];reason=Counter();guard=Counter();restored=[];scored=[];accepted=[]
            for p in proposals:
                reason[p.get('refusal','accepted' if p.get('accepted') else 'interrupted_before_final_decision')]+=1
                for r in p.get('candidate_guard',{}).get('refusal_reasons',[]):guard[r]+=1
                if 'candidate_state_scaled' in p:restored.append(p['candidate_state_scaled'])
                if 'full_assessment' in p:scored.append(p['candidate_state_scaled'])
                if p.get('accepted'):accepted.append(p['candidate_state_scaled'])
            callbacks=[]
            for e in c['keyed_direction_evidence']['callbacks']:callbacks.append(e['state_scaled'])
            actual_fk=[e['state_scaled'] for e in a['events'] if e['kind']=='FK_callback']
            records.append({'source_inventory_index':c['source_inventory_index'],
                'proposed_states':stats([p['proposal_state_scaled'] for p in proposals]),
                'restored_candidates':stats(restored),'fully_scored_nonbaseline_proposal_states':stats(scored),
                'all_full_callback_states_including_original_and_terminal':stats(callbacks),
                'actual_cheap_FK_callback_states':stats(actual_fk),'historically_accepted_states':stats(accepted),
                'selected_shift_mm':(np.asarray(c['palm_displacement_world_m'])*1000).tolist(),
                'visited_pattern_radii_scaled':sorted({v['radius_scaled'] for v in proposals if v.get('radius_scaled') is not None},reverse=True),
                'fully_scored_proposal_attempts_include_cache_hits':True,
                'fully_scored_attempts_with_lower_total_cost':sum(v['full_assessment']['objective']<a['baseline']['objective'] for v in proposals if 'full_assessment' in v),
                'proposal_final_decision_counts':dict(reason),'guard_reason_counts_nonexclusive':dict(guard),
                'source_fit_record_rejection_counts_not_new_measurements':True})
    result={'schema':'hug_free_palm_exploration_trace_summary_v1','input_sha256':inputs,'cases':sorted(records,key=lambda c:c['source_inventory_index']),
        'scope':'Read-only arithmetic on immutable primary state/proposal/refusal records; no new geometry queries, tuning or candidate selection.',
        'new_eval_data':False,'physics_steps':0}
    with (OUT/'exploration_summary.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
