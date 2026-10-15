"""Deterministic two-cohort merger; compact selected poses, lossless full records."""
from pathlib import Path
import json,hashlib,gzip
import numpy as np
OUT=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def main():
    p=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json');records=[];provenance=[];rows=[]
    for indices in p['allowed_cohorts']:
        path=OUT/('candidate_cohort_'+'_'.join(map(str,indices))+'.json');c=read(path)
        assert c['protocol_sha256']==ph and c['cohort_case_indices']==indices and [v['source_inventory_index'] for v in c['cases']]==indices
        assert c['guards']['actual']==0 and c['protected_hashes_unchanged']
        raw=path.read_bytes();compressed=gzip.compress(raw,mtime=0);gz=path.with_suffix('.json.gz')
        with gz.open('xb') as f:f.write(compressed)
        assert gzip.decompress(compressed)==raw
        provenance.append({'path':str(path),'sha256':sha(path),'gzip_path':str(gz),'gzip_sha256':sha(gz),
            'runtime_seconds':c['runtime_seconds'],'source_input_hashes_rechecked':True})
        for full in c['cases']:
            index=full['source_inventory_index'];audit=full['solver_audit'];evidence=full['keyed_direction_evidence']
            for callback in evidence['callbacks']:
                trace=Path(callback['trace_path']);assert sha(trace)==callback['trace_compressed_sha256']
                rawtrace=gzip.decompress(trace.read_bytes());assert hashlib.sha256(rawtrace).hexdigest()==callback['trace_uncompressed_sha256']
                tracevalue=json.loads(rawtrace);assert tracevalue['state_scaled']==callback['state_scaled']
            compact={k:v for k,v in full.items() if k not in ('surface_reference_by_part','direction_reference_by_part','keyed_direction_evidence','solver_audit')}
            compact['solver_audit']={k:v for k,v in audit.items() if k not in ('proposals','events')}
            compact['complete_record_binding']={'cohort_path':str(path),'cohort_sha256':sha(path),'gzip_path':str(gz),'gzip_sha256':sha(gz),'source_inventory_index':index,'not_repeated_in_compact':'Full proposals, events, selected quadrature/direction reference and all per-callback gzip trace bindings'}
            records.append(compact)
            rows.append({'source_inventory_index':index,'object_id':full['object_id'],'source_case':full['source_case'],
                'selected_new_pose':audit['surface_candidate_returned'],'palm_displacement_world_mm':(np.asarray(full['palm_displacement_world_m'])*1000).tolist(),
                'palm_displacement_norm_mm':full['palm_displacement_norm_m']*1000,
                'original_objective':audit['baseline']['objective'],'selected_objective':audit['final_assessment']['objective'],
                'original_mean_tip_mm':audit['baseline']['mean_fingertip_error_m']*1000,'selected_mean_tip_mm':audit['final_assessment']['mean_fingertip_error_m']*1000,
                'individual_tip_delta_mm':(np.asarray(audit['final_assessment']['tip_error_norm_m'])-np.asarray(audit['baseline']['tip_error_norm_m'])).tolist(),
                'tip_delta_array_units':'m; retained explicit key correction below','all_original_exact_caps_pass':audit['final_assessment']['individual_tip_nonworsening_pass'],
                'termination':audit['termination'],'callback_or_geometry_refusal':audit['callback_or_geometry_refusal'],
                'counts':audit['counters'],'guard_refusals':sum(v.get('refusal')=='candidate_guard_refused' for v in audit['proposals']),
                'proposal_count':len(audit['proposals']),'accepted_count':len(audit['accepted_candidates']),
                'prior_original_caps_reproduced_exactly':full['prior_original_caps_reproduced_exactly'],
                'analytic_jacobian_gate_pass':full['analytic_jacobian_baseline_check']['passed'],
                'analytic_jacobian_maxabs_m_per_state':full['analytic_jacobian_baseline_check']['maxabs_difference_m_per_scaled_state'],
                'mj_forward_calls_in_set_pose':full['mj_forward_calls_in_set_pose']})
            rows[-1]['individual_tip_delta_mm']=(np.asarray(rows[-1]['individual_tip_delta_mm'])*1000).tolist();rows[-1].pop('tip_delta_array_units')
    records.sort(key=lambda v:v['source_inventory_index']);rows.sort(key=lambda v:v['source_inventory_index'])
    assert [c['source_inventory_index'] for c in records]==[8,12,16,20,24,28]
    for table in ('source_sha256','implementation_sha256'):
        for path,h in p[table].items():assert sha(path)==h,path
    result={'schema':'hug_free_palm_selected_targets_v1','protocol_sha256':ph,'cases':records,'cohort_provenance':provenance,
        'new_eval_data':False,'new_generator_proposals':False,'physics_steps':0,'mapping_acceptance':False,'production_changes':False,
        'note':'Static development selection only; independent frozen geometry audit required. Complete trace/refusal evidence losslessly bound in cohort gzip and callback gzip files.'}
    write(OUT/'candidate_targets.json',result)
    summary={'schema':'hug_free_palm_fit_summary_v1','protocol_sha256':ph,'candidate_sha256':sha(OUT/'candidate_targets.json'),
        'cases':rows,'cohort_provenance':provenance,'total_full_callbacks':sum(r['counts']['full'] for r in rows),
        'total_cheap_FK_callbacks':sum(r['counts']['FK'] for r in rows),'total_solver_analytic_jacobian_callbacks':sum(r['counts']['analytic_Jacobians'] for r in rows),
        'total_mj_forward_calls_in_staging':sum(r['mj_forward_calls_in_set_pose'] for r in rows),
        'new_eval_data':False,'physics_steps':0,'training':False,'scientific_mapping_acceptance':False}
    write(OUT/'fit_summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
