"""Summarize all six guarded development cases without hiding failures."""
from pathlib import Path
import gzip,json,hashlib
BASE=Path(__file__).parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[BASE/'candidate_validation_8,12,16.json.gz',BASE/'candidate_validation_20,24,28.json.gz']
reports=[json.loads(gzip.decompress(p.read_bytes())) for p in files]
assert all(r['protocol_sha256']==sha(BASE/'protocol.json') for r in reports)
assert len({r['candidate_sha256'] for r in reports})==1
assert len({r['script_sha256'] for r in reports})==1
records=sum([r['cases'] for r in reports],[]);assert len(records)==6;assert len({r['native_sha256'] for r in records})==6
p=read(BASE/'protocol.json');sp=read(Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/sampling/protocol.json'));eps=str(sp['primary_allowance_m']);parts=['palm','thumb','index','middle','ring','little'];result=[]
def amount(prox,band):
    x=prox[eps]['bands'][str(band)]['independent_parts']
    return {'sum_independent_part_lost_area_lower_mm2':sum(x[part]['lost_native_area']['lower_mm2'] for part in parts),'sum_independent_part_added_area_lower_mm2':sum(x[part]['new_mapped_area']['lower_mm2'] for part in parts),'per_part':{part:{'lost_lower_mm2':x[part]['lost_native_area']['lower_mm2'],'added_lower_mm2':x[part]['new_mapped_area']['lower_mm2'],'native_retention_interval':x[part]['native_retention_interval']} for part in parts},'note':'Part masks may overlap; sums are sums of six separate anatomical measurements, not unique object area.'}
def normal(ori,band):
    c=ori['bands'][str(band)]['common_same_label'];s=c['pointwise_sign_check'];return {'candidate':c['candidate_denominator'],'valid':c['valid_denominator'],'outside_visual_eligible':s['eligible_outside_visual_denominator'],'sampled_sign_reversals':s['flip_denominators'],'sampled_reversal_count':sum(x['query_count'] for x in s['flip_denominators'].values()),'note':'Eligibility can change after mapping; fewer eligible flips alone does not establish improved direction preservation. Matching points do not bound continuous normals.'}
for c in sorted(records,key=lambda c:c['source_inventory_index']):
    rec={'case_id':c['case_id'],'native_sha256':c['native_sha256'],'source_inventory_index':c['source_inventory_index'],'engineering_guard_audit':c['engineering_guard_independent_audit'],'root_policy':c['root_policy'],'resolutions':{}}
    for density,r in c['resolutions'].items():
        rr={'query_count':r['query_count'],'query_cache_sha256':r['query_cache_sha256'],'strict_status':r['frozen_acceptance_application']['strict_relationship_status_conditional'],'proximity_status':r['frozen_acceptance_application']['proximity_status_conditional'],'pointwise_orientation_status':r['frozen_acceptance_application']['pointwise_orientation_status'],'baseline_strict_status':r['baseline_acceptance']['strict_relationship_status_conditional'],'bands':{}}
        for band in (1,2,5):
            oldp=r['baseline_conditional_sampling'];newp=r['conditional_sampling'];oldw=oldp[eps]['bands'][str(band)]['whole_hand_unsigned_proximity'];neww=newp[eps]['bands'][str(band)]['whole_hand_unsigned_proximity'];rr['bands'][str(band)]={'baseline_unsigned_whole_IoU_centroid':oldw['centroid_estimates']['iou'],'candidate_unsigned_whole_IoU_centroid':neww['centroid_estimates']['iou'],'baseline_same_part':amount(oldp,band),'candidate_same_part':amount(newp,band),'baseline_orientation':normal(r['baseline_orientation'],band),'candidate_orientation':normal(r['orientation'],band)}
        rec['resolutions'][density]=rr
    result.append(rec)
statuses=[r['strict_status'] for c in result for r in c['resolutions'].values()];assert len(statuses)==9
summary={'schema':'hug_guarded_refinement_static_summary_v1','source_reports_sha256':{p.name:sha(p) for p in files},'candidate_sha256':reports[0]['candidate_sha256'],'validation_protocol_sha256':sha(BASE/'protocol.json'),'summary_script_sha256':sha(__file__),'case_count':6,'resolution_count':len(statuses),'strict_fail_resolution_count':statuses.count('FAIL'),'strict_pass_resolution_count':statuses.count('PASS'),'strict_unresolved_resolution_count':statuses.count('UNRESOLVED'),'all_existing_engineering_tip_guards_pass':all(c['engineering_guard_audit']['existing_engineering_guard_status']=='PASS' for c in result),'min_candidate_mean_tip_mm':min(c['engineering_guard_audit']['mean_fingertip_error_m']*1000 for c in result),'max_candidate_mean_tip_mm':max(c['engineering_guard_audit']['mean_fingertip_error_m']*1000 for c in result),'physics_steps':0,'guards_actual_attempts':sum(r['guards']['actual'] for r in reports),'new_eval_data':False,'new_inference':False,'training':False,'full_contact_equivalence':'UNRESOLVED','failure_label_preservation':'NOT_TESTED','independent_generalization':'NOT_TESTED','cases':result}
(BASE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))
for c in result:
    r=c['resolutions']['2'];b=r['bands']['2'];old=b['baseline_same_part'];new=b['candidate_same_part'];print(json.dumps({'case':c['case_id'],'strict_status':r['strict_status'],'mean_tip_mm':c['engineering_guard_audit']['mean_fingertip_error_m']*1000,'baseline_unsigned_whole_IoU_2mm':b['baseline_unsigned_whole_IoU_centroid'],'candidate_unsigned_whole_IoU_2mm':b['candidate_unsigned_whole_IoU_centroid'],'baseline_lost_part_area_lower_2mm':old['sum_independent_part_lost_area_lower_mm2'],'candidate_lost_part_area_lower_2mm':new['sum_independent_part_lost_area_lower_mm2'],'baseline_added_part_area_lower_2mm':old['sum_independent_part_added_area_lower_mm2'],'candidate_added_part_area_lower_2mm':new['sum_independent_part_added_area_lower_mm2'],'baseline_eligible_flips_2mm':b['baseline_orientation']['sampled_reversal_count'],'candidate_eligible_flips_2mm':b['candidate_orientation']['sampled_reversal_count']}))
