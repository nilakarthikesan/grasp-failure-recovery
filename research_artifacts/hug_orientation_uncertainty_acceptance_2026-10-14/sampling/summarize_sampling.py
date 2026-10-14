"""Readable summaries of frozen static sampling report; no independent-site CI."""
from pathlib import Path
import json,hashlib
OUT=Path(__file__).parent

def main():
    p=json.loads((OUT/'protocol.json').read_text());r=json.loads((OUT/'sampling_uncertainty.json').read_text());required=['palm','index','middle','ring','little','thumb'];records=[];repeats=[]
    for case in r['cases']:
        by={}
        for density,res in case['resolutions'].items():
            row={'case_id':case['case_id'],'object_id':case['object_id'],'density_mm':int(density),'query_count':res['query_count'],'surface_area_mm2':res['surface_area_mm2'],'cell_radius_mm':res['cell_radius_mm'],'bands':{},'numeric_budget_sensitivity':{}}
            for eps in p['numerical_allowances_m']:
                er=res['numerical_allowances'][str(eps)];sen={}
                for band in p['bands_mm']:
                    w=er['bands'][str(band)]['whole_hand_unsigned_proximity'];parts=er['bands'][str(band)]['independent_parts'];fail=[part for part in required if parts[part]['lost_native_area']['lower_mm2']>0 or parts[part]['new_mapped_area']['lower_mm2']>0]
                    sen[str(band)]={'parts_with_definite_relationship_change':fail,'whole_lost_lower_mm2':w['lost_native_area']['lower_mm2'],'whole_new_lower_mm2':w['new_mapped_area']['lower_mm2'],'whole_ambiguous_area_mm2':w['whole_relationship_ambiguous_area_mm2'],'whole_iou_interval':w['iou_interval']}
                    if eps==p['primary_allowance_m']:
                        row['bands'][str(band)]={'whole':w,'whole_ambiguous_area_fraction':w['whole_relationship_ambiguous_area_mm2']/res['surface_area_mm2'],'anatomical_parts_with_definite_change':fail,'required_anatomical_part_intervals':{part:parts[part] for part in required},'strict_gate_status':'FAIL_CONDITIONAL' if fail else 'UNRESOLVED'}
                row['numeric_budget_sensitivity'][str(eps)]=sen
            row['all_bands_strict_gate_status']='FAIL_CONDITIONAL' if any(b['anatomical_parts_with_definite_change'] for b in row['bands'].values()) else 'UNRESOLVED'
            records.append(row);by[density]=row
        if '1' in by:
            changes={}
            for band in p['bands_mm']:
                coarse=by['2']['bands'][str(band)]['whole'];fine=by['1']['bands'][str(band)]['whole'];co=coarse['centroid_estimates'];fi=fine['centroid_estimates']
                changes[str(band)]={'centroid_iou_fine_minus_coarse':fi['iou']-co['iou'],'centroid_native_area_fine_minus_coarse_mm2':fi['native_area_mm2']-co['native_area_mm2'],'centroid_mapped_area_fine_minus_coarse_mm2':fi['mapped_area_mm2']-co['mapped_area_mm2'],'ambiguous_area_fine_minus_coarse_mm2':fine['whole_relationship_ambiguous_area_mm2']-coarse['whole_relationship_ambiguous_area_mm2'],'iou_interval_coarse':coarse['iou_interval'],'iou_interval_fine':fine['iou_interval']}
            repeats.append({'case_id':case['case_id'],'bands':changes,'meaning':'Nested geometricdensitydiagnostics, notindependentstatisticalobservations or continuumconvergenceproof.'})
    summary={'schema':'hug_sampling_conditional_summary_v1','source_report_sha256':hashlib.sha256((OUT/'sampling_uncertainty.json').read_bytes()).hexdigest(),'protocol_sha256':r['protocol_sha256'],'primary_numerical_allowance_m':p['primary_allowance_m'],'conditional_on_uniform_numeric_error_allowance':True,'case_count':len(r['cases']),'resolution_count':len(records),'coarse_case_count':sum(x['density_mm']==2 for x in records),'finer_repeat_count':sum(x['density_mm']==1 for x in records),'native_carrier_reference_available':False,'required_parts':required,'part_areas_not_additive':True,'records':records,'density_repeats':repeats,'strict_fail_resolution_count':sum(x['all_bands_strict_gate_status']=='FAIL_CONDITIONAL' for x in records),'new_eval_data':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps({'strict_fail_resolution_count':summary['strict_fail_resolution_count'],'resolution_count':len(records)}))
if __name__=='__main__':main()
