"""Read-only post-measurement aggregation; no geometry, fitting or selection."""
from pathlib import Path
import collections,hashlib,json
HERE=Path(__file__).resolve().parent
def aggregate():
    allposes=[]
    for i in [8,12,16,20,24,28]:allposes+=json.loads((HERE/f'case_{i}.json').read_text())['poses']
    summary={'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),'cohort':json.loads((HERE/'cohort_8_12_16_20_24_28.json').read_text()),'rows':[],'aggregate':{}}
    for role in ['original','pooled']:
        ps=[x for x in allposes if x['pose_role']==role];rs=[r for x in ps for r in x['rows']]
        ag={'row_count':len(rs),'exposure':dict(collections.Counter(r['part_nearest_component_candidate']['exposure_status'] for r in rs)),'occupancy':dict(collections.Counter(r['occupancy_status'] for r in rs)),'by_kind':{},'unique_part_coordinate_sites':sum(len({(r['expected_part'],tuple(r['query_object_m'])) for r in x['rows']}) for x in ps)}
        for kind in ['proximity','orientation']:
            sub=[r for r in rs if r['row_kind']==kind];ag['by_kind'][kind]={'n':len(sub),'exposure':dict(collections.Counter(r['part_nearest_component_candidate']['exposure_status'] for r in sub)),'strict_inside':sum(r['occupancy_status']=='STRICT_COMPONENT_INTERIOR' for r in sub),'historical_passed':sum(r['historical_literal_row_passed'] for r in sub)}
            if kind=='proximity':ag['by_kind'][kind]['persist_by_direction']={d:{'rows':sum(r['source_relationship_change']==d for r in sub),'persist':sum(r['source_relationship_change']==d and r['conditional_whole_cell_counterexample_persists'] for r in sub)} for d in ['lost_native','new_mapped']}
            else:ag['by_kind'][kind]['historical_valid']=sum(r['historical_orientation_valid'] for r in sub)
        summary['aggregate'][role]=ag
    for x in allposes:
        rs=x['rows'];prox=[r for r in rs if r['row_kind']=='proximity'];ori=[r for r in rs if r['row_kind']=='orientation']
        summary['rows'].append({'index':x['source_inventory_index'],'pose':x['pose_role'],'total':len(rs),'proximity_rows':len(prox),'orientation_rows':len(ori),'part_exposed':sum(r['part_exposed_subset_distance_exact_conditional'] for r in rs),'part_buried':sum(r['part_nearest_component_candidate']['exposure_status']=='STRICTLY_BURIED_IN_OTHER_COMPONENT' for r in rs),'proximity_counterexamples_persist':sum(r['conditional_whole_cell_counterexample_persists'] for r in prox),'persist_parts':dict(collections.Counter(r['expected_part'] for r in prox if r['conditional_whole_cell_counterexample_persists'])),'buried_parts':dict(collections.Counter(r['expected_part'] for r in rs if r['part_nearest_component_candidate']['exposure_status']=='STRICTLY_BURIED_IN_OTHER_COMPONENT')),'burier_parts':dict(collections.Counter(part for r in rs for part in r['part_nearest_component_candidate']['strict_burying_parts'])),'ori_buried':sum(r['part_nearest_component_candidate']['exposure_status']=='STRICTLY_BURIED_IN_OTHER_COMPONENT' for r in ori),'source_tip_mean_m':x['exact_mean_tip_m']})
    summary['all_bound_order_consistent']=all(r['bound_order_consistent'] for x in allposes for r in x['rows']);summary['no_oracle_inconsistency']=all(r['oracle_consistent'] for x in allposes for r in x['rows']);summary['component_point_queries']=sum(x['component_point_queries'] for x in allposes);summary['candidate_point_queries']=sum(x['candidate_point_queries'] for x in allposes)
    return summary
if __name__=='__main__':print(json.dumps(aggregate(),indent=2))
