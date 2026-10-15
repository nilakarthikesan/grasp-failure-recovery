"""Recompute descriptive totals and retain deterministic illustrative witnesses."""
from pathlib import Path
import collections,hashlib,json
ROOT=Path('/private/tmp/hug_exposed_geometry_2026-10-15_v1')
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=read(ROOT/'summary.json');w=read('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json')
poses=[(i,p) for i in [8,12,16,20,24,28] for p in read(ROOT/f'case_{i}.json')['poses']]
aggregate={};examples=[]
for role in ['original','pooled']:
    pairs=[(i,r) for i,p in poses if p['pose_role']==role for r in p['rows']];rows=[r for i,r in pairs]
    a={'row_count':len(rows),'exposure':dict(collections.Counter(r['part_nearest_component_candidate']['exposure_status'] for r in rows)),
       'occupancy':dict(collections.Counter(r['occupancy_status'] for r in rows)),
       'by_kind':{},'unique_part_coordinate_sites':len({(i,r['expected_part'],tuple(r['query_object_m'])) for i,r in pairs})}
    for kind in ['proximity','orientation']:
        rr=[r for r in rows if r['row_kind']==kind]
        b={'n':len(rr),'exposure':dict(collections.Counter(r['part_nearest_component_candidate']['exposure_status'] for r in rr)),
           'strict_inside':sum(r['occupancy_status']=='STRICT_COMPONENT_INTERIOR' for r in rr),'historical_passed':sum(r['historical_literal_row_passed'] for r in rr)}
        if kind=='proximity':
            b['persist_by_direction']={d:{'rows':sum(r['source_relationship_change']==d for r in rr),'persist':sum(r['source_relationship_change']==d and r['conditional_whole_cell_counterexample_persists'] for r in rr)} for d in ['lost_native','new_mapped']}
        else:b['historical_valid']=sum(r['historical_orientation_valid'] for r in rr)
        a['by_kind'][kind]=b
    assert a==summary['aggregate'][role],role
    aggregate[role]=a
for index,p in poses:
    # Illustration rule: first persistent proximity row in frozen source order.
    r=next(r for r in p['rows'] if r.get('conditional_whole_cell_counterexample_persists'))
    native=w['whole_cell_rows'][int(r['saved_witness_json_pointer'].rsplit('/',1)[-1])]['row']
    examples.append({'index':index,'pose_role':p['pose_role'],'selection_rule':'first persistent proximity row in unchanged source order; illustrative only',
        'saved_witness_json_pointer':r['saved_witness_json_pointer'],'source_report_sha256':r['source_report_sha256'],
        'part':r['expected_part'],'change':r['source_relationship_change'],'band_mm':r['band_mm'],
        'native_centroid_distance_m':native['native_centroid_distance_m'],'native_cell_radius_plus_allowance_m':native['centroid_radius_m']+native['numerical_allowance_m'],
        'mapped_exposed_cell_lower_m':r['conditional_exposed_subset_cell_lower_bound_m'],
        'mapped_exposed_cell_upper_m':r['conditional_exposed_subset_cell_upper_bound_m'],
        'conditional_counterexample_persists':True})
report={'passed':True,'aggregate':aggregate,'all_six_cases_have_surviving_counterexamples_at_both_poses':True,'illustrative_examples':examples,
        'source_bindings':{str(ROOT/'summary.json'):sha(ROOT/'summary.json'),str(HERE/'diagnostic_output_review.json'):sha(HERE/'diagnostic_output_review.json')},
        'limitations':['Counts are not an improvement score: capped witnesses include repeated cells/sites/bands/resolutions, and do not measure total mismatch area.',
                       'A candidate can make a retained cell straddle a threshold or have an unavailable exposed upper bound; failure to prove persistence is not repair.',
                       'The native labels remain those of the historical MANO triangle reference. This is not native occupied-union certification.',
                       'Exposed closest-point status does not establish normal uniqueness, force closure, contact outcome, or global mapping validity.'],
        'new_geometry_queries':0,'new_eval_data':False}
out=HERE/'aggregate_review.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'sha256':sha(out),'examples':examples},indent=2))
