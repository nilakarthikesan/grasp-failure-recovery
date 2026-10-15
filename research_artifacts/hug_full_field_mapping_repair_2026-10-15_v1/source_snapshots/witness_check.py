"""Fixed saved witness predicates; runtime geometry is injected, never reconstructed here."""
import numpy as np

PARTS=('palm','index','middle','ring','little','thumb','forearm_carrier','wrist_carrier')

def anatomical_part(body):
    leaf=body.split('/')[-1]
    if leaf=='forearm':return 'forearm_carrier'
    if leaf=='wrist':return 'wrist_carrier'
    for prefix,part in [('ff','index'),('mf','middle'),('rf','ring'),('lf','little'),('th','thumb')]:
        if leaf.startswith(prefix):return part
    if leaf=='palm':return 'palm'
    raise ValueError('Unknown active hand owner: '+body)

def descendant(parentids,body,root):
    while body>0:
        if body==root:return True
        body=int(parentids[body])
    return False

def fixed_rows(saved,index,native_hash):
    pi=[i for i,r in enumerate(saved['whole_cell_rows']) if r['source_inventory_index']==index]
    oi=[i for i,r in enumerate(saved['orientation_rows']) if r['source_inventory_index']==index]
    prox=[saved['whole_cell_rows'][i] for i in pi];ori=[saved['orientation_rows'][i] for i in oi]
    assert all(r['row']['native_sha256']==native_hash for r in prox)
    assert all(r['native_sha256']==native_hash for r in ori)
    assert all(r['row']['anatomical_part'] in PARTS[:6] for r in prox)
    assert all(r['row']['native_part'] in PARTS[:6] for r in ori)
    return pi,oi,prox,ori

def evaluate_fixed(hand,visual_tri,prox,ori,orientation_protocol,geometry,query_part,direction,core):
    """No row selection or threshold changes. Duplicate rows stay duplicate."""
    pq=np.asarray([r['row']['centroid_object_m'] for r in prox],float).reshape(-1,3)
    pd=np.empty(len(prox));pcp=np.empty_like(pq);pn=np.empty_like(pq)
    labels=[r['row']['anatomical_part'] for r in prox]
    for part in PARTS[:6]:
        ii=np.asarray([i for i,x in enumerate(labels) if x==part],int)
        if len(ii):pd[ii],pcp[ii],pn[ii]=query_part(geometry,[g for g in hand if g['part']==part],pq[ii])
    p=core.proximity_constraints(pd,[r['row']['centroid_radius_m'] for r in prox],
        [r['row']['numerical_allowance_m'] for r in prox],[r['row']['band_mm']/1000 for r in prox],
        [r['row']['relationship_change']=='lost_native' for r in prox])
    oq=np.asarray([r['row']['query_object_m'] for r in ori],float).reshape(-1,3)
    on=np.asarray([r['row']['object_outward_normal'] for r in ori],float).reshape(-1,3)
    expected=np.asarray([PARTS.index(r['row']['native_part']) for r in ori],int)
    score=np.asarray([r['row']['native_facing_compatibility'] for r in ori],float)
    assert np.all(np.abs(score)>orientation_protocol['facing_dot_numerical_deadzone'])
    if len(ori):fields=direction.mapped_direction_fields(oq,on,hand,visual_tri,orientation_protocol,expected_part=expected)
    else:fields={'valid':np.empty(0,bool),'score':np.empty(0),'distance_m':np.empty(0),'reason_masks':{}}
    o=core.orientation_constraints(fields['score'],fields['distance_m'],np.sign(score),
        [r['band_mm']/1000 for r in ori],orientation_protocol['facing_dot_numerical_deadzone'],
        orientation_protocol['query_boundary_tolerance_m'],fields['valid'])
    return {'proximity_distance_m':pd,'proximity_closest_object_m':pcp,'proximity_outward_normal':pn,
        'proximity':p,'exact_orientation_fields':fields,'exact_orientation':o,
        'literal_surface_pass':np.concatenate((p['passed'],o['passed'])),
        'summary':{'proximity_rows':len(prox),'proximity_passed':int(p['passed'].sum()),
            'orientation_rows':len(ori),'orientation_passed':int(o['passed'].sum()),
            'orientation_invalid':int((~fields['valid']).sum()),
            'orientation_coverage':'PRESENT_RETAINED_EXAMPLES' if len(ori) else 'UNAVAILABLE_NO_SAVED_RELIABLE_WITNESS',
            'all_retained_rows_pass':bool(p['passed'].all() and o['passed'].all()),
            'full_grid_mapping_acceptance':False}}
