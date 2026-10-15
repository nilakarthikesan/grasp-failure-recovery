"""Pure saved-output classification: no model imports, geometry, selection or gate changes."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def classify(proximity,orientation_fields,orientation,prox_rows,ori_rows,protocol):
    near=np.array([r['row']['relationship_change']=='lost_native' for r in prox_rows],bool)
    band=np.array([r['row']['band_mm']/1000 for r in prox_rows]);lower=np.array(proximity['lower_m']);upper=np.array(proximity['upper_m'])
    passed=np.asarray(proximity['passed'],bool)
    assert np.array_equal(passed,np.where(near,upper<=band,lower>band))
    lost=near&(lower>band);added=(~near)&(upper<=band);uncertain=(~passed)&(~lost)&(~added)
    assert np.all(passed.astype(int)+lost+added+uncertain==1)
    valid=np.asarray(orientation_fields['valid'],bool);score=np.asarray(orientation_fields['score'],float);distance=np.asarray(orientation_fields['distance_m'],float)
    signs=np.sign([r['row']['native_facing_compatibility'] for r in ori_rows]);bands=np.array([r['band_mm']/1000 for r in ori_rows]);deadzone=protocol['facing_dot_numerical_deadzone']
    inband=(distance<=bands)&(distance>protocol['query_boundary_tolerance_m']);eligible=valid&inband
    signed=signs*score;preserved=np.asarray(orientation['passed'],bool)
    opposite=eligible&(signed < -deadzone);uncertain_sign=eligible&(~preserved)&(~opposite)
    unavailable=(~eligible)
    assert np.all(preserved.astype(int)+opposite+uncertain_sign+unavailable==1)
    assert not np.any(preserved&~eligible)
    perpart={}
    for part in ('palm','index','middle','ring','little','thumb'):
        pi=np.asarray([r['row']['anatomical_part']==part for r in prox_rows],bool)
        oi=np.asarray([r['row']['native_part']==part for r in ori_rows],bool)
        perpart[part]={'proximity_rows':int(pi.sum()),'preserved':int((pi&passed).sum()),'definite_lost':int((pi&lost).sum()),'definite_added':int((pi&added).sum()),'proximity_unresolved':int((pi&uncertain).sum()),'orientation_rows':int(oi.sum()),'orientation_preserved':int((oi&preserved).sum()),'valid_opposite_within_source_band':int((oi&opposite).sum()),'orientation_ineligible_or_outside_source_band':int((oi&unavailable).sum())}
    return {'proximity':{'total':len(passed),'preserved':int(passed.sum()),'definite_lost':int(lost.sum()),'definite_added':int(added.sum()),'unresolved':int(uncertain.sum()),'definite_lost_indices':np.flatnonzero(lost).tolist(),'definite_added_indices':np.flatnonzero(added).tolist(),'unresolved_indices':np.flatnonzero(uncertain).tolist()},
        'orientation':{'total':len(valid),'preserved':int(preserved.sum()),'valid_opposite_within_source_band':int(opposite.sum()),'valid_sign_unresolved_within_source_band':int(uncertain_sign.sum()),'invalid_eligibility':int((~valid).sum()),'eligible_outside_source_band':int((valid&~inband).sum()),'ineligible_or_outside_source_band':int(unavailable.sum()),'opposite_indices':np.flatnonzero(opposite).tolist(),'invalid_indices':np.flatnonzero(~valid).tolist(),'no_orientation_evidence':len(valid)==0},'per_part':perpart}

def main():
    p=read(HERE/'protocol.json');result=read(HERE/'results.json');saved=read(p['saved_witnesses']);orientation=read(p['orientation_protocol']);rows=[]
    assert result['protocol_sha256']==sha(HERE/'protocol.json')
    for item in result['cases']:
        assert sha(item['path'])==item['sha256'];c=read(item['path']);target=c['frozen_pose'];w=c['current_witness']
        prox=[saved['whole_cell_rows'][i] for i in target['proximity_source_indices']];ori=[saved['orientation_rows'][i] for i in target['orientation_source_indices']]
        classified=classify(w['proximity'],w['exact_orientation_fields'],w['exact_orientation'],prox,ori,orientation)
        old=c['historical_reference_witness'];pd=np.array(w['proximity_distance_m'])-np.array(old['proximity_distance_m'])
        od=np.array(w['exact_orientation_fields']['distance_m'])-np.array(old['orientation_distance_m'])
        sd=np.array(w['exact_orientation_fields']['score'])-np.array(old['orientation_score'])
        rows.append({'pose_id':c['pose_id'],'source_inventory_index':c['source_inventory_index'],'role':target['role'],**classified,
          'tip_status':c['current_tip_status_against_literal_historical_caps'],'historical_tip_cap_excess_m':c['historical_tip_cap_excess_m'],
          'historical_root_delta_m':c['root_position_change_from_saved_m'],'historical_root_rotation_matrix_delta':c['root_rotation_matrix_change_from_saved'],
          'max_absolute_proximity_distance_change_from_historical_m':float(np.max(np.abs(pd),initial=0)),
          'max_absolute_orientation_distance_change_from_historical_m':float(np.max(np.abs(od),initial=0)),
          'max_absolute_orientation_score_change_from_historical':float(np.max(np.abs(sd),initial=0)),
          'historical_proximity_passed':int(np.asarray(old['proximity_passed']).sum()),'historical_orientation_passed':int(np.asarray(old['orientation_passed']).sum()),
          'historical_orientation_invalid':int((~np.asarray(old['orientation_valid'],bool)).sum())})
    report={'schema':'hug_current_representation_saved_witness_summary_v1','primary_result_sha256':sha(HERE/'results.json'),'summary_code_sha256':sha(__file__),'poses':rows,
        'interpretation':'Counts are overlapping retained witness rows, not independent grasps, area, success labels or generalization evidence. Definite proximity changes use unchanged interval bounds. Invalid direction eligibility is distinct from a reliable opposite direction. Source precision and frames differ for current nominal poses; original cap values remain literal and unchanged.',
        'physics_steps':0,'new_queries':0,'full_current_mapping_status':'NOT_CERTIFIED_BY_RETAINED_SUBSET'}
    out=HERE/'saved_output_summary.json';assert not out.exists();out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for r in rows:print(json.dumps({'pose':r['pose_id'],'proximity':r['proximity'],'orientation':r['orientation'],'mean_tip_m':r['tip_status']['mean_m']}))
    print('summary_sha256='+sha(out))
if __name__=='__main__':main()
