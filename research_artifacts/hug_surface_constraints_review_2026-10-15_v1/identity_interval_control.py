"""Manufactured identity control for the unchanged interval certificate.

No native proposal, robot pose, geometry query, optimizer, physics or eval data.
"""
from pathlib import Path
import datetime,hashlib,importlib.util,json
import numpy as np

OUT=Path(__file__).parent
SOURCE=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/sampling/cell_intervals.py')
RULE=SOURCE.parent.parent/'acceptance/acceptance_rule.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('unchanged_frozen_cell_intervals',SOURCE)
cells=importlib.util.module_from_spec(spec);spec.loader.exec_module(cells)

def main():
    assert not (OUT/'identity_interval_control.json').exists()
    original_hash=sha(SOURCE)
    # A manufactured cell with identical native and target center distances.
    # Its radius straddles the fixed 2mm proximity threshold in both fields.
    native=np.array([.002]);target=native.copy();radius=np.array([.0005]);area=np.array([1.])
    band=.002;eps=1e-8
    nd,np_=cells.masks(native,radius,band,eps);md,mp=cells.masks(target,radius,band,eps)
    assert np.array_equal(native,target)
    assert nd.tolist()==[False] and np_.tolist()==[True]
    assert md.tolist()==[False] and mp.tolist()==[True]
    result=cells.pair_intervals(native,target,radius,area,band,eps)
    assert result['lost_native_area']['lower_mm2']==0. and result['new_mapped_area']['lower_mm2']==0.
    assert result['lost_native_area']['upper_mm2']==1. and result['new_mapped_area']['upper_mm2']==1.
    # Every admissible mapped interval classification leaves at least one
    # possible mismatch when the native interval straddles the threshold.
    truth=[]
    for label,mdef,mpossible in [('far',False,False),('ambiguous',False,True),('near',True,True)]:
        lost_upper=(True and not mdef)
        added_upper=(mpossible and not False)
        assert lost_upper or added_upper
        truth.append({'mapped_class':label,'mapped_definite':mdef,'mapped_possible':mpossible,'lost_upper_mask':lost_upper,'added_upper_mask':added_upper})
    # Identity fields that are resolved do not have this certificate obstacle.
    resolved=[]
    for label,distance in [('near',.0002),('far',.004)]:
        d=np.array([distance]);r=cells.pair_intervals(d,d.copy(),radius,area,band,eps)
        assert r['lost_native_area']['upper_mm2']==0. and r['new_mapped_area']['upper_mm2']==0.
        resolved.append({'class':label,'distance_m':distance,'lost_upper_mm2':r['lost_native_area']['upper_mm2'],'added_upper_mm2':r['new_mapped_area']['upper_mm2']})
    assert sha(SOURCE)==original_hash
    report={'schema':'hug_frozen_interval_identity_control_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'source_path':str(SOURCE),'source_sha256':original_hash,'acceptance_rule_sha256':sha(RULE),'script_sha256':sha(__file__),'identity_inputs':{'native_and_target_centroid_distance_m':native.tolist(),'centroid_radius_m':radius.tolist(),'area_mm2':area.tolist(),'band_m':band,'original_numerical_allowance_m':eps},'identity_outputs':result,'native_ambiguous_truth_table':truth,'resolved_identity_controls':resolved,'proposition':'For a positive-area native-ambiguous cell (Npossible=true,Ndefinite=false), zero lost upper requires Mdefinite=true. Mdefinite implies Mpossible=true, giving positive added upper. Therefore at least one upper mismatch remains for every mapped interval classification.','meaning':'The separate per-field interval estimator cannot certify relationship identity for this manufactured ambiguous cell, even when native and target distance fields are identical. This is a certificate-coverage limitation, not an actual mismatch or proof of impossible mapping.','scientific_criteria_changed':False,'new_surface_queries':0,'new_FK':0,'new_fitting':False,'physics_steps':0,'new_eval_data':False,'no_real_saved_case_measurement':True}
    with (OUT/'identity_interval_control.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'passed':True,'report_sha256':sha(OUT/'identity_interval_control.json'),'identity_lost_upper_mm2':1.,'identity_added_upper_mm2':1.,'native_ambiguous_classes_checked':3}))

if __name__=='__main__':main()
