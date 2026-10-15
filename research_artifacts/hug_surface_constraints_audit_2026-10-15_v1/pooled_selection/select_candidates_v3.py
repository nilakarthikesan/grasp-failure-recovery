"""Deterministic selection of already independently reviewed V1/V2 final states.

No fitting, geometric query, simulation, label generation, or criterion change.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,math,re
OUT=Path(__file__).parent
V1=Path('/private/tmp/hug_surface_constraints_2026-10-15_v1/candidate_targets.json')
V2=Path('/private/tmp/hug_surface_constraints_2026-10-15_v2/candidate_targets.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def choose(ranks):
    assert len(ranks)==2 and all(len(r)==5 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in r) for r in ranks)
    return min(range(2),key=lambda version:(tuple(ranks[version]),version))
def selftest():
    rows=[([[0,2,10,5,5],[1,0,0,0,0]],0,'feasible_first'),([[1,0,10,5,5],[1,1,0,0,0]],0,'eligible_count_first'),([[1,0,2,8,8],[1,0,3,0,0]],0,'worst_exact_violation'),([[1,0,2,2,8],[1,0,2,3,0]],0,'failed_rows'),([[1,0,2,2,8],[1,0,2,2,3]],1,'pose_final'),([[1,0,2,2,3],[1,0,2,2,3]],0,'stable_V1_tie')]
    for ranks,want,label in rows:assert choose(ranks)==want,label
    return [{'test':label,'pass':True} for _,_,label in rows]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--review-receipt',type=Path,nargs='+');args=ap.parse_args()
    if args.selftest:write(OUT/'policy_controls.json',{'scope':'Synthetic rank selection controls only; no native queries or states','checks':selftest(),'all_checks_pass':True});return
    assert args.review_receipt, 'Independent numerical PASS receipts required before final selection'
    policy=read(OUT/'policy_v3.json');assert sha(__file__)==policy['selection_script_sha256'] and sha(V1)==policy['V1_candidate_sha256'];assert not (OUT/'protocol.json').exists() and not (OUT/'candidate_targets.json').exists()
    candidates=[read(V1),read(V2)];paths=[V1,V2];input_hashes={str(OUT/'policy_v3.json'):sha(OUT/'policy_v3.json'),str(OUT/'policy_controls.json'):sha(OUT/'policy_controls.json'),str(Path(__file__)):sha(__file__)}
    def collect(v):
        if isinstance(v,dict):
            for k,w in v.items():
                if isinstance(k,str) and k.startswith('/') and isinstance(w,str) and re.fullmatch('[a-f0-9]{64}',w):assert sha(k)==w,k;input_hashes[k]=w
                collect(w)
            if isinstance(v.get('path'),str) and isinstance(v.get('sha256'),str) and Path(v['path']).is_file():assert sha(v['path'])==v['sha256'];input_hashes[v['path']]=v['sha256']
        elif isinstance(v,list):
            for w in v:collect(w)
    protocols=[]
    for path,c in zip(paths,candidates):
        assert c['schema']==('hug_saved_surface_witness_constraint_candidates_v1' if path==V1 else 'hug_saved_surface_witness_constraint_candidates_v2') and c['physics_steps']==0 and not c['new_inference'] and not c['new_eval_data']
        pp=path.parent/'protocol.json';p=read(pp);assert c['protocol_sha256']==sha(pp);assert p['scientific_acceptance_rule_sha256']==policy['scientific_acceptance_rule_sha256'];protocols.append(p)
        input_hashes[str(path)]=sha(path);input_hashes[str(pp)]=sha(pp);collect(p);collect(c)
        assert sorted(x['source_inventory_index'] for x in c['cases'])==policy['case_indices']
    root_rule=Path(policy['root_selection_rule']['path']);assert sha(root_rule)==policy['root_selection_rule']['sha256'];input_hashes[str(root_rule)]=sha(root_rule)
    receipts=[];passed_receipts=[]
    for path in args.review_receipt:
        receipt=read(path);assert receipt.get('passed') is True, 'Numerical review receipt must declare PASS';collect(receipt);input_hashes[str(path)]=sha(path);receipts.append({'path':str(path),'sha256':sha(path)});passed_receipts.append(receipt)
    for path in paths:
        assert any(receipt.get('source_bindings',{}).get(str(path))==sha(path) for receipt in passed_receipts), 'Both exact source candidate hashes must be covered by numerical PASS receipts'
    maps=[{x['source_inventory_index']:x for x in c['cases']} for c in candidates]
    # Comparability and literal eligibility only, before any ranking.
    for index in policy['case_indices']:
        a,b=[m[index] for m in maps]
        for field in ('native_sha256','joint_names','original_qpos_rad','authoritative_original_tip_caps_sqrt_einsum_m','bounds_scaled','domain','constraint_source_rows','original_root_target_position_m','original_root_target_quaternion_wxyz','root_target_quaternion_wxyz'):
            assert a[field]==b[field],(index,field)
        for c in (a,b):
            d=c['selected_constraint_diagnostics'];assert d['in_domain'] and d['tip_status']['passed'] and d['tip_status']['mean_pass'] and all(d['tip_status']['individual_pass'])
            assert all(v<=cap for v,cap in zip(c['selected_tip_norms_sqrt_einsum_m'],c['authoritative_original_tip_caps_sqrt_einsum_m']))
            assert c['audited_pose_role'] in ('surface_constraint_selected','tip_feasible_constraint_refused_diagnostic','original_fallback')
            assert d['rank'][0]==int(not d['all_witness_constraints_pass']) and d['rank'][1]==sum(not x for x in d['exact_orientation_fields']['valid']) and d['rank'][3]==sum(not x for x in d['literal_surface_pass'])
    protocol={'schema':'hug_surface_constraints_pooled_selection_protocol_v1','policy_sha256':sha(OUT/'policy_v3.json'),'input_sha256':input_hashes,'review_receipts':receipts,'selection_rule':policy['selection_rule'],'scientific_acceptance_rule_sha256':policy['scientific_acceptance_rule_sha256'],'frozen_utc_before_ranking':datetime.now(timezone.utc).isoformat(),'source_versions':['V1','V2'],'new_eval_data':False,'physics_steps':0,'new_inference':False,'production_promotion':False}
    write(OUT/'protocol.json',protocol)
    selected=[];rows=[]
    for index in policy['case_indices']:
        options=[m[index] for m in maps];ranks=[c['selected_constraint_diagnostics']['rank'] for c in options];version=choose(ranks);chosen=copy.deepcopy(options[version]);original_selection_source=chosen.get('selection_source')
        chosen['selection_source']={'pooled_source_version':['V1','V2'][version],'source_candidate_path':str(paths[version]),'source_candidate_sha256':sha(paths[version]),'source_protocol_sha256':candidates[version]['protocol_sha256'],'original_selection_source':original_selection_source,'V1_rank':ranks[0],'V2_rank':ranks[1],'stable_V1_tie':ranks[0]==ranks[1]}
        selected.append(chosen);rows.append({'source_inventory_index':index,**chosen['selection_source'],'audited_pose_role':chosen['audited_pose_role'],'constraint_selection_accepted':chosen['constraint_selection_accepted']})
    result={'schema':'hug_pooled_surface_constraint_candidates_v1','protocol_sha256':sha(OUT/'protocol.json'),'cases':selected,'selection_rows':rows,'new_eval_data':False,'new_inference':False,'physics_steps':0,'production_changed':False,'mapping_acceptance':'NOT_ESTABLISHED_REQUIRES_UNCHANGED_FULL_GRID_AUDIT'}
    write(OUT/'candidate_targets.json',result);write(OUT/'selection_report.json',{'schema':'hug_surface_constraints_pooling_report_v1','candidate_sha256':sha(OUT/'candidate_targets.json'),'protocol_sha256':sha(OUT/'protocol.json'),'records':rows,'pure_selection_no_new_optimization':True,'new_eval_data':False,'physics_steps':0})
    print(json.dumps({'candidate_sha256':sha(OUT/'candidate_targets.json'),'protocol_sha256':sha(OUT/'protocol.json'),'selected_versions':{r['source_inventory_index']:r['pooled_source_version'] for r in rows}},indent=2))
if __name__=='__main__':main()
