from pathlib import Path
import gzip,json,hashlib,datetime
OUT=Path(__file__).parent
PRIMARY=Path('/private/tmp/hug_free_palm_fit_2026-10-15_v1')
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
p=read(PRIMARY/'protocol.json');c=read(PRIMARY/'candidate_targets.json')
assert p['execution_authorized_by_root_review']
assert c['protocol_sha256']==sha(PRIMARY/'protocol.json')
assert not (OUT/'verification_protocol.json').exists() and not (OUT/'verification_report.json').exists()
inputs={}
for table in ('source_sha256','implementation_sha256','query_cache_sha256','full_cache_sha256'):inputs.update(p[table])
for name in ('protocol.json','candidate_targets.json','fit_summary.json'):inputs[str(PRIMARY/name)]=sha(PRIMARY/name)
for prov in c['cohort_provenance']:
    path=Path(prov['gzip_path']);raw=gzip.decompress(path.read_bytes());assert sha(path)==prov['gzip_sha256'] and hashlib.sha256(raw).hexdigest()==prov['sha256']
    inputs[str(path)]=sha(path);full=json.loads(raw)
    for row in full['cases']:
        for t in row['keyed_direction_evidence']['callbacks']:
            assert sha(t['trace_path'])==t['trace_compressed_sha256'];inputs[t['trace_path']]=t['trace_compressed_sha256']
prior=ROOT/'research_artifacts/hug_canonical_mcp_calibration_2026-10-14_v1/runtime_v3/candidate_targets.json'
assert sha(prior)=='84496804c00267024af56f3364457943696db8481b9c1853bca2c93c6c824b9d'
inputs[str(prior)]=sha(prior)
inputs[str(OUT/'verify_outputs.py')]=sha(OUT/'verify_outputs.py')
for path,h in inputs.items():assert sha(path)==h,path
result={'schema':'independent_hug_free_palm_output_verification_protocol_v1','freeze_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_before_first_independent_candidate_FK_or_trace_arithmetic':True,'execution_authorized_by_root':True,'script_sha256':sha(OUT/'verify_outputs.py'),'primary_protocol':str(PRIMARY/'protocol.json'),'candidate_path':str(PRIMARY/'candidate_targets.json'),'inventory_path':str(ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json'),'historical_candidate_path':str(prior),'source_urdf':str(ROOT/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf'),'input_sha256':inputs,'case_indices':[8,12,16,20,24,28],'review_preflight_sha256':sha(OUT/'preexecution_review.json'),'scope':'Independent actual action/native WORLD/captured-scene and source-URDF FK, complete27D trace and native direction guard arithmetic; no solver replay or surface query. Six already inspected development cases only.','old_caps_reconstruction':'Exact byte equality to independent prior canonical archive original-cap fields and source action reconstruction; candidate <= literal historical originalcaps zero allowance.','guards':['mj_step','mj_step1','mj_step2'],'new_eval_data':False,'scientific_mapping_acceptance':False}
(OUT/'verification_protocol.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'input_count':len(inputs),'protocol_sha256':sha(OUT/'verification_protocol.json')}))
