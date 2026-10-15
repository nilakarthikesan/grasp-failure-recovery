from pathlib import Path
import hashlib,json,shutil,subprocess,datetime
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
B=W/'research_artifacts/dga_practical_retargeting_2026-10-15_v1'
C=Path('/private/tmp/dga_practical_retargeting_20261015_v1/control')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
prior=[]
for name,m in [('dexgrasp_anything_mapping_validation_2026-10-15_component_constraints_v1','evidence_manifest_v1.json'),('dexgrasp_anything_mapping_validation_2026-10-15_acceptance_feasibility_v2','evidence_manifest.json')]:
    r=W/'research_artifacts'/name;d=json.loads((r/m).read_text());bad=[]
    for rel,meta in d['files'].items():
        if not (r/rel).is_file() or 'sha256:'+sha(r/rel)!=meta['sha256']:bad.append(rel)
    assert not bad,bad
    prior.append({'study':name,'manifest_sha256':sha(r/m),'files_checked':len(d['files']),'mismatches':bad})
write(C/'prior_preservation_final_v2.json',{'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':prior,'all_prior_artifacts_unchanged':True})
for name in ['owned_doc_updates.json','prior_preservation_final_v2.json','close_archive.py']:
    shutil.copyfile(C/name,B/'mapper/control'/name)
reviews=['authoritative_all20_independent_review_v1.json','positive_row4_thumb_independent_review_v1.json']
for name in reviews:
    assert (B/'protocol'/name).is_file(),name
readme=B/'README.md';text=readme.read_text()
text+='''

Final independent review verifies 385 aggregate checks and 2,159,607 retained leaf bounds across disjoint partitions, with a separate 41-check positive-result audit using independently coded scalar FK. [All-output review](protocol/authoritative_all20_independent_review_v1.json) and [positive thumb review](protocol/positive_row4_thumb_independent_review_v1.json) bind the final summaries and selected poses. [Exact replay commands and environment notes](mapper/control/reproducibility_commands_v1.txt) and [source/dependency audit](mapper/control/source_dependency_audit_v1.json) describe replay prerequisites. One inherited prose sentence in the original-four method supplement's output-schema description mentions 16/11; its actual scope, loader, arrays and recorded accounting consistently contain four inputs and two mapped outputs. That frozen sentence is retained transparently, not silently edited.
'''
readme.write_text(text)
summary=B/'final_summary_v1.json';s=json.loads(summary.read_text());s['independent_reviews']={n:sha(B/'protocol'/n) for n in reviews};s['prior_evidence_preservation']=prior;write(summary,s)
files={}
for p in sorted(B.rglob('*')):
    if p.is_file() and p.name!='evidence_manifest.json':files[str(p.relative_to(B))]={'sha256':'sha256:'+sha(p),'bytes':p.stat().st_size}
oversized={k:v['bytes'] for k,v in files.items() if v['bytes']>95_000_000}
assert not oversized,oversized
manifest={'schema_version':'dga_practical_static_evidence_manifest_v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'STATIC_PILOT_ADMITTED_TESTED_SCOPE','files':files,'historical_results_unchanged':prior,'target_and_source_hashes_unchanged':True,'no_physics_evaluation_training':True}
write(B/'evidence_manifest.json',manifest)
print(json.dumps({'manifest_sha256':sha(B/'evidence_manifest.json'),'file_count':len(files),'bytes':sum(v['bytes'] for v in files.values()),'largest_file':max(files,key=lambda k:files[k]['bytes']),'all_prior_preserved':True},indent=2))
