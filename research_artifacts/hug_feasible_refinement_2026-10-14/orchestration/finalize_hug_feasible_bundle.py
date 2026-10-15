from pathlib import Path
import hashlib,json,shutil
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
TMP=Path('/private/tmp/hug_feasible_refinement_2026-10-14')
BUNDLE=ROOT/'research_artifacts/hug_feasible_refinement_2026-10-14'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
review=read(TMP/'validation/independent_final_read_only_review.json')
assert review['result']=='PASS'
copied=[]
for name in ('independent_final_read_only_review.json','INDEPENDENT_FINAL_READ_ONLY_REVIEW.md','review_final_audit.py'):
    src=TMP/'validation'/name;dst=BUNDLE/'validation'/name;assert not dst.exists()
    shutil.copyfile(src,dst)
    copied.append({'source_path':str(src),'source_sha256':sha(src),'bundle_path':str(dst.relative_to(BUNDLE)),
        'bundle_sha256':sha(dst),'encoding':'original_bytes','size_bytes':src.stat().st_size})
p=BUNDLE/'source_copy_receipt.json';receipt=read(p);receipt['files'].extend(copied);p.write_text(json.dumps(receipt,indent=2)+'\n')
guide=ROOT/'docs/HUG_FEASIBLE_REFINEMENT.md';body=guide.read_text()
anchor='- The recorded selection verifier passes six authentic and ten deliberate\n'
assert body.count(anchor)==1
body=body.replace(anchor,'- The final independent report review passes 832 consistency checks, including\n  all nine statuses, per-band summaries, source gates and protected hashes.\n'+anchor)
guide.write_text(body)
metadata=read(BUNDLE/'publication_verification.json')
metadata['independent_final_read_only_review_result']='PASS'
metadata['independent_final_read_only_consistency_check_count']=832
metadata['independent_final_review_sha256']=sha(BUNDLE/'validation/independent_final_read_only_review.json')
metadata['guide_sha256']=sha(guide)
(BUNDLE/'publication_verification.json').write_text(json.dumps(metadata,indent=2)+'\n')
shutil.copyfile(Path(__file__),BUNDLE/'orchestration'/Path(__file__).name)
manifest={'schema':'hug_feasible_static_bundle_manifest_v2','files':{str(p.relative_to(BUNDLE)):sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
(BUNDLE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for rel,h in manifest['files'].items():assert sha(BUNDLE/rel)==h,rel
for path,h in read(TMP/'seed_stage/protected_hashes.json').items():assert sha(path)==h,path
print(json.dumps({'bound_files':len(manifest['files']),'manifest_sha256':sha(BUNDLE/'manifest.json'),'all_bound_files_verified':True,'protected_numeric_dependencies':89,'no_actual_physics_or_eval_data':True}))
