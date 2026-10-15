from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
TMP=Path('/private/tmp/hug_feasible_refinement_2026-10-14')
BUNDLE=ROOT/'research_artifacts/hug_feasible_refinement_2026-10-14'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
staged=TMP/'staged_working_draft.tex'
scope=read(BUNDLE/'git_staging_scope.json')
assert sha(staged)==scope['staged_paper_sha256']
for src in (staged,TMP/'final_compiler_results.json'):
    shutil.copyfile(src,BUNDLE/src.name)
paper=ROOT/'docs/paper/working_draft.tex'
receipt={'schema':'hug_feasible_final_compiler_checkpoint_receipt_v2',
    'staged_checkpoint_sha256':sha(staged),'staged_HUG_only_checkpoint_compile':'success',
    'shared_working_paper_sha256_after_compile':sha(paper),'shared_working_paper_compile':'success',
    'open_paper_path_unchanged':str(paper),'other_provider_sections_preserved_and_unstaged':True,
    'raw_tool_results':'final_compiler_results.json'}
(BUNDLE/'final_compiler_checkpoint_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copyfile(Path(__file__),BUNDLE/'orchestration'/Path(__file__).name)
manifest={'schema':'hug_feasible_static_bundle_manifest_v2','files':{str(p.relative_to(BUNDLE)):sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p!=BUNDLE/'manifest.json'}}
(BUNDLE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for rel,h in manifest['files'].items():assert sha(BUNDLE/rel)==h,rel
for path,h in read(TMP/'seed_stage/protected_hashes.json').items():assert sha(path)==h,path
assert read(BUNDLE/'validation/independent_final_read_only_review.json')['result']=='PASS'
assert sha(ROOT/'src/grasp_failure_prediction/evaluation/feasible_surface_refinement.py')=='2b4d8c30fe639d70bd8e4da0aa6406c2782ee2a3e7997683ed89db9edb935c4e'
subprocess.run(['git','add','--',str(BUNDLE.relative_to(ROOT))],cwd=ROOT,check=True)
assert subprocess.check_output(['git','show',':docs/paper/working_draft.tex'],cwd=ROOT)==staged.read_bytes()
print(json.dumps({'final_manifest_sha256':sha(BUNDLE/'manifest.json'),'bound_files':len(manifest['files']),
    'all_protected_inputs_unchanged':True,'both_manuscript_versions_compiled':True}))
