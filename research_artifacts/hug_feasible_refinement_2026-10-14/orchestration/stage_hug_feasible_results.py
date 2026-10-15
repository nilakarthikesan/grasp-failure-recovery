"""Stage only coordinator-owned HUG changes; preserve other chats' edits."""
from pathlib import Path
import hashlib,json,subprocess,shutil
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
BUNDLE=ROOT/'research_artifacts/hug_feasible_refinement_2026-10-14'
def git(*args,input=None):return subprocess.check_output(['git',*args],cwd=ROOT,input=input)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert git('branch','--show-current').decode().strip()=='codex/project-work'
assert git('rev-parse','HEAD').decode().strip()=='76a4b2cf464926d5a531372b1bc7fd239d955445'
assert not git('diff','--cached','--name-only').strip(),'Another writer staged changes; do not include them.'
paper='docs/paper/working_draft.tex'
base=git('show','HEAD:'+paper).decode();current=(ROOT/paper).read_text()
def span(text):
    begin='% BEGIN GFP SECTION hug\n';end='% END GFP SECTION hug'
    assert text.count(begin)==text.count(end)==1
    a=text.index(begin)+len(begin);b=text.index(end,a)
    return a,b
a,b=span(base);ca,cb=span(current)
assert hashlib.sha256(current[ca:cb].encode()).hexdigest()=='dde285566bd5a9373b03baa02019d1bf3a758b899604e6242945c38e6cf0c953', 'Owned HUG block changed; reconcile before staging.'
staged=base[:a]+current[ca:cb]+base[b:]
assert staged!=base and staged!=current
scope={'schema':'hug_feasible_owned_git_staging_scope_v2','base_commit':'76a4b2cf464926d5a531372b1bc7fd239d955445',
    'paper_path':paper,'working_paper_sha256':hashlib.sha256(current.encode()).hexdigest(),
    'staged_paper_sha256':hashlib.sha256(staged.encode()).hexdigest(),
    'paper_staging_policy':'Only HUG marked block over HEAD; all provider working-paper blocks remain untouched and unstaged.',
    'source_code_staging_policy':'Only new experimental feasible API and its manufactured tests.',
    'unrelated_provider_code_tests_reports_evidence_preserved':True}
(BUNDLE/'git_staging_scope.json').write_text(json.dumps(scope,indent=2)+'\n')
shutil.copyfile(Path(__file__),BUNDLE/'orchestration'/Path(__file__).name)
manifest={'schema':'hug_feasible_static_bundle_manifest_v2','files':{str(p.relative_to(BUNDLE)):sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
(BUNDLE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
owned=['docs/HUG_FEASIBLE_REFINEMENT.md','docs/RESEARCH_DECISIONS.md','docs/PAPER_PROGRESS.md','docs/THREAD_HANDOFF.md',
       'src/grasp_failure_prediction/evaluation/feasible_surface_refinement.py','tests/test_feasible_surface_refinement.py']
owned += [str(p.relative_to(ROOT)) for p in sorted(BUNDLE.rglob('*')) if p.is_file()]
git('add','--',*owned)
oid=git('hash-object','-w','--stdin',input=staged.encode()).decode().strip()
git('update-index','--add','--cacheinfo','100644',oid,paper)
names=set(git('diff','--cached','--name-only').decode().splitlines())
assert names==set(owned+[paper]),names.symmetric_difference(owned+[paper])
assert (ROOT/paper).read_text()==current,'Working paper was altered while staging.'
assert git('show',':'+paper).decode()==staged
print(json.dumps({'staged_owned_file_count':len(names),'paper_only_HUG_block_staged':True,
    'working_paper_unchanged':True,'staged_paper_sha256':scope['staged_paper_sha256'],'manifest_sha256':sha(BUNDLE/'manifest.json')}))
