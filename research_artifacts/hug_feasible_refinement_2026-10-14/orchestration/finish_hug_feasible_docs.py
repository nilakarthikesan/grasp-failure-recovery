from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
TMP=Path('/private/tmp/hug_feasible_refinement_2026-10-14')
BUNDLE=ROOT/'research_artifacts/hug_feasible_refinement_2026-10-14'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
decisions='''

## 14 October 2026 — HUG feasible-step repair; full mapping remains failed

**Previous plan:** Six surface fits were refused because every eligible major/raw
candidate violated an original-relative fingertip cap. The user authorizes
continued HUG diagnosis/repair, concurrent provider work and ongoing paper updates,
with no new eval data.

**Measured diagnosis:** All 649 recorded major fingertip vectors replay exactly;
all six original cap Jacobians have rank five and admit inward feasible steps.
Cheap FK controls restore all six old raw poses. The first surface optimizer
takes large nonlinear steps and lacks feasible-step restoration; its refused
trajectory does not establish contradictory constraints or global infeasibility.
A bounded 60-call stencil observes nearby direction-eligibility jumps but does
not identify them as the sole cause of all six failures. No demonstrated frame,
cache or constraint-sign bug was found in these recorded checks.

**Implemented and frozen:** An isolated feasible-refinement API restores/checks
cheap FK before new full-surface scores, retains original caps/source bounds and
native poor relationships, and independently recomputes terminal results. The
actual study is seed-only on one immutable old raw pose per original case, no
pattern sweeps. Its 1-nm inward target tightens projection; computed final caps
have zero added allowance, and the five-tip mean remains strictly ≤10 mm. The
scientific 1/2/5-mm surface rule is unchanged. Twenty-six new manufactured and
120 prior surface checks pass (146 total).

**Completed result:** All six poses improve frozen cost by 1.52–38.09% while
passing all 30 original-relative fingertip caps and independent source/scene
checks. This restoration stage uses 18 full calls in 49.57 s, reusing previously
computed raw seeds; it is not a from-scratch speed claim. Two cheap projections
converge, four exhaust their 40-iteration budget while returning verified
feasible improving points. All six source preflights PASS. The unchanged six
coarse/three fine surface audits give 0 PASS/9 FAIL; pointwise orientation is
9 FAIL, versus 7 FAIL/2 UNRESOLVED on original fallback poses. Larger computed
added-area lower bounds and new reliable reversal witnesses show that the
fitting objective cannot certify contact preservation. Counts have pose-specific
eligibility and are not physical failure rates.

**Decision and scope:** Retain the feasibility repair as experimental. Do not
promote it to the default mapper, loosen acceptance, or call complete HUG mapping
validated. Study the remaining anatomical contact/surface witnesses separately
with this feasibility mechanism. The general pattern path has manufactured
evidence only. All 89 bound numerical dependencies, saved inputs and predictor
remain unchanged; mutable paper/unrelated provider code is outside that closure.
The same manuscript's HUG subsection is updated and compiles successfully.
No inference, integrated physics, eval episodes, outcome labels or training ran.
Original scientific bundles/protocols remain immutable.
[Completed phase](HUG_FEASIBLE_REFINEMENT.md).
'''
progress='''

## 14 October 2026 — HUG feasible restoration completed; fidelity still fails

The old constraint failures are traced to the unsuccessful surface-search
trajectory, with feasible movement demonstrated on all six saved cases. A new
frozen seed-restoration stage returns six improving poses and passes all 30
original-relative fingertip caps plus independent source checks. It uses 18
surface calls in 49.57 s, reusing the old raw fits. Two cheap projections converge
and four reach their iteration budget; final feasibility is independently checked.
All 146 new/prior surface API tests pass.

The unchanged nine-resolution audit still returns 0 PASS/9 FAIL, now with nine
pointwise orientation FAIL results. This repairs constraint handling, not complete
grasp preservation. The default mapper, predictor and historical results remain
unchanged. No new eval data or training was generated. The existing paper's HUG
subsection is updated and compiles. [Evidence and remaining work](HUG_FEASIBLE_REFINEMENT.md).
'''
for file,extra in [('docs/RESEARCH_DECISIONS.md',decisions),('docs/PAPER_PROGRESS.md',progress)]:
    path=ROOT/file;old=path.read_text();assert extra.splitlines()[2] not in old
    path.write_text(old+extra)
handoff=ROOT/'docs/THREAD_HANDOFF.md';old=handoff.read_text()
notice='''> **Latest continuation, 14 October 2026:** Read the
> [parallel-provider coordination guide](PARALLEL_GENERATOR_VALIDATION_HANDOFF.md)
> and [completed HUG feasibility repair](HUG_FEASIBLE_REFINEMENT.md) before the
> historical snapshot below. Six improved HUG fits now pass all fingertip caps,
> but all nine strict surface comparisons still FAIL. The current title is
> *Evaluating Simulation Diversity for Grasp Failure Prediction*. No new eval
> data or training is authorized in this mapping-validation phase.

'''
assert 'Latest continuation, 14 October 2026' not in old
first,rest=old.split('\n',1);handoff.write_text(first+'\n\n'+notice+rest.lstrip('\n'))
paper=ROOT/'docs/paper/working_draft.tex'
assert sha(paper)=='f347ddbb9537fd9d147e03ecb935a8678ec7e9ebea0b6ec66c8612e4a295b278'
receipt={'schema':'hug_feasible_paper_compile_receipt_v2','source_path':str(paper),
    'source_sha256':sha(paper),'HUG_section_update_receipt':json.loads((TMP/'paper_final_section_update.json').read_text()),
    'compiler':'Codex desktop standalone LaTeX compiler','observed_result':'success',
    'observed_message':'The current source compiled successfully with the desktop editor\'s compiler.',
    'recorded_utc':datetime.now(timezone.utc).isoformat(),'concurrent_provider_sections_preserved':True}
    # Compiler success observed after the last HUG edit; exact current bytes
    # match that update receipt. Numeric closure intentionally excludes prose.
(BUNDLE/'paper_compile_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
for name in ('hug_final_fragment.tex','paper_final_section_update.json'):
    shutil.copyfile(TMP/name,BUNDLE/name)
for src in [Path(__file__),Path('/private/tmp/publish_hug_feasible_results.py'),Path('/private/tmp/update_hug_feasible_paper_final.py')]:
    shutil.copyfile(src,BUNDLE/'orchestration'/src.name)
manifest={'schema':'hug_feasible_static_bundle_manifest_v2','files':{str(p.relative_to(BUNDLE)):sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
(BUNDLE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'updated_ledgers':True,'paper_sha256':sha(paper),'bound_files':len(manifest['files']),'manifest_sha256':sha(BUNDLE/'manifest.json')}))
