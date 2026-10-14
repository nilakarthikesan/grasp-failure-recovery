"""Record the newly authorized parallel workflow and writing phase separately."""
from pathlib import Path
import hashlib
import json

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
HERE=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    master=ROOT/'docs/PARALLEL_GENERATOR_VALIDATION_HANDOFF.md'
    t=master.read_text()
    t=t.replace('and do not later save a stale buffer over the coordinated manuscript.',
        'and do not later save a stale buffer over the coordinated manuscript.\nRemote compute workers deliver evidence and fragments for local integration;\nthis local advisory lock does not coordinate separate-host manuscript replicas.')
    master.write_text(t)
    p=ROOT/'docs/PAPER_STRUCTURE_PLAN.md'
    old=p.read_text()
    new='''# Paper structure and writing plan

## Current direction, 14 October 2026 — parallel audits and prediction-centered paper

The user authorizes parallel individual validation of ContactGen, DexGrasp
Anything and UniDexGrasp, with HUG/shared dependencies coordinated in the main
chat. The accepted working title is **Evaluating Simulation Diversity for
Grasp Failure Prediction**. The central question is whether training on proposals
from multiple generators improves generalization to unfamiliar objects and
grasp configurations relative to single-generator training. Mapping validity
supports this prediction study; it is not the sole research contribution.

The [working manuscript](paper/working_draft.tex) now contains an updated
abstract/introduction and source-specific mapping/verification subsections,
with completed HUG evidence and explicit missing actual-native evidence for the
other three sources. Historical HUG predictor/executor results are preserved as
development motivation. The one/two/three/four-source comparison remains
proposed, with matched training counts as a methods control and generator
identity separated from count. No training or new evaluation data occurs here.

Current structure: introduction/related work; common mapping validity method;
HUG/ContactGen/DexGrasp Anything/UniDexGrasp subsections; historical HUG execution
and development diagnostics; planned multiple-source training/generalization
comparison; limitations/evidence; references. Provider threads maintain only
their owned marked subsections using the shared lock/hash editor. The coordinator
owns abstract/introduction, common claims, bibliography and global ledgers.
[Assignments and coordination](PARALLEL_GENERATOR_VALIDATION_HANDOFF.md).

This is a growing research draft, not a completed comparison or submission-ready
PDF. The earlier diagnostic-only workshop plan below remains a dated alternative.
The broader direction supersedes its HUG-only research-question and exclusion of
four-generator work; it does not change the workshop's page/template rules or
claim future findings as completed. Final submission scope/format must be chosen
from evidence actually completed at submission time.

## Historical diagnostic-only submission plan (preserved)

'''+old.replace('# Paper structure and writing plan for Sim2Real2Field\n\n','',1)
    p.write_text(new)
    entry='''

## 14 October 2026 — Parallel provider validation and shared-paper writing authorized

**Previous plan and new user decision:** Individual audits had been sequenced
behind HUG. The user requests three handoffs for concurrent ContactGen,
DexGrasp Anything and UniDexGrasp validation, plus simultaneous updates to the
same manuscript, including abstract/introduction. This supersedes the earlier
sequential provider order. No new eval data is requested or generated.

**Evidence and coordination choice:** The completed HUG batch already used six
concurrent fits and two audit groups; expensive mesh queries share the host's
CPU. Separate chats advance provider-specific code/contracts and writing in
parallel, but do not add compute hardware. Shared MANO mapping affects HUG and
ContactGen; shared Shadow bridges, assets and measurement code can affect other
audits. Give each provider unique code/tests/evidence/report ownership, reserve
shared dependencies and Git operations for the coordinator, and bind each run's
actual scientific dependency closure. Evolving prose is recorded separately.

**Implemented:** Three ready-to-paste provider handoffs and a common coordination
guide are saved. The same existing paper now has marked source-specific blocks,
an updated multi-generator prediction abstract/introduction, supporting mapping
methods and a proposed one/two/three/four-source comparison. Its accepted title
is Evaluating Simulation Diversity for Grasp Failure Prediction. A lock/reread/
section-hash editor merges owned subsections without losing other writers'
changes; all 12 installed tests pass, including concurrent subprocess updates.
The standalone manuscript compiles successfully with the desktop compiler.
Historical HUG numerical result text is preserved exactly.

**Measured status and remaining work:** HUG accepted no correction and still
FAILS the unchanged strict claim at six coarse/three fine resolutions. These
are nine repeated static comparisons, not nine physical failures. ContactGen
and UniDexGrasp have synthetic/reference implementation evidence; DexGrasp
Anything needs its dedicated decoder/worker/registration. No actual-native
validation is claimed for those three. Missing authentic outputs/assets are
blockers to full native verification; synthetic fixtures cannot replace them.
Source diversity improving predictor generalization remains a hypothesis.

**Provenance:** Today's explicitly authorized manuscript revision follows the
completed HUG run and does not rewrite that run's old paper/source snapshot.
All prior scientific bundles and results remain immutable; shared model and
simulation inputs/code remain unchanged. No fresh inference, integration,
eval episodes, task labels or predictor fitting occurs in this writing phase.
[Parallel handoffs](PARALLEL_GENERATOR_VALIDATION_HANDOFF.md).
'''
    for name in ('RESEARCH_DECISIONS.md','PAPER_PROGRESS.md'):
        q=ROOT/'docs'/name;q.write_text(q.read_text()+entry)
    q=ROOT/'docs/THREAD_HANDOFF.md'
    q.write_text(q.read_text()+'''

## 14 October 2026 — Latest: parallel provider audits and one coordinated paper

The latest user instruction supersedes the earlier sequential provider order:
prepare three ContactGen/DexGrasp Anything/UniDexGrasp handoffs so the user can
start their chats concurrently; keep all contributions in the existing working
paper. [Start here](PARALLEL_GENERATOR_VALIDATION_HANDOFF.md). Each provider file
contains a copy-paste prompt, exact file ownership, status/blockers and paper
update procedure. User-owned chats have not been created automatically.

The manuscript remains docs/paper/working_draft.tex in the codex/project-work
checkout. Abstract/introduction and mapping subsections now reflect the accepted
multi-generator failure-prediction direction. Historical HUG numbers remain
exact; mixed-source training/generalization results remain unexecuted. Compile
succeeded. All paper writers use scripts/update_paper_section.py, which locks,
rereads and atomically replaces one owned block with a verified section hash.
Twelve installed concurrency/refusal tests pass. Coordinator owns shared
sections, ledgers, mapping/simulator dependencies and Git staging/commits/pushes.

HUG's last run remains 0 accepted corrections,6 exact originals,9 strict FAIL;
source checks pass, contact/failure-label equivalence is unconfirmed. The other
three sources currently lack validated real native outputs/actual source-hand
evidence; their threads first inventory saved inputs, audit/repair unique
provider code and document honest scoped results. No new eval data, physics,
predictor training or fresh generator inference ran in this handoff/writing task.
The old HUG source snapshot's paper hash describes that completed phase; the
new authorized manuscript revision is separately recorded in
research_artifacts/parallel_validation_coordination_2026-10-14/.
Preserve previous run records, assets, existing untracked files and this paper
path. Do not rerun unchanged HUG audits or loosen its rule to obtain a PASS.
''')
    q=ROOT/'docs/README.md';text=q.read_text()
    text+='''

## Parallel generator validation and shared manuscript

[Coordination and three thread prompts](PARALLEL_GENERATOR_VALIDATION_HANDOFF.md)
assign independent ContactGen, DexGrasp Anything and UniDexGrasp audits, shared
dependency ownership and locked subsection updates to the existing paper.
The prediction-centered opening and provider methods are drafted; actual-native
validation and mixed-source training remain pending where stated.
''';q.write_text(text)
    evidence=ROOT/'research_artifacts/parallel_validation_coordination_2026-10-14'
    verification=json.loads((evidence/'initialization_verification.json').read_text())
    verification['paper_compile_status']='PASS'
    verification['paper_compile_tool']='desktop standalone LaTeX compiler'
    verification['installed_concurrency_test_result']='12 passed;19 subtests passed'
    verification['installed_test_sha256']=sha(ROOT/'tests/test_paper_section_updates.py')
    verification['paper_after_sha256']=sha(ROOT/'docs/paper/working_draft.tex')
    original=json.loads((ROOT/'research_artifacts/hug_constrained_refinement_2026-10-14/fit/protected_hashes.json').read_text())
    paper=str(ROOT/'docs/paper/working_draft.tex')
    other={p:h for p,h in original.items() if p!=paper}
    assert len(other)==113 and all(sha(p)==h for p,h in other.items())
    verification['previous_HUG_nonpaper_protected_files_unchanged']=113
    (evidence/'initialization_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (evidence/'finish_coordination_docs.py').write_bytes(Path(__file__).read_bytes())
    names=[ROOT/'scripts/update_paper_section.py',ROOT/'tests/test_paper_section_updates.py',
           ROOT/'docs/paper/working_draft.tex',master,ROOT/'docs/PAPER_STRUCTURE_PLAN.md']
    names.extend((ROOT/'docs/parallel_validation').glob('*HANDOFF.md'))
    manifest={'schema':'parallel_provider_handoff_and_paper_manifest_v1','date':'2026-10-14',
        'files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in names},
        'evidence_files_sha256':{p.name:sha(p) for p in evidence.iterdir() if p.is_file() and p.name!='manifest.json'},
        'no_new_eval_data':True,'no_new_native_inference':True,'physics_steps':0}
    (evidence/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
