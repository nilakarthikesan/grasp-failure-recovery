"""Append only this study's decisions and update the owned DGA paper section."""
from pathlib import Path
import hashlib
import json

WORK = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
STUDY = WORK/'research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_acceptance_feasibility_v2'
MARKER = '<!-- dga-acceptance-feasibility-2026-10-15-v2 -->'


def sha(data): return hashlib.sha256(data).hexdigest()


def append(path, text):
    assert MARKER not in path.read_text(), 'Do not duplicate a decision'
    with path.open('a') as f: f.write('\n\n'+MARKER+'\n'+text+'\n')


def main():
    # Final visual status must have been independently resolved before writing.
    final = json.loads((STUDY/'control/final_summary_v1.json').read_text())
    assert final['new_candidate_visual_preservation'] == 'FAIL'
    report = '''# DGA acceptance feasibility audit

15 October 2026. **DGA remains unaccepted under the unchanged target and criteria. Two required poses have a verified mathematical incompatibility; this is more specific than an unsuccessful optimizer.**

The shared MuJoCo importer fixes remain intact. Independent reconstruction finds no additional rotation, joint-order, unit or source-mesh conversion error. Native and target fingertip meshes genuinely differ: the index/ring source distal mesh has 970 triangles and the target visual mesh has 5,374. Their local extents differ by at least 2.171777 mm in one direction; that is a geometric witness, not a whole-surface Hausdorff estimate.

All 16 saved native outputs remain accounted for. Eleven source-valid outputs received new static surface diagnostics; the five source-invalid outputs remain rejected. The only pose also legal on the target, raw row 6, fails index/ring visual proximity preservation at every 0.5/1/2 mm band without angle correction. At the 1 mm band, conservative lost-area lower bounds are 50.25 and 66.9375 mm² respectively. Its palm comparison passes only for empty nearsets and no occupancy; direction remains unresolved. The other ten raw poses remain target-illegal diagnostics.

Exact rational interval arithmetic now proves unavoidable added palm occupancy for old development row 1 and new native row 0 under their **original pose-specific landmark guards**. The minimum certified changed squares are 0.000104947 and 0.525625 mm² respectively. The proof covers every proper target palm placement and finger configuration satisfying those guards. It removes the previous assumed 1 nm source-transform bridge. Its scope is exact declared geometry and real-arithmetic guards with the recorded constants; arbitrary floating guard implementations, physical outcomes and other poses are not covered.

A separate 2,238-state, three-start correction test retained all tightened landmark and 18 component guards. It selected a 0.00001 rad THJ3 change with unchanged palm and maximum landmark error; the thumb maximum fitting error improved only about 46.4 nm. Fresh independent grids still FAIL strict visual preservation. No acceptance is inferred from fitting loss or the two eligible intermediate proposals.

2,082 exact-math controls, 51 raw-surface controls, nine band-proposal controls and independent certificate, mesh and saved-output reviews support the result. A discovered summary unit-label error was corrected in v2; original reports and measurements are preserved. NumPy matmul warnings remain disclosed, with finite outputs and independent arithmetic checks; no blanket numerical soundness claim is made.

**Decision:** no provider promotion or evaluation start. A mapping covering these required poses cannot satisfy the current exact declared-palm criterion and original landmark guards together. Repeating fitting on those obstructed domains cannot establish acceptance. This does not prove all DGA poses impossible or invalidate a separately scoped executed-Shadow prediction task. No target, criterion or supported input domain was changed.

No physics, new native inference, evaluation episodes/outcome labels, or training ran. All 949 files in the previous DGA component-study manifest rehash unchanged. Original DexGraspNet, native MultiDex and each correction study remain separate.

[Complete evidence and manifests](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_acceptance_feasibility_v2/README.md).
'''
    target = WORK/'docs/DGA_ACCEPTANCE_FEASIBILITY_AUDIT.md'
    assert not target.exists(); target.write_text(report)
    decision = '''## 15 October 2026 — DGA exact feasibility obstruction and unchanged-angle failures

**Previous plan.** Continue repairing and independently testing DGA on the corrected importer while preserving target, criteria and originals; examine the previously untested target-legal raw pose and resolve the palm proof's numerical premise.

**Implemented evidence.** All eleven source-valid saved outputs received static raw-pose surface diagnostics, with all sixteen ledger entries retained. Raw row 6 is source/target legal but fails index/ring proximity at all three bands. Independent mesh reconstruction finds exact agreement with saved source triangles and identifies genuine distal shape differences, not a missing/doubled transform. Its palm nearset/occupancy pass is an empty-set result; directions remain unresolved. Ten other tested raw poses remain target-illegal.

Exact outward rational intervals reconstruct saved robust-6D source poses and literal wrist FK, removing the old assumed 1 nm source-transform bridge. Old development row 1 and new native row 0 each have an unavoidable positive palm-occupancy mismatch under their ORIGINAL max/SSE guards: at least 0.000104947 and 0.525625 mm². These quantifiers cover proper target palm placement and arbitrary finger articulation in the stated mathematical guard domains. Independent review recounts 516 certificates, nine literal boxes and guard provenance; 2,082 controls pass. The tighter component-guard result remains separate. Arbitrary floating guard predicate soundness and physical outcomes are not certified.

A separately frozen simultaneous joint/palm band-fitting experiment explores 2,238 states in 600.5 seconds. Two trial states pass strict component improvement and all existing guards; selection changes only THJ3 by 0.00001 rad. Thumb maximum fitting difference improves about 46.4 nm, palm and maximum landmark error are unchanged. Fresh full visual grids at (13/37,19/41) still FAIL. Budget exhaustion is not an infeasibility proof for this other pose. Independent adoption review and saved-output recount pass. A reporting-unit label defect was fixed in a separate summary version without changing measurements. All warnings and unsuccessful attempts remain preserved.

**Decision and next boundary.** DGA remains unsupported/unaccepted. The required cohort cannot all meet the unchanged exact declared-palm and landmark requirements; more optimization cannot remove the two certified conflicts. Other poses remain case-specific questions, not established impossibility. Do not silently substitute a passing subset, change geometry or infer physical grasp failure from mapping rejection. Progress to evaluation under a different target/fidelity claim/declared input domain would be a separate scientific scope decision, not a code fix; none was made. All 949 prior-study files remain unchanged. No physics, new inference, eval episodes/labels or training. [Report](DGA_ACCEPTANCE_FEASIBILITY_AUDIT.md).
'''
    progress = '''## 15 October 2026 — DGA acceptance boundary resolved for two required poses

Updated only the owned DGA paper section with exact interval palm-occupancy obstructions under original guards, new unchanged-angle row 6 failures, verified distal mesh differences and the bounded band-fitting result. The paper distinguishes impossibility for two named poses from global DGA impossibility, empty-set palm passes from direction preservation, and real-arithmetic theorem scope from arbitrary floating implementations. DGA remains unaccepted; no target/criteria/domain change, physics, new native inference, evaluation data or training. [Evidence-backed report](DGA_ACCEPTANCE_FEASIBILITY_AUDIT.md). Compilation is recorded after the desktop compiler check.
'''
    append(WORK/'docs/RESEARCH_DECISIONS.md', decision)
    append(WORK/'docs/PAPER_PROGRESS.md', progress)
    paper = WORK/'docs/paper/working_draft.tex'; before = paper.read_bytes()
    start = b'% BEGIN GFP SECTION dexgrasp_anything'; end = b'% END GFP SECTION dexgrasp_anything'
    a,b = before.index(start),before.index(end)
    assert before.count(start)==before.count(end)==1
    paragraph = r'''
\paragraph{Exact feasibility audit after the compiler repair.}
A separate static audit retains all sixteen saved native outputs and measures
the eleven source-valid raw poses. The only jointly source/target-legal raw
pose, row 6, fails index and ring visual proximity preservation at all three
bands without angle correction. At 1 mm, conservative lost-area lower bounds
are 50.25 and 66.9375 mm$^2$. Independent reconstruction reproduces all eight
index/ring runtime triangle sets exactly; distal meshes genuinely differ
(970 versus 5,374 triangles), despite matching nominal link frames. The palm
nearset/occupancy pass concerns empty sets; direction remains unresolved.

Outward rational intervals reconstruct the saved robust-6D source pose and
literal wrist FK, removing the former assumed 1 nm source-transform bridge.
For old development row 1 and new native row 0, respectively, any proper
target palm placement and finger configuration satisfying their original
pose-specific maximum/SSE guards leaves an added declared-palm occupancy
square of at least 0.000104947 and 0.525625 mm$^2$. These are exact declared
geometry statements with real-arithmetic guards and the recorded constants,
not a proof about arbitrary floating implementations or every DGA pose.
There are 2,082 manufactured arithmetic/geometry controls; independent review
recounts 516 certificates and verifies the nine collider declarations and
original guard provenance. The required cohort therefore cannot all satisfy
the unchanged exact criterion, independently of optimizer strength.

A separately frozen 2,238-state simultaneous joint/palm band-fitting trial
retains the tighter landmark and eighteen component guards. It selects only
a $10^{-5}$ rad THJ3 adjustment, reducing the maximum thumb fitting difference
by about 46.4 nm with unchanged palm and maximum landmark error. Fresh grids
at fractions $(13/37,19/41)$ still fail visual preservation. This bounded
search does not prove infeasibility for that other pose. Original outputs and
criteria remain unchanged, no provider is promoted, and no physics, new native
inference, evaluation episodes/labels or training occurs.

'''
    after = before[:b] + paragraph.encode() + before[b:]
    assert paper.read_bytes()==before, 'Concurrent paper edit: reread rather than overwrite'
    paper.write_bytes(after)
    assert after[:a]==before[:a] and after[after.index(end):]==before[b:]
    (STUDY/'control/paper_before.tex').write_bytes(before)
    (STUDY/'control/paper_after.tex').write_bytes(after)
    (STUDY/'control/documentation_update_receipt.json').write_text(json.dumps({
        'paper_before_sha256':sha(before),'paper_after_sha256':sha(after),
        'only_dga_section_changed':True,
        'dga_section_sha256':sha(after[a:after.index(end)+len(end)]),
        'report':str(target),'decision_marker':MARKER},indent=2)+'\n')
    # These exact append payloads allow selective staging without absorbing
    # earlier uncommitted research sections owned by other work.
    (STUDY/'control/decision_append.txt').write_text('\n\n'+MARKER+'\n'+decision+'\n')
    (STUDY/'control/progress_append.txt').write_text('\n\n'+MARKER+'\n'+progress+'\n')
    (STUDY/'control/paper_addition.tex').write_text(paragraph)
    print(str(target))


if __name__ == '__main__': main()
