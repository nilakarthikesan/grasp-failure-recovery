from pathlib import Path
import hashlib, json, datetime

W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
B=W/'research_artifacts/dga_practical_retargeting_2026-10-15_v1'
C=Path('/private/tmp/dga_practical_retargeting_20261015_v1/control')
V=Path('/private/tmp/dga_practical_validation_20261015_v1')
def sha(b): return hashlib.sha256(b).hexdigest()

reports={n:json.loads((V/n/'report.json').read_text()) for n in ['recent16_authoritative_v3_part_a','recent16_authoritative_v3_part_b','original4_authoritative_v3']}
counts={}
for d in reports.values():
    for k,v in d['counts'].items(): counts[k]=counts.get(k,0)+v
assert counts['source_valid_mapped']==13 and counts['source_invalid']==7
assert counts['certified_corrected_branches']==1 and counts['explicit_baseline_fallback_branches']==22
row=json.loads((V/'recent16_authoritative_v3_part_a/row_04.json').read_text())
assert row['branches']['thumb']['adopt_candidate']
assert not row['branches']['little']['adopt_candidate']

decision='''
<!-- dga-practical-static-2026-10-15-v1 -->
## 15 October 2026 — DGA practical retargeting admitted for the tested static scope

**Previous plan and authorized change.** Strict native-preservation trials and two exact palm-obstruction results remain failed and unchanged. The user authorized a distinct practical DGA-plus-retargeter protocol on the same Shadow geometry. Criteria were frozen before testing (SHA-256 `6eb351dc097daffb4d08f04117615714de2120d3395367f900e4cefb444d4d34`). The new estimand is the eventual executed Shadow pipeline, not equivalent native-hand outcomes. No outcome data are collected here.

**Implemented criteria.** Source validity precedes explicit named-joint projection into literal target intervals. Palm, wrists and already target-legal finger branches remain fixed. Corrections must preserve per-finger maximum landmark error and SSE relative to that legal reference and independently bound the worst uncapped native-relative distance residual over the whole 50 mm cube surface. Admission compares candidate upper bounds to reference lower bounds; unresolved corrections fall back. The conditional 1 nm per-distance arithmetic allowance is numerical, not a physical fidelity tolerance. No empirically justified absolute distortion threshold is available, so none was invented to admit these outputs. The bounded worst-residual property does not assert pointwise/contact-region preservation.

**Completed evidence and decision.** All 20 saved MultiDex outputs are accounted for: 13 source-valid poses pass the declared static mapping gate and seven native-invalid outputs remain explicit refusals. All are previously inspected regression data. Of 23 proposed branch corrections, one (recent row 4 thumb) passes the independent whole-surface test; 22 are unresolved and use the reference. The thumb's worst surface-residual interval decreases from [2.508736, 2.947318] mm to [1.678521, 2.117103] mm. Its maximum landmark error decreases from 0.852864 to 0.584096 mm, with SSE reduced as well. The final whole-pose maximum is 0.605692 mm because the little-finger proposal was not admitted. Across the cohort, maximum landmark error still reaches 6.075600 mm. Existing native/target palm overlap remains, and no native-fidelity claim is accepted.

**Status.** `STATIC_PILOT_ADMITTED_TESTED_SCOPE` applies only to this auditable DGA-plus-retargeter implementation on the saved cube cohort under the conditional numerical bounds. Literal limits, independent XML/FK and compiled MuJoCo kinematics pass; final poses and every refusal/fallback are retained. Provider execution stays disabled. Original DexGraspNet, native MultiDex, historical corrections and intermediate practical methods remain separate. The mapper's capped-objective mismatch, exception fallback, face-referenced-vertex bound, omitted thumb visual box and validation runtime were corrected in additive versions before final admission. Independent complete-surface coverage/adoption recounts are archived.

**Next check and limits.** A frozen pipeline needs fresh native outputs on additional objects before claiming input coverage beyond this regression cube. Physical adequacy and failure-predictor utility are separate later questions, requiring authorized execution and outcome-based evidence; static nonregression does not answer them. Distortion descriptors, mapper identity, native-invalid refusals and the full attempted-input denominator must be retained in any later dataset. No physics, new native inference, evaluation episodes/labels or training ran in this study. See [protocol](DGA_PRACTICAL_RETARGETING_PROTOCOL.md) and [versioned evidence](../research_artifacts/dga_practical_retargeting_2026-10-15_v1/README.md).
'''

progress='''
<!-- dga-practical-static-2026-10-15-v1 -->
## 15 October 2026 — Practical DGA static contract completed on saved outputs

The existing DGA manuscript section now distinguishes historical failed native preservation from the newly authorized practical Shadow pipeline. All 20 saved MultiDex outputs are accounted for: 13 native-valid static mappings, seven explicit native-invalid refusals, one independently admitted thumb correction and 22 unresolved branch proposals falling back to the fixed legal reference. The positive bound concerns each finger's worst uncapped distance residual over the complete cube surface, conditional on the declared arithmetic allowance; it does not prove pointwise contact preservation. Maximum landmark error remains 6.075600 mm, known palm overlap remains, and physical/native-fidelity acceptance is not asserted.

Status is `STATIC_PILOT_ADMITTED_TESTED_SCOPE` for these previously inspected cube outputs, with execution disabled. No new native inference, physics, evaluation episodes/labels or training. Historical protocols/results and other manuscript sections remain preserved. [Protocol and evidence](DGA_PRACTICAL_RETARGETING_PROTOCOL.md). The canonical paper is compiled after this edit; the source snapshot and compiler receipt bind the exact working file.
'''

paper=r'''
\paragraph{A distinct practical static protocol.}
Following explicit authorization, we freeze a new contract for DGA plus an
explicit retargeter on the unchanged Shadow hand; the preceding exact
native-preservation failures remain unchanged. Source-invalid outputs are
refused before retargeting. The legal reference projects finger angles onto
literal target intervals, retaining the native palm, both wrists and already
legal finger branches. This minimizes joint-space change only. Corrections
must not increase any finger's maximum landmark error or squared-error sum.
An independent implementation also bounds the worst uncapped unsigned
native-relative distance residual over the entire 50 mm cube surface.
The residual is 2-Lipschitz; shifted 1, 0.5 and 0.25 mm grids and a complete
adaptive cell-cover ledger bound its supremum, conditional on 1 nm arithmetic
error per distance. Candidate upper bounds must not exceed reference lower
bounds, and independent strict improvement is required. Unresolved proposals
return the recorded reference. This property concerns a worst-case residual,
not pointwise nonregression, signed occupancy or preserved contact regions.

All twenty saved MultiDex outputs are accounted for: thirteen source-valid
poses satisfy the static contract and seven retain native-domain refusals.
All have previously informed development; no untouched holdout is claimed.
Of 23 proposed branch corrections, one passes the independent gate and 22
are unresolved. For recent row 4, the thumb's continuous maximum residual
interval falls from $[2.508736,2.947318]$ to $[1.678521,2.117103]$ mm;
its maximum landmark error falls from 0.852864 to 0.584096 mm with lower SSE.
The little-finger correction is not admitted, leaving final whole-pose maximum
error 0.605692 mm. Across the cohort, error still reaches 6.075600 mm.
Independent compiled kinematics and literal target limits pass. Fixed palm
geometry prevents additional palm change relative to the reference but does
not remove native/target palm overlap. No absolute fidelity tolerance has been
empirically calibrated. Admission is therefore limited to the tested static
implementation; native equivalence, physical adequacy and unseen-object
generalization remain unestablished. Any future labels would describe the
executed Shadow generator-plus-mapper pipeline, with distortion and refusal
accounting retained. Provider execution remains disabled. No physics, new
native inference, evaluation episodes/labels or training is performed.

'''

updates={'docs/RESEARCH_DECISIONS.md':decision,'docs/PAPER_PROGRESS.md':progress}
receipts=[]
for rel,addition in updates.items():
    p=W/rel; old=p.read_text(); assert 'dga-practical-static-2026-10-15-v1' not in old
    new=old+'\n'+addition
    assert p.read_text()==old
    p.write_text(new)
    receipts.append({'path':rel,'before_sha256':sha(old.encode()),'after_sha256':sha(new.encode()),'operation':'append_only'})
p=W/'docs/paper/working_draft.tex';old=p.read_text()
start='% BEGIN GFP SECTION dexgrasp_anything';end='% END GFP SECTION dexgrasp_anything'
assert old.count(start)==old.count(end)==1
i=old.index(start); j=old.index(end); outside=old[:i]+old[j:]
assert r'\paragraph{A distinct practical static protocol.}' not in old
new=old[:j]+paper+old[j:]
assert p.read_text()==old
p.write_text(new)
assert new[:new.index(start)]+new[new.index(end):]==outside
receipts.append({'path':'docs/paper/working_draft.tex','before_sha256':sha(old.encode()),'after_sha256':sha(new.encode()),'outside_dga_before_sha256':sha(outside.encode()),'outside_dga_after_sha256':sha(outside.encode()),'operation':'append_inside_owned_dga_section'})
proto=W/'docs/DGA_PRACTICAL_RETARGETING_PROTOCOL.md'
text=proto.read_text()
text+='''

## Completed static pilot, 15 October 2026

All 20 saved MultiDex outputs were retained: 13 source-valid mappings pass the static contract and seven native-invalid outputs remain refusals. One of 23 proposed branch corrections is admitted (recent row 4 thumb); 22 remain unresolved and fall back. The independently bounded worst thumb residual falls from [2.508736, 2.947318] mm to [1.678521, 2.117103] mm, with lower maximum landmark error and SSE. This is a worst-residual improvement over the whole cube surface, not pointwise nonregression at every location or preservation of all near/contact regions.

The final maximum landmark error across the cohort remains 6.075600 mm and existing palm overlap remains. The scoped status is `STATIC_PILOT_ADMITTED_TESTED_SCOPE`, with no native-fidelity or physically harmless-distortion claim. All outputs are previously inspected cube regression data. Provider execution remains disabled.

The final validator uses face-referenced vertex bounds, conservative triangle sphere/AABB pruning and an exact triangulation of the target thumb's 1 mm visual box. Adaptive refinement retains every leaf's bound in a verified full-surface covering ledger. It changes runtime, not the frozen grid levels or acceptance inequalities. Earlier capped proposer runs, incomplete checker runs and all code versions remain separate in the evidence bundle.
'''
proto.write_text(text)
(C/'owned_doc_updates.json').write_text(json.dumps({'append_updates':updates,'paper_append':paper,'receipts':receipts},indent=2)+'\n')
print(json.dumps(receipts,indent=2))
