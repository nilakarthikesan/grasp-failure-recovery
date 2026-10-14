"""Document only complete canonical static results; retain dated historical checkpoints."""
from pathlib import Path
import json

ROOT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
BUNDLE = ROOT / 'research_artifacts/hug_constrained_refinement_2026-10-14'

def main():
    s = json.loads((BUNDLE / 'validation/summary.json').read_text())
    assert s['case_count'] == 6 and s['resolution_count'] == 9
    assert s['all_engineering_source_preflights_pass'] and s['all_protected_hashes_unchanged']
    n = s['strict_status_counts']
    fallback = sum(c['selection']['source'] == 'original_fallback' for c in s['cases'])
    corrected = 6 - fallback
    verdict = f"{n['PASS']} PASS, {n['FAIL']} FAIL and {n['UNRESOLVED']} UNRESOLVED"
    result = (f"All six saved-grasp fits are complete. The solver returned {corrected} "
              f"eligible surface corrections and retained {fallback} exact original poses. "
              f"Independent original/final source preflight passes all six; the unchanged "
              f"six-coarse/three-fine strict mapping audit returns {verdict}.")
    lead = ("14 October 2026. **Fingertip constraints and native surface-direction fitting "
            "are implemented and tested on all six existing HUG grasps.** " + result + " "
            "The default mapper remains unchanged; this is not HUG mapping sign-off. "
            "The previous staged correction remains a separate historical result; see "
            "[the previous result](HUG_SURFACE_REFINEMENT.md).")
    doc = ROOT / 'docs/HUG_CONSTRAINED_REFINEMENT.md'
    text = doc.read_text()
    oldlead = text.split('\n\n', 2)[1]
    assert 'pending' in oldlead
    text = text.replace(oldlead, lead, 1)
    text = text.replace('Only completed results from the canonical run will determine its\nmeasured geometry result.', 'The completed canonical run and its independently rerun audit below determine\nthis scoped development result.')
    text = text.replace('## Selection, tests and pending acceptance', '## Selection, tests and independent acceptance')
    text = text.replace('The ongoing canonical runtime uses six independent singleton processes, two', 'The completed canonical runtime used six independent singleton processes, two')
    old = ('After all six fits finish, independent source preflight will recompute every\n'
           'original/final fingertip error, physical/trust bounds, root convention and\n'
           'recorded candidate selection. The unchanged six-coarse/three-fine audit then\n'
           'uses the original 1/2/5 mm bands, query cells, anatomical definitions, normal\n'
           'eligibility, numerical allowances and strict acceptance rule.')
    new = ('Independent source preflight recomputed every original/final fingertip\n'
           'error, physical/trust bounds and root convention, and checked the recorded\n'
           'candidate selection. It did not independently recompute all historical\n'
           'objectives. The complete nine-resolution audit used the original 1/2/5 mm\n'
           'bands, query cells, anatomical definitions, normal eligibility, numerical\n'
           'allowances and strict acceptance rule.')
    assert old in text
    text = text.replace(old, new, 1)
    rows = []
    for c in s['cases']:
        oldtip = c['original_landmarks']; tip = c['constrained_landmarks']; sel = c['selection']
        major = sel['major_iteration']
        selected = sel['source'] + (f" (recorded major callback {major})" if major is not None else '')
        callbacks = sel['recorded_selection_independent_audit'].get('major_iteration_count')
        rows.append(f"- **{c['case_id']}**: {selected}; {sel['eligible_major_count']} eligible improving major callbacks; solver status {sel['solver_status']} ({sel['solver_message']}), {sel['iterations']} optimizer iterations, budget exhausted {sel['budget_exhausted']}. Mean TIP {oldtip['mean_fingertip_error_m']*1000:.3f} → {tip['mean_fingertip_error_m']*1000:.3f} mm; maximum TIP {oldtip['maximum_fingertip_error_m']*1000:.3f} → {tip['maximum_fingertip_error_m']*1000:.3f} mm.")
    measured = ('\n## Completed measured result\n\n' + result + '\n\n' + '\n'.join(rows) + '\n\n'
        'Exact-original fallback means that case was not repaired. A raw improvement\n'
        'which violates any fingertip cap is refused; the original source proposal\n'
        'stays in the audit. Stored callback sequence numbers are distinct from\n'
        'the optimizer iteration counter. Nonconvergence and budget exhaustion do\n'
        'not prove global infeasibility of a faithful mapping.\n\n'
        '[The independent report](../research_artifacts/hug_constrained_refinement_2026-10-14/validation/REPORT.md)\n'
        'records all five fingertip and middle errors, raw refusals, whole/per-part\n'
        'proximity bounds, actual normal domains and signed witnesses.\n'
        '[The complete summary](../research_artifacts/hug_constrained_refinement_2026-10-14/validation/summary.json)\n'
        'and [bound evidence](../research_artifacts/hug_constrained_refinement_2026-10-14/results_manifest.json)\n'
        'preserve all six cases and nine resolutions.\n\n'
        '![Per-finger native position errors](../research_artifacts/hug_constrained_refinement_2026-10-14/validation/constrained_fingertip_errors.png)\n\n'
        '![Independent mapping measurements](../research_artifacts/hug_constrained_refinement_2026-10-14/validation/constrained_candidate_tradeoffs.png)\n\n')
    text = text.replace('\nLower fitting error or passing fingertip guards', measured + 'Lower fitting error or passing fingertip guards', 1)
    doc.write_text(text)
    bundle_readme = ('# HUG constrained refinement evidence\n\n14 October 2026. ' + result + '\n\n'
        '[The method](../../docs/HUG_CONSTRAINED_REFINEMENT.md), [independent report](validation/REPORT.md)\n'
        'and [all-case summary](validation/summary.json) distinguish engineering\n'
        'source checks, candidate selection and strict scientific mapping acceptance.\n'
        'A retained original is an explicit refusal, not a corrected mapping.\n\n'
        '`implementation/` preserves the earlier implementation-only checkpoint,\n'
        'whose pending status was accurate at its checkpoint. Its immutable manifest\n'
        'binds the isolated API, 50 new manufactured tests and 120 installed API checks.\n'
        '`fit/` preserves complete canonical runtime V2.1 singleton reports, the merged\n'
        'candidate and all recorded major-callback histories, compressed byte-exact.\n'
        '[The clarified fit report](fit/REPORT_COUNTER_CLARIFIED.md) is the reader\n'
        'version. The original draft/report and its manifest remain byte-exact;\n'
        'a separately bound note corrects only the wording of the 120-iteration\n'
        'budget. Actual solver iterations and major callbacks are separate counters.\n'
        'The pre-audit fit summary retains its then-pending audit status; the completed\n'
        'independent result is in `validation/summary.json`.\n'
        '`validation/` holds the independent all-six preflight and full nine-resolution\n'
        'reports, unchanged rule, summary, scripts and diagnostic figures. `direction/`\n'
        'retains the actual normal helper, manufactured fixtures and baseline parity.\n'
        '`history/` preserves interrupted, import-aborted and unexecuted versions plus\n'
        'separate measured performance fixtures; none is pooled into mapping results.\n\n'
        '[results_manifest.json](results_manifest.json) hashes all stored evidence.\n'
        '[compression_bindings.json](compression_bindings.json) binds the full original\n'
        'canonical report/log bytes to compressed copies. Licensed source assets,\n'
        'saved scene binaries and query arrays stay in their existing runtime paths.\n'
        'Frozen scripts retain the recorded absolute input paths; adapting paths\n'
        'requires a separately recorded reproduction configuration.\n\n'
        'All 114 original protected source/model/paper/input bytes remain unchanged;\n'
        'the installed new API/test are separately bound. Canonical fit guard selftests\n'
        'total 18, with zero actual physics calls; the independent guards also report\n'
        'zero. No inference, eval episode, outcome/label, predictor training or advance\n'
        'to another generator occurred. The default mapper does not invoke this API.\n')
    (BUNDLE / 'README.md').write_text(bundle_readme)
    continuation = ('\n\n## 14 October 2026 — Joint fingertip/direction fitting and unchanged audit completed\n\n'
        + result + '\n\n'
        'The isolated 24-joint constrained API and native-facing objective are\n'
        'implemented. Each fingertip must retain at most its original native error\n'
        'plus the separately declared 10 nm engineering arithmetic allowance; the\n'
        'computed mean remains ≤10 mm. Invalid fixed direction sites retain penalties,\n'
        'native-away targets are preserved, missing part mass is not redistributed,\n'
        'and no successful-lift filter is used. Fifty new manufactured tests and all\n'
        '120 installed new/prior surface API checks pass. These are implementation\n'
        'checks, not full geometric equivalence. The default mapper is not promoted.\n\n'
        'Final independent FK/source checks and the nine unchanged static surface\n'
        'audits are complete. Optimizer nonconvergence/refusal is neither proof of\n'
        'global infeasibility nor a native grasp-failure label. Preserve the frozen\n'
        '1/2/5 mm rule, query cells, numerical assumptions, all source poor proposals\n'
        'and each versioned result. Incomplete/import-aborted/unexecuted histories\n'
        'and unused performance helpers remain separate. All 114 protected original\n'
        'bytes, predictor and manuscript match; the new API/test are separately\n'
        'bound. No new inference, physical eval data, outcome labels, training or\n'
        'other-generator work occurred. [Completed evidence](HUG_CONSTRAINED_REFINEMENT.md).\n\n'
        'Next technical work, if authorized, is a separately frozen constraint-feasible\n'
        'refinement or a documented change of the mapping/embodiment claim followed\n'
        'by appropriate validation. Do not loosen acceptance or call a fallback a\n'
        'repair. Contact/failure-label equivalence, unseen-object validation and\n'
        'predictor adequacy remain unresolved or untested.\n')
    for name in ('RESEARCH_DECISIONS.md', 'PAPER_PROGRESS.md', 'THREAD_HANDOFF.md'):
        p = ROOT / 'docs' / name
        p.write_text(p.read_text() + continuation)
    appendix = ('\n\n## 14 October 2026 — Completed constrained continuation\n\n' + result + '\n\n'
        '[Fingertip and native-direction method/evidence](HUG_CONSTRAINED_REFINEMENT.md)\n'
        'supersedes the proposed next solve as a completed development attempt. The\n'
        'previous method and its results above remain separate history. The same\n'
        'acceptance rule, domains and numerical assumptions apply. No default mapper\n'
        'promotion, new eval data, predictor training or next-generator advance\n'
        'follows from this result.\n')
    for name in ('HUG_SURFACE_REFINEMENT.md', 'HUG_MAPPING_CONFIRMATION.md', 'HUG_MAPPING_ACCEPTANCE.md'):
        p = ROOT / 'docs' / name
        p.write_text(p.read_text() + appendix)
    p = ROOT / 'docs/HUG_STATIC_VALIDATION_PROTOCOL.md'
    t = p.read_text()
    t = t.replace('Version 3 of this measurement/acceptance status note.', 'Version 4 of this measurement/acceptance status note.', 1)
    start = t.index('## Current correction status, 14 October 2026')
    end = t.index('## Completed checks and frozen acceptance, 14 October 2026', start)
    current = ('## Current correction status, 14 October 2026\n\n' + result + '\n\n'
        '[The completed constrained/direction-aware solve](HUG_CONSTRAINED_REFINEMENT.md)\n'
        'retains the original five-finger caps, strict 10 mm mean and unchanged\n'
        'scientific acceptance rule. Original-pose fallback is not a surface repair.\n'
        'The earlier staged/interpolated correction remains separate documented\n'
        'history. Experimental APIs stay separate from the default mapper; no new\n'
        'eval data or other-generator advance occurred. No current result certifies\n'
        'native contact/failure-label equivalence or independent predictor adequacy.\n\n')
    p.write_text(t[:start] + current + t[end:])
    print(json.dumps({'completed_cases': 6, 'completed_resolutions': 9, 'corrections_returned': corrected, 'original_fallback_count': fallback, 'strict_status_counts': n}))

if __name__ == '__main__':
    main()
