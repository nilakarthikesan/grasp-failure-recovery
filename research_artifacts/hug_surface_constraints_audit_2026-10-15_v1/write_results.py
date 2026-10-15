"""Render a concise evidence index from completed, independently checked reports."""
from pathlib import Path
import json
D=Path(__file__).parent;s=json.loads((D/'summary.json').read_bytes());assert s['all_checks_pass']
lines=['# Direct HUG surface-constraint audit results','',
'This is a static development-case mapping audit of the same six existing HUG grasps. It contains no new inference, physics rollout, evaluation dataset, outcome labels or model training. The Shadow Hand and original acceptance criteria were retained.','',
f"All six independent source/FK/root/domain and original fingertip-cap checks passed. The {s['fresh_full_grid_measurements']} fresh grid comparisons returned proximity {s['proximity_status_counts']}, pointwise orientation {s['pointwise_orientation_status_counts']}, and strict relationship {s['strict_status_counts']}. These counts are case-resolution comparisons, not independent grasps.",'',
'The predeclared cross-version selection uses the identical exact retained-witness rank with stable V1 ties. It retained V1 for all six cases; five were exact ties, and Mug A had a better V1 rank. This is a fitting selection, not scientific acceptance. Original fallback and refused diagnostics remain explicitly labeled.','',
'| Case index | Role | Palm shift (mm) | Original → selected mean tip error (mm) | Resolution: proximity / orientation / strict |',
'|---|---|---:|---:|---|']
for c in s['records']:
    results='; '.join(f"{den} mm: {r['proximity']} / {r['pointwise_orientation']} / {r['strict_relationship']}" for den,r in c['resolutions'].items())
    lines.append(f"| {c['index']} | {c['audit_selection_role']['audited_pose_role']} | {c['palm_displacement_norm_mm']:.6f} | {c['original_mean_tip_error_mm']:.6f} → {c['selected_mean_tip_error_mm']:.6f} | {results} |")
lines += ['', '## Saved fitting witnesses are separate evidence','',
'The following counts refer only to the fixed capped retained rows, including repeated nested-band/history records. They are not full-grid coverage or independent sites. Orientation with no saved row is unavailable, not passed.','',
'| Case | Proximity retained rows passed: original → selected | Orientation retained rows passed: original → selected | Orientation coverage |',
'|---|---:|---:|---|']
for c in s['records']:
    w=c['fit_retained_witness_diagnostics'];b=w['baseline'];a=w['selected']
    lines.append(f"| {c['index']} | {b['proximity_rows_pass']}/{b['proximity_retained_rows']} → {a['proximity_rows_pass']}/{a['proximity_retained_rows']} | {b['orientation_rows_pass']}/{b['orientation_retained_rows']} → {a['orientation_rows_pass']}/{a['orientation_retained_rows']} | {a['orientation_witness_coverage']} |")
lines += ['', '## Palm bounds on the unchanged coarse grid','',
'Values below use the same 2 mm band, 2 mm grid and primary 10 nm arithmetic allowance. They are conditional lower/upper bounds on anatomical proximity mismatch. Anatomical parts overlap and must not be summed as physical contact area. A smaller lower bound alone does not prove smaller total mismatch.','',
'| Case | Lost lower: original → selected (mm²) | Lost upper (mm²) | Added lower (mm²) | Added upper (mm²) |',
'|---|---:|---:|---:|---:|']
for c in s['records']:
    p=c['resolutions']['2']['same_2mm_band_part_comparison_original_vs_audited']['palm'];vals=[]
    for direction,bound in [('lost','lower'),('lost','upper'),('added','lower'),('added','upper')]:
        vals.append(f"{p['original_'+direction+'_'+bound+'_mm2']:.6f} → {p['audited_'+direction+'_'+bound+'_mm2']:.6f}")
    lines.append('| '+str(c['index'])+' | '+' | '.join(vals)+' |')
lines += ['', '## Interpretation and reproducibility','',
'These numerical checks do not establish faithful surface mapping or failure-label preservation. Definite proximity mismatches and reliable sign reversals remain failures. Absence of a sampled reversal remains unresolved for continuous orientation. The independent-interval identity control also shows why ambiguous cells cannot certify zero possible mismatch even when surfaces agree; that certificate limitation does not erase the observed definite failures.','',
'Replaying the identical fitter import sequence showed four effective Numba threads per process, overriding the requested two. The unchanged launch drivers contain no later reset, so the primary archives disclose the inferred four-thread launch setting rather than claiming compliance with the requested limit. The final independent grid audit resets threads after imports and verifies two active threads per process, with at most two audit processes.','',
f"- Original acceptance-rule SHA: `{s['acceptance_rule_sha256']}`.",
f"- Final audit protocol SHA: `{s['protocol_sha256']}`.",
f"- Pooled selected candidate SHA: `{s['candidate_sha256']}`.",
f"- Independent arithmetic/source read-back: {s['consistency_checks']} checks passed; {s['saved_witness_records_across_nested_bands']} saved orientation witness records, {s['distinct_saved_witnesses_by_case_density_query_direction']} distinct by case/density/query/direction.",
f"- Preflight time: {s['preflight_runtime_seconds']:.3f} s. Cohort times: {', '.join(f'{x:.3f} s' for x in s['cohort_runtime_seconds'])}.",
'- See `summary.json` for per-part lower/upper comparisons and orientation denominators; full reports retain all bands and allowances.',
'- Separate saved-row and document-review supplements preserve their own scopes. Document review covers method and retained-row interpretation, not pending-at-review full-grid outcome prose.','']
(D/'RESULTS.md').write_text('\n'.join(lines))
print(str(D/'RESULTS.md'))
