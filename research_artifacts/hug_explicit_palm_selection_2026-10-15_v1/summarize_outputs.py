"""Summarize completed supplemental results without changing selection."""
from pathlib import Path
import json,hashlib
import numpy as np
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
candidate=read(OUT/'candidate_targets.json');report=read(OUT/'selection_report.json')
assert report['passed'] and report['candidate_sha256']==sha(OUT/'candidate_targets.json')
rows=[]
for case in candidate['cases']:
    origin=case['explicit_translation_proposal_origin'];caps=np.asarray(case['authoritative_original_tip_caps_sqrt_einsum_m']);tips=np.asarray(case['selected_tip_norms_sqrt_einsum_m']);requested=np.asarray(origin['proposal_translation_world_m'])
    rows.append({'source_inventory_index':case['source_inventory_index'],'method':origin['source_version'],'seed_index_zero_based':origin['proposal_seed_index'],'seed_label':origin['proposal_seed_label'],'proposal_sequence':origin['proposal_sequence'],'callback_sequence':origin['callback_sequence'],'requested_translation_world_m':requested.tolist(),'requested_translation_norm_m':float(np.linalg.norm(requested)),'actual_palm_translation_world_m':case['palm_displacement_world_m'],'actual_palm_translation_norm_m':case['palm_displacement_norm_m'],'baseline_objective':case['baseline_fresh_diagnostics']['objective'],'candidate_objective':case['selected_fresh_diagnostics']['objective'],'original_mean_tip_error_m':float(caps.mean()),'candidate_mean_tip_error_m':float(tips.mean()),'max_exact_original_cap_excess_m':float(np.max(tips-caps)),'direction_guard_pass':case['direction_guard_pass'],'direction_guard_refusal_reasons':case['extra_direction_guard']['refusal_reasons'],'diagnostic_not_promoted':True})
with (OUT/'compact_review_table.json').open('x') as f:json.dump({'schema':'hug_explicit_palm_seed_diagnostic_table_v1','candidate_sha256':sha(OUT/'candidate_targets.json'),'selection_report_sha256':sha(OUT/'selection_report.json'),'rows':rows},f,indent=2);f.write('\n')
lines=['# Explicit translation-seed coverage supplement','',
'This one post-hoc supplement preserves all original selection results. It uses three existing grasps, no new fitting, and no physical episodes. All listed poses remain refused by the extra direction selector and are not promoted.','',
'| Case | Method / seed / proposal / callback | Requested shift norm (mm) | Actual shift norm (mm) | Objective original → candidate | Tip mean original → candidate (mm) | Largest signed cap excess (m) |',
'| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for r in rows:
    lines.append(f"| {r['source_inventory_index']} | {r['method']} / {r['seed_index_zero_based']} / {r['proposal_sequence']} / {r['callback_sequence']} | {1000*r['requested_translation_norm_m']:.6f} | {1000*r['actual_palm_translation_norm_m']:.6f} | {r['baseline_objective']:.9g} → {r['candidate_objective']:.9g} | {1000*r['original_mean_tip_error_m']:.6f} → {1000*r['candidate_mean_tip_error_m']:.6f} | {r['max_exact_original_cap_excess_m']:.9g} |")
lines+=['','Seed indices are zero based; proposal and callback sequences are one based. Nonpositive cap excess means every literal original cap passes, without an acceptance allowance.']
for r in rows:lines+=['',f"- {r['source_inventory_index']}: "+'; '.join(r['direction_guard_refusal_reasons'])]
with (OUT/'compact_review_table.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({'table_sha256':sha(OUT/'compact_review_table.json'),'rows':len(rows)}))
