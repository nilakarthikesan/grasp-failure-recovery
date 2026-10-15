"""Descriptive report of frozen completed selection; never ranks or modifies poses."""
from pathlib import Path
import hashlib, json
import numpy as np

OUT = Path(__file__).parent
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol = read(OUT/'protocol.json')
candidate = read(OUT/'candidate_targets.json')
report = read(OUT/'selection_report.json')
assert report['passed'] and report['candidate_sha256'] == sha(OUT/'candidate_targets.json')
assert candidate['protocol_sha256'] == report['protocol_sha256'] == sha(OUT/'protocol.json')
sources = {method['name']: {row['source_inventory_index']: row for row in read(method['candidate_path'])['cases']} for method in protocol['methods']}
rows = []
for case in candidate['cases']:
    selected = case['selection_source']
    original = sources[selected['method_name']][case['source_inventory_index']]
    caps = np.asarray(case['authoritative_original_tip_caps_sqrt_einsum_m'])
    tips = np.asarray(case['selected_tip_norms_sqrt_einsum_m'])
    rows.append({
        'source_inventory_index': case['source_inventory_index'],
        'object_id': case['object_id'],
        'audited_pose_role': case['audited_pose_role'],
        'selected_method': selected['method_name'],
        'callback_sequence': selected['callback_sequence'],
        'palm_displacement_world_m': case['palm_displacement_world_m'],
        'palm_displacement_norm_m': case['palm_displacement_norm_m'],
        'baseline_objective': original['baseline_fresh_diagnostics']['objective'],
        'candidate_objective': selected['objective'],
        'baseline_mean_fingertip_error_m': float(caps.mean()),
        'candidate_mean_fingertip_error_m': float(tips.mean()),
        'maximum_literal_original_cap_excess_m': float(np.max(tips-caps)),
        'all_five_caps_pass': bool(np.all(tips<=caps)),
        'direction_guard_pass': case['direction_guard_pass'],
        'direction_guard_refusal_reasons': case['extra_direction_guard']['refusal_reasons'],
        'guarded_mapper_selected_fallback': case['guarded_mapper_selected_fallback'],
        'diagnostic_pose_not_promoted': case['diagnostic_pose_not_promoted']
    })
result = {'schema': 'hug_translated_palm_diagnostic_compact_review_table_v1',
          'candidate_sha256': sha(OUT/'candidate_targets.json'),
          'selection_report_sha256': sha(OUT/'selection_report.json'),
          'protocol_sha256': sha(OUT/'protocol.json'),
          'table': rows,
          'scope': 'Descriptive frozen-output summary. Direction guard is an extra selector, not the separate scientific surface acceptance.'}
with (OUT/'compact_review_table.json').open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
lines = ['# Static translated-palm diagnostic summary', '',
         'These poses are diagnostic only. Original mapper outputs and extra direction refusals are retained; scientific surface acceptance is reported separately.', '',
         '| Index | Method / callback | Palm XYZ shift (mm) | Shift (mm) | Objective: original → candidate | Tip mean: original → candidate (mm) | Max exact cap excess (m) | Extra direction guard |',
         '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
for row in rows:
    xyz=', '.join(f'{1000*x:.6f}' for x in row['palm_displacement_world_m'])
    lines.append(f"| {row['source_inventory_index']} | {row['selected_method']} / {row['callback_sequence']} | ({xyz}) | {1000*row['palm_displacement_norm_m']:.6f} | {row['baseline_objective']:.9g} → {row['candidate_objective']:.9g} | {1000*row['baseline_mean_fingertip_error_m']:.6f} → {1000*row['candidate_mean_fingertip_error_m']:.6f} | {row['maximum_literal_original_cap_excess_m']:.9g} | {'PASS' if row['direction_guard_pass'] else 'REFUSED'} |")
lines += ['', 'A nonpositive cap excess means all five literal original-cap constraints pass. The table does not add any numerical tolerance.', '']
for row in rows:
    lines.append(f"- {row['source_inventory_index']}: " + ('No extra guard refusal.' if row['direction_guard_pass'] else '; '.join(row['direction_guard_refusal_reasons'])))
with (OUT/'compact_review_table.md').open('x') as stream: stream.write('\n'.join(lines)+'\n')
print(json.dumps({'rows':len(rows),'json_sha256':sha(OUT/'compact_review_table.json'),'markdown_sha256':sha(OUT/'compact_review_table.md')}))
