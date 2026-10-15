"""Read-only arithmetic replay of already independently reviewed exact rows."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = Path('/private/tmp/hug_surface_constraints_2026-10-15_v1/candidate_targets.json')
CLAIMS = Path('/private/tmp/hug_surface_constraints_local_change_2026-10-15_v1/retained_row_changes.json')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
claims = json.loads(CLAIMS.read_text())
source = json.loads(SOURCE.read_text())
checks = []
def check(name, condition):
    assert condition, name
    checks.append(name)

check('source_hash', claims['source_sha256'] == sha(SOURCE))
check('source_path', claims['source_path'] == str(SOURCE))
cases = []
for c in source['cases']:
    idx = c['source_inventory_index']
    claim = next(x for x in claims['cases'] if x['index'] == idx)
    a, b = c['baseline_constraint_diagnostics'], c['selected_constraint_diagnostics']
    mean_mm = sum(c['selected_tip_norms_sqrt_einsum_m']) / 5 * 1000
    check(f'{idx}:mean', abs(mean_mm - claim['mean_tip_mm']) <= 2e-15)
    output = {'index': idx, 'mean_tip_mm': mean_mm, 'changes': {}}
    for kind in ('proximity', 'orientation'):
        group = 'proximity' if kind == 'proximity' else 'exact_orientation'
        rows = c['constraint_source_rows'][kind]
        old, new = a[group]['passed'], b[group]['passed']
        check(f'{idx}:{kind}:lengths', len(rows) == len(old) == len(new))
        counts, gained, lost = {}, [], []
        for j, (row, before, after) in enumerate(zip(rows, old, new)):
            part = row['row']['anatomical_part' if kind == 'proximity' else 'native_part']
            count = counts.setdefault(part, dict(total=0, baseline_pass=0, selected_pass=0, gained=0, lost=0))
            for key, inc in [('total', 1), ('baseline_pass', before), ('selected_pass', after),
                             ('gained', after and not before), ('lost', before and not after)]:
                count[key] += int(inc)
            if before == after:
                continue
            out = dict(row_index=j, part=part,
                       band_mm=row['row']['band_mm'] if kind == 'proximity' else row['band_mm'],
                       source_report=row['source_report'], source_json_pointer=row['source_json_pointer'],
                       baseline_pass=before, selected_pass=after)
            if kind == 'proximity':
                out.update(relationship_change=row['row']['relationship_change'],
                           baseline_margin_m=a[group]['margin_m'][j], selected_margin_m=b[group]['margin_m'][j])
            else:
                out.update(baseline_eligible=a['exact_orientation_fields']['valid'][j],
                           selected_eligible=b['exact_orientation_fields']['valid'][j],
                           baseline_sign_margin=a[group]['sign_margin'][j], selected_sign_margin=b[group]['sign_margin'][j])
            (gained if after else lost).append(out)
        reconstructed = dict(by_part=counts, gained_rows=gained, lost_rows=lost)
        check(f'{idx}:{kind}:complete_exact_replay', reconstructed == claim['changes'][kind])
        output['changes'][kind] = reconstructed
    cases.append(output)

mug = next(c for c in cases if c['index'] == 24)
mugpro = mug['changes']['proximity']
check('mug_10_of_57', sum(p['selected_pass'] for p in mugpro['by_part'].values()) == 10 and
      sum(p['total'] for p in mugpro['by_part'].values()) == 57)
check('mug_part_distribution', {p: q['selected_pass'] for p,q in mugpro['by_part'].items()} ==
      dict(index=2, middle=2, ring=3, little=2, palm=1, thumb=0))
check('mug_all_gained_lost_native', len(mugpro['gained_rows']) == 10 and all(r['relationship_change'] == 'lost_native' for r in mugpro['gained_rows']))

apple = next(c for c in source['cases'] if c['source_inventory_index'] == 12)
lost = next(c for c in cases if c['index'] == 12)['changes']['orientation']['lost_rows']
check('apple_two_lost', len(lost) == 2)
check('apple_band_and_part', [r['band_mm'] for r in lost] == [2,5] and all(r['part'] == 'middle' for r in lost))
arows = [apple['constraint_source_rows']['orientation'][r['row_index']] for r in lost]
check('apple_same_query', arows[0]['row']['query_index'] == arows[1]['row']['query_index'] and
      arows[0]['row']['query_object_m'] == arows[1]['row']['query_object_m'] and
      arows[0]['original_query_cache_sha256'] == arows[1]['original_query_cache_sha256'])
check('apple_valid_both', all(r['baseline_eligible'] and r['selected_eligible'] for r in lost))
check('apple_sign_crossing', all(r['baseline_sign_margin'] > 0 > r['selected_sign_margin'] for r in lost))
mean_range = [min(c['mean_tip_mm'] for c in cases), max(c['mean_tip_mm'] for c in cases)]
check('mean_range', all(abs(x-y) <= 2e-15 for x,y in zip(mean_range,claims['mean_tip_range_mm'])))

report = dict(passed=True, scope='Saved-row identity and arithmetic only; zero new FK, geometry, fitting or eval.',
              source_bindings={str(p):sha(p) for p in (SOURCE,CLAIMS,Path(__file__))},
              checks_passed=len(checks), checks=checks, cases=cases, mean_tip_range_mm=mean_range,
              interpretation='Mug A repairs 10 of 57 retained example rows only; this does not establish total area reduction. Apple B loses two retained orientation records at one eligible query repeated at 2/5 mm. Sign margins include the frozen deadzone, not raw dot products.')
(BASE/'retained_row_change_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=True,checks_passed=len(checks),path=str(BASE/'retained_row_change_review.json'),sha256=sha(BASE/'retained_row_change_review.json'))))
