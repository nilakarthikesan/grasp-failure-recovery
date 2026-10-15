"""Add transparent candidate gates and separate fitting-row summaries before freeze."""
from pathlib import Path
import ast
D=Path(__file__).parent;assert not (D/'protocol.json').exists()
f=D/'summarize_audit.py';s=f.read_text()
needle="p=read(OUT/'protocol.json');candidate_path=Path(p['candidate_path']);candidate=read(candidate_path)"
block="""def fitting_row_summary(diag):
    prox=diag['proximity']['passed'];ori=diag['exact_orientation']['passed'];valid=diag['exact_orientation_fields']['valid'];literal=diag['literal_surface_pass']
    check('fitting_row_count_lengths',len(ori)==len(valid) and len(literal)==len(prox)+len(ori))
    check('fitting_row_pass_aggregate',literal==prox+ori and diag['all_witness_constraints_pass']==all(literal))
    return {'proximity_retained_rows':len(prox),'proximity_rows_pass':sum(prox),'proximity_rows_fail':len(prox)-sum(prox),'orientation_retained_rows':len(ori),'orientation_rows_pass':sum(ori),'orientation_rows_fail':len(ori)-sum(ori),'orientation_rows_eligible':sum(valid),'orientation_witness_coverage':diag['orientation_witness_coverage'],'all_retained_rows_pass':diag['all_witness_constraints_pass'],'rank':diag['rank'],'row_coverage_is_full_grid_acceptance':False}
"""
assert s.count(needle)==1;s=s.replace(needle,block+needle)
needle="    check(prefix+'role_preserved'"
idx=s.index(needle)
block="""    row['exact_original_tip_caps_pass']=True;row['original_tip_caps_mm']=[v*1000 for v in eng['frozen_authoritative_tip_caps_sqrt_einsum_m']];row['selected_tip_errors_mm']=[v*1000 for v in eng['candidate_tip_norms_sqrt_einsum_m']]
    row['fit_retained_witness_diagnostics']={'scope':'Fixed capped saved witness examples, including duplicate nested-band/history rows. Their satisfaction is not a full-grid mapping acceptance claim. Counts are recomputed from supplied per-row truth; separate independent fit review verifies arithmetic and exact geometry.','baseline':fitting_row_summary(targets[i]['baseline_constraint_diagnostics']),'selected':fitting_row_summary(targets[i]['selected_constraint_diagnostics'])}
    dom=c['candidate_domain_audit'];check(prefix+'declared_domain',dom['scaled_state_and_world_translation_within_declared_frozen_bounds'] and dom['domain_endpoint_arithmetic_allowance']==0 and all(b[0]<=v<=b[1] for v,b in zip(dom['selected_state_scaled'],dom['bounds_scaled'])) and all(b[0]<=v<=b[1] for v,b in zip(root['palm_displacement_world_m'],dom['world_displacement_AABB_m'])))
"""
s=s[:idx]+block+s[idx:];ast.parse(s);f.write_text(s)
print('Summary preparation complete; no geometry or physics execution.')
