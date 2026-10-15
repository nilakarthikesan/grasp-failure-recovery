"""Read-only frozen-report/logic review; no geometry, simulator or fitting."""
from pathlib import Path
import json,hashlib,ast
B=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14')
O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[B/'acceptance/acceptance_rule.json',B/'acceptance/evaluate_acceptance.py',B/'sampling/cell_intervals.py',B/'sampling/sampling_uncertainty.json',B/'sampling/protocol.json',B/'orientation/protocol.json',Path('/private/tmp/hug_surface_constraints_review_2026-10-15_v1/identity_interval_control.json')]
bindings={str(p):sha(p) for p in paths};rule=json.loads(paths[0].read_text());assert sha(paths[0])=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4'
report=json.loads(paths[3].read_text());protocol=json.loads(paths[4].read_text());assert report['protocol_sha256']==sha(paths[4]);epsilon=protocol['primary_allowance_m']
classes={'far':(False,False),'ambiguous':(False,True),'near':(True,True)};table=[]
for n,(nd,np) in classes.items():
 for m,(md,mp) in classes.items():
  lo=nd and not mp;lu=np and not md;ao=md and not np;au=mp and not nd
  zero=not lu and not au;expected=(n==m and n!='ambiguous');assert zero==expected
  table.append({'native_class':n,'mapped_class':m,'lost_lower_mask':lo,'lost_upper_mask':lu,'added_lower_mask':ao,'added_upper_mask':au,'zero_both_upper':zero})
rows=[]
for c in report['cases']:
 for density,res in c['resolutions'].items():
  blocked=[]
  for band in rule['bands_mm']:
   for part in rule['anatomical_parts']:
    v=res['numerical_allowances'][str(epsilon)]['bands'][str(band)]['independent_parts'][part]['native_area'];ambiguous_count=v['possible_cell_count']-v['definite_cell_count'];area=v['uncertain_mm2'];assert area>=0 and ambiguous_count>=0
    if area>0:
     assert ambiguous_count>0
     blocked.append({'part':part,'band_mm':band,'native_ambiguous_area_mm2':area,'native_ambiguous_cell_count':ambiguous_count})
  rows.append({'case_id':c['case_id'],'source_inventory_index':c['source_inventory_index'],'density_mm':int(density),'source_report_cache_sha256':res['cache_sha256'],'required_part_band_checks':len(rule['bands_mm'])*len(rule['anatomical_parts']),'blocked_part_band_checks':len(blocked),'native_ambiguity_precludes_zero_upper_certificate_for_any_mapped_pose':bool(blocked),'blocked_checks':blocked,'max_single_part_band_ambiguous_area_mm2':max([b['native_ambiguous_area_mm2'] for b in blocked],default=0.)})
code=paths[1].read_text();tree=ast.parse(code);strict=[];coverage=[]
for node in ast.walk(tree):
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='strict' for t in node.targets):strict.append({'line':node.lineno,'expression':ast.unparse(node.value),'string_constants':sorted({n.value for n in ast.walk(node.value) if isinstance(n,ast.Constant) and isinstance(n.value,str)})})
 if isinstance(node,ast.Dict):
  for k,v in zip(node.keys,node.values):
   if isinstance(k,ast.Constant) and k.value=='continuous_orientation_coverage_status':coverage.append({'line':node.lineno,'value':ast.literal_eval(v)})
assert len(strict)==1 and 'PASS' not in strict[0]['string_constants'];assert coverage and all(x['value']=='UNRESOLVED' for x in coverage)
result={'schema':'hug_acceptance_contract_review_v1','source_bindings':bindings,'primary_epsilon_m':epsilon,'truth_table':table,'proximity_zero_upper_iff':'Every positive-area cell is certified-near in both fields or certified-far in both fields; neither field is ambiguous.','fixed_native_ambiguity_theorem':'If native possible=true and definite=false on a positive-area cell, zero lost upper requires mapped definite=true, which implies mapped possible=true and hence added upper>0. No mapped pose can make both separate-field upper bounds zero for that fixed cell.','scope_of_impossibility':'The fixed independent-bound certificate, not geometric equivalence, morphology reachability or all possible validation methods.','current_saved_grid_count':len(rows),'saved_grids_with_native_only_certificate_blocker':sum(r['native_ambiguity_precludes_zero_upper_certificate_for_any_mapped_pose'] for r in rows),'saved_report_rows':rows,'evaluator_strict_assignments':strict,'evaluator_orientation_coverage_assignments':coverage,'full_strict_pass_reachable_in_frozen_evaluator':False,'frozen_rule_changed':False,'real_geometry_queries':0,'FK_calls':0,'new_proposals':0,'new_eval_data':False,'physics_steps':0}
(O/'contract_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:print(r['source_inventory_index'],r['density_mm'],r['blocked_part_band_checks'],r['max_single_part_band_ambiguous_area_mm2'])
print('sha256',sha(O/'contract_evidence.json'))
