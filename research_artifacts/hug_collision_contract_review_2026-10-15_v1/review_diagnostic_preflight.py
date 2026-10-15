"""Independent JSON/AST review only; does not import the scene runtime."""
from pathlib import Path
import ast, hashlib, json
import numpy as np

HERE=Path(__file__).resolve().parent
FIT=Path('/private/tmp/hug_exposed_geometry_2026-10-15_v1')
WIT=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json')
ACCEPTANCE_SHA='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(condition, label):
    assert condition,label
    checks.append(label)

protocol_path=FIT/'protocol.draft.json'
p=read(protocol_path)
initial=sha(protocol_path)
bindings={}
for group in ('input_sha256','implementation_sha256','preflight_sha256'):
    for path,digest in p[group].items():
        check(sha(path)==digest,'hash:'+path)
        bindings[path]=digest
check(not p['execution_authorized_after_independent_review'],'draft_not_yet_authorized')
check(p['cases']==[8,12,16,20,24,28],'fixed_six_cases')
check(p['no_acceptance_change'] and not p['new_eval_data'] and not p['new_inference'] and not p['new_fit'] and not p['production_promotion'] and p['physics_steps']==0,'scope_fixed')
check(p['numerical_classification_tolerance_m']==1e-8,'inherited_classification_tolerance')
acceptances=[(path,digest) for path,digest in bindings.items() if path.endswith('/acceptance/acceptance_rule.json')]
check(len(acceptances)==1 and acceptances[0][1]==ACCEPTANCE_SHA,'scientific_acceptance_unchanged')
w=read(WIT);pool=read(p['pooled_candidate_path'])
cases={c['source_inventory_index']:c for c in pool['cases']}
check(len(w['whole_cell_rows'])==178 and len(w['orientation_rows'])==134,'all_312_saved_rows_preserved')
per_case=[]
for index in p['cases']:
    frozen=p['source_rows'][str(index)];c=cases[index]
    for field,key in [('proximity_indices','whole_cell_rows'),('orientation_indices','orientation_rows')]:
        expected=[j for j,r in enumerate(w[key]) if r['source_inventory_index']==index]
        check(frozen[field]==expected,f'{index}:{field}:ordered_exact')
        for j in expected:
            r=w[key][j]
            native=r['row']['native_sha256'] if key=='whole_cell_rows' else r['native_sha256']
            check(native==c['native_sha256'],f'{index}:{key}:{j}:native')
            check(bindings[r['source_report']]==r['source_report_sha256'],f'{index}:{key}:{j}:report_hash')
    check(frozen['native_sha256']==c['native_sha256'],f'{index}:frozen_native')
    check(np.array_equal(frozen['original_state_scaled'],np.zeros(27)),f'{index}:exact_original_zero')
    check(np.array_equal(frozen['pooled_state_scaled'],c['selected_state_scaled']),f'{index}:exact_pooled_state')
    b=np.asarray(c['bounds_scaled']);x=np.asarray(c['selected_state_scaled'])
    check(b.shape==(27,2) and np.all(x>=b[:,0]) and np.all(x<=b[:,1]),f'{index}:saved_scaled_domain')
    for role,key in [('original','baseline_constraint_diagnostics'),('pooled','selected_constraint_diagnostics')]:
        d=c[key];P=len(frozen['proximity_indices']);O=len(frozen['orientation_indices'])
        check(len(d['part_distance_m'])==P+O,f'{index}:{role}:distance_count')
        check(len(d['proximity']['passed'])==P and len(d['exact_orientation']['passed'])==O,f'{index}:{role}:literal_row_counts')
    per_case.append({'source_inventory_index':index,'proximity_rows':len(frozen['proximity_indices']),'orientation_rows':len(frozen['orientation_indices']),'row_pose_records':2*(len(frozen['proximity_indices'])+len(frozen['orientation_indices']))})
check(sum(c['row_pose_records'] for c in per_case)==624,'624_fixed_row_pose_records')
check(p['finite_budget']['robot_poses']==12 and p['finite_budget']['retained_row_pose_records']==624 and p['finite_budget']['hand_component_query_points']==624*45*3 and p['finite_budget']['query_batch_calls']==12*45*2,'finite_budget_arithmetic')
for name in ('run_diagnostic.py','exposed_classifier.py'):
    tree=ast.parse((FIT/name).read_text())
    bad=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('mj_step','mj_step1','mj_step2','minimize','least_squares')]
    check(not bad,name+':no_integrator_or_optimizer_calls')
for name in ('classifier_controls_review.json','classifier_final_controls/classifier_controls_review.json','compiled_contract.json','independent_union_controls.json'):
    path=HERE/name
    bindings[str(path)]=sha(path)
primary=FIT/'manufactured_controls.json';bindings[str(primary)]=sha(primary)
check(read(HERE/'classifier_controls_review.json')['passed'],'independent_classifier_controls_pass')
check(read(HERE/'classifier_final_controls/classifier_controls_review.json')['passed'],'final_independent_classifier_controls_pass')
check(read(HERE/'classifier_final_controls/classifier_controls_review.json')['source_bindings'][str(FIT/'exposed_classifier.py')]==sha(FIT/'exposed_classifier.py'),'controls_bind_final_classifier')
check(read(primary)['passed'],'primary_manufactured_controls_pass')
for path,digest in bindings.items():check(sha(path)==digest,'lifecycle_after:'+path)
check(sha(protocol_path)==initial,'draft_unchanged_during_review')
report={'schema':'independent_exposed_geometry_preflight_review_v1','passed':True,
        'protocol_draft_sha256':initial,'source_bindings':bindings,'checks_count':len(checks),'checks':checks,
        'per_case':per_case,'unchanged_scientific_acceptance_sha256':ACCEPTANCE_SHA,
        'new_saved_grasp_queries':0,'new_fk_calls':0,'physics_steps':0,'new_eval_data':False,
        'manual_review':['Exterior nearest-component distance is conditional exact distance to occupied-union boundary; interior uses bounds only.',
                         'Per-part distance is to boundary-of-union intersect same-part component boundaries, not all component surfaces.',
                         'Retained-cell persistence uses fixed source radius/allowance and additional conservative classification eps; absence of persistence proof is not repair.',
                         'No orientation recomputation or normal-uniqueness claim. Historical literal rows are preserved.',
                         'No new inference, fitting, object grid, integrator, or production promotion.',
                         'Runtime assertions bind exact protocol rows/states and refuse source mutation/output overwrite; final numerical outputs require later review.'],
        'limitations':['Floating oracle-dependent bounds, not interval arithmetic certificates.',
                       'Capped saved witnesses only, not all cells or held-out evidence.',
                       'Union boundary is not identical to MuJoCo pairwise contact behavior or finite-object accessibility.',
                       'Prior strict failures remain unchanged.']}
out=HERE/'diagnostic_preflight_review.json'
assert not out.exists(),'preserve earlier review if rerun needed'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'checks_count':len(checks),'path':str(out),'sha256':sha(out),'draft_sha256':initial,'per_case':per_case},indent=2))
