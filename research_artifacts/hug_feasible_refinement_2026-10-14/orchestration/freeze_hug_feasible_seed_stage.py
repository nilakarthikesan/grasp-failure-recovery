from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json

OUT=Path('/private/tmp/hug_feasible_refinement_2026-10-14/seed_stage')
OLD=Path('/private/tmp/hug_constrained_refinement_runtime_v2_1_2026-10-14')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def function_ast(path,name):
    found=[n for n in ast.walk(ast.parse(Path(path).read_text())) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(found)==1
    return ast.dump(found[0],include_attributes=False)
assert not (OUT/'protocol.json').exists()
p=json.loads((OUT/'protocol.draft.json').read_text())
assert sha(OUT/'run_seed_stage.py')=='a3964fad49f6acdb8a49080b3211b6c0816f34c361b945c507ef7d2f23426808'
assert sha(OUT/'feasible_surface_refinement.py')=='2b4d8c30fe639d70bd8e4da0aa6406c2782ee2a3e7997683ed89db9edb935c4e'
for name in ('set_pose','evaluate','evaluate_tips'):
    assert function_ast(OUT/'run_seed_stage.py',name)==function_ast(OLD/'run_constrained_refinement.py',name),name
for table in ('source_sha256','implementation_sha256'):
    for path,h in p[table].items(): assert sha(path)==h,path
protected=json.loads((OUT/'protected_hashes.json').read_text())
assert len(protected)==89
for path,h in protected.items():
    assert sha(path)==h,path
    assert '/docs/paper/' not in path and '/proposals/contactgen.py' not in path and '/proposals/unidexgrasp.py' not in path
assert p['solver_configuration']['maximum_pattern_sweeps']==0
assert p['solver_configuration']['maximum_full_geometry_evaluations']==4
assert p['solver_configuration']['restoration_inward_target_m']==1e-9
assert p['scientific_kernel_AST_identical']==dict.fromkeys(('set_pose','evaluate','evaluate_tips'),True)
p['frozen_utc']=datetime.now(timezone.utc).isoformat()
p['execution_authorized_by_root_review']=True
p['draft_only_not_executed']=False
p['root_review']='Core API plus 26 manufactured tests and independent read-only review; literal old set_pose/evaluate/evaluate_tips AST checked. One all-six cohort, seed-only common configuration, 4 full calls maximum, exact original caps; no acceptance changes.'
(OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps({'frozen_protocol_sha256':sha(OUT/'protocol.json'),'protected_numeric_closure_count':len(protected),'before_new_restored_surface_metrics':True}))
