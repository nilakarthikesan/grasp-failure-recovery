from pathlib import Path
import ast,hashlib,json,platform
from datetime import datetime,timezone
OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OLD=ROOT/'research_artifacts/hug_feasible_refinement_2026-10-14/seed_stage'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def function_ast(path,name):
    nodes=[n for n in ast.walk(ast.parse(Path(path).read_text())) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1
    return ast.dump(nodes[0],include_attributes=False)
p=json.loads((OUT/'protocol.draft.json').read_text())
for name in ('set_pose','evaluate','evaluate_tips'):
    assert function_ast(OLD/'run_seed_stage.py',name)==function_ast(OUT/'run_guarded_geometry.py',name),name
for name in ('geometry_helpers.py','exact_direction.py'):assert sha(OLD/name)==sha(OUT/name)
p['implementation_sha256']={str(OUT/name):sha(OUT/name) for name in ('run_guarded_geometry.py','feasible_surface_refinement.py','surface_component_guard.py','keyed_direction_evidence.py','test_keyed_direction_evidence.py','exact_direction.py','geometry_helpers.py')}
for table in ('source_sha256','implementation_sha256','query_cache_sha256','full_cache_sha256'):
    for path,h in p[table].items():assert sha(path)==h,path
for path,h in json.loads((OUT/'protected_hashes.json').read_text()).items():assert sha(path)==h,path
p['runtime_dependency_path']='/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages'
p['draft_review_ready_utc']=datetime.now(timezone.utc).isoformat()
(OUT/'protocol.draft.json').write_text(json.dumps(p,indent=2)+'\n')
(OUT/'pre_execution_driver_checks.json').write_text(json.dumps({'scientific_kernel_AST_identical':p['scientific_kernel_AST_identical'],'literal_helpers_exact_copies':True,'all_pre_execution_input_hashes_verified':True,'manufactured_keyed_evidence_tests':19,'manufactured_tests_passed':True,'first_test_collection_attempt':'Failed due supplemental existing dependency paths omitted; no tests or geometry ran. Reran with recorded installed dependency paths.','actual_geometry_executed':False,'draft_protocol_sha256':sha(OUT/'protocol.draft.json'),'driver_sha256':sha(OUT/'run_guarded_geometry.py')},indent=2)+'\n')
print(json.dumps({'review_ready':True,'driver_sha256':sha(OUT/'run_guarded_geometry.py'),'draft_protocol_sha256':sha(OUT/'protocol.draft.json'),'bound_implementations':len(p['implementation_sha256']),'bound_sources':len(p['source_sha256'])}))
