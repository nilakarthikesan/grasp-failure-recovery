from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,platform,importlib.metadata
OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OLD=ROOT/'research_artifacts/hug_feasible_refinement_2026-10-14/seed_stage'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fn(path,name):
    found=[n for n in ast.walk(ast.parse(Path(path).read_text())) if isinstance(n,ast.FunctionDef) and n.name==name];assert len(found)==1
    return ast.dump(found[0],include_attributes=False)
p=json.loads((OUT/'protocol.draft.json').read_text())
assert p['execution_authorized_by_root_review'] is False and p['draft_only_not_executed']
assert p['allowed_cohorts']==[[8,16,24],[12,20,28]] and p['maximum_concurrent_processes']==2
assert p['solver_configuration']['maximum_pattern_sweeps']==2 and p['solver_configuration']['maximum_full_geometry_evaluations']==64
for table in ('source_sha256','implementation_sha256','query_cache_sha256','full_cache_sha256'):
    for path,h in p[table].items():assert sha(path)==h,path
assert sha(OUT/'protected_hashes.json')==p['protected_snapshot_sha256']
for path,h in json.loads((OUT/'protected_hashes.json').read_text()).items():assert sha(path)==h,path
for name in ('set_pose','evaluate','evaluate_tips'):assert fn(OUT/'run_guarded_geometry.py',name)==fn(OLD/'run_seed_stage.py',name)
for name in ('geometry_helpers.py','exact_direction.py'):assert sha(OUT/name)==sha(OLD/name)
p.update(frozen_utc=datetime.now(timezone.utc).isoformat(),execution_authorized_by_root_review=True,draft_only_not_executed=False,
    root_review='Root read scientific kernels, keyed evidence and finite budgets and approved launch. Scheduling-only update: two fixed independent cohorts [8,16,24] and [12,20,28], maximum two processes x two Numba threads; freeze strict merger before execution. No scientific tuning or reruns under this version.',
    manufactured_keyed_evidence_tests_passed=19,
    python_version=platform.python_version(),runtime_packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','mujoco','numba')})
with (OUT/'protocol.json').open('x') as f:json.dump(p,f,indent=2,allow_nan=False);f.write('\n')
checks={'scientific_kernel_AST_identical':dict.fromkeys(('set_pose','evaluate','evaluate_tips'),True),'literal_helpers_exact_copies':True,
    'all_pre_execution_input_hashes_verified':True,'manufactured_keyed_evidence_tests_passed':19,'actual_geometry_executed_before_freeze':False,
    'frozen_protocol_sha256':sha(OUT/'protocol.json'),'driver_sha256':sha(OUT/'run_guarded_geometry.py'),'merger_sha256':sha(OUT/'merge_guarded_geometry.py'),
    'root_review_scheduling_change':'Two disjoint predetermined cohorts, maximum two processes x two Numba threads.'}
with (OUT/'frozen_pre_execution_checks.json').open('x') as f:json.dump(checks,f,indent=2);f.write('\n')
print(json.dumps(checks))
