from pathlib import Path
import json,hashlib,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/private/tmp/hug_se3_protected_fit_2026-10-15_v1')
from feasible_path_solver import DEFAULT_CONFIGURATION
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=Path('/private/tmp/hug_se3_segment_check_2026-10-15_v1/protocol.json')
p=json.loads(source.read_text());bindings=dict(p['bindings']);bindings[str(source)]=sha(source)
for name in ('feasible_path_solver.py','test_feasible_path_solver.py','manufactured_controls.log','run_feasible.py','freeze_protocol.py'):
 f=HERE/name;bindings[str(f)]=sha(f)
cases={}
for i in (20,28):
 f=source.parent/f'case_{i}.json';bindings[str(f)]=sha(f);cases[str(i)]={'path':str(f),'sha256':sha(f)}
review=Path('/private/tmp/hug_feasible_path_independent_review_2026-10-15_v1/preexecution_review.json')
assert review.exists(), 'Need independent preexecution receipt'
r=json.loads(review.read_text());assert r['approved_for_bounded_execution'] is True
for name in ('feasible_path_solver.py','run_feasible.py'):
 assert r['bindings'][str(HERE/name)]==sha(HERE/name)
bindings[str(review)]=sha(review)
for s,h in bindings.items():assert sha(s)==h,s
out={'schema':'hug_feasible_accepted_path_protocol_v1','execution_authorized':True,'case_indices':[20,28],'source_cases':cases,'bindings':bindings,'solver':DEFAULT_CONFIGURATION,'driver_start_binding_extra_exact_checks_per_case':1,'solver_budget_excludes_driver_start_binding':True,'no_new_eval_data':True,'no_new_native_grasps':True,'no_physics_steps':True,'scope':'Existing saved endpoint retargeting only. Literal original caps/domain/passed pooled rows/eligibility; exact accepted transitions improve current diagnostic rank; no full-field acceptance claim.','start_replay_requirement':'Fresh exact rank, literal predicates, orientation eligibility, authoritative original tip norms, caps and bounds reproduce accepted segment source.'}
target=HERE/'protocol.json';assert not target.exists();target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'protocol':str(target),'sha256':sha(target),'bindings':len(bindings)}))
