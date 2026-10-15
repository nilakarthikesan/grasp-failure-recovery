from pathlib import Path
import json,hashlib,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/private/tmp/hug_se3_protected_fit_2026-10-15_v1')
from feasible_eligibility_solver import DEFAULT_CONFIGURATION
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=Path('/private/tmp/hug_feasible_path_2026-10-15_v1/protocol.json');p=json.loads(source.read_text());bindings=dict(p['bindings']);bindings[str(source)]=sha(source)
for name in ('eligibility_case.py','extended_policy.py','feasible_eligibility_solver.py','run_eligibility.py','freeze_protocol.py'):
 f=HERE/name;bindings[str(f)]=sha(f)
cases={};refs={}
for i in (20,28):
 f=source.parent/'pilot'/str(i)/'result.json';bindings[str(f)]=sha(f);cases[str(i)]={'path':str(f),'sha256':sha(f)}
 r=Path('/private/tmp/hug_se3_segment_check_2026-10-15_v1')/f'case_{i}.json';refs[str(i)]={'path':str(r),'sha256':sha(r)}
review=Path('/private/tmp/hug_eligibility_guidance_review_2026-10-15_v1/preexecution_review.json')
assert review.exists(),'Need independent review';r=json.loads(review.read_text());assert r['approved_for_bounded_execution'] is True
for f,h in r['bindings'].items():assert sha(f)==h;bindings[f]=h
bindings[str(review)]=sha(review)
for s,h in bindings.items():assert sha(s)==h,s
out={'schema':'hug_feasible_eligibility_guidance_protocol_v1','execution_authorized':True,'case_indices':[20,28],'source_cases':cases,'pooled_reference_cases':refs,'bindings':bindings,'solver':DEFAULT_CONFIGURATION,'driver_binding_exact_checks_per_case':2,'solver_budgets_exclude_driver_binding_checks':True,'scope':'Add continuous actual global closest-point visual-exterior and nested mesh facet-tie margins to fixedpooledeligible hard search rows. Original exact scientific predicates, tipcaps, originaldomain, protectedrows, rank unchanged. Failed startpreflight saves refusal and executes no solve.','facet_tie_logic':'min_g max(d_g-(D+tie), min_conflicting_triangles(d_triangle-(d_g+tie))); own pergeometry closest point/normal as originalexacttie test.','finite_positive_gap_ceiling_m':1.0,'ceiling_is_search_margin_only':True,'no_new_eval_data':True,'no_new_native_grasps':True,'no_physics_steps':True}
target=HERE/'protocol.json';assert not target.exists();target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'protocol':str(target),'sha256':sha(target),'bindings':len(bindings)}))
