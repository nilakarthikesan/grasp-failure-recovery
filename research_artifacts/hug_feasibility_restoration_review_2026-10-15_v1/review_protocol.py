"""Pre-execution binding/static review; no real FK, geometry or projection."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np

OUT=Path(__file__).resolve().parent
PRIMARY=Path('/private/tmp/hug_feasibility_restoration_2026-10-15_v1')
V2=Path('/private/tmp/hug_surface_constraints_2026-10-15_v2')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
draft=PRIMARY/'protocol.draft.json';p=json.loads(draft.read_text())
proof=json.loads((OUT/'source_endpoint_review.json').read_text())
controls=json.loads((OUT/'independent_core_controls.json').read_text())
pool=json.loads(Path(p['pooled_candidate_path']).read_text())
source=json.loads((V2/'candidate_targets.json').read_text())
checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)
check('not_authorized_or_executed',not p['execution_authorized_after_independent_review'] and not list(PRIMARY.glob('case_*_endpoint_*.json')))
for section in ('input_sha256','implementation_sha256','preflight_sha256'):
    for path,h in p[section].items():check(f'{section}:{path}',sha(path)==h)
check('two_frozen_cohorts',p['allowed_cohorts']==[[8,16,24],[12,20,28]])
expected=dict(trust_half_width_scaled=.25,maximum_FK_executions_per_endpoint=400,maximum_analytic_jacobians_per_endpoint=60,maximum_iterations_per_endpoint=40,inward_target_m=1e-9,optimizer_ftol=1e-12,cache_size=128)
check('exact_fixed_configuration',p['projection_configuration']==expected)
check('controls_use_final_core_hash',controls['passed'] and controls['checks_passed']==18 and controls['source_bindings'][str(PRIMARY/'restoration_core.py')]==sha(PRIMARY/'restoration_core.py'))
check('source_proof_pass',proof['passed'] and proof['checks_passed']==116)
for name in ('constraint_core.py','constraint_engine.py','free_palm_kinematics.py','exact_direction.py','geometry_helpers.py','surface_component_guard.py'):
    check('unchanged_scientific_and_kinematic_source:'+name,sha(PRIMARY/name)==sha(V2/name))
for case in pool['cases']:
    idx=case['source_inventory_index'];cp=p['cases'][str(idx)];a=case['baseline_constraint_diagnostics'];b=case['selected_constraint_diagnostics']
    check(f'{idx}:native_identity',case['native_sha256']==cp['native_sha256'])
    check(f'{idx}:unchanged_caps',case['authoritative_original_tip_caps_sqrt_einsum_m']==cp['original_tip_caps_m'])
    check(f'{idx}:reference_state_and_rank',case['selected_state_scaled']==cp['reference_state_scaled'] and b['rank']==cp['reference_exact_rank'])
    for tag,diag in [('original',a),('reference',b)]:
        check(f'{idx}:{tag}_orientation_pass',diag['exact_orientation']['passed']==cp[tag+'_orientation_pass'])
        check(f'{idx}:{tag}_orientation_valid',diag['exact_orientation_fields']['valid']==cp[tag+'_orientation_valid'])
    for tag,group,field in [('valid','exact_orientation_fields','valid'),('pass','exact_orientation','passed')]:
        expected_mask=np.asarray(a[group][field],bool)|np.asarray(b[group][field],bool)
        check(f'{idx}:protected_union_{tag}',expected_mask.tolist()==cp['protected_orientation_'+tag])
    check(f'{idx}:two_fixed_endpoints',[x['start_index'] for x in cp['endpoints']]==[0,1])
    old=next(x for x in source['cases'] if x['source_inventory_index']==idx)
    for endpoint in cp['endpoints']:
        start=endpoint['start_index'];ind=next(x for x in proof['endpoints'] if x['source_inventory_index']==idx and x['start_index']==start)
        check(f'{idx}:{start}:exact_state_and_event',endpoint['state_scaled']==ind['source_state_scaled'] and endpoint['state_float64_sha256']==ind['source_state_bytes_sha256'] and endpoint['source_terminal_trace_event']==ind['source_trace_event'])
        check(f'{idx}:{start}:complete_terminal',endpoint['source_terminal_record']==old['solver_audit']['optimizer_starts'][start])
        check(f'{idx}:{start}:trace_binding',endpoint['source_trace_path']==ind['source_trace'] and endpoint['source_trace_sha256']==proof['source_bindings'][ind['source_trace']])
        e=np.array(endpoint['source_tip_error_vectors_m']);norm=np.sqrt(np.einsum('ij,ij->i',e,e));caps=np.array(cp['original_tip_caps_m'])
        check(f'{idx}:{start}:source_norms',norm.tolist()==endpoint['source_tip_status']['norms_m'])
        check(f'{idx}:{start}:original_terminal_cap_refusal',not np.all(norm<=caps) and norm.mean()<=.010)
core=ast.parse((PRIMARY/'restoration_core.py').read_text())
imports=[]
for n in ast.walk(core):
    if isinstance(n,ast.Import):imports.extend(x.name for x in n.names)
    if isinstance(n,ast.ImportFrom):imports.append(n.module)
check('projection_core_no_geometry_or_runtime_imports',set(imports)=={'hashlib','collections','numpy','scipy.optimize'})
check('fixed_historical_acceptance_bound',any(path.endswith('/acceptance_rule.json') and h=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4' for path,h in p['input_sha256'].items()))
for field in ('new_eval_data','new_inference','production_promotion'):check('forbidden_action:'+field,p[field] is False)
check('zero_physics',p['physics_steps']==0)
source_gate=V2/'all_six_source_preflight.json'
prerequisites={str(x):sha(x) for x in [OUT/'source_endpoint_review.json',OUT/'independent_core_controls.json',OUT/'primary_manufactured_controls_rerun.log',source_gate]}
report=dict(passed=True,protocol_draft_sha256=sha(draft),checks_passed=len(checks),checks=checks,
            source_bindings={str(draft):sha(draft),str(Path(__file__)):sha(Path(__file__)),**p['input_sha256'],**p['implementation_sha256'],**p['preflight_sha256'],**prerequisites},
            prerequisites_sha256=prerequisites,
            manufactured_controls={'independent':18,'primary_independently_rerun':14,'source_endpoint_identity_checks':116},
            static_findings=['Complete27 keys and literal historical individual caps with inclusive10mm mean, no tolerance/rebase; inward1nm is optimizer guidance only.',
                             'Exactly12 prior V2 terminals; .25 scaled per-coordinate box intersects original domain; nearest observed eligible checkpoint, not global projection.',
                             'Only endpoint/major/terminal checkpoints eligible; raw probes retained but never selected; selection uses only endpoint distance before any surface query.',
                             '400 tipFK plus60 analytic staging callbacks each invoke two mj_forward; maximum920 projection forwards per endpoint. Reserved freshFK included; scene/source/exact stages counted separately.',
                             'Driver asserts raw/exact surface counters unchanged during each projection. Up to one retained exact check per chosen projection plus reference/final/fallback checks, max5/case explicitly declared.',
                             'Strict exact-rank improvement plus original-or-pooled passed AND eligible row preservation; protects native-away direction via unchanged literal engine.',
                             'Fresh final gate/rank equality; otherwise explicit pooled fallback. No coarse audit unless eligible, no production promotion.',
                             'Geometry,Jacobian,staging and exact-rank engine byte-identical to already independently validated V2; post-import Numba2 asserted in driver.'],
            limitations=['Pre-execution review does not certify future numerical output or mapped-surface acceptance.','Protected eligible-row noninvalidation is extra conservative selection, not the frozen scientific criterion.','Inherited analytic-Jacobian manufactured/source gates are reused; this review performs no real FK or new geometry.'],
            no_real_projection_executed_by_reviewer=True,no_new_eval_data=True)
target=OUT/'preflight_review.json';target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),protocol_draft_sha256=sha(draft),review_sha256=sha(target))))
