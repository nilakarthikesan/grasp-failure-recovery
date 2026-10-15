"""Read saved source bytes and freeze all twelve endpoints; no scene/FK imports."""
from pathlib import Path
import gzip,hashlib,json
import numpy as np
OUT=Path(__file__).parent;V2=Path('/private/tmp/hug_surface_constraints_2026-10-15_v2')
POOL=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/pooled_selection/candidate_targets.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
prior=read(V2/'protocol.json');endpoints=read(V2/'candidate_targets.json');refs={c['source_inventory_index']:c for c in read(POOL)['cases']}
inputs=dict(prior['input_sha256']);inputs.update(prior['implementation_sha256'])
extra=[V2/'protocol.json',V2/'candidate_targets.json',POOL,Path('/private/tmp/hug_surface_constraints_review_2026-10-15_v2/final_output_review.json'),Path('/private/tmp/hug_surface_constraints_review_2026-10-15_v2/pooled_selection_review.json')]
inputs.update({str(p):sha(p) for p in extra});cases={}
for case in endpoints['cases']:
 i=case['source_inventory_index'];ref=refs[i];trace=Path(case['solver_audit']['trace_path']);assert sha(trace)==case['solver_audit']['trace_sha256'];inputs[str(trace)]=sha(trace)
 trace_terms={};raw={}
 for line in gzip.open(trace,'rt'):
  event=json.loads(line)
  if event['kind']=='optimizer_terminal':trace_terms[event['start_index']]=event
  if event['kind'] in ('raw_callback','exact_checkpoint','selected_fresh_terminal'):
   v=event['value'];raw[np.asarray(v['state_scaled'],np.float64).tobytes()]=(event['event'],v)
 frozen=[]
 for terminal in case['solver_audit']['optimizer_starts']:
  assert terminal['status']==9 and terminal['nit']==40
  start=terminal['start_index'];t=trace_terms[start];x=np.asarray(terminal['terminal_state_scaled'],np.float64);bounds=np.asarray(ref['bounds_scaled'])
  assert np.array_equal(x,np.asarray(t['terminal_state_scaled'])) and np.isfinite(x).all() and np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1])
  ev,v=raw[x.tobytes()];assert not v['tip_status']['passed']
  frozen.append({'start_index':start,'state_scaled':x.tolist(),'state_float64_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'source_terminal_record':terminal,'source_terminal_trace_event':t['event'],'source_FK_value_trace_event':ev,'source_trace_path':str(trace),'source_trace_sha256':sha(trace),'source_tip_status':v['tip_status'],'source_tip_error_vectors_m':v['task_error_vectors_m'][:5]})
 assert len(frozen)==2 and [r['start_index'] for r in frozen]==[0,1]
 b=ref['baseline_constraint_diagnostics'];s=ref['selected_constraint_diagnostics']
 cases[str(i)]={'native_sha256':ref['native_sha256'],'original_tip_caps_m':ref['authoritative_original_tip_caps_sqrt_einsum_m'],'reference_state_scaled':ref['selected_state_scaled'],'reference_exact_rank':s['rank'],'reference_orientation_pass':s['exact_orientation']['passed'],'reference_orientation_valid':s['exact_orientation_fields']['valid'],'original_orientation_pass':b['exact_orientation']['passed'],'original_orientation_valid':b['exact_orientation_fields']['valid'],'protected_orientation_pass':np.logical_or(b['exact_orientation']['passed'],s['exact_orientation']['passed']).tolist(),'protected_orientation_valid':np.logical_or(b['exact_orientation_fields']['valid'],s['exact_orientation_fields']['valid']).tolist(),'endpoints':frozen}
files=['restoration_core.py','run_restoration.py','constraint_core.py','constraint_engine.py','free_palm_kinematics.py','exact_direction.py','geometry_helpers.py','surface_component_guard.py','test_restoration_core.py']
p={'schema':'hug_saved_endpoint_feasibility_restoration_protocol_v1','candidate_schema':'hug_saved_endpoint_feasibility_restoration_candidates_v1','execution_authorized_after_independent_review':False,'cases':cases,'allowed_cohorts':[[8,16,24],[12,20,28]],'pooled_candidate_path':str(POOL),'input_sha256':inputs,'implementation_sha256':{str(OUT/f):sha(OUT/f) for f in files},'preflight_sha256':{str(OUT/'manufactured_tests_v2.log'):sha(OUT/'manufactured_tests_v2.log')},'projection_configuration':{'trust_half_width_scaled':.25,'maximum_FK_executions_per_endpoint':400,'maximum_analytic_jacobians_per_endpoint':60,'maximum_iterations_per_endpoint':40,'inward_target_m':1e-9,'optimizer_ftol':1e-12,'cache_size':128},
 'source_selection':'Exactly twelve saved V2 terminal states, both fixed starts for each existing native grasp. No clipping, new seeds, geometry-ranked endpoint selection or additional endpoints. Full source terminal and trace/FK provenance retained.',
 'objective':'One FK-only SLSQP nearest-endpoint restoration minimizing0.5*sum((x-endpoint)^2). This differs from previous surface minimax search. Choose closest among observed exact-tip/domain/trust-feasible endpoint/major/terminal checkpoints only, stable earliest tie; no global nearest-projection guarantee. Raw probes cannot be selected. No surface query during restoration.',
 'trust_domain':'Box half-width0.25 in each normalized coordinate, intersect original frozen bounds. Each joint changes by at most0.00625rad and each world palm coordinate by at most0.00125m from saved endpoint; not an overall Euclidean1.25mm sphere. Original physical∩q0±.35 and conservative source-chain translationAABB stay fixed; native targets/root quaternion fixed.',
 'tip_constraints':'Original literal sqrt(einsum) individual errors<=historical caps and mean<=0.010m; no tolerance or rebase. The1nm inward target guides optimizer only, never an acceptance allowance. Complete27 key, zero norm Jacobian convention0, fresh selectedFK outsidecache.',
 'budget_accounting':'Per endpoint400 actual tipFK callback executions (includes one reserved fresh call; cachehits excluded),60 analytic Jacobian callbacks and40 optimizeriterations. Each tipFK and each analytic Jacobian callback stages two mj_forward calls, hence at most920 projection mj_forward calls/endpoint; driver explicitly counts/asserts this. Jacobian FK is additional, not hidden inside400. Scene initialization, original source check, exact surface checks and independent selected-source FK are separately counted. No extensions/backtracking searches/new starts.',
 'surface_checks':'At most one exact retained-surface evaluation for the chosen feasible projection checkpoint per endpoint. Per case one fresh pooled-reference exact check, up to two restored checks, one selected fresh check and at most one original pooled fallback check: maximum5. All312 retained source rows remain. No full-grid geometry during fitting.',
 'selection':'Improvement only if literal original tip/mean/domain pass, exact five-component historical diagnostic_rank strictly improves over pooled reference, all orientation rows passed by either original OR pooled pose still pass exact source-band/sign/eligibility, and all rows valid in either reference remain valid. The extra noninvalidation gate is conservative and not the scientific full-surface criterion. Stable endpoint-order tie. Full fresh chosen check must preserve exact rank and gates; otherwise retained pooled fallback. No productionpromotion.',
 'coarse_audit_policy':'Only genuinely improved eligible candidates proceed to unchanged coarse-grid native-relative surface audit after independent source/FK/cap/trust review. Any definite coarse mismatch is a failure and ends that candidate audit. Finer grids/coverage are considered only without a coarse counterexample; do not automatically rerun allnine. Historical surface criteria unchanged.',
 'scientific_scope':'Historical strict native-relative Shadow surface-equivalence and original-cap nonworsening are project-designed gates, not published HUG thresholds. Static restoration results cannot establish native HUG lift success, graspquality, outcome preservation or predictor generalization. Same six development grasps only.',
 'concurrency':{'maximum_processes':2,'numba_threads_per_process':2,'numba_pool_capacity':4,'thread_mask_set_and_asserted_after_all_frozen_imports':True},'new_eval_data':False,'new_inference':False,'physics_steps':0,'production_promotion':False}
target=OUT/'protocol.draft.json';assert not target.exists();target.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'draft':str(target),'sha256':sha(target),'fixed_endpoints':sum(len(c['endpoints']) for c in cases.values()),'source_files':len(inputs)}))
