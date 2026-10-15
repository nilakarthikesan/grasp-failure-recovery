"""Build an unexecuted draft, binding the exact existing witness/start inputs."""
from pathlib import Path
import hashlib,json
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OUT=Path(__file__).parent;V1=Path('/private/tmp/hug_free_palm_fit_2026-10-15_v1')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=read(V1/'protocol.json');saved_path=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json');saved=read(saved_path)
initial=Path('/private/tmp/hug_free_palm_review_2026-10-15_v1/diagnostic_selection/candidate_targets.json')
explicit=Path('/private/tmp/hug_explicit_palm_selection_2026-10-15_v1/candidate_targets.json')
targets={c['source_inventory_index']:(c,initial) for c in read(initial)['cases']}
targets.update({c['source_inventory_index']:(c,explicit) for c in read(explicit)['cases']})
sources=dict(old['source_sha256']);sources.update(old['implementation_sha256']);sources.update(saved['input_sha256'])
sources.update({str(p):sha(p) for p in (saved_path,initial,explicit,V1/'protocol.json',V1/'candidate_cohort_8_16_24.json',V1/'candidate_cohort_12_20_28.json',Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/cached_failure_row_counts.json'))})
cases={}
for i in (8,12,16,20,24,28):
    c,path=targets[i]
    cases[str(i)]={'native_sha256':c['native_sha256'],'proximity_row_count':sum(r['source_inventory_index']==i for r in saved['whole_cell_rows']),
        'orientation_row_count':sum(r['source_inventory_index']==i for r in saved['orientation_rows']),
        'frozen_starts':[{'label':'original_saved_pose','state_scaled':[0.]*27},
            {'label':'prior_fixed_deliberate_palm_diagnostic','state_scaled':c['selected_state_scaled'],'source_candidate_path':str(path),'source_candidate_sha256':sha(path),'source_selection':c['selection_source']}]}
files=['constraint_core.py','constraint_engine.py','constraint_solver.py','run_constraints.py','free_palm_kinematics.py','exact_direction.py','geometry_helpers.py','surface_component_guard.py','test_constraint_solver.py','source_preflight.py']
protocol={'schema':'hug_saved_surface_witness_constraint_protocol_v1','branch':'codex/project-work','source_branch_commit':'90e27e0',
    'execution_authorized_after_independent_review':False,'candidate_schema':'hug_saved_surface_witness_constraint_candidates_v1',
    'cases':cases,'allowed_cohorts':[[8,16,24],[12,20,28]],'input_sha256':sources,
    'implementation_sha256':{str(OUT/f):sha(OUT/f) for f in files},
    'preflight_sha256':{str(p):sha(p) for p in [Path('/private/tmp/hug_surface_constraints_root_2026-10-15_v1/all_six_source_preflight.json'),
        Path('/private/tmp/hug_surface_constraints_review_2026-10-15_v1/manufactured_constraint_controls.json'),OUT/'solver_tests_v2.log']},
    'source_preflight_path':'/private/tmp/hug_surface_constraints_root_2026-10-15_v1/all_six_source_preflight.json',
    'preflight_control_counts':{'independent_core_controls':36,'solver_controls':8,'source_cases':6,'source_analytic_jacobian_cases':6},
    'solver_configuration':{'maximum_iterations_per_start':40,'maximum_raw_callback_executions_per_case':3000,
        'maximum_exact_checks_per_case':90,'terminal_raw_and_exact_reserve':2,'surface_difference_step_scaled':1e-4,
        'raw_cache_size':128,'epigraph_slack_upper':1000.,'tip_inward_target_m':1e-9,'optimizer_ftol':1e-9},
    'state':{'order':'24 original-relative joint deltas/0.025rad, then3 world palm displacements/0.005m',
        'joints':'original q±0.35 intersect source physical limits; inward representable endpoints unchanged',
        'translation_domain':'prior source-chain reach plus ORIGINAL native tip caps conservativeAABB;1um outer pad only',
        'root':'original floating forearm quaternion fixed; q-dependent palm compensation plus world translation; wrist joints remain variables'},
    'constraints':{'proximity':'All178 saved capped original definite mismatch cells; literal error=radius+old_eps,lower=max(0,d-error),upper=d+error. Native-definite-near requires mappedupper<=band; native-definite-far requires mappedlower>band. These sufficient witness constraints exceed merely removing definite failure.',
        'orientation':'All134 saved reliable signed-direction rows, including85 native-away and49 native-facing; preserve sourceband, same nearest anatomicalpart, exterior query, unique feature, unburied closestpoint, and closestpoint outside literal visual object. Exact inherited eligibility gates unchanged.',
        'optimizer_surrogate':'Epigraph minimum worst normalized constraint violation; proximity distances scaled1mm, sign dimensionless. Raw expected-part normals/source-band/querydistance plus nearest-part gap and whole-hand query-exterior margins. Complete tie/burial/visual eligibility checked at every tip/domain-feasible major iterate and terminal, not differentiated in inner search.',
        'inward_targets':'1nm stricter distance/tip/mean targets and1e-8 stricter signed-score target guide optimization only; literal final predicates have no added allowance.',
        'tip_gates':'Each exact sqrt(einsum) norm<=historical original cap; original mean<=0.010m. No rebase.',
        'duplicates':'All source rows retained with source pointers. Duplicate inequalities are repeated without summed weighting; max violation and feasibility unchanged. Refusal counts include repeated rows, not independent sites.',
        'counts':'312 retained source witness rows=178 proximity+134 orientation.848 raw epigraph inequalities=178 proximity+3*134 orientation conditions+2*134 nearestpart/exterior guidance.6 additional exact-tip/mean optimization inequalities percase are outside epigraph slack.',
        'missing_orientation':'Case24 has zero saved reliable orientation witnesses; UNAVAILABLE, never an orientation PASS.'},
    'selection':'Only exact checked original/start/major/terminal states satisfying originaltip/mean/domain are eligible. Lexicographic:all literal witness rows feasible first; otherwise invalid orientation-row count, maximum exact normalized violation without optimizerbuffers, failed-row count, normalized pose squared distance; stable earliest tie. No raw probe selection. All states nonproduction; allsurfacePASS still requires full independent audit.',
    'budget_semantics':'3000 actual raw geometry callback executions including raw work inside exact checks, cachehits not charged.90exactchecks.2raw+exact calls reserved for fresh final and possible originalfallback. No extension, resampling, or additionalstarts if budget reached. AnalyticJacobian/FK staging counted separately.',
    'concurrency':{'maximum_processes':2,'numba_threads_per_process':2,'numba_pool_capacity':4},
    'scientific_acceptance_rule_sha256':old['strict_acceptance_rule_sha256'],
    'coverage':'178 retained capped examples out of44,558 original definite part/band/direction/cell rows;44,380 not constrained.134 retained orientation examples also capped/repeated. This is development repair, not exhaustive region coverage or held-out validation.',
    'independent_audit':'Unchanged6coarse+3fine comparisons on all original gridcells; no inference/physics/eval/training. Nativeambiguouscell intervals and continuous orientation coverage can remain UNRESOLVED even after all fitted rows pass.',
    'new_eval_data':False,'new_inference':False,'physics_steps':0,'production_promotion':False}
target=OUT/'protocol.draft.json';target.write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps({'draft':str(target),'sha256':sha(target),'input_count':len(sources)}))
