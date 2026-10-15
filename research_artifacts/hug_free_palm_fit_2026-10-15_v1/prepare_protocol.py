from pathlib import Path
import hashlib,json,ast,gzip
OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OLD=Path('/private/tmp/hug_guarded_geometry_2026-10-14_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((OLD/'protocol.json').read_text())
inherited=json.loads((OLD/'protocol.json').read_text())
for key in ('frozen_utc','root_review','manufactured_keyed_evidence_tests_passed','seed_bindings',
            'palm_anchor','maximum_full_surface_evaluations_per_case','scientific_kernel_AST_identical',
            'selection_policy','source_branch_commit','draft_utc','draft_review_ready_utc',
            'before_first_new_restored_seed_surface_metrics','old_canonical_fit_metrics_already_observed',
            'old_protocol_path','old_protocol_sha256','before_first_new_guarded_pattern_surface_metrics',
            'old_seed_stage_protocol_path','old_seed_stage_protocol_sha256'):
    p.pop(key,None)
p['inherited_protocol_history']={'path':str(OLD/'protocol.json'),'sha256':sha(OLD/'protocol.json'),
    'historical_source_commit':inherited['source_branch_commit'],
    'only_inherited_scientific_contents':'Surface objective/labels/sites/direction guard definitions and protected input closure; original method/domain/seed/budget statements superseded by this new version.'}
p['source_branch_commit']='114ce9b'
p['palm_anchor']='Original native wrist + per-grasp world translation. Actual floating-root quaternion fixed; forearm position compensates joint-dependent wrist-to-palm FK and requested palm displacement.'
p['maximum_full_surface_evaluations_per_case']=128
p['selection_policy']='26 declared translation seeds followed by at most4 deterministic normalized-state pattern sweeps,128fullcallback limit; exact original tipcaps/10mm mean and additional per-part native direction guard; fresh terminal checks.'
p['cache_policy']='Exact full27 normalized-state float64 bytes key; complete detached callback values/native-normal diagnostics recorded losslessly; duplicate full state must reproduce values exactly.'
p.update(schema='hug_free_palm_development_protocol_v1',method_version=1,method_stage='free_palm_normalized_pattern_v1',
    execution_authorized_by_root_review=False,draft_only_not_executed=True,
    direction_helper_path=str(OUT/'exact_direction.py'),protected_snapshot_sha256=sha(OUT/'protected_hashes.json'),
    state_coordinates={'dimension':27,'indices_0_23':'(actual q_rad - original q_rad)/0.025rad; original named order',
        'indices_24_26':'world XYZ palm displacement/0.005m','q0_state':[0.]*27,'joint_scale_rad':.025,'translation_scale_m':.005,
        'search_scale_not_acceptance_tolerance':True},
    placement='Preserve actual saved floating-forearm root quaternion exactly. Set joints and baseline forearm position, mj_forward, then compensate forearm position to native wrist + variable world displacement; mj_forward only. Native targets unmodified.',
    domain={'joints':'literal source URDF bounds intersect original q +/-0.35rad, normalized endpoints rounded inward only until decoded joints remain within literal bounds',
        'translation':'AABB enclosing intersection of five balls centered at ORIGINAL native world fingertip targets, radii source palm-to-tip chain translation-length sums + ORIGINAL exact caps +1um outer computational padding',
        'translation_padding_is_only_outer_search_domain_not_acceptance':True,'new_root_orientation':False},
    predeclared_seeds={'count':26,'order':'18 seeds: distances2.5,5,10mm; XYZ axis; signs+,-; then8 length5mm cube diagonals with XYZ signs+,- lexicographically; all original joints, clamp normalized full state to frozen derived domain',
        'units':'search hyperparameters, not measured witness gaps or acceptance thresholds','seeds_frozen_before_first_case_surface_callback':True},
    analytic_jacobian='mj_jac at each actual URDF virtual fingertip world point on its parent distal body MINUS mj_jac at actual palm origin, named joint dof columns*.025rad; append .005m identity world displacement. Norm Jacobian uses error dot vectorJ/norm; zero norm rows zero. All declared gates must pass before search.',
    analytic_jacobian_verification={'step':1e-4,'tolerance_m_per_state':1e-8},
    prior_original_cap_bindings={},
    actual_geometry_measured_before_freeze=False,
    trace_policy='Every full callback complete residual/native-normal diagnostics saved losslessly to immutable deterministic gzip; exact27-state metadata cache retains values. Cohort JSON records every proposal/refusal/event; merger writes compact selected poses and lossless cohort gzip.',
    literal_scientific_acceptance_unchanged=True,
    direction_guard_role='Additional conservative development selector requiring each available fixed native direction component not worsen and original valid bins remain valid; not the scientific acceptance rule.',
    limitations=['Six previously inspected development grasps only. Finite local search cannot prove global infeasibility or convergence.',
        'Source URDF reach domain is a conservative necessary outer box, not a guarantee every anchor is fingertip feasible.',
        'Same literal part-labelled collider surfaces, skin-weight correspondence, fixed quadrature and native-away targets as prior frozen study.',
        'All old original-native caps and10mm mean unchanged. Passing these engineering checks alone is not scientific mapping acceptance.',
        'Independent full proximity/orientation audits required. No inferred outcomes, lift success, inference, physics integration, eval episode or predictor training.'])
p['solver_configuration'].update(initial_pattern_radius_scaled=1.,minimum_pattern_radius_scaled=.125,
    maximum_pattern_sweeps=4,maximum_full_geometry_evaluations=128,maximum_FK_evaluations=6000,
    maximum_restorations=256,restoration_maxiter=40)
p['solver_configuration'].pop('initial_pattern_radius_rad',None);p['solver_configuration'].pop('minimum_pattern_radius_rad',None)
p['solver_configuration'].pop('joint_delta_limit_rad',None)
p['solver_configuration'].pop('finite_difference_step_rad',None)
p['optimization_order']='All26 translation seeds then up to4 pattern sweeps; common tip-inward minimum-norm direction, translation XYZ +/-,24 jointcoords +/-, SVD norm-tangentbasis. First accepted cost improvement ends sweep, otherwise radius halves.'
p['callback_budgets']='Per case max128full callbacks incl terminal reserves,6000cheap-tip callbacks incl terminal reserves,6000analyticJac callbacks,256restorations. Pre-solver bounded finite-difference gate and final source/FK verification outside solver counted separately. Each staging call performs exactly2mj_forward; no mj_step.'
caproot=ROOT/'research_artifacts/hug_guarded_geometry_validation_2026-10-14_v1'
for path in sorted(caproot.glob('*_validation.json.gz')):
    d=json.loads(gzip.decompress(path.read_bytes()))
    if 'source_inventory_index' not in d:continue
    index=d['source_inventory_index'];p['prior_original_cap_bindings'][str(index)]={'path':str(path),'sha256':sha(path)}
    p['source_sha256'][str(path)]=sha(path)
assert len(p['prior_original_cap_bindings'])==6
names=('run_free_palm.py','free_palm_solver.py','free_palm_kinematics.py','keyed_direction_evidence.py','surface_component_guard.py','geometry_helpers.py','exact_direction.py','merge_free_palm.py','test_free_palm.py','test_keyed_direction_evidence.py')
p['implementation_sha256']={str(OUT/name):sha(OUT/name) for name in names}
p['manufactured_controls']={'count':26,'result':'PASS','log_path':str(OUT/'manufactured_tests.log'),'log_sha256':sha(OUT/'manufactured_tests.log'),'only_manufactured_inputs':True}
for name in ('geometry_helpers.py','exact_direction.py','surface_component_guard.py'):assert sha(OUT/name)==sha(OLD/name)
for path,h in p['source_sha256'].items():assert sha(path)==h,path
(OUT/'protocol.draft.json').write_text(json.dumps(p,indent=2,allow_nan=False)+'\n')
print(json.dumps({'draft_protocol_sha256':sha(OUT/'protocol.draft.json'),'implementation_files':len(names),'original_cap_bindings':6,'manufactured_tests':26}))
