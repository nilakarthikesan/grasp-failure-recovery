"""Create the reviewed operational protocol BEFORE real candidate fitting."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import ast,hashlib,json
from constrained_refinement import ConstrainedRefinementConfig
from grasp_failure_prediction.evaluation.surface_refinement import RefinementConfig

OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
HELPER=Path('/private/tmp/hug_constrained_validation_2026-10-14/exact_direction.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def main():
    if (OUT/'protocol.json').exists():raise FileExistsError('Frozen protocol must not be overwritten')
    old=read(ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/protocol.json')
    fullroot=Path('/private/tmp/hug_sampling_checks_2026-10-14/ephemeral')
    full={str(p):sha(p) for p in fullroot.glob('*density2mm_full_current.npz')};assert len(full)==6
    assert sha(HELPER)=='c20380677f727206830e3e87beebd91948009ffb83dc341c53dad2fc5df9baae'
    assert sha(OUT/'constrained_refinement.py')=='0b29f867337cd11bb9128f5aa6ca92800338543055b286b40873d10f4d74f409'
    protected=read(OUT/'protected_hashes.json')
    assert len(protected)==114 and sha(OUT/'protected_hashes.json')=='d1f97309f366962ee3b32c11d24cc576870899f37dbb8837080ea15fcbb9c530'
    for path,h in protected.items():assert sha(path)==h,path
    scripts=[OUT/'constrained_refinement.py',OUT/'run_constrained_refinement.py',OUT/'geometry_helpers.py',OUT/'merge_cohorts.py',HELPER]
    for f in scripts:ast.parse(f.read_text())
    sourcepaths=[ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json',
        ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/static_saved_hug_fk.json',
        ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces/audit_other_surfaces.py',
        ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/check_orientation.py',
        ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/protocol.json',
        ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/acceptance/acceptance_rule.json',
        ROOT/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf',
        Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug/assets/mano_models/models/MANO_RIGHT.pkl')]
    p={
        'schema':'hug_global_constrained_surface_protocol_v1',
        'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'before_first_real_candidate_metrics':True,'root_review_required_before_execution':True,
        'scope':'Six original saved seed-0 HUG grasps only, two independent fixed three-case cohorts, all 24 joints jointly from original q. No new inference, eval episodes, physics, task labels or training.',
        'cases':[{**c,'densities_mm':[2]} for c in old['cases']],
        'allowed_cohorts':[[8,12,16],[20,24,28]],
        'cohort_filename_policy':'candidate_cohort_8_12_16.json and candidate_cohort_20_24_28.json with matching noncolliding partial files. Merge binds complete six unique original hashes and common protocol/source/scripts.',
        'solver_configuration':asdict(ConstrainedRefinementConfig()),
        'surface_configuration':asdict(RefinementConfig()),
        'direction_configuration':{
            'bins_per_part':64,'score_weight':1.0,'invalid_penalty':2.0,
            'fixed_mass_per_part':1/6,'score_endpoint_arithmetic_allowance':1e-12,
            'native_and_baseline_eligibility':'Fixed original 5 mm common-same-label exterior, unique normal/point/label, unburied, both closest points outside actual visible object under literal old normal protocol. No eligible sites means unavailable part, with no mass redistribution.',
            'candidate_eligibility':'Reevaluate actual nearest full-hand part/normal/ties/inside/buried/miss/closest-visual membership at fixed reference bins. Invalid site retains nonzero residual 2.0 instead of being dropped.'},
        'query_cache_root':old['original_query_cache_root'],
        'query_cache_sha256':{p:h for p,h in old['query_cache_sha256'].items() if 'density2' in p},
        'original_full_distance_root':str(fullroot),'full_cache_sha256':full,
        'direction_helper_path':str(HELPER),'implementation_sha256':{str(f):sha(f) for f in scripts},
        'source_sha256':{str(f):sha(f) for f in sourcepaths},
        'protected_snapshot_sha256':sha(OUT/'protected_hashes.json'),
        'protected_source_count':114,'source_branch_commit':'d577818',
        'strict_acceptance_rule_sha256':'87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4',
        'compound_objective':'Fixed original six-part near/broad quadrature, native unsigned distances and closest-point vectors with the prior settings; each part mass 1/6. Actual native-relative facing-score residual uses normalized per-part weights and fixed 1/sqrt(6) scale; invalid residual is 2.0 before weighting, fixed length, native-away scores retained. Original all-ten anchor residual 0.25/10mm/sqrt(30); original full-24 joint regularizer 0.05/0.35rad/sqrt(24). Concatenate distance, closest-vector, facing, anchor and joint blocks. Equivalent soft-L1 cost with f_scale 0.2.',
        'constraint_policy':'Internal SLSQP: every tip squared error is at most its ORIGINAL squared error; mean error at most .010 m. Final per-tip norm allows only 10 nm explicit arithmetic error, while computed mean stays strictly at most .010 m. Actual source-URDF bounds intersect original q +/- .35 rad. No clipping or replacing constraints with a loss.',
        'candidate_selection':'Lowest frozen-objective FEASIBLE SLSQP major iterate or raw-final; finite-difference and line-search probes cannot be selected. Shared pre-metric policy. If none improves original by 1e-12, retain EXACT original q/root. Record each major feasibility/cost, raw-final refusal, selected iteration and convergence/budget separately.',
        'palm_anchor':'Native wrist equals Shadow palm POSITION. Original carrier rotation and world-operator/base q-dependent palm-rotation convention maintained; carrier translation compensates ONLY changed wrist-to-palm offset. Literal original root when q is original. No object/hand scaling or arbitrary shift.',
        'cache_policy':'Fresh solver/model data per case. Full residual LRU 64 and FK-tip LRU 64; original/overlap full-versus-cheap tip components agree within 1e-12 m. Constraint derivatives use FK-only callback. Final state/surface/direction/FK fields are recomputed freshly OUTSIDE solver cache.',
        'numba_threads_per_process':1,'progress_full_evaluation_interval':25,
        'guarded_APIs':['mj_step','mj_step1','mj_step2'],
        'guards_active_after_all_imports_and_end_cohort':True,
        'settings_identical_across_cases':True,
        'limitations':[
            'Local nonsmooth constrained optimization is neither a global optimum nor proof of feasibility/infeasibility of strict morphology mapping.',
            'Actual mesh/analytic normals are fitted only at declared original paired-eligible bins. Other lost/new/unavailable/tied/penetrating regions remain separate required scientific audit evidence.',
            'Fixed part mass never redistributes missing normal evidence; absence cannot become PASS.',
            'Closest-part distances can include buried collider interfaces; normal eligibility excludes buried features. No exact globally accessible contact boundary proof.',
            'Conditional arithmetic assumptions and lack of exhaustive self-intersection proof remain; numerical constraint allowance is not a scientific mapping margin.',
            'Objective improvement or engineering passage cannot certify strict contact states, forces, failure labels or unseen objects.',
            'Six development poses from one MANO shape are not independent held-out validation.',
            'Original default mapper, predictor, manuscript and runtime assets remain unchanged. No new eval data authorized.']}
    (OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps({'protocol_sha256':sha(OUT/'protocol.json'),'implementation_sha256':p['implementation_sha256'],
        'protected_count':114,'solver_configuration':p['solver_configuration'],'direction_configuration':p['direction_configuration']}))

if __name__=='__main__':main()
