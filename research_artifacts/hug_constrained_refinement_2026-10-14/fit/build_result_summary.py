"""Summarize frozen canonical records without geometry, model, solver or physics calls."""
from pathlib import Path
import hashlib
import json

OUT = Path(__file__).parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    merged = read(OUT / 'candidate_targets.json')
    verification = read(OUT / 'verification.json')
    assert verification['candidate_sha256'] == sha(OUT / 'candidate_targets.json')
    assert merged['runtime_protocol_version'] == 2 and merged['runtime_amendment'] == 1
    assert merged['complete_original_six_case_coverage'] and len(merged['cases']) == 6
    cases = []
    for c in merged['cases']:
        a = c['solver_audit']
        base, raw, final = a['baseline'], a['raw_assessment'], a['final_assessment']
        cases.append({
            'source_inventory_index': c['source_inventory_index'],
            'source_case': c['source_case'], 'object_id': c['object_id'],
            'native_sha256': c['native_sha256'],
            'runtime_seconds': c['runtime_seconds'],
            'solver_status': a['solver_status'], 'solver_success': a['solver_success'],
            'solver_message': a['solver_message'], 'iterations': a['iterations'],
            'major_iteration_count': len(a['major_iteration_assessments']),
            'eligible_lower_objective_major_iteration_count': sum(
                row['eligible_lower_objective_candidate'] for row in a['major_iteration_assessments']),
            'function_evaluations': a['function_evaluations'],
            'full_surface_evaluations_including_fresh_final': c['full_callback_count_including_fresh_final'],
            'budget_exhausted': a['budget_exhausted'],
            'selected_candidate_source': a['selected_candidate_source'],
            'selected_major_iteration': a['selected_major_iteration'],
            'surface_candidate_returned': a['surface_candidate_returned'],
            'exact_original_fallback': a['exact_original_fallback'],
            'raw_final_refusal_reasons': a['raw_final_refusal_reasons'],
            'original_objective': base['objective'], 'raw_final_objective': raw['objective'],
            'selected_objective': final['objective'],
            'raw_final_objective_reduction_fraction': (base['objective'] - raw['objective']) / base['objective'],
            'original_tip_error_norm_m': base['tip_error_norm_m'],
            'raw_final_tip_error_norm_m': raw['tip_error_norm_m'],
            'selected_tip_error_norm_m': final['tip_error_norm_m'],
            'raw_final_tip_error_increase_m': raw['tip_error_increase_m'],
            'raw_final_individual_tip_cap_pass': raw['allowance_individual_tip_nonworsening_pass'],
            'original_mean_tip_error_m': base['mean_fingertip_error_m'],
            'raw_final_mean_tip_error_m': raw['mean_fingertip_error_m'],
            'selected_mean_tip_error_m': final['mean_fingertip_error_m'],
            'original_mean_middle_error_m': c['baseline_fresh_diagnostics']['mean_middle_error_m'],
            'selected_mean_middle_error_m': c['selected_fresh_diagnostics']['mean_middle_error_m'],
            'original_task_link_error_norm_m': c['baseline_fresh_diagnostics']['task_link_error_norm_m'],
            'selected_task_link_error_norm_m': c['selected_fresh_diagnostics']['task_link_error_norm_m'],
            'fixed_direction_bin_count': c['baseline_fresh_diagnostics']['normal_bin_count'],
            'selected_invalid_direction_bin_count': c['selected_fresh_diagnostics']['normal_invalid_bin_count'],
            'fresh_final_recomputation_outside_solver_cache': c['final_state_freshly_recomputed_outside_solver_cache'],
            'independent_fixed_URDF_vs_scene_max_link_error_m': c['independent_fixed_URDF_vs_scene_max_link_error_m'],
            'time_s': c['time_s'], 'scientific_mapping_acceptance_claim': False,
        })
    summary = {
        'schema': 'hug_global_constrained_canonical_fit_summary_v1',
        'canonical_runtime_protocol_version': 2, 'canonical_runtime_amendment': 1,
        'candidate_sha256': sha(OUT / 'candidate_targets.json'),
        'protocol_sha256': sha(OUT / 'protocol.json'),
        'verification_sha256': sha(OUT / 'verification.json'),
        'summary_script_sha256': sha(__file__),
        'source_order_of_tip_errors': ['thumb', 'index', 'middle', 'ring', 'little'],
        'selected_surface_candidate_count': sum(c['surface_candidate_returned'] for c in cases),
        'exact_original_fallback_count': sum(c['exact_original_fallback'] for c in cases),
        'solver_success_count': sum(c['solver_success'] for c in cases),
        'solver_budget_exhaustion_count': sum(c['budget_exhausted'] for c in cases),
        'all_original_six_cases_retained': True,
        'protected_source_count': merged['protected_source_count'],
        'protected_and_original_inputs_unchanged': merged['protected_hashes_unchanged'] and merged['original_inputs_and_implementations_unchanged'],
        'guards': merged['guards'], 'physics_steps': 0, 'new_eval_data': False,
        'new_generator_proposal': False, 'training': False, 'production_changes': False,
        'strict_acceptance_rule_sha256': merged['strict_acceptance_rule_unchanged_sha256'],
        'scientific_mapping_acceptance_claim': False,
        'independent_full_grid_audit_status': 'pending_separate_original_rule_audit',
        'interpretation': [
            'A lower internal native-relative surface objective alone does not authorize a correction.',
            'The fixed source-relative individual fingertip caps and the existing strict 10 mm mean guard determine engineering eligibility.',
            'An exact original fallback retains the existing mapping; it does not establish scientific contact or orientation fidelity.',
            'A refused local nonsmooth SLSQP result does not prove global infeasibility or morphological impossibility.',
        ],
        'historical_runtime_statuses': [
            {'version': 1, 'status': 'interrupted_no_complete_candidate', 'status_path': '/private/tmp/hug_constrained_refinement_2026-10-14/interrupted_status.json'},
            {'version': 2, 'status': 'six_import_failures_before_real_case_solves', 'status_path': '/private/tmp/hug_constrained_refinement_runtime_v2_2026-10-14/startup_abort_status.json'},
            {'version': 3, 'status': 'frozen_unexecuted_fitting_protocol', 'status_path': '/private/tmp/hug_constrained_refinement_runtime_v3_2026-10-14/UNEXECUTED_STATUS.json'},
        ],
        'cases': cases,
    }
    dest = OUT / 'fit_summary.json'
    if dest.exists():
        raise FileExistsError('Existing canonical summary must not be overwritten')
    dest.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = [
        '# Canonical global constrained HUG static correction — 2026-10-14', '',
        f"The frozen runtime V2.1 run completed all six existing cases. It returned {summary['selected_surface_candidate_count']} eligible corrections and retained the exact original mapping in {summary['exact_original_fallback_count']} cases. No new proposals, physics steps, evaluation data or training were produced. The production mapper remains unchanged.", '',
        f"The protocol is `{summary['protocol_sha256']}`; the complete candidate record is `{summary['candidate_sha256']}`. The JSON/hash/URDF verifier checked the source and input hashes, physical joint bounds, trust region, exact fallback identity, source-relative individual fingertip caps, strict existing 10 mm mean guard and fresh final diagnostics. These are engineering checks. The independent original-rule full-grid scientific audit is recorded separately; this report claims no mapping acceptance.", '',
        'All cases used the same fixed native-relative distance, closest-point-vector and actual facing-score targets; source-away scores remained the targets. Fixed unreliable candidate direction sites incurred a residual of 2.0. All 24 joints started from their original values, with physical limits intersected with a 0.35 rad trust region. SLSQP had 120 major iterations, fixed finite differences and the same soft-L1 objective. Only a feasible lower-objective major iterate or the raw final candidate was eligible; otherwise the exact original pose was returned.', '',
        'The palm position stayed at the native wrist position, with the declared q-dependent palm orientation and root compensation. There was no arbitrary hand/object shift or scaling. The final geometry and landmark diagnostics were recomputed outside solver caches.', '',
    ]
    for c in cases:
        lines.append(f"- Case {c['source_inventory_index']} ({c['object_id']}): status {c['solver_status']}, {c['iterations']} iterations, {c['major_iteration_count']} major callbacks, {c['eligible_lower_objective_major_iteration_count']} eligible improved major iterates. Original/raw/selected objective: {c['original_objective']:.6f} / {c['raw_final_objective']:.6f} / {c['selected_objective']:.6f}. Original/raw/selected mean fingertip error: {1000*c['original_mean_tip_error_m']:.6f} / {1000*c['raw_final_mean_tip_error_m']:.6f} / {1000*c['selected_mean_tip_error_m']:.6f} mm. Selection: {c['selected_candidate_source']}; raw refusal: {', '.join(c['raw_final_refusal_reasons']) or 'none'}. Runtime: {c['runtime_seconds']/60:.2f} minutes.")
    lines.extend(['',
        'A reduced internal objective does not resolve an individual fingertip cap violation. These local nonsmooth constrained solves do not establish global infeasibility or prove that a faithful mapping is impossible. Any new solver, constraint Jacobian or restoration strategy is a proposed subsequent protocol and has not been evaluated here.', '',
        'Runtime histories remain separate: V1 was interrupted before any complete candidate; V2 failed import before any saved-case solve; V2.1 supplied the canonical outcomes; V3 was frozen and import-tested but its fitting protocol was never executed. The two-thread and exact-burial benchmarks are performance evidence only. No probes or candidates from earlier runtime versions were reused in V2.1.', '',
    ])
    report = OUT / 'REPORT.md'
    if report.exists():
        raise FileExistsError('Existing canonical report must not be overwritten')
    report.write_text('\n'.join(lines))
    print(json.dumps({'fit_summary_sha256': sha(dest), 'report_sha256': sha(report),
                      'selected': summary['selected_surface_candidate_count'],
                      'fallback': summary['exact_original_fallback_count']}))


if __name__ == '__main__':
    main()
