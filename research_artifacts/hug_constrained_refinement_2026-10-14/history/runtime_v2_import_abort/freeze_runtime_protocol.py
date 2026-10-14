"""Freeze a runtime-only derivative, without loading a case or simulator."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json, math

OUT = Path(__file__).parent
V1 = Path('/private/tmp/hug_constrained_refinement_2026-10-14')
V1_PROTOCOL_SHA = 'aa3fe3556ee7e838a104e0cf78df69e694344c30604789c4542299d1637aa021'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
RUNTIME_ENVIRONMENT = {
    'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '2',
    'OMP_DYNAMIC': 'FALSE', 'MKL_DYNAMIC': 'FALSE',
}
TWO_THREAD_REPORT_SHA = '85f27b67512fb2872be5a56c8cd3ed2bcb514c4e95369336ae42ca4c24829952'
TWO_THREAD_PROTOCOL_SHA = '5c0106e4c3450631bb0ba34a6763a53469331f7084342e3db1de8506092738cb'


def nested_functions(path):
    tree = ast.parse(Path(path).read_text())
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in ('set_pose', 'evaluate', 'evaluate_tips')}


def validate_thread_report(path, parent):
    assert sha(path) == TWO_THREAD_REPORT_SHA
    report = read(path)
    benchmark_protocol_path = path.parent / 'protocol.json'
    assert sha(benchmark_protocol_path) == report['protocol_sha256'] == TWO_THREAD_PROTOCOL_SHA
    benchmark_protocol = read(benchmark_protocol_path)
    assert report['schema'] == 'hug_unchanged_normal_two_thread_benchmark_v1'
    assert benchmark_protocol['schema'] == 'hug_unchanged_normal_numba_two_thread_protocol_v1'
    assert benchmark_protocol['thread_variants'] == [1, 2]
    assert benchmark_protocol['perturb_inventory_indices'] == [8, 16, 24]
    assert benchmark_protocol['perturb_offset_rad'] == 0.02
    assert report['guards']['actual'] == 0 and report['guards']['selftests'] == 3
    assert report['guards']['testing'] is False and report['physics_steps'] == 0
    assert report['source_and_protected_hashes_unchanged'] is True
    for flag in ('new_inference', 'new_eval_data', 'constrained_solver_run',
                 'ongoing_V1_modified', 'AABB_shortcut_activated'):
        assert report[flag] is False, flag
    assert benchmark_protocol['physics_steps'] == 0
    assert benchmark_protocol['new_eval_data'] is False and benchmark_protocol['new_inference'] is False
    for source, digest in benchmark_protocol['source_sha256'].items():
        assert sha(source) == digest, source
    assert benchmark_protocol['source_sha256'][parent['direction_helper_path']] == sha(parent['direction_helper_path'])
    planned = {row['source_inventory_index']: row['native_sha256'] for row in parent['cases']}
    expected = {(index, 'original') for index in planned} | {(index, 'fixed_plus') for index in (8, 16, 24)}
    records = report['records']
    assert len(records) == len(expected) == 9
    assert len({(row['source_inventory_index'], row['pose']) for row in records}) == 9
    assert {(row['source_inventory_index'], row['pose']) for row in records} == expected
    for row in records:
        assert row['native_sha256'] == planned[row['source_inventory_index']]
        assert row['offset_rad'] == (0.0 if row['pose'] == 'original' else 0.02)
        assert row['time_s'] == 0 and row['object_pose_unchanged'] is True
        assert row['all_original_eligible_fields_exact'] is True
        assert row['reference_sites_and_coverage_exact'] is True
        variants = row['normal_field_variants']
        assert [(variant['variant'], variant['threads']) for variant in variants] == [('original_1thread', 1), ('original_2threads', 2)]
        assert variants[0]['eligible_bin_count'] == variants[1]['eligible_bin_count']
        for variant in variants:
            assert variant['all_fields_masks_scores_and_diagnostics_exact'] is True
            assert len(variant['timing']['seconds']) == benchmark_protocol['timing_repeats'] == 2
            assert all(math.isfinite(value) and value > 0 for value in variant['timing']['seconds'])
    return benchmark_protocol_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread-parity-report', type=Path)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    destination = OUT / ('protocol.json' if args.freeze else 'protocol_draft.json')
    if destination.exists():
        raise FileExistsError('Versioned protocol evidence must not be overwritten')
    assert sha(V1 / 'protocol.json') == V1_PROTOCOL_SHA
    original = read(V1 / 'protocol.json')
    for path, digest in original['implementation_sha256'].items():
        assert sha(path) == digest, path
    for name in ('constrained_refinement.py', 'geometry_helpers.py', 'protected_hashes.json'):
        assert sha(OUT / name) == sha(V1 / name), name
    assert nested_functions(OUT / 'run_constrained_refinement.py') == nested_functions(V1 / 'run_constrained_refinement.py')
    protected = read(OUT / 'protected_hashes.json')
    for path, digest in protected.items():
        assert sha(path) == digest, path
    for group in ('source_sha256', 'query_cache_sha256', 'full_cache_sha256'):
        for path, digest in original[group].items():
            assert sha(path) == digest, path
    if args.freeze and args.thread_parity_report is None:
        raise ValueError('Final freeze requires the reviewed exact 1-versus-2 thread report')
    benchmark_protocol_path = validate_thread_report(args.thread_parity_report, original) if args.thread_parity_report else None
    scripts = [OUT / 'constrained_refinement.py', OUT / 'run_constrained_refinement.py',
               OUT / 'geometry_helpers.py', OUT / 'merge_cohorts.py',
               Path(original['direction_helper_path'])]
    for script in scripts:
        ast.parse(script.read_text())
    protocol = dict(original)
    protocol.update({
        'schema': 'hug_global_constrained_surface_protocol_runtime_v2',
        'runtime_protocol_version': 2,
        'frozen_utc': datetime.now(timezone.utc).isoformat() if args.freeze else None,
        'before_first_real_candidate_metrics': False,
        'before_first_runtime_v2_candidate_metrics': True,
        'prior_runtime_v1_callback_metrics_already_observed': True,
        'new_scientific_protocol': False,
        'scientific_settings_predeclared_under_parent_protocol': V1_PROTOCOL_SHA,
        'scope': 'Six original saved seed-0 HUG grasps, six independent singleton CPU cohorts. All scientific settings, original seeds, sites, exact geometry, constraints, objective and selection are unchanged from V1. No inference, eval episodes, physics, task labels or training.',
        'allowed_cohorts': [[8], [12], [16], [20], [24], [28]],
        'cohort_filename_policy': 'candidate_cohort_8.json through candidate_cohort_28.json, one case per declared singleton cohort, with noncolliding partial files. Merge requires all six unique original hashes and a common reviewed runtime protocol/source/helper.',
        'implementation_sha256': {str(script): sha(script) for script in scripts},
        'numba_threads_per_process': 2,
        'runtime_environment': RUNTIME_ENVIRONMENT,
        'maximum_concurrent_processes': 6,
        'parent_runtime_protocol_path': str(V1 / 'protocol.json'),
        'parent_runtime_protocol_sha256': V1_PROTOCOL_SHA,
        'runtime_only_change': True,
        'scientific_settings_unchanged_from_parent': True,
        'scientific_kernel_AST_identical': ['set_pose', 'evaluate', 'evaluate_tips'],
        'parent_execution_preservation': 'V1 source/protocol/logs are preserved byte-for-byte. Its interruption, if authorized, is recorded separately; no V1 probes or partial records are merged into V2. Each V2 case starts at original q with fresh model/solver/cache.',
        'candidate_schema': 'hug_global_constrained_surface_candidate_v1',
        'thread_parity_report_path': str(args.thread_parity_report.resolve()) if args.thread_parity_report else None,
        'thread_parity_report_sha256': sha(args.thread_parity_report) if args.thread_parity_report else None,
        'thread_parity_protocol_path': str(benchmark_protocol_path) if benchmark_protocol_path else None,
        'thread_parity_protocol_sha256': sha(benchmark_protocol_path) if benchmark_protocol_path else None,
    })
    immutable = ['cases', 'solver_configuration', 'surface_configuration', 'direction_configuration',
                 'query_cache_root', 'query_cache_sha256', 'original_full_distance_root',
                 'full_cache_sha256', 'direction_helper_path', 'source_sha256',
                 'protected_snapshot_sha256', 'protected_source_count',
                 'strict_acceptance_rule_sha256', 'compound_objective', 'constraint_policy',
                 'candidate_selection', 'palm_anchor', 'cache_policy', 'limitations']
    assert all(protocol[key] == original[key] for key in immutable)
    manifest = {
        'schema': 'hug_global_constrained_runtime_derivative_manifest_v2',
        'parent_protocol_sha256': V1_PROTOCOL_SHA,
        'science_configuration_fields_identical': immutable,
        'scientific_kernel_AST_identical': protocol['scientific_kernel_AST_identical'],
        'unchanged_solver_sha256': sha(OUT / 'constrained_refinement.py'),
        'unchanged_geometry_helpers_sha256': sha(OUT / 'geometry_helpers.py'),
        'unchanged_direction_helper_sha256': sha(protocol['direction_helper_path']),
        'changes': ['Numba threads per process 1 to 2', 'BLAS/OpenMP thread controls explicitly 1',
                    'Two three-case cohorts to six singleton cohorts', 'Runtime/cohort provenance fields only'],
        'new_candidate_measurement_executed_by_freezer': False,
    }
    destination.write_text(json.dumps(protocol, indent=2) + '\n')
    (OUT / ('runtime_change_manifest.json' if args.freeze else 'runtime_change_manifest_draft.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'protocol_path': str(destination), 'protocol_sha256': sha(destination),
                      'implementation_sha256': protocol['implementation_sha256'],
                      'frozen': args.freeze, 'candidate_execution': False}))


if __name__ == '__main__':
    main()
