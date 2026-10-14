"""Stage explicitly allowlisted HUG history, without running scientific code.

This script reads source bytes and JSON metadata only. It imports no project,
NumPy, simulator, licensed asset or experiment helper. All writes remain under
its own /private/tmp directory. Existing differing staging files are refused.
"""
from pathlib import Path
import collections
import gzip
import hashlib
import io
import json
import math

BASE = Path('/private/tmp/hug_constrained_history_archival_2026-10-14')
STAGE = BASE / 'staged_history'
CANONICAL = Path('/private/tmp/hug_constrained_refinement_runtime_v2_1_2026-10-14')
CANONICAL_PROTOCOL_SHA = 'e42b74b787a8a1a5b5a661596466fe66b24c14eea39db25ad039fda2a462b8ff'
SOLVER_SHA = '0b29f867337cd11bb9128f5aa6ca92800338543055b286b40873d10f4d74f409'
CASE8_SHA = 'ce4153660c096900244d16af0364ab96c8c7cbab6f298860b68bf99a627e7290'

# Literal file lists are the complete copy authorization. No glob or recursive
# source copy occurs. Duplicate partial benchmark reports and V2 drafts remain
# at their original paths and are explicitly excluded from this staging.
GROUPS = [
    ('runtime_v1_interrupted', '/private/tmp/hug_constrained_refinement_2026-10-14',
     'INTERRUPTED_WITH_NO_COMPLETED_CASE',
     'Two cohorts stopped before any selected-candidate report; progress probes are not eligible final results. Final integration counters were not serialized, so a complete-run zero-step guard report is unavailable.',
     ['METHODS_NOTES.md', 'SOLVER_API.md', 'api_installation_verification.json',
      'check_rigid_refresh.py', 'cohort_20_24_28.log', 'cohort_8_12_16.log',
      'constrained_refinement.py', 'freeze_protocol.py', 'geometry_helpers.py',
      'install_constrained_api.py', 'installed_api_test_verification.json',
      'interrupted_status.json', 'manufactured_protocol.json',
      'manufactured_verification.json', 'merge_cohorts.py',
      'parent_protocol_review.json', 'protected_hashes.json', 'protocol.json',
      'read_only_code_review.json', 'rigid_refresh_run.log',
      'rigid_refresh_sanity.json', 'run_constrained_refinement.py',
      'run_manufactured_checks.py', 'save_implementation_checkpoint.py',
      'solver_source_manifest.json', 'test_constrained_surface_refinement.py',
      'verify_candidate_records.py']),
    ('runtime_v2_import_abort', '/private/tmp/hug_constrained_refinement_runtime_v2_2026-10-14',
     'IMPORT_ABORT_BEFORE_MAIN_CASE_OR_SOLVER',
     'All six singleton launches stopped during import because pool capacity two could not support the historical temporary four-thread mask. No case loading, solver callback or candidate record occurred; final whole-run guard counters are unavailable.',
     ['SOLVER_API.md', 'cohort_8.log', 'cohort_12.log', 'cohort_16.log',
      'cohort_20.log', 'cohort_24.log', 'cohort_28.log',
      'constrained_refinement.py', 'freeze_runtime_protocol.py',
      'geometry_helpers.py', 'manufactured_protocol.json',
      'manufactured_verification.json', 'merge_cohorts.py',
      'parent_protocol_review.json', 'protected_hashes.json', 'protocol.json',
      'run_constrained_refinement.py', 'runtime_change_manifest.json',
      'startup_abort_status.json', 'test_constrained_surface_refinement.py',
      'verify_candidate_records.py']),
    ('runtime_v3_unexecuted', '/private/tmp/hug_constrained_refinement_runtime_v3_2026-10-14',
     'FROZEN_FITTING_PREPARATION_NOT_EXECUTED',
     'The exact-buried helper preparation had an import smoke check, but no V3 case loading or solver execution and no candidate report. The progressing canonical V2.1 fits were retained.',
     ['SOLVER_API.md', 'UNEXECUTED_STATUS.json', 'check_import_runtime.py',
      'constrained_refinement.py', 'freeze_runtime_protocol.py',
      'geometry_helpers.py', 'helper_equivalence_review.json',
      'helper_implementation.diff', 'import_runtime_smoke.json',
      'manufactured_protocol.json', 'manufactured_verification.json',
      'merge_cohorts.py', 'protected_hashes.json', 'protocol.json',
      'run_constrained_refinement.py', 'runtime_change_manifest.json',
      'test_constrained_surface_refinement.py', 'verify_candidate_records.py']),
    ('performance_one_two_threads', '/private/tmp/hug_numba_two_thread_verification_2026-10-14',
     'MEASURED_STATIC_PARITY_RUNTIME_MASK_TWO_USED_BY_CANONICAL_V2_1',
     'Six originals and three fixed static FK perturbations establish observed exact helper-field/reference parity between one and two threads. This is runtime development evidence, not a constrained fit or scientific mapping verdict.',
     ['REPORT.md', 'benchmark.json', 'benchmark.log', 'benchmark_two_threads.py',
      'manifest.json', 'protocol.json', 'summary.json']),
    ('performance_aabb_unused', '/private/tmp/hug_visual_aabb_acceleration_2026-10-14',
     'MEASURED_STATIC_PILOT_NOT_ACTIVATED_IN_CONSTRAINED_FITS',
     'Manufactured fixtures and nine saved/static-perturbation records test an AABB membership shortcut. The shortcut was not activated in any constrained fit or the canonical helper.',
     ['REPORT.md', 'benchmark.json', 'benchmark.log',
      'benchmark_acceleration.py', 'manifest.json', 'protocol.json',
      'visual_membership_aabb.py']),
    ('performance_exact_buried_tested_unused', '/private/tmp/hug_exact_buried_acceleration_2026-10-14',
     'MEASURED_STATIC_HELPER_NOT_USED_BY_CANONICAL_FITTING',
     'Twenty-one manufactured full-field calls and nine saved/static-perturbation records support observed parity of conditional exact buried queries. This helper was prepared for unexecuted V3 and was not used by canonical V2.1 fits.',
     ['REPORT.md', 'benchmark.json', 'benchmark.log', 'benchmark_buried.py',
      'check_manufactured.py', 'exact_direction_buried.py', 'manifest.json',
      'manufactured.log', 'manufactured_buried_parity.json',
      'manufactured_direction_tests.json', 'parent_helper_review.json',
      'protocol.json', 'summary.json']),
    ('validation_v1_unexecuted', '/private/tmp/hug_constrained_validation_2026-10-14/runtime_v1_unexecuted',
     'FROZEN_WRAPPER_NOT_EXECUTED_ON_REAL_CANDIDATE',
     'The V1 candidate runtime produced no completed case, so this separately bound audit wrapper was retained without real-candidate execution.',
     ['status.json', 'protocol.json', 'validate_candidate.py']),
    ('validation_v3_unexecuted', '/private/tmp/hug_constrained_validation_runtime_v3_2026-10-14',
     'FROZEN_DEVELOPMENT_WRAPPER_NOT_EXECUTED_ON_REAL_CANDIDATE',
     'V3 fitting and real-candidate audit remained unexecuted. Fast-helper fixtures are separate development evidence; the actual canonical V2.1 audit is packaged separately by the root task.',
     ['development_unexecuted_status.json', 'protocol.json', 'validate_candidate.py']),
]
ALLOWED_SUFFIXES = {'.json', '.py', '.md', '.log', '.diff'}
README = '''# Preserved HUG runtime and performance history

14 October 2026. This staging preserves distinct source/protocol versions and
their recorded statuses. It does not merge scientific protocols or historical
results. The current canonical V2.1 candidate fits and their independent audit
are packaged separately and are not copied into this history selection.

- `runtime_v1_interrupted`: two interrupted cohorts, zero completed cases and
  no selected-candidate report. Logged objective/geometry probes are incomplete
  optimizer evaluations. Final full-run integration counters were not serialized;
  a complete-run zero-step guard report is unavailable.
- `runtime_v2_import_abort`: six startup-only failures before `main`, case
  loading or any solver callback. Final whole-run guard counters are unavailable.
- `runtime_v3_unexecuted`: frozen faster-helper preparation with import smoke
  evidence only. V3 fitting never ran and produced no candidate records.
- `performance_one_two_threads`: measured static helper/reference parity on six
  originals plus three fixed FK perturbations. Canonical V2.1 uses the verified
  two-thread mask with its unchanged original helper. Timing scope is the normal
  helper only; it is not complete-fit speed or mapping acceptance.
- `performance_aabb_unused`: measured static AABB pilot, never activated in a
  constrained fit or in canonical V2.1.
- `performance_exact_buried_tested_unused`: measured static and manufactured
  exact-buried helper parity, prepared for V3 but unused in canonical fitting.
- `validation_v1_unexecuted` and `validation_v3_unexecuted`: separately bound
  independent wrapper sources/protocols that were not run on real candidates.

Each group retains its original protocol. Original metadata statements remain
historical statements with their original scope. In particular, the preserved
`runtime_v1_interrupted/read_only_code_review.json` describes the canonical code
review while real-grasp results were pending at that review; it is not a V1 fit
verdict. `reviews/case8_read_only_review.json` separately records the completed
canonical case-8 trace review, bound to its report hash, without copying that
canonical fit report into history. Exact fallback does not prove global
infeasibility or scientific mapping PASS.

Only literal allowlisted metadata, text source, notes, diffs and logs are copied.
No licensed assets, scene binaries, query NPZ/NPY arrays or raw point arrays are
included. Duplicate partial benchmark reports and V2 draft protocols/manifests
remain preserved at their original paths and are excluded here. The canonical
direction helper and its original fixtures are packaged separately by root.

All logs are gzip-compressed with an empty filename and `mtime=0`; even the
small logs use the same deterministic rule. `manifest.json` binds both each
original uncompressed byte SHA-256 and each stored byte SHA-256, with sizes and
source paths. The archiver verifies decompression and rechecks source bytes after
copying. The archiver reads metadata only, imports no experiment code, reruns no
tests, and writes only beneath its own temporary staging directory.
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def save(relative, data):
    path = STAGE / relative
    assert path.resolve().is_relative_to(STAGE.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Differing staged evidence must not be overwritten: {path}')
    else:
        path.write_bytes(data)
    assert path.read_bytes() == data
    return path


def case8_review():
    report_path = CANONICAL / 'candidate_cohort_8.json'
    assert digest(report_path.read_bytes()) == CASE8_SHA
    assert digest((CANONICAL / 'protocol.json').read_bytes()) == CANONICAL_PROTOCOL_SHA
    report = read_json(report_path)
    protocol = read_json(CANONICAL / 'protocol.json')
    case = report['cases'][0]
    assert len(report['cases']) == 1 and case['source_inventory_index'] == 8
    audit = case['solver_audit']
    original = audit['baseline']
    raw = audit['raw_assessment']
    rows = audit['major_iteration_assessments']
    norm = lambda v: math.sqrt(sum(x*x for x in v))
    original_norms = [norm(v) for v in original['tip_error_vectors_m']]
    allowance = protocol['solver_configuration']['feasibility_error_allowance_m']
    config = protocol['solver_configuration']
    bounds = audit['bounded_correction_domain_rad']

    def eligible(q, assessment):
        return (q is not None and assessment is not None and len(q) == 24
                and all(math.isfinite(v) and lo <= v <= hi for v, (lo, hi) in zip(q, bounds))
                and all(norm(v) <= baseline + allowance for v, baseline in
                        zip(assessment['tip_error_vectors_m'], original_norms))
                and sum(norm(v) for v in assessment['tip_error_vectors_m']) / 5
                    <= config['maximum_mean_fingertip_error_m']
                and assessment['objective'] < original['objective'] - config['minimum_objective_improvement'])

    eligible_major = [r['major_iteration'] for r in rows if eligible(r['qpos_rad'], r['assessment'])]
    assert not eligible_major and not eligible(audit['raw_candidate_qpos_rad'], raw)
    assert all(eligible(r['qpos_rad'], r['assessment']) == r['eligible_lower_objective_candidate'] for r in rows)
    assert case['qpos_rad'] == case['original_qpos_rad'] == audit['returned_qpos_rad']
    assert case['root_target_position_m'] == case['original_root_target_position_m']
    assert case['root_target_quaternion_wxyz'] == case['original_root_target_quaternion_wxyz']
    assert audit['selected_candidate_source'] == 'original_fallback' and not audit['surface_candidate_returned']
    protected = read_json(CANONICAL / 'protected_hashes.json')
    bound_hashes = [('source', protocol['source_sha256']),
                    ('implementation', protocol['implementation_sha256']),
                    ('protected', protected)]
    assert all(digest(Path(path).read_bytes()) == sha for _, mapping in bound_hashes for path, sha in mapping.items())
    assert report['protocol_sha256'] == CANONICAL_PROTOCOL_SHA
    assert report['solver_implementation_sha256'] == SOLVER_SHA
    assert report['script_sha256'] == digest((CANONICAL / 'run_constrained_refinement.py').read_bytes())
    assert report['direction_implementation_sha256'] == digest(Path(protocol['direction_helper_path']).read_bytes())
    assert case['selected_fresh_diagnostics']['task_link_error_vectors_m'][:5] == audit['final_assessment']['tip_error_vectors_m']
    assert case['baseline_fresh_diagnostics']['task_link_error_vectors_m'][:5] == original['tip_error_vectors_m']
    raw_norms = [norm(v) for v in raw['tip_error_vectors_m']]
    return {
        'schema': 'hug_canonical_case8_independent_read_only_trace_review_v1',
        'review_date': '2026-10-14', 'status': 'NO_ACTIONABLE_CORRECTNESS_FINDINGS',
        'source_report': str(report_path), 'source_report_sha256': CASE8_SHA,
        'canonical_protocol_sha256': CANONICAL_PROTOCOL_SHA, 'solver_sha256': SOLVER_SHA,
        'case_index': 8, 'native_sha256': case['native_sha256'],
        'q_root_position_and_root_quaternion_exact_original': True,
        'selected_candidate_source': audit['selected_candidate_source'],
        'selected_major_iteration': audit['selected_major_iteration'],
        'tip_order': ['thumb', 'index', 'middle', 'ring', 'little'],
        'original_tip_error_norm_m': original_norms, 'raw_tip_error_norm_m': raw_norms,
        'raw_tip_error_increase_m': [a-b for a, b in zip(raw_norms, original_norms)],
        'raw_tip_nonworsening_with_10nm_pass': [a <= b + allowance for a, b in zip(raw_norms, original_norms)],
        'strict_mean_limit_m': config['maximum_mean_fingertip_error_m'],
        'baseline_mean_tip_error_m': original['mean_fingertip_error_m'],
        'raw_mean_tip_error_m': raw['mean_fingertip_error_m'],
        'baseline_objective': original['objective'], 'raw_objective': raw['objective'],
        'final_objective': audit['final_assessment']['objective'],
        'major_callback_count': len(rows), 'independently_eligible_major_iterations': eligible_major,
        'major_eligibility_flags_independently_verified': True,
        'major_refusal_counts': dict(collections.Counter(reason for r in rows for reason in r['refusal_reasons'])),
        'raw_refusal_reasons': audit['raw_final_refusal_reasons'],
        'solver_iterations': audit['iterations'], 'solver_jacobian_evaluations': audit['jacobian_evaluations'],
        'solver_full_geometry_evaluations': audit['geometry_callback_evaluations'],
        'driver_full_evaluations_including_fresh_baseline_and_final': case['full_callback_count_including_fresh_final'],
        'solver_success': audit['solver_success'], 'solver_status': audit['solver_status'],
        'solver_message': audit['solver_message'], 'budget_exhausted': audit['budget_exhausted'],
        'counter_interpretation': 'Solver-reported iteration count, callback count, Jacobian count and geometry evaluation count are distinct counters.',
        'full_FK_callback_agreement_max_error_m': audit['full_FK_callback_agreement_max_error_m'],
        'fresh_original_final_tip_bindings_verified': True,
        'source_implementation_and_protected_hashes_verified': True,
        'protected_source_count': len(protected), 'reported_guards': report['guards'],
        'time_s': case['time_s'], 'physics_steps': case['physics_steps'],
        'new_eval_data': case['new_eval_data'], 'training': case['training'],
        'files_changed_by_review': False, 'tests_or_simulations_rerun': False,
        'global_infeasibility_claimed': False, 'scientific_mapping_PASS_claimed': False,
        'scope': 'Read-only recorded selection/source-binding review of canonical case 8 only; not an independent geometry rerun, all-six verdict or convergence/global-feasibility certificate.',
    }


def main():
    assert Path(__file__).resolve().parent == BASE.resolve()
    source_records = []
    group_records = []
    for group, source_dir, status, meaning, names in GROUPS:
        root = Path(source_dir)
        assert len(names) == len(set(names))
        group_records.append({'directory': group, 'source_directory': source_dir,
                              'status': status, 'interpretation': meaning,
                              'file_count': len(names),
                              'protocol_sha256': digest((root/'protocol.json').read_bytes())})
        for name in names:
            assert Path(name).name == name and Path(name).suffix in ALLOWED_SUFFIXES
            source = root / name
            assert not source.is_symlink() and source.is_file()
            raw = source.read_bytes()
            stored = raw
            relative = f'{group}/{name}'
            compressed = name.endswith('.log')
            if compressed:
                stream = io.BytesIO()
                with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as gz:
                    gz.write(raw)
                stored = stream.getvalue()
                assert gzip.decompress(stored) == raw
                relative += '.gz'
            save(relative, stored)
            assert source.read_bytes() == raw, f'Source changed while archiving: {source}'
            source_records.append({'source_path': str(source), 'stored_path': relative,
                                   'original_bytes': len(raw), 'original_byte_sha256': digest(raw),
                                   'stored_bytes': len(stored), 'stored_byte_sha256': digest(stored),
                                   'encoding': 'gzip_mtime0_empty_filename' if compressed else 'original_bytes',
                                   'source_unchanged_after_copy': True})
    generated = []
    for relative, value in [('README.md', README.encode()),
                            ('reviews/case8_read_only_review.json', json_bytes(case8_review()))]:
        save(relative, value)
        generated.append({'stored_path': relative, 'stored_bytes': len(value),
                          'stored_byte_sha256': digest(value), 'origin': 'generated_scoped_archival_metadata'})
    manifest = {
        'schema': 'hug_constrained_runtime_and_performance_history_staging_manifest_v1',
        'archive_date': '2026-10-14', 'staging_directory': str(STAGE),
        'archiver_path': str(Path(__file__).resolve()),
        'archiver_sha256': digest(Path(__file__).read_bytes()),
        'groups': group_records, 'copied_file_count': len(source_records),
        'original_byte_count': sum(x['original_bytes'] for x in source_records),
        'stored_byte_count': sum(x['stored_bytes'] for x in source_records),
        'copied_files': source_records, 'generated_files': generated,
        'canonical_V2_1_fit_and_audit_copied': False,
        'canonical_case8_used_only_as_read_only_source_for_scoped_review': True,
        'source_or_checkout_writes': False, 'scientific_code_imported_or_run': False,
        'tests_rerun': False, 'scientific_protocol_versions_merged': False,
        'licensed_assets_scene_binaries_query_arrays_or_raw_point_arrays_copied': False,
        'explicit_exclusions': ['All nonallowlisted files', 'benchmark.partial.json in each performance directory',
                                'runtime V2 protocol_draft.json and runtime_change_manifest_draft.json',
                                'Canonical V2.1 candidate and audit files', 'Licensed assets, scenes and raw query arrays'],
        'limitations': ['Historical incomplete run guards are unavailable as stated in original statuses.',
                        'Static helper parity is measured-domain development evidence, not a global proof or mapping verdict.',
                        'Case 8 exact fallback establishes recorded selection correctness only.'],
    }
    save('manifest.json', json_bytes(manifest))
    print(json.dumps({'staged_history': str(STAGE), 'manifest': str(STAGE/'manifest.json'),
                      'copied_files': len(source_records), 'groups': len(group_records),
                      'all_source_bytes_preserved': True, 'tests_or_scientific_code_run': False}))


if __name__ == '__main__':
    main()
