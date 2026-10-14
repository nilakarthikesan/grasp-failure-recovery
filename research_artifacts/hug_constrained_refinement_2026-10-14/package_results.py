"""Package complete static results; never accept a partial run or change source evidence."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import shutil

ROOT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
BUNDLE = ROOT / 'research_artifacts/hug_constrained_refinement_2026-10-14'
FIT = Path('/private/tmp/hug_constrained_refinement_runtime_v2_1_2026-10-14')
AUDIT = Path('/private/tmp/hug_constrained_validation_runtime_v2_1_2026-10-14')
DIRECTION = Path('/private/tmp/hug_constrained_validation_2026-10-14')
HISTORY = Path('/private/tmp/hug_constrained_history_archival_2026-10-14/staged_history')
SELECTION_REVIEW = HISTORY.parent / 'all_six_selection_read_only_review.json'
FINAL_REVIEW = HISTORY.parent / 'final_report_read_only_review.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def main():
    candidate = read(FIT / 'candidate_targets.json')
    summary = read(AUDIT / 'summary.json')
    assert candidate['complete_original_six_case_coverage']
    assert candidate['protocol_sha256'] == sha(FIT / 'protocol.json')
    assert summary['candidate_protocol_sha256'] == candidate['protocol_sha256']
    assert summary['protocol_sha256'] == sha(AUDIT / 'protocol.json')
    assert summary['case_count'] == 6 and summary['resolution_count'] == 9
    assert summary['all_engineering_source_preflights_pass']
    assert summary['all_protected_hashes_unchanged']
    assert candidate['guards']['actual'] == summary['guards_actual_attempts'] == 0
    assert candidate['guards']['selftests'] == 18
    assert candidate['physics_steps'] == 0 and candidate['new_eval_data'] is False
    protected = read(FIT / 'protected_hashes.json')
    assert len(protected) == 114
    for p, h in protected.items():
        assert sha(p) == h, p
    api = ROOT / 'src/grasp_failure_prediction/evaluation/constrained_surface_refinement.py'
    test = ROOT / 'tests/test_constrained_surface_refinement.py'
    assert sha(api) == candidate['solver_implementation_sha256'] == sha(FIT / 'constrained_refinement.py')
    assert sha(test) == sha(FIT / 'test_constrained_surface_refinement.py')
    for subdir in ('fit', 'validation', 'direction', 'history'):
        assert not (BUNDLE / subdir).exists(), subdir
    compression = {}
    def copy_group(source, target, names, compressed=()):
        target.mkdir(parents=True)
        for name in names:
            src = source / name
            assert src.is_file(), src
            raw = src.read_bytes()
            dest_name = name + '.gz' if name in compressed else name
            dest = target / dest_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(gzip.compress(raw, mtime=0) if name in compressed else raw)
            if name in compressed:
                assert gzip.decompress(dest.read_bytes()) == raw
                compression[str(dest.relative_to(BUNDLE))] = {
                    'source_path': str(src), 'uncompressed_sha256': sha(src),
                    'gzip_sha256': sha(dest), 'uncompressed_size_bytes': len(raw),
                    'exact_original_bytes_preserved': True,
                }
    fit_plain = [
        'protocol.json', 'run_constrained_refinement.py', 'merge_cohorts.py',
        'constrained_refinement.py', 'geometry_helpers.py', 'protected_hashes.json',
        'verify_candidate_records.py', 'runtime_change_manifest.json',
        'parent_protocol_review.json', 'check_import_runtime.py',
        'import_runtime_smoke.json', 'launch_sessions.json',
        'verification.json', 'fit_summary.json', 'build_result_summary.py', 'REPORT.md',
        'REPORT_COUNTER_CLARIFIED.md', 'counter_wording_clarification.json',
        'selected_artifact_manifest.json',
    ]
    fit_compressed = ['candidate_targets.json']
    for i in (8,12,16,20,24,28):
        fit_compressed.extend([f'candidate_cohort_{i}.json', f'cohort_{i}.log'])
    for name in ('candidate_record_verification.json', 'completed_status.json'):
        if (FIT / name).is_file():
            fit_plain.append(name)
    audit_names = [
        'protocol.json', 'validate_candidate.py', 'protected_hashes.json',
        'parent_wrapper_review.json', 'binding_diff_review.json',
        'selection_audit_tests.json', 'selection_test_reuse.json',
        'selection_test_import_abort.json', 'test_selection_audit.py',
        'presentation_preparation.json', 'presentation_binding_addendum.json',
        'build_summary.py', 'plot_summary.py', 'build_report.py',
        'prepare_audit_shards.py', 'audit_shard_plan.json',
        'candidate_validation_preflight_only.json.gz',
        'candidate_validation_preflight_only.json.gz.binding.json',
        'candidate_validation_8,12,16.json.gz',
        'candidate_validation_8,12,16.json.gz.binding.json',
        'candidate_validation_20,24,28.json.gz',
        'candidate_validation_20,24,28.json.gz.binding.json',
        'summary.json', 'REPORT.md', 'report_metadata.json',
        'constrained_candidate_tradeoffs.png', 'constrained_fingertip_errors.png',
        'figure_metadata.json',
        'preflight.log', 'cohort_a/audit.log', 'cohort_b/audit.log',
        'cohort_a/protocol.json', 'cohort_b/protocol.json',
        'final_presentation_revision.json',
    ]
    for script in ('build_summary', 'plot_summary', 'build_report'):
        audit_names.extend([
            f'presentation_binding_addendum_to_final_{script}.diff',
            f'presentation_post_audit_pre_layout_to_final_{script}.diff',
            f'presentation_history/hash_recovered_binding_addendum/{script}.py',
            f'presentation_history/post_audit_pre_layout_revision/{script}.py',
        ])
    for name in ('summary.json', 'REPORT.md', 'report_metadata.json',
                 'figure_metadata.json', 'constrained_candidate_tradeoffs.png',
                 'constrained_fingertip_errors.png'):
        audit_names.append('presentation_history/post_audit_pre_layout_revision/'+name)
    for name in ('selected_artifact_manifest.json', 'publication_manifest.json'):
        if (AUDIT / name).is_file():
            audit_names.append(name)
    publication = read(AUDIT / 'selected_publication_manifest.json')
    assert publication['query_caches_and_source_assets_not_included']
    for row in publication['files']:
        name = row['relative_path']
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert Path(name).suffix in ('.py', '.json', '.gz', '.md', '.png', '.diff', '.log')
        assert sha(AUDIT/name) == row['sha256']
        if name not in audit_names:
            audit_names.append(name)
    audit_names.append('selected_publication_manifest.json')
    direction_names = [
        'exact_direction.py', 'test_exact_direction.py',
        'manufactured_direction_tests.json', 'check_original_direction_parity.py',
        'original_direction_eligibility_parity.json',
    ]
    assert HISTORY.is_dir()
    assert SELECTION_REVIEW.is_file()
    assert FINAL_REVIEW.is_file()
    for source, names in ((FIT, fit_plain+fit_compressed), (AUDIT, audit_names), (DIRECTION, direction_names)):
        for name in names:
            assert (source/name).is_file(), source/name
    copy_group(FIT, BUNDLE / 'fit', fit_plain + fit_compressed, fit_compressed)
    copy_group(AUDIT, BUNDLE / 'validation', audit_names,
               ['preflight.log', 'cohort_a/audit.log', 'cohort_b/audit.log'])
    # Preserve the publication manifest's original log bindings alongside the
    # separately compressed runtime log copies.
    (BUNDLE/'validation/preflight.log').write_bytes((AUDIT/'preflight.log').read_bytes())
    for row in publication['files']:
        assert sha(BUNDLE/'validation'/row['relative_path']) == row['sha256']
    acceptance = ROOT / 'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/acceptance/acceptance_rule.json'
    assert sha(acceptance) == candidate['strict_acceptance_rule_unchanged_sha256']
    (BUNDLE / 'validation' / 'acceptance_rule.json').write_bytes(acceptance.read_bytes())
    (BUNDLE / 'validation' / 'selection_read_only_review.json').write_bytes(SELECTION_REVIEW.read_bytes())
    (BUNDLE / 'validation' / 'final_report_read_only_review.json').write_bytes(FINAL_REVIEW.read_bytes())
    copy_group(DIRECTION, BUNDLE / 'direction', direction_names)
    shutil.copytree(HISTORY, BUNDLE / 'history')
    history_archiver = HISTORY.parent / 'prepare_history.py'
    assert sha(history_archiver) == read(HISTORY / 'manifest.json')['archiver_sha256']
    (BUNDLE / 'history' / 'prepare_history.py').write_bytes(history_archiver.read_bytes())
    (BUNDLE / 'package_results.py').write_bytes(Path(__file__).read_bytes())
    write_json(BUNDLE / 'compression_bindings.json', compression)
    write_json(BUNDLE / 'completion_verification.json', {
        'schema': 'hug_complete_constrained_static_bundle_verification_v1',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_sha256': sha(FIT / 'candidate_targets.json'),
        'candidate_protocol_sha256': candidate['protocol_sha256'],
        'independent_protocol_sha256': summary['protocol_sha256'],
        'strict_acceptance_rule_sha256': candidate['strict_acceptance_rule_unchanged_sha256'],
        'installed_solver_sha256': sha(api), 'installed_test_sha256': sha(test),
        'protected_source_count': len(protected), 'all_protected_hashes_unchanged': True,
        'case_count': 6, 'resolution_count': 9,
        'all_engineering_source_preflights_pass': True,
        'strict_status_counts': summary['strict_status_counts'],
        'fit_guard_selftests': 18, 'actual_physics_calls': 0,
        'new_eval_data': False, 'new_inference': False, 'training': False,
        'production_mapper_changed': False, 'next_generator_advanced': False,
        'incomplete_and_unexecuted_history_excluded_from_results': True,
    })
    write_json(BUNDLE / 'results_manifest.json', {
        'schema': 'hug_complete_constrained_static_results_manifest_v1',
        'scope': 'Six original HUG development poses, complete constrained fits and nine independent unchanged-rule static audits. Runtime histories remain separate. No new eval data.',
        'files_sha256': {str(p.relative_to(BUNDLE)): sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p.name != 'results_manifest.json'},
        'new_eval_data': False, 'new_inference': False, 'physics_steps': 0,
    })
    print(json.dumps({'bundle': str(BUNDLE), 'manifest_sha256': sha(BUNDLE/'results_manifest.json'), 'file_count': len(read(BUNDLE/'results_manifest.json')['files_sha256'])}))

if __name__ == '__main__':
    main()
