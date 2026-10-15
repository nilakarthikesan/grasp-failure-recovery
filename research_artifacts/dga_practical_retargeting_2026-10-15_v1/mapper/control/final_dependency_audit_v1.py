"""Read-only audit of fixed DGA source/dependency bindings; no fitting."""
from datetime import datetime, timezone, timedelta
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path('/private/tmp/dga_practical_retargeting_20261015_v1')
WORK = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
PROTOCOL = Path('/private/tmp/dga_practical_protocol_20261015_v1/criteria_frozen_v1.json')
SOURCE_PATHS = [
    'src/grasp_failure_prediction/proposals/dga_practical.py',
    'src/grasp_failure_prediction/proposals/dexgrasp_anything.py',
    'tests/test_dga_practical.py',
    'tests/test_dga_practical_adversarial.py',
    'tests/test_dexgrasp_anything_provider.py',
    'scripts/infer_dexgrasp_anything_proposals.py',
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return 'sha256:' + h.hexdigest()

historical = WORK / 'research_artifacts/dexgrasp_anything_mapping_validation_2026-10-14_contract_v1/evidence_manifest.json'
old_entries = {x['path']: 'sha256:' + x['sha256'] for x in json.loads(historical.read_text())['entries']}
sources = []
for relative in SOURCE_PATHS:
    path = WORK / relative
    tree = ast.parse(path.read_text())
    imports = sorted(set(
        node.module or '' if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])))
    digest = sha(path)
    assert relative not in old_entries or digest == old_entries[relative], relative
    sources.append({'path': str(path), 'sha256': digest, 'imports': imports,
                    'historical_contract_unchanged': digest == old_entries.get(relative) if relative in old_entries else None})

cohorts = []
for method_name, result_name in [('method_frozen_v3.json', 'results_v3'),
                                 ('method_original_four_regression_frozen_v3.json', 'results_original_four_regression_v3')]:
    method_path, result_path = ROOT / method_name, ROOT / result_name
    method = json.loads(method_path.read_text())
    report = json.loads((result_path / 'report.json').read_text())
    assert report['method_sha256'] == sha(method_path)
    assert report['protocol_sha256'] == sha(PROTOCOL) == method['protocol']['sha256']
    for path, digest in method['dependency_sha256'].items():
        assert sha(path) == digest, path
    for name, digest in report['outputs'].items():
        assert sha(result_path / name) == digest, name
    inferred_start = datetime.fromisoformat(report['created_at_utc']) - timedelta(seconds=report['elapsed_seconds'])
    assert datetime.fromisoformat(method['created_at_utc']) < inferred_start
    bindings = []
    with np.load(result_path / 'poses.npz', allow_pickle=False) as poses:
        mapped_ids = list(map(int, poses['global_sample_indices']))
        for record in report['records']:
            item = {'row': record['global_sample_index'], 'status': record['status']}
            for field in ['native_manifest', 'mapping_reference']:
                assert sha(record[field]['path']) == record[field]['sha256']
            native_path = Path(record['native_manifest']['path'])
            native = json.loads(native_path.read_text())
            raw_path = native_path.parent / native['output']['path']
            assert sha(raw_path) == native['output']['sha256']
            ref_path = Path(record['mapping_reference']['path'])
            reference = json.loads(ref_path.read_text())
            reference_npz = ref_path.parent / reference['arrays']['path']
            assert sha(reference_npz) == reference['arrays']['sha256']
            assert reference['native_output_sha256'] == sha(raw_path)
            with np.load(raw_path, allow_pickle=False) as raw:
                row = raw['hand_qpos'][record['within_batch_sample_index']]
                # Original runner canonicalizes float values to float64 before hashing.
                row_hash = 'sha256:' + hashlib.sha256(np.asarray(row, dtype=np.float64).tobytes()).hexdigest()
                assert row_hash == record['raw33_row_sha256']
                if record['global_sample_index'] in mapped_ids:
                    assert np.array_equal(poses['source_hand_qpos'][mapped_ids.index(record['global_sample_index'])], row)
            item.update({'raw_archive': str(raw_path), 'raw_archive_sha256': sha(raw_path),
                         'source_runtime_npz': str(reference_npz), 'source_runtime_npz_sha256': sha(reference_npz),
                         'raw33_row_sha256': row_hash, 'precision_policy': reference['precision_policy']})
            bindings.append(item)
    cohorts.append({'method': str(method_path), 'method_sha256': sha(method_path),
                    'frozen_dependency_count': len(method['dependency_sha256']),
                    'all_frozen_dependencies_match': True,
                    'freeze_utc': method['created_at_utc'],
                    'inferred_run_start_utc': inferred_start.isoformat(),
                    'report': str(result_path / 'report.json'), 'report_sha256': sha(result_path / 'report.json'),
                    'output_files_rehashed': len(report['outputs']),
                    'accounting': report['accounting'], 'runtime': method['runtime'],
                    'rows': bindings})

audit = {
    'schema_version': 'dga_practical_final_source_dependency_audit_v1',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Read-only source/dependency and existing proposal-output provenance audit; no fitting rerun.',
    'sources': sources,
    'historical_contract_manifest': {'path': str(historical), 'sha256': sha(historical)},
    'additional_file_required_for_complete_provider_tests': 'scripts/infer_dexgrasp_anything_proposals.py',
    'all_new_source_test_files_to_include': SOURCE_PATHS,
    'already_tracked_import_dependencies': ['pyproject.toml', 'src/grasp_failure_prediction/__init__.py',
        'src/grasp_failure_prediction/proposals/__init__.py', 'src/grasp_failure_prediction/proposals/schema.py',
        'src/grasp_failure_prediction/evaluation/__init__.py', 'src/grasp_failure_prediction/evaluation/schema.py'],
    'import_execution_review': {
        'worker_main_guarded': True, 'worker_imports_inference_or_physics': False,
        'provider_tests_use_manufactured_fixtures': True,
        'sole_provider_test_subprocess': 'Import-only child explicitly rejects torch/mujoco/isaacgym/pytorch3d/csdf/hydra',
        'tests_reexecuted_in_this_audit': False,
    },
    'cohorts': cohorts,
    'full_experiment_reproducibility_limit': 'Frozen experiment runners retain absolute paths and external runtime assets/dependencies. Restore exact paths and bytes before replay, or create a new separately versioned relocation implementation. Committed core tests are separately self-contained with their six-file closure and declared Python dependencies.',
    'proposal_output_claim': 'These are provisional proposal outputs, not final independently admitted poses. Independent validator applies branch fallback separately.',
    'zero_execution': {'fitting': 0, 'inference': 0, 'physics_steps': 0, 'training': 0, 'evaluation_episodes': 0},
}
out = ROOT / 'control/source_dependency_audit_v1.json'
with out.open('x') as stream:
    json.dump(audit, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'audit': str(out), 'sha256': sha(out),
                  'sources': len(sources), 'dependency_counts': [x['frozen_dependency_count'] for x in cohorts],
                  'output_files_rehashed': sum(x['output_files_rehashed'] for x in cohorts),
                  'native_rows_verified': sum(len(x['rows']) for x in cohorts)}))
