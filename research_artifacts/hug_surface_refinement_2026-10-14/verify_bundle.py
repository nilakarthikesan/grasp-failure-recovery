"""Read-only byte/protocol verifier, not a scientific acceptance test."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_bytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    manifest = read(BASE / 'manifest.json')
    for name, digest in manifest['files_sha256'].items():
        assert sha(BASE / name) == digest, name
    for relative, digest in manifest['installed_static_API_and_tests_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    checked = 0
    if not args.offline:
        protected = read(BASE / 'prototype/protected_hashes.json')
        assert len(protected) == 110
        for path, digest in protected.items():
            assert sha(path) == digest, path
        checked = len(protected)
    raw_bytes = gzip.decompress((BASE / 'prototype/candidate_targets.json.gz').read_bytes())
    candidate_bytes = gzip.decompress((BASE / 'prototype/guarded_candidate_targets.json.gz').read_bytes())
    gp = read(BASE / 'prototype/guard_protocol.json')
    assert hashlib.sha256(raw_bytes).hexdigest() == gp['raw_v2_report_sha256']
    candidate = json.loads(candidate_bytes)
    assert candidate['protocol_sha256'] == sha(BASE / 'prototype/guard_protocol.json')
    audit = read(BASE / 'validation/summary.json')
    assert audit['candidate_sha256'] == hashlib.sha256(candidate_bytes).hexdigest()
    assert audit['validation_protocol_sha256'] == sha(BASE / 'validation/protocol.json')
    assert (audit['case_count'], audit['resolution_count'], audit['strict_fail_resolution_count']) == (6, 9, 9)
    assert audit['strict_pass_resolution_count'] == 0
    rule = ROOT / 'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/acceptance/acceptance_rule.json'
    assert sha(rule) == read(BASE / 'prototype/protocol.json')['strict_acceptance_rule_unchanged_sha256']
    all_cases = []
    for name in ('candidate_validation_8,12,16.json.gz', 'candidate_validation_20,24,28.json.gz'):
        archive = BASE / 'validation' / name
        binding = read(BASE / 'validation' / (name + '.binding.json'))
        raw = gzip.decompress(archive.read_bytes())
        assert sha(archive) == binding['gzip_sha256'] == audit['source_reports_sha256'][name]
        assert hashlib.sha256(raw).hexdigest() == binding['uncompressed_sha256']
        report = json.loads(raw)
        assert report['candidate_sha256'] == audit['candidate_sha256'] == binding['candidate_sha256']
        assert report['protocol_sha256'] == audit['validation_protocol_sha256'] == binding['protocol_sha256']
        assert report['script_sha256'] == sha(BASE / 'validation/validate_candidate.py')
        all_cases.extend(report['cases'])
    assert len(all_cases) == 6
    assert {c['source_inventory_index'] for c in all_cases} == {8,12,16,20,24,28}
    statuses = [r['frozen_acceptance_application']['strict_relationship_status_conditional']
                for c in all_cases for r in c['resolutions'].values()]
    assert statuses == ['FAIL'] * 9
    assert audit['physics_steps'] == 0 and audit['new_eval_data'] is False
    print(json.dumps({'published_files_checked': len(manifest['files_sha256']),
                      'installed_API_and_test_files_checked': len(manifest['installed_static_API_and_tests_sha256']),
                      'original_protected_files_checked': checked,
                      'runtime_protected_check_available': not args.offline,
                      'candidate_protocol_audit_bindings_match': True,
                      'strict_fail_domains': 9, 'physics_steps': 0,
                      'meaning': 'Integrity/status verification; no independent geometry or acceptance proof.'}))

if __name__ == '__main__':
    main()
