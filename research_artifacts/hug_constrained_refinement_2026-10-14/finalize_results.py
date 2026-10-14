"""Bind completed docs/evidence and verify every protected source; no geometry reruns."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import gzip
import hashlib
import json
import re

ROOT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
B = ROOT / 'research_artifacts/hug_constrained_refinement_2026-10-14'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
write = lambda p,v: Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    m = read(B/'results_manifest.json')
    for name, h in m['files_sha256'].items():
        if name != 'README.md':
            assert sha(B/name) == h, name
    d = ROOT/'docs/HUG_CONSTRAINED_REFINEMENT.md'
    text = d.read_text().replace('their errors, including regressions, will be reported.',
                              'their errors, including regressions, are reported.')
    text = text.replace('120 iterations, eps 1e-5 rad, ftol 1e-9 and robust f_scale 0.2.',
                        'a 120-iteration budget, eps 1e-5 rad, ftol 1e-9 and robust f_scale 0.2.')
    d.write_text(text)
    (B/'documentation_results.py').write_bytes(Path('/private/tmp/update_hug_constrained_docs_2026-10-14.py').read_bytes())
    (B/'finalize_results.py').write_bytes(Path(__file__).read_bytes())
    publication = read(B/'validation/selected_publication_manifest.json')
    for row in publication['files']:
        p = B/'validation'/row['relative_path']
        assert p.stat().st_size == row['size_bytes'] and sha(p) == row['sha256']
    compression = read(B/'compression_bindings.json')
    for name, row in compression.items():
        data = (B/name).read_bytes()
        raw = gzip.decompress(data)
        assert hashlib.sha256(data).hexdigest() == row['gzip_sha256']
        assert hashlib.sha256(raw).hexdigest() == row['uncompressed_sha256']
        assert len(raw) == row['uncompressed_size_bytes']
    history = read(B/'history/manifest.json')
    for row in history['copied_files']:
        data = (B/'history'/row['stored_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['stored_byte_sha256']
        if row['encoding'].startswith('gzip'):
            data = gzip.decompress(data)
        assert hashlib.sha256(data).hexdigest() == row['original_byte_sha256']
    protected = read(B/'fit/protected_hashes.json')
    assert len(protected) == 114
    for p,h in protected.items():
        assert sha(p) == h, p
    assert sha(ROOT/'src/grasp_failure_prediction/evaluation/constrained_surface_refinement.py') == sha(B/'implementation/constrained_refinement.py')
    assert sha(ROOT/'tests/test_constrained_surface_refinement.py') == sha(B/'implementation/test_constrained_surface_refinement.py')
    for p in B.rglob('*.py'):
        ast.parse(p.read_text())
    docs = [ROOT/'docs'/name for name in ('HUG_CONSTRAINED_REFINEMENT.md',
        'HUG_MAPPING_ACCEPTANCE.md','HUG_MAPPING_CONFIRMATION.md',
        'HUG_STATIC_VALIDATION_PROTOCOL.md','HUG_SURFACE_REFINEMENT.md',
        'PAPER_PROGRESS.md','RESEARCH_DECISIONS.md','THREAD_HANDOFF.md')]
    checked_links = 0
    for p in [d,B/'README.md']:
        for dest in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in dest and not dest.startswith('#'):
                target = (p.parent / dest.split('#',1)[0]).resolve()
                assert target.exists(), (str(p), dest)
                checked_links += 1
    write(B/'final_binding_verification.json', {
        'schema': 'hug_complete_bundle_and_docs_final_binding_verification_v1',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'all114_protected_original_file_hashes_unchanged': True,
        'installed_new_API_and_tests_byte_identical_to_frozen_sources': True,
        'selected_publication_files_verified': len(publication['files']),
        'canonical_gzip_copies_verified': len(compression),
        'historical_source_copies_verified': len(history['copied_files']),
        'local_primary_document_links_verified': checked_links,
        'python_sources_AST_parse': 'PASS',
        'changed_docs_sha256': {str(p.relative_to(ROOT)):sha(p) for p in docs},
        'case_count':6,'strict_resolution_count':9,'strict_FAIL_count':9,
        'returned_surface_corrections':0,'exact_original_fallback_count':6,
        'new_eval_data':False,'new_inference':False,'physics_steps':0,
        'production_mapper_or_manuscript_changed':False,
    })
    m['files_sha256'] = {str(p.relative_to(B)):sha(p) for p in sorted(B.rglob('*')) if p.is_file() and p.name != 'results_manifest.json'}
    write(B/'results_manifest.json',m)
    assert all(sha(B/name)==h for name,h in m['files_sha256'].items())
    print(json.dumps({'all_file_bindings':'PASS','bound_file_count':len(m['files_sha256']),
        'stored_bytes':sum(p.stat().st_size for p in B.rglob('*') if p.is_file()),
        'manifest_sha256':sha(B/'results_manifest.json'),'primary_doc':str(d)}))

if __name__ == '__main__':
    main()
