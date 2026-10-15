"""Archive this study and its hash-bound inputs without changing old evidence."""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

ROOT = Path('/private/tmp/dga_mapping_acceptance_20261015_v2')
WORK = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
DEST = WORK/'research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_acceptance_feasibility_v2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return 'sha256:'+h.hexdigest()


def main():
    assert not DEST.exists(), 'Never replace an existing study'
    shutil.copytree(ROOT, DEST, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    deps = {}
    for p in ROOT.rglob('*.json'):
        x = json.loads(p.read_text())
        if not isinstance(x, dict): continue
        for key in ['dependency_sha256', 'dependencies_sha256']:
            d = x.get(key)
            if not isinstance(d, dict): continue
            for path, digest in d.items():
                if isinstance(digest, str) and Path(path).is_file():
                    deps.setdefault((path, digest), []).append(str(p.relative_to(ROOT)))
    blobs = DEST/'dependency_blobs'; blobs.mkdir()
    relocation = []
    unavailable = []
    local_hash_index = {sha(p):p for p in ROOT.rglob('*') if p.is_file()
                        and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    for (path, digest), referenced_by in sorted(deps.items()):
        source = Path(path)
        if sha(path) != digest:
            source = local_hash_index.get(digest)
            if source is None:
                unavailable.append({'original_path':path,'expected_sha256':digest,
                                    'referenced_by':referenced_by,
                                    'scope':'Superseded historical dependency version not present; not used to substantiate current conclusions.'})
                continue
        if source.is_relative_to(ROOT):
            target = DEST/source.relative_to(ROOT)
        else:
            target = blobs/digest.removeprefix('sha256:')
            if not target.exists(): shutil.copyfile(source, target)
        if sha(target) != digest: raise ValueError('Archive changed bytes: '+path)
        relocation.append({'original_path':path,'archive_path':str(target.relative_to(DEST)),
                           'sha256':digest,'bytes':source.stat().st_size,
                           'referenced_by':referenced_by})
    (DEST/'dependency_archive.json').write_text(json.dumps({
        'schema_version':'dga_static_acceptance_input_archive_v1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'original_absolute_paths_preserved_in_historical_policies':True,
        'relocation_purpose':'Hash-bound byte-identical dependencies; runtime environment still requires matching packages. Policies are not rewritten.',
        'dependency_path_count':len(relocation),
        'deduplicated_external_blob_count':len(list(blobs.iterdir())),
        'files':relocation,'superseded_unavailable_versions':unavailable},indent=2)+'\n')
    print(json.dumps({'destination':str(DEST),'dependencies':len(relocation),
                      'blobs':len(list(blobs.iterdir())),
                      'superseded_unavailable_versions':len(unavailable)}))


if __name__ == '__main__': main()
