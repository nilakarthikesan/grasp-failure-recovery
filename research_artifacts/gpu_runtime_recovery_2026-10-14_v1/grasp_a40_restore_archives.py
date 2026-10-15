"""Restore verified backups into empty destinations; never run a model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

root = Path('/workspace/grasp-runtime-recovery')
archives = [
    (root / 'runtime_snapshot_a40_20261014_v1.tar.gz', Path('/opt'),
     '2524a50a6f4bf0533493097de27bd3b0255607df6dfc6c0f9ca5f6d87c9e51de'),
    (root / 'dga_native_smoke_assets.tar', Path('/workspace/dga'),
     '3b65bcd0c8d1e501063684fa52dd3c261ba4b663494056d0c7f868859f46408b'),
]
def digest(handle):
    h = hashlib.sha256()
    for block in iter(lambda: handle.read(8388608), b''):
        h.update(block)
    return h.hexdigest()

assert not (root / 'archive_restore_report.json').exists()
assert not Path('/opt/dga').exists()
assert not Path('/workspace/dga').exists()
for archive, destination, expected in archives:
    with archive.open('rb') as f:
        assert digest(f) == expected, 'Transferred archive hash mismatch'
    with tarfile.open(archive) as members:
        assert all(not x.name.startswith('/') and '..' not in Path(x.name).parts for x in members)

Path('/workspace/dga').mkdir()
rows = []
for archive, destination, expected in archives:
    subprocess.run(['tar', '--no-same-owner', '-xf', str(archive), '-C', str(destination)], check=True)
    files = 0
    links = 0
    with tarfile.open(archive) as members:
        for member in members:
            target = destination / member.name
            if member.isfile():
                with members.extractfile(member) as original, target.open('rb') as restored:
                    assert digest(original) == digest(restored), 'Restored file differs: ' + member.name
                files += 1
            elif member.issym():
                assert target.is_symlink() and os.readlink(target) == member.linkname
                links += 1
    rows.append({'archive': archive.name, 'sha256': expected,
                 'restore_root': str(destination), 'regular_files_verified': files,
                 'symlinks_verified': links})
report = {'status': 'PASS', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'scope': 'byte_preserving_backup_restore_only', 'archives': rows,
          'original_pod_untouched': True, 'new_inference_calls': 0,
          'new_eval_episodes': 0, 'training_runs': 0}
(root / 'archive_restore_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
