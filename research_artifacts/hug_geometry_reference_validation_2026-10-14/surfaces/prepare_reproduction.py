from pathlib import Path
import sys, gzip, shutil, hashlib
bundle, target = map(Path, sys.argv[1:3])
assert not target.exists(), f'Reproduction output already exists: {target}'
reference = bundle.parent / 'reference'
protected = bundle.parent / 'protected_input_sha256.json'
assert reference.is_dir() and protected.is_file(), 'Run from published parent bundle'
fixed = Path('/private/tmp/hug_current_stage_protected_2026-10-14.json')
if fixed.exists():
    assert fixed.read_bytes() == protected.read_bytes(), 'Protected snapshot differs'
else:
    shutil.copyfile(protected, fixed)
raw = gzip.decompress((reference / 'hug_reference_comparison_full.json.gz').read_bytes())
assert hashlib.sha256(raw).hexdigest() == '0b7a2df411b9763f29c9cda6a5a252417716869dc5728de90369ef5f632738f6'
(target / 'reference').mkdir(parents=True)
(target / 'reference/hug_reference_comparison.json').write_bytes(raw)
shutil.copyfile(reference / 'protocol.json', target / 'reference/protocol.json')
for name in ('compare_surfaces.py', 'protocol_v3.json'):
    shutil.copyfile(bundle / name, target / name)
print(target)
