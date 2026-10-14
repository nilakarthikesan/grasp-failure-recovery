"""Install only the two new isolated API files; preserve every protected byte."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
protected = json.loads((HERE / 'protected_hashes.json').read_text())
for path, expected in protected.items():
    assert sha(path) == expected, path
pairs = [
    ('constrained_refinement.py', 'src/grasp_failure_prediction/evaluation/constrained_surface_refinement.py',
     '0b29f867337cd11bb9128f5aa6ca92800338543055b286b40873d10f4d74f409'),
    ('test_constrained_surface_refinement.py', 'tests/test_constrained_surface_refinement.py',
     'fabb3db393cf8becafc1ccf8ee24a86b7a068c0dc1c59b816bb11bce00dfb612'),
]
for source, target, expected in pairs:
    assert sha(HERE / source) == expected
    destination = ROOT / target
    assert not destination.exists(), destination
for source, target, expected in pairs:
    destination = ROOT / target
    destination.write_bytes((HERE / source).read_bytes())
    assert sha(destination) == expected
for path, expected in protected.items():
    assert sha(path) == expected, path
report = {'new_api_and_test_installed_byte_identical': True,
          'protected_file_count': len(protected), 'protected_hashes_unchanged': True,
          'default_mapper_changed': False, 'physics_steps': 0, 'new_eval_data': False,
          'files': [{'path': target, 'sha256': expected} for _, target, expected in pairs]}
(HERE / 'api_installation_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
