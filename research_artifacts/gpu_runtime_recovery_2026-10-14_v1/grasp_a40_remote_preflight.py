"""Runtime-only recovery check: no generator, checkpoint load, or physics."""
import hashlib
import importlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

class ProhibitedImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'isaacgym', 'mujoco', 'pybullet', 'models', 'datasets'}:
            raise ImportError('Runtime recovery prohibits this import: ' + fullname)

sys.meta_path.insert(0, ProhibitedImports())
root = Path('/workspace/dga')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()

def source_hashes():
    return {str(p.relative_to(root / 'source')): digest(p)
            for p in sorted((root / 'source').rglob('*'))
            if p.is_file() and '.git' not in p.relative_to(root / 'source').parts}

before = source_hashes()
modules = ['torch', 'torchvision', 'numpy', 'scipy', 'hydra', 'omegaconf',
           'pytorch_kinematics', 'pointops_cuda', 'pointnet2._ext', 'csdf', 'pytorch3d._C']
import torch
load_calls = []
def deny_load(*args, **kwargs):
    load_calls.append(True)
    raise RuntimeError('Checkpoint deserialization prohibited during recovery')
torch.load = deny_load
torch.jit.load = deny_load
loaded = {name: importlib.import_module(name) for name in modules}
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 1
assert 'A40' in torch.cuda.get_device_name(0)
left = torch.tensor([[1., 2.], [3., 4.]], device='cuda')
product = left @ left
torch.cuda.synchronize()
expected = torch.tensor([[7., 10.], [15., 22.]])
assert torch.equal(product.cpu(), expected), 'Fixed CUDA arithmetic mismatch'
expected_inputs = {
    'DexGraspNet.pth': 'f7675d8eb93b3978b2c2d72ec9f54ead6b936d1dd307e07605daa0c02bcac14f',
    'input/manifest.json': 'e00f34eb513de1812ad2ab163fc7cb5278e84e88d8c64bb35c3fb160147b4854',
    'input/analytic_geometry_contract.json': '45c1162a4f815d65c43254f2f499f53bde1eecea4d39912d8ce36d2c5cd92a81',
    'input/mesh.obj': 'e3abe7b101ecc6ebe02e38931674fd7991e445b752dca0d41f51eeaef148cbac',
    'input/points.npy': '03d9c6ffd06868427010d5dbc6b5db0edd848f9d62bd0209a970c328554f1d71',
    'input/normals.npy': '20bb8a503d8c0cebf3bf1537328fb08a26577fdd5daa0d7106d548833c681bb4',
}
actual_inputs = {name: digest(root / name) for name in expected_inputs}
assert actual_inputs == expected_inputs, 'Restored input byte mismatch'
after = source_hashes()
assert before == after, 'Protected source changed during import check'
assert not load_calls
head = subprocess.check_output(['git', '-C', str(root / 'source'), 'rev-parse', 'HEAD'], text=True).strip()
assert head == '86353d710ca71d0f547f0d8b15cc8ca959c78163'
check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
assert check.returncode == 0, check.stdout + check.stderr
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
report = {
    'status': 'PASS', 'scope': 'runtime_restore_and_fixed_cuda_arithmetic_only',
    'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'python': sys.version, 'executable': sys.executable,
    'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
    'gpu': gpu, 'compute_capability': list(torch.cuda.get_device_capability(0)),
    'fixed_cuda_product': product.cpu().tolist(),
    'module_origins': {name: getattr(module, '__file__', None) for name, module in loaded.items()},
    'source_commit': head, 'source_files_checked': len(before), 'source_unchanged': before == after,
    'input_hashes': actual_inputs, 'pip_check': check.stdout.strip(),
    'checkpoint_deserialization_calls': len(load_calls),
    'generator_imports_prohibited': True, 'physics_imports_prohibited': True,
    'new_inference_calls': 0, 'new_eval_episodes': 0, 'training_runs': 0,
    'preflight_script_sha256': digest(Path(__file__)),
}
out = Path('/workspace/grasp-runtime-recovery/preflight_report.json')
assert not out.exists(), 'Refusing to overwrite a previous recovery report'
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
