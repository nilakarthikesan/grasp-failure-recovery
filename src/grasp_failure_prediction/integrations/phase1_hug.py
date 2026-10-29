"""Bounded fresh official HUG inference on an immutable approved scene capture.

This local CPU stage generates three genuine proposals; it does not execute a
hand, retarget joints, train a model or establish physical grasp success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import types

import numpy as np
from PIL import Image

from .hug import normalize_hug_prediction
from .hug_preprocessing import seeded_point_cloud, tensor_content_hash
from .hug_rng import seed_cpu_generation
from .observations import check_manifest, validate_depth_encoding

HUG_REVISION = '8d1c52d4c24bfae5a369e32e3f134f5601a02630'
CHECKPOINT_SHA256 = '515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae'
SEEDS = (0, 1, 2)
STEPS = 50
POINT_CLOUD_SEED = 42


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def save_arrays(path, arrays):
    """Write an array-only archive that loads with allow_pickle=False."""
    clean = {name: np.asarray(value) for name, value in arrays.items()}
    for name, value in clean.items():
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('nonfinite or unsafe output array: ' + name)
    np.savez_compressed(path, **clean)


def source_files(scene):
    return {path.name: sha(path) for path in sorted(Path(scene).iterdir()) if path.is_file()}


def require_fixed_protocol(seeds, steps):
    if tuple(seeds) != SEEDS or type(steps) is not int or steps != STEPS:
        raise ValueError('approved protocol requires seeds [0,1,2], 50 steps and point-cloud seed 42')


def validate_capture(scene):
    """Bind a completed cube capture before any model loading or output writes."""
    scene = Path(scene).resolve()
    report = json.loads((scene / 'validation_report.json').read_text())
    if report.get('scene_preparation_checks_passed') is not True:
        raise ValueError('completed scene preparation report required')
    for name, digest in report['artifact_sha256'].items():
        if Path(name).name != name or sha(scene / name) != digest:
            raise ValueError('approved scene artifact changed: ' + name)
    manifest = json.loads((scene / 'observation_manifest.json').read_text())
    for key in ('rgb_path', 'depth_m_path', 'intrinsics_path', 'mask_path'):
        if Path(manifest[key]).is_absolute() or not (scene / manifest[key]).resolve().is_relative_to(scene):
            raise ValueError('observation path must remain inside scene capture')
    rgb = np.asarray(Image.open(scene / manifest['rgb_path']).convert('RGB'))
    depth = np.load(scene / manifest['depth_m_path'], allow_pickle=False)
    encoded = np.asarray(Image.open(scene / 'depth.png'))
    intrinsic = np.load(scene / manifest['intrinsics_path'], allow_pickle=False)
    camera = np.load(scene / 'T_world_camera.npy', allow_pickle=False)
    mask = np.asarray(Image.open(scene / manifest['mask_path']).convert('L'))
    if rgb.shape != (224, 224, 3) or encoded.shape != (224, 224):
        raise ValueError('this approved HUG stage requires the original 224x224 capture')
    validate_depth_encoding(depth, encoded)
    u, v = manifest['selection_uv']
    if (type(u) is not int or type(v) is not int or not (0 <= u < 224 and 0 <= v < 224)
            or mask[v, u] == 0 or encoded[v, u] == 0):
        raise ValueError('stored selected point must have valid object depth; no silent fallback')
    if (camera.shape != (4, 4) or camera.dtype != np.float64 or not np.isfinite(camera).all()
            or not np.allclose(camera[3], [0, 0, 0, 1])
            or not np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-10)
            or not np.isclose(np.linalg.det(camera[:3, :3]), 1.)):
        raise ValueError('original float64 proper camera-to-world transform required')
    return manifest, rgb, encoded, intrinsic, camera


def export_proposal(path, grasp, *, prediction, captured, shape, camera):
    normalize_hug_prediction(grasp)
    aa = captured['postAA32']
    transforms = captured['source_transforms_abs32']
    translation = np.asarray(grasp['t']).reshape(3).copy()
    for value, expected in ((aa, (48,)), (transforms, (16, 4, 4)),
                            (translation, (3,)), (shape, (1, 10))):
        if value.dtype != np.float32 or value.shape != expected or not np.isfinite(value).all():
            raise ValueError('native source output must retain literal float32 shape and bytes')
    arrays = {**grasp, 'prediction99': prediction, 'postAA32': aa,
              'translation32': translation, 'shape32': shape,
              'source_transforms_abs32': transforms, 'cameraWorld64': camera}
    save_arrays(path / 'proposal.npz', arrays)
    # This pickle is produced locally from the actual upstream output; never consumed as an external input.
    (path / 'proposal.pkl').write_bytes(pickle.dumps(grasp))
    return {'proposal_sha256': sha(path / 'proposal.pkl'),
            'proposal_npz_sha256': sha(path / 'proposal.npz')}


def generate(scene, output, hug_root, checkpoint, *, dependency_dirs=(), seeds=SEEDS, steps=STEPS):
    require_fixed_protocol(seeds, steps)
    scene, output, hug_root, checkpoint = (Path(path).resolve() for path in (scene, output, hug_root, checkpoint))
    if output.exists():
        raise FileExistsError('HUG output must be a fresh directory')
    manifest, rgb, depth_mm, intrinsic, camera = validate_capture(scene)
    initial_scene_hashes = source_files(scene)
    revision = subprocess.check_output(['git', '-C', str(hug_root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != HUG_REVISION or subprocess.check_output(['git', '-C', str(hug_root), 'status', '--porcelain'], text=True).strip():
        raise ValueError('clean pinned official HUG source required')
    if sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('pinned official EMA checkpoint differs')
    cache = hug_root.parent / 'hf_cache'
    if not cache.is_dir():
        raise ValueError('existing offline DINO cache required')
    os.environ.update(HF_HOME=str(cache), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for directory in reversed(tuple(dependency_dirs)):
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise ValueError('dependency directory absent: ' + str(directory))
        sys.path.insert(0, str(directory))
    import torch
    import transformers
    import torch_cluster
    import manotorch
    from safetensors import safe_open
    package = types.ModuleType('hug')
    package.__path__ = [str(hug_root / 'src')]
    sys.modules['hug'] = package
    from hug.prepare_inputs import prepare_pkl
    from hug.dataloader.grasp_dataset import GraspDataset
    from hug.inference import load_model, load_raw_checkpoint
    from hug.models.mano import mano_params_to_grasp_dict
    from hug.utils import pcl_utils
    output.mkdir(parents=True, exist_ok=False)
    events = output / 'events.jsonl'
    def event(kind, **fields):
        with events.open('a') as stream:
            stream.write(json.dumps({'event': kind, **fields}, sort_keys=True, allow_nan=False) + '\n')
    event('fresh_inference_started', seeds=list(seeds), steps=steps)
    checks = check_manifest(scene / 'observation_manifest.json', output / 'observation_checks')
    if not checks['preflight_passed']:
        raise ValueError(str(checks['errors']))
    prepared = prepare_pkl(rgb, depth_mm, intrinsic, 'cube', output / 'inputs', object_name='phase1_synthetic_cube')
    entry = pickle.loads(prepared.read_bytes())
    entry['object_mask'] = (scene / manifest['mask_path']).read_bytes()
    entry['condition_point'] = np.asarray(manifest['selection_uv'], dtype=np.float32)
    prepared.write_bytes(pickle.dumps(entry))
    class FixedDataset(GraspDataset):
        def _build_pcl(self, depth_m, rgb_np, camera_K, point_xyz=None):
            xyz, colors = seeded_point_cloud(depth_m.numpy(), rgb_np, camera_K,
                seed=POINT_CLOUD_SEED, backproject=pcl_utils.backproject_to_pcl,
                sample=pcl_utils.sample_fixed_n, center=point_xyz,
                crop_radius=self.pcl_crop_radius, n_points=self.n_points_input)
            return torch.from_numpy(xyz), torch.from_numpy(colors)
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = load_model(checkpoint, use_ema=True, device='cpu').float().eval()
    raw = load_raw_checkpoint(checkpoint, 'cpu')
    state = raw.get('ema')
    if state is None:
        state = raw['model']
    state = {key.removeprefix('module.'): value for key, value in state.items() if key != 'n_averaged'}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(not key.startswith(('image_encoder.model.', 'mano.mano_layer.')) for key in incompatible.missing_keys):
        raise ValueError('released checkpoint has missing/unexpected learned parameters')
    if model.mano.mano_layer.th_hands_mean.dtype != torch.float32 or bool(torch.any(model.mano.mano_layer.th_hands_mean != 0)):
        raise ValueError('literal flat zero hand mean required')
    batch = FixedDataset(str(output / 'inputs'), split='eval', pcl_crop_radius=model.pcl_crop_radius)[0]
    tensor_hash = tensor_content_hash({key: value for key, value in batch.items() if torch.is_tensor(value)})
    shape = model.fixed_betas.detach().cpu().numpy().reshape(1, 10).copy()
    bindings = {str(path): sha(path) for path in sorted((hug_root / 'src').rglob('*.py'))}
    for path in [checkpoint, Path(__file__), hug_root / 'assets/mano_rhand_shape.npy',
                 hug_root / 'assets/mano_models/models/MANO_RIGHT.pkl', hug_root / 'assets/mano_rhand_mesh_faces.npy']:
        bindings[str(path)] = sha(path)
    dinofiles = [path for path in sorted((cache / 'hub').rglob('*'))
                if path.is_file() and '/snapshots/' in str(path) and path.name in ('config.json', 'model.safetensors')]
    for path in dinofiles:
        bindings[str(path)] = sha(path)
    runtime_modules = (np, torch, transformers, torch_cluster, manotorch)
    runtime = {module.__name__: {'version': getattr(module, '__version__', None),
                                'source_path': module.__file__, 'source_sha256': sha(module.__file__)}
               for module in runtime_modules}
    runtime['platform'] = platform.platform()
    with safe_open(str(checkpoint), framework='pt') as archive:
        weights_kind = json.loads(archive.metadata().get('weights_kind', '"model"'))
    protocol = {'schema_version': 'phase1_fresh_hug_v1', 'scene': str(scene),
                'scene_hashes': initial_scene_hashes, 'hug_revision': revision,
                'checkpoint_sha256': CHECKPOINT_SHA256, 'weights_kind': weights_kind,
                'seeds': list(seeds), 'sampling_steps': steps, 'point_cloud_seed': POINT_CLOUD_SEED,
                'device': 'cpu', 'dtype': 'float32', 'batch_size': 1, 'runtime': runtime,
                'input_tensor_hash': tensor_hash, 'prepared_input_sha256': sha(prepared),
                'bindings': bindings, 'offline_only': True, 'no_retarg_or_physics': True}
    write_json(output / 'generation_protocol.json', protocol)
    # Supply neutral source geometry for the exact generated beta, without retargeting.
    with torch.inference_mode():
        neutral = model.mano.mano_layer(torch.zeros((1, 48)), model.fixed_betas)
    save_arrays(output / 'neutral_source.npz', {'shape32': shape,
        'vertices32': neutral.verts[0].detach().cpu().numpy(),
        'joints32': neutral.joints[0].detach().cpu().numpy(),
        'source_transforms_abs32': neutral.transforms_abs[0].detach().cpu().numpy(),
        'skinning_weights32': model.mano.mano_layer.th_weights.detach().cpu().numpy(),
        'mesh_faces': model.mesh_faces,
        'parents': np.asarray(model.mano.mano_layer.kintree_parents, dtype=np.int64)})
    captured = {}
    def hook(module, arguments, result):
        captured['postAA32'] = result.full_poses.detach().cpu().numpy()[0].copy()
        captured['source_transforms_abs32'] = result.transforms_abs.detach().cpu().numpy()[0].copy()
    handle = model.mano.mano_layer.register_forward_hook(hook)
    rows = []
    try:
        for seed in seeds:
            directory = output / f'seed_{seed:03d}'
            directory.mkdir(exist_ok=False)
            sample_start = time.perf_counter()
            try:
                seed_cpu_generation(seed, torch_module=torch)
                with torch.inference_mode():
                    prediction = model.sample(batch['point_uv'][None], batch['camera_K'][None],
                        steps=steps, rgb=batch['rgb'][None], pcl_xyz=batch['pcl_xyz'][None],
                        pcl_rgb=batch['pcl_rgb'][None])[0]
                    captured.clear()
                    grasp = mano_params_to_grasp_dict(prediction, model.fixed_betas.squeeze(0),
                                                       model.mano, intrinsic, model.mesh_faces)
                receipt = export_proposal(directory, grasp,
                    prediction=prediction.detach().cpu().numpy().copy(), captured=captured,
                    shape=shape, camera=camera)
                row = {'seed': seed, 'real_hug_inference_passed': True,
                       'device': 'cpu', 'dtype': 'float32', 'sampling_steps': steps,
                       'point_cloud_seed': POINT_CLOUD_SEED, 'input_tensor_hash': tensor_hash,
                       'generation_rng': 'torch.manual_seed and libc.srand immediately before batch-size-one model.sample',
                       'elapsed_inference_s': time.perf_counter() - sample_start,
                       'source_scene': str(scene), 'scene_manifest_sha256': initial_scene_hashes['observation_manifest.json'],
                       'checkpoint_sha256': CHECKPOINT_SHA256, 'hug_code_commit': revision,
                       'generation_protocol_sha256': sha(output / 'generation_protocol.json'),
                       'source_frames': 'camera xyz metres; post-mean literal MANO AA32; cameraWorld64 is camera-to-world',
                       'retargeting_executed': False, 'simulation_execution_tested': False, **receipt}
            except Exception as exc:
                row = {'seed': seed, 'real_hug_inference_passed': False,
                       'error_type': type(exc).__name__, 'error': str(exc)}
            write_json(directory / 'inference_report.json', row)
            rows.append(row)
            event('proposal_completed', **row)
            print(json.dumps({'seed': seed, 'real_hug_inference_passed': row['real_hug_inference_passed'],
                              'output': str(directory)}), flush=True)
    finally:
        handle.remove()
    unchanged = source_files(scene) == initial_scene_hashes
    report = {'schema_version': 'phase1_fresh_hug_report_v1', 'proposals': rows,
              'real_hug_inference_passed': len(rows) == 3 and all(row['real_hug_inference_passed'] for row in rows),
              'scene_unchanged': unchanged, 'elapsed_total_s': time.perf_counter() - started,
              'generation_protocol_sha256': sha(output / 'generation_protocol.json'),
              'scope': 'Three bounded actual official HUG samples; no retargeting, physics, training or grasp-success claim.'}
    write_json(output / 'inference_report.json', report)
    if not unchanged:
        raise ValueError('approved scene capture changed during inference')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--hug-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dependency-dir', action='append', default=[])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--steps', type=int, default=STEPS)
    args = parser.parse_args(argv)
    try:
        report = generate(args.scene, args.output, args.hug_root, args.checkpoint,
                          dependency_dirs=args.dependency_dir, seeds=args.seeds, steps=args.steps)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        print('Fresh HUG inference rejected: ' + str(exc), flush=True)
        return 1
    return 0 if report['real_hug_inference_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
