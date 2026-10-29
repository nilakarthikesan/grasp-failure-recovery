"""Validate one real HUG proposal against a native MANOSim hand, statically.

No integration, controller trial, outcome label, or training is performed.
Licensed hand assets remain external; the combined XML references their paths.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

# Existing optional runtime dependencies; base NumPy/MuJoCo are already loaded.
for _dependency in (Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps'),):
    if _dependency.is_dir() and str(_dependency) not in sys.path:
        sys.path.append(str(_dependency))

from grasp_failure_prediction.evaluation import manosim_pose as mp
from grasp_failure_prediction.integrations.scene_preparation import reload_saved_scene

RUNTIME = Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check')
DEFAULT_ASSETS = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/manosim_native_transfer_2026-10-15_v2/build_run_v1/raw')
SNAP = np.array([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3])
TIP_VERTICES = np.array([745, 317, 444, 556, 673])
TIP_BONES = np.array([15, 3, 6, 12, 9])
POSITION_LIMIT_M = 5e-6
ROTATION_COMPONENT_LIMIT = 1e-6


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _literal(value, shape, dtype, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.dtype(dtype) or not np.isfinite(a).all():
        raise ValueError(f'Expected finite literal {name}: {shape} {dtype}')
    return a


def axis_angle_rotations(aa):
    """Proper float64 Exp of original float32 source coordinates; no fitting."""
    a = _literal(aa, (48,), 'float32', 'postAA32').reshape(16, 3).astype(np.float64)
    matrices = []
    for vector in a:
        x, y, z = vector
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        theta2 = float(vector @ vector)
        if theta2 < 1e-12:
            s = 1 - theta2 / 6 + theta2 * theta2 / 120
            c = .5 - theta2 / 24 + theta2 * theta2 / 720
        else:
            theta = np.sqrt(theta2)
            s, c = np.sin(theta) / theta, (1 - np.cos(theta)) / theta2
        matrices.append(np.eye(3) + s * skew + c * (skew @ skew))
    return np.asarray(matrices)


def world_points(points, camera):
    return np.einsum('ij,nj->ni', camera[:3, :3], points, optimize=False) + camera[:3, 3]


def load_proposal(directory, scene):
    path = Path(directory) / 'proposal.npz'
    with np.load(path, allow_pickle=False) as z:
        values = {key: z[key].copy() for key in z.files}
    required = {'postAA32', 'translation32', 'shape32', 'source_transforms_abs32',
                'cameraWorld64', 'mesh_vertices', 'landmarks_3d', 'mesh_faces'}
    if not required <= set(values):
        raise ValueError('Proposal lacks native source fields: ' + ', '.join(sorted(required - set(values))))
    for key, shape, dtype in [('postAA32', (48,), 'float32'), ('translation32', (3,), 'float32'),
                              ('shape32', (1, 10), 'float32'), ('source_transforms_abs32', (16, 4, 4), 'float32'),
                              ('cameraWorld64', (4, 4), 'float64'), ('mesh_vertices', (778, 3), 'float32'),
                              ('landmarks_3d', (21, 3), 'float32')]:
        _literal(values[key], shape, dtype, key)
    camera = np.load(Path(scene) / 'T_world_camera.npy', allow_pickle=False)
    if not np.array_equal(values['cameraWorld64'], camera):
        raise ValueError('Proposal camera differs from captured scene camera')
    if not np.array_equal(camera[3], [0., 0., 0., 1.]):
        raise ValueError('Invalid camera homogeneous transform')
    if np.max(np.abs(camera[:3, :3].T @ camera[:3, :3] - np.eye(3))) > 1e-12:
        raise ValueError('Invalid camera rotation')
    if 'shape' in values and not np.array_equal(values['shape'].reshape(1, 10), values['shape32']):
        raise ValueError('Official proposal shape differs from source shape')
    if 't' in values and not np.array_equal(values['t'].reshape(3), values['translation32']):
        raise ValueError('Official proposal translation differs from source translation')
    values['proposal_sha256'] = _sha(path)
    return values


def source_reference(values, runtime=RUNTIME):
    """Independent actual licensed MANO forward; never load a HUG grasp model."""
    import torch
    # Load base NumPy/Torch before external dependency directories.
    for folder in reversed([runtime / 'hug-deps', runtime / 'deps']):
        if folder.is_dir() and str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    module = importlib.import_module('manotorch.manolayer')
    expected = runtime / 'hug-deps/manotorch/manolayer.py'
    if Path(module.__file__).resolve() != expected.resolve():
        raise ValueError('Unexpected MANO implementation source')
    assets = runtime / 'hug/assets/mano_models'
    layer = module.ManoLayer(rot_mode='axisang', side='right', center_idx=0,
                             use_pca=False, flat_hand_mean=True, mano_assets_root=str(assets)).cpu().float().eval()
    if bool(torch.count_nonzero(layer.th_hands_mean)):
        raise ValueError('Nonzero source hand mean is outside the approved flat-mean domain')
    torch.set_num_threads(2)
    with torch.no_grad():
        posed = layer(torch.from_numpy(values['postAA32'][None].copy()), torch.from_numpy(values['shape32'].copy()))
        neutral = layer(torch.zeros((1, 48), dtype=torch.float32), torch.from_numpy(values['shape32'].copy()))
    t = torch.from_numpy(values['translation32'].copy())
    # The source translation addition occurs in float32 before world64 mapping.
    vertices = (posed.verts[0] + t).detach().numpy()
    landmarks = (posed.joints[0] + t).detach().numpy()
    anchors = (posed.transforms_abs[0, :, :3, 3] + t).detach().numpy()
    transforms = posed.transforms_abs[0].detach().numpy()
    return dict(vertices=vertices, landmarks=landmarks, anchors=anchors, transforms=transforms,
                faces=layer.get_mano_closed_faces().detach().numpy(),
                neutral_vertices=neutral.verts[0].detach().numpy(),
                source_sha256=_sha(module.__file__), licensed_asset_sha256=_sha(assets / 'models/MANO_RIGHT.pkl'))


def select_assets(root, shape):
    root = Path(root).resolve()
    shape_path = root / 'assets/mano_rhand_shape.npy'
    original = np.load(shape_path, allow_pickle=False)
    if original.dtype != shape.dtype or original.shape != shape.shape or original.tobytes() != shape.tobytes():
        raise ValueError('Fresh proposal shape does not match existing literal MANOSim assets; a separately bound shape build is required')
    xml = root / 'assets/literal_native/mano_rhand/capsule_hand.xml'
    receipt = root / 'build_report.json'
    expected = _sha(shape_path)
    closure = mp.source_asset_closure(xml, shape_path, source_root=root,
        expected_shape_sha256=expected, builder_provenance={'path': str(receipt), 'sha256': _sha(receipt)})
    return xml, closure, expected


def assemble_scene(scene_directory, hand_xml):
    """Preserve cube scene; scope authored hand defaults and external meshes."""
    scene = ET.parse(Path(scene_directory) / 'scene.xml').getroot()
    for node in list(scene.findall('keyframe')):
        scene.remove(node)  # Old seven-qpos keyframes cannot describe the added hand.
    # Respect explicit hand inertias rather than the fixture's geom inference.
    scene.find('compiler').set('inertiafromgeom', 'auto')
    hand = ET.parse(hand_xml).getroot()
    defaults = scene.find('default')
    if defaults is None:
        defaults = ET.SubElement(scene, 'default')
    named = ET.SubElement(defaults, 'default', {'class': 'phase1_native_hand'})
    for child in hand.find('default'):
        named.append(deepcopy(child))
    asset = scene.find('asset')
    if asset is None:
        asset = ET.Element('asset')
        scene.insert(1, asset)
    for mesh in hand.find('asset'):
        copied = deepcopy(mesh)
        copied.set('file', str((Path(hand_xml).parent / copied.attrib['file']).resolve()))
        asset.append(copied)
    for body in hand.find('worldbody'):
        copied = deepcopy(body)
        copied.set('childclass', 'phase1_native_hand')
        scene.find('worldbody').append(copied)
    for section in ('actuator', 'contact', 'tendon', 'equality'):
        child = hand.find(section)
        if child is not None:
            target = scene.find(section)
            if target is None:
                target = ET.SubElement(scene, section)
            for item in child:
                target.append(deepcopy(item))
    return ET.tostring(scene, encoding='unicode')


def _id(model, kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError('Missing compiled scene element: ' + name)
    return result


def mapping_metrics(state, values, reference):
    camera = values['cameraWorld64']
    source_anchors = world_points(reference['anchors'], camera)
    source_rotations = np.einsum('ij,njk->nik', camera[:3, :3], reference['transforms'][:, :3, :3], optimize=False)
    anchor_error = np.linalg.norm(state.joint_anchor_world_m - source_anchors, axis=1)
    rotation_error = np.max(np.abs(state.world_body_rotations - source_rotations), axis=(1, 2))
    return {
        'source_vertex_reconstruction_max_m': float(np.linalg.norm(reference['vertices'] - values['mesh_vertices'], axis=1).max()),
        'source_landmark_reconstruction_max_m': float(np.linalg.norm(reference['landmarks'] - values['landmarks_3d'], axis=1).max()),
        'exported_source_transforms_max_component_error': float(np.max(np.abs(reference['transforms'] - values['source_transforms_abs32']))),
        'face_table_exact': bool(np.array_equal(reference['faces'], values['mesh_faces'])),
        'bone_anchor_errors_m': anchor_error.tolist(), 'bone_rotation_component_errors': rotation_error.tolist(),
        'max_bone_anchor_error_m': float(anchor_error.max()), 'max_bone_rotation_component_error': float(rotation_error.max()),
        'saved_landmark_anchor_max_m': float(np.linalg.norm(state.joint_anchor_world_m - world_points(values['landmarks_3d'][SNAP], camera), axis=1).max()),
        'independent_compiled_fk': state.independent_readback,
    }


def validate_native(scene_directory, proposal_directory, output_directory, *, assets_root=DEFAULT_ASSETS, runtime=RUNTIME, render=True):
    scene, proposal, output = map(lambda p: Path(p).resolve(), (scene_directory, proposal_directory, output_directory))
    if output.exists():
        raise FileExistsError('Native validation output must be a fresh directory')
    values = load_proposal(proposal, scene)
    hand_xml, closure, expected = select_assets(assets_root, values['shape32'])
    reference = source_reference(values, Path(runtime))
    old_model, old_data = reload_saved_scene(scene)
    xml = assemble_scene(scene, hand_xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    old_joint = _id(old_model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_free')
    joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_free')
    qa, va = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    oq, ov = int(old_model.jnt_qposadr[old_joint]), int(old_model.jnt_dofadr[old_joint])
    data.qpos[qa:qa+7] = old_data.qpos[oq:oq+7]
    data.qvel[va:va+6] = old_data.qvel[ov:ov+6]
    adapter = mp.ManoSimStaticPoseAdapter(model, data, source_assets=closure,
        asset_shape_identity=closure['asset_shape_identity'], expected_shape_sha256=expected)
    rotations = axis_angle_rotations(values['postAA32'])
    wrist = np.eye(4)
    wrist[:3, :3] = values['cameraWorld64'][:3, :3] @ rotations[0]
    wrist[:3, 3] = world_points(values['translation32'][None].astype(np.float64), values['cameraWorld64'])[0]
    state = adapter.stage(wrist, {f'mano_joint_{i}': rotations[i] for i in range(1, 16)}, asset_shape_identity=closure['asset_shape_identity'])
    metrics = mapping_metrics(state, values, reference)
    old_cube = _id(old_model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
    new_cube = _id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
    cube_preserved = bool(old_model.body_mass[old_cube] == model.body_mass[new_cube]
        and np.array_equal(old_model.body_inertia[old_cube], model.body_inertia[new_cube])
        and np.array_equal(adapter.data.qpos[qa:qa+7], old_data.qpos[oq:oq+7])
        and np.array_equal(adapter.data.qvel[va:va+6], old_data.qvel[ov:ov+6]))
    metrics['captured_cube_preserved'] = cube_preserved
    metrics['cube_mass_kg'] = float(model.body_mass[new_cube])
    metrics['cube_diagonal_inertia_kg_m2'] = model.body_inertia[new_cube].tolist()
    passed = bool(cube_preserved and metrics['face_table_exact'] and metrics['source_vertex_reconstruction_max_m'] <= POSITION_LIMIT_M
        and metrics['source_landmark_reconstruction_max_m'] <= POSITION_LIMIT_M
        and metrics['exported_source_transforms_max_component_error'] <= ROTATION_COMPONENT_LIMIT
        and metrics['max_bone_anchor_error_m'] <= POSITION_LIMIT_M
        and metrics['saved_landmark_anchor_max_m'] <= POSITION_LIMIT_M
        and metrics['max_bone_rotation_component_error'] <= ROTATION_COMPONENT_LIMIT)
    # Rigid distal markers intentionally differ from posed/skinned fingertips.
    neutral_rots = np.repeat(np.eye(3)[None], 16, axis=0)
    neutral = adapter.stage(np.eye(4), {f'mano_joint_{i}': neutral_rots[i] for i in range(1, 16)}, asset_shape_identity=closure['asset_shape_identity'])
    local_tips = np.einsum('nji,nj->ni', neutral.world_body_rotations[TIP_BONES],
                           reference['neutral_vertices'][TIP_VERTICES] - neutral.body_origin_world_m[TIP_BONES], optimize=False)
    rigid_tips = np.einsum('nij,nj->ni', state.world_body_rotations[TIP_BONES], local_tips, optimize=False) + state.body_origin_world_m[TIP_BONES]
    source_tips = world_points(values['landmarks_3d'][[4, 8, 12, 16, 20]], values['cameraWorld64'])
    adapter.stage_serialized(state.serialize_pose())
    output.mkdir(parents=True)
    (output / 'scene.xml').write_text(xml)
    np.savez_compressed(output / 'pose.npz', qpos=adapter.data.qpos, qvel=adapter.data.qvel,
        bone_anchors_world=state.joint_anchor_world_m, source_anchors_world=world_points(reference['anchors'], values['cameraWorld64']),
        rigid_tips_world=rigid_tips, source_tips_world=source_tips, wrist_world=wrist)
    _write(output / 'named_pose.json', state.serialize_pose())
    image_name = None
    if render:
        config = json.loads((scene / 'scene_config.json').read_text())
        with mujoco.Renderer(model, height=config['camera']['height'], width=config['camera']['width']) as renderer:
            renderer.update_scene(adapter.data, camera='capture')
            rgb = renderer.render().copy()
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        k = np.load(scene / 'intrinsics.npy', allow_pickle=False)
        camera = values['cameraWorld64']
        def pixels(points):
            p = np.einsum('ji,nj->ni', camera[:3, :3], points - camera[:3, 3], optimize=False)
            uv = p @ k.T
            return uv[:, :2] / uv[:, 2:3]
        for p in pixels(source_tips):
            if np.isfinite(p).all():
                x, y = p
                draw.ellipse((x-2, y-2, x+2, y+2), outline=(0, 255, 255), width=1)
        for p in pixels(rigid_tips):
            if np.isfinite(p).all():
                x, y = p
                draw.line((x-2, y, x+2, y), fill=(255, 255, 0), width=1)
                draw.line((x, y-2, x, y+2), fill=(255, 255, 0), width=1)
        image.save(output / 'mapping.png')
        image_name = str(output / 'mapping.png')
    report = {'schema_version': 'phase1_native_static_validation_v1', 'mapping_checks_passed': passed,
        'scope': 'Static numerical MANO-to-MANOSim mapping, not physical/contact fidelity or a grasp outcome',
        'physics_integration_steps': 0, 'grasp_trials_executed': 0, 'prediction_or_training_performed': False,
        'proposal_sha256': values['proposal_sha256'], 'scene_xml_sha256': _sha(scene / 'scene.xml'),
        'hand_source_closure_sha256': closure['closure_sha256'], 'shape_asset_sha256': expected,
        'source_mano_code_sha256': reference['source_sha256'], 'licensed_asset_sha256': reference['licensed_asset_sha256'],
        'captured_scene_time_s': float(old_data.time), 'static_analysis_time_s': float(adapter.data.time),
        'position_numerical_limit_m': POSITION_LIMIT_M, 'rotation_component_numerical_limit': ROTATION_COMPONENT_LIMIT,
        'checks': metrics, 'fingertip_rigid_to_skinned_residuals_m': np.linalg.norm(rigid_tips-source_tips, axis=1).tolist(),
        'fingertip_order': ['thumb', 'index', 'middle', 'ring', 'little'],
        'geometry_approximation': 'Authored MANOSim phalanx capsules and convex segmented palm are collision approximations. Rigid distal markers do not reproduce pose-dependent MANO skinning. Fingertip residuals are diagnostics, not mapping admission thresholds.',
        'mapping_image': image_name, 'image_legend': 'cyan circles: original HUG skinned tips; yellow crosses: rigid MANOSim distal markers',
        'mujoco_version': mujoco.__version__, 'outputs': {p.name: _sha(p) for p in output.iterdir() if p.is_file()}}
    _write(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--proposal', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--assets-root', default=DEFAULT_ASSETS, type=Path)
    parser.add_argument('--runtime', default=RUNTIME, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_native(args.scene, args.proposal, args.output, assets_root=args.assets_root, runtime=args.runtime), indent=2))


if __name__ == '__main__':
    main()
