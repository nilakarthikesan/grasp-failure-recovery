"""Saved-array reconstruction audit. No engine, MANO, FK or physics imports.

This verifies the durable evidence, not replayability of a lost temp closure.
Run with the project Python and --output pointing to a fresh JSON path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'research_artifacts/manosim_native_source_pipeline_2026-10-15_v1'
RECEIPT_SHA = '9a5d3a551a29d45049aa360dd1012fab86c32a97f45b38c4477be77b32d5a13e'
PROTOCOL_SHA = '92b5f9bc2552b7a56cc0698aa981ae565f1701d665395095ae55ec6558c2ec9b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normal(vertices, faces):
    tri = vertices[:, faces]
    n = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
    lengths = np.linalg.norm(n, axis=-1)
    if not np.isfinite(lengths).all() or np.any(lengths == 0):
        raise ValueError('Invalid saved triangle')
    return n / lengths[..., None]


def distance(a, b):
    return float(np.max(np.linalg.norm(a - b, axis=-1)))


def audit():
    if sys.flags.optimize:
        raise RuntimeError('This assertion-based evidence audit must run without Python optimization')
    receipt_path = PRIOR / 'run_v1/independent_readback.json'
    protocol_path = PRIOR / 'protocol.json'
    assert sha(receipt_path) == RECEIPT_SHA
    assert sha(protocol_path) == PROTOCOL_SHA
    receipt = json.loads(receipt_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    checked = {}
    for path, record in receipt['artifact_inventory'].items():
        p = Path(path)
        assert p.is_file() and p.stat().st_size == record['bytes']
        assert sha(p) == record['sha256']
        checked[path] = record
    missing = sorted(path for path in protocol['bindings'] if not Path(path).is_file())
    cases = []
    for case in (20, 28):
        raw = ROOT / f'runs/manosim_native_source_pipeline_2026-10-15_v1/run_v1/raw/{case}'
        domain_path = raw / 'source_initialized_domain.npz'
        fields_path = raw / 'all109_native_source_engine_readbacks.npz'
        for path in (domain_path, fields_path):
            assert str(path) in checked
        with np.load(domain_path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        with np.load(fields_path, allow_pickle=False) as z:
            f = {k: z[k] for k in z.files}
        assert len(d['pose_ids']) == 109 and len(set(d['pose_ids'].tolist())) == 109
        assert np.array_equal(d['source_initialized_qgrid'], f['actual_named_qpos'])
        assert d['source_initialized_qgrid'][0].tobytes() == d['source_initialized_q0'].tobytes()
        assert f['actual_recovered_axisangles_float32'][0].tobytes() == d['original_axisangles_float32'].tobytes()
        assert f['actual_recovered_translations_float32'][0].tobytes() == d['original_camera_translation_float32'].tobytes()
        for a in [*d.values(), *f.values()]:
            if np.issubdtype(a.dtype, np.number):
                assert np.isfinite(a).all()
        camera = d['original_cameraWorld64']
        assert np.array_equal(camera[3], [0., 0., 0., 1.])
        assert np.max(np.abs(camera[:3, :3].T @ camera[:3, :3] - np.eye(3))) < 1e-12
        assert abs(np.linalg.det(camera[:3, :3]) - 1) < 1e-12
        camera32 = np.add(f['source_wrist_vertices_float32'], f['actual_recovered_translations_float32'][:, None], dtype=np.float32)
        assert camera32.tobytes() == f['source_translated_camera_vertices_float32'].tobytes()
        # Preserve the recorded per-pose contraction order; batching changes
        # floating-point reduction and cannot support a byte-equality assertion.
        world = np.array([np.einsum('ij,kj->ki', camera[:3, :3], a, optimize=False) + camera[:3, 3] for a in camera32])
        anchors = np.array([np.einsum('ij,kj->ki', camera[:3, :3], a, optimize=False) + camera[:3, 3] for a in f['source_translated_camera_anchors_float32']])
        rotations = np.array([np.einsum('ij,njk->nik', camera[:3, :3], a, optimize=False) for a in f['source_camera_rotations_float32']])
        reduction_errors = dict(vertices=float(np.max(np.abs(world - f['source_vertices_world']))),
            anchors=float(np.max(np.abs(anchors - f['source_joint_anchors_world']))),
            rotations=float(np.max(np.abs(rotations - f['source_body_rotations_world']))))
        # The producer's source arrays can have a different stride/reduction
        # order. Verify arithmetic within double precision; retain the original
        # strict scientific gates below and expose these residuals separately.
        assert max(reduction_errors.values()) <= 1e-15
        faces = d['faces']
        assert np.array_equal(faces, f['faces'])
        sn = normal(world, faces)
        en = normal(f['compiled_vertices_world_m'], faces)
        original_n = normal(d['original_native_target_world_m'][None], faces)
        measured = dict(
            vertex_error_m=distance(f['compiled_vertices_world_m'], world),
            normal_error=distance(en, sn),
            compiled_normal_array_error=distance(en, f['compiled_normals_world']),
            anchor_error_m=distance(f['compiled_joint_anchors_world'], anchors),
            rotation_component_error=float(np.max(np.abs(f['compiled_body_rotations_world'] - rotations))),
            original_q0_vertex_error_m=distance(f['compiled_vertices_world_m'][0], d['original_native_target_world_m']),
            original_q0_normal_error=distance(en[0], original_n[0]),
            source_coordinate_rotation_roundtrip_error=float(np.max(f['coordinate_rotation_roundtrip_errors'])),
        )
        gates = protocol['gates']
        limits = dict(vertex_error_m=gates['engine_source_vertex_comparison_m'],
            normal_error=gates['engine_source_normal_component_comparison'],
            compiled_normal_array_error=1e-12, anchor_error_m=5e-6,
            rotation_component_error=1e-6, original_q0_vertex_error_m=5e-6,
            original_q0_normal_error=1e-6, source_coordinate_rotation_roundtrip_error=1e-6)
        checks = {k: measured[k] <= limit for k, limit in limits.items()}
        assert not np.any(f['warning_counts'])
        assert not np.any(f['retained_qvel']) and not np.any(f['retained_ctrl'])
        assert not np.any(f['geom_contype']) and not np.any(f['geom_conaffinity'])
        assert not np.any(f['flex_contype']) and not np.any(f['flex_conaffinity'])
        cases.append(dict(case=case, poses=109, measurements=measured, limits=limits, checks=checks,
            camera_transform_reduction_residuals=reduction_errors, arithmetic_residual_limit=1e-15,
            exact_q0_source_bytes=True, actual_named_qgrid_equal=True,
            source_float32_translation_order_verified=True,
            all_checks_pass=all(checks.values())))
    # Inspect pickle opcodes only. Do not execute a dataset pickle.
    import pickletools
    manifest_path = Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_benchmark_review_2026-10-09/download_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    oracle = []
    for item in manifest['files']:
        if not item['local_path'].endswith('.pkl'):
            continue
        p = Path(item['local_path'])
        assert sha(p) == item['sha256']
        ops = list(pickletools.genops(p.read_bytes()))
        indices = [i for i, (_, value, _) in enumerate(ops) if value == 'grasp']
        assert len(indices) == 1
        i = indices[0]
        value_op = ops[i+2][0].name if ops[i+1][0].name == 'MEMOIZE' else ops[i+1][0].name
        oracle.append(dict(path=str(p), sha256=sha(p), grasp_value_opcode=value_op, contains_grasp_label=value_op != 'NONE'))
    if (manifest['dataset'] != 'kevinywu/1m-hugs'
        or manifest['revision'] != '56ea3f100b1a4ec38bb35e986c0dd14d38977283'
        or len(oracle) != 10 or any(row['contains_grasp_label'] for row in oracle)):
        raise ValueError('Expected ten pinned evaluation frames with grasp=None')
    return dict(schema_version='manosim_reconstruction_durable_readback_v1',
        status='PASS' if all(c['all_checks_pass'] for c in cases) else 'FAIL', cases=cases,
        prior_independent_receipt_sha256=RECEIPT_SHA, prior_protocol_sha256=PROTOCOL_SHA,
        verified_durable_artifacts=checked,
        prior_full_dependency_closure_live_reverified=False,
        prior_dependency_paths_currently_missing=missing,
        oracle_observation_audit=dict(dataset=manifest['dataset'], revision=manifest['revision'],
            manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path), frames=oracle,
            finding='These ten downloaded evaluation frames have grasp=None; they are not recorded-human oracle labels.'),
        operations=dict(native_model_constructions=0, MANO_or_skin_evaluations=0, FK_or_collision_queries=0, physics_steps=0),
        verifier_development_history=['Two source-only audit attempts stopped before output at transform byte-equality assertions. Per-pose contractions still differ by double-precision reduction/stride rounding (observed 5.55e-17 m). The verifier explicitly reports these residuals at a 1e-15 arithmetic bound.', 'The first persisted readback.json checked engine/source against reconstruction allowances. Peer review prompted this readback_v2 to use the original stricter engine/source gates (1e-12 m,1e-10 normal), add coordinate-roundtrip checks, refuse python -O and require ten pinned null-grasp frames. The first readback is preserved; no saved operand or historical result changed.'],
        limits='Saved operands establish finite geometric reconstruction, not dynamic/contact fidelity or universal grasp coverage.',
        collection_or_training_allowed=False, verifier_source_sha256=sha(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'cases', 'operations')}))
    sys.exit(0 if result['status'] == 'PASS' else 1)
