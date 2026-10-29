"""Offline boundary tests; these tests never load HUG or generate grasps."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from grasp_failure_prediction.integrations.phase1_hug import (
    export_proposal, require_fixed_protocol, save_arrays, sha, source_files,
    validate_capture, write_json,
)


def test_fixed_approved_generation_protocol():
    require_fixed_protocol([0, 1, 2], 50)
    for seeds, steps in (([0], 50), ([2, 1, 0], 50), ([0, 1, 2], 1), ([0, 1, 2, 3], 50)):
        with pytest.raises(ValueError):
            require_fixed_protocol(seeds, steps)


def test_array_archive_rejects_objects_and_nonfinite_values(tmp_path):
    for value in (np.array([object()]), np.array([np.nan]), np.array(['unsafe'])):
        with pytest.raises(ValueError):
            save_arrays(tmp_path / 'bad.npz', {'value': value})
    assert not (tmp_path / 'bad.npz').exists()


def test_export_preserves_literal_source_arrays_and_official_grasp_fields(tmp_path):
    position = np.array([[0., 0., .5]], dtype=np.float32)
    points = np.zeros((21, 3), dtype=np.float32) + position
    grasp = {'pose': np.zeros((1, 15, 3), dtype=np.float32),
             'shape': np.zeros((1, 10), dtype=np.float32),
             't': position, 'R_6d': np.array([[1, 0, 0, 0, 1, 0]], dtype=np.float32),
             'T_camera_wrist': np.eye(4, dtype=np.float32),
             'landmarks_3d': points, 'landmarks_2d': np.zeros((21, 2), dtype=np.float32),
             'mesh_vertices': np.zeros((778, 3), dtype=np.float32),
             'mesh_faces': np.tile(np.array([[0, 1, 2]], dtype=np.int64), (1552, 1))}
    aa = np.arange(48, dtype=np.float32) / 100
    transforms = np.broadcast_to(np.eye(4, dtype=np.float32), (16, 4, 4)).copy()
    captured = {'postAA32': aa, 'source_transforms_abs32': transforms}
    camera = np.eye(4, dtype=np.float64)
    receipt = export_proposal(tmp_path, grasp, prediction=np.zeros(99, dtype=np.float32),
                               captured=captured, shape=grasp['shape'], camera=camera)
    with np.load(tmp_path / 'proposal.npz', allow_pickle=False) as archive:
        for key in grasp:
            np.testing.assert_array_equal(archive[key], grasp[key])
        assert archive['postAA32'].dtype == np.float32
        assert archive['postAA32'].tobytes() == aa.tobytes()
        assert archive['source_transforms_abs32'].tobytes() == transforms.tobytes()
        assert archive['cameraWorld64'].dtype == np.float64
        assert archive['translation32'].shape == (3,)
    assert receipt['proposal_sha256'] == sha(tmp_path / 'proposal.pkl')
    assert receipt['proposal_npz_sha256'] == sha(tmp_path / 'proposal.npz')
    captured['postAA32'] = aa.astype(np.float64)
    with pytest.raises(ValueError, match='literal float32'):
        export_proposal(tmp_path, grasp, prediction=np.zeros(99), captured=captured,
                          shape=grasp['shape'], camera=camera)


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / 'scene'
    root.mkdir()
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    metric = np.full((224, 224), .5, dtype=np.float32)
    Image.fromarray(rgb).save(root / 'rgb.png')
    Image.fromarray(np.full((224, 224), 500, dtype=np.uint16)).save(root / 'depth.png')
    Image.fromarray(np.full((224, 224), 255, dtype=np.uint8)).save(root / 'object_mask.png')
    np.save(root / 'depth_m.npy', metric)
    np.save(root / 'intrinsics.npy', [[100., 0, 111.5], [0, 100, 111.5], [0, 0, 1]])
    np.save(root / 'T_world_camera.npy', np.eye(4, dtype=np.float64))
    manifest = {'rgb_path': 'rgb.png', 'depth_m_path': 'depth_m.npy',
                'intrinsics_path': 'intrinsics.npy', 'mask_path': 'object_mask.png',
                'selection_uv': [111, 111]}
    write_json(root / 'observation_manifest.json', manifest)
    write_json(root / 'validation_report.json', {'scene_preparation_checks_passed': True,
                                                'artifact_sha256': source_files(root)})
    return root


def test_scene_gate_is_read_only_and_rejects_tampered_artifacts(capture):
    before = source_files(capture)
    manifest, rgb, encoded, intrinsic, camera = validate_capture(capture)
    assert source_files(capture) == before
    assert manifest['selection_uv'] == [111, 111]
    assert rgb.shape == (224, 224, 3) and encoded.dtype == np.uint16
    with (capture / 'rgb.png').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='approved scene artifact changed'):
        validate_capture(capture)


def test_invalid_selected_depth_is_rejected_without_fallback(capture):
    encoded = np.asarray(Image.open(capture / 'depth.png')).copy()
    encoded[111, 111] = 0
    metric = np.load(capture / 'depth_m.npy')
    metric[111, 111] = 0
    Image.fromarray(encoded).save(capture / 'depth.png')
    np.save(capture / 'depth_m.npy', metric)
    report = json.loads((capture / 'validation_report.json').read_text())
    report['artifact_sha256']['depth.png'] = sha(capture / 'depth.png')
    report['artifact_sha256']['depth_m.npy'] = sha(capture / 'depth_m.npy')
    write_json(capture / 'validation_report.json', report)
    with pytest.raises(ValueError, match='no silent fallback'):
        validate_capture(capture)
