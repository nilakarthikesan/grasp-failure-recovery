"""Static native validation guards and source/target comparison tests."""
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from grasp_failure_prediction.integrations import phase1_manosim as native


def test_source_axisangle_exp_preserves_nonidentity_root_and_named_fingers():
    aa = np.zeros(48, dtype=np.float32)
    aa[2] = np.float32(np.pi / 2)
    aa[3] = np.float32(.4)
    rotations = native.axis_angle_rotations(aa)
    np.testing.assert_allclose(rotations[0] @ [1., 0., 0.], [0., 1., 0.], atol=1e-7)
    np.testing.assert_allclose(rotations[1] @ [0., 1., 0.], [0., np.cos(.4), np.sin(.4)], atol=1e-7)
    np.testing.assert_allclose(rotations @ np.transpose(rotations, (0, 2, 1)), np.broadcast_to(np.eye(3), (16, 3, 3)), atol=1e-15)
    assert np.linalg.det(rotations).min() > .999999999999


def test_source_rotation_rejects_double_precision_substitute_and_nonfinite():
    with pytest.raises(ValueError, match='literal'):
        native.axis_angle_rotations(np.zeros(48, dtype=np.float64))
    aa = np.zeros(48, dtype=np.float32)
    aa[0] = np.nan
    with pytest.raises(ValueError, match='literal'):
        native.axis_angle_rotations(aa)


def test_existing_hand_cannot_be_selected_for_different_shape(tmp_path):
    (tmp_path / 'assets').mkdir()
    np.save(tmp_path / 'assets/mano_rhand_shape.npy', np.zeros((1, 10), dtype=np.float32))
    with pytest.raises(ValueError, match='separately bound shape build'):
        native.select_assets(tmp_path, np.ones((1, 10), dtype=np.float32))


def test_combined_scene_keeps_cube_defaults_and_namespaces_hand_defaults(tmp_path):
    scene = tmp_path / 'scene'
    scene.mkdir()
    (scene / 'scene.xml').write_text('''<mujoco><compiler inertiafromgeom="true"/><default><geom friction="1 .005 .0001" solref=".02 1"/></default><worldbody><body name="cube"><freejoint name="cube_free"/><geom name="cube" type="box" size=".025 .025 .025" mass=".1"/></body><camera name="capture"/></worldbody><keyframe><key name="capture" qpos="0 0 .025 1 0 0 0"/></keyframe></mujoco>''')
    hand = tmp_path / 'capsule_hand.xml'
    hand.write_text('''<mujoco><default><geom solref=".04 1"/></default><asset><mesh name="palm" file="meshes/palm.obj"/></asset><worldbody><body name="mano_body_0"><geom type="mesh" mesh="palm"/></body></worldbody><actuator/></mujoco>''')
    assembled = ET.fromstring(native.assemble_scene(scene, hand))
    assert assembled.find('keyframe') is None
    assert assembled.find('default/geom').get('solref') == '.02 1'
    assert assembled.find("default/default[@class='phase1_native_hand']/geom").get('solref') == '.04 1'
    assert assembled.find("worldbody/body[@name='mano_body_0']").get('childclass') == 'phase1_native_hand'
    assert assembled.find("worldbody/body[@name='cube']/geom").get('size') == '.025 .025 .025'
    assert assembled.find('compiler').get('inertiafromgeom') == 'auto'
    assert Path(assembled.find('asset/mesh').get('file')).is_absolute()


def fixture_values():
    return dict(postAA32=np.zeros(48, np.float32), translation32=np.zeros(3, np.float32),
                shape32=np.zeros((1, 10), np.float32), source_transforms_abs32=np.tile(np.eye(4, dtype=np.float32), (16, 1, 1)),
                cameraWorld64=np.eye(4), mesh_vertices=np.zeros((778, 3), np.float32),
                landmarks_3d=np.zeros((21, 3), np.float32), mesh_faces=np.zeros((1552, 3), np.int64))


def test_proposal_camera_must_match_actual_saved_capture(tmp_path):
    proposal = tmp_path / 'proposal'
    proposal.mkdir()
    scene = tmp_path / 'scene'
    scene.mkdir()
    np.save(scene / 'T_world_camera.npy', np.eye(4))
    values = fixture_values()
    np.savez(proposal / 'proposal.npz', **values)
    loaded = native.load_proposal(proposal, scene)
    assert loaded['shape32'].shape == (1, 10)
    values['cameraWorld64'][0, 3] = .01
    np.savez(proposal / 'proposal.npz', **values)
    with pytest.raises(ValueError, match='camera differs'):
        native.load_proposal(proposal, scene)


def test_actual_source_geometry_check_detects_error_even_when_compiled_fk_agrees():
    values = fixture_values()
    reference = dict(vertices=values['mesh_vertices'].copy(), landmarks=values['landmarks_3d'].copy(),
                     anchors=np.zeros((16, 3)), transforms=values['source_transforms_abs32'].copy(), faces=values['mesh_faces'])
    state = SimpleNamespace(joint_anchor_world_m=np.zeros((16, 3)), world_body_rotations=np.tile(np.eye(3), (16, 1, 1)),
                            independent_readback={'anchor_max_error_m': 0.})
    reference['vertices'][4, 0] = .001
    metrics = native.mapping_metrics(state, values, reference)
    assert metrics['independent_compiled_fk']['anchor_max_error_m'] == 0.
    assert metrics['max_bone_anchor_error_m'] == 0.
    assert metrics['source_vertex_reconstruction_max_m'] > native.POSITION_LIMIT_M


def test_existing_output_rejected_before_reading_source_or_scene(tmp_path):
    output = tmp_path / 'already'
    output.mkdir()
    sentinel = output / 'keep'
    sentinel.write_text('original')
    with pytest.raises(FileExistsError):
        native.validate_native('missing', 'missing', output)
    assert sentinel.read_text() == 'original'
