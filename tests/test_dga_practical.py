"""Manufactured static boundary tests, never physical/native-population evidence."""
from dataclasses import replace
import numpy as np
import pytest
from grasp_failure_prediction.proposals import dexgrasp_anything as dga
from grasp_failure_prediction.proposals import dga_practical as practical
from test_dexgrasp_anything_provider import hands, native_values, make_hand, rigid, rz


def native(hands, values, transform=None):
    return dga.decode_dga_hand_pose(values, object_from_input=np.eye(4) if transform is None else transform,
                                    source_hand_urdf=hands[0])


def narrower(hands, monkeypatch):
    make_hand(hands[1], finger_upper=.1)
    monkeypatch.setattr(dga, 'TARGET_URDF_SHA256', dga.file_sha256(hands[1]))


def test_identity_and_named_root_roundtrip(hands, native_values):
    n = native(hands, native_values)
    r = practical.build_practical_reference(n, target_hand_urdf=hands[1])
    assert r['joint_targets_rad'] == n.joint_targets_rad and r['joint_changes'] == {}
    f = dga.DGAHandTree(hands[1], dga.TARGET_URDF_SHA256).forward(r['joint_targets_rad'])
    np.testing.assert_allclose(np.array(r['object_from_target_root']) @ f['palm'], n.object_from_palm, atol=1e-14)
    assert r['structural_status'] == 'PASS' and r['geometry_status'] == 'NOT_ASSESSED'
    assert not r['execution_eligible']


def test_explicit_projection_minimality_and_native_preservation(hands, native_values, monkeypatch):
    narrower(hands, monkeypatch)
    n = native(hands, native_values)
    q_before = dict(n.joint_targets_rad); p_before = n.object_from_palm.tobytes()
    r = practical.build_practical_reference(n, target_hand_urdf=hands[1])
    assert len(r['joint_changes']) == 22
    assert n.joint_targets_rad == q_before and n.object_from_palm.tobytes() == p_before
    for k, change in r['joint_changes'].items():
        assert change['target_rad'] == .1 and change['delta_rad'] < 0
        for alternative in (-1., -.5, 0., .1):
            assert abs(alternative - q_before[k]) >= abs(change['delta_rad'])


def test_source_invalid_is_rejected_before_projection(hands, native_values, monkeypatch):
    n = native(hands, native_values)
    q = dict(n.joint_targets_rad); q['THJ1'] = 2.
    with pytest.raises(dga.DGAContractError, match='outside'):
        practical.build_practical_reference(replace(n, joint_targets_rad=q), target_hand_urdf=hands[1])


def test_nonidentity_frame_equivariance(hands, native_values, monkeypatch):
    narrower(hands, monkeypatch)
    transform = rigid(rz(.7), [.2, -.1, .3])
    a = practical.build_practical_reference(native(hands, native_values), target_hand_urdf=hands[1])
    b = practical.build_practical_reference(native(hands, native_values, transform), target_hand_urdf=hands[1])
    assert a['joint_targets_rad'] == b['joint_targets_rad']
    np.testing.assert_allclose(np.array(b['object_from_target_root']), transform @ a['object_from_target_root'], atol=1e-14)


@pytest.mark.parametrize('defect', ['palm', 'wrist', 'legal_branch', 'nan', 'limit'])
def test_candidate_refuses_structural_corruption(hands, native_values, defect):
    n = native(hands, native_values); q = dict(n.joint_targets_rad); palm = n.object_from_palm.copy()
    if defect == 'palm': palm[0, 3] += 1e-12
    elif defect == 'wrist': q['WRJ2'] += .01
    elif defect == 'legal_branch': q['FFJ2'] += .01
    elif defect == 'nan': q['FFJ2'] = np.nan
    elif defect == 'limit': q['FFJ2'] = 1.00000000001
    with pytest.raises(dga.DGAContractError):
        practical.validate_practical_pose(n, joint_targets_rad=q, object_from_palm=palm, target_hand_urdf=hands[1])


def test_mutated_source_frames_rejected(hands, native_values):
    n = native(hands, native_values)
    frames = {k: v.copy() for k, v in n.object_from_links.items()}
    frames['ff0'][0, 3] += .001
    with pytest.raises(dga.DGAContractError):
        practical.build_practical_reference(replace(n, object_from_links=frames), target_hand_urdf=hands[1])
