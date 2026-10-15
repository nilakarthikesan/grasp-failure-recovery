"""Independent manufactured static corruption controls; no native evidence."""
from dataclasses import replace
from xml.etree import ElementTree as ET
import numpy as np
import pytest
from grasp_failure_prediction.proposals import dexgrasp_anything as dga
from grasp_failure_prediction.proposals import dga_practical as practical
from test_dexgrasp_anything_provider import hands, native_values


def decoded(hands, values):
    return dga.decode_dga_hand_pose(values, object_from_input=np.eye(4), source_hand_urdf=hands[0])


def reject(code, callback):
    with pytest.raises(dga.DGAContractError) as error:
        callback()
    assert error.value.code == code


@pytest.mark.parametrize('number', [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize('which', ['source', 'candidate'])
def test_nonfinite_angles_cannot_be_projected_or_admitted(hands, native_values, number, which):
    n = decoded(hands, native_values)
    q = dict(n.joint_targets_rad)
    q['THJ4'] = number
    if which == 'source':
        reject('native_numeric_invalid', lambda: practical.build_practical_reference(
            replace(n, joint_targets_rad=q), target_hand_urdf=hands[1]))
    else:
        reject('native_numeric_invalid', lambda: practical.validate_practical_pose(
            n, joint_targets_rad=q, object_from_palm=n.object_from_palm, target_hand_urdf=hands[1]))


def test_source_valid_target_invalid_wrist_is_not_repaired(hands, native_values, monkeypatch):
    n = decoded(hands, native_values)
    xml = ET.parse(hands[1])
    xml.find("./joint[@name='WRJ2']/limit").set('upper', '.1')
    xml.write(hands[1], encoding='unicode')
    monkeypatch.setattr(dga, 'TARGET_URDF_SHA256', dga.file_sha256(hands[1]))
    before = n.joint_targets_rad['WRJ2']
    reject('practical_wrist_unsupported', lambda: practical.build_practical_reference(n, target_hand_urdf=hands[1]))
    assert n.joint_targets_rad['WRJ2'] == before


@pytest.mark.parametrize('which', ['source', 'candidate'])
@pytest.mark.parametrize('defect', ['missing', 'extra'])
def test_joint_names_are_complete_and_exact(hands, native_values, which, defect):
    n = decoded(hands, native_values)
    q = dict(n.joint_targets_rad)
    if defect == 'missing':
        del q['THJ2']
    else:
        q['UNKNOWN'] = 0.
    if which == 'source':
        reject('native_joint_set_mismatch', lambda: practical.build_practical_reference(
            replace(n, joint_targets_rad=q), target_hand_urdf=hands[1]))
    else:
        reject('native_joint_set_mismatch', lambda: practical.validate_practical_pose(
            n, joint_targets_rad=q, object_from_palm=n.object_from_palm, target_hand_urdf=hands[1]))


@pytest.mark.parametrize('defect', ['root_translation', 'native_palm_translation', 'source_palm_link_translation'])
def test_inconsistent_root_palm_and_link_chains_rejected(hands, native_values, defect):
    n = decoded(hands, native_values)
    if defect == 'root_translation':
        root = n.object_from_forearm.copy(); root[0, 3] += .01
        n = replace(n, object_from_forearm=root)
    elif defect == 'native_palm_translation':
        palm = n.object_from_palm.copy(); palm[1, 3] += .01
        n = replace(n, object_from_palm=palm)
    else:
        links = {k:v.copy() for k,v in n.object_from_links.items()}
        links['palm'][2, 3] += .01
        n = replace(n, object_from_links=links)
    reject('native_frame_inconsistent', lambda: practical.build_practical_reference(n, target_hand_urdf=hands[1]))


@pytest.mark.parametrize('defect', ['reflection', 'scale', 'nan'])
def test_candidate_palm_is_a_finite_proper_transform(hands, native_values, defect):
    n = decoded(hands, native_values)
    palm = n.object_from_palm.copy()
    if defect == 'reflection':
        palm[:3, 0] *= -1
    elif defect == 'scale':
        palm[:3, :3] *= 1.1
    else:
        palm[0, 3] = np.nan
    reject('native_numeric_invalid' if defect == 'nan' else 'native_transform_invalid',
           lambda: practical.validate_practical_pose(n, joint_targets_rad=n.joint_targets_rad,
                  object_from_palm=palm, target_hand_urdf=hands[1]))


def test_all_native_arrays_and_joint_values_remain_unchanged(hands, native_values):
    n = decoded(hands, native_values)
    snapshot = (dict(n.joint_targets_rad), n.object_from_forearm.tobytes(),
                n.object_from_palm.tobytes(), {k:v.tobytes() for k,v in n.object_from_links.items()})
    ref = practical.build_practical_reference(n, target_hand_urdf=hands[1])
    practical.validate_practical_pose(n, joint_targets_rad=ref['joint_targets_rad'],
                                     object_from_palm=ref['object_from_palm'], target_hand_urdf=hands[1])
    current = (dict(n.joint_targets_rad), n.object_from_forearm.tobytes(),
               n.object_from_palm.tobytes(), {k:v.tobytes() for k,v in n.object_from_links.items()})
    assert current == snapshot
    assert ref['geometry_status'] == 'NOT_ASSESSED'
    assert ref['physical_status'] == 'NOT_ASSESSED'
    assert ref['execution_eligible'] is False
