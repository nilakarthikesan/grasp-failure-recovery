"""Manufactured DGA contract tests: no checkpoint, inference or physics.

The fixture hands intentionally differ from production assets. Every fixture
monkeypatches the two expected hashes; none can establish actual-native fidelity.
"""

from __future__ import annotations

import json
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from pydantic import ValidationError

from grasp_failure_prediction.proposals import dexgrasp_anything as dga


def rx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def ry(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def rigid(rotation, translation):
    value = np.eye(4)
    value[:3, :3], value[:3, 3] = rotation, translation
    return value


def columns(rotation):
    return np.concatenate([rotation[:, 0], rotation[:, 1]])


def make_hand(path, *, wrist_offset=(0., -.01, .213), finger_upper=1.):
    """A genuine tree with 24 named revolutes and independently known wrists."""
    robot = ET.Element("robot", name="manufactured_dga_contract_hand")
    for name in ("forearm", "wrist", "palm"):
        ET.SubElement(robot, "link", name=name)

    def joint(name, parent, child, xyz, axis, upper=1.):
        element = ET.SubElement(robot, "joint", name=name, type="revolute")
        ET.SubElement(element, "parent", link=parent)
        ET.SubElement(element, "child", link=child)
        ET.SubElement(element, "origin", xyz=" ".join(map(str, xyz)), rpy="0 0 0")
        ET.SubElement(element, "axis", xyz=" ".join(map(str, axis)))
        ET.SubElement(element, "limit", lower="-1", upper=str(upper))

    joint("WRJ2", "forearm", "wrist", wrist_offset, (0, 1, 0))
    joint("WRJ1", "wrist", "palm", (0, 0, .034), (1, 0, 0))
    for prefix, names in (
        ("ff", ("FFJ4", "FFJ3", "FFJ2", "FFJ1")),
        ("mf", ("MFJ4", "MFJ3", "MFJ2", "MFJ1")),
        ("rf", ("RFJ4", "RFJ3", "RFJ2", "RFJ1")),
        ("lf", ("LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1")),
        ("th", ("THJ5", "THJ4", "THJ3", "THJ2", "THJ1")),
    ):
        parent = "palm"
        for index, name in enumerate(names):
            child = prefix + str(index)
            ET.SubElement(robot, "link", name=child)
            joint(name, parent, child, (.003, 0, .02), (1, 0, 0), finger_upper)
            parent = child
    ET.ElementTree(robot).write(path, encoding="unicode")
    return path


@pytest.fixture
def hands(tmp_path, monkeypatch):
    source = make_hand(tmp_path / "source.urdf")
    target = make_hand(tmp_path / "target.urdf", wrist_offset=(.008, 0., .19))
    monkeypatch.setattr(dga, "SOURCE_URDF_SHA256", dga.file_sha256(source))
    monkeypatch.setattr(dga, "TARGET_URDF_SHA256", dga.file_sha256(target))
    return source, target


@pytest.fixture
def native_values():
    values = np.zeros(33)
    values[:3] = [.01, -.02, .03]
    values[3:9] = columns(rx(.4) @ ry(-.2))
    values[9:] = .15
    values[9:11] = [.2, -.3]
    return values


def expect_code(code, callback):
    with pytest.raises(dga.DGAContractError) as caught:
        callback()
    assert caught.value.code == code


def test_nonidentity_object_and_forearm_compose_both_wrists_analytically(hands, native_values):
    object_from_input = rigid(rz(.7), [.2, -.1, .3])
    pose = dga.decode_dga_hand_pose(native_values, object_from_input=object_from_input,
                                    source_hand_urdf=hands[0])
    input_rotation = rx(.4) @ ry(-.2)
    root_rotation = rz(.7) @ input_rotation
    root_translation = rz(.7) @ native_values[:3] + [.2, -.1, .3]
    wrist2, wrist1 = native_values[9:11]
    palm_local = np.array([.034 * np.sin(wrist2), -.01, .213 + .034 * np.cos(wrist2)])
    expected_palm_rotation = root_rotation @ ry(wrist2) @ rx(wrist1)
    expected_palm_translation = root_translation + root_rotation @ palm_local
    np.testing.assert_allclose(pose.object_from_forearm, rigid(root_rotation, root_translation), atol=1e-14)
    np.testing.assert_allclose(pose.object_from_palm,
                               rigid(expected_palm_rotation, expected_palm_translation), atol=1e-14)
    assert pose.joint_targets_rad["WRJ2"] == wrist2
    assert pose.joint_targets_rad["WRJ1"] == wrist1
    assert tuple(pose.joint_targets_rad) == dga.SOURCE_JOINT_NAMES
    # A free-palm misinterpretation would lose a roughly quarter-metre offset.
    assert np.linalg.norm(pose.object_from_palm[:3, 3] - root_translation) > .2


def test_candidate_compensates_target_root_without_double_wrist_fk(hands, native_values):
    pose = dga.decode_dga_hand_pose(native_values, object_from_input=rigid(rz(.7), [.2, -.1, .3]),
                                    source_hand_urdf=hands[0])
    candidate = dga.map_dga_candidate(pose, target_hand_urdf=hands[1])
    wrist2, wrist1 = native_values[9:11]
    target_local = rigid(ry(wrist2) @ rx(wrist1),
                         [.008 + .034 * np.sin(wrist2), 0, .19 + .034 * np.cos(wrist2)])
    reconstructed = np.asarray(candidate["object_from_target_root"]) @ target_local
    np.testing.assert_allclose(reconstructed, pose.object_from_palm, atol=1e-14)
    assert candidate["joint_targets_rad"] == pose.joint_targets_rad
    assert candidate["execution_eligible"] is False
    assert candidate["acceptance"] == "UNRESOLVED"
    assert candidate["actual_native_validation"] == "NOT_TESTED"
    assert candidate["surface_fidelity"] == "NOT_TESTED"
    assert candidate["physics_steps"] == 0


def test_target_limits_reject_without_clipping(hands, native_values, monkeypatch):
    make_hand(hands[1], finger_upper=.1)
    monkeypatch.setattr(dga, "TARGET_URDF_SHA256", dga.file_sha256(hands[1]))
    pose = dga.decode_dga_hand_pose(native_values, object_from_input=np.eye(4), source_hand_urdf=hands[0])
    before = dict(pose.joint_targets_rad)
    expect_code("joint_limit_violation", lambda: dga.map_dga_candidate(pose, target_hand_urdf=hands[1]))
    assert pose.joint_targets_rad == before


@pytest.mark.parametrize("mutation,code", [
    ("source_hash", "source_hand_mismatch"), ("root_scale", "native_transform_invalid"),
    ("palm_translation", "native_frame_inconsistent"), ("missing_link", "native_link_set_mismatch"),
    ("extra_link", "native_link_set_mismatch"), ("finger_translation", "native_frame_inconsistent"),
    ("finger_reflection", "native_transform_invalid"), ("joint_value", "native_frame_inconsistent"),
    ("root_nan", "native_numeric_invalid"),
])
def test_public_native_pose_is_revalidated_after_dataclass_replacement(hands, native_values, mutation, code):
    pose = dga.decode_dga_hand_pose(native_values, object_from_input=np.eye(4), source_hand_urdf=hands[0])
    if mutation == "source_hash":
        changed = replace(pose, source_hand_sha256="sha256:" + "0" * 64)
    elif mutation == "root_scale":
        changed = replace(pose, object_from_forearm=np.diag([2., 1., 1., 1.]))
    elif mutation == "root_nan":
        changed = replace(pose, object_from_forearm=np.full((4, 4), np.nan))
    elif mutation == "palm_translation":
        palm = pose.object_from_palm.copy()
        palm[0, 3] += .01
        changed = replace(pose, object_from_palm=palm)
    elif mutation == "joint_value":
        joints = dict(pose.joint_targets_rad)
        joints["FFJ4"] += .1
        changed = replace(pose, joint_targets_rad=joints)
    else:
        links = {name: frame.copy() for name, frame in pose.object_from_links.items()}
        if mutation == "missing_link":
            links.pop("ff0")
        elif mutation == "extra_link":
            links["unknown_link"] = np.eye(4)
        elif mutation == "finger_translation":
            links["ff0"][0, 3] += .01
        elif mutation == "finger_reflection":
            links["ff0"][:3, 0] *= -1
        changed = replace(pose, object_from_links=links)
    expect_code(code, lambda: dga.map_dga_candidate(changed, target_hand_urdf=hands[1]))


def test_mutable_shared_palm_array_cannot_forge_consistent_native_geometry(hands, native_values):
    pose = dga.decode_dga_hand_pose(native_values, object_from_input=np.eye(4), source_hand_urdf=hands[0])
    pose.object_from_links["palm"][0, 3] += .01
    expect_code("native_frame_inconsistent", lambda: dga.map_dga_candidate(pose, target_hand_urdf=hands[1]))


def test_source_asset_is_reverified_when_mapping_existing_pose(hands, native_values):
    pose = dga.decode_dga_hand_pose(native_values, object_from_input=np.eye(4), source_hand_urdf=hands[0])
    hands[0].write_text(hands[0].read_text() + "\n")
    expect_code("hand_asset_hash_mismatch", lambda: dga.map_dga_candidate(pose, target_hand_urdf=hands[1]))


def test_symmetric_6d_orthogonalization_is_not_gram_schmidt():
    result = dga.rotation_from_6d([1., 0., 0., .6, .8, 0.])
    # The two supplied directions subtend atan2(.8,.6); symmetric correction
    # splits their departure from a right angle equally between both columns.
    expected_angle = (np.arctan2(.8, .6) - np.pi / 2) / 2
    np.testing.assert_allclose(result, rz(expected_angle), atol=1e-14)
    assert not np.allclose(result, np.eye(3))
    np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-14)
    assert np.linalg.det(result) == pytest.approx(1.)


@pytest.mark.parametrize("value", [[0.] * 6, [1., 0., 0., 1., 0., 0.], [1., 0., 0., -1., 0., 0.]])
def test_degenerate_6d_rotation_rejected(value):
    expect_code("native_rotation_degenerate", lambda: dga.rotation_from_6d(value))


def test_denormalized_27_inverse_rotation_once_preserves_all_24_joints():
    values = np.arange(27, dtype=float) / 100
    rotation = rx(.3) @ ry(-.5) @ rz(.2)
    result = dga.compose_denormalized_prediction(values, rotation)
    assert result.shape == (33,)
    np.testing.assert_allclose(result[:3], rotation.T @ values[:3], atol=1e-14)
    np.testing.assert_allclose(dga.rotation_from_6d(result[3:9]), rotation.T, atol=1e-14)
    np.testing.assert_array_equal(result[9:], values[3:])


@pytest.mark.parametrize("values", [np.zeros(27), np.zeros(32), np.zeros((1, 33)), ["x"] * 33,
                                    [True] * 33, [np.inf] * 33, [np.nan] * 33])
def test_malformed_native_values_rejected(hands, values):
    expect_code("native_numeric_invalid", lambda: dga.decode_dga_hand_pose(
        values, object_from_input=np.eye(4), source_hand_urdf=hands[0]))


@pytest.mark.parametrize("bad_transform", [np.diag([2., 1., 1., 1.]), np.diag([-1., 1., 1., 1.]),
                                          np.zeros((4, 4)), np.eye(3)])
def test_scaled_reflected_or_malformed_object_transform_rejected(hands, native_values, bad_transform):
    with pytest.raises(dga.DGAContractError):
        dga.decode_dga_hand_pose(native_values, object_from_input=bad_transform, source_hand_urdf=hands[0])


def test_pinned_hand_hash_required(hands, native_values):
    hands[0].write_text(hands[0].read_text() + "\n")
    expect_code("hand_asset_hash_mismatch", lambda: dga.decode_dga_hand_pose(
        native_values, object_from_input=np.eye(4), source_hand_urdf=hands[0]))


def test_source_limits_reject_without_modifying_native_values(hands, native_values):
    native_values[12] = 1.1
    before = native_values.copy()
    expect_code("joint_limit_violation", lambda: dga.decode_dga_hand_pose(
        native_values, object_from_input=np.eye(4), source_hand_urdf=hands[0]))
    np.testing.assert_array_equal(native_values, before)


@pytest.mark.parametrize("missing", ["WRJ2", "WRJ1", "THJ1"])
def test_tree_forward_requires_every_joint_including_both_wrists(hands, missing):
    tree = dga.DGAHandTree(hands[0], dga.SOURCE_URDF_SHA256)
    values = dict.fromkeys(dga.SOURCE_JOINT_NAMES, 0.)
    values.pop(missing)
    expect_code("native_joint_set_mismatch", lambda: tree.forward(values))


@pytest.mark.parametrize("kind", ["continuous", "prismatic", "floating"])
def test_tree_rejects_unrepresented_joint_types(hands, kind):
    root = ET.parse(hands[0])
    root.getroot().find("joint").set("type", kind)
    root.write(hands[0])
    expect_code("hand_joint_unsupported", lambda: dga.DGAHandTree(hands[0], dga.file_sha256(hands[0])))


def test_tree_rejects_mimic_instead_of_silently_expanding_it(hands):
    root = ET.parse(hands[0])
    ET.SubElement(root.getroot().find("joint"), "mimic", joint="WRJ1")
    root.write(hands[0])
    expect_code("hand_joint_unsupported", lambda: dga.DGAHandTree(hands[0], dga.file_sha256(hands[0])))


def test_tree_rejects_second_parent(hands):
    root = ET.parse(hands[0])
    root.getroot().findall("joint")[2].find("child").set("link", "palm")
    root.write(hands[0])
    expect_code("hand_tree_invalid", lambda: dga.DGAHandTree(hands[0], dga.file_sha256(hands[0])))


def write_reference(root, name, value):
    path = root / name
    if name.endswith(".npy"):
        np.save(path, value)
    elif name.endswith(".npz"):
        np.savez(path, **value)
    else:
        path.write_text(value)
    return {"path": name, "sha256": dga.file_sha256(path)}


def sync_bundle_log(bundle):
    path, manifest, _ = bundle
    payload = {
        "schema_version": "dga_export_log_v1", "evidence_origin": manifest["evidence_origin"],
        "configuration_sha256": manifest["configuration"]["sha256"],
        "output_sha256": manifest["output"]["sha256"],
        "object_id": manifest["object_id"], "geometry_content_hash": manifest["geometry_content_hash"],
        "object_mesh_sha256": manifest["object_mesh"]["sha256"],
        "point_cloud_sha256": manifest["point_cloud"]["sha256"], "normals_sha256": manifest["normals"]["sha256"],
        "source_hand_scale": manifest["source_hand_scale"], "physical_object_scale": manifest["physical_object_scale"],
        "object_from_input": manifest["object_from_input"],
        "joint_parameter_names": list(dga.SOURCE_JOINT_NAMES), "state": manifest["state"],
        "fixture_note": "Manufactured contract fixture; no inference run.",
    }
    manifest["generation_log"] = write_reference(path.parent, "generation.json", json.dumps(payload))
    return payload


@pytest.fixture
def bundle(tmp_path, hands, native_values):
    manifest = {
        "schema_version": "dga_native_export_v1", "evidence_origin": "manufactured_contract_fixture",
        "source_repository": dga.SOURCE_REPOSITORY, "source_commit": dga.SOURCE_COMMIT,
        "checkpoint_sha256": dga.CHECKPOINT_SHA256, "dataset_name": "DexGraspNet",
        "normalizer_source_sha256": dga.NORMALIZER_SHA256,
        "normalize_joints": True, "normalize_translation": True,
        "state": "denormalized_inverse_rotated_33", "physics_guided_sampling": False,
        "postprocessing": "none", "source_root": "forearm", "source_hand_scale": 1.0,
        "rotation_representation": "robust_6d_two_columns", "translation_units": "m", "joint_units": "rad",
        "input_frame": "input_point_cloud_physical_m", "normal_vectors": "preserved_as_conditioned",
        "joint_names": list(dga.SOURCE_JOINT_NAMES), "object_id": "manufactured_triangle",
        "geometry_content_hash": "sha256:" + "a" * 64, "physical_object_scale": .2,
        "object_from_input": np.eye(4).tolist(), "seed": 17, "sample_count": 2,
        "source_hand": {"path": hands[0].name, "sha256": dga.file_sha256(hands[0])},
    }
    values = np.stack([native_values, native_values.copy()])
    # Deliberately retain a distant proposal; the decoder must not filter by
    # object proximity, quality, or any later controller outcome.
    values[1, :3] = [2., -3., 4.]
    manifest["output"] = write_reference(tmp_path, "outputs.npz", {"hand_qpos": values})
    manifest["object_mesh"] = write_reference(tmp_path, "object.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    manifest["point_cloud"] = write_reference(tmp_path, "points.npy", np.array([[0., 0., 0.], [.1, 0., 0.]]))
    manifest["normals"] = write_reference(tmp_path, "normals.npy", np.array([[0., 0., 1.], [0., 0., 1.]]))
    keys = ("source_commit", "checkpoint_sha256", "dataset_name", "normalizer_source_sha256",
            "normalize_joints", "normalize_translation", "physics_guided_sampling", "postprocessing", "seed", "sample_count")
    config = {key: manifest[key] for key in keys}
    config["optimizer"] = {"enabled": False}
    manifest["configuration"] = write_reference(tmp_path, "config.json", json.dumps(config))
    result = (tmp_path / "manifest.json", manifest, config)
    sync_bundle_log(result)
    return result


def load_bundle(bundle, *, expected_object_id="manufactured_triangle", expected_geometry_hash="sha256:" + "a" * 64):
    path, manifest, _ = bundle
    path.write_text(json.dumps(manifest))
    return dga.load_dga_export(path, expected_object_id=expected_object_id, expected_geometry_hash=expected_geometry_hash)


def test_manufactured_bundle_remains_classified_and_retains_native_miss(bundle, hands):
    manifest, poses = load_bundle(bundle)
    assert manifest.evidence_origin == "manufactured_contract_fixture"
    assert len(poses) == 2
    np.testing.assert_allclose(poses[1].object_from_forearm[:3, 3], [2., -3., 4.])
    # physical_object_scale is provenance only; it cannot shrink the hand pose.
    assert manifest.physical_object_scale == .2
    for pose in poses:
        candidate = dga.map_dga_candidate(pose, target_hand_urdf=hands[1])
        assert candidate["actual_native_validation"] == "NOT_TESTED"
        assert candidate["execution_eligible"] is False


@pytest.mark.parametrize("key,value", [
    ("state", "raw_normalized_27"), ("postprocessing", "force_tested"),
    ("dataset_name", "real_dex"), ("rotation_representation", "row_major_6d"),
    ("translation_units", "mm"), ("joint_units", "degree"),
    ("source_root", "palm"), ("source_commit", "f" * 40),
    ("source_hand_scale", 2.0), ("input_frame", "normalized_unit_sphere"),
    ("normal_vectors", "normalized_by_decoder"),
    ("checkpoint_sha256", "sha256:" + "0" * 64), ("physical_object_scale", -1.),
    ("normalize_joints", "true"), ("sample_count", True),
    ("evidence_origin", "accepted_generated_mapping"), ("unexpected_field", 1),
])
def test_manifest_stage_dataset_units_and_provenance_fail_closed(bundle, key, value):
    bundle[1][key] = value
    with pytest.raises(ValidationError):
        load_bundle(bundle)


def test_joint_order_requires_exact_24_order(bundle):
    bundle[1]["joint_names"][0:2] = ["WRJ1", "WRJ2"]
    expect_code("native_joint_order_mismatch", lambda: load_bundle(bundle))


@pytest.mark.parametrize("kwargs", [{"expected_object_id": "different"}, {"expected_geometry_hash": "sha256:" + "b" * 64}])
def test_request_object_and_geometry_hash_binding(bundle, kwargs):
    expect_code("native_object_mismatch", lambda: load_bundle(bundle, **kwargs))


@pytest.mark.parametrize("role", ["output", "source_hand", "object_mesh", "point_cloud", "normals", "configuration", "generation_log"])
def test_every_bundle_artifact_hash_is_checked(bundle, role):
    bundle[1][role]["sha256"] = "sha256:" + "0" * 64
    expect_code("native_artifact_hash_mismatch", lambda: load_bundle(bundle))


@pytest.mark.parametrize("path", ["../outside.npy", "/private/tmp/outside.npy", "subdir/../../outside.npy"])
def test_bundle_references_cannot_escape_root(bundle, path):
    bundle[1]["point_cloud"]["path"] = path
    expect_code("native_path_invalid", lambda: load_bundle(bundle))


def test_symlink_escape_rejected(bundle, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "_outside.npy")
    outside.write_bytes(b"untrusted outside fixture")
    link = tmp_path / "escape.npy"
    link.symlink_to(outside)
    bundle[1]["point_cloud"] = {"path": link.name, "sha256": dga.file_sha256(outside)}
    expect_code("native_path_invalid", lambda: load_bundle(bundle))


@pytest.mark.parametrize("key", ["source_commit", "checkpoint_sha256", "dataset_name", "normalizer_source_sha256",
                                 "normalize_joints", "normalize_translation", "physics_guided_sampling", "postprocessing", "seed", "sample_count"])
def test_configuration_must_bind_every_generation_field(bundle, key):
    path, manifest, config = bundle
    config.pop(key)
    manifest["configuration"] = write_reference(path.parent, "config.json", json.dumps(config))
    expect_code("native_configuration_mismatch", lambda: load_bundle(bundle))


@pytest.mark.parametrize("value", [None, False, "disabled"])
def test_optimizer_configuration_required_even_when_disabled(bundle, value):
    path, manifest, config = bundle
    config["optimizer"] = value
    manifest["configuration"] = write_reference(path.parent, "config.json", json.dumps(config))
    expect_code("native_configuration_mismatch", lambda: load_bundle(bundle))


@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_optimizer_enabled_state_cannot_disagree_or_coerce(bundle, value):
    path, manifest, config = bundle
    config["optimizer"]["enabled"] = value
    manifest["configuration"] = write_reference(path.parent, "config.json", json.dumps(config))
    expect_code("native_configuration_mismatch", lambda: load_bundle(bundle))


def test_guided_generation_export_rejected_until_optimizer_contract_audited(bundle):
    path, payload, config = bundle
    payload["physics_guided_sampling"] = True
    config["physics_guided_sampling"] = True
    config["optimizer"] = {"enabled": True, "grad_scale": .1, "loss_weights": {"ERF_loss": .3}}
    payload["configuration"] = write_reference(path.parent, "config.json", json.dumps(config))
    sync_bundle_log(bundle)
    with pytest.raises(ValidationError):
        load_bundle(bundle)


@pytest.mark.parametrize("key,value", [("normalize_joints", 1), ("normalize_translation", "true"),
                                        ("seed", "17"), ("sample_count", 2.)])
def test_configuration_values_cannot_silently_coerce(bundle, key, value):
    path, manifest, config = bundle
    config[key] = value
    manifest["configuration"] = write_reference(path.parent, "config.json", json.dumps(config))
    expect_code("native_configuration_mismatch", lambda: load_bundle(bundle))


@pytest.mark.parametrize("key", ["schema_version", "evidence_origin", "configuration_sha256", "output_sha256",
                                 "object_id", "geometry_content_hash", "object_mesh_sha256", "point_cloud_sha256",
                                 "normals_sha256", "source_hand_scale", "physical_object_scale", "object_from_input",
                                 "joint_parameter_names", "state"])
def test_generation_log_binds_runtime_stage_inputs_and_order(bundle, key):
    path, manifest, _ = bundle
    log = json.loads((path.parent / manifest["generation_log"]["path"]).read_text())
    log.pop(key)
    manifest["generation_log"] = write_reference(path.parent, "generation.json", json.dumps(log))
    expect_code("native_log_mismatch", lambda: load_bundle(bundle))


@pytest.mark.parametrize("role,value", [
    ("point_cloud", np.empty((0, 3))), ("point_cloud", np.zeros((2, 4))),
    ("point_cloud", np.array([[np.nan, 0., 0.], [0., 0., 0.]])),
    ("normals", np.zeros((2, 3))),
    ("normals", np.zeros((1, 3))), ("normals", np.full((2, 3), np.inf)),
])
def test_invalid_point_clouds_and_normals_rejected(bundle, role, value):
    path, manifest, _ = bundle
    manifest[role] = write_reference(path.parent, role + ".npy", value)
    sync_bundle_log(bundle)
    with pytest.raises(dga.DGAContractError):
        load_bundle(bundle)


def test_scaled_conditioning_normals_are_preserved_without_normalization(bundle):
    path, manifest, _ = bundle
    conditioned = np.array([[0., 0., .2], [0., 0., .4]])
    manifest["normals"] = write_reference(path.parent, "normals.npy", conditioned)
    sync_bundle_log(bundle)
    loaded, poses = load_bundle(bundle)
    assert loaded.normal_vectors == "preserved_as_conditioned"
    assert len(poses) == 2
    np.testing.assert_array_equal(np.load(path.parent / "normals.npy"), conditioned)


@pytest.mark.parametrize("values", [np.zeros((2, 27)), np.zeros((1, 33)), np.full((2, 33), np.nan), np.full((2, 33), "text")])
def test_invalid_output_shape_count_or_numbers_rejected(bundle, values):
    path, manifest, _ = bundle
    manifest["output"] = write_reference(path.parent, "outputs.npz", {"hand_qpos": values})
    sync_bundle_log(bundle)
    expect_code("native_numeric_invalid", lambda: load_bundle(bundle))


def test_output_cannot_smuggle_additional_fields(bundle, native_values):
    path, manifest, _ = bundle
    manifest["output"] = write_reference(path.parent, "outputs.npz", {"hand_qpos": np.stack([native_values] * 2), "accepted": np.array([True])})
    sync_bundle_log(bundle)
    expect_code("native_fields_unsupported", lambda: load_bundle(bundle))


def test_output_pickle_arrays_never_loaded(bundle):
    path, manifest, _ = bundle
    manifest["output"] = write_reference(path.parent, "outputs.npz", {"hand_qpos": np.full((2, 33), object(), dtype=object)})
    sync_bundle_log(bundle)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_bundle(bundle)


@pytest.fixture
def offline_worker():
    # Normal repository location, plus colocated standalone test staging.
    path = Path(__file__).resolve().parents[1] / "scripts" / "infer_dexgrasp_anything_proposals.py"
    if not path.is_file():
        path = Path(__file__).with_name("infer_dexgrasp_anything_proposals.py")
    spec = importlib.util.spec_from_file_location("dga_manufactured_offline_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_retains_target_rejection_in_mixed_batch_without_physical_label(
        offline_worker, bundle, hands, native_values, monkeypatch):
    path, manifest, _ = bundle
    make_hand(hands[1], finger_upper=.1)
    monkeypatch.setattr(dga, "TARGET_URDF_SHA256", dga.file_sha256(hands[1]))
    valid = native_values.copy()
    valid[11:] = .05
    target_incompatible = native_values.copy()
    target_incompatible[:3] = [2., -3., 4.]
    manifest["output"] = write_reference(path.parent, "outputs.npz", {"hand_qpos": np.stack([valid, target_incompatible])})
    sync_bundle_log(bundle)
    load_bundle(bundle)
    output = path.parent / "offline_inspection"
    report = offline_worker.inspect_saved_export(
        path, object_id=manifest["object_id"], geometry_hash=manifest["geometry_content_hash"],
        target_urdf=hands[1], output=output, expected_seed=17, expected_count=2)
    assert report["sample_count"] == 2
    assert report["candidate_count"] == 1
    assert report["rejection_count"] == 1
    assert report["evidence_origin"] == "manufactured_contract_fixture"
    assert report["execution_eligible"] is False
    assert report["native_inference_executed"] is False
    assert report["physics_steps"] == 0
    candidate = json.loads((output / "sample_0000" / "candidate.json").read_text())
    rejection = json.loads((output / "sample_0001" / "rejection.json").read_text())
    assert candidate["sample_index"] == 0
    assert candidate["evidence_origin"] == "manufactured_contract_fixture"
    assert candidate["execution_eligible"] is False
    assert rejection["sample_index"] == 1
    assert rejection["code"] == "joint_limit_violation"
    assert rejection["physical_failure_label"] is None
    assert rejection["evidence_origin"] == "manufactured_contract_fixture"
    assert rejection["execution_eligible"] is False
    assert rejection["native_inference_executed"] is False
    assert rejection["joint_targets_rad"]["WRJ2"] == target_incompatible[9]
    assert rejection["joint_targets_rad"]["WRJ1"] == target_incompatible[10]
    assert rejection["joint_targets_rad"]["FFJ4"] == .15
    np.testing.assert_allclose(np.asarray(rejection["object_from_forearm"])[:3, 3], [2., -3., 4.])
    assert not list(output.rglob("proposal.json"))


@pytest.mark.parametrize("seed,count,code", [(18, 2, "native_seed_mismatch"), (17, 3, "native_count_mismatch")])
def test_worker_request_binding_rejects_before_output(offline_worker, bundle, hands, seed, count, code):
    path, manifest, _ = bundle
    load_bundle(bundle)
    output = path.parent / "invalid_request"
    expect_code(code, lambda: offline_worker.inspect_saved_export(
        path, object_id=manifest["object_id"], geometry_hash=manifest["geometry_content_hash"],
        target_urdf=hands[1], output=output, expected_seed=seed, expected_count=count))
    assert not output.exists()


def test_worker_preserves_existing_evidence_directory(offline_worker, bundle, hands):
    path, manifest, _ = bundle
    load_bundle(bundle)
    output = path.parent / "existing_evidence"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("historical evidence")
    expect_code("output_exists", lambda: offline_worker.inspect_saved_export(
        path, object_id=manifest["object_id"], geometry_hash=manifest["geometry_content_hash"],
        target_urdf=hands[1], output=output))
    assert marker.read_text() == "historical evidence"
    assert sorted(p.name for p in output.iterdir()) == ["preserve.txt"]


def test_offline_module_and_worker_import_without_model_or_physics_runtime(offline_worker):
    script = """
import builtins, importlib.util, sys
denied = {'torch', 'mujoco', 'isaacgym', 'pytorch3d', 'csdf', 'hydra'}
original_import = builtins.__import__
def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.')[0] in denied:
        raise AssertionError('Offline inspection attempted runtime import: ' + name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = checked_import
import grasp_failure_prediction.proposals
provider_name = 'grasp_failure_prediction.proposals.dexgrasp_anything'
provider_spec = importlib.util.spec_from_file_location(provider_name, sys.argv[1])
provider = importlib.util.module_from_spec(provider_spec)
sys.modules[provider_name] = provider
provider_spec.loader.exec_module(provider)
worker_spec = importlib.util.spec_from_file_location('offline_worker_import_smoke', sys.argv[2])
worker = importlib.util.module_from_spec(worker_spec)
worker_spec.loader.exec_module(worker)
assert callable(worker.inspect_saved_export)
assert not denied.intersection(name.split('.')[0] for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script, dga.__file__, offline_worker.__file__],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
