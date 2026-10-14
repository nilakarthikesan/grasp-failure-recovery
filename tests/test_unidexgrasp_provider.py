"""Synthetic adapter-contract tests, never official model output or eval data."""

import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation.schema import canonical_json, canonical_sha256
from grasp_failure_prediction.proposals.artifacts import file_sha256, load_proposal
from grasp_failure_prediction.proposals.dexgraspnet import SOURCE_JOINT_NAMES, SOURCE_TIP_SITES
from grasp_failure_prediction.proposals.registry import load_provider_registry
from grasp_failure_prediction.proposals.schema import ProposalRequest, RigidTransform
from grasp_failure_prediction.proposals.translation import TranslationError, translate_native_pose
from grasp_failure_prediction.proposals import unidexgrasp as adapter
from grasp_failure_prediction.proposals.generate import generate_proposals


HASH = "sha256:" + "a" * 64
IDENTITY = {"translation_m": [0., 0., 0.], "quaternion_xyzw": [0., 0., 0., 1.]}


def _request(count=1):
    return ProposalRequest.model_validate({
        "request_id": "synthetic_unidex_test", "provider": {"id": "unidexgrasp_shadow_v1"},
        "object": {"id": "synthetic_object", "geometry_content_hash": HASH},
        "provider_input_path": "bundle/native_manifest.json", "seed": 11, "sample_count": count,
    })


def _synthetic_source(path: Path, *, root_pose='pos="0.01 -0.02 0.03"'):
    """An artificial skeleton. It does not copy an official hand asset.

    Positive tests explicitly override the certificate to this synthetic
    skeleton. Production rejects it; its success is not a scientific result.
    """
    bodies = []
    for index, name in enumerate(SOURCE_JOINT_NAMES):
        bodies.append(f'<body name="synthetic_{index}" pos="0 0 {index / 1000}"><inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/><joint name="{name}" type="hinge" axis="0 0 1" range="-2 2" limited="true"/></body>')
    sites = ''.join(f'<site name="{name}" pos="{index / 100} 0 0"/>' for index, name in enumerate(SOURCE_TIP_SITES))
    path.write_text(f'<mujoco model="synthetic_adapter_test"><compiler angle="radian"/><worldbody><body name="robot0:palm" {root_pose}><inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/>{"".join(bodies)}{sites}</body></worldbody></mujoco>')
    return path


def _allow_synthetic_certificate(source, monkeypatch):
    model = mujoco.MjModel.from_xml_path(str(source))
    monkeypatch.setattr(adapter, "KINEMATICS_CERTIFICATE_SHA256", adapter._certificate(model))


def _bundle(root: Path, *, count=1):
    root.mkdir()
    source = _synthetic_source(root / "source_hand.xml")
    output = root / "model_output.npz"
    values = np.zeros((count, 28))
    values[:, :3] = [0.04, -0.05, 0.06]
    values[:, 6:] = np.arange(22) / 100.
    np.savez(output, hand_pose=values)
    report = {
        "schema_version": "unidexgrasp_inference_report_v1", "classification": "fresh_model_inference",
        "source_repository": adapter.SOURCE_REPOSITORY, "source_commit": adapter.SOURCE_COMMIT,
        "object_id": "synthetic_object", "geometry_content_hash": HASH,
        "sample_count": count, "seeds": [11 + index for index in range(count)],
        "object_scale": 0.1, "object_to_input": IDENTITY,
        "checkpoint_sha256": {"rotation": HASH, "pose": HASH},
        "config_sha256": {"rotation": HASH, "pose": HASH},
        "source_hand_mjcf_sha256": file_sha256(source), "output_sha256": file_sha256(output),
        "input_manifest_sha256": HASH, "point_cloud_sha256": HASH, "tta_applied": False,
        "test_fixture_origin": "synthetic_unit_contract_only_not_official_model_output",
    }
    report_file = root / "inference_report.json"
    report_file.write_bytes(canonical_json(report))
    manifest = {
        "schema_version": "unidexgrasp_native_bundle_v1", "source_repository": adapter.SOURCE_REPOSITORY,
        "source_commit": adapter.SOURCE_COMMIT, "object_id": "synthetic_object",
        "geometry_content_hash": HASH, "object_scale": 0.1, "object_to_input": IDENTITY,
        "pose_frame": "input_point_cloud", "translation_units": "m", "rotation_representation": "axis_angle",
        "joint_units": "rad", "joint_names": list(SOURCE_JOINT_NAMES), "pose_field": "hand_pose",
        "output_path": output.name, "output_sha256": file_sha256(output),
        "report_path": report_file.name, "report_sha256": file_sha256(report_file),
        "source_hand_mjcf_path": source.name, "source_hand_mjcf_sha256": file_sha256(source),
    }
    path = root / "native_manifest.json"
    path.write_bytes(canonical_json(manifest))
    return path, source, report_file


def test_prediction_conversion_transposes_upstream_rotation_and_does_not_scale_translation():
    # Native sampled rotation canonicalizes input; a 90-degree Z rotation
    # rotates canonical translation into input through its transpose.
    sampled = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    joints = np.arange(22) / 10.
    pose = adapter.compose_unidexgrasp_prediction([0.1, 0.2, 0.3], sampled, joints)
    np.testing.assert_allclose(pose[:3], [0.2, -0.1, 0.3], atol=1e-12)
    np.testing.assert_allclose(adapter._axis_angle_matrix(pose[3:6]), sampled.T, atol=1e-12)
    np.testing.assert_array_equal(pose[6:], joints)
    with pytest.raises(TranslationError, match="proper rotation"):
        adapter.compose_unidexgrasp_prediction([0., 0., 0.], np.diag([1., 1., -1.]), joints)


def test_decoder_composes_nonidentity_input_hand_and_actual_source_palm(tmp_path, monkeypatch):
    source = _synthetic_source(tmp_path / "synthetic.xml", root_pose='pos="0.01 -0.02 0.03" quat="0.7071067811865476 0 0 0.7071067811865476"')
    _allow_synthetic_certificate(source, monkeypatch)
    values = np.zeros(28)
    values[:3] = [0.04, 0.05, 0.06]
    values[3:6] = [np.pi / 2., 0., 0.]
    values[6:] = np.arange(22) / 100.
    transform = RigidTransform(translation_m=(0.1, 0.2, 0.3), quaternion_xyzw=(0., 0., np.sqrt(0.5), np.sqrt(0.5)))
    native = adapter.decode_unidexgrasp_hand_pose(values, object_to_input=transform, source_hand_mjcf=source, source_hand_sha256=file_sha256(source), kinematics_output=tmp_path / "derived.xml")
    # object_R = Z90, input_R = X90, source palm offset + source_R=Z90.
    object_rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    input_rotation = np.asarray([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    expected_translation = np.asarray(transform.translation_m) + object_rotation @ (values[:3] + input_rotation @ [0.01, -0.02, 0.03])
    np.testing.assert_allclose(native.object_to_root.translation_m, expected_translation, atol=1e-12)
    expected_rotation = object_rotation @ input_rotation @ object_rotation
    q = native.object_to_root.quaternion_xyzw
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(q)[[3, 0, 1, 2]])
    np.testing.assert_allclose(rotation.reshape(3, 3), expected_rotation, atol=1e-12)
    translated = translate_native_pose(native, adapter.TRANSLATION_STRATEGY)
    assert translated.joint_targets_rad["FFJ4"] == -values[6]
    assert translated.joint_targets_rad["MFJ4"] == -values[10]
    assert translated.joint_targets_rad["THJ1"] == -values[-1]
    assert translated.joint_targets_rad["WRJ1"] == translated.joint_targets_rad["WRJ2"] == 0.
    assert native.payload["actual_source_hand_sha256"] == file_sha256(source)


def test_kinematics_certificate_rejects_arbitrary_hand_and_inherited_axis_change(tmp_path, monkeypatch):
    source = _synthetic_source(tmp_path / "synthetic.xml")
    with pytest.raises(TranslationError) as exc:
        adapter._prepare_source(source, tmp_path / "derived.xml", file_sha256(source))
    assert exc.value.code == "source_kinematics_not_equivalent"
    _allow_synthetic_certificate(source, monkeypatch)
    adapter._prepare_source(source, tmp_path / "accepted.xml", file_sha256(source))
    # Effective inherited defaults participate in the certificate.
    xml = source.read_text().replace('<worldbody>', '<default><joint axis="1 0 0"/></default><worldbody>').replace('axis="0 0 1" ', '')
    source.write_text(xml)
    with pytest.raises(TranslationError) as exc:
        adapter._prepare_source(source, tmp_path / "changed.xml", file_sha256(source))
    assert exc.value.code == "source_kinematics_not_equivalent"


@pytest.mark.parametrize("change", ["hash", "path", "revision", "scale", "seed", "frame", "joints", "pose_field", "report_transform", "checkpoints", "config"])
def test_bundle_contract_fails_closed_before_any_output(tmp_path, change):
    bundle, _, report_file = _bundle(tmp_path / "bundle")
    manifest = json.loads(bundle.read_text())
    report = json.loads(report_file.read_text())
    if change == "hash":
        manifest["output_sha256"] = HASH
    elif change == "path":
        manifest["output_path"] = "../model_output.npz"
    elif change == "revision":
        manifest["source_commit"] = "0" * 40
    elif change == "scale":
        manifest["object_scale"] = 2.
    elif change == "seed":
        report["seeds"] = [12]
    elif change == "frame":
        manifest["pose_frame"] = "object"
    elif change == "joints":
        manifest["joint_names"].reverse()
    elif change == "pose_field":
        manifest["pose_field"] = "guess_best"
    elif change == "report_transform":
        report["object_to_input"]["translation_m"][0] = 0.1
    elif change == "checkpoints":
        report.pop("checkpoint_sha256")
    elif change == "config":
        report.pop("config_sha256")
    report_file.write_bytes(canonical_json(report))
    manifest["report_sha256"] = file_sha256(report_file)
    bundle.write_bytes(canonical_json(manifest))
    with pytest.raises((TranslationError, ValueError)):
        adapter._load_export(bundle, _request())
    assert not (tmp_path / "proposals").exists()


@pytest.mark.parametrize("bad", ["object_dtype", "nan", "count"])
def test_numeric_native_payload_rejects_pickle_nonfinite_and_wrong_count(tmp_path, bad):
    bundle, _, report_file = _bundle(tmp_path / "bundle")
    manifest = json.loads(bundle.read_text())
    report = json.loads(report_file.read_text())
    value = np.zeros((1, 28), dtype=object if bad == "object_dtype" else float)
    if bad == "nan": value[0, 0] = np.nan
    if bad == "count": value = np.zeros((2, 28))
    output = bundle.parent / manifest["output_path"]
    np.savez(output, hand_pose=value)
    manifest["output_sha256"] = report["output_sha256"] = file_sha256(output)
    report_file.write_bytes(canonical_json(report))
    manifest["report_sha256"] = file_sha256(report_file)
    bundle.write_bytes(canonical_json(manifest))
    with pytest.raises((TranslationError, ValueError)):
        adapter._load_export(bundle, _request())


def test_provider_synthetic_contract_uses_real_mapping_and_keeps_checkpoint_provenance(tmp_path, monkeypatch):
    bundle, source, _ = _bundle(tmp_path / "bundle", count=2)
    _allow_synthetic_certificate(source, monkeypatch)
    # Synthetic skeleton deliberately is not anatomically correct. The FK
    # validator is replaced only in this unit contract test, never production.
    monkeypatch.setattr(adapter, "validate_fk_translation", lambda native, translated, **kwargs: {"maximum_fingertip_error_m": 0., "synthetic_unit_test": True})
    spec = load_provider_registry().resolve("unidexgrasp_shadow_v1")
    provider = adapter.UniDexGraspProposalProvider(spec, project_root=tmp_path)
    paths = provider.generate(_request(2), tmp_path / "proposals")
    assert len(paths) == 2
    first, second = map(load_proposal, paths)
    assert first.provider.code_revision == adapter.SOURCE_COMMIT
    assert first.provider.checkpoint_hash == canonical_sha256({"rotation": HASH, "pose": HASH})
    assert first.generation.seed == 11 and second.generation.seed == 12
    assert first.provider_metrics["classification"] == "native_model_export"
    assert first.provider_metrics["object_scale"] == 0.1
    # Translation remains physical meters, not multiplied by object_scale.
    np.testing.assert_allclose(first.object_to_palm.translation_m, [0.05, -0.07, 0.09])
    assert first.input_binding.input_manifest_hash == file_sha256(bundle)
    for reference in first.native_artifacts:
        assert file_sha256(paths[0].parent / reference.path) == reference.sha256
    copied = paths[0].parent / "native/native_manifest.json"
    adapter._load_export(copied, _request(2))


def test_fk_gate_failure_writes_no_canonical_proposals(tmp_path, monkeypatch):
    _, source, _ = _bundle(tmp_path / "bundle")
    _allow_synthetic_certificate(source, monkeypatch)
    def reject(*args, **kwargs):
        raise TranslationError("forward_kinematics_residual_exceeded", "synthetic rejection")
    monkeypatch.setattr(adapter, "validate_fk_translation", reject)
    provider = adapter.UniDexGraspProposalProvider(load_provider_registry().resolve("unidexgrasp_shadow_v1"), project_root=tmp_path)
    with pytest.raises(TranslationError, match="synthetic rejection"):
        provider.generate(_request(), tmp_path / "proposals")
    assert not (tmp_path / "proposals").exists()


def test_public_generation_dispatch_accepts_only_certified_synthetic_unit_contract(tmp_path, monkeypatch):
    _, source, _ = _bundle(tmp_path / "bundle")
    _allow_synthetic_certificate(source, monkeypatch)
    monkeypatch.setattr(adapter, "validate_fk_translation", lambda *args, **kwargs: {"maximum_fingertip_error_m": 0., "synthetic_unit_test": True})
    request = tmp_path / "request.json"
    request.write_bytes(canonical_json(_request()))
    manifest = generate_proposals(request, tmp_path / "out", project_root=tmp_path)
    assert manifest.provider_id == "unidexgrasp_shadow_v1"
    assert len(manifest.proposals) == 1
    proposal = load_proposal(tmp_path / "out" / manifest.proposals[0].proposal_path)
    assert proposal.provider.id == manifest.provider_id
    assert proposal.input_binding.generation_request_group_id == canonical_sha256(_request())


def test_fresh_hook_rejects_stale_original_input_binding(tmp_path):
    bundle, _, _ = _bundle(tmp_path / "bundle")
    original = tmp_path / "input.json"
    original.write_text('{"schema_version":"unidexgrasp_object_input_v1"}')
    provider = adapter.UniDexGraspProposalProvider(load_provider_registry().resolve("unidexgrasp_shadow_v1"), project_root=tmp_path)
    with pytest.raises(TranslationError) as exc:
        provider.generate_from_bundle(_request(), tmp_path / "out", bundle, binding_path=original)
    assert exc.value.code == "native_input_binding_mismatch"
    assert not (tmp_path / "out").exists()


def test_worker_module_import_is_side_effect_free_and_validates_input(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/infer_unidexgrasp_proposals.py"
    spec = importlib.util.spec_from_file_location("synthetic_unidex_worker_import", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points = tmp_path / "synthetic_points.npy"
    np.save(points, np.ones((1024, 3), dtype=np.float32))
    raw = {
        "schema_version": "unidexgrasp_object_input_v1", "object_id": "synthetic_object",
        "geometry_content_hash": HASH, "object_scale": 0.1, "object_to_input": IDENTITY,
        "point_cloud_path": points.name, "point_cloud_sha256": file_sha256(points),
        "point_cloud_units": "m", "point_cloud_frame": "input_point_cloud", "plane_input": [0., 0., 1., 0.],
    }
    manifest = tmp_path / "input.json"
    manifest.write_bytes(canonical_json(raw))
    _, _, _, loaded = module.read_worker_input(manifest, _request())
    np.testing.assert_array_equal(loaded, np.ones((1024, 3)))
    raw["point_cloud_units"] = "mm"
    manifest.write_bytes(canonical_json(raw))
    with pytest.raises(ValueError, match="physical meters"):
        module.read_worker_input(manifest, _request())
    import sys
    import types
    monkeypatch.setitem(sys.modules, "network.synthetic_wrong_location", types.SimpleNamespace(__file__=str(tmp_path / "outside_checkout.py")))
    with pytest.raises(ValueError, match="outside its pinned generation checkout"):
        module._assert_official_modules(tmp_path / "pinned_generation")
