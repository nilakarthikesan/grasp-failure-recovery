"""Synthetic unit contracts only: no inference, simulation or evaluation data."""

from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_failure_prediction.proposals import contactgen
from grasp_failure_prediction.proposals.artifacts import load_proposal
from grasp_failure_prediction.proposals.contactgen import (
    CONTACTGEN_SOURCE_REVISION, ContactGenNativeBundle, ContactGenObjectInput,
    ContactGenOptimizedSample, ContactGenProposalProvider, axis_angle_rotation,
    contactgen_mano_landmarks, convert_contactgen_bundle, load_contactgen_bundle,
)
from grasp_failure_prediction.proposals.registry import load_provider_registry
from grasp_failure_prediction.proposals.schema import ProposalRequest


GEOMETRY = "sha256:" + "1" * 64
SYNTHETIC_MESH = "v 0.01 -0.02 0.03\nv 0.03 -0.02 0.03\nv 0.02 0.01 0.03\nf 1 2 3\n"
SYNTHETIC_MESH_SHA256 = "sha256:" + hashlib.sha256(SYNTHETIC_MESH.encode()).hexdigest()


def _synthetic_bundle():
    # These arbitrary array values test indexing/transforms; they are not model
    # outputs and must never be used as evaluation grasps.
    rotation = axis_angle_rotation((0, 0, 0.5))
    frames = np.tile(np.eye(4), (16, 1, 1))
    frames[:, :3, :3] = rotation
    frames[:, :3, 3] = np.arange(48).reshape(16, 3) * 0.001
    vertices = np.arange(778 * 3).reshape(778, 3) * 0.00001
    return {
        "schema_version": "contactgen_native_bundle_v1",
        "source_revision": CONTACTGEN_SOURCE_REVISION,
        "generation_stage": "refined_mano",
        "mano_configuration": "right_pca26_nonflat_mean",
        "coordinate_frame": "vertex_mean_centered_object_mesh", "units": "meters",
        "object_input": {
            "schema_version": "contactgen_object_input_v1",
            "object": {"id": "synthetic_object", "geometry_content_hash": GEOMETRY},
            "mesh_path": "synthetic_mesh.obj", "mesh_sha256": SYNTHETIC_MESH_SHA256,
            "units": "meters",
            "object_to_mesh": {
                "translation_m": [0.1, -0.2, 0.3],
                "quaternion_xyzw": [0, 0, 2**-0.5, 2**-0.5],
            },
        },
        "mesh_vertex_mean_m": [0.02, -0.01, 0.03],
        "checkpoint_sha256": "sha256:" + "3" * 64,
        "hand_sdf_checkpoint_sha256": "sha256:" + "4" * 64,
        "mano_asset_sha256": "sha256:" + "5" * 64,
        "global_iterations": 200, "pose_iterations": 1000,
        "samples": [{
            "sample_index": 0, "seed": 7,
            "global_pose_axis_angle_rad": [0, 0, 0.5],
            "mano_pose_pca": [0] * 26, "mano_shape": [0] * 10,
            # Deliberately different from frame[0] to test shaped wrist origin.
            "mano_translation_m": [0.004, 0.005, 0.006],
            "hand_vertices_centered_m": vertices.tolist(),
            "hand_frames_centered_m": frames.tolist(),
        }],
    }


def test_mano_reorders_tips_and_restores_nonidentity_object_frame():
    bundle = ContactGenNativeBundle.model_validate(_synthetic_bundle())
    sample = bundle.samples[0]
    grasp = contactgen_mano_landmarks(sample, bundle.object_input, bundle.mesh_vertex_mean_m)
    mesh_object = contactgen.rigid_transform_matrix(bundle.object_input.object_to_mesh)
    frames = np.asarray(sample.hand_frames_centered_m)
    vertices = np.asarray(sample.hand_vertices_centered_m)
    assert grasp.landmarks_3d.shape == (21, 3)
    for output_index, source_point in (
        (0, frames[0, :3, 3]), (1, frames[13, :3, 3]),
        (4, vertices[745]), (8, vertices[317]), (12, vertices[444]),
        (16, vertices[556]), (20, vertices[673]),
    ):
        expected = mesh_object[:3, :3] @ (source_point + bundle.mesh_vertex_mean_m) + mesh_object[:3, 3]
        np.testing.assert_allclose(grasp.landmarks_3d[output_index], expected, atol=1e-12)
    np.testing.assert_allclose(grasp.T_camera_wrist[:3, :3], mesh_object[:3, :3] @ frames[0, :3, :3])
    np.testing.assert_allclose(grasp.T_camera_wrist[:3, 3], grasp.landmarks_3d[0])
    # Native outputs are in meters; no upstream Manopth *1000 is applied.
    assert np.max(np.abs(grasp.landmarks_3d)) < 1


@pytest.mark.parametrize("field,value", [
    ("units", "millimeters"), ("generation_stage", "contact_maps"),
    ("mano_configuration", "left_pca26_nonflat_mean"),
    ("coordinate_frame", "world"), ("source_revision", "0" * 40),
])
def test_native_contract_rejects_ambiguous_or_incompatible_exports(field, value):
    raw = _synthetic_bundle()
    raw[field] = value
    with pytest.raises(ValueError):
        ContactGenNativeBundle.model_validate(raw)


@pytest.mark.parametrize("mutation", ["reflection", "shape", "global_rotation", "nonfinite"])
def test_native_contract_rejects_bad_mano_geometry(mutation):
    raw = _synthetic_bundle()
    sample = raw["samples"][0]
    if mutation == "reflection":
        sample["hand_frames_centered_m"][1][2][2] = -1
    elif mutation == "shape":
        sample["mano_pose_pca"] = [0] * 45
    elif mutation == "global_rotation":
        sample["global_pose_axis_angle_rad"] = [0, 0, 0]
    else:
        sample["hand_vertices_centered_m"][0][0] = float("nan")
    with pytest.raises(ValueError):
        ContactGenNativeBundle.model_validate(raw)


def test_contact_only_exports_need_official_refinement(tmp_path):
    path = tmp_path / "contact_map.json"
    path.write_text(json.dumps({"contacts_object": [0.5], "generation_stage": "contact_maps"}))
    with pytest.raises(ValueError, match="optimize_pose"):
        load_contactgen_bundle(path)


def _request(tmp_path, raw=None):
    raw = raw or _synthetic_bundle()
    source = tmp_path / "synthetic_native.json"
    source.write_text(json.dumps(raw))
    (tmp_path / "synthetic_mesh.obj").write_text(SYNTHETIC_MESH)
    return ProposalRequest.model_validate({
        "request_id": "synthetic_contract_only", "provider": {"id": "contactgen_mano_shadow_v1"},
        "object": raw["object_input"]["object"], "provider_input_path": source.name,
        "seed": 7, "sample_count": 1,
    })


def _stub_mesh_loader(monkeypatch):
    class SyntheticMesh:
        vertices = np.array([[0.01, -0.02, 0.03], [0.03, -0.02, 0.03], [0.02, 0.01, 0.03]])
        faces = np.array([[0, 1, 2]])
    monkeypatch.setitem(sys.modules, "trimesh", SimpleNamespace(Trimesh=SyntheticMesh, load=lambda path: SyntheticMesh()))


def _stub_translation(monkeypatch, mean_error=0.001):
    _stub_mesh_loader(monkeypatch)
    # Monkeypatch both retargeting and FK so tests never generate a model pose
    # or run a MuJoCo simulation/forward pass.
    pose = SimpleNamespace(joint_names=("joint_a",), qpos=np.array([0.1]))
    monkeypatch.setattr(contactgen, "translate_mano_with_official_retargeter", lambda grasp, strategy: pose)
    class Validator:
        def __init__(self, root):
            pass
        def validate(self, value):
            return SimpleNamespace(mean_fingertip_error_m=mean_error, maximum_fingertip_error_m=0.002)
    monkeypatch.setattr(contactgen, "ShadowPoseValidator", Validator)
    monkeypatch.setattr(contactgen, "contactgen_object_palm_pose", lambda landmarks, value, validator: contactgen._matrix_pose(landmarks.T_camera_wrist))


def test_import_provider_preserves_hashes_binding_and_native_mode(tmp_path, monkeypatch):
    from grasp_failure_prediction.proposals.generate import _expected_input_binding
    _stub_translation(monkeypatch)
    request = _request(tmp_path)
    spec = load_provider_registry().resolve(request.provider.id)
    outputs = ContactGenProposalProvider(spec, project_root=tmp_path).generate(request, tmp_path / "output")
    proposal = load_proposal(outputs[0], verify_native_artifacts=True)
    assert proposal.provider.checkpoint_hash == "sha256:" + "3" * 64
    assert proposal.provider.code_revision == CONTACTGEN_SOURCE_REVISION
    assert proposal.provider_metrics["fresh_inference"] is False
    assert proposal.provider_metrics["generation_stage"] == "refined_mano"
    assert proposal.provider_metrics["mano_asset_sha256"] == "sha256:" + "5" * 64
    assert proposal.provider_metrics["hand_sdf_checkpoint_sha256"] == "sha256:" + "4" * 64
    assert proposal.input_binding.model_dump() == _expected_input_binding(request, tmp_path)


def test_failed_alignment_writes_no_proposals(tmp_path, monkeypatch):
    _stub_translation(monkeypatch, mean_error=0.02)
    request = _request(tmp_path)
    spec = load_provider_registry().resolve(request.provider.id)
    with pytest.raises(ValueError, match="alignment gate"):
        ContactGenProposalProvider(spec, project_root=tmp_path).generate(request, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_bundle_request_seed_and_geometry_are_bound_before_retargeting(tmp_path, monkeypatch):
    request = _request(tmp_path)
    spec = load_provider_registry().resolve(request.provider.id)
    monkeypatch.setattr(contactgen, "ShadowPoseValidator", lambda root: pytest.fail("must reject before FK"))
    for field, value in (("seed", 8), ("sample_count", 2)):
        bundle = ContactGenNativeBundle.model_validate(_synthetic_bundle())
        with pytest.raises(ValueError, match="seed|sample count"):
            convert_contactgen_bundle(bundle, request.model_copy(update={field: value}), spec, project_root=tmp_path, output_directory=tmp_path / "output")
    raw = _synthetic_bundle()
    raw["object_input"]["object"]["geometry_content_hash"] = "sha256:" + "6" * 64
    with pytest.raises(ValueError, match="object/geometry"):
        convert_contactgen_bundle(ContactGenNativeBundle.model_validate(raw), request, spec, project_root=tmp_path, output_directory=tmp_path / "output")


def test_object_input_rejects_unbound_mesh_paths_and_units():
    raw = _synthetic_bundle()["object_input"]
    for field, value in (("mesh_path", "../outside.obj"), ("mesh_path", "/tmp/outside.obj"), ("units", "millimeters")):
        malformed = copy.deepcopy(raw)
        malformed[field] = value
        with pytest.raises(ValueError):
            ContactGenObjectInput.model_validate(malformed)


def test_worker_rejects_changed_upstream_code_without_importing_cuda(monkeypatch, tmp_path):
    script = Path(__file__).parents[1] / "scripts/infer_contactgen_object.py"
    module_spec = importlib.util.spec_from_file_location("contactgen_worker_test", script)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: CONTACTGEN_SOURCE_REVISION)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match="modified tracked files"):
        module.verify_contactgen_source(tmp_path)


def test_native_mesh_hash_and_center_are_checked_before_translation(tmp_path):
    pytest.importorskip("trimesh")
    bundle = ContactGenNativeBundle.model_validate(_synthetic_bundle())
    path = tmp_path / "synthetic_mesh.obj"
    path.write_text(SYNTHETIC_MESH)
    contactgen.validate_contactgen_mesh(bundle.object_input, tmp_path, expected_vertex_mean_m=bundle.mesh_vertex_mean_m)
    with pytest.raises(ValueError, match="vertex mean"):
        contactgen.validate_contactgen_mesh(bundle.object_input, tmp_path, expected_vertex_mean_m=(0, 0, 0))
    path.write_text(SYNTHETIC_MESH + "# modified synthetic fixture\n")
    with pytest.raises(ValueError, match="mesh hash"):
        contactgen.validate_contactgen_mesh(bundle.object_input, tmp_path)


def test_palm_rotation_matches_hug_origin_policy_with_nonidentity_frames(monkeypatch):
    bundle = ContactGenNativeBundle.model_validate(_synthetic_bundle())
    landmarks = contactgen_mano_landmarks(bundle.samples[0], bundle.object_input, bundle.mesh_vertex_mean_m)
    base_palm = axis_angle_rotation((0.2, 0, 0))
    validator = SimpleNamespace(
        model=object(), data=SimpleNamespace(xmat=np.array([base_palm.reshape(9)])),
        set_pose=lambda pose: None,
    )
    monkeypatch.setattr(contactgen.mujoco, "mj_name2id", lambda *args: 0)
    result = contactgen.contactgen_object_palm_pose(landmarks, object(), validator)
    matrix = contactgen.rigid_transform_matrix(result)
    np.testing.assert_allclose(matrix[:3, 3], landmarks.T_camera_wrist[:3, 3])
    np.testing.assert_allclose(matrix[:3, :3], landmarks.T_camera_wrist[:3, :3] @ contactgen.MANO_TO_OPERATOR_RIGHT.T @ base_palm, atol=1e-12)


def test_runtime_assets_require_exact_hash_before_loading(tmp_path):
    script = Path(__file__).parents[1] / "scripts/infer_contactgen_object.py"
    module_spec = importlib.util.spec_from_file_location("contactgen_asset_test", script)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    path = tmp_path / "synthetic_checkpoint_placeholder.bin"
    path.write_bytes(b"synthetic hash fixture only; not a checkpoint")
    expected = contactgen.file_sha256(path)
    assert module.verify_runtime_asset(path, expected, "checkpoint") == path
    with pytest.raises(ValueError, match="asset pin"):
        module.verify_runtime_asset(path, "sha256:" + "0" * 64, "checkpoint")
    with pytest.raises(ValueError, match="expected hash"):
        module.verify_runtime_asset(path, "bad", "checkpoint")


def test_fresh_object_input_gets_worker_guidance(tmp_path):
    path = tmp_path / "object_input.json"
    path.write_text(json.dumps(_synthetic_bundle()["object_input"]))
    with pytest.raises(ValueError, match="fresh inference.*external-command"):
        load_contactgen_bundle(path)


@pytest.mark.parametrize("channels", [
    {"observation_path": "synthetic_observation"},
    {"provider_inputs": [{"role": "provider_input", "path": "synthetic_native.json"}]},
])
def test_input_channel_ambiguity_rejected_before_retargeting(tmp_path, monkeypatch, channels):
    request = _request(tmp_path)
    raw = request.model_dump(mode="json")
    raw.update(channels)
    malformed = ProposalRequest.model_validate(raw)
    spec = load_provider_registry().resolve(request.provider.id)
    monkeypatch.setattr(contactgen, "ShadowPoseValidator", lambda root: pytest.fail("must reject ambiguous input before FK"))
    with pytest.raises(ValueError, match="provider input channel|exactly one"):
        ContactGenProposalProvider(spec, project_root=tmp_path).generate(malformed, tmp_path / "output")


def test_sole_provider_input_role_has_exact_native_binding(tmp_path, monkeypatch):
    from grasp_failure_prediction.proposals.generate import _expected_input_binding
    _stub_translation(monkeypatch)
    request = _request(tmp_path)
    raw = request.model_dump(mode="json")
    raw["provider_inputs"] = [{"role": "provider_input", "path": raw.pop("provider_input_path")}]
    request = ProposalRequest.model_validate(raw)
    spec = load_provider_registry().resolve(request.provider.id)
    outputs = ContactGenProposalProvider(spec, project_root=tmp_path).generate(request, tmp_path / "output")
    assert load_proposal(outputs[0]).input_binding.model_dump() == _expected_input_binding(request, tmp_path)


def test_registered_contactgen_native_path_works_through_common_generator(tmp_path, monkeypatch):
    from grasp_failure_prediction.proposals.generate import generate_proposals
    from grasp_failure_prediction.proposals.artifacts import proposal_content_hash
    _stub_translation(monkeypatch)
    request = _request(tmp_path)
    request_path = tmp_path / "synthetic_request.json"
    request_path.write_text(json.dumps(request.model_dump(mode="json")))
    output = tmp_path / "synthetic_unit_outputs"
    manifest = generate_proposals(request_path, output, project_root=tmp_path)
    assert manifest.provider_id == "contactgen_mano_shadow_v1"
    assert len(manifest.proposals) == 1
    entry = manifest.proposals[0]
    proposal = load_proposal(output / entry.proposal_path, expected_hash=entry.proposal_hash, verify_native_artifacts=True)
    assert proposal_content_hash(proposal) == entry.proposal_hash
    assert proposal.generation.seed == 7
    assert proposal.generation.sample_index == 0
    assert proposal.provider_metrics["generation_mode"] == "native_model_output_import"
    assert proposal.input_binding.input_manifest_hash == contactgen.file_sha256(tmp_path / "synthetic_native.json")
