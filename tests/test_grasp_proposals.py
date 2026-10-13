from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from grasp_failure_prediction.evaluation.schema import canonical_sha256
from grasp_failure_prediction.proposals.artifacts import (
    load_proposal,
    proposal_content_hash,
)
from grasp_failure_prediction.proposals.generate import generate_proposals
from grasp_failure_prediction.proposals.registry import load_provider_registry
from grasp_failure_prediction.proposals.schema import GraspProposal, RigidTransform


GEOMETRY_HASH = "sha256:45c1162a4f815d65c43254f2f499f53bde1eecea4d39912d8ce36d2c5cd92a81"


def _request(tmp_path: Path) -> Path:
    source = tmp_path / "saved.json"
    source.write_text(
        (Path(__file__).parent / "fixtures/provider_saved_cube.json").read_text()
    )
    request = {
        "schema_version": "proposal_request_v1",
        "request_id": "cube_saved_smoke",
        "provider": {"id": "shadow_saved_record_v1"},
        "object": {"id": "object01", "geometry_content_hash": GEOMETRY_HASH},
        "embodiment_id": "shadow_hand_right",
        "provider_input_path": "saved.json",
        "seed": 7,
        "sample_count": 1,
    }
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(request, sort_keys=False))
    return path


def test_saved_provider_generates_content_addressed_canonical_artifact(tmp_path):
    manifest = generate_proposals(
        _request(tmp_path), tmp_path / "proposals", project_root=tmp_path
    )
    assert manifest.provider_id == "shadow_saved_record_v1"
    assert len(manifest.proposals) == 1
    path = tmp_path / "proposals" / manifest.proposals[0].proposal_path
    proposal = load_proposal(path, expected_hash=manifest.proposals[0].proposal_hash)
    assert proposal.object.geometry_content_hash == GEOMETRY_HASH
    assert proposal.input_binding.observation_group_id.startswith("sha256:")
    assert proposal.input_binding.generation_request_group_id.startswith("sha256:")
    assert proposal_content_hash(proposal) == manifest.proposals[0].proposal_hash


def test_native_artifacts_are_optional_but_deep_audit_is_strict(tmp_path):
    manifest = generate_proposals(
        _request(tmp_path), tmp_path / "proposals", project_root=tmp_path
    )
    path = tmp_path / "proposals" / manifest.proposals[0].proposal_path
    raw = json.loads(path.read_text())
    raw["native_artifacts"] = [{
        "role": "provider_output",
        "path": "native/missing.bin",
        "sha256": "sha256:" + "0" * 64,
    }]
    proposal = GraspProposal.model_validate(raw)
    path.write_text(json.dumps(proposal.model_dump(mode="json")))
    load_proposal(path)
    with pytest.raises(FileNotFoundError, match="native proposal artifact"):
        load_proposal(path, verify_native_artifacts=True)


def test_rigid_transform_requires_a_unit_quaternion():
    with pytest.raises(ValueError, match="unit length"):
        RigidTransform(translation_m=(0, 0, 0), quaternion_xyzw=(0, 0, 0, 2))


def test_provider_registry_hash_is_stable():
    registry = load_provider_registry()
    spec = registry.resolve("hug_mano_shadow_v1")
    assert registry.config_hash(spec.provider_id) == canonical_sha256(spec)
    assert spec.compatible_execution_protocols == ("fixed_grasp_lift_v2",)


def test_http_provider_sends_hashed_inputs_and_accepts_canonical_response(
    tmp_path, monkeypatch
):
    local = generate_proposals(
        _request(tmp_path), tmp_path / "local", project_root=tmp_path
    )
    proposal_path = tmp_path / "local" / local.proposals[0].proposal_path
    proposal = json.loads(proposal_path.read_text())
    registry = load_provider_registry()
    spec = registry.resolve("dexgrasp_anything_shadow_v1")
    proposal["provider"]["id"] = spec.provider_id
    proposal["provider"]["config_hash"] = canonical_sha256(spec)
    (tmp_path / "object.obj").write_text("v 0 0 0\n")
    (tmp_path / "points.npy").write_bytes(b"test-point-cloud")
    request_payload = yaml.safe_load(_request(tmp_path).read_text())
    del request_payload["provider_input_path"]
    request_payload["provider_inputs"] = [
        {"role": "object_mesh", "path": "object.obj"},
        {"role": "object_point_cloud", "path": "points.npy"},
    ]
    request_path = tmp_path / "dga_request.yaml"
    request_path.write_text(yaml.safe_dump(request_payload, sort_keys=False))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"proposals": [proposal]}).encode()

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        assert timeout == 120
        assert body["provider_config_hash"] == canonical_sha256(spec)
        assert {item["role"] for item in body["inputs"]} == {
            "object_mesh", "object_point_cloud"
        }
        assert {item["path"] for item in body["inputs"]} == {
            "object.obj", "points.npy"
        }
        assert body["inputs"][0]["sha256"].startswith("sha256:")
        proposal["input_binding"] = body["expected_input_binding"]
        return Response()

    monkeypatch.setattr(
        "grasp_failure_prediction.proposals.generate.urlopen", fake_urlopen
    )
    manifest = generate_proposals(
        request_path,
        tmp_path / "remote",
        provider_id=spec.provider_id,
        endpoint="https://worker.invalid/v1/grasp-proposals",
        project_root=tmp_path,
    )
    assert manifest.provider_id == spec.provider_id


def test_provider_requirements_fail_before_worker_invocation(tmp_path, monkeypatch):
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        "grasp_failure_prediction.proposals.generate.urlopen", forbidden
    )
    with pytest.raises(ValueError, match="object_mesh.*object_point_cloud"):
        generate_proposals(
            _request(tmp_path),
            tmp_path / "remote",
            provider_id="dexgrasp_anything_shadow_v1",
            endpoint="https://worker.invalid/v1/grasp-proposals",
            project_root=tmp_path,
        )
    assert called is False
    assert not (tmp_path / "remote").exists()
