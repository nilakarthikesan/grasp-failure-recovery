from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter
from grasp_failure_prediction.proposals.providers import _write_mapping_record
from grasp_failure_prediction.proposals.translation import translate_mano_with_mapping_record
from test_retargeting import synthetic_hug_prediction


def test_recorded_legacy_solve_preserves_fresh_commands_and_raw_bounds():
    grasp, _ = synthetic_hug_prediction()
    expected = ShadowHandRetargeter().retarget(grasp)
    pose, record = translate_mano_with_mapping_record(grasp, "mano_right_to_shadow_hand_right_v1")
    np.testing.assert_array_equal(pose.qpos, expected.qpos)
    assert record["fresh_instance_per_proposal"]
    assert record["mapping_status"] == "ok"
    assert record["outcome"] is None
    assert len(record["raw_refinement_qpos_rad"]) == 20
    assert len(record["optimizer_status_codes"]) == 20
    assert record["optimizer_status_gate_passed"]
    assert record["provenance"]["profile_id"] == "legacy_shadow_vector_v1"
    limits = np.asarray(record["joint_limits_rad"])
    final_raw = np.asarray(record["raw_refinement_qpos_rad"][-1])
    np.testing.assert_array_equal(np.clip(final_raw, limits[:, 0], limits[:, 1]), pose.qpos)


@pytest.mark.parametrize("statuses", [[None] * 20, [-1] * 20, [1] * 19])
def test_missing_or_failed_status_cannot_qualify_a_finite_pose(monkeypatch, statuses):
    import grasp_failure_prediction.evaluation.fresh_mapping_qualification as capture
    monkeypatch.setattr(
        "grasp_failure_prediction.proposals.translation._official_mano_retargeter",
        lambda _: SimpleNamespace(joint_limits=np.array([[-1., 1.]]), optimizer_iterations=20),
    )
    monkeypatch.setattr(capture, "capture_solve", lambda *args: (object(), {
        "status": "returned", "optimizer_status_codes": statuses,
    }))
    pose, record = translate_mano_with_mapping_record(None, "ignored")
    assert pose is None
    assert record["mapping_status"] == "refused_invalid_retarget"
    assert record["outcome"] is None


def test_refusal_retains_partial_solver_trace(monkeypatch):
    import grasp_failure_prediction.evaluation.fresh_mapping_qualification as capture
    monkeypatch.setattr(
        "grasp_failure_prediction.proposals.translation._official_mano_retargeter",
        lambda _: SimpleNamespace(joint_limits=np.array([[-1., 1.]]), optimizer_iterations=20),
    )
    monkeypatch.setattr(capture, "capture_solve", lambda *args: (None, {
        "status": "refused", "raw_refinement_qpos_rad": [[.2]],
        "optimizer_status_codes": [-1], "message": "solver failed",
    }))
    pose, record = translate_mano_with_mapping_record(None, "ignored")
    assert pose is None
    assert record["raw_refinement_qpos_rad"] == [[.2]]
    assert record["message"] == "solver failed"
    assert record["outcome"] is None


def test_nonfinite_refusal_evidence_is_explicit_json_not_lost(tmp_path):
    path = tmp_path / "mapping_record.json"
    _write_mapping_record(path, {"mapping_status": "refused_invalid_retarget", "outcome": None,
                                 "raw_refinement_qpos_rad": [[float("nan"), float("inf")]]})
    record = json.loads(path.read_text())
    assert record["raw_refinement_qpos_rad"] == [["nan", "inf"]]
    assert record["outcome"] is None


def test_hug_subprocess_uses_only_explicit_dependency_bundle(tmp_path, monkeypatch):
    from grasp_failure_prediction.proposals.providers import HugProposalProvider
    from grasp_failure_prediction.proposals.registry import load_provider_registry
    from grasp_failure_prediction.proposals.schema import ProposalRequest
    bundle = tmp_path / "declared_dependencies"; bundle.mkdir()
    hug = tmp_path / "hug"; hug.mkdir()
    checkpoint = tmp_path / "weights"; checkpoint.write_bytes(b"manufactured")
    observation = tmp_path / "capture"; observation.mkdir()
    geometry = "sha256:" + "0" * 64
    (observation / "scene_contract.json").write_text(json.dumps({"object_id": "object01", "geometry_content_hash": geometry}))
    (observation / "observation_manifest.json").write_text("{}")
    np.save(observation / "T_world_camera.npy", np.eye(4))
    request = ProposalRequest(request_id="runtime_test", provider={"id": "hug_mano_shadow_v1"},
        object={"id": "object01", "geometry_content_hash": geometry}, observation_path="capture")
    provider = HugProposalProvider(load_provider_registry().resolve("hug_mano_shadow_v1"),
        project_root=tmp_path, hug_root=hug, checkpoint=checkpoint, runtime_dependency_dirs=(bundle,))
    calls = []
    def stop(command, **kwargs):
        calls.append(command)
        raise RuntimeError("stopped before real inference")
    monkeypatch.setattr("grasp_failure_prediction.proposals.providers.subprocess.run", stop)
    with pytest.raises(RuntimeError, match="before real inference"):
        provider.generate(request, tmp_path / "output")
    command = calls[0]
    assert command[1] == str(provider.adapter_root / "scripts/infer_sim_observation.py")
    assert command.count("--dependency-dir") == 1
    assert command[command.index("--dependency-dir") + 1] == str(bundle)


def test_hug_runtime_refuses_missing_declared_bundle_before_inference(tmp_path):
    from grasp_failure_prediction.proposals.providers import HugProposalProvider
    with pytest.raises(FileNotFoundError, match="dependency directory"):
        HugProposalProvider(None, project_root=tmp_path, hug_root=tmp_path,
                            checkpoint=tmp_path / "unused", runtime_dependency_dirs=(tmp_path / "missing",))
