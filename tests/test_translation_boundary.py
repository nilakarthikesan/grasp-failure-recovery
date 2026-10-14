from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from grasp_failure_prediction.evaluation.object_assets import load_object_geometry
from grasp_failure_prediction.proposals.execution import ProposalRejected, _scene_preflight, _validate_pose
from grasp_failure_prediction.proposals.artifacts import load_proposal, proposal_content_hash, write_proposal
from grasp_failure_prediction.proposals.generate import generate_proposals
from grasp_failure_prediction.proposals.execution import run_proposal_case
from grasp_failure_prediction.proposals.geometry_inputs import prepare_geometry_inputs
from grasp_failure_prediction.proposals.registry import load_provider_registry
from grasp_failure_prediction.proposals.schema import GraspProposal
from grasp_failure_prediction.proposals.translation import (
    NativeHandPose,
    SyntheticShadowRecordDecoder,
    TranslationError,
    load_translation_registry,
    translate_native_pose,
)
from test_object_assets import write_box_mesh
from test_grasp_proposals import GEOMETRY_HASH, _request


def _asset(root: Path) -> Path:
    root.mkdir()
    write_box_mesh(root / "box.obj", [0, 0, 0], [0.03, 0.02, 0.01])
    path = root / "object.xml"
    path.write_text('''<mujoco model="box"><compiler meshdir="." angle="radian"/>
    <asset><mesh name="box" file="box.obj"/></asset><worldbody><body name="object">
    <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/><freejoint/>
    <geom type="mesh" mesh="box"/></body></worldbody></mujoco>''')
    return path


def test_translation_registry_is_separate_strict_and_fail_closed():
    registry = load_translation_registry()
    exact = registry.resolve("dex_urdf_shadow_identity_v1")
    assert exact.strategy_kind == "exact_kinematic_mapping"
    assert exact.source_asset_hash == exact.target_asset_hash
    with pytest.raises(ValueError, match="unknown registry ID"):
        registry.resolve("unknown_provider_mapping")


def test_named_shadow_decoder_and_identity_mapping_are_exact():
    fixture = Path(__file__).parent / "fixtures/provider_saved_cube.json"
    native = SyntheticShadowRecordDecoder().decode(fixture)
    translated = translate_native_pose(native, "dex_urdf_shadow_identity_v1")
    assert translated.object_to_palm == native.object_to_root
    assert translated.joint_targets_rad == native.joint_targets_rad

    bad = NativeHandPose(
        native.source_hand_model_id,
        native.source_root_link,
        native.object_to_root,
        {**native.joint_targets_rad, "invented_joint": 0.0},
    )
    with pytest.raises(TranslationError) as exc:
        translate_native_pose(bad, "dex_urdf_shadow_identity_v1")
    assert exc.value.code == "native_joint_set_mismatch"
    assert exc.value.rejection.status == "translation_failed"
    assert exc.value.rejection.task_label is None


def test_changed_mapping_evidence_fails_closed(monkeypatch):
    fixture = Path(__file__).parent / "fixtures/provider_saved_cube.json"
    native = SyntheticShadowRecordDecoder().decode(fixture)

    class ChangedDigest:
        def hexdigest(self):
            return "0" * 64

    monkeypatch.setattr(
        "grasp_failure_prediction.proposals.translation.hashlib.sha256",
        lambda content: ChangedDigest(),
    )
    with pytest.raises(TranslationError) as exc:
        translate_native_pose(native, "dex_urdf_shadow_identity_v1")
    assert exc.value.code == "mapping_evidence_hash_mismatch"


def test_provider_classification_license_metadata_and_disabled_dexgraspnet():
    registry = load_provider_registry()
    saved = registry.resolve("shadow_saved_record_v1")
    assert saved.kind == "dataset_replay"
    assert saved.provenance.code.status == "verified"
    dex = registry.resolve("dexgraspnet_saved_record_v1")
    assert dex.kind == "dataset_replay" and not dex.enabled
    assert "saved-record semantics" in dex.unsupported_reason


def test_geometry_inputs_are_deterministic_and_content_sensitive(tmp_path):
    asset = _asset(tmp_path / "asset")
    geometry = load_object_geometry(asset)
    first = prepare_geometry_inputs(geometry, tmp_path / "first", seed=7, point_count=128)
    second = prepare_geometry_inputs(geometry, tmp_path / "second", seed=7, point_count=128)
    assert first == second
    np.testing.assert_array_equal(
        np.load(tmp_path / "first/object_points.npy"),
        np.load(tmp_path / "second/object_points.npy"),
    )
    assert (tmp_path / "first/object_points.npy").read_bytes() == (tmp_path / "second/object_points.npy").read_bytes()
    changed_seed = prepare_geometry_inputs(geometry, tmp_path / "seed", seed=8, point_count=128)
    changed_config = prepare_geometry_inputs(geometry, tmp_path / "count", seed=7, point_count=64)
    assert changed_seed["manifest_hash"] != first["manifest_hash"]
    assert changed_config["manifest_hash"] != first["manifest_hash"]
    (asset.parent / "box.obj").write_text((asset.parent / "box.obj").read_text() + "# changed\n")
    changed_geometry = load_object_geometry(asset)
    changed_asset = prepare_geometry_inputs(changed_geometry, tmp_path / "asset_changed", seed=7, point_count=128)
    assert changed_asset["manifest_hash"] != first["manifest_hash"]


def test_structured_rejections_have_null_task_label_and_precede_rollout():
    raw = json.loads((Path(__file__).parent / "fixtures/provider_saved_cube.json").read_text())
    raw["object_to_palm"]["translation_m"] = [1.0, 0.0, 0.0]
    proposal = type("P", (), {"object_to_palm": type("T", (), raw["object_to_palm"])()})()
    with pytest.raises(ProposalRejected) as exc:
        _scene_preflight(proposal)
    rejection = exc.value.rejection
    assert rejection.status == "proposal_rejected"
    assert rejection.stage == "scene_preflight"
    assert rejection.code == "palm_implausibly_distant"
    assert rejection.task_label is None


def test_preflight_tolerance_is_registered_and_hashed():
    from grasp_failure_prediction.evaluation.schema import canonical_sha256
    from grasp_failure_prediction.proposals.preflight import load_preflight_registry

    registry = load_preflight_registry()
    spec = registry.resolve("shadow_proposal_preflight_v1")
    assert spec.maximum_object_to_palm_distance_m == 0.5
    assert registry.config_hash(spec.preflight_id) == canonical_sha256(spec)


def test_rejected_proposal_never_constructs_rollout_runner(tmp_path, monkeypatch):
    manifest = generate_proposals(_request(tmp_path), tmp_path / "proposals", project_root=tmp_path)
    entry = manifest.proposals[0]
    proposal_path = tmp_path / "proposals" / entry.proposal_path
    proposal = load_proposal(proposal_path)
    proposal = proposal.model_copy(update={
        "object_to_palm": proposal.object_to_palm.model_copy(update={"translation_m": (1.0, 0.0, 0.0)})
    })
    write_proposal(proposal_path, proposal)
    case = {
        "schema_version": "eval_case_v2",
        "case_id": "rejected_before_rollout",
        "environment": {"id": "adroit_shadow_tabletop_v2"},
        "embodiment": {"id": "shadow_hand_right"},
        "execution_protocol": {"id": "fixed_grasp_lift_v2"},
        "object": {"id": "object01", "mass_kg": 0.18, "friction_profile_id": "nominal_v1"},
        "grasp": {
            "id": proposal.proposal_id,
            "proposal_path": str(Path("proposals") / entry.proposal_path),
            "expected_proposal_hash": proposal_content_hash(proposal),
        },
        "initial_condition": {
            "id": "centered_upright",
            "object_position_m": [0.0, 0.0, 0.03],
            "object_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "motion_profile_id": "nominal_lift_v1",
        "seed": 19,
    }
    case_path = tmp_path / "case.yaml"
    case_path.write_text(yaml.safe_dump(case, sort_keys=False))
    monkeypatch.setattr(
        "grasp_failure_prediction.proposals.execution._runner_class",
        lambda resolved: pytest.fail("rollout runner constructed for rejected proposal"),
    )
    with pytest.raises(ProposalRejected) as exc:
        run_proposal_case(case_path, tmp_path / "run", project_root=tmp_path, record_video=False)
    assert exc.value.rejection.code == "palm_implausibly_distant"
    assert not (tmp_path / "run").exists()
