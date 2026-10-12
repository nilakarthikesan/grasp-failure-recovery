"""Treatment fidelity tests that never advance a grasp simulation."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_failure_prediction.evaluation import paired_wrist_execution as execution
from grasp_failure_prediction.evaluation.noisy_rollouts import NoisyShadowRunner, _sha256
from grasp_failure_prediction.evaluation.registry import load_protocol_registry, resolve_case
from grasp_failure_prediction.evaluation.retargeting import RetargetedHandPose, default_dex_urdf_root
from grasp_failure_prediction.evaluation.schema import EvaluationCase, canonical_sha256
from grasp_failure_prediction.evaluation.table_clearance import plan_table_clearance
from test_evaluation_schema import valid_case


def frozen_case(pose, position, quaternion, plan):
    fingers = [name for name in pose.joint_names if name not in {"WRJ2", "WRJ1"}]
    return dict(
        pair_id="fixture", eligible=True,
        intended_world_palm_position_m=position.tolist(),
        intended_world_palm_quaternion_wxyz=quaternion.tolist(),
        intended_finger_joint_names=fingers,
        intended_finger_target_rad=[float(pose.qpos[pose.joint_names.index(n)]) for n in fingers],
        poses={"A": dict(
            wrist_rad=list(plan.adjusted_wrist_rad), root_position_m=plan.root_position_m.tolist(),
            root_quaternion_wxyz=plan.root_quaternion_wxyz.tolist(),
            nominal_table_clearance_m=plan.minimum_table_clearance_m,
            nominal_proximal_object_distance_m=plan.minimum_proximal_object_distance_m,
            passes_geometry_gate=True)},
    )


@pytest.fixture
def setup():
    fixture = json.loads((Path(__file__).parent / "fixtures/shadow_hug_cube_target.json").read_text())
    pose = RetargetedHandPose(
        tuple(fixture["joint_names"]), np.asarray(fixture["joints"]),
        np.zeros((21, 3)), np.zeros((10, 3)), np.eye(4))
    position = np.array([0.0, 0.0, 0.13])
    quaternion = np.array([np.cos(np.pi / 6), 0.0, np.sin(np.pi / 6), 0.0])
    base = load_protocol_registry().resolve("fixed_grasp_lift_v2")
    # Build static geometry with the existing runner, without mj_step.
    planner = NoisyShadowRunner(base, default_dex_urdf_root())
    reset = dict(seed=1, object_mass_kg=0.18, object_position_m=np.array([1., 1., 1.]),
                 object_orientation_wxyz=np.array([1., 0., 0., 0.]))
    planner.reset(**reset)
    plan = plan_table_clearance(planner, pose, position, quaternion)
    assert plan.feasible
    case = frozen_case(pose, position, quaternion, plan)
    protocol = execution.paired_wrist_protocol(
        base, manifest_sha256="0" * 64, pair_id="fixture", posture="A",
        wrist_rad=case["poses"]["A"]["wrist_rad"])
    runner = execution.FrozenWristShadowRunner(protocol, default_dex_urdf_root(), frozen_case=case)
    runner.reset(**reset)
    return runner, pose, position, quaternion, case


def forbid_motion(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a rejected frozen treatment must not enter controller execution")
    monkeypatch.setattr(NoisyShadowRunner, "execute", forbidden)


def test_protocol_versions_exact_treatment_without_changing_task_dynamics():
    base = load_protocol_registry().resolve("fixed_grasp_lift_v2")
    before = canonical_sha256(base)
    protocol = execution.paired_wrist_protocol(
        base, manifest_sha256="0" * 64, pair_id="pair", posture="A", wrist_rad=[0.1, -0.2])
    assert protocol.parameters == base.parameters
    assert protocol.success == base.success
    assert protocol.states == base.states
    assert protocol.runtime_table_margin_m == 0.001
    assert protocol.clearance_configuration["check_closing_targets"] is False
    assert protocol.base_protocol_config_hash == before
    assert protocol.explicit_wrist_target_rad == (0.1, -0.2)
    assert protocol.execution_protocol_id == "fixed_grasp_lift_v4_frozen_wrist_pair"
    assert canonical_sha256(base) == before
    with pytest.raises(ValueError, match="fixed_grasp_lift_v2"):
        execution.paired_wrist_protocol(load_protocol_registry().resolve("fixed_grasp_lift_v1"),
                                       manifest_sha256="0" * 64, pair_id="pair", posture="A",
                                       wrist_rad=[0., 0.])


def test_exact_safe_wrist_preserves_fingers_and_world_palm_without_replanning(setup, monkeypatch):
    runner, pose, position, quaternion, case = setup
    forwarded, before = {}, pose.qpos.copy()
    sentinel = object()
    def capture(self, explicit, **kwargs):
        forwarded.update(pose=explicit, **kwargs)
        return sentinel
    monkeypatch.setattr(NoisyShadowRunner, "execute", capture)
    result = runner.execute(pose, grasp_palm_position_m=position,
                            grasp_palm_quaternion_wxyz=quaternion)
    assert result is sentinel
    target = forwarded["pose"]
    for name, angle in zip(("WRJ2", "WRJ1"), case["poses"]["A"]["wrist_rad"], strict=True):
        assert target.qpos[target.joint_names.index(name)] == angle
    for i, name in enumerate(pose.joint_names):
        if name not in {"WRJ2", "WRJ1"}:
            assert target.qpos[i] == before[i]
    np.testing.assert_array_equal(forwarded["grasp_palm_position_m"], position)
    np.testing.assert_array_equal(forwarded["grasp_palm_quaternion_wxyz"], quaternion)
    np.testing.assert_array_equal(pose.qpos, before)
    assert runner.clearance_plan.evaluated_candidates == 1
    assert runner.frozen_pose_certificate["automatic_substitution"] is False
    assert max(runner.frozen_pose_certificate["nominal_world_geometry_max_absolute_errors"].values()) < 1e-13
    assert runner.data.time == 0
    assert runner.action_history == []


def test_unsafe_exact_wrist_cannot_be_silently_replaced_by_a_safe_candidate(setup, monkeypatch):
    runner, pose, position, quaternion, case = setup
    original = [float(pose.qpos[pose.joint_names.index(name)]) for name in ("WRJ2", "WRJ1")]
    case["poses"]["A"]["wrist_rad"] = original
    runner.protocol = runner.protocol.model_copy(update={"explicit_wrist_target_rad": tuple(original)})
    forbid_motion(monkeypatch)
    with pytest.raises(execution.FrozenWristRejected, match="substitution is forbidden"):
        runner.execute(pose, grasp_palm_position_m=position, grasp_palm_quaternion_wxyz=quaternion)
    assert runner.clearance_plan.feasible  # Existing planner found a replacement.
    assert runner.clearance_plan.adjusted_wrist_rad != tuple(original)
    assert runner.data.time == 0
    assert runner.action_history == []


@pytest.mark.parametrize("field,index", [("root_position_m", 0), ("root_quaternion_wxyz", 0)])
def test_carrier_mismatch_rejects_before_motion(setup, monkeypatch, field, index):
    runner, pose, position, quaternion, case = setup
    case["poses"]["A"][field][index] += 1e-5
    forbid_motion(monkeypatch)
    with pytest.raises(execution.FrozenWristRejected, match=field):
        runner.execute(pose, grasp_palm_position_m=position, grasp_palm_quaternion_wxyz=quaternion)
    assert runner.data.time == 0


@pytest.mark.parametrize("changed", ["palm", "finger", "table_clearance", "proximal_distance", "direction"])
def test_frozen_geometry_and_direction_are_enforced_before_motion(setup, monkeypatch, changed):
    runner, pose, position, quaternion, case = setup
    extra = {}
    if changed == "palm":
        position = position.copy()
        position[0] += 1e-5
    elif changed == "finger":
        qpos = pose.qpos.copy()
        qpos[pose.joint_names.index("FFJ1")] += 1e-5
        pose = replace(pose, qpos=qpos)
    elif changed == "table_clearance":
        case["poses"]["A"]["nominal_table_clearance_m"] += 1e-5
    elif changed == "proximal_distance":
        case["poses"]["A"]["nominal_proximal_object_distance_m"] += 1e-5
    else:
        extra["approach_direction_world"] = [1., 0., -1.]
    forbid_motion(monkeypatch)
    with pytest.raises(execution.FrozenWristRejected):
        runner.execute(pose, grasp_palm_position_m=position, grasp_palm_quaternion_wxyz=quaternion, **extra)
    assert runner.data.time == 0
    assert runner.action_history == []


@pytest.mark.parametrize("amplitude", [0.01, -0.1, float("nan")])
def test_nonzero_noise_is_rejected_before_runner_model_creation(monkeypatch, amplitude):
    base = load_protocol_registry().resolve("fixed_grasp_lift_v2")
    protocol = execution.paired_wrist_protocol(
        base, manifest_sha256="0" * 64, pair_id="pair", posture="A", wrist_rad=[0., 0.])
    calls = []
    monkeypatch.setattr(NoisyShadowRunner, "__init__", lambda *args, **kwargs: calls.append(1))
    with pytest.raises(ValueError, match="zero"):
        execution.FrozenWristShadowRunner(protocol, frozen_case={"poses": {"A": {}}},
                                         noise_amplitude_rad=amplitude)
    assert calls == []


def manifest_fixture(tmp_path):
    (tmp_path / "model.npz").write_bytes(b"immutable model")
    source = tmp_path / "source.py"
    source.write_text("unchanged source")
    manifest = dict(
        schema_version=execution.MANIFEST_SCHEMA, gates=execution.GATES.copy(),
        source_sha256={"repository:source.py": _sha256(source)},
        model_contract=dict(
            weights_normalization_features_unchanged=True, failure_alert_threshold=0.5,
            prediction_time_s=2.96, history_steps=20,
            fields=["hand_qpos", "hand_qvel", "commanded_target_rad"], outcomes_not_inputs=True,
            model_file="model.npz", model_sha256=_sha256(tmp_path / "model.npz")),
        cases=[dict(pair_id="fixture", eligible=True)])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, _sha256(path), manifest


@pytest.mark.parametrize("changed", ["manifest", "source", "model", "gates", "cutoff"])
def test_frozen_manifest_and_source_tampering_is_rejected(tmp_path, changed):
    path, expected, manifest = manifest_fixture(tmp_path)
    if changed == "manifest":
        path.write_text(path.read_text() + " ")
    elif changed == "source":
        (tmp_path / "source.py").write_text("changed source")
    elif changed == "model":
        (tmp_path / "model.npz").write_bytes(b"changed model")
    else:
        if changed == "gates":
            manifest["gates"]["runtime_table_margin_m"] = 0.0001
        else:
            manifest["model_contract"]["prediction_time_s"] = 3.0
        path.write_text(json.dumps(manifest))
        expected = _sha256(path)
    with pytest.raises(ValueError, match="SHA256|gates|contract"):
        execution.load_frozen_manifest(path, expected, repository_root=tmp_path, project_root=tmp_path)


def test_frozen_manifest_accepts_exact_binding_without_loading_predictor(tmp_path):
    path, expected, manifest = manifest_fixture(tmp_path)
    assert execution.load_frozen_manifest(path, expected, repository_root=tmp_path,
                                          project_root=tmp_path) == manifest


def test_runtime_stop_remains_unlabeled_without_scoring_or_outcome_bundle(
        setup, tmp_path, monkeypatch):
    _, pose, position, quaternion, row = setup
    prepared = SimpleNamespace(
        resolved=resolve_case(EvaluationCase.model_validate(valid_case())), object_geometry=None,
        source={}, pose=pose, palm_position=position, palm_quaternion=quaternion, validation=None)
    case = tmp_path / "case.yaml"
    case.write_text(json.dumps(valid_case()))
    row.update(case_path="case.yaml", case_sha256=_sha256(case), source_binding={},
               resolved_environment=prepared.resolved.environment.model_dump(mode="json"),
               environment_config_hash=prepared.resolved.environment_config_hash,
               resolved_base_execution_protocol=prepared.resolved.execution_protocol.model_dump(mode="json"),
               base_execution_protocol_config_hash=prepared.resolved.execution_protocol_config_hash)
    monkeypatch.setattr(execution, "load_frozen_manifest", lambda *a, **k: {"cases": [row]})
    monkeypatch.setattr(execution, "_prepare_case", lambda *a, **k: prepared)
    def rejected(self, *args, **kwargs):
        self.clearance_plan = SimpleNamespace(as_record=lambda: {"feasible": True})
        self.runtime_guard_event = dict(time_s=0.002, actual_table_clearance_m=0.0009,
                                       required_table_clearance_m=0.001)
        self.data.time = 0.002
        raise execution.MotionClearanceRejected("runtime margin reached")
    def forbidden(*args, **kwargs):
        pytest.fail("stopped treatment has no task outcome or model score")
    monkeypatch.setattr(execution.FrozenWristShadowRunner, "execute", rejected)
    monkeypatch.setattr(execution, "score_trace", forbidden)
    monkeypatch.setattr(execution, "write_evaluation_artifacts", forbidden)
    result = execution.execute_frozen_wrist_case(
        case, tmp_path / "stopped", project_root=tmp_path, manifest_path="unused",
        expected_manifest_sha256="0" * 64, pair_id="fixture", posture="A")
    assert result["status"] == "runtime_clearance_rejected"
    assert result["label"] is None
    assert result["predictor_evaluated_by_executor"] is False
    assert sorted(path.name for path in (tmp_path / "stopped").iterdir()) == [
        "collection_metadata.json", "runtime_guard_state.npz"]
    with np.load(tmp_path / "stopped/runtime_guard_state.npz", allow_pickle=False) as state:
        assert float(state["time_s"]) == 0.002


def test_frozen_case_and_scene_binding_cannot_change(tmp_path):
    resolved = resolve_case(EvaluationCase.model_validate(valid_case()))
    prepared = SimpleNamespace(resolved=resolved, source={"proposal_sha256": "original"})
    case = tmp_path / "case.yaml"
    case.write_text(json.dumps(valid_case()))
    row = dict(case_sha256=_sha256(case), source_binding=prepared.source.copy(),
               resolved_environment=resolved.environment.model_dump(mode="json"),
               environment_config_hash=resolved.environment_config_hash,
               resolved_base_execution_protocol=resolved.execution_protocol.model_dump(mode="json"),
               base_execution_protocol_config_hash=resolved.execution_protocol_config_hash)
    execution._validate_prepared(prepared, row, case)
    row["source_binding"] = {"proposal_sha256": "changed"}
    with pytest.raises(ValueError, match="source binding"):
        execution._validate_prepared(prepared, row, case)
    row["source_binding"] = prepared.source.copy()
    case.write_text("changed case")
    with pytest.raises(ValueError, match="case SHA256"):
        execution._validate_prepared(prepared, row, case)
