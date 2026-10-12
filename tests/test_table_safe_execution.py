"""Contract tests for the separately versioned clearance execution path."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation import table_safe_execution as execution
from grasp_failure_prediction.evaluation.noisy_rollouts import NoisyShadowRunner
from grasp_failure_prediction.evaluation.registry import load_protocol_registry, resolve_case
from grasp_failure_prediction.evaluation.retargeting import (
    RetargetedHandPose, default_dex_urdf_root,
)
from grasp_failure_prediction.evaluation.schema import EvaluationCase, canonical_sha256
from grasp_failure_prediction.evaluation.runner import ExecutionState
from grasp_failure_prediction.evaluation.table_clearance import TableClearanceConfiguration
from test_evaluation_schema import valid_case


@pytest.fixture
def setup():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/shadow_hug_cube_target.json").read_text()
    )
    base = load_protocol_registry().resolve("fixed_grasp_lift_v2")
    runner = execution.TableSafeNoisyShadowRunner(
        execution.table_safe_protocol(base), default_dex_urdf_root(),
        noise_amplitude_rad=0.15, noise_seed=123,
    )
    runner.reset(
        seed=1, object_mass_kg=0.18,
        object_position_m=np.array([1.0, 1.0, 1.0]),
        object_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    pose = RetargetedHandPose(
        tuple(fixture["joint_names"]), np.asarray(fixture["joints"]),
        np.zeros((21, 3)), np.zeros((10, 3)), np.eye(4),
    )
    position = np.array([0.0, 0.0, 0.13])
    quaternion = np.array([np.cos(np.pi / 6), 0.0, np.sin(np.pi / 6), 0.0])
    return runner, pose, position, quaternion


def test_protocol_preserves_original_task_but_hashes_planning_configuration():
    base = load_protocol_registry().resolve("fixed_grasp_lift_v2")
    original_hash = canonical_sha256(base)
    safe = execution.table_safe_protocol(base)
    larger_margin = execution.table_safe_protocol(
        base, TableClearanceConfiguration(clearance_m=0.02)
    )
    assert safe.parameters == base.parameters
    assert safe.success == base.success
    assert safe.states == base.states
    assert safe.base_protocol_config_hash == original_hash
    assert safe.base_protocol_id == "fixed_grasp_lift_v2"
    assert safe.execution_protocol_id != base.execution_protocol_id
    assert canonical_sha256(safe) != original_hash
    assert canonical_sha256(safe) != canonical_sha256(larger_margin)
    assert canonical_sha256(base) == original_hash
    assert execution.table_safe_protocol(base) == safe
    with pytest.raises(ValueError, match="fixed_grasp_lift_v2"):
        execution.table_safe_protocol(load_protocol_registry().resolve("fixed_grasp_lift_v1"))


def test_runner_rejects_unversioned_protocol_before_model_creation(monkeypatch):
    calls = []
    monkeypatch.setattr(NoisyShadowRunner, "__init__", lambda *args, **kwargs: calls.append(1))
    with pytest.raises(TypeError, match="version"):
        execution.TableSafeNoisyShadowRunner(
            load_protocol_registry().resolve("fixed_grasp_lift_v2"), default_dex_urdf_root()
        )
    assert calls == []


@pytest.mark.parametrize("direction", [
    [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, -1.0],
    [float("nan"), 0.0, -1.0], [0.0, 0.0, float("inf")], [0.0, -1.0],
])
def test_invalid_approach_is_rejected_before_planning_or_motion(setup, monkeypatch, direction):
    runner, pose, position, quaternion = setup

    def forbidden(*args, **kwargs):
        pytest.fail("invalid approach must not plan or start simulation")

    monkeypatch.setattr(execution, "plan_table_clearance", forbidden)
    monkeypatch.setattr(NoisyShadowRunner, "execute", forbidden)
    with pytest.raises(ValueError, match="direction|vertical"):
        runner.execute(
            pose, grasp_palm_position_m=position, grasp_palm_quaternion_wxyz=quaternion,
            approach_direction_world=np.asarray(direction),
        )
    assert runner.data.time == 0
    assert runner.action_history == []
    assert runner.clearance_plan is None


def test_safe_plan_passes_adjusted_wrist_and_original_world_palm_to_controller(setup, monkeypatch):
    runner, pose, position, quaternion = setup
    original_pose = pose.qpos.copy()
    original_position = position.copy()
    original_rotation = quaternion.copy()
    forwarded = {}
    sentinel = object()
    callback = lambda step: None

    def capture(self, adjusted_pose, **kwargs):
        forwarded.update(pose=adjusted_pose, **kwargs)
        return sentinel

    monkeypatch.setattr(NoisyShadowRunner, "execute", capture)
    result = runner.execute(
        pose, grasp_palm_position_m=position, grasp_palm_quaternion_wxyz=quaternion,
        step_callback=callback,
    )
    assert result is sentinel
    assert runner.clearance_plan.feasible
    assert runner.clearance_plan.original_table_clearance_m < 0
    assert forwarded["pose"] is runner.clearance_plan.adjusted_pose
    np.testing.assert_array_equal(forwarded["grasp_palm_position_m"], original_position)
    np.testing.assert_array_equal(forwarded["grasp_palm_quaternion_wxyz"], original_rotation)
    assert forwarded["step_callback"] is callback
    for index, name in enumerate(pose.joint_names):
        if name not in {"WRJ2", "WRJ1"}:
            assert forwarded["pose"].qpos[index] == original_pose[index]
    np.testing.assert_array_equal(pose.qpos, original_pose)
    np.testing.assert_array_equal(position, original_position)
    np.testing.assert_array_equal(quaternion, original_rotation)


def test_infeasible_plan_rejects_before_parent_execution_or_steps(setup, monkeypatch):
    runner, pose, position, quaternion = setup
    position[2] = -0.1

    def forbidden(*args, **kwargs):
        pytest.fail("clearance rejection must precede controller execution")

    monkeypatch.setattr(NoisyShadowRunner, "execute", forbidden)
    with pytest.raises(execution.ClearanceRejected, match="no wrist posture"):
        runner.execute(pose, grasp_palm_position_m=position, grasp_palm_quaternion_wxyz=quaternion)
    assert runner.clearance_plan.feasible is False
    assert runner.data.time == 0
    assert runner.action_history == []


def test_clearance_rejection_writes_unlabeled_metadata_without_outcome_bundle(
    setup, monkeypatch, tmp_path,
):
    _, pose, position, quaternion = setup
    position[2] = -0.1
    prepared = SimpleNamespace(
        resolved=resolve_case(EvaluationCase.model_validate(valid_case())),
        object_geometry=None, source={}, pose=pose,
        palm_position=position, palm_quaternion=quaternion, validation=None,
    )
    monkeypatch.setattr(execution, "_prepare_case", lambda *args: prepared)

    def forbidden(*args, **kwargs):
        pytest.fail("a rejected proposal must not be scored or published as an outcome")

    monkeypatch.setattr(execution, "score_trace", forbidden)
    monkeypatch.setattr(execution, "write_evaluation_artifacts", forbidden)
    (tmp_path / "saved_case.yaml").write_text(json.dumps(valid_case()))
    output = tmp_path / "rejected"
    record = execution.execute_table_safe_case(
        "saved_case.yaml", output, project_root=tmp_path,
    )
    assert record["status"] == "clearance_rejected"
    assert record["label"] is None
    assert record["clearance_plan"]["feasible"] is False
    assert sorted(path.name for path in output.iterdir()) == ["collection_metadata.json"]
    assert json.loads((output / "collection_metadata.json").read_text())["label"] is None
    assert record["execution_protocol"]["execution_protocol_id"] == "fixed_grasp_lift_v3_table_clearance"


def test_actual_below_margin_geometry_raises_runtime_rejection(setup):
    runner, _, _, _ = setup
    root = runner._root_qpos_address
    runner.data.qpos[root + 2] = -0.5
    with pytest.raises(execution.MotionClearanceRejected, match="runtime table margin"):
        runner._guard_table_clearance(ExecutionState.APPROACH)
    event = runner.runtime_guard_event
    assert event["actual_table_clearance_m"] < runner.protocol.runtime_table_margin_m
    assert event["required_table_clearance_m"] == runner.protocol.runtime_table_margin_m
    assert event["phase"] == "approach"
    assert event["body"] in {runner.model.body(body).name for body in runner._hand_body_ids}
    assert "label" not in event
    assert runner.data.time == 0
    assert runner.action_history == []


def test_runtime_guard_reads_geometry_without_mutating_live_state(setup):
    runner, _, _, _ = setup
    root = runner._root_qpos_address
    runner.data.qpos[root + 2] = 1.0
    runner.data.qvel[:] = np.linspace(-0.3, 0.3, runner.model.nv)
    runner.data.ctrl[:] = np.linspace(-0.2, 0.2, runner.model.nu)
    runner.data.mocap_pos[:] = [0.1, 0.2, 1.0]
    runner.data.mocap_quat[:] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(runner.model, runner.data)
    fields = (
        "qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "xpos",
        "geom_xpos", "xmat", "geom_xmat", "qacc", "qacc_warmstart",
    )
    before = {name: getattr(runner.data, name).copy() for name in fields}
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_before = np.empty(mujoco.mj_stateSize(runner.model, state_spec))
    state_after = np.empty_like(state_before)
    mujoco.mj_getState(runner.model, runner.data, state_before, state_spec)
    runner._guard_table_clearance(ExecutionState.APPROACH)
    mujoco.mj_getState(runner.model, runner.data, state_after, state_spec)
    np.testing.assert_array_equal(state_after, state_before)
    for name in fields:
        np.testing.assert_array_equal(getattr(runner.data, name), before[name])
    assert runner.runtime_guard_event is None


def test_runtime_rejection_saves_diagnostics_without_scoring_or_binary_label(
    setup, monkeypatch, tmp_path,
):
    _, pose, position, quaternion = setup
    prepared = SimpleNamespace(
        resolved=resolve_case(EvaluationCase.model_validate(valid_case())),
        object_geometry=None, source={}, pose=pose,
        palm_position=position, palm_quaternion=quaternion, validation=None,
    )
    monkeypatch.setattr(execution, "_prepare_case", lambda *args: prepared)
    event = dict(
        time_s=0.002, phase="approach", geom_id=2, body="forearm",
        actual_table_clearance_m=0.0005, required_table_clearance_m=0.001,
    )

    def rejected(self, *args, **kwargs):
        self.clearance_plan = SimpleNamespace(as_record=lambda: {"feasible": True})
        self.runtime_guard_event = event.copy()
        self.data.time = event["time_s"]
        raise execution.MotionClearanceRejected("actual clearance reached runtime margin")

    def forbidden(*args, **kwargs):
        pytest.fail("runtime rejection must not be scored or written as a task outcome")

    monkeypatch.setattr(execution.TableSafeNoisyShadowRunner, "execute", rejected)
    monkeypatch.setattr(execution, "score_trace", forbidden)
    monkeypatch.setattr(execution, "write_evaluation_artifacts", forbidden)
    (tmp_path / "saved_case.yaml").write_text(json.dumps(valid_case()))
    output = tmp_path / "runtime_rejected"
    record = execution.execute_table_safe_case("saved_case.yaml", output, project_root=tmp_path)
    assert record["status"] == "runtime_clearance_rejected"
    assert record["label"] is None
    assert record["runtime_guard_event"] == event
    assert sorted(path.name for path in output.iterdir()) == [
        "collection_metadata.json", "runtime_guard_state.npz",
    ]
    saved = json.loads((output / "collection_metadata.json").read_text())
    assert saved["label"] is None
    assert saved["status"] == "runtime_clearance_rejected"
    with np.load(output / "runtime_guard_state.npz", allow_pickle=False) as state:
        assert float(state["time_s"]) == event["time_s"]
        assert np.all(np.isfinite(state["qpos"]))


@pytest.mark.parametrize("status,label", [
    ("completed", 0), ("completed", 1),
    ("clearance_rejected", None), ("runtime_clearance_rejected", None),
])
def test_cli_signals_rejection_as_nonzero_exit(monkeypatch, capsys, status, label):
    monkeypatch.setattr(sys, "argv", ["table-safe", "case.yaml", "--output", "out"])
    monkeypatch.setattr(
        execution, "execute_table_safe_case", lambda *args, **kwargs: {"status": status, "label": label}
    )
    if status == "completed":
        assert execution.main() is None
    else:
        with pytest.raises(SystemExit) as exception:
            execution.main()
        assert exception.value.code == 2
    assert f"{status}: label={label}" in capsys.readouterr().out


def test_nonempty_output_is_preserved_before_preparation(monkeypatch, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    artifact = output / "result.json"
    artifact.write_text("existing outcome")

    def forbidden(*args, **kwargs):
        pytest.fail("existing output must be rejected before preparation")

    monkeypatch.setattr(execution, "_prepare_case", forbidden)
    with pytest.raises(FileExistsError):
        execution.execute_table_safe_case("case.yaml", output, project_root=tmp_path)
    assert artifact.read_text() == "existing outcome"
