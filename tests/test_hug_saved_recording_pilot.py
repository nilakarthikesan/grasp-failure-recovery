"""No physics: synthetic driver transactions plus frozen-model smoke checks."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation.prediction_recording import PredictionRecorder


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("saved_recording_pilot", ROOT / "scripts/run_hug_saved_recording_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)
MODEL_DIR = ROOT / "research_artifacts/task_outcome_baseline_v1"
JOINTS = json.loads((MODEL_DIR / "report.json").read_text())["input_contract"]["joint_names"]


@pytest.fixture(autouse=True)
def no_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("all physics integration APIs forbidden in driver static tests")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)


def fake_result(*, failure=None):
    recorder = PredictionRecorder(JOINTS, np.tile([-2., 2.], (24, 1)))
    for sample in range(1, 76):
        recorder.append(time_s=round(sample * .04, 10), action_interval_start_s=round((sample - 1) * .04, 10),
            phase="close_fingers" if sample <= 74 else "lift", hand_qpos=np.zeros(24), hand_qvel=np.zeros(24),
            requested_target_rad=np.zeros(24), applied_target_rad=np.zeros(24),
            diagnostics={"full_qpos": np.zeros(38), "full_qvel": np.zeros(36)})
    return {"recorder": recorder, "observables": recorder.arrays(),
            "diagnostics": {"full_qpos": np.zeros((75, 38)), "full_qvel": np.zeros((75, 36))},
            "outcome": {"success": failure is None, "failure_type": failure, "maximum_lift_m": .15},
            "first_failure_time_s": .5 if failure == "approach_collision" else None,
            "validation": {"synthetic_test": True}, "model_sha256": "sha256:" + "a" * 64}


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    # Manufactured transaction tests start after scientific admission. The real
    # gate remains unmocked in test_hug_fidelity_admission.py.
    from grasp_failure_prediction.evaluation import mapping_fidelity
    monkeypatch.setattr(mapping_fidelity, "require_fidelity_pass", lambda *args, **kwargs: None)
    evidence, raw = tmp_path / "portable", tmp_path / "ignored_raw"
    evidence.mkdir()
    inventory = []
    cases = {}
    for index in range(32):
        inventory.append({"object_id": f"object_{index // 8}", "observation_group_id": f"observation_{index // 4}"})
        if index % 2 == 0:
            cases[str(index)] = {"native_sha256": "sha256:" + "b" * 64,
                "geometry_group_id": f"geometry_{index // 8}", "proposal_group_id": f"proposal_{index}",
                "command": {"source_index": index}, "input_bindings": {}, "asset_sha256": "sha256:" + "c" * 64}
    inventory_path = tmp_path / "inventory.json"
    pilot.write(inventory_path, {"modern_cases": inventory})
    scorer = ROOT / "src/grasp_failure_prediction/evaluation/scoring.py"
    models = {str(MODEL_DIR / name): pilot.sha(MODEL_DIR / name) for name in (
        "report.json", "proprioception_and_commands.npz", "commands_only.npz")}
    protocol = {"schema_version": pilot.SCHEMA, "project": str(ROOT), "inputs_project": str(ROOT),
        "raw_output": str(raw), "inventory": str(inventory_path), "schedule": pilot.schedule(), "cases": cases,
        "quota": dict(pilot.QUOTA), "hand_id": "shadow_hand_right", "execution_protocol_id": "fixed_grasp_lift_v2",
        "command_noise_rad": 0., "sensor_noise": 0., "approach_direction_world": [0., 0., -1.],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__},
        "source_bindings": {str(scorer): pilot.sha(scorer)}, "input_bindings": {str(inventory_path): pilot.sha(inventory_path)},
        "tolerances": {"determinism_trace_abs": 1e-10},
        "prefix": {"protocol_id": "synthetic_v1", "prediction_time_s": 2.96, "history_steps": 20, "control_timestep_s": .04},
        "saved_models_dir": str(MODEL_DIR), "saved_model_bindings": models,
        "model_smoke_scope": "synthetic compatibility test; no model fitting"}
    pilot.write(evidence / "protocol.json", protocol)
    reports = []
    for name in ("static", "camera"):
        report = tmp_path / f"{name}.json"
        pilot.write(report, {"synthetic_test": True})
        reports.append({"path": str(report), "sha256": pilot.sha(report)})
    receipt = {"schema_version": "hug_saved_recording_pilot_authorization_v1", "authorized": True,
        "protocol_sha256": pilot.sha(evidence / "protocol.json"), "accepted_saved_replay_profile": True,
        "camera_checks_pass": True, "static_profile_report": reports[0], "camera_report": reports[1]}
    pilot.write(evidence / "authorization.json", receipt)
    return evidence, raw, protocol, receipt


def executor(protocol, source, frozen, attempt_dir, budget):
    budget["total"] += 1
    return fake_result()


def test_declared_schedule_is_exactly_sixteen_even_commands_plus_four_repeats():
    plan = pilot.schedule()
    assert [row["source_inventory_index"] for row in plan] == list(range(0, 32, 2)) + [0, 8, 16, 24]
    assert len(plan) == 20
    assert [row["repeat_of"] for row in plan[-4:]] == ["attempt_0000", "attempt_0004", "attempt_0008", "attempt_0012"]


def test_read_command_uses_nominal_saved_endpoint_and_rejects_noise(tmp_path):
    path = tmp_path / "actions.npz"
    values = {"joint_names": np.asarray(JOINTS), "nominal_target_rad": np.ones((2, 24)),
              "root_target_position_m": np.ones((2, 3)),
              "root_target_quaternion_wxyz": np.tile([1., 0., 0., 0.], (2, 1)),
              "applied_noise_rad": np.zeros((2, 24)), "full_qpos": np.full((2, 38), 999.)}
    np.savez_compressed(path, **values)
    source = {"stored_shadow_command": {"path": str(path), "sha256": pilot.sha(path), "row_index": 1}}
    result = pilot.read_command(source)
    assert result["qpos_rad"] == [1.] * 24
    values["applied_noise_rad"][1, 0] = .01
    np.savez_compressed(path, **values)
    source["stored_shadow_command"]["sha256"] = pilot.sha(path)
    with pytest.raises(ValueError, match="zero applied noise"):
        pilot.read_command(source)


def test_source_inertia_tensor_and_mass_scaling_are_independent_of_compiled_principal_axes():
    xml = b'<mujoco><worldbody><body><inertial mass="2" pos="0 0 0" fullinertia="1 2 3 .1 .2 .3"/></body></worldbody></mujoco>'
    center, tensor = pilot.expected_object_inertia(SimpleNamespace(source_xml=xml), 4.)
    np.testing.assert_array_equal(center, [0., 0., 0.])
    np.testing.assert_allclose(tensor, [[2., .2, .4], [.2, 4., .6], [.4, .6, 6.]])
    _, cube = pilot.expected_object_inertia(None, .18)
    np.testing.assert_allclose(cube, np.eye(3) * .000075)


def test_tracking_is_report_only_and_measures_achieved_against_applied():
    values = fake_result()["observables"]
    values["hand_qpos"][:] = .1
    assert pilot.phase_tracking(values)["close_fingers"]["mean_abs_rad"] == pytest.approx(.1)


def fake_physics_data():
    return SimpleNamespace(time=1., qpos=np.zeros(38), qvel=np.zeros(36), qacc=np.zeros(36), ctrl=np.zeros(24),
                           warning=[SimpleNamespace(number=0) for _ in range(8)])


@pytest.mark.parametrize("mode", ["warning_reset", "nonfinite_acceleration", "time_reset", "time_stall"])
def test_guard_detects_warnings_finite_autoreset_and_numerical_failures_immediately(mode):
    data = fake_physics_data()
    model = SimpleNamespace(opt=SimpleNamespace(timestep=.002))
    def step(model, state):
        state.time += .002
        if mode == "warning_reset":
            state.warning[int(mujoco.mjtWarning.mjWARN_BADQPOS)].number += 1
            state.qpos[:] = 0.  # an automatic reset can look finite afterward
        elif mode == "nonfinite_acceleration":
            state.qacc[0] = np.nan
        elif mode == "time_reset":
            state.time = .002
        elif mode == "time_stall":
            state.time -= .002
    with pytest.raises(RuntimeError, match="warning|nonfinite|time increment"):
        pilot.guarded_physics_step(model, data, step, {"total": 0, "this_attempt": 0}, pilot.QUOTA,
                                   time_tolerance_s=1e-12)


def test_guard_preserves_exact_physics_budget_and_rejects_preexisting_nonfinite_state():
    data = fake_physics_data()
    model = SimpleNamespace(opt=SimpleNamespace(timestep=.002))
    budget = {"total": 0, "this_attempt": 0}
    pilot.guarded_physics_step(model, data, lambda model, state: setattr(state, "time", state.time + .002),
                               budget, pilot.QUOTA, time_tolerance_s=1e-12)
    assert budget == {"total": 1, "this_attempt": 1}
    data.qvel[0] = np.inf
    with pytest.raises(RuntimeError, match="before physics step"):
        pilot.guarded_physics_step(model, data, lambda *a: pytest.fail("must not advance"),
                                   budget, pilot.QUOTA, time_tolerance_s=1e-12)


def test_determinism_checks_all_scored_outcome_fields_and_full_state():
    first, second = fake_result(), fake_result()
    assert pilot.compare_replay(first, second, 1e-10)["passed"]
    second["diagnostics"]["full_qvel"][0, 0] = 2e-10
    with pytest.raises(ValueError, match="exceeded tolerance"):
        pilot.compare_replay(first, second, 1e-10)
    second = fake_result()
    second["outcome"]["maximum_lift_m"] += 1e-12
    with pytest.raises(ValueError, match="exact scored outcome"):
        pilot.compare_replay(first, second, 1e-10)


@pytest.mark.parametrize("change", [
    {"authorized": False}, {"camera_checks_pass": False}, {"accepted_saved_replay_profile": False},
    {"protocol_sha256": "sha256:" + "0" * 64},
])
def test_unaccepted_gates_cannot_start_pilot(frozen, change):
    evidence, raw, protocol, receipt = frozen
    receipt.update(change)
    pilot.write(evidence / "authorization.json", receipt)
    with pytest.raises(ValueError, match="authorization"):
        pilot.run(evidence, evidence / "authorization.json", executor=executor)
    assert not raw.exists() and not (evidence / "summary.json").exists()


def test_cli_bootstraps_existing_dependencies_before_guarded_input_failure(frozen):
    evidence, raw, protocol, receipt = frozen
    original = ROOT / "research_artifacts/hug_native_mano_comparison_2026-10-15_v1/measurement/protocol.json"
    protocol["dependency_dirs"] = json.loads(original.read_text())["dependency_dirs"]
    pilot.write(evidence / "protocol.json", protocol)
    receipt.update(protocol_sha256=pilot.sha(evidence / "protocol.json"))
    pilot.write(evidence / "authorization.json", receipt)
    Path(protocol["inventory"]).write_text("{}")  # guarantees stop before any attempt
    code = """import importlib.util, mujoco, runpy, sys
from pathlib import Path
# The subprocess tests dependency bootstrap after admission, using synthetic
# records. Load only the lightweight gate file; importing evaluation/__init__
# before the pilot bootstrap would itself require the missing Pydantic runtime.
name = 'grasp_failure_prediction.evaluation.mapping_fidelity'
spec = importlib.util.spec_from_file_location(name, Path('src/grasp_failure_prediction/evaluation/mapping_fidelity.py'))
mapping_fidelity = importlib.util.module_from_spec(spec)
sys.modules[name] = mapping_fidelity
spec.loader.exec_module(mapping_fidelity)
mapping_fidelity.require_fidelity_pass = lambda *args, **kwargs: None
def forbidden(*args, **kwargs):
    raise RuntimeError('physics forbidden in CLI bootstrap regression')
for name in ('mj_step','mj_step1','mj_step2'):
    setattr(mujoco, name, forbidden)
runpy.run_path('scripts/run_hug_saved_recording_pilot.py', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code, "run", "--output", str(evidence),
                             "--authorization", str(evidence / "authorization.json")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert '"status": "HARD_STOPPED"' in result.stdout
    assert "ModuleNotFoundError" not in result.stderr
    assert not raw.exists()


def test_changed_input_hard_stops_with_entire_quota_not_run(frozen):
    evidence, raw, protocol, receipt = frozen
    Path(protocol["inventory"]).write_text("{}")
    result = pilot.run(evidence, evidence / "authorization.json", executor=executor)
    assert result["status"] == "HARD_STOPPED"
    assert all(row["status"] == "NOT_RUN" for row in result["schedule"])
    assert not raw.exists()


def test_cannot_enlarge_predeclared_physics_budget(frozen):
    evidence, raw, protocol, receipt = frozen
    protocol["quota"]["maximum_total_physics_steps"] += 1
    with pytest.raises(ValueError, match="bounded schedule"):
        pilot.verify_bindings(protocol)


def test_mocked_twenty_attempt_roundtrip_keeps_raw_private_and_scores_saved_weights(frozen, monkeypatch):
    evidence, raw, protocol, receipt = frozen
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **k: pytest.fail("physics forbidden in static test"))
    result = pilot.run(evidence, evidence / "authorization.json", executor=executor)
    assert result["status"] == "COMPLETED_BOUNDED_QUOTA"
    assert result["counts"]["completed"] == 20
    assert result["counts"]["not_run"] == 0
    assert all(row["determinism"]["passed"] for row in result["schedule"][-4:])
    assert len(list(raw.rglob("observables.npz"))) == 20
    assert len(list(evidence.rglob("*.npz"))) == 1  # portable predictor prefixes only
    smoke = json.loads((evidence / "model_input_smoke.json").read_text())
    assert smoke["model_training_performed"] is False
    assert len(smoke["models"]["proprioception_and_commands"]["p_success"]) == 20
    assert smoke["models"]["proprioception_and_commands"]["smoke_metrics"]["failure_roc_auc"] is None


def test_runtime_error_stops_remaining_quota_and_has_null_outcome(frozen):
    evidence, raw, protocol, receipt = frozen
    def fail_third(*args):
        if args[3].name == "attempt_0002":
            args[4]["total"] += 1
            raise RuntimeError("synthetic integration error")
        return executor(*args)
    result = pilot.run(evidence, evidence / "authorization.json", executor=fail_third)
    assert result["status"] == "HARD_STOPPED"
    assert result["counts"]["completed"] == 2 and result["counts"]["not_run"] == 17
    error = json.loads((raw / "attempts/attempt_0002/incomplete/attempt.json").read_text())
    assert error["attempt"]["outcome"] is None and error["attempt"]["execution_status"] == "runtime_error"


def test_mapping_refusal_is_unexecuted_and_stops_pilot(frozen):
    evidence, raw, protocol, receipt = frozen
    def refuse(*args):
        raise pilot.AttemptStopped("refused_guard", "test", "synthetic static guard")
    result = pilot.run(evidence, evidence / "authorization.json", executor=refuse)
    assert result["counts"]["not_run"] == 19
    metadata = json.loads((raw / "attempts/attempt_0000/incomplete/attempt.json").read_text())["attempt"]
    assert metadata["outcome"] is None and metadata["execution_status"] == "not_executed"


def test_pre_cutoff_approach_failures_are_reported_without_repairing_quota(frozen):
    evidence, raw, protocol, receipt = frozen
    def collision(*args):
        args[4]["total"] += 1
        return fake_result(failure="approach_collision")
    result = pilot.run(evidence, evidence / "authorization.json", executor=collision)
    assert result["counts"]["approach_collision"] == 20
    assert result["counts"]["completed"] == 20
    assert result["export"]["status"] == "NO_ELIGIBLE_PREFIXES"


def test_recording_write_failure_preserves_null_error_in_separate_record(frozen):
    evidence, raw, protocol, receipt = frozen
    def broken_writer(*args):
        result = executor(*args)
        def fail(*args, **kwargs):
            raise OSError("synthetic write failure")
        result["recorder"].write = fail
        return result
    result = pilot.run(evidence, evidence / "authorization.json", executor=broken_writer)
    assert result["status"] == "HARD_STOPPED"
    assert result["counts"]["completed"] == 0 and result["counts"]["not_run"] == 19
    error = json.loads((raw / "attempts/attempt_0000/incomplete/attempt.json").read_text())
    assert error["attempt"]["outcome"] is None
    assert (raw / "attempts/attempt_0000/partial_diagnostics.npz").is_file()
    assert "synthetic write failure" in (raw / "attempts/attempt_0000/failure_traceback.txt").read_text()


def test_non_valueerror_postprocessing_failure_is_hard_stop(frozen, monkeypatch):
    evidence, raw, protocol, receipt = frozen
    def fail(*args, **kwargs):
        raise OSError("synthetic model artifact read failure")
    monkeypatch.setattr(pilot, "model_smoke", fail)
    result = pilot.run(evidence, evidence / "authorization.json", executor=executor)
    assert result["status"] == "HARD_STOPPED"
    assert result["counts"]["completed"] == 20
    assert result["export"]["error_type"] == "OSError"
    assert (raw / "postprocessing_traceback.txt").exists()


def test_saved_weight_smoke_rejects_joint_permutation(frozen):
    _, _, protocol, _ = frozen
    contract = deepcopy(json.loads((MODEL_DIR / "report.json").read_text())["input_contract"])
    contract["fields"][-1] = "applied_target_rad"
    contract["joint_names"] = contract["joint_names"][::-1]
    with pytest.raises(ValueError, match="immutable saved model contract"):
        pilot.model_smoke(SimpleNamespace(X=np.zeros((1, 1440)), y=np.ones(1, dtype=np.uint8),
                                        manifest={"input_contract": contract}), protocol)
