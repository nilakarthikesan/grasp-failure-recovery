import pytest
import mujoco

from grasp_failure_prediction.evaluation.prediction_recording import (
    AttemptMetadata, PHYSICAL_FAILURES, PredictionRecorder, write_incomplete_attempt,
)
from grasp_failure_prediction.prediction.failure_analysis import (
    PHYSICAL_PHASES, failure_phase, summarize_attempts,
)


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("failure analysis tests use manufactured recordings; physics forbidden")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)


HASH = "sha256:" + "a" * 64


def recorded(tmp_path, name, *, execution="completed", failure_type=None,
             mapping="ok", hand_id="manufactured_shadow", protocol_id="manufactured_controller_v1",
             label_rule_id="manufactured_rule_v1", label_hash=HASH, object_id="manufactured_object"):
    """Write the real recording schema with invented values, never simulate."""
    payload = dict(attempt_id=name, hand_id=hand_id, execution_protocol_id=protocol_id,
                   object_id=object_id, geometry_group_id="manufactured_geometry",
                   observation_group_id="manufactured_capture", proposal_group_id="manufactured_proposal",
                   mapping=dict(status=mapping), execution_status=execution, outcome=None,
                   provenance_hashes={key: HASH for key in ("code", "configuration", "inputs", "model", "assets",
                                                          "native_pose", "mapped_pose", "label_rule")})
    payload["provenance_hashes"]["label_rule"] = label_hash
    if mapping != "ok":
        payload["mapping"].update(code="manufactured_refusal", message="manufactured static refusal")
    if execution == "completed":
        payload["outcome"] = dict(success=failure_type is None, failure_type=failure_type,
                                  label_rule_id=label_rule_id, first_failure_time_s=None)
    metadata = AttemptMetadata(**payload)
    path = tmp_path / name
    if execution == "completed":
        recorder = PredictionRecorder(["joint_0"], [[-1., 1.]])
        recorder.append(time_s=.04, action_interval_start_s=0., phase="hold", hand_qpos=[.1],
                        hand_qvel=[0.], requested_target_rad=[.2], applied_target_rad=[.2], diagnostics={})
        recorder.write(path, metadata)
    else:
        write_incomplete_attempt(path, metadata)
    return path


@pytest.mark.parametrize("cause,phase", [
    ("approach_collision", "pre_contact"), ("no_contact", "pre_contact"),
    ("failed_acquisition", "acquisition"), ("drop", "hold"),
    ("slip_during_hold", "hold"), ("timeout", "unknown")])
def test_physical_phase_taxonomy_is_explicit(cause, phase):
    assert failure_phase(cause, success=False) == phase


@pytest.mark.parametrize("cause", ["invalid_retarget", "unreachable_pregrasp", "runtime_error", None])
def test_mapping_and_runtime_faults_never_get_physical_phases(cause):
    with pytest.raises(ValueError):
        failure_phase(cause, success=False)


def test_success_and_failure_cause_cannot_mix():
    assert failure_phase(None, success=True) is None
    with pytest.raises(ValueError):
        failure_phase("drop", success=True)


def test_source_physical_taxonomy_is_completely_covered():
    assert set(PHYSICAL_PHASES) == PHYSICAL_FAILURES


def test_summary_preserves_null_denominators_and_only_groups_completed_causes(tmp_path):
    paths = [recorded(tmp_path, "success"),
             recorded(tmp_path, "collision", failure_type="approach_collision"),
             recorded(tmp_path, "acquisition", failure_type="failed_acquisition"),
             recorded(tmp_path, "drop", failure_type="drop"),
             recorded(tmp_path, "timeout", failure_type="timeout"),
             recorded(tmp_path, "mapping_refusal", execution="not_executed", mapping="refused_guard"),
             recorded(tmp_path, "planner_refusal", execution="not_executed", mapping="unreachable_pregrasp"),
             recorded(tmp_path, "unexecuted", execution="not_executed"),
             recorded(tmp_path, "runtime", execution="runtime_error")]
    result = summarize_attempts(paths)
    assert result["counts"] == dict(attempted=9, mapped=7, executed=6, physically_labeled=5,
                                    success=1, physical_failure=4, null_outcome=4,
                                    refused_or_unexecuted=3, runtime_error=1)
    assert result["physical_failure_phases"] == dict(pre_contact=1, acquisition=1, hold=1, unknown=1)
    assert result["physical_failure_causes"] == dict(approach_collision=1, failed_acquisition=1, drop=1, timeout=1)
    assert result["cohort"] == dict(hand_id="manufactured_shadow", execution_protocol_id="manufactured_controller_v1",
                                    label_rule_id="manufactured_rule_v1", label_rule_sha256=HASH)
    null_rows = [row for row in result["attempts"] if row["physical_outcome"] is None]
    assert len(null_rows) == 4 and all(row["failure_phase"] is None for row in null_rows)


@pytest.mark.parametrize("identity", ["hand_id", "protocol_id"])
@pytest.mark.parametrize("execution,mapping", [("completed", "ok"), ("runtime_error", "ok"),
                                                ("not_executed", "unreachable_pregrasp")])
def test_mixed_hand_or_controller_refused_even_for_null_attempt(tmp_path, identity, execution, mapping):
    paths = [recorded(tmp_path, "first"),
             recorded(tmp_path, "second", execution=execution, mapping=mapping, **{identity: "different_v2"})]
    with pytest.raises(ValueError, match="hand/controller protocols"):
        summarize_attempts(paths)


@pytest.mark.parametrize("change", [dict(label_rule_id="different_rule_v2"),
                                  dict(label_hash="sha256:" + "b" * 64)])
def test_completed_label_rule_id_and_hash_cannot_be_pooled(tmp_path, change):
    paths = [recorded(tmp_path, "first"), recorded(tmp_path, "second", failure_type="no_contact", **change)]
    with pytest.raises(ValueError, match="label rules or hashes"):
        summarize_attempts(paths)


def test_null_attempts_do_not_invent_or_select_a_physical_label_rule(tmp_path):
    paths = [recorded(tmp_path, "refused", execution="not_executed", mapping="refused_guard"),
             recorded(tmp_path, "runtime", execution="runtime_error", label_hash="sha256:" + "b" * 64)]
    result = summarize_attempts(paths)
    assert result["counts"]["attempted"] == result["counts"]["null_outcome"] == 2
    assert result["counts"]["physically_labeled"] == 0
    assert result["physical_failure_causes"] == result["physical_failure_phases"] == {}
    assert result["cohort"]["label_rule_id"] is result["cohort"]["label_rule_sha256"] is None


def test_duplicate_attempt_identity_refused(tmp_path):
    path = recorded(tmp_path, "same")
    with pytest.raises(ValueError, match="duplicate attempt"):
        summarize_attempts([path, path])
