from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from grasp_failure_prediction.evaluation.prediction_recording import (
    AttemptMetadata, MappingDecision, MappingStatus, PhysicalOutcome,
    PredictionRecorder, file_sha256, load_attempt, validate_observables,
    write_incomplete_attempt, write_unexecuted_attempt,
)
from grasp_failure_prediction.prediction.recorded_dataset import (
    PrefixSpec, export_recorded_dataset, load_recorded_dataset,
)


HASH = "sha256:" + "a" * 64
JOINTS = [f"joint_{index}" for index in range(24)]
LIMITS = np.tile([-1., 1.], (24, 1))


def metadata(attempt_id="attempt_1", *, mapping="ok", status="completed", success=True,
             failure_type=None, first_failure_time_s=None, **changes):
    payload = {
        "attempt_id": attempt_id, "hand_id": "shadow_study_v1",
        "execution_protocol_id": "fixed_grasp_lift_v2", "object_id": "object_cube",
        "geometry_group_id": "cube_mesh", "observation_group_id": "capture_new_1",
        "proposal_group_id": "proposal_family_new_1",
        "mapping": {"status": mapping}, "execution_status": status,
        "outcome": {"success": success, "failure_type": failure_type,
                    "label_rule_id": "historical_task_outcome_rule_v1",
                    "first_failure_time_s": first_failure_time_s},
        "provenance_hashes": {key: HASH for key in (
            "code", "configuration", "inputs", "model", "assets", "native_pose",
            "mapped_pose", "label_rule")},
    }
    if mapping != "ok":
        payload["mapping"].update(code="test_refusal", message="synthetic mapping refusal")
    if status != "completed":
        payload["outcome"] = None
    payload.update(changes)
    return AttemptMetadata.model_validate(payload)


def recorder(*, future_value=100., bad_phase=None, cutoff_gap=False):
    result = PredictionRecorder(JOINTS, LIMITS)
    for index in range(1, 24):
        time = round(index * .04, 10)
        phase = "close_fingers" if index <= 20 else "lift"
        if bad_phase and index == bad_phase[0]:
            phase = bad_phase[1]
        if cutoff_gap:
            time += .001
        measured = index / 100 if index <= 20 else future_value
        result.append(
            time_s=time, action_interval_start_s=(0. if index == 1 else round((index - 1) * .04, 10) + (.001 if cutoff_gap else 0)),
            phase=phase, hand_qpos=np.full(24, measured), hand_qvel=np.full(24, 2 * measured),
            requested_target_rad=np.full(24, 2.), applied_target_rad=np.ones(24),
            diagnostics={"contact_count": np.asarray(index), "full_qpos": np.full(38, 500.)},
        )
    return result


def prefix():
    return PrefixSpec(protocol_id="synthetic_close_prefix_v1", prediction_time_s=.8,
                      history_steps=20, control_timestep_s=.04)


def rewrite_observables(root: Path, edit):
    with np.load(root / "observables.npz", allow_pickle=False) as source:
        values = {key: source[key].copy() for key in source.files}
    edit(values)
    np.savez_compressed(root / "observables.npz", **values)
    payload = json.loads((root / "attempt.json").read_text())
    payload["artifacts_sha256"]["observables.npz"] = file_sha256(root / "observables.npz")
    (root / "attempt.json").write_text(json.dumps(payload))


@pytest.mark.parametrize("status", [value.value for value in MappingStatus if value is not MappingStatus.OK])
def test_mapping_refusal_cannot_carry_physical_label(status):
    with pytest.raises(ValidationError, match="outcome null"):
        metadata(mapping=status)
    refusal = metadata(mapping=status, status="not_executed")
    assert refusal.outcome is None


def test_runtime_error_before_first_sample_preserves_denominator_without_fake_state(tmp_path):
    good, error = tmp_path / "good", tmp_path / "error"
    recorder().write(good, metadata("good"))
    payload = write_incomplete_attempt(error, metadata("error", status="runtime_error"))
    assert payload["recording"] is None and payload["artifacts_sha256"] == {}
    assert load_attempt(error)[2] is None
    report = export_recorded_dataset([good, error], tmp_path / "out", prefix_spec=prefix())
    assert report["counts"]["attempted"] == report["counts"]["executed"] == 2
    assert report["counts"]["physically_labeled"] == report["counts"]["included"] == 1
    assert report["excluded"][0]["reason"] == "unexecuted_or_runtime_error"


def test_empty_recorder_can_finalize_runtime_error_but_not_completed_outcome(tmp_path):
    result = PredictionRecorder(JOINTS, LIMITS)
    payload = result.write(tmp_path / "error", metadata(status="runtime_error"))
    assert payload["attempt"]["outcome"] is None
    with pytest.raises(ValueError, match="no samples"):
        PredictionRecorder(JOINTS, LIMITS).write(tmp_path / "completed", metadata())


@pytest.mark.parametrize("failure", ["invalid_retarget", "unreachable_pregrasp", "runtime_error"])
def test_setup_errors_are_not_physical_failure_types(failure):
    with pytest.raises(ValidationError, match="physical task failure"):
        PhysicalOutcome(success=False, failure_type=failure, label_rule_id="rule_v1")


@pytest.mark.parametrize("changes", [
    {"success": 0, "failure_type": "drop"},
    {"success": True, "failure_type": "drop"},
    {"success": False, "failure_type": None},
    {"success": True, "first_failure_time_s": .1},
    {"success": False, "failure_type": "drop", "first_failure_time_s": float("nan")},
    {"success": False, "failure_type": "drop", "first_failure_time_s": True},
])
def test_outcome_consistency_and_strict_boolean(changes):
    with pytest.raises(ValidationError):
        PhysicalOutcome(label_rule_id="rule_v1", **changes)


def test_refusal_requires_reason_and_frozen_hashes():
    with pytest.raises(ValidationError, match="code and message"):
        MappingDecision(status="refused_guard")
    with pytest.raises(ValidationError, match="code, configuration and input hashes"):
        metadata(provenance_hashes={})
    with pytest.raises(ValidationError, match="sha256"):
        metadata(provenance_hashes={"code": "unknown", "configuration": HASH, "inputs": HASH})


def test_immutable_samples_align_requested_applied_and_measured_arrays(tmp_path):
    result = recorder()
    result.write(tmp_path / "attempt", metadata())
    attempt, payload, arrays = load_attempt(tmp_path / "attempt")
    assert attempt.outcome.success is True
    assert arrays["hand_qpos"].shape == arrays["hand_qvel"].shape == (23, 24)
    np.testing.assert_array_equal(arrays["requested_target_rad"], 2.)
    np.testing.assert_array_equal(arrays["applied_target_rad"], 1.)
    assert "full_qpos" not in arrays and "contact_count" not in arrays
    assert payload["recording"]["samples"] == 23
    with np.load(tmp_path / "attempt/diagnostics.npz", allow_pickle=False) as source:
        np.testing.assert_array_equal(source["time_s"], arrays["time_s"])
    with pytest.raises(ValueError, match="already been finalized"):
        result.write(tmp_path / "other", metadata())


def test_recorder_copies_inputs_and_returns_copies():
    result = PredictionRecorder(JOINTS, LIMITS)
    source = np.zeros(24)
    diagnostic = np.zeros(38)
    result.append(time_s=.04, action_interval_start_s=0, phase="close_fingers",
                  hand_qpos=source, hand_qvel=source, requested_target_rad=source,
                  applied_target_rad=source, diagnostics={"full_qpos": diagnostic})
    source[:] = 5.
    diagnostic[:] = 5.
    values = result.arrays()
    np.testing.assert_array_equal(values["hand_qpos"], 0.)
    values["hand_qpos"][:] = 8.
    np.testing.assert_array_equal(result.arrays()["hand_qpos"], 0.)


@pytest.mark.parametrize("bad_key", ["hand_qvel", "phase", "joint_names"])
def test_missing_observable_fields_fail_closed(bad_key):
    values = recorder().arrays()
    values.pop(bad_key)
    with pytest.raises(ValueError, match="strict allowlist"):
        validate_observables(values)


@pytest.mark.parametrize("edit, message", [
    (lambda a: a.update(hand_qvel=np.zeros((22, 24))), "shape"),
    (lambda a: a["time_s"].__setitem__(1, a["time_s"][0]), "strictly increasing"),
    (lambda a: a["action_interval_start_s"].__setitem__(2, .001), "aligned"),
    (lambda a: a["hand_qpos"].__setitem__((0, 0), np.nan), "finite"),
    (lambda a: a["applied_target_rad"].__setitem__((0, 0), 2.), "clipped"),
    (lambda a: a.update(contact_count=np.zeros(23)), "strict allowlist"),
])
def test_bad_shapes_timestamps_controls_and_diagnostics_are_rejected(edit, message):
    values = recorder().arrays()
    edit(values)
    with pytest.raises(ValueError, match=message):
        validate_observables(values)


def test_recorder_rejects_unaligned_append_without_keeping_bad_row():
    result = recorder()
    with pytest.raises(ValueError, match="contiguous"):
        result.append(time_s=1.0, action_interval_start_s=.8, phase="close_fingers",
                      hand_qpos=np.zeros(24), hand_qvel=np.zeros(24),
                      requested_target_rad=np.zeros(24), applied_target_rad=np.zeros(24))
    assert result.sample_count == 23


def test_prediction_export_excludes_refusals_errors_and_early_collisions(tmp_path):
    good, refusal, error, early = [tmp_path / name for name in ("good", "refusal", "error", "early")]
    recorder().write(good, metadata())
    write_unexecuted_attempt(refusal, metadata("refused", mapping="refused_guard", status="not_executed"))
    recorder().write(error, metadata("runtime", status="runtime_error"))
    recorder().write(early, metadata("early", success=False, failure_type="approach_collision"))
    report = export_recorded_dataset([good, refusal, error, early], tmp_path / "derived", prefix_spec=prefix())
    assert report["counts"] == {
        "attempted": 4, "mapped": 3, "executed": 3, "physically_labeled": 2,
        "included": 1, "excluded": 3, "success": 1, "failure": 0,
        "proposal_groups": 1, "observation_groups": 1, "geometry_groups": 1,
    }
    assert {row["reason"] for row in report["excluded"]} == {
        "mapping_refusal", "unexecuted_or_runtime_error", "failure_at_or_before_cutoff"}
    bundle = load_recorded_dataset(tmp_path / "derived")
    assert bundle.X.shape == (1, 1440)
    assert bundle.y.tolist() == [1]
    assert bundle.episode_ids.tolist() == ["attempt_1"]
    assert "unassigned" in report["split_policy"]


def test_mutated_refusal_label_fails_instead_of_becoming_negative_training_example(tmp_path):
    root = tmp_path / "refusal"
    write_unexecuted_attempt(root, metadata(mapping="refused_invalid_retarget", status="not_executed"))
    payload = json.loads((root / "attempt.json").read_text())
    payload["attempt"]["outcome"] = {"success": False, "failure_type": "drop", "label_rule_id": "rule_v1"}
    (root / "attempt.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="outcome null"):
        export_recorded_dataset([root], tmp_path / "dataset", prefix_spec=prefix())


def test_export_only_uses_causal_measured_and_applied_history(tmp_path):
    outputs = []
    for index, future in enumerate((100., -1000.)):
        source = tmp_path / f"raw{index}"
        recorder(future_value=future).write(source, metadata())
        # The diagnostic artifact need not be readable by a predictor process.
        (source / "diagnostics.npz").write_bytes(b"not an npz predictor input")
        export_recorded_dataset([source], tmp_path / f"out{index}", prefix_spec=prefix())
        outputs.append(load_recorded_dataset(tmp_path / f"out{index}").X)
    np.testing.assert_array_equal(*outputs)
    values = outputs[0].reshape(1, 20, 3, 24)
    np.testing.assert_array_equal(values[0, -1, 0], np.float32(.2))
    np.testing.assert_array_equal(values[0, -1, 1], np.float32(.4))
    np.testing.assert_array_equal(values[0, -1, 2], 1.)
    # Existing model feature summarization accepts the same 20×3×24 layout;
    # the historical training CLI intentionally retains its old schema loader.
    from grasp_failure_prediction.prediction.model import summarize_prefix
    assert summarize_prefix(outputs[0]).shape == (1, 288)


@pytest.mark.parametrize("bad_phase, cutoff_gap, message", [
    ((1, "lift"), False, "before any lift"),
    ((10, "approach"), False, "history must be closing"),
    (None, True, "insufficient samples|cutoff endpoint"),
])
def test_export_enforces_phase_and_exact_cutoff(tmp_path, bad_phase, cutoff_gap, message):
    recorder(bad_phase=bad_phase, cutoff_gap=cutoff_gap).write(tmp_path / "raw", metadata())
    with pytest.raises(ValueError, match=message):
        export_recorded_dataset([tmp_path / "raw"], tmp_path / "out", prefix_spec=prefix())


def test_export_excludes_explicit_failure_on_cutoff(tmp_path):
    recorder().write(tmp_path / "good", metadata("good"))
    recorder().write(tmp_path / "early", metadata("early", success=False,
        failure_type="drop", first_failure_time_s=.8))
    report = export_recorded_dataset([tmp_path / "good", tmp_path / "early"],
                                     tmp_path / "out", prefix_spec=prefix())
    assert report["counts"]["failure"] == 0
    assert report["excluded"][0]["reason"] == "failure_at_or_before_cutoff"


def test_hash_and_field_validation_at_artifact_boundary(tmp_path):
    recorder().write(tmp_path / "raw", metadata())
    (tmp_path / "raw/observables.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_attempt(tmp_path / "raw")


def test_extra_observable_field_is_rejected_even_with_updated_hash(tmp_path):
    recorder().write(tmp_path / "raw", metadata())
    rewrite_observables(tmp_path / "raw", lambda a: a.update(object_height=np.zeros(23)))
    with pytest.raises(ValueError, match="strict allowlist"):
        load_attempt(tmp_path / "raw")


def test_recording_refuses_overwrite_or_samples_for_unexecuted_mapping(tmp_path):
    result = recorder()
    with pytest.raises(ValueError, match="unexecuted/refused"):
        result.write(tmp_path / "refused", metadata(mapping="refused_guard", status="not_executed"))
    result.write(tmp_path / "raw", metadata())
    with pytest.raises(ValueError, match="nonempty"):
        recorder().write(tmp_path / "raw", metadata("other"))


def test_export_rejects_mixed_label_protocol_and_joint_order(tmp_path):
    recorder().write(tmp_path / "one", metadata("one"))
    recorder().write(tmp_path / "two", metadata("two", execution_protocol_id="changed_controller_v1"))
    with pytest.raises(ValueError, match="cannot mix"):
        export_recorded_dataset([tmp_path / "one", tmp_path / "two"], tmp_path / "out", prefix_spec=prefix())
    recorder().write(tmp_path / "three", metadata("three"))
    rewrite_observables(tmp_path / "three", lambda a: a.update(joint_names=a["joint_names"][::-1]))
    with pytest.raises(ValueError, match="joint ordering"):
        export_recorded_dataset([tmp_path / "one", tmp_path / "three"], tmp_path / "other", prefix_spec=prefix())


def test_excluded_attempt_denominator_cannot_mix_controller_protocols(tmp_path):
    recorder().write(tmp_path / "one", metadata("one"))
    write_incomplete_attempt(tmp_path / "error", metadata("error", status="runtime_error",
        execution_protocol_id="unrelated_controller_v1"))
    with pytest.raises(ValueError, match="cannot mix"):
        export_recorded_dataset([tmp_path / "one", tmp_path / "error"], tmp_path / "out", prefix_spec=prefix())


def test_exported_dataset_cannot_smuggle_diagnostic_feature_contract(tmp_path):
    recorder().write(tmp_path / "raw", metadata())
    export_recorded_dataset([tmp_path / "raw"], tmp_path / "out", prefix_spec=prefix())
    payload = json.loads((tmp_path / "out/manifest.json").read_text())
    payload["input_contract"]["fields"][-1] = "contact_count"
    (tmp_path / "out/manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="strict allowlist"):
        load_recorded_dataset(tmp_path / "out")
