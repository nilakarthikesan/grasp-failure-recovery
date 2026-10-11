from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from grasp_failure_prediction.evaluation.result_viewer import discover_results, load_frozen_evaluation


def test_discover_results_indexes_metrics_and_existing_video(tmp_path: Path) -> None:
    run = tmp_path / "batch" / "run_00"
    run.mkdir(parents=True)
    (run / "rollout.mp4").write_bytes(b"video")
    (run / "result.json").write_text(
        json.dumps(
            {
                "case_id": "case_00",
                "status": "completed",
                "success": True,
                "failure_type": None,
                "resolved_execution": {
                    "environment_id": "environment_v2",
                    "execution_protocol_id": "protocol_v2",
                },
                "retargeting": {"mean_fingertip_error_m": 0.004},
                "outcome": {"maximum_lift_m": 0.15, "hold_duration_s": 2.0},
                "artifacts": {"video": "rollout.mp4"},
            }
        )
    )
    (run / "resolved_case.json").write_text(
        json.dumps(
            {
                "case": {
                    "object": {"id": "cube", "mass_kg": 0.18},
                    "grasp": {"id": "grasp_00", "inference_seed": 0},
                    "seed": 9000,
                }
            }
        )
    )

    results = discover_results(tmp_path)

    assert len(results) == 1
    assert results[0]["run_id"] == "batch/run_00"
    assert results[0]["has_video"] is True
    assert results[0]["video_url"] == "/files/batch/run_00/rollout.mp4"
    assert results[0]["maximum_lift_m"] == 0.15


def test_discover_results_tolerates_historical_bundle_without_video(tmp_path: Path) -> None:
    run = tmp_path / "old"
    run.mkdir()
    (run / "result.json").write_text(
        json.dumps({"case_id": "old_case", "success": False, "artifacts": {}})
    )

    result = discover_results(tmp_path)[0]

    assert result["has_video"] is False
    assert result["video_url"] is None
    assert "evaluation" not in result


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload))


def _bind_predictions(evaluation: Path) -> None:
    report = json.loads((evaluation / "report.json").read_text())
    report["manifest_sha256"] = _hash(evaluation / "manifest.json")
    report["predictions_sha256"] = _hash(evaluation / "predictions.csv")
    _write_json(evaluation / "report.json", report)


@pytest.fixture
def frozen_evaluation(tmp_path: Path):
    collection, evaluation = tmp_path / "collection", tmp_path / "evaluation"
    collection.mkdir()
    evaluation.mkdir()
    episodes, rows = [], []
    for index, (label, p_failure) in enumerate(((1, .8), (0, .3), (0, .9))):
        episode_id = f"episode_{index:06d}"
        raw = collection / episode_id
        raw.mkdir()
        video = index == 1
        if video:
            (raw / "rollout.mp4").write_bytes(b"video")
        _write_json(raw / "result.json", dict(status="completed", success=bool(label),
                    artifacts={"video": "rollout.mp4"} if video else {}, case_id=episode_id))
        _write_json(raw / "collection_metadata.json", dict(status="completed", label=label,
                    noise_amplitude_rad=.15, geometry_group_id="geometry"))
        _write_json(raw / "resolved_case.json", {"case": {"object": {"id": "new_object"}}})
        episodes.append(dict(episode_id=episode_id, source_task_label=label,
                             geometry_novelty="new_geometry", geometry_group_id="geometry",
                             noise_amplitude_rad=.15, observed_at_s=2.9599999999998956,
                             raw_files_sha256={name: _hash(raw / name) for name in
                                               ("collection_metadata.json", "result.json")}))
        for model in ("training_prevalence", "proprioception_and_commands", "commands_only"):
            rows.append([episode_id, "fresh_evaluation", label, model, 1 - p_failure,
                         p_failure, int(p_failure >= .5)])
    counts = dict(indexed=3, included=3, excluded=0, success=1, failure=2)
    _write_json(evaluation / "manifest.json", dict(schema_version="fresh_task_prefix_dataset_v1",
                                                 counts=counts, episodes=episodes))
    _write_json(evaluation / "report.json", dict(schema_version="frozen_fresh_task_outcome_evaluation_v1",
                prediction_time_s=2.96, counts=counts,
                models={name: {} for name in ("training_prevalence", "proprioception_and_commands", "commands_only")}))
    with (evaluation / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_id", "partition", "label_success", "model", "p_success", "p_failure", "failure_alert"])
        writer.writerows(rows)
    _bind_predictions(evaluation)
    return collection, evaluation


def test_frozen_predictions_match_outcomes_and_prioritize_rendered_examples(frozen_evaluation) -> None:
    collection, evaluation = frozen_evaluation
    results = discover_results(collection, evaluation)
    assert results[0]["run_id"] == "episode_000001"
    by_id = {item["run_id"]: item["evaluation"] for item in results}
    assert by_id["episode_000000"]["model_error"] == "false_alarm"
    assert by_id["episode_000001"]["model_error"] == "missed_failure"
    assert by_id["episode_000002"]["model_error"] == "correct"
    assert by_id["episode_000001"]["p_failure"] == .3
    assert by_id["episode_000000"]["observed_at_s"] == pytest.approx(2.96)
    assert by_id["episode_000000"]["geometry_novelty"] == "new_geometry"
    assert by_id["episode_000000"]["noise_amplitude_rad"] == .15


def test_frozen_view_rejects_prediction_content_changed_after_evaluation(frozen_evaluation) -> None:
    collection, evaluation = frozen_evaluation
    with (evaluation / "predictions.csv").open("a") as handle:
        handle.write("changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        discover_results(collection, evaluation)


@pytest.mark.parametrize("change", ["nonfinite", "complement", "threshold", "duplicate", "missing"])
def test_frozen_view_rejects_inconsistent_prediction_rows(frozen_evaluation, change: str) -> None:
    collection, evaluation = frozen_evaluation
    path = evaluation / "predictions.csv"
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if change == "nonfinite":
        rows[1][5] = "nan"
    elif change == "complement":
        rows[1][4] = ".4"
    elif change == "threshold":
        rows[1][6] = "0"
    elif change == "duplicate":
        rows.append(rows[1])
    else:
        rows.pop(1)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    _bind_predictions(evaluation)
    with pytest.raises(ValueError):
        load_frozen_evaluation(collection, evaluation)


def test_frozen_view_rejects_actual_label_disagreement_even_with_valid_hashes(frozen_evaluation) -> None:
    collection, evaluation = frozen_evaluation
    result = collection / "episode_000000" / "result.json"
    payload = json.loads(result.read_text())
    payload["success"] = False
    _write_json(result, payload)
    manifest_path = evaluation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"][0]["raw_files_sha256"]["result.json"] = _hash(result)
    _write_json(manifest_path, manifest)
    _bind_predictions(evaluation)
    with pytest.raises(ValueError, match="differs from the raw attempt"):
        load_frozen_evaluation(collection, evaluation)


def test_frozen_view_rejects_report_outcome_counts_that_do_not_match_rows(frozen_evaluation) -> None:
    collection, evaluation = frozen_evaluation
    for filename in ("report.json", "manifest.json"):
        path = evaluation / filename
        payload = json.loads(path.read_text())
        payload["counts"].update(success=0, failure=3)
        _write_json(path, payload)
    _bind_predictions(evaluation)
    with pytest.raises(ValueError, match="label counts"):
        load_frozen_evaluation(collection, evaluation)


def test_frozen_view_rejects_raw_result_changed_after_evaluation(frozen_evaluation) -> None:
    collection, evaluation = frozen_evaluation
    (collection / "episode_000000" / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="raw result.json hash mismatch"):
        load_frozen_evaluation(collection, evaluation)


def test_frozen_view_rejects_prediction_history_after_frozen_cutoff(frozen_evaluation) -> None:
    collection, evaluation = frozen_evaluation
    path = evaluation / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["episodes"][0]["observed_at_s"] = 3.0
    _write_json(path, manifest)
    _bind_predictions(evaluation)
    with pytest.raises(ValueError, match="invalid frozen evaluation"):
        load_frozen_evaluation(collection, evaluation)
