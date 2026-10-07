from __future__ import annotations

import json
from pathlib import Path

from grasp_failure_prediction.evaluation.result_viewer import discover_results


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
