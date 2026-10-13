from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from grasp_failure_prediction.grasp_eval import build_parser, main
from grasp_failure_prediction.proposals.workflow import (
    experiment_status,
    load_experiment,
    run_experiment,
    validate_experiment,
)


GEOMETRY_HASH = "sha256:45c1162a4f815d65c43254f2f499f53bde1eecea4d39912d8ce36d2c5cd92a81"


def _experiment(tmp_path: Path) -> Path:
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "saved.json").write_text(
        (Path(__file__).parent / "fixtures/provider_saved_cube.json").read_text()
    )
    request = {
        "schema_version": "proposal_request_v1",
        "request_id": "saved_cube_request",
        "provider": {"id": "shadow_saved_record_v1"},
        "object": {"id": "object01", "geometry_content_hash": GEOMETRY_HASH},
        "embodiment_id": "shadow_hand_right",
        "provider_input_path": "inputs/saved.json",
        "seed": 7,
        "sample_count": 1,
    }
    (tmp_path / "request.yaml").write_text(yaml.safe_dump(request, sort_keys=False))
    experiment = {
        "schema_version": "grasp_eval_experiment_v1",
        "experiment_id": "saved_cube_pilot",
        "output_root": "runs/saved_cube_pilot",
        "record_video": False,
        "jobs": [
            {
                "id": "saved_cube",
                "request": "request.yaml",
                "evaluations": [
                    {
                        "case_id": "saved_cube_nominal",
                        "proposal_sample_index": 0,
                        "environment": {"id": "adroit_shadow_tabletop_v2"},
                        "embodiment": {"id": "shadow_hand_right"},
                        "execution_protocol": {"id": "fixed_grasp_lift_v2"},
                        "object": {
                            "id": "object01",
                            "mass_kg": 0.18,
                            "friction_profile_id": "nominal_v1",
                        },
                        "initial_condition": {
                            "id": "centered_upright",
                            "object_position_m": [0.0, 0.0, 0.03],
                            "object_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                        "motion_profile_id": "nominal_lift_v1",
                        "seed": 19,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment, sort_keys=False))
    return path


def test_experiment_validation_resolves_provider_and_cases(tmp_path):
    path = _experiment(tmp_path)
    summary = validate_experiment(path, project_root=tmp_path)
    assert summary == {
        "schema_version": "grasp_eval_experiment_v1",
        "experiment_id": "saved_cube_pilot",
        "job_count": 1,
        "case_count": 1,
        "jobs": [
            {
                "job_id": "saved_cube",
                "provider_id": "shadow_saved_record_v1",
                "request_id": "saved_cube_request",
                "proposal_count": 1,
                "case_count": 1,
            }
        ],
        "valid": True,
    }


def test_experiment_rejects_duplicate_case_ids(tmp_path):
    path = _experiment(tmp_path)
    payload = yaml.safe_load(path.read_text())
    duplicate = dict(payload["jobs"][0]["evaluations"][0])
    payload["jobs"][0]["evaluations"].append(duplicate)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(ValueError, match="case IDs must be unique"):
        load_experiment(path)


def test_run_and_resume_saved_provider_experiment(tmp_path):
    path = _experiment(tmp_path)
    summary = run_experiment(path, project_root=tmp_path)
    assert summary["case_count"] == 1
    output = tmp_path / "runs/saved_cube_pilot"
    assert (output / "proposals/saved_cube/manifest.json").is_file()
    assert (output / "cases/saved_cube_nominal.json").is_file()
    assert (output / "runs/saved_cube_nominal/result.json").is_file()
    result_before = (output / "runs/saved_cube_nominal/result.json").read_bytes()

    resumed = run_experiment(path, project_root=tmp_path, resume=True)
    assert resumed == summary
    assert (output / "runs/saved_cube_nominal/result.json").read_bytes() == result_before
    assert experiment_status(path, project_root=tmp_path) == {
        "experiment_id": "saved_cube_pilot",
        "output_root": str(output),
        "expected": 1,
        "completed": 1,
        "remaining": 0,
        "successful": summary["success_count"],
        "failed": summary["failure_count"],
        "complete": True,
    }


def test_cli_commands_are_discoverable_and_validate_prints_json(tmp_path, capsys):
    parser = build_parser()
    assert parser.parse_args(["status", "experiment.yaml"]).command == "status"
    path = _experiment(tmp_path)
    assert main(["validate", str(path), "--project-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True


def test_view_command_uses_experiment_run_directory(tmp_path, monkeypatch):
    path = _experiment(tmp_path)
    observed = {}

    def fake_serve(root, host, port, *, open_browser, evaluation_root=None):
        observed.update(root=root, host=host, port=port, open_browser=open_browser)

    monkeypatch.setattr("grasp_failure_prediction.grasp_eval.serve_results", fake_serve)
    assert main([
        "view",
        str(path),
        "--project-root",
        str(tmp_path),
        "--port",
        "9001",
        "--no-open",
    ]) == 0
    assert observed == {
        "root": tmp_path / "runs/saved_cube_pilot/runs",
        "host": "127.0.0.1",
        "port": 9001,
        "open_browser": False,
    }
