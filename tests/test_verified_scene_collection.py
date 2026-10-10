"""Collector content-binding tests; these fixtures are not scientific captures.

The tiny RGB-D files and inference report below are deliberate test fixtures.
Only a previously generated local, explicitly selected HUG pickle is loaded;
this suite neither generates a fresh proposal nor certifies its capture truth.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from grasp_failure_prediction.evaluation import noisy_rollouts
from grasp_failure_prediction.evaluation.case_runner import _geometry_content_hash
from grasp_failure_prediction.evaluation.registry import resolve_case
from grasp_failure_prediction.evaluation.scene_contracts import build_scene_contract
from grasp_failure_prediction.evaluation.schema import EvaluationCase
from test_evaluation_schema import valid_case


TRUSTED_SOURCE = (Path(os.environ["GRASP_PILOT_TRUSTED_SOURCE"])
                  if os.environ.get("GRASP_PILOT_TRUSTED_SOURCE") else None)


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")


@pytest.fixture
def bound_case(tmp_path):
    if (TRUSTED_SOURCE is None or not (TRUSTED_SOURCE / "proposal.pkl").is_file()
            or not (TRUSTED_SOURCE / "T_world_camera.npy").is_file()):
        pytest.skip("GRASP_PILOT_TRUSTED_SOURCE must select a locally generated HUG proposal")
    Image = pytest.importorskip("PIL.Image")
    observation = tmp_path / "source_observation"
    observation.mkdir()
    Image.fromarray(np.full((2, 2, 3), [30, 80, 140], dtype=np.uint8)).save(observation / "rgb.png")
    Image.fromarray(np.full((2, 2), 500, dtype=np.uint16)).save(observation / "depth.png")
    Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(observation / "object_mask.png")
    np.save(observation / "depth_m.npy", np.full((2, 2), 0.5, dtype=np.float32))
    np.save(observation / "intrinsics.npy", np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]))
    shutil.copyfile(TRUSTED_SOURCE / "T_world_camera.npy", observation / "T_world_camera.npy")
    file_hashes = {path.name: digest(path) for path in sorted(observation.iterdir())}
    payload = valid_case()
    payload["grasp"]["prediction_path"] = "proposal/proposal.pkl"
    payload["grasp"]["observation_path"] = "source_observation"
    resolved = resolve_case(EvaluationCase.model_validate(payload))
    contract = build_scene_contract(
        object_id=payload["object"]["id"], geometry_content_hash=_geometry_content_hash(None),
        mass_kg=payload["object"]["mass_kg"],
        object_position_m=payload["initial_condition"]["object_position_m"],
        object_orientation_xyzw=payload["initial_condition"]["object_orientation_xyzw"],
        environment_config_hash=resolved.environment_config_hash,
        camera_sha256=file_hashes["T_world_camera.npy"], observation_files_sha256=file_hashes,
    )
    write_json(observation / "scene_contract.json", contract)
    proposal = tmp_path / "proposal"
    shutil.copytree(observation, proposal)
    # This is the only pickle loaded, selected by the test operator from prior
    # locally generated output. Never download or unpickle external test data.
    shutil.copyfile(TRUSTED_SOURCE / "proposal.pkl", proposal / "proposal.pkl")
    report = {
        "test_fixture_only": True,
        "real_hug_inference_passed": True,
        "proposal_sha256": digest(proposal / "proposal.pkl").removeprefix("sha256:"),
        "scene_contract_sha256": digest(proposal / "scene_contract.json").removeprefix("sha256:"),
        "observation_hashes": {
            name: file_hashes[file_name].removeprefix("sha256:")
            for name, file_name in {
                "rgb_path": "rgb.png", "depth_m_path": "depth_m.npy",
                "intrinsics_path": "intrinsics.npy", "mask_path": "object_mask.png",
            }.items()
        },
    }
    write_json(proposal / "inference_report.json", report)
    case_path = tmp_path / "case.yaml"
    case_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    manifest = tmp_path / "cases.json"
    write_json(manifest, ["case.yaml"])
    return SimpleNamespace(project=tmp_path, payload=payload, case_path=case_path,
                           observation=observation, proposal=proposal, manifest=manifest)


def test_bound_cube_case_collects_one_physical_episode_with_verified_metadata(bound_case):
    fixture = bound_case
    prepared = noisy_rollouts._prepare_case(fixture.case_path, fixture.project)
    assert prepared.source["scene_contract_status"] == "verified"
    assert prepared.source["scene_contract_sha256"] == digest(fixture.observation / "scene_contract.json")
    output = fixture.project / "collected"
    summary = noisy_rollouts.collect_noisy_rollouts(
        fixture.manifest, output, project_root=fixture.project,
        noise_amplitudes_rad=[0.0], repetitions=1,
    )
    assert summary["planned"] == summary["completed"] == 1
    assert summary["errors"] == 0
    assert summary["successes"] + summary["failures"] == 1
    metadata = json.loads((output / "episode_000000" / "collection_metadata.json").read_text())
    result = json.loads((output / "episode_000000" / "result.json").read_text())
    events = json.loads((output / "episode_000000" / "events.json").read_text())
    assert metadata["scene_contract_status"] == "verified"
    assert metadata["label"] == int(result["success"]) == events["binary_outcome"]["label"]
    assert metadata["geometry_group_id"] == prepared.source["geometry_group_id"]
    with np.load(output / "episode_000000" / "trajectory.npz", allow_pickle=False) as trace:
        assert len(trace["time_s"]) > 1
        assert np.all(np.diff(trace["time_s"]) > 0)


@pytest.mark.parametrize(("mismatch", "error_fragment"), [
    ("mass", "mass_kg mismatch"),
    ("pose", "object_position_m mismatch"),
    ("observation_bytes", "observation_files_sha256 mismatch"),
    ("proposal_bytes", "inference report does not bind"),
    ("proposal_report", "inference report does not bind"),
    ("contract_report", "inference report does not bind"),
    ("input_report", "inference inputs do not match"),
    ("contract_copy", "must match the captured observation contract"),
    ("missing_proposal_contract", "scene contract"),
])
def test_scene_mismatches_are_unlabeled_validation_errors(bound_case, mismatch, error_fragment):
    fixture = bound_case
    if mismatch == "mass":
        fixture.payload["object"]["mass_kg"] = 0.21
    elif mismatch == "pose":
        fixture.payload["initial_condition"]["object_position_m"][0] += 0.01
    elif mismatch == "observation_bytes":
        path = fixture.observation / "rgb.png"
        path.write_bytes(path.read_bytes() + b"\n")
    elif mismatch == "proposal_bytes":
        # Trailing bytes retain the decoded trusted proposal while changing its
        # byte identity; the copied inference report must reject this change.
        path = fixture.proposal / "proposal.pkl"
        path.write_bytes(path.read_bytes() + b"\n")
    elif mismatch in {"proposal_report", "contract_report", "input_report"}:
        path = fixture.proposal / "inference_report.json"
        report = json.loads(path.read_text())
        if mismatch == "proposal_report":
            report["proposal_sha256"] = "0" * 64
        elif mismatch == "contract_report":
            report["scene_contract_sha256"] = "0" * 64
        else:
            report["observation_hashes"]["rgb_path"] = "0" * 64
        write_json(path, report)
    elif mismatch == "contract_copy":
        # A semantically identical copy still differs in raw bytes. The
        # capture sidecar must have been copied unchanged into inference.
        path = fixture.proposal / "scene_contract.json"
        path.write_text(path.read_text() + "\n")
    else:
        (fixture.proposal / "scene_contract.json").unlink()
    fixture.case_path.write_text(yaml.safe_dump(fixture.payload, sort_keys=False))
    output = fixture.project / "rejected"
    summary = noisy_rollouts.collect_noisy_rollouts(
        fixture.manifest, output, project_root=fixture.project,
        noise_amplitudes_rad=[0.0], repetitions=1,
    )
    assert summary == {"planned": 1, "completed": 0, "successes": 0, "failures": 0, "errors": 1}
    row = json.loads((output / "index.jsonl").read_text())
    assert row["status"] == "validation_failed"
    assert row["label"] is None
    assert error_fragment in row["error"]
    assert not (output / "episode_000000" / "result.json").exists()


def test_baseline_once_keeps_one_zero_trial_per_case_and_paired_distinct_repetition_seeds(tmp_path, monkeypatch):
    manifest = tmp_path / "cases.json"
    write_json(manifest, ["case_a.yaml", "case_b.yaml"])
    captured_entries = []

    def prepare(case_path, project):
        name = case_path.stem
        return SimpleNamespace(source={
            "group_id": "proposal_" + name, "observation_group_id": "scene_" + name,
            "object_id": "object01", "geometry_group_id": "same_cube_geometry",
        })

    def record(prepared, output, entry):
        captured_entries.append(dict(entry))
        return dict(entry, status="completed", label=1, failure_type=None)

    monkeypatch.setattr(noisy_rollouts, "_prepare_case", prepare)
    monkeypatch.setattr(noisy_rollouts, "_run_episode", record)
    output = tmp_path / "plan_output"
    summary = noisy_rollouts.collect_noisy_rollouts(
        manifest, output, project_root=tmp_path, noise_amplitudes_rad=[0.0, 0.15, 0.6],
        repetitions=3, seed=9000, baseline_once=True,
    )
    expected_count = 2 * (1 + 2 * 3)
    assert summary["planned"] == summary["completed"] == expected_count
    assert summary["errors"] == 0
    assert len(captured_entries) == expected_count
    plan = json.loads((output / "plan.json").read_text())
    assert plan["baseline_once"] is True
    assert plan["episodes"] == captured_entries
    case_seeds = []
    for case_index in range(2):
        rows = [row for row in captured_entries if row["case_index"] == case_index]
        baseline = [row for row in rows if row["noise_amplitude_rad"] == 0]
        assert len(baseline) == 1 and baseline[0]["repetition"] == 0
        seeds = set()
        for repetition in range(3):
            noisy = [row for row in rows if row["repetition"] == repetition and row["noise_amplitude_rad"] > 0]
            assert {row["noise_amplitude_rad"] for row in noisy} == {0.15, 0.6}
            assert len({row["noise_seed"] for row in noisy}) == 1
            seeds.add(noisy[0]["noise_seed"])
        assert len(seeds) == 3
        assert baseline[0]["noise_seed"] in seeds
        case_seeds.append(seeds)
    assert case_seeds[0].isdisjoint(case_seeds[1])
