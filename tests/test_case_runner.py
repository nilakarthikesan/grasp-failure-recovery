from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from grasp_failure_prediction.evaluation.case_runner import (
    _hug_world_palm_pose,
    _run_case_inference,
    _table_parallel_world_wrist,
    run_case,
)
from grasp_failure_prediction.evaluation.pose_validation import (
    ShadowPoseValidator,
    _body_id,
)
from grasp_failure_prediction.evaluation.retargeting import (
    ShadowHandRetargeter,
    default_dex_urdf_root,
)
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
from grasp_failure_prediction.evaluation.schema import EvaluationCase
from grasp_failure_prediction.evaluation.registry import RegistryError
from test_evaluation_schema import valid_case
from test_retargeting import synthetic_hug_prediction


def write_test_case(root: Path, *, bad_hash: bool = False) -> Path:
    grasp, _ = synthetic_hug_prediction()
    prediction = root / "grasps" / "object01" / "grasp_003.pkl"
    prediction.parent.mkdir(parents=True)
    fields = ("pose", "shape", "t", "T_camera_wrist", "landmarks_3d", "mesh_vertices")
    with prediction.open("wb") as handle:
        pickle.dump({name: getattr(grasp, name) for name in fields}, handle)
    payload = valid_case()
    if bad_hash:
        payload["environment"]["expected_config_hash"] = "sha256:" + "0" * 64
    case_path = root / "case.yaml"
    case_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return case_path


def test_one_case_command_writes_valid_result_bundle(tmp_path) -> None:
    case_path = write_test_case(tmp_path)
    output = tmp_path / "run"
    result = run_case(case_path, output, inference=False, project_root=tmp_path)
    assert result.status == "completed"
    assert (output / "result.json").is_file()
    payload = json.loads((output / "result.json").read_text())
    assert payload["resolved_execution"]["execution_protocol_id"] == "fixed_grasp_lift_v2"


def test_hash_mismatch_fails_before_output_or_simulation(tmp_path) -> None:
    case_path = write_test_case(tmp_path, bad_hash=True)
    output = tmp_path / "run"
    with pytest.raises(RegistryError, match="configuration hash mismatch"):
        run_case(case_path, output, inference=False, project_root=tmp_path)
    assert not output.exists()


def test_saved_proposal_mode_requires_prediction_path(tmp_path) -> None:
    payload = valid_case()
    del payload["grasp"]["prediction_path"]
    payload["grasp"]["observation_path"] = "observations/object01"
    case_path = tmp_path / "case.yaml"
    case_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="saved-proposal mode requires"):
        run_case(
            case_path, tmp_path / "run", inference=False, project_root=tmp_path
        )


def test_repeated_case_produces_identical_trajectory(tmp_path) -> None:
    case_path = write_test_case(tmp_path)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    run_case(case_path, first_output, inference=False, project_root=tmp_path)
    run_case(case_path, second_output, inference=False, project_root=tmp_path)
    first = np.load(first_output / "trajectory.npz")
    second = np.load(second_output / "trajectory.npz")
    assert set(first.files) == set(second.files)
    for key in first.files:
        np.testing.assert_array_equal(first[key], second[key])


def test_case_inference_isolated_from_existing_proposal(tmp_path, monkeypatch) -> None:
    payload = valid_case()
    payload["grasp"]["observation_path"] = "observations/object01"
    payload["grasp"]["inference_seed"] = 17
    case = EvaluationCase.model_validate(payload)
    observation = tmp_path / "observations" / "object01"
    observation.mkdir(parents=True)
    (observation / "proposal.pkl").write_bytes(b"stale")
    hug_root = tmp_path / "hug"
    hug_root.mkdir()
    checkpoint = tmp_path / "hug.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "run"

    def fake_run(command, *, cwd, check):
        inference_dir = output / "hug_inference"
        assert not (inference_dir / "proposal.pkl").exists()
        assert command[-2:] == ["--seed", "17"]
        assert cwd == tmp_path
        assert check is True
        (inference_dir / "proposal.pkl").write_bytes(b"fresh")

    monkeypatch.setattr(
        "grasp_failure_prediction.evaluation.case_runner.subprocess.run", fake_run
    )
    proposal = _run_case_inference(
        case,
        tmp_path,
        output,
        hug_root=hug_root,
        checkpoint=checkpoint,
    )
    assert proposal.read_bytes() == b"fresh"


def test_wrist_is_horizontal_and_centered_above_object() -> None:
    position, quaternion = _table_parallel_world_wrist(
        np.array([0.2, 0.1, 0.03]), 0.04
    )

    # The Shadow body origin is 4.5 cm behind its visible palm center. With the
    # hand facing down, shifting the origin +y puts the mesh center over object.
    np.testing.assert_allclose(position, [0.2, 0.145, 0.07], atol=1e-8)
    np.testing.assert_allclose(
        quaternion, [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], atol=1e-8
    )


def test_hug_world_pose_composes_corrected_operator_and_robot_palm_frames(
    tmp_path,
) -> None:
    grasp, _ = synthetic_hug_prediction()
    pose = ShadowHandRetargeter().retarget(grasp)
    validator = ShadowPoseValidator(default_dex_urdf_root())
    np.save(tmp_path / "T_world_camera.npy", np.eye(4))

    position, quaternion = _hug_world_palm_pose(
        grasp, pose, validator, tmp_path
    )
    actual_rotation = np.empty(9)
    import mujoco

    mujoco.mju_quat2Mat(actual_rotation, quaternion)
    base_to_palm = validator.data.xmat[
        _body_id(validator.model, "palm")
    ].reshape(3, 3)
    expected_rotation = (
        grasp.T_camera_wrist[:3, :3]
        @ MANO_TO_OPERATOR_RIGHT.T
        @ base_to_palm
    )
    np.testing.assert_allclose(position, grasp.T_camera_wrist[:3, 3])
    np.testing.assert_allclose(actual_rotation.reshape(3, 3), expected_rotation)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific viewer launch")
def test_viewer_flag_is_exposed_by_cli() -> None:
    # Interactive launch is intentionally manual; this test documents that the
    # supported macOS path lives in run_case rather than a separate simulator.
    assert "viewer" in run_case.__annotations__
