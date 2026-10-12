from __future__ import annotations

import json
import numpy as np

from grasp_failure_prediction.evaluation.artifacts import write_evaluation_artifacts
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
from grasp_failure_prediction.evaluation.registry import resolve_case
from grasp_failure_prediction.evaluation.retargeting import (
    ShadowHandRetargeter,
    default_dex_urdf_root,
)
from grasp_failure_prediction.evaluation.runner import AdroitShadowRunner
from grasp_failure_prediction.evaluation.schema import EvaluationCase, EvaluationResult
from grasp_failure_prediction.evaluation.scoring import score_trace
from test_evaluation_schema import valid_case
from test_retargeting import synthetic_hug_prediction


def test_result_bundle_contains_replay_and_provenance_artifacts(tmp_path) -> None:
    resolved = resolve_case(EvaluationCase.model_validate(valid_case()))
    grasp, _ = synthetic_hug_prediction()
    retargeter = ShadowHandRetargeter()
    pose = retargeter.retarget(grasp)
    validation = ShadowPoseValidator(default_dex_urdf_root()).validate(pose)
    runner = AdroitShadowRunner(
        resolved.execution_protocol, default_dex_urdf_root()
    )
    runner.reset(
        seed=resolved.case.seed,
        object_mass_kg=resolved.case.object.mass_kg,
        object_position_m=np.asarray(resolved.case.initial_condition.object_position_m),
        object_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    trace = runner.execute(
        pose,
        grasp_palm_position_m=np.array([0.0, 0.0, 0.09]),
        grasp_palm_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    outcome = score_trace(
        trace,
        resolved.execution_protocol,
        control_timestep_s=runner.control_timestep_s,
    )
    result = write_evaluation_artifacts(
        tmp_path,
        resolved=resolved,
        trace=trace,
        pose_validation=validation,
        outcome=outcome,
        final_qpos=runner.data.qpos.copy(),
        final_qvel=runner.data.qvel.copy(),
    )
    assert result.case_id == resolved.case.case_id
    for name in ("trajectory.npz", "resolved_case.json", "final_state.npz", "result.json"):
        assert (tmp_path / name).is_file()
    loaded = EvaluationResult.model_validate(json.loads((tmp_path / "result.json").read_text()))
    assert loaded.resolved_execution.environment_config_hash == resolved.environment_config_hash
    trajectory = np.load(tmp_path / "trajectory.npz")
    assert len(trajectory["time_s"]) == len(trace.steps)
    assert "opposing_hand_object_contact" in trajectory.files
    assert "maximum_contact_normal_force_n" in trajectory.files
    assert "maximum_actuator_force_fraction" in trajectory.files
