"""Write a self-contained result bundle for one evaluation case."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from .pose_validation import PoseValidation
from .registry import ResolvedCase
from .runner import ExecutionTrace
from .schema import (
    ArtifactReferences,
    EvaluationResult,
    OutcomeMetrics,
    ResolvedExecution,
    RetargetingMetrics,
)
from .scoring import ScoredOutcome


def _code_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_evaluation_artifacts(
    output_dir: str | Path,
    *,
    resolved: ResolvedCase,
    trace: ExecutionTrace,
    pose_validation: PoseValidation,
    outcome: ScoredOutcome,
    final_qpos: np.ndarray,
    final_qvel: np.ndarray,
    video: Path | None = None,
    hug_proposal: Path | None = None,
    inference_report: Path | None = None,
) -> EvaluationResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory.npz"
    resolved_case_path = output / "resolved_case.json"
    final_state_path = output / "final_state.npz"
    result_path = output / "result.json"

    np.savez_compressed(
        trajectory_path,
        index=np.asarray([step.index for step in trace.steps], dtype=np.int64),
        time_s=np.asarray([step.time_s for step in trace.steps]),
        state=np.asarray([step.state.value for step in trace.steps]),
        palm_position_m=np.stack([step.palm_position_m for step in trace.steps]),
        hand_qpos=np.stack([step.hand_qpos for step in trace.steps]),
        object_position_m=np.stack([step.object_position_m for step in trace.steps]),
        contact_count=np.asarray([step.contact_count for step in trace.steps]),
        hand_object_contact=np.asarray(
            [step.hand_object_contact for step in trace.steps]
        ),
        hand_table_contact=np.asarray([step.hand_table_contact for step in trace.steps]),
        object_table_contact=np.asarray(
            [step.object_table_contact for step in trace.steps]
        ),
        opposing_hand_object_contact=np.asarray(
            [step.opposing_hand_object_contact for step in trace.steps]
        ),
        maximum_contact_normal_force_n=np.asarray(
            [step.maximum_contact_normal_force_n for step in trace.steps]
        ),
        maximum_actuator_force_fraction=np.asarray(
            [step.maximum_actuator_force_fraction for step in trace.steps]
        ),
    )
    np.savez_compressed(final_state_path, qpos=final_qpos, qvel=final_qvel)

    resolved_payload = {
        "case": resolved.case.model_dump(mode="json"),
        "environment": resolved.environment.model_dump(mode="json"),
        "environment_config_hash": resolved.environment_config_hash,
        "execution_protocol": resolved.execution_protocol.model_dump(mode="json"),
        "execution_protocol_config_hash": resolved.execution_protocol_config_hash,
    }
    _write_json(resolved_case_path, resolved_payload)

    try:
        dex_version = importlib.metadata.version("dex-retargeting")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        dex_version = "unknown"
    protocol_parameters = resolved.execution_protocol.parameters.model_dump(mode="json")
    result = EvaluationResult(
        case_id=resolved.case.case_id,
        status="completed",
        success=outcome.success,
        failure_type=outcome.failure_type.value if outcome.failure_type else None,
        resolved_execution=ResolvedExecution(
            environment_id=resolved.environment.environment_id,
            environment_config_hash=resolved.environment_config_hash,
            embodiment_id=resolved.case.embodiment.id,
            execution_protocol_id=resolved.execution_protocol.execution_protocol_id,
            execution_protocol_config_hash=resolved.execution_protocol_config_hash,
            execution_protocol_parameters=protocol_parameters,
            mujoco_version=mujoco.__version__,
            code_commit=_code_commit(),
        ),
        retargeting=RetargetingMetrics(
            library=f"dex-retargeting=={dex_version}",
            config_id="shadow_hand_right",
            mean_fingertip_error_m=pose_validation.mean_fingertip_error_m,
            maximum_fingertip_error_m=pose_validation.maximum_fingertip_error_m,
        ),
        outcome=OutcomeMetrics(
            acquired_object=outcome.acquired_object,
            maximum_lift_m=outcome.maximum_lift_m,
            hold_duration_s=outcome.hold_duration_s,
            final_object_height_m=outcome.final_object_height_m,
            opposing_contact_duration_s=sum(
                step.opposing_hand_object_contact for step in trace.steps
            )
            * resolved.environment.simulator.control_timestep_s,
            peak_contact_normal_force_n=max(
                (step.maximum_contact_normal_force_n for step in trace.steps),
                default=0.0,
            ),
            peak_actuator_force_fraction=max(
                (step.maximum_actuator_force_fraction for step in trace.steps),
                default=0.0,
            ),
        ),
        artifacts=ArtifactReferences(
            trajectory=Path("trajectory.npz"),
            resolved_case=Path("resolved_case.json"),
            simulator_state=Path("final_state.npz"),
            video=video,
            hug_proposal=hug_proposal,
            inference_report=inference_report,
        ),
    )
    _write_json(result_path, result.model_dump(mode="json"))
    return result
