"""Execute one canonical grasp proposal in the registered MuJoCo environment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import yaml

from grasp_failure_prediction.evaluation.artifacts import _code_commit, _write_json
from grasp_failure_prediction.evaluation.case_runner import (
    _ensure_empty_output,
    _execute,
    _geometry_content_hash,
    _resolve_object_geometry,
    _runner_class,
    _validate_case_configuration,
)
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
from grasp_failure_prediction.evaluation.registry import resolve_case
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
from grasp_failure_prediction.evaluation.schema import OutcomeMetrics, ResolvedExecution
from grasp_failure_prediction.evaluation.scoring import score_trace

from .artifacts import load_proposal, proposal_content_hash
from .registry import load_provider_registry
from .preflight import load_preflight_registry
from .schema import (
    EvaluationCaseV2,
    ProposalArtifactReferences,
    ProposalExecutionResult,
    ProposalExecutionValidation,
    ProposalPreflightRejection,
    ProposalRejectionArtifact,
    ResolvedProviderMetadata,
)


class ProposalRejected(ValueError):
    """A structured pre-rollout rejection which has no physical task label."""

    def __init__(self, rejection: ProposalPreflightRejection):
        self.rejection = rejection
        super().__init__(rejection.message)


def _reject(stage: str, code: str, message: str) -> None:
    raise ProposalRejected(ProposalPreflightRejection(
        status="proposal_rejected", stage=stage, code=code, message=message
    ))


@dataclass(frozen=True)
class NamedJointPose:
    joint_names: tuple[str, ...]
    qpos: np.ndarray


def load_execution_case(path: str | Path) -> EvaluationCaseV2:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation case must contain a YAML/JSON mapping")
    return EvaluationCaseV2.model_validate(payload)


def _matrix(position_m, quaternion_xyzw) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("pose quaternion must be finite and nonzero")
    quaternion_wxyz = (quaternion / norm)[[3, 0, 1, 2]]
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quaternion_wxyz)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.reshape(3, 3)
    result[:3, 3] = np.asarray(position_m, dtype=np.float64)
    return result


def _world_palm(case: EvaluationCaseV2, proposal) -> tuple[np.ndarray, np.ndarray]:
    world_object = _matrix(
        case.initial_condition.object_position_m,
        case.initial_condition.object_orientation_xyzw,
    )
    object_palm = _matrix(
        proposal.object_to_palm.translation_m,
        proposal.object_to_palm.quaternion_xyzw,
    )
    world_palm = world_object @ object_palm
    quaternion_wxyz = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion_wxyz, world_palm[:3, :3].reshape(9))
    return world_palm[:3, 3].copy(), quaternion_wxyz


def _validate_pose(proposal) -> tuple[NamedJointPose, ProposalExecutionValidation]:
    validator = ShadowPoseValidator(default_dex_urdf_root())
    expected = validator.joint_names
    actual = set(proposal.joint_targets_rad)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        _reject(
            "structural",
            "shadow_joint_set_mismatch",
            f"proposal joint set does not match Shadow Hand; missing={missing}, "
            f"unexpected={unexpected}",
        )
    qpos = np.asarray([proposal.joint_targets_rad[name] for name in expected])
    lower = validator.model.jnt_range[:, 0]
    upper = validator.model.jnt_range[:, 1]
    limited = validator.model.jnt_limited.astype(bool)
    tolerance = 1e-9
    if np.any(qpos[limited] < lower[limited] - tolerance) or np.any(
        qpos[limited] > upper[limited] + tolerance
    ):
        _reject("kinematic", "shadow_joint_limit_violation", "proposal joint target violates the Shadow Hand joint limits")
    margins = np.minimum(qpos[limited] - lower[limited], upper[limited] - qpos[limited])
    pose = NamedJointPose(expected, qpos)
    validator.set_pose(pose)
    return pose, ProposalExecutionValidation(
        joint_count=len(expected),
        minimum_joint_limit_margin_rad=max(0.0, float(np.min(margins))),
        self_contact_count=int(validator.data.ncon),
        provider_validation_status=proposal.provider_validation.status,
        provider_metrics=proposal.provider_validation.metrics,
    )


def _scene_preflight(proposal) -> None:
    preflight = load_preflight_registry().resolve("shadow_proposal_preflight_v1")
    translation = np.asarray(proposal.object_to_palm.translation_m, dtype=np.float64)
    if float(np.linalg.norm(translation)) > preflight.maximum_object_to_palm_distance_m:
        _reject(
            "scene_preflight",
            "palm_implausibly_distant",
            "palm exceeds the registered maximum distance from the object frame",
        )


def _write_artifacts(
    output: Path,
    *,
    resolved,
    proposal,
    proposal_hash: str,
    proposal_validation: ProposalExecutionValidation,
    trace,
    outcome,
    runner,
    video: Path | None,
    provider_spec,
) -> ProposalExecutionResult:
    trajectory = output / "trajectory.npz"
    final_state = output / "final_state.npz"
    resolved_path = output / "resolved_case.json"
    result_path = output / "result.json"
    np.savez_compressed(
        trajectory,
        index=np.asarray([step.index for step in trace.steps], dtype=np.int64),
        time_s=np.asarray([step.time_s for step in trace.steps]),
        state=np.asarray([step.state.value for step in trace.steps]),
        palm_position_m=np.stack([step.palm_position_m for step in trace.steps]),
        hand_qpos=np.stack([step.hand_qpos for step in trace.steps]),
        object_position_m=np.stack([step.object_position_m for step in trace.steps]),
        contact_count=np.asarray([step.contact_count for step in trace.steps]),
        hand_object_contact=np.asarray([step.hand_object_contact for step in trace.steps]),
        hand_table_contact=np.asarray([step.hand_table_contact for step in trace.steps]),
        object_table_contact=np.asarray([step.object_table_contact for step in trace.steps]),
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
    np.savez_compressed(final_state, qpos=runner.data.qpos, qvel=runner.data.qvel)
    resolved_payload = {
        "case": resolved.case.model_dump(mode="json"),
        "environment": resolved.environment.model_dump(mode="json"),
        "environment_config_hash": resolved.environment_config_hash,
        "execution_protocol": resolved.execution_protocol.model_dump(mode="json"),
        "execution_protocol_config_hash": resolved.execution_protocol_config_hash,
        "proposal": proposal.model_dump(mode="json"),
        "proposal_hash": proposal_hash,
    }
    _write_json(resolved_path, resolved_payload)
    protocol = resolved.execution_protocol.parameters.model_dump(mode="json")
    result = ProposalExecutionResult(
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
            execution_protocol_parameters=protocol,
            mujoco_version=mujoco.__version__,
            code_commit=_code_commit(),
        ),
        proposal_id=proposal.proposal_id,
        provider_id=proposal.provider.id,
        provider_config_hash=proposal.provider.config_hash,
        provider_metadata=ResolvedProviderMetadata(
            kind=provider_spec.kind,
            translation_strategy_id=provider_spec.translation_strategy_id,
            source_revision=provider_spec.provenance.source_revision,
            license_statuses={
                name: getattr(provider_spec.provenance, name).status
                for name in ("code", "checkpoint", "dataset", "object_assets")
            },
        ),
        proposal_hash=proposal_hash,
        generation_seed=proposal.generation.seed,
        proposal_sample_index=proposal.generation.sample_index,
        geometry_group_id=proposal.input_binding.geometry_group_id,
        observation_group_id=proposal.input_binding.observation_group_id,
        generation_request_group_id=proposal.input_binding.generation_request_group_id,
        proposal_validation=proposal_validation,
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
                (step.maximum_contact_normal_force_n for step in trace.steps), default=0.0
            ),
            peak_actuator_force_fraction=max(
                (step.maximum_actuator_force_fraction for step in trace.steps), default=0.0
            ),
        ),
        artifacts=ProposalArtifactReferences(
            trajectory=Path("trajectory.npz"),
            resolved_case=Path("resolved_case.json"),
            proposal=Path("proposal.json"),
            simulator_state=Path("final_state.npz"),
            video=video,
        ),
    )
    _write_json(result_path, result.model_dump(mode="json"))
    return result


def run_proposal_case(
    case_path: str | Path,
    output_dir: str | Path,
    *,
    viewer: bool = False,
    record_video: bool = True,
    verify_native_artifacts: bool = False,
    project_root: str | Path | None = None,
) -> ProposalExecutionResult:
    project = Path(project_root or ".").resolve()
    output = Path(output_dir)
    _ensure_empty_output(output)
    case = load_execution_case(case_path)
    resolved = resolve_case(case)
    _validate_case_configuration(case)
    geometry = _resolve_object_geometry(case, project)
    proposal_path = (project / case.grasp.proposal_path).resolve()
    if not proposal_path.is_relative_to(project):
        raise ValueError("proposal path escapes the project root")
    proposal = load_proposal(
        proposal_path,
        expected_hash=case.grasp.expected_proposal_hash,
        verify_native_artifacts=verify_native_artifacts,
    )
    if proposal.proposal_id != case.grasp.id:
        raise ValueError("proposal ID does not match the evaluation case")
    providers = load_provider_registry()
    spec = providers.resolve(proposal.provider.id, proposal.provider.config_hash)
    if proposal.object.id != case.object.id:
        raise ValueError("proposal object ID does not match the evaluation case")
    if proposal.object.geometry_content_hash != _geometry_content_hash(geometry):
        raise ValueError("proposal geometry hash does not match the evaluation case")
    if proposal.embodiment_id != case.embodiment.id or spec.embodiment_id != case.embodiment.id:
        raise ValueError("proposal/provider embodiment does not match the evaluation case")
    if case.execution_protocol.id not in proposal.compatible_execution_protocols:
        raise ValueError("proposal is incompatible with the selected execution protocol")
    if case.execution_protocol.id not in spec.compatible_execution_protocols:
        raise ValueError("provider is incompatible with the selected execution protocol")
    if proposal.approach_semantics != spec.approach_semantics:
        raise ValueError("proposal approach semantics do not match its provider registry entry")
    try:
        pose, validation = _validate_pose(proposal)
        _scene_preflight(proposal)
    except ProposalRejected as exc:
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(proposal_path, output / "proposal.json")
        artifact = ProposalRejectionArtifact(
            case_id=case.case_id,
            proposal_id=proposal.proposal_id,
            provider_id=proposal.provider.id,
            proposal_hash=proposal_content_hash(proposal),
            rejection=exc.rejection,
        )
        _write_json(output / "rejection.json", artifact.model_dump(mode="json"))
        raise
    runner_type = _runner_class(resolved)
    runner = runner_type(
        resolved.execution_protocol,
        default_dex_urdf_root(),
        physics_timestep_s=resolved.environment.simulator.physics_timestep_s,
        control_timestep_s=resolved.environment.simulator.control_timestep_s,
        object_geometry=geometry,
    )
    initial = case.initial_condition
    object_xyzw = np.asarray(initial.object_orientation_xyzw, dtype=np.float64)
    object_xyzw /= np.linalg.norm(object_xyzw)
    runner.reset(
        seed=case.seed,
        object_mass_kg=case.object.mass_kg,
        object_position_m=np.asarray(initial.object_position_m),
        object_orientation_wxyz=object_xyzw[[3, 0, 1, 2]],
    )
    palm_position, palm_quaternion = _world_palm(case, proposal)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(proposal_path, output / "proposal.json")
    trace = _execute(
        runner,
        pose,
        palm_position,
        palm_quaternion,
        np.array([0.0, 0.0, -1.0]),
        viewer,
        output / "rollout.mp4" if record_video else None,
    )
    outcome = score_trace(
        trace,
        resolved.execution_protocol,
        control_timestep_s=runner.control_timestep_s,
    )
    result = _write_artifacts(
        output,
        resolved=resolved,
        proposal=proposal,
        proposal_hash=proposal_content_hash(proposal),
        proposal_validation=validation,
        trace=trace,
        outcome=outcome,
        runner=runner,
        video=Path("rollout.mp4") if record_video else None,
        provider_spec=spec,
    )
    print(json.dumps({
        "case_id": result.case_id,
        "provider_id": result.provider_id,
        "success": result.success,
        "failure_type": result.failure_type,
        "output": str(output),
    }, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-video", dest="record_video", action="store_false")
    parser.add_argument("--verify-native-artifacts", action="store_true")
    parser.set_defaults(record_video=True)
    args = parser.parse_args()
    run_proposal_case(
        args.case,
        args.output,
        viewer=args.viewer,
        record_video=args.record_video,
        verify_native_artifacts=args.verify_native_artifacts,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()
