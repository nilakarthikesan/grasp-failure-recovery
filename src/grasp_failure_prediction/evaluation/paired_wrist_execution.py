"""Execute a frozen, explicit wrist posture without replanning its treatment.

This opt-in v4 protocol retains the v3 runtime guard and v2 task dynamics. A
candidate that no longer reproduces its static certificate is rejected before
the first physics step; the planner must never substitute a different wrist.
It neither evaluates the predictor nor changes a historical protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from .artifacts import write_evaluation_artifacts
from .noisy_rollouts import (
    NoisyShadowRunner, _prepare_case, _relative_path, _runtime_versions, _sha256,
    _source_hashes, _validate_execution, _write_json,
)
from .registry import ExecutionProtocolSpec, ResolvedCase
from .retargeting import default_dex_urdf_root, reorder_for_mujoco
from .rollout_events import annotate_rollout_events
from .schema import RegistryReference, canonical_sha256
from .scoring import score_trace
from .table_clearance import (
    TableClearanceConfiguration, _descendant, _rotation, plan_table_clearance,
)
from .table_safe_execution import (
    ClearanceRejected, MotionClearanceRejected, TableSafeNoisyShadowRunner,
    TableSafeProtocolSpec,
)


MANIFEST_SCHEMA = "explicit_wrist_pair_pose_manifest_v1"
GATES = dict(
    nominal_table_clearance_m=0.010,
    nominal_proximal_object_clearance_m=0.001,
    runtime_table_margin_m=0.001,
    runtime_check_interval_s=0.002,
    check_closing_targets=False,
    invariance_absolute_tolerance=1e-10,
    pose_reproduction_absolute_tolerance=1e-12,
)


class PairedWristProtocolSpec(TableSafeProtocolSpec):
    execution_protocol_id: Literal["fixed_grasp_lift_v4_frozen_wrist_pair"]
    controller: Literal["explicit_palm_preserving_wrist_v1"]
    frozen_pose_manifest_sha256: str
    pair_id: str
    posture: Literal["A", "B"]
    explicit_wrist_target_rad: tuple[float, float]


class FrozenWristRejected(ClearanceRejected):
    """The declared treatment cannot reproduce its frozen geometry certificate."""


def _digest(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("SHA256 binding must be a string")
    value = value.removeprefix("sha256:")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid SHA256 binding")
    return value


def _assert_close(actual, expected, tolerance: float, field: str) -> None:
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (actual.shape != expected.shape or not np.all(np.isfinite(actual))
            or not np.all(np.isfinite(expected))
            or not np.allclose(actual, expected, rtol=0, atol=tolerance)):
        raise FrozenWristRejected(f"frozen {field} differs from the prepared geometry")


def load_frozen_manifest(manifest_path, expected_manifest_sha256, *,
                         repository_root, project_root):
    """Validate the frozen manifest and every declared file before model setup."""
    path = Path(manifest_path).resolve()
    if _digest(_sha256(path)) != _digest(expected_manifest_sha256):
        raise ValueError("frozen pose manifest SHA256 differs")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported explicit wrist manifest schema")
    if manifest.get("gates") != GATES:
        raise ValueError("frozen wrist manifest must retain the declared geometry/runtime gates")
    bindings = manifest.get("source_sha256")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("frozen wrist manifest needs source SHA256 bindings")
    roots = {"repository": Path(repository_root).resolve(),
             "runtime_assets": Path(project_root).resolve()}
    for key, digest in bindings.items():
        prefix, separator, relative = key.partition(":")
        if not separator or prefix not in roots:
            raise ValueError(f"unsupported source binding: {key}")
        source = roots[prefix] / _relative_path(relative)
        if _digest(_sha256(source)) != _digest(digest):
            raise ValueError(f"frozen source SHA256 differs: {key}")
    contract = manifest.get("model_contract", {})
    expected_fields = ["hand_qpos", "hand_qvel", "commanded_target_rad"]
    if (contract.get("weights_normalization_features_unchanged") is not True
            or contract.get("failure_alert_threshold") != 0.5
            or contract.get("prediction_time_s") != 2.96
            or contract.get("history_steps") != 20
            or contract.get("fields") != expected_fields
            or contract.get("outcomes_not_inputs") is not True):
        raise ValueError("frozen model input/threshold/cutoff contract changed")
    model = roots["repository"] / _relative_path(contract["model_file"])
    if _digest(_sha256(model)) != _digest(contract["model_sha256"]):
        raise ValueError("frozen model SHA256 differs")
    cases = manifest.get("cases", [])
    if (not isinstance(cases, list) or not cases
            or len({case["pair_id"] for case in cases}) != len(cases)):
        raise ValueError("frozen manifest needs uniquely identified cases")
    return manifest


def paired_wrist_protocol(base: ExecutionProtocolSpec, *, manifest_sha256,
                          pair_id, posture, wrist_rad):
    if base.execution_protocol_id != "fixed_grasp_lift_v2":
        raise ValueError("explicit wrist execution requires fixed_grasp_lift_v2 input")
    wrist = np.asarray(wrist_rad, dtype=float)
    if wrist.shape != (2,) or not np.all(np.isfinite(wrist)):
        raise ValueError("explicit wrist target must be a finite two-vector")
    return PairedWristProtocolSpec.model_validate(dict(
        base.model_dump(mode="json"),
        execution_protocol_id="fixed_grasp_lift_v4_frozen_wrist_pair",
        controller="explicit_palm_preserving_wrist_v1",
        base_protocol_id=base.execution_protocol_id,
        base_protocol_config_hash=canonical_sha256(base),
        clearance_configuration=asdict(TableClearanceConfiguration()),
        runtime_table_margin_m=GATES["runtime_table_margin_m"],
        frozen_pose_manifest_sha256=_digest(manifest_sha256),
        pair_id=pair_id, posture=posture, explicit_wrist_target_rad=tuple(wrist),
    ))


class FrozenWristShadowRunner(TableSafeNoisyShadowRunner):
    """Use the declared angles exactly, then inherit the unchanged guarded stepper."""

    def __init__(self, protocol, *args, frozen_case, **kwargs):
        if not isinstance(protocol, PairedWristProtocolSpec):
            raise TypeError("use paired_wrist_protocol to version the explicit treatment")
        if kwargs.get("noise_amplitude_rad", 0.0) != 0.0:
            raise ValueError("frozen wrist comparison requires zero finger and wrist noise")
        self.frozen_case = frozen_case
        self.frozen_pose = frozen_case["poses"][protocol.posture]
        self.frozen_pose_certificate = None
        super().__init__(protocol, *args, **kwargs)
        if self.physics_timestep_s != GATES["runtime_check_interval_s"]:
            raise ValueError("frozen wrist guard requires the original 2 ms physics interval")
        if self.control_timestep_s != 0.04:
            raise ValueError("frozen wrist comparison requires the original 40 ms control interval")

    def _geometry(self, pose, position, quaternion):
        """Independent FK check of the compensated palm/finger target geometry."""
        model, scratch = self.model, mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self._hand_qpos_addresses] = reorder_for_mujoco(
            pose.joint_names, pose.qpos, self._hand_joint_names)
        root = self._root_qpos_address
        scratch.qpos[root:root + 7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_forward(model, scratch)
        rotation = _rotation(quaternion) @ scratch.xmat[self._palm_body_id].reshape(3, 3).T
        root_position = np.asarray(position) - rotation @ scratch.xpos[self._palm_body_id]
        root_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(root_quaternion, rotation.reshape(9))
        scratch.qpos[root:root + 7] = np.r_[root_position, root_quaternion]
        mujoco.mj_forward(model, scratch)
        fingers = [g for g in range(model.ngeom)
                   if int(model.geom_bodyid[g]) != self._palm_body_id
                   and _descendant(model, int(model.geom_bodyid[g]), self._palm_body_id)]
        return dict(
            root_position_m=root_position, root_quaternion_wxyz=root_quaternion,
            palm_position_m=scratch.xpos[self._palm_body_id].copy(),
            palm_rotation=scratch.xmat[self._palm_body_id].copy(),
            finger_position_m=scratch.geom_xpos[fingers].copy(),
            finger_rotation=scratch.geom_xmat[fingers].copy(),
        )

    def execute(self, pose, *, grasp_palm_position_m, grasp_palm_quaternion_wxyz,
                **kwargs):
        direction = np.asarray(kwargs.get("approach_direction_world", [0, 0, -1]), dtype=float)
        norm = float(np.linalg.norm(direction))
        if (direction.shape != (3,) or not np.all(np.isfinite(direction)) or norm <= 0
                or not np.allclose(direction / norm, [0, 0, -1], atol=1e-12, rtol=0)):
            raise FrozenWristRejected("frozen wrist execution requires vertical descent")
        case, expected = self.frozen_case, self.frozen_pose
        tolerance = GATES["pose_reproduction_absolute_tolerance"]
        _assert_close(grasp_palm_position_m, case["intended_world_palm_position_m"],
                      tolerance, "world palm position")
        _assert_close(grasp_palm_quaternion_wxyz, case["intended_world_palm_quaternion_wxyz"],
                      tolerance, "world palm quaternion")
        finger_names = tuple(name for name in pose.joint_names if name not in {"WRJ2", "WRJ1"})
        if (len(finger_names) != 22 or len(set(finger_names)) != 22
                or set(finger_names) != set(case["intended_finger_joint_names"])):
            raise FrozenWristRejected("frozen 22 finger joint names differ")
        actual_fingers = [pose.qpos[pose.joint_names.index(name)]
                          for name in case["intended_finger_joint_names"]]
        _assert_close(actual_fingers, case["intended_finger_target_rad"], tolerance,
                      "22 finger targets")
        wrist = np.asarray(expected["wrist_rad"], dtype=float)
        _assert_close(wrist, self.protocol.explicit_wrist_target_rad, 0.0, "protocol wrist")
        qpos = pose.qpos.copy()
        for name, value in zip(("WRJ2", "WRJ1"), wrist, strict=True):
            qpos[pose.joint_names.index(name)] = value
        explicit = replace(pose, qpos=qpos)
        try:
            self.clearance_plan = plan_table_clearance(
                self, explicit, grasp_palm_position_m, grasp_palm_quaternion_wxyz,
                configuration=self.clearance_configuration)
        except ValueError as error:
            raise FrozenWristRejected(str(error)) from error
        plan = self.clearance_plan
        if (not plan.feasible or plan.adjusted_wrist_rad != tuple(wrist)
                or plan.evaluated_candidates != 1):
            raise FrozenWristRejected("exact frozen wrist is unsafe; planner substitution is forbidden")
        if not expected["passes_geometry_gate"]:
            raise FrozenWristRejected("frozen candidate was not declared geometrically eligible")
        for field in ("root_position_m", "root_quaternion_wxyz"):
            _assert_close(getattr(plan, field), expected[field], tolerance, field)
        _assert_close(plan.minimum_table_clearance_m, expected["nominal_table_clearance_m"],
                      tolerance, "nominal table clearance")
        _assert_close(plan.minimum_proximal_object_distance_m,
                      expected["nominal_proximal_object_distance_m"], tolerance,
                      "nominal proximal object clearance")
        original = self._geometry(pose, grasp_palm_position_m, grasp_palm_quaternion_wxyz)
        treated = self._geometry(explicit, grasp_palm_position_m, grasp_palm_quaternion_wxyz)
        errors = {}
        for field in ("palm_position_m", "palm_rotation", "finger_position_m", "finger_rotation"):
            _assert_close(treated[field], original[field], GATES["invariance_absolute_tolerance"], field)
            errors[field] = float(np.max(np.abs(treated[field] - original[field])))
        self.frozen_pose_certificate = dict(
            explicit_angles_unchanged=True, automatic_substitution=False,
            nominal_world_geometry_max_absolute_errors=errors,
            scope="nominal intended target geometry; actual dynamics require separate measurement")
        # Bypass TableSafe.execute: re-entering its planner could choose a new
        # treatment. Its guarded _advance remains active through normal dispatch.
        return NoisyShadowRunner.execute(
            self, explicit, grasp_palm_position_m=grasp_palm_position_m,
            grasp_palm_quaternion_wxyz=grasp_palm_quaternion_wxyz, **kwargs)


def _validate_prepared(prepared, row, case_path):
    if _digest(_sha256(case_path)) != _digest(row["case_sha256"]):
        raise ValueError("frozen case SHA256 differs")
    if prepared.source != row["source_binding"]:
        raise ValueError("frozen HUG/observation/object source binding differs")
    resolved = prepared.resolved
    for actual, expected, field in (
        (resolved.environment.model_dump(mode="json"), row["resolved_environment"], "environment"),
        (resolved.environment_config_hash, row["environment_config_hash"], "environment hash"),
        (resolved.execution_protocol.model_dump(mode="json"),
         row["resolved_base_execution_protocol"], "base protocol"),
        (resolved.execution_protocol_config_hash, row["base_execution_protocol_config_hash"],
         "base protocol hash"),
    ):
        if actual != expected:
            raise ValueError(f"frozen {field} differs")


def execute_frozen_wrist_case(case_path, output_dir, *, project_root, manifest_path,
                             expected_manifest_sha256, pair_id, posture,
                             noise_amplitude_rad=0.0, noise_seed=9000,
                             repository_root=None):
    """Run one explicitly frozen zero-noise A/B treatment; never call the model."""
    if noise_amplitude_rad != 0.0:
        raise ValueError("frozen wrist comparison requires zero finger and wrist noise")
    if posture not in {"A", "B"}:
        raise ValueError("posture must be A or B")
    project = Path(project_root).resolve()
    repository = Path(repository_root or Path(__file__).resolve().parents[3]).resolve()
    manifest = load_frozen_manifest(
        manifest_path, expected_manifest_sha256, repository_root=repository, project_root=project)
    matches = [row for row in manifest["cases"] if row["pair_id"] == pair_id]
    if len(matches) != 1 or not matches[0].get("eligible"):
        raise ValueError("requested pair is not geometrically eligible in the frozen manifest")
    row = matches[0]
    relative = _relative_path(row["case_path"])
    expected_path = (project / relative).resolve()
    requested_path = Path(case_path)
    if not requested_path.is_absolute():
        requested_path = project / requested_path
    if requested_path.resolve() != expected_path:
        raise ValueError("case path differs from the requested frozen pair")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be new or empty: {output}")
    prepared = _prepare_case(expected_path, project)
    _validate_prepared(prepared, row, expected_path)
    protocol = paired_wrist_protocol(
        prepared.resolved.execution_protocol, manifest_sha256=expected_manifest_sha256,
        pair_id=pair_id, posture=posture, wrist_rad=row["poses"][posture]["wrist_rad"])
    config_hash = canonical_sha256(protocol)
    case = prepared.resolved.case.model_copy(update={"execution_protocol": RegistryReference(
        id=protocol.execution_protocol_id, expected_config_hash=config_hash)})
    resolved: ResolvedCase = replace(
        prepared.resolved, case=case, execution_protocol=protocol,
        execution_protocol_config_hash=config_hash)
    runner = FrozenWristShadowRunner(
        protocol, default_dex_urdf_root(), frozen_case=row,
        physics_timestep_s=resolved.environment.simulator.physics_timestep_s,
        control_timestep_s=resolved.environment.simulator.control_timestep_s,
        noise_amplitude_rad=0.0, noise_seed=noise_seed, object_geometry=prepared.object_geometry)
    runner.reset(
        seed=case.seed, object_mass_kg=case.object.mass_kg,
        object_position_m=np.asarray(case.initial_condition.object_position_m),
        object_orientation_wxyz=np.asarray(case.initial_condition.object_orientation_xyzw)[[3, 0, 1, 2]])
    source_hashes = _source_hashes()
    source_hashes.update({name: _sha256(Path(__file__).parent / name) for name in (
        "paired_wrist_execution.py", "table_safe_execution.py", "table_clearance.py")})
    metadata = dict(
        prepared.source, schema_version="frozen_wrist_pair_rollout_v1",
        case_path=str(expected_path), original_case_sha256=_sha256(expected_path),
        original_resolved_case=prepared.resolved.case.model_dump(mode="json"),
        resolved_environment=resolved.environment.model_dump(mode="json"),
        environment_config_hash=resolved.environment_config_hash,
        execution_protocol=protocol.model_dump(mode="json"),
        execution_protocol_config_hash=config_hash, source_file_sha256=source_hashes,
        runtime_versions=_runtime_versions(), pair_id=pair_id, posture=posture,
        frozen_pose_manifest_sha256=_digest(expected_manifest_sha256),
        noise_amplitude_rad=0.0, noise_seed=int(noise_seed), noise_mode="disabled",
        intended_world_palm_position_m=prepared.palm_position.tolist(),
        intended_world_palm_quaternion_wxyz=prepared.palm_quaternion.tolist(),
        purpose="frozen-model development wrist diagnostic; not a fresh generalization test",
        label_scope="unchanged final grasp/lift/hold scorer; rejection has no label",
        privileged_replay_only_fields=["full_qpos", "full_qvel", "integration_state"],
        predictor_evaluated_by_executor=False)
    try:
        trace = runner.execute(
            prepared.pose, grasp_palm_position_m=prepared.palm_position,
            grasp_palm_quaternion_wxyz=prepared.palm_quaternion)
    except ClearanceRejected as error:
        output.mkdir(parents=True, exist_ok=True)
        metadata.update(
            status=("runtime_clearance_rejected" if isinstance(error, MotionClearanceRejected)
                    else "frozen_pose_rejected"), label=None, error=str(error),
            clearance_plan=(None if runner.clearance_plan is None
                            else runner.clearance_plan.as_record()),
            frozen_pose_certificate=runner.frozen_pose_certificate)
        if runner.runtime_guard_event is not None:
            metadata["runtime_guard_event"] = runner.runtime_guard_event
            np.savez_compressed(output / "runtime_guard_state.npz", qpos=runner.data.qpos.copy(),
                                qvel=runner.data.qvel.copy(), time_s=float(runner.data.time))
        if runner.action_history:
            runner.write_action_artifacts(output)
            metadata["partial_actions_scope"] = "diagnostics only; rejected attempt remains unlabeled and unscored"
        _write_json(output / "collection_metadata.json", metadata)
        return metadata
    _validate_execution(runner, trace)
    outcome = score_trace(trace, protocol, control_timestep_s=runner.control_timestep_s)
    output.mkdir(parents=True, exist_ok=True)
    runner.write_action_artifacts(output)
    _write_json(output / "events.json", annotate_rollout_events(
        trace, protocol, control_timestep_s=runner.control_timestep_s, outcome=outcome))
    metadata.update(
        status="completed", label=int(outcome.success),
        failure_type=outcome.failure_type.value if outcome.failure_type else None,
        clearance_plan=runner.clearance_plan.as_record(),
        frozen_pose_certificate=runner.frozen_pose_certificate,
        runtime_guard="1 mm table-plane margin checked at every 2 ms physics endpoint",
        sampled_hand_table_contact=any(step.hand_table_contact for step in trace.steps),
        state_replay="scene.mjb plus actions.npz integration_state (mjSTATE_INTEGRATION)")
    _write_json(output / "collection_metadata.json", metadata)
    write_evaluation_artifacts(
        output, resolved=resolved, trace=trace, pose_validation=prepared.validation,
        outcome=outcome, final_qpos=runner.data.qpos.copy(), final_qvel=runner.data.qvel.copy())
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--posture", choices=("A", "B"), required=True)
    args = parser.parse_args()
    record = execute_frozen_wrist_case(
        args.case, args.output, project_root=args.project_root, manifest_path=args.manifest,
        expected_manifest_sha256=args.manifest_sha256, pair_id=args.pair_id, posture=args.posture)
    print(f'{record["status"]}: label={record["label"]}')
    if record["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
