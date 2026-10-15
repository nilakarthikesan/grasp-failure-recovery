"""Opt-in HUG proposal execution transaction for prediction_recording_v1.

This adds no controller, mapper, scorer, collection quota or train/test split.
Execution requires native-surface fidelity independently of numerical mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import shutil
import traceback
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from grasp_failure_prediction.evaluation import scoring
from grasp_failure_prediction.evaluation.case_runner import _source_scene_binding
from grasp_failure_prediction.evaluation.prediction_recording import (
    AttemptMetadata, MappingStatus, RECORDING_SCHEMA, file_sha256, write_incomplete_attempt, write_json,
)
from grasp_failure_prediction.evaluation.recorded_runner import RecordedActuatedShadowRunner
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root, _validate_rigid_transform
from grasp_failure_prediction.evaluation.runner import ActuatedShadowRunner
from grasp_failure_prediction.evaluation.schema import canonical_sha256


HUG_PROVIDER_ID = "hug_mano_shadow_v1"


class RecordingAdmissionError(ValueError):
    def __init__(self, mapping_status: MappingStatus, code: str, message: str):
        self.mapping_status, self.code = mapping_status, code
        super().__init__(message)


def _finite_state(data) -> None:
    if not np.isfinite(data.time) or not all(np.isfinite(getattr(data, name)).all()
            for name in ("qpos", "qvel", "qacc", "ctrl")):
        raise RuntimeError("nonfinite simulator state at recorded control boundary")


class CheckedRecordedShadowRunner(RecordedActuatedShadowRunner):
    """Same controller, with refusal of numerical errors at control boundaries."""

    def _advance(self, *args, **kwargs):
        _finite_state(self.data)
        before = float(self.data.time)
        warnings = np.asarray([self.data.warning[i].number for i in range(int(mujoco.mjtWarning.mjNWARNING))]).copy()
        if warnings.any():
            raise RuntimeError("MuJoCo warning before recorded control interval")
        super()._advance(*args, **kwargs)
        _finite_state(self.data)
        after = np.asarray([self.data.warning[i].number for i in range(len(warnings))])
        if np.any(after != warnings):
            raise RuntimeError("MuJoCo warning during recorded control interval")
        if not np.isclose(float(self.data.time) - before, self.control_timestep_s, rtol=0, atol=1e-12):
            raise RuntimeError("unexpected recorded control interval or simulator reset")


def _code_files() -> dict[Path, str]:
    package = Path(__file__).resolve().parents[1]
    names = (
        "proposals/execution.py", "proposals/prediction_recording.py", "proposals/artifacts.py",
        "proposals/schema.py", "proposals/registry.py", "proposals/preflight.py",
        "proposals/configs/preflight.yaml", "proposals/configs/providers.yaml",
        "evaluation/runner.py", "evaluation/recorded_runner.py", "evaluation/prediction_recording.py",
        "evaluation/scoring.py", "evaluation/retargeting.py", "evaluation/pose_validation.py",
        "evaluation/mapping_fidelity.py",
        "evaluation/object_assets.py", "evaluation/case_runner.py", "evaluation/scene_contracts.py",
        "evaluation/registry.py", "evaluation/schema.py", "integrations/hug.py",
        "evaluation/configs/environments.yaml", "evaluation/configs/execution_protocols.yaml",
    )
    return {package / name: file_sha256(package / name) for name in names}


def _hand_files() -> dict[Path, str]:
    urdf = Path(default_dex_urdf_root()) / "shadow_hand/shadow_hand_right.urdf"
    tree = ET.fromstring(urdf.read_bytes())
    paths = {urdf} | {(urdf.parent / node.attrib["filename"]).resolve() for node in tree.findall(".//mesh")}
    return {path: file_sha256(path) for path in sorted(paths)}


class HugPredictionRecording:
    """Persist a denominator-preserving attempt even if setup fails early."""

    def __init__(self, case_path: str | Path, output: Path, project: Path) -> None:
        self.output, self.project = output, project
        self.stage_output = output / "_pending_execution"
        self.requested_case_path = Path(case_path).resolve()
        self.input_files: dict[Path, str] = {}
        self.source_files = _code_files()
        self.asset_files: dict[Path, str] = {}
        self.case = self.resolved = self.geometry = self.proposal = self.proposal_path = self.provider_spec = None
        self.runner = self.trace = self.outcome = None
        self.execution_started = False
        self.native_verified = False
        self.native_sha256 = None
        self.scene_binding = None
        self.model_sha256 = None
        if self.requested_case_path.is_file():
            self.remember(self.requested_case_path)

    def remember(self, path: Path) -> None:
        path = path.resolve()
        digest = file_sha256(path)
        if path in self.input_files and self.input_files[path] != digest:
            raise ValueError(f"input changed during recorded execution: {path.name}")
        self.input_files[path] = digest

    def bind(self, **values) -> None:
        for key, value in values.items():
            if key not in {"case", "resolved", "geometry", "proposal", "proposal_path", "provider_spec", "runner", "trace", "outcome"}:
                raise ValueError("unknown prediction-recording context field")
            setattr(self, key, value)
        if "proposal_path" in values:
            self.remember(values["proposal_path"])

    def validate_inputs(self) -> None:
        proposal, resolved = self.proposal, self.resolved
        if proposal.provider.id != HUG_PROVIDER_ID:
            raise ValueError("hug_prediction_recording_v1 requires the HUG proposal provider")
        if proposal.input_binding.observation_group_id is None:
            raise ValueError("recorded HUG execution requires a captured observation identity")
        expected_geometry = canonical_sha256({"geometry_content_hash": proposal.object.geometry_content_hash})
        expected_observation = canonical_sha256({
            "input_manifest_hash": proposal.input_binding.input_manifest_hash,
            "scene_contract_hash": proposal.input_binding.scene_contract_hash,
            "object_geometry": proposal.object.geometry_content_hash})
        if (proposal.input_binding.geometry_group_id != expected_geometry
                or proposal.input_binding.observation_group_id != expected_observation):
            raise ValueError("HUG group identities do not match captured input content")
        if proposal.provider_validation.status != "passed":
            raise RecordingAdmissionError(MappingStatus.REFUSED_GUARD, "provider_validation_failed",
                                          "recorded HUG execution requires passed provider validation")
        mean_error = proposal.provider_validation.metrics.get("mean_fingertip_error_m")
        guard = resolved.execution_protocol.parameters.maximum_mean_fingertip_error_m
        if (isinstance(mean_error, bool) or not isinstance(mean_error, (int, float))
                or not np.isfinite(mean_error) or mean_error < 0 or mean_error > guard):
            raise RecordingAdmissionError(MappingStatus.REFUSED_GUARD, "fingertip_engineering_guard",
                                          "HUG provider did not pass the unchanged fingertip engineering guard")
        outputs = [item for item in proposal.native_artifacts if item.role == "provider_output"]
        reports = [item for item in proposal.native_artifacts if item.role == "provider_report"]
        if len(outputs) != 1 or len(reports) != 1:
            raise ValueError("recorded HUG execution requires exactly one native output and inference report")
        native = (self.proposal_path.parent / outputs[0].path).resolve()
        report = (self.proposal_path.parent / reports[0].path).resolve()
        if report != native.parent / "inference_report.json":
            raise ValueError("HUG inference report must accompany its copied native capture")
        # load_proposal(deep=True) already enforces bundle containment/hashes;
        # repeat at this boundary to reject a mutation between the two reads.
        for reference in proposal.native_artifacts:
            target = (self.proposal_path.parent / reference.path).resolve()
            if not target.is_relative_to(self.proposal_path.parent.resolve()) or file_sha256(target) != reference.sha256:
                raise ValueError("native artifact containment/hash mismatch")
            self.remember(target)
        inference = json.loads(report.read_text(encoding="utf-8"))
        if proposal.provider.checkpoint_hash is None or (
            "sha256:" + str(inference.get("checkpoint_sha256")) != proposal.provider.checkpoint_hash
        ):
            raise ValueError("native inference checkpoint differs from the proposal provenance")
        if inference.get("hug_code_commit") != proposal.provider.code_revision:
            raise ValueError("native inference code revision differs from proposal provenance")
        _validate_rigid_transform(np.load(native.parent / "T_world_camera.npy", allow_pickle=False))
        observation = native.parent.relative_to(self.project)
        # Reuse the captured-scene validator. This view changes only the input
        # field names from eval_case_v2 to the already implemented HUG binding API.
        scene_case = SimpleNamespace(
            object=self.case.object, initial_condition=self.case.initial_condition,
            grasp=SimpleNamespace(observation_path=observation, inference_seed=proposal.generation.seed),
        )
        binding = _source_scene_binding(scene_case, resolved, self.project, native, self.geometry)
        if (binding["scene_contract_status"] != "verified"
                or binding["scene_contract_sha256"] != proposal.input_binding.scene_contract_hash
                or binding["observation_manifest_sha256"] != proposal.input_binding.input_manifest_hash):
            raise ValueError("proposal input binding differs from its verified native capture")
        for name in (*binding["observation_files_sha256"], "scene_contract.json", "observation_manifest.json"):
            self.remember(native.parent / name)
        self.asset_files = _hand_files()
        if self.geometry is not None:
            for name, expected in self.geometry.files_sha256.items():
                path = self.geometry.mjcf_path.parent / name
                if file_sha256(path) != expected:
                    raise ValueError("object geometry changed since resolution")
                self.asset_files[path.resolve()] = expected
        self.native_sha256 = outputs[0].sha256
        self.scene_binding, self.native_verified = binding, True

    def runner_class(self, registered):
        if registered is not ActuatedShadowRunner:
            raise ValueError("recording mode requires the unchanged registered ActuatedShadowRunner")
        self.require_fidelity_for_execution()
        return CheckedRecordedShadowRunner

    def require_fidelity_for_execution(self) -> None:
        """Refuse before runner creation; static input inspection stays usable."""
        from grasp_failure_prediction.evaluation.mapping_fidelity import MappingFidelityRefused, require_fidelity_pass
        references = [] if self.proposal is None else [item for item in self.proposal.native_artifacts
                                                       if item.role == "mapping_fidelity_evidence"]
        reference = None
        if len(references) == 1 and self.proposal_path is not None:
            item = references[0]
            reference = {"path": str(self.proposal_path.parent / item.path), "sha256": item.sha256}
        try:
            require_fidelity_pass(reference, mapped_path=self.proposal_path)
        except MappingFidelityRefused as error:
            raise RecordingAdmissionError(MappingStatus.REFUSED_GUARD, "native_surface_fidelity_unqualified",
                                          str(error)) from error

    def verify_bindings(self) -> None:
        for path, expected in {**self.source_files, **self.input_files, **self.asset_files}.items():
            if file_sha256(path) != expected:
                raise ValueError(f"recorded execution binding changed: {path.name}")

    def metadata(self, *, mapping="ok", execution_status, outcome=None, code=None, message=None):
        unresolved = canonical_sha256({"case_request": str(self.requested_case_path), "files": {
            str(path): value for path, value in self.input_files.items()}})
        case, proposal, resolved = self.case, self.proposal, self.resolved
        configuration = {"recording_schema": RECORDING_SCHEMA}
        if resolved is not None:
            configuration.update(environment=resolved.environment.model_dump(mode="json"),
                                 execution_protocol=resolved.execution_protocol.model_dump(mode="json"),
                                 case=resolved.case.model_dump(mode="json"))
        if self.provider_spec is not None:
            configuration["provider"] = self.provider_spec.model_dump(mode="json")
        hashes = {
            "code": canonical_sha256({str(path): value for path, value in self.source_files.items()}),
            "configuration": canonical_sha256(configuration),
            "inputs": canonical_sha256({"case_requested": str(self.requested_case_path), "files": {
                str(path): value for path, value in self.input_files.items()}}),
        }
        if proposal is not None:
            hashes["mapped_pose"] = canonical_sha256({
                "joint_targets_rad": proposal.joint_targets_rad,
                "object_to_palm": proposal.object_to_palm.model_dump(mode="json"),
                "pose_frame": proposal.pose_frame, "palm_link": proposal.palm_link})
        if self.native_sha256 is not None:
            hashes["native_pose"] = self.native_sha256
        if self.model_sha256 is not None:
            hashes["model"] = self.model_sha256
        if self.asset_files:
            hashes["assets"] = canonical_sha256({str(path): value for path, value in self.asset_files.items()})
        if resolved is not None:
            hashes["label_rule"] = canonical_sha256({"scorer": file_sha256(Path(scoring.__file__)),
                "success": resolved.execution_protocol.success.model_dump(mode="json"),
                "control_timestep_s": resolved.environment.simulator.control_timestep_s})
        return AttemptMetadata(
            attempt_id=case.case_id if case is not None else "unresolved_" + unresolved[-16:],
            hand_id=case.embodiment.id if case is not None else "unresolved",
            execution_protocol_id=case.execution_protocol.id if case is not None else "unresolved",
            object_id=case.object.id if case is not None else "unresolved",
            geometry_group_id=proposal.input_binding.geometry_group_id if proposal is not None else "unresolved_" + unresolved,
            observation_group_id=(proposal.input_binding.observation_group_id if proposal is not None else None) or "unresolved_" + unresolved,
            proposal_group_id=(canonical_sha256({"native_pose_sha256": self.native_sha256,
                "observation_group_id": proposal.input_binding.observation_group_id,
                "geometry_group_id": proposal.input_binding.geometry_group_id}) if self.native_verified else "unresolved_" + unresolved),
            mapping={"status": mapping, "code": code, "message": message},
            execution_status=execution_status, outcome=outcome, provenance_hashes=hashes,
        )

    def _receipt(self, metadata, *, error=None) -> dict:
        return {
            "schema_version": "hug_proposal_recording_receipt_v1", "recording_schema": RECORDING_SCHEMA,
            "attempt_id": metadata.attempt_id, "attempt_record": "prediction_recording/attempt.json",
            "mapping_status": metadata.mapping.status.value, "execution_status": metadata.execution_status,
            "outcome": None if metadata.outcome is None else metadata.outcome.model_dump(mode="json"),
            "source_generator_verification": "verified_hug" if self.native_verified else "unresolved",
            "input_files_sha256": {str(path): value for path, value in self.input_files.items()},
            "source_files_sha256": {str(path): value for path, value in self.source_files.items()},
            "asset_files_sha256": {str(path): value for path, value in self.asset_files.items()},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__},
            "numerical_guard": "control-boundary finite qpos/qvel/qacc/ctrl, warning counters, timestep within1e-12s",
            "error": error,
        }

    def complete(self, trace, outcome) -> None:
        if not self.native_verified or self.runner is None:
            raise ValueError("recorded completion requires verified native provenance and a runner")
        self.verify_bindings()
        mujoco.mj_saveModel(self.runner.model, str(self.stage_output / "scene.mjb"))
        self.model_sha256 = file_sha256(self.stage_output / "scene.mjb")
        failure = None if outcome.failure_type is None else outcome.failure_type.value
        first_failure = None
        if failure == "approach_collision":
            first_failure = next((step.time_s - self.runner.physics_timestep_s for step in trace.steps
                if step.hand_table_contact and step.state.value in {"move_to_pregrasp", "approach"}), None)
        metadata = self.metadata(execution_status="completed", outcome={
            "success": bool(outcome.success), "failure_type": failure,
            "label_rule_id": self.resolved.execution_protocol.execution_protocol_id + "_task_outcome_v1",
            "first_failure_time_s": first_failure,
        })
        self.runner.prediction_recorder.write(self.stage_output / "prediction_recording", metadata)
        write_json(self.stage_output / "recording_manifest.json", self._receipt(metadata))
        self.verify_bindings()
        # Publish the existing completed-result marker last. Default-mode files
        # never use this transaction and preserve their historical layout.
        for path in sorted(self.stage_output.iterdir(), key=lambda p: (p.name == "result.json", p.name)):
            path.rename(self.output / path.name)
        self.stage_output.rmdir()

    def fail(self, error: Exception) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        quarantine = self.output / "_incomplete_execution"
        if self.stage_output.exists():
            self.stage_output.rename(quarantine)
        else:
            quarantine.mkdir()
        for name in ("prediction_recording", "result.json", "recording_manifest.json"):
            path = self.output / name
            if path.exists():
                path.rename(quarantine / ("published_" + name))
        for path in quarantine.rglob("attempt.json"):
            path.rename(path.with_name("quarantined_attempt.json"))
        (quarantine / "exception_traceback.txt").write_text(traceback.format_exc())
        if (quarantine / "rejection.json").exists():
            shutil.copy2(quarantine / "rejection.json", self.output / "rejection.json")
        try:
            if self.runner is not None and self.runner.prediction_recorder.sample_count:
                recorder = self.runner.prediction_recorder
                np.savez_compressed(quarantine / "partial_observables.npz", **recorder.arrays())
                rows = recorder._diagnostics
                if rows and rows[0]:
                    np.savez_compressed(quarantine / "partial_diagnostics.npz",
                                        **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
        except Exception as snapshot_error:
            write_json(quarantine / "partial_recording_error.json", {"type": type(snapshot_error).__name__, "message": str(snapshot_error)})
        rejection = getattr(error, "rejection", None)
        if isinstance(error, RecordingAdmissionError):
            mapping, code = error.mapping_status, error.code
        elif rejection is not None:
            mapping = (MappingStatus.REFUSED_INVALID_RETARGET if rejection.stage in {"structural", "kinematic", "translation"}
                       else MappingStatus.REFUSED_GUARD)
            code = rejection.code
        else:
            mapping = MappingStatus.OK if self.execution_started else MappingStatus.SETUP_ERROR
            code = type(error).__name__
        metadata = self.metadata(mapping=mapping, execution_status="runtime_error" if mapping is MappingStatus.OK else "not_executed",
                                 code=code if mapping is not MappingStatus.OK else None,
                                 message=str(error) if mapping is not MappingStatus.OK else None)
        write_incomplete_attempt(self.output / "prediction_recording", metadata)
        receipt = self._receipt(metadata, error={"type": type(error).__name__, "message": str(error)})
        receipt["quarantined_artifacts"] = "_incomplete_execution"
        write_json(self.output / "recording_manifest.json", receipt)
