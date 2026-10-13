"""Built-in local providers for canonical Shadow Hand proposals."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Protocol

import mujoco
import numpy as np

from grasp_failure_prediction.evaluation.case_runner import _hug_world_palm_pose
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
from grasp_failure_prediction.evaluation.retargeting import (
    ShadowHandRetargeter,
    default_dex_urdf_root,
)
from grasp_failure_prediction.evaluation.schema import canonical_sha256
from grasp_failure_prediction.integrations.hug import load_hug_prediction

from .artifacts import file_sha256, write_proposal
from .registry import ProviderSpec
from .schema import (
    GenerationRecord,
    GraspProposal,
    InputBinding,
    NativeArtifactReference,
    ProposalObjectBinding,
    ProposalRequest,
    ProviderProvenance,
    ProviderValidation,
    RigidTransform,
)


class ProposalProvider(Protocol):
    def generate(self, request: ProposalRequest, output_directory: Path) -> list[Path]: ...


def _code_revision(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40


def _project_input(project_root: Path, relative: Path, role: str) -> Path:
    path = (project_root / relative).resolve()
    if not path.is_relative_to(project_root.resolve()):
        raise ValueError(f"{role} path escapes the project root")
    return path


def _matrix_from_pose(position_m, orientation_xyzw) -> np.ndarray:
    xyzw = np.asarray(orientation_xyzw, dtype=np.float64)
    wxyz = xyzw[[3, 0, 1, 2]]
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, wxyz)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.reshape(3, 3)
    transform[:3, 3] = np.asarray(position_m, dtype=np.float64)
    return transform


def _rigid_transform(transform: np.ndarray) -> RigidTransform:
    quaternion_wxyz = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion_wxyz, transform[:3, :3].reshape(9))
    quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
    if quaternion_xyzw[3] < 0:
        quaternion_xyzw *= -1
    return RigidTransform(
        translation_m=tuple(float(value) for value in transform[:3, 3]),
        quaternion_xyzw=tuple(float(value) for value in quaternion_xyzw),
    )


def _binding(
    request: ProposalRequest,
    *,
    input_manifest_hash: str | None,
    scene_contract_hash: str | None,
) -> InputBinding:
    observation_identity = {
        "input_manifest_hash": input_manifest_hash,
        "scene_contract_hash": scene_contract_hash,
        "object_geometry": request.object.geometry_content_hash,
    }
    return InputBinding(
        geometry_group_id=canonical_sha256(
            {"geometry_content_hash": request.object.geometry_content_hash}
        ),
        observation_group_id=(
            canonical_sha256(observation_identity)
            if input_manifest_hash is not None or scene_contract_hash is not None
            else None
        ),
        generation_request_group_id=canonical_sha256(request),
        input_manifest_hash=input_manifest_hash,
        scene_contract_hash=scene_contract_hash,
    )


class HugProposalProvider:
    """Run official HUG, retarget MANO, and emit execution-ready proposals."""

    def __init__(self, spec: ProviderSpec, *, project_root: Path, hug_root: Path, checkpoint: Path):
        self.spec = spec
        self.project_root = project_root
        self.hug_root = hug_root.resolve()
        self.checkpoint = checkpoint.resolve()
        if not self.hug_root.is_dir():
            raise FileNotFoundError(f"HUG checkout does not exist: {self.hug_root}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"HUG checkpoint does not exist: {self.checkpoint}")

    def generate(self, request: ProposalRequest, output_directory: Path) -> list[Path]:
        if request.observation_path is None:
            raise ValueError("HUG provider requires observation_path")
        observation = _project_input(
            self.project_root, request.observation_path, "observation"
        )
        if not observation.is_dir():
            raise FileNotFoundError(f"observation directory does not exist: {observation}")
        contract_path = observation / "scene_contract.json"
        manifest_path = observation / "observation_manifest.json"
        camera_path = observation / "T_world_camera.npy"
        for required in (contract_path, manifest_path, camera_path):
            if not required.is_file():
                raise FileNotFoundError(f"HUG observation is missing {required.name}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract.get("object_id") != request.object.id:
            raise ValueError("scene contract object ID does not match proposal request")
        if contract.get("geometry_content_hash") != request.object.geometry_content_hash:
            raise ValueError("scene contract geometry hash does not match proposal request")

        registry_hash = canonical_sha256(self.spec)
        outputs: list[Path] = []
        for sample_index in range(request.sample_count):
            sample_seed = request.seed + sample_index
            if sample_seed > 2**32 - 1:
                raise ValueError("sample seed exceeds unsigned 32-bit range")
            sample_dir = output_directory / f"sample_{sample_index:04d}"
            native = sample_dir / "native"
            if sample_dir.exists():
                raise FileExistsError(f"proposal sample directory already exists: {sample_dir}")
            shutil.copytree(observation, native)
            for stale in (native / "proposal.pkl", native / "inference_report.json"):
                stale.unlink(missing_ok=True)
            command = [
                sys.executable,
                str(self.project_root / "scripts" / "infer_sim_observation.py"),
                "--hug-root", str(self.hug_root),
                "--observation", str(native),
                "--checkpoint", str(self.checkpoint),
                "--seed", str(sample_seed),
            ]
            subprocess.run(command, cwd=self.project_root, check=True)
            native_proposal = native / "proposal.pkl"
            report_path = native / "inference_report.json"
            if not native_proposal.is_file() or not report_path.is_file():
                raise RuntimeError("HUG provider did not produce its declared native artifacts")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            grasp = load_hug_prediction(native_proposal)
            retargeter = ShadowHandRetargeter(default_dex_urdf_root())
            pose = retargeter.retarget(grasp)
            validator = ShadowPoseValidator(default_dex_urdf_root())
            validation = validator.validate(pose)
            maximum = self.spec.maximum_mean_fingertip_error_m
            if maximum is None or validation.mean_fingertip_error_m > maximum:
                raise ValueError(
                    "HUG retargeting alignment gate failed: "
                    f"{validation.mean_fingertip_error_m:.6f} m"
                )
            palm_position, palm_quaternion_wxyz = _hug_world_palm_pose(
                grasp, pose, validator, native
            )
            world_palm = np.eye(4, dtype=np.float64)
            palm_rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(palm_rotation, palm_quaternion_wxyz)
            world_palm[:3, :3] = palm_rotation.reshape(3, 3)
            world_palm[:3, 3] = palm_position
            world_object = _matrix_from_pose(
                contract["object_position_m"], contract["object_orientation_xyzw"]
            )
            object_palm = np.linalg.inv(world_object) @ world_palm
            proposal = GraspProposal(
                proposal_id=f"{request.request_id}_sample_{sample_index:04d}",
                provider=ProviderProvenance(
                    id=self.spec.provider_id,
                    config_hash=registry_hash,
                    code_revision=str(report.get("hug_code_commit") or "unknown"),
                    adapter_code_revision=_code_revision(self.project_root),
                    checkpoint_hash="sha256:" + str(report["checkpoint_sha256"]),
                ),
                generation=GenerationRecord(seed=sample_seed, sample_index=sample_index),
                object=ProposalObjectBinding(
                    id=request.object.id,
                    geometry_content_hash=request.object.geometry_content_hash,
                ),
                object_to_palm=_rigid_transform(object_palm),
                joint_targets_rad={
                    name: float(value)
                    for name, value in zip(pose.joint_names, pose.qpos, strict=True)
                },
                compatible_execution_protocols=self.spec.compatible_execution_protocols,
                input_binding=_binding(
                    request,
                    input_manifest_hash=file_sha256(manifest_path),
                    scene_contract_hash=file_sha256(contract_path),
                ),
                provider_validation=ProviderValidation(
                    status="passed",
                    metrics={
                        "mean_fingertip_error_m": validation.mean_fingertip_error_m,
                        "maximum_fingertip_error_m": validation.maximum_fingertip_error_m,
                    },
                ),
                provider_metrics={"sampling_steps": report.get("sampling_steps", 50)},
                native_artifacts=(
                    NativeArtifactReference(
                        role="provider_output",
                        path=Path("native/proposal.pkl"),
                        sha256=file_sha256(native_proposal),
                    ),
                    NativeArtifactReference(
                        role="provider_report",
                        path=Path("native/inference_report.json"),
                        sha256=file_sha256(report_path),
                    ),
                ),
            )
            path = sample_dir / "proposal.json"
            write_proposal(path, proposal)
            outputs.append(path)
        return outputs


class SavedShadowProposalProvider:
    """Import one reviewed, directly specified Shadow target."""

    def __init__(self, spec: ProviderSpec, *, project_root: Path):
        self.spec = spec
        self.project_root = project_root

    def generate(self, request: ProposalRequest, output_directory: Path) -> list[Path]:
        if request.sample_count != 1:
            raise ValueError("saved-record provider emits exactly one proposal")
        if request.provider_input_path is None:
            raise ValueError("saved-record provider requires provider_input_path")
        source = _project_input(
            self.project_root, request.provider_input_path, "saved provider input"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "shadow_saved_record_v1":
            raise ValueError("saved Shadow record has the wrong schema version")
        if payload.get("object_id") != request.object.id:
            raise ValueError("saved Shadow record object ID mismatch")
        if payload.get("geometry_content_hash") != request.object.geometry_content_hash:
            raise ValueError("saved Shadow record geometry hash mismatch")
        sample_dir = output_directory / "sample_0000"
        if sample_dir.exists():
            raise FileExistsError(f"proposal sample directory already exists: {sample_dir}")
        sample_dir.mkdir(parents=True)
        revision = _code_revision(self.project_root)
        proposal = GraspProposal(
            proposal_id=f"{request.request_id}_sample_0000",
            provider=ProviderProvenance(
                id=self.spec.provider_id,
                config_hash=canonical_sha256(self.spec),
                code_revision=revision,
                adapter_code_revision=revision,
            ),
            generation=GenerationRecord(seed=request.seed, sample_index=0),
            object=ProposalObjectBinding(
                id=request.object.id,
                geometry_content_hash=request.object.geometry_content_hash,
            ),
            object_to_palm=RigidTransform.model_validate(payload["object_to_palm"]),
            joint_targets_rad=payload["joint_targets_rad"],
            compatible_execution_protocols=self.spec.compatible_execution_protocols,
            input_binding=_binding(
                request, input_manifest_hash=file_sha256(source), scene_contract_hash=None
            ),
            provider_validation=ProviderValidation(status="not_applicable"),
            provider_metrics={"source_record_sha256": file_sha256(source)},
        )
        path = sample_dir / "proposal.json"
        write_proposal(path, proposal)
        return [path]


def load_local_provider(
    spec: ProviderSpec,
    *,
    project_root: Path,
    hug_root: Path | None = None,
    checkpoint: Path | None = None,
) -> ProposalProvider:
    """Construct a registered in-process provider without dynamic code loading."""

    if spec.provider_id == "hug_mano_shadow_v1":
        if hug_root is None or checkpoint is None:
            raise ValueError("HUG provider requires --hug-root and --checkpoint")
        return HugProposalProvider(
            spec, project_root=project_root, hug_root=hug_root, checkpoint=checkpoint
        )
    if spec.provider_id == "shadow_saved_record_v1":
        return SavedShadowProposalProvider(spec, project_root=project_root)
    raise ValueError(
        f"provider {spec.provider_id!r} has no built-in local adapter; "
        "use --external-command or --endpoint"
    )
