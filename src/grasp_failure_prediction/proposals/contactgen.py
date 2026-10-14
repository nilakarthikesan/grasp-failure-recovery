"""ContactGen refined-MANO exports translated through the existing MANO path.

The official demo only saves OBJ meshes. The companion CUDA worker preserves
its optimized MANO parameters, 16 bone frames and 778 vertices in a strict JSON
bundle. Contact maps alone are not executable grasp proposals. Coordinates in
that bundle are metric and relative to the vertex-mean-centered input mesh;
conversion restores the center and the declared mesh-to-object transform.

Official contract: stevenlsw/contactgen@1aea2b6ce498ec90c4af0dd4c5fbdd4a23db761a,
``demo.py``, ``contactgen/contact/contact_optimizer.py``, and the repository's
modified ``manopth/manolayer.py``. That layer returns meters without the upstream
Manopth millimeter scaling. The standard right MANO tip/reorder convention is
from hassony2/manopth@4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b's
``manopth/manolayer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from pydantic import Field, field_validator, model_validator

from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
from grasp_failure_prediction.evaluation.schema import SHA256_PATTERN, StrictModel, canonical_sha256
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT

from .artifacts import file_sha256, write_proposal
from .registry import ProviderSpec
from .schema import (
    GenerationRecord, GraspProposal, InputBinding, NativeArtifactReference,
    ProposalObjectBinding, ProposalRequest, ProviderProvenance, ProviderValidation,
    RigidTransform,
)
from .translation import TranslationError, translate_mano_with_official_retargeter


CONTACTGEN_SOURCE_REVISION = "1aea2b6ce498ec90c4af0dd4c5fbdd4a23db761a"
CONTACTGEN_PROVIDER_ID = "contactgen_mano_shadow_v1"
MANO_RIGHT_TIP_VERTEX_INDICES = (745, 317, 444, 556, 673)
MANO_THUMB_FIRST_JOINT_INDICES = (
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20,
)


def _finite_array(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return result


def _rigid_frames(value) -> np.ndarray:
    frames = _finite_array("hand_frames_centered_m", value, (16, 4, 4))
    rotation = frames[:, :3, :3]
    if not np.allclose(frames[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("MANO frames must have homogeneous last rows")
    if not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=2e-5):
        raise ValueError("MANO frame rotations must be orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1, atol=2e-5):
        raise ValueError("MANO frame rotations must be right-handed")
    return frames


def axis_angle_rotation(axis_angle_rad) -> np.ndarray:
    """Small NumPy Rodrigues helper; ContactGen's global pose is axis-angle."""
    vector = _finite_array("global_pose_axis_angle_rad", axis_angle_rad, (3,))
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def rigid_transform_matrix(pose: RigidTransform) -> np.ndarray:
    matrix = np.eye(4)
    quaternion_wxyz = np.asarray(pose.quaternion_xyzw)[[3, 0, 1, 2]]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion_wxyz)
    matrix[:3, :3] = rotation.reshape(3, 3)
    matrix[:3, 3] = pose.translation_m
    return matrix


def _matrix_pose(matrix: np.ndarray) -> RigidTransform:
    quaternion_wxyz = np.empty(4)
    mujoco.mju_mat2Quat(quaternion_wxyz, matrix[:3, :3].reshape(9))
    quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
    if quaternion_xyzw[3] < 0:
        quaternion_xyzw *= -1
    return RigidTransform(
        translation_m=tuple(float(v) for v in matrix[:3, 3]),
        quaternion_xyzw=tuple(float(v) for v in quaternion_xyzw),
    )


class ContactGenObjectInput(StrictModel):
    """Fresh worker input; vertices in mesh coordinates must already be meters.

    ``object_to_mesh`` maps original input-mesh coordinates into the canonical
    MuJoCo object body frame, before ContactGen subtracts the vertex mean.
    ``mesh_sha256`` is the exact file hash, separate from the registered geometry
    content hash used by the evaluation scene.
    """

    schema_version: Literal["contactgen_object_input_v1"]
    object: ProposalObjectBinding
    mesh_path: Path
    mesh_sha256: str = Field(pattern=SHA256_PATTERN)
    units: Literal["meters"]
    object_to_mesh: RigidTransform

    @field_validator("mesh_path")
    @classmethod
    def relative_mesh(cls, value):
        if value.is_absolute() or ".." in value.parts or not value.name:
            raise ValueError("mesh_path must be a project-relative file")
        return value


class ContactGenOptimizedSample(StrictModel):
    sample_index: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    global_pose_axis_angle_rad: tuple[float, float, float]
    mano_pose_pca: tuple[float, ...]
    mano_shape: tuple[float, ...]
    mano_translation_m: tuple[float, float, float]
    hand_vertices_centered_m: list[list[float]]
    hand_frames_centered_m: list[list[list[float]]]

    @model_validator(mode="after")
    def valid_optimized_geometry(self):
        _finite_array("mano_pose_pca", self.mano_pose_pca, (26,))
        _finite_array("mano_shape", self.mano_shape, (10,))
        _finite_array("mano_translation_m", self.mano_translation_m, (3,))
        _finite_array("hand_vertices_centered_m", self.hand_vertices_centered_m, (778, 3))
        frames = _rigid_frames(self.hand_frames_centered_m)
        if not np.allclose(frames[0, :3, :3], axis_angle_rotation(self.global_pose_axis_angle_rad), atol=2e-5):
            raise ValueError("MANO wrist frame rotation disagrees with optimized global axis-angle")
        return self


class ContactGenNativeBundle(StrictModel):
    """Completed ContactGen + optimize_pose output, never a contact-only map."""

    schema_version: Literal["contactgen_native_bundle_v1"]
    source_revision: Literal[CONTACTGEN_SOURCE_REVISION]
    generation_stage: Literal["refined_mano"]
    mano_configuration: Literal["right_pca26_nonflat_mean"]
    coordinate_frame: Literal["vertex_mean_centered_object_mesh"]
    units: Literal["meters"]
    object_input: ContactGenObjectInput
    mesh_vertex_mean_m: tuple[float, float, float]
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    hand_sdf_checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    mano_asset_sha256: str = Field(pattern=SHA256_PATTERN)
    global_iterations: int = Field(ge=1)
    pose_iterations: int = Field(ge=1)
    samples: tuple[ContactGenOptimizedSample, ...] = Field(min_length=1)

    @field_validator("mesh_vertex_mean_m")
    @classmethod
    def finite_center(cls, value):
        _finite_array("mesh_vertex_mean_m", value, (3,))
        return value

    @model_validator(mode="after")
    def unique_samples(self):
        indices = [sample.sample_index for sample in self.samples]
        if len(indices) != len(set(indices)):
            raise ValueError("ContactGen bundle sample indices must be unique")
        return self


def load_contactgen_bundle(path: Path) -> ContactGenNativeBundle:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ContactGen native bundle must be a JSON object")
    if raw.get("schema_version") == "contactgen_object_input_v1":
        raise ValueError("ContactGen object input requires fresh inference; use scripts/infer_contactgen_object.py through --external-command")
    if raw.get("generation_stage") != "refined_mano":
        raise TranslationError(
            "contactgen_refinement_required",
            "ContactGen contact maps/parts/directions require optimize_pose before MANO translation",
        )
    return ContactGenNativeBundle.model_validate(raw)


@dataclass(frozen=True)
class ManoLandmarkInput:
    """The existing retargeter interface; frame is object rather than camera."""

    landmarks_3d: np.ndarray
    T_camera_wrist: np.ndarray


def contactgen_mano_landmarks(
    sample: ContactGenOptimizedSample, object_input: ContactGenObjectInput,
    mesh_vertex_mean_m,
) -> ManoLandmarkInput:
    """Restore mesh centering and construct the standard 21 right MANO joints.

    ContactGen returns 16 MANO bone frames in original MANO joint order. Add
    five mesh fingertips and reorder wrist/thumb/index/middle/ring/little. The
    wrist frame translation is frame[0], which includes the shaped MANO root;
    ``mano_translation_m`` alone is not the wrist position.
    """
    frames = _rigid_frames(sample.hand_frames_centered_m)
    vertices = np.asarray(sample.hand_vertices_centered_m, dtype=np.float64)
    joints = np.concatenate((frames[:, :3, 3], vertices[list(MANO_RIGHT_TIP_VERTEX_INDICES)]))
    joints = joints[list(MANO_THUMB_FIRST_JOINT_INDICES)]
    center = _finite_array("mesh_vertex_mean_m", mesh_vertex_mean_m, (3,))
    object_mesh = rigid_transform_matrix(object_input.object_to_mesh)
    landmarks = (joints + center) @ object_mesh[:3, :3].T + object_mesh[:3, 3]
    mesh_wrist = frames[0].copy()
    mesh_wrist[:3, 3] += center
    return ManoLandmarkInput(landmarks_3d=landmarks, T_camera_wrist=object_mesh @ mesh_wrist)


def contactgen_object_palm_pose(landmarks: ManoLandmarkInput, pose, validator) -> RigidTransform:
    """Apply the same wrist-origin/palm-origin policy as the HUG adapter."""
    validator.set_pose(pose)
    palm_id = mujoco.mj_name2id(validator.model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    if palm_id < 0:
        raise ValueError("Shadow validation model is missing palm")
    base_palm_rotation = validator.data.xmat[palm_id].reshape(3, 3)
    object_palm = landmarks.T_camera_wrist.copy()
    object_palm[:3, :3] = object_palm[:3, :3] @ MANO_TO_OPERATOR_RIGHT.T @ base_palm_rotation
    return _matrix_pose(object_palm)


def _input_path(request: ProposalRequest, project_root: Path) -> Path:
    if request.observation_path is not None:
        raise ValueError("ContactGen requires one provider input channel; observation_path is unsupported")
    if request.provider_input_path is not None and request.provider_inputs:
        raise ValueError("ContactGen requires exactly one of provider_input_path or a sole provider_input role")
    if request.provider_inputs and (
        len(request.provider_inputs) != 1 or request.provider_inputs[0].role != "provider_input"
    ):
        raise ValueError("ContactGen provider_inputs must contain only one provider_input role")
    relative = request.provider_input_path
    if relative is None:
        relative = next((item.path for item in request.provider_inputs if item.role == "provider_input"), None)
    if relative is None:
        raise ValueError("ContactGen requires provider_input_path or provider_input role")
    path = (project_root / relative).resolve()
    if not path.is_relative_to(project_root.resolve()):
        raise ValueError("ContactGen provider input escapes project root")
    if not path.is_file():
        raise FileNotFoundError(f"ContactGen provider input does not exist: {path}")
    return path


def validate_contactgen_mesh(
    object_input: ContactGenObjectInput, project_root: Path, *, expected_vertex_mean_m=None,
):
    """Hash-check the exact mesh and verify the declared centering operation.

    Trimesh is an optional evaluation dependency; loading a mesh performs no
    model inference, sampling, simulation, or evaluation-data generation.
    """
    path = (project_root / object_input.mesh_path).resolve()
    if not path.is_relative_to(project_root.resolve()):
        raise ValueError("ContactGen mesh path escapes project root")
    if not path.is_file():
        raise FileNotFoundError(f"ContactGen object mesh does not exist: {path}")
    if file_sha256(path) != object_input.mesh_sha256:
        raise ValueError("ContactGen input mesh hash does not match declared mesh_sha256")
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError('ContactGen mesh validation requires Trimesh; install the "eval" extra') from exc
    mesh = trimesh.load(path)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("ContactGen requires one nonempty triangle mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("ContactGen mesh vertices must be a finite Nx3 array")
    center = vertices.mean(axis=0)
    if expected_vertex_mean_m is not None:
        declared = _finite_array("mesh_vertex_mean_m", expected_vertex_mean_m, (3,))
        if not np.allclose(center, declared, rtol=0, atol=1e-7):
            raise ValueError("ContactGen declared vertex mean does not match the source mesh")
    return mesh, center


def convert_contactgen_bundle(
    bundle: ContactGenNativeBundle, request: ProposalRequest, spec: ProviderSpec, *,
    project_root: Path, output_directory: Path,
    generation_mode: Literal["native_model_output_import", "fresh_official_inference"] = "native_model_output_import",
) -> list[Path]:
    """Validate and translate declared optimized samples without model inference.

    Hashes bind exact source files. A native export is a trusted input, rather
    than an independent authentication of its model/checkpoint provenance.
    """
    # Lazy imports avoid the generate -> providers -> ContactGen import cycle.
    from .generate import _expected_input_binding
    from .providers import _code_revision

    _input_path(request, project_root)
    if spec.translation_strategy_id != "mano_right_to_shadow_hand_right_v1":
        raise ValueError("ContactGen requires the existing right MANO-to-Shadow translation strategy")
    if spec.provider_id != CONTACTGEN_PROVIDER_ID or request.provider.id != spec.provider_id:
        raise ValueError("ContactGen provider ID does not match request/specification")
    if spec.provenance.source_revision != bundle.source_revision:
        raise ValueError("ContactGen bundle source revision does not match provider registry")
    if bundle.object_input.object.id != request.object.id or bundle.object_input.object.geometry_content_hash != request.object.geometry_content_hash:
        raise ValueError("ContactGen bundle object/geometry does not match request")
    if len(bundle.samples) != request.sample_count:
        raise ValueError("ContactGen bundle sample count does not match request")
    if request.seed + request.sample_count - 1 > 2**32 - 1:
        raise ValueError("ContactGen sample seed exceeds unsigned 32-bit range")
    ordered = sorted(bundle.samples, key=lambda sample: sample.sample_index)
    for index, sample in enumerate(ordered):
        if sample.sample_index != index or sample.seed != request.seed + index:
            raise ValueError("ContactGen bundle sample index/seed does not match request")
        sample_dir = output_directory / f"sample_{index:04d}"
        if sample_dir.exists():
            raise FileExistsError(f"proposal sample directory already exists: {sample_dir}")
    validate_contactgen_mesh(bundle.object_input, project_root, expected_vertex_mean_m=bundle.mesh_vertex_mean_m)
    binding = InputBinding.model_validate(_expected_input_binding(request, project_root.resolve()))
    validator = ShadowPoseValidator(default_dex_urdf_root())
    proposals = []
    for sample in ordered:
        landmarks = contactgen_mano_landmarks(sample, bundle.object_input, bundle.mesh_vertex_mean_m)
        pose = translate_mano_with_official_retargeter(landmarks, spec.translation_strategy_id)
        validation = validator.validate(pose)
        maximum = spec.maximum_mean_fingertip_error_m
        if maximum is None or validation.mean_fingertip_error_m > maximum:
            raise TranslationError(
                "contactgen_mano_alignment_failed",
                f"ContactGen MANO retargeting alignment gate failed: {validation.mean_fingertip_error_m:.6f} m",
            )
        proposal = GraspProposal(
            proposal_id=f"{request.request_id}_sample_{sample.sample_index:04d}",
            provider=ProviderProvenance(
                id=spec.provider_id, config_hash=canonical_sha256(spec),
                code_revision=bundle.source_revision,
                adapter_code_revision=_code_revision(project_root),
                checkpoint_hash=bundle.checkpoint_sha256,
            ),
            generation=GenerationRecord(seed=sample.seed, sample_index=sample.sample_index),
            object=bundle.object_input.object,
            object_to_palm=contactgen_object_palm_pose(landmarks, pose, validator),
            joint_targets_rad={name: float(value) for name, value in zip(pose.joint_names, pose.qpos, strict=True)},
            compatible_execution_protocols=spec.compatible_execution_protocols,
            input_binding=binding,
            provider_validation=ProviderValidation(status="passed", metrics={
                "mean_fingertip_error_m": validation.mean_fingertip_error_m,
                "maximum_fingertip_error_m": validation.maximum_fingertip_error_m,
                "translation_strategy_id": spec.translation_strategy_id,
            }),
            provider_metrics={
                "generation_mode": generation_mode,
                "fresh_inference": generation_mode == "fresh_official_inference",
                "generation_stage": bundle.generation_stage,
                "global_iterations": bundle.global_iterations,
                "pose_iterations": bundle.pose_iterations,
                "hand_sdf_checkpoint_sha256": bundle.hand_sdf_checkpoint_sha256,
                "mano_asset_sha256": bundle.mano_asset_sha256,
                "mesh_sha256": bundle.object_input.mesh_sha256,
            },
        )
        proposals.append((sample, proposal))
    outputs = []
    for sample, proposal in proposals:
        sample_dir = output_directory / f"sample_{sample.sample_index:04d}"
        native = sample_dir / "native/contactgen_bundle.json"
        native.parent.mkdir(parents=True, exist_ok=False)
        single = bundle.model_copy(update={"samples": (sample,)})
        native.write_text(json.dumps(single.model_dump(mode="json"), sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        proposal = proposal.model_copy(update={"native_artifacts": (NativeArtifactReference(
            role="provider_output", path=Path("native/contactgen_bundle.json"), sha256=file_sha256(native),
        ),)})
        output = sample_dir / "proposal.json"
        write_proposal(output, proposal)
        outputs.append(output)
    return outputs


class ContactGenProposalProvider:
    """Import a completed official ContactGen bundle through MANO retargeting.

    Fresh inference uses ``scripts/infer_contactgen_object.py`` as the existing
    external-command worker. Local native imports never invoke a model.
    """

    def __init__(self, spec: ProviderSpec, *, project_root: Path):
        self.spec = spec
        self.project_root = project_root.resolve()

    def generate(self, request: ProposalRequest, output_directory: Path) -> list[Path]:
        path = _input_path(request, self.project_root)
        bundle = load_contactgen_bundle(path)
        return convert_contactgen_bundle(
            bundle, request, self.spec, project_root=self.project_root,
            output_directory=output_directory,
        )
