"""DGA native export decoder and UNREGISTERED static 24-joint candidate.

No model/checkpoint loading, inference, dynamics, optimization or clipping lives
here. A candidate is deliberately not a GraspProposal: scientific native/surface
acceptance and shared registry integration remain separate gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


SOURCE_REPOSITORY = "https://github.com/4DVLab/DexGrasp-Anything"
SOURCE_COMMIT = "86353d710ca71d0f547f0d8b15cc8ca959c78163"
SOURCE_URDF_SHA256 = "sha256:932fca7c411ef9ae4e6a8f17a892b3629840b5b5e1254610e0616e184d1f95c7"
TARGET_URDF_SHA256 = "sha256:29a50063cced2fba81dfe3d208ba54499fc12635c9189d0822f7caa2771c36de"
CHECKPOINT_SHA256 = "sha256:f7675d8eb93b3978b2c2d72ec9f54ead6b936d1dd307e07605daa0c02bcac14f"
NORMALIZER_SHA256 = "sha256:495f09cf24f9e847376a3c858b06b28b509f01a28a4fede28dced7840f59622d"
SOURCE_JOINT_NAMES = (
    "WRJ2", "WRJ1", "FFJ4", "FFJ3", "FFJ2", "FFJ1",
    "MFJ4", "MFJ3", "MFJ2", "MFJ1", "RFJ4", "RFJ3", "RFJ2", "RFJ1",
    "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1", "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
)
CANDIDATE_STRATEGY = "dga_24_joint_palm_compensated_candidate_v1"
SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class DGAContractError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _numeric(value, shape, role):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise DGAContractError("native_numeric_invalid", f"{role} must be finite numeric {shape}")
    return array.astype(np.float64)


def proper_transform(value, role="object_from_input") -> np.ndarray:
    matrix = _numeric(value, (4, 4), role)
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-7, rtol=0)):
        raise DGAContractError("native_transform_invalid", f"{role} must be rigid, without scale/reflection")
    return matrix


def rotation_from_6d(value) -> np.ndarray:
    """Match pinned robust_compute_rotation_matrix_from_ortho6d.

    Six numbers are two consecutive COLUMN vectors. The symmetric middle/diff
    construction matters for nonorthogonal input; Gram-Schmidt is different.
    Degenerate columns fail closed instead of returning a singular matrix.
    """
    values = _numeric(value, (6,), "rotation6d")
    def unit(vector):
        length = np.linalg.norm(vector)
        if length < 1e-8:
            raise DGAContractError("native_rotation_degenerate", "6D rotation columns are degenerate")
        return vector / length
    x, y = unit(values[:3]), unit(values[3:])
    middle, difference = unit(x + y), unit(x - y)
    x, y = unit(middle + difference), unit(middle - difference)
    return np.column_stack((x, y, unit(np.cross(x, y))))


def compose_denormalized_prediction(values, scene_rotation) -> np.ndarray:
    """27 physical values -> exact visualizer 33-vector; no object rescaling.

    The input must ALREADY be denormalized using the chosen dataset's flags.
    scene_rotation maps input points into the augmented sampling frame.
    Upstream uses identity root orientation there and undoes rotation once.
    """
    prediction = _numeric(values, (27,), "denormalized diffusion prediction")
    rotation = _numeric(scene_rotation, (3, 3), "scene rotation")
    transform = np.eye(4)
    transform[:3, :3] = rotation
    proper_transform(transform, "scene rotation")
    inverse = rotation.T
    return np.concatenate((inverse @ prediction[:3], inverse.T[:2].reshape(6), prediction[3:]))


def _axis_rotation(axis, angle):
    x, y, z = axis / np.linalg.norm(axis)
    cross = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _vector(node, key, default="0 0 0"):
    value = default if node is None else node.get(key, default)
    return _numeric(np.fromstring(value, sep=" "), (3,), key)


class DGAHandTree:
    """Geometry-free pinned URDF FK; never compiles a dynamics model."""

    def __init__(self, path: Path, expected_sha256: str):
        self.sha256 = file_sha256(path)
        if self.sha256 != expected_sha256:
            raise DGAContractError("hand_asset_hash_mismatch", "hand URDF does not match its pinned hash")
        xml = ET.parse(path).getroot()
        self.links = tuple(node.attrib["name"] for node in xml.findall("link"))
        self.joints = {}
        for node in xml.findall("joint"):
            if node.find("mimic") is not None or node.get("type") not in {"fixed", "revolute"}:
                raise DGAContractError("hand_joint_unsupported", "only fixed/revolute, unmimicked URDF joints supported")
            origin = node.find("origin")
            roll, pitch, yaw = _vector(origin, "rpy")
            matrix = np.eye(4)
            matrix[:3, :3] = (_axis_rotation(np.array([0., 0., 1.]), yaw)
                              @ _axis_rotation(np.array([0., 1., 0.]), pitch)
                              @ _axis_rotation(np.array([1., 0., 0.]), roll))
            matrix[:3, 3] = _vector(origin, "xyz")
            limit = node.find("limit")
            axis = _vector(node.find("axis"), "xyz", "1 0 0")
            if node.get("type") == "revolute" and (limit is None or np.linalg.norm(axis) < 1e-8):
                raise DGAContractError("hand_joint_unsupported", "revolute joints require finite limits and a nonzero axis")
            limits = None if node.get("type") == "fixed" else _numeric(
                [float(limit.attrib["lower"]), float(limit.attrib["upper"])], (2,), "joint limits")
            if limits is not None and limits[0] > limits[1]:
                raise DGAContractError("hand_joint_unsupported", "joint limits are reversed")
            name = node.attrib["name"]
            if name in self.joints:
                raise DGAContractError("hand_tree_invalid", "duplicate joint")
            self.joints[name] = (node.find("parent").attrib["link"], node.find("child").attrib["link"], matrix, axis, limits)
        parents = [joint[0] for joint in self.joints.values()]
        children = [joint[1] for joint in self.joints.values()]
        roots = set(self.links) - set(children)
        if len(roots) != 1 or len(set(children)) != len(children) or len(set(self.links)) != len(self.links):
            raise DGAContractError("hand_tree_invalid", "URDF must be a single rooted tree")
        if not set(parents + children).issubset(self.links):
            raise DGAContractError("hand_tree_invalid", "joint references an unknown link")
        self.root = next(iter(roots))
        self.joint_names = tuple(name for name, joint in self.joints.items() if joint[4] is not None)
        if set(self.joint_names) != set(SOURCE_JOINT_NAMES) or "palm" not in self.links:
            raise DGAContractError("hand_joint_set_mismatch", "hand must contain palm and the exact 24 DGA names")
        self.forward(dict.fromkeys(SOURCE_JOINT_NAMES, 0.), check_limits=False)

    def forward(self, joints, *, check_limits=True):
        if set(joints) != set(SOURCE_JOINT_NAMES):
            raise DGAContractError("native_joint_set_mismatch", "all 24 joints, including wrists, are required")
        values = dict(zip(SOURCE_JOINT_NAMES, _numeric([joints[n] for n in SOURCE_JOINT_NAMES], (24,), "joints"), strict=True))
        frames = {self.root: np.eye(4)}
        pending = dict(self.joints)
        while pending:
            progressed = False
            for name, (parent, child, origin, axis, limits) in list(pending.items()):
                if parent not in frames:
                    continue
                local = origin.copy()
                if limits is not None:
                    angle = values[name]
                    if check_limits and not limits[0] <= angle <= limits[1]:
                        raise DGAContractError("joint_limit_violation", f"{name}={angle} outside [{limits[0]}, {limits[1]}]; no clipping")
                    local[:3, :3] = origin[:3, :3] @ _axis_rotation(axis, angle)
                frames[child] = frames[parent] @ local
                del pending[name]
                progressed = True
            if not progressed:
                raise DGAContractError("hand_tree_invalid", "disconnected or cyclic URDF")
        if set(frames) != set(self.links):
            raise DGAContractError("hand_tree_invalid", "unreachable link")
        return frames


@dataclass(frozen=True)
class DGANativePose:
    object_from_forearm: np.ndarray
    object_from_palm: np.ndarray
    joint_targets_rad: dict[str, float]
    object_from_links: dict[str, np.ndarray]
    source_hand_sha256: str
    source_hand_urdf: Path


def decode_dga_hand_pose(values, *, object_from_input, source_hand_urdf: Path) -> DGANativePose:
    values = _numeric(values, (33,), "DGA native pose")
    source = DGAHandTree(source_hand_urdf, SOURCE_URDF_SHA256)
    if source.root != "forearm":
        raise DGAContractError("source_root_mismatch", "DGA source root must be forearm")
    joints = dict(zip(SOURCE_JOINT_NAMES, values[9:], strict=True))
    frames = source.forward(joints)
    input_from_forearm = np.eye(4)
    input_from_forearm[:3, :3] = rotation_from_6d(values[3:9])
    input_from_forearm[:3, 3] = values[:3]
    object_from_forearm = proper_transform(proper_transform(object_from_input) @ input_from_forearm, "object_from_forearm")
    object_frames = {name: object_from_forearm @ frame for name, frame in frames.items()}
    return DGANativePose(object_from_forearm, object_frames["palm"], joints, object_frames, source.sha256, source_hand_urdf.resolve())


def map_dga_candidate(native: DGANativePose, *, target_hand_urdf: Path) -> dict:
    """Unregistered approximation: preserve palm pose and all joint values.

    Counter-transform the target root to prevent applying wrist FK twice.
    Source/target hand geometry is different. No scientific tolerance is applied.
    """
    if native.source_hand_sha256 != SOURCE_URDF_SHA256:
        raise DGAContractError("source_hand_mismatch", "candidate requires the pinned DGA native hand")
    source = DGAHandTree(native.source_hand_urdf, SOURCE_URDF_SHA256)
    source_frames = source.forward(native.joint_targets_rad)
    source_root = proper_transform(native.object_from_forearm, "native object_from_forearm")
    palm = proper_transform(native.object_from_palm, "native object_from_palm")
    if set(native.object_from_links) != set(source_frames):
        raise DGAContractError("native_link_set_mismatch", "native frames must contain the exact source link set")
    for name, frame in native.object_from_links.items():
        if not np.allclose(proper_transform(frame, name), source_root @ source_frames[name], atol=1e-10, rtol=0):
            raise DGAContractError("native_frame_inconsistent", f"native {name} frame disagrees with pinned source FK")
    if not np.allclose(palm, native.object_from_links["palm"], atol=1e-10, rtol=0):
        raise DGAContractError("native_frame_inconsistent", "native palm transform disagrees with source FK")
    target = DGAHandTree(target_hand_urdf, TARGET_URDF_SHA256)
    frames = target.forward(native.joint_targets_rad)
    object_from_target_root = native.object_from_palm @ np.linalg.inv(frames["palm"])
    mapped = {name: object_from_target_root @ frame for name, frame in frames.items()}
    common = sorted(set(native.object_from_links) & set(mapped))
    return {
        "schema_version": "dga_static_mapping_candidate_v1",
        "strategy_id": CANDIDATE_STRATEGY, "execution_eligible": False,
        "acceptance": "UNRESOLVED", "actual_native_validation": "NOT_TESTED",
        "surface_fidelity": "NOT_TESTED", "physics_steps": 0,
        "source_hand_sha256": native.source_hand_sha256, "target_hand_sha256": target.sha256,
        "object_from_palm": native.object_from_palm.tolist(),
        "object_from_target_root": object_from_target_root.tolist(),
        "joint_targets_rad": {name: float(value) for name, value in native.joint_targets_rad.items()},
        "link_origin_error_m": {name: float(np.linalg.norm(native.object_from_links[name][:3, 3] - mapped[name][:3, 3])) for name in common},
    }


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class DGAFile(_Strict):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)


class DGAExportManifest(_Strict):
    schema_version: Literal["dga_native_export_v1"]
    evidence_origin: Literal["authentic_saved_native", "manufactured_contract_fixture"]
    source_repository: Literal[SOURCE_REPOSITORY]
    source_commit: Literal[SOURCE_COMMIT]
    checkpoint_sha256: Literal[CHECKPOINT_SHA256]
    dataset_name: Literal["DexGraspNet"]
    normalizer_source_sha256: Literal[NORMALIZER_SHA256]
    normalize_joints: bool
    normalize_translation: bool
    state: Literal["denormalized_inverse_rotated_33"]
    # Guidance-enabled exports need a separately reviewed resolved optimizer
    # contract (weights, scales and clipping). A Boolean alone is insufficient.
    physics_guided_sampling: Literal[False]
    postprocessing: Literal["none"]
    source_root: Literal["forearm"]
    source_hand_scale: Literal[1.0]
    rotation_representation: Literal["robust_6d_two_columns"]
    translation_units: Literal["m"]
    joint_units: Literal["rad"]
    input_frame: Literal["input_point_cloud_physical_m"]
    normal_vectors: Literal["preserved_as_conditioned"]
    joint_names: list[str]
    object_id: str = Field(min_length=1)
    geometry_content_hash: str = Field(pattern=SHA256_PATTERN)
    physical_object_scale: float = Field(gt=0)
    object_from_input: list[list[float]]
    seed: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    output: DGAFile
    source_hand: DGAFile
    object_mesh: DGAFile
    point_cloud: DGAFile
    normals: DGAFile
    configuration: DGAFile
    generation_log: DGAFile


def _verified_child(root: Path, reference: DGAFile) -> Path:
    relative = Path(reference.path)
    child = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not child.is_relative_to(root.resolve()):
        raise DGAContractError("native_path_invalid", "artifact must stay inside its export bundle")
    if not child.is_file() or file_sha256(child) != reference.sha256:
        raise DGAContractError("native_artifact_hash_mismatch", f"hash mismatch for {reference.path}")
    return child


def load_dga_export(manifest_path: Path, *, expected_object_id: str, expected_geometry_hash: str):
    """Hash-checked numeric interchange, never torch/pickle.

    Hashes establish bundle integrity, not authenticity of self-reported origin.
    Saved-output provenance still requires independent acquisition/log review.
    Physical scale is provenance; it is never multiplied into hand translation.
    """
    manifest = DGAExportManifest.model_validate_json(manifest_path.read_bytes())
    if manifest.joint_names != list(SOURCE_JOINT_NAMES):
        raise DGAContractError("native_joint_order_mismatch", "the explicit 24-joint tensor order is required")
    if (manifest.object_id != expected_object_id or manifest.geometry_content_hash != expected_geometry_hash):
        raise DGAContractError("native_object_mismatch", "export object binding differs from request")
    proper_transform(manifest.object_from_input)
    paths = {name: _verified_child(manifest_path.parent, getattr(manifest, name)) for name in (
        "output", "source_hand", "object_mesh", "point_cloud", "normals", "configuration", "generation_log")}
    if manifest.source_hand.sha256 != SOURCE_URDF_SHA256:
        raise DGAContractError("source_hand_mismatch", "source hand must be the pinned DGA URDF")
    config = json.loads(paths["configuration"].read_text(encoding="utf-8"))
    for key in ("source_commit", "checkpoint_sha256", "dataset_name", "normalizer_source_sha256",
                "normalize_joints", "normalize_translation", "physics_guided_sampling", "postprocessing", "seed", "sample_count"):
        if key not in config or type(config[key]) is not type(getattr(manifest, key)) or config[key] != getattr(manifest, key):
            raise DGAContractError("native_configuration_mismatch", f"configuration must explicitly bind {key}")
    if not isinstance(config.get("optimizer"), dict):
        raise DGAContractError("native_configuration_mismatch", "record optimizer configuration even when guidance is disabled")
    if config["optimizer"].get("enabled") is not manifest.physics_guided_sampling:
        raise DGAContractError("native_configuration_mismatch", "actual optimizer enabled state must match the declared stage")
    log = json.loads(paths["generation_log"].read_text(encoding="utf-8"))
    log_expected = {
        "schema_version": "dga_export_log_v1", "evidence_origin": manifest.evidence_origin,
        "configuration_sha256": manifest.configuration.sha256, "output_sha256": manifest.output.sha256,
        "object_id": manifest.object_id, "geometry_content_hash": manifest.geometry_content_hash,
        "object_mesh_sha256": manifest.object_mesh.sha256,
        "point_cloud_sha256": manifest.point_cloud.sha256, "normals_sha256": manifest.normals.sha256,
        "source_hand_scale": manifest.source_hand_scale, "physical_object_scale": manifest.physical_object_scale,
        "object_from_input": manifest.object_from_input,
        "joint_parameter_names": list(SOURCE_JOINT_NAMES), "state": manifest.state,
    }
    if any(log.get(key) != value for key, value in log_expected.items()):
        raise DGAContractError("native_log_mismatch", "generation log must bind runtime joint order, configuration, state and physical input")
    with np.load(paths["output"], allow_pickle=False) as archive:
        if set(archive.files) != {"hand_qpos"}:
            raise DGAContractError("native_fields_unsupported", "export must contain only explicit hand_qpos")
        values = _numeric(archive["hand_qpos"], (manifest.sample_count, 33), "hand_qpos")
    points = np.load(paths["point_cloud"], allow_pickle=False)
    normals = np.load(paths["normals"], allow_pickle=False)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise DGAContractError("native_input_invalid", "point cloud must be nonempty Nx3 physical input")
    points = _numeric(points, points.shape, "point cloud")
    normals = _numeric(normals, points.shape, "point normals")
    # Upstream average-scale handling can change normal magnitudes. Preserve
    # the ACTUAL model conditioning bytes; silently normalizing changes input.
    if np.any(np.max(np.abs(normals), axis=1) < 1e-12):
        raise DGAContractError("native_input_invalid", "input normals must be nonzero; magnitudes are preserved")
    poses = [decode_dga_hand_pose(row, object_from_input=manifest.object_from_input, source_hand_urdf=paths["source_hand"]) for row in values]
    return manifest, poses
