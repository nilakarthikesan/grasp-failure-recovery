"""UniDexGrasp native model export -> verified existing Shadow translation.

This adapter never loads a PyTorch pickle or starts inference. The separate
worker is the opt-in GPU boundary. Native exports are numeric NPZ plus JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from pydantic import Field, field_validator

from grasp_failure_prediction.evaluation.schema import (
    SHA256_PATTERN, StrictModel, canonical_sha256,
)
from .artifacts import file_sha256, write_proposal
from .dexgraspnet import (
    SOURCE_JOINT_NAMES, SOURCE_TIP_SITES, _derive_kinematic_mjcf,
    _quaternion_xyzw, validate_fk_translation,
)
from .registry import ProviderSpec
from .schema import (
    GenerationRecord, GraspProposal, NativeArtifactReference,
    ProposalObjectBinding, ProposalRequest, ProviderProvenance,
    ProviderValidation, RigidTransform,
)
from .translation import NativeHandPose, TranslationError, translate_native_pose


SOURCE_REPOSITORY = "https://github.com/PKU-EPIC/UniDexGrasp"
SOURCE_COMMIT = "36c9bfcf7cedc987dd15e2384c548e51d5aadc21"
TRANSLATION_STRATEGY = "dexgraspnet_shadow_to_shadow_hand_right_v1"
# Finger subtree only, with effective MuJoCo defaults/compiler semantics.
# Derived from registered DexGraspNet@bd1b13d shadow_hand_wrist_free.xml whose
# complete-file hash is 39b1f1e1150c9c3821748ba70cfaeaefc2517100d45b5d6d5ef683a7d132de8a.
# The actual UniDex asset retains its own raw hash and is accepted only when
# this certificate demonstrates equivalent finger kinematics.
KINEMATICS_CERTIFICATE_SHA256 = "sha256:7aeae22adce14e36eff79ce01bd6417302333bbcb65d3c740ea800ef1951da2e"


def _child(root: Path, value: str, role: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise TranslationError("native_path_invalid", f"{role} must be a relative file path")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise TranslationError("native_path_invalid", f"{role} escapes its bundle")
    return resolved


def _verify_file(path: Path, expected: str, role: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise TranslationError("native_artifact_hash_mismatch", f"{role} hash does not match its manifest")


def _numeric(value, shape: tuple[int, ...], role: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise TranslationError("native_pose_invalid", f"{role} must be a finite numeric array with shape {shape}")
    return array.astype(np.float64)


def _axis_angle_matrix(value) -> np.ndarray:
    axis_angle = _numeric(value, (3,), "axis angle")
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = axis_angle / angle
    cross = np.asarray([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * (cross @ cross)


def compose_unidexgrasp_prediction(canon_translation, sampled_rotation, hand_qpos) -> np.ndarray:
    """Match upstream add_rotation_to_hand_pose with zero extra rotation.

    ``sampled_rotation`` maps input column coordinates to canonical ones.
    The hand in input coordinates therefore has R=sampled_rotation.T and
    t=sampled_rotation.T @ canon_translation. Joints are unchanged radians.
    """
    translation = _numeric(canon_translation, (3,), "canonical translation")
    rotation = _numeric(sampled_rotation, (3, 3), "sampled rotation")
    joints = _numeric(hand_qpos, (22,), "hand qpos")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
        raise TranslationError("native_rotation_invalid", "sampled rotation must be a proper rotation matrix")
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.T.reshape(9))
    if quat[0] < 0:
        quat *= -1
    length = float(np.linalg.norm(quat[1:]))
    axis_angle = np.zeros(3) if length < 1e-12 else quat[1:] / length * (2. * np.arctan2(length, quat[0]))
    return np.concatenate((rotation.T @ translation, axis_angle, joints))


def _certificate(model: mujoco.MjModel) -> str:
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0:palm")
    if palm != 1 or model.body_parentid[palm] != 0:
        raise TranslationError("source_root_mismatch", "UniDex source chain must start at robot0:palm")
    names = lambda index: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
    def descendant(body):
        while body > 0:
            if body == palm:
                return True
            body = int(model.body_parentid[body])
        return False
    bodies = [body for body in range(model.nbody) if descendant(body)]
    vector = lambda value: [round(float(component), 12) for component in value]
    payload = {
        "bodies": [{
            "name": names(body),
            "parent": None if body == palm else names(int(model.body_parentid[body])),
            "pos": [0., 0., 0.] if body == palm else vector(model.body_pos[body]),
            "quat": [1., 0., 0., 0.] if body == palm else vector(model.body_quat[body]),
        } for body in bodies],
        "joints": [], "tips": [],
    }
    for joint in range(model.njnt):
        if int(model.jnt_bodyid[joint]) not in bodies:
            raise TranslationError("source_joint_set_mismatch", "source has joints outside the palm subtree")
        payload["joints"].append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
            "body": names(int(model.jnt_bodyid[joint])),
            "type": int(model.jnt_type[joint]),
            "pos": vector(model.jnt_pos[joint]), "axis": vector(model.jnt_axis[joint]),
            "range": vector(model.jnt_range[joint]),
            "limited": bool(model.jnt_limited[joint]),
            "reference_rad": round(float(model.qpos0[model.jnt_qposadr[joint]]), 12),
        })
    for name in SOURCE_TIP_SITES:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site < 0:
            raise TranslationError("source_tip_site_missing", f"source hand is missing {name}")
        payload["tips"].append({
            "name": name, "body": names(int(model.site_bodyid[site])),
            "pos": vector(model.site_pos[site]), "quat": vector(model.site_quat[site]),
        })
    return canonical_sha256(payload)


def _prepare_source(source: Path, output: Path, expected_hash: str) -> mujoco.MjModel:
    _verify_file(source, expected_hash, "source hand MJCF")
    xml = ET.parse(source).getroot()
    compiler = xml.find("compiler")
    if compiler is None or compiler.get("angle") != "radian":
        raise TranslationError("source_units_unsupported", "source MJCF must explicitly use radians")
    if any(item.tag in {"include", "frame", "replicate", "attach", "composite", "flexcomp"} for item in xml.iter()):
        raise TranslationError("source_mjcf_unsupported", "source MJCF must be a self-contained simple body/joint tree")
    for body in xml.iter("body"):
        if any(key in body.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis", "mocap")):
            raise TranslationError("source_mjcf_unsupported", "source body rotations must use explicit quaternions")
    _derive_kinematic_mjcf(source, output)
    model = mujoco.MjModel.from_xml_path(str(output))
    if _certificate(model) != KINEMATICS_CERTIFICATE_SHA256:
        raise TranslationError("source_kinematics_not_equivalent", "UniDex source palm/finger kinematics differ from the registered Shadow mapping")
    return model


def decode_unidexgrasp_hand_pose(
    hand_pose, *, object_to_input: RigidTransform, source_hand_mjcf: Path,
    source_hand_sha256: str, kinematics_output: Path,
) -> NativeHandPose:
    """Decode one upstream 28-vector after certifying the source hand asset.

    Distances are already physical meters. ``object_scale`` is geometry
    provenance and must never be applied a second time to hand translation.
    The source palm offset is read from FK rather than assuming identity.
    """
    values = _numeric(hand_pose, (28,), "hand pose")
    model = _prepare_source(source_hand_mjcf, kinematics_output, source_hand_sha256)
    data = mujoco.MjData(model)
    data.qpos[:] = values[6:]
    mujoco.mj_forward(model, data)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0:palm")
    input_root = np.eye(4)
    input_root[:3, :3] = _axis_angle_matrix(values[3:6])
    input_root[:3, 3] = values[:3]
    root_palm = np.eye(4)
    root_palm[:3, :3] = data.xmat[palm].reshape(3, 3)
    root_palm[:3, 3] = data.xpos[palm]
    from .providers import _matrix_from_pose
    object_input = _matrix_from_pose(object_to_input.translation_m, object_to_input.quaternion_xyzw)
    object_palm = object_input @ input_root @ root_palm
    return NativeHandPose(
        # This is an explicit kinematic-equivalence bridge for the registered
        # mapping. It is never the provenance identity of the UniDex asset.
        source_hand_model_id="dexgraspnet_shadow_mjcf_bd1b13d",
        source_root_link="robot0:palm",
        object_to_root=RigidTransform(
            translation_m=tuple(float(item) for item in object_palm[:3, 3]),
            quaternion_xyzw=_quaternion_xyzw(object_palm[:3, :3]),
        ),
        joint_targets_rad=dict(zip(SOURCE_JOINT_NAMES, (float(item) for item in values[6:]), strict=True)),
        payload={
            "actual_source_hand_model_id": "unidexgrasp_shadow_external_mjcf",
            "actual_source_hand_sha256": source_hand_sha256,
            "kinematic_equivalence_certificate_sha256": KINEMATICS_CERTIFICATE_SHA256,
            "source_kinematics_path": str(kinematics_output),
            "source_kinematics_sha256": file_sha256(kinematics_output),
        },
    )


class UniDexGraspExportManifest(StrictModel):
    schema_version: str
    source_repository: str
    source_commit: str
    object_id: str
    geometry_content_hash: str = Field(pattern=SHA256_PATTERN)
    object_scale: float = Field(gt=0, allow_inf_nan=False)
    object_to_input: RigidTransform
    pose_frame: str
    translation_units: str
    rotation_representation: str
    joint_units: str
    joint_names: tuple[str, ...]
    pose_field: str
    output_path: str
    output_sha256: str = Field(pattern=SHA256_PATTERN)
    report_path: str
    report_sha256: str = Field(pattern=SHA256_PATTERN)
    source_hand_mjcf_path: str
    source_hand_mjcf_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("schema_version")
    @classmethod
    def schema(cls, value):
        if value != "unidexgrasp_native_bundle_v1":
            raise ValueError("unsupported UniDex native bundle schema")
        return value


def _load_export(bundle: Path, request: ProposalRequest):
    manifest = UniDexGraspExportManifest.model_validate_json(bundle.read_text(encoding="utf-8"))
    expected = {
        "source_repository": SOURCE_REPOSITORY, "source_commit": SOURCE_COMMIT,
        "pose_frame": "input_point_cloud", "translation_units": "m",
        "rotation_representation": "axis_angle", "joint_units": "rad",
        "joint_names": SOURCE_JOINT_NAMES,
    }
    for key, value in expected.items():
        if getattr(manifest, key) != value:
            raise TranslationError("native_contract_mismatch", f"UniDex export {key} is unsupported")
    if manifest.pose_field not in {"hand_pose", "tta_hand_pose"}:
        raise TranslationError("native_pose_field_unsupported", "select hand_pose or tta_hand_pose explicitly")
    if manifest.object_id != request.object.id or manifest.geometry_content_hash != request.object.geometry_content_hash:
        raise TranslationError("native_object_mismatch", "UniDex object binding does not match request")
    output = _child(bundle.parent, manifest.output_path, "model output")
    report_file = _child(bundle.parent, manifest.report_path, "inference report")
    source = _child(bundle.parent, manifest.source_hand_mjcf_path, "source hand")
    for path, digest, role in (
        (output, manifest.output_sha256, "model output"),
        (report_file, manifest.report_sha256, "inference report"),
        (source, manifest.source_hand_mjcf_sha256, "source hand"),
    ):
        _verify_file(path, digest, role)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    for key, value in {
        "schema_version": "unidexgrasp_inference_report_v1", "classification": "fresh_model_inference",
        "source_repository": SOURCE_REPOSITORY, "source_commit": SOURCE_COMMIT,
        "object_id": request.object.id, "geometry_content_hash": request.object.geometry_content_hash,
        "sample_count": request.sample_count, "seeds": [request.seed + index for index in range(request.sample_count)],
        "object_scale": manifest.object_scale, "output_sha256": manifest.output_sha256,
        "object_to_input": manifest.object_to_input.model_dump(mode="json"),
        "source_hand_mjcf_sha256": manifest.source_hand_mjcf_sha256,
    }.items():
        if report.get(key) != value:
            raise TranslationError("native_report_mismatch", f"UniDex inference report {key} does not match export/request")
    checkpoints = report.get("checkpoint_sha256")
    checkpoint_roles = {"rotation", "pose", "contact"} if manifest.pose_field == "tta_hand_pose" else {"rotation", "pose"}
    if not isinstance(checkpoints, dict) or set(checkpoints) != checkpoint_roles:
        raise TranslationError("native_checkpoint_missing", "UniDex report must bind both learned model checkpoints")
    import re
    required_hashes = list(checkpoints.values()) + [report.get("input_manifest_sha256"), report.get("point_cloud_sha256")]
    configs = report.get("config_sha256")
    if not isinstance(configs, dict) or set(configs) != {"rotation", "pose"}:
        raise TranslationError("native_config_missing", "UniDex report must bind effective IPDF and Glow configurations")
    required_hashes.extend(configs.values())
    if any(not isinstance(value, str) or re.fullmatch(SHA256_PATTERN, value) is None for value in required_hashes):
        raise TranslationError("native_checkpoint_missing", "UniDex checkpoint hashes must be SHA-256 digests")
    if report.get("tta_applied") is not (manifest.pose_field == "tta_hand_pose"):
        raise TranslationError("native_report_mismatch", "selected pose field disagrees with reported TTA status")
    if request.seed + request.sample_count - 1 > 2**32 - 1:
        raise TranslationError("native_seed_invalid", "sample seeds exceed unsigned 32-bit range")
    with np.load(output, allow_pickle=False) as arrays:
        if set(arrays.files) - {"hand_pose", "tta_hand_pose"} or manifest.pose_field not in arrays.files:
            raise TranslationError("native_output_schema_invalid", "UniDex output must contain only declared numeric hand poses")
        poses = _numeric(arrays[manifest.pose_field], (request.sample_count, 28), "batch hand poses")
    return manifest, report, poses, output, report_file, source


class UniDexGraspProposalProvider:
    """Translate a hash-bound fresh model export; fresh execution uses worker."""

    def __init__(self, spec: ProviderSpec, *, project_root: Path):
        self.spec = spec
        self.project_root = project_root.resolve()
        if spec.translation_strategy_id != TRANSLATION_STRATEGY:
            raise ValueError("UniDex provider requires the evidenced Shadow equivalence mapping")

    def generate(self, request: ProposalRequest, output_directory: Path) -> list[Path]:
        if request.provider_input_path is None or request.observation_path is not None or request.provider_inputs:
            raise ValueError("UniDex provider requires provider_input_path as its sole input channel")
        from .providers import _project_input
        bundle = _project_input(self.project_root, request.provider_input_path, "UniDex native bundle")
        return self.generate_from_bundle(request, output_directory, bundle)

    def generate_from_bundle(self, request: ProposalRequest, output_directory: Path, bundle: Path, *, binding_path: Path | None = None) -> list[Path]:
        """Worker hook retaining the caller's original input binding."""
        from .providers import _binding, _code_revision
        manifest, report, poses, output, report_file, source = _load_export(bundle, request)
        if binding_path is not None:
            if file_sha256(binding_path) != report["input_manifest_sha256"]:
                raise TranslationError("native_input_binding_mismatch", "inference report was created from a different caller input manifest")
            original = json.loads(binding_path.read_text(encoding="utf-8"))
            for key, expected in {
                "schema_version": "unidexgrasp_object_input_v1",
                "point_cloud_sha256": report["point_cloud_sha256"],
                "object_to_input": manifest.object_to_input.model_dump(mode="json"),
                "object_scale": manifest.object_scale,
            }.items():
                if original.get(key) != expected:
                    raise TranslationError("native_input_binding_mismatch", f"caller input {key} differs from inference export")
            points = _child(binding_path.parent, original["point_cloud_path"], "caller point cloud")
            _verify_file(points, report["point_cloud_sha256"], "caller point cloud")
        # Complete all decoding/translation/FK gates before writing proposals.
        decoded = []
        with tempfile.TemporaryDirectory(prefix="unidexgrasp-kinematics-") as temp:
            for index, values in enumerate(poses):
                native = decode_unidexgrasp_hand_pose(
                    values, object_to_input=manifest.object_to_input,
                    source_hand_mjcf=source, source_hand_sha256=manifest.source_hand_mjcf_sha256,
                    kinematics_output=Path(temp) / f"source_{index}.xml",
                )
                translated = translate_native_pose(native, self.spec.translation_strategy_id)
                validation = validate_fk_translation(native, translated, maximum_error_m=0.002)
                decoded.append((translated, validation))
        for index in range(request.sample_count):
            sample = output_directory / f"sample_{index:04d}"
            if sample.exists():
                raise FileExistsError(f"proposal sample directory already exists: {sample}")
        result = []
        revision = _code_revision(self.project_root)
        for index, (translated, validation) in enumerate(decoded):
            sample = output_directory / f"sample_{index:04d}"
            native_dir = sample / "native"
            native_dir.mkdir(parents=True)
            artifacts = []
            for role, original, name in (
                ("provider_output", output, "model_output.npz"),
                ("provider_report", report_file, "inference_report.json"),
                ("source_hand_asset", source, "source_hand.xml"),
            ):
                target = native_dir / name
                shutil.copy2(original, target)
                artifacts.append(NativeArtifactReference(role=role, path=Path("native") / name, sha256=file_sha256(target)))
            copied_manifest = manifest.model_copy(update={
                "output_path": "model_output.npz", "report_path": "inference_report.json",
                "source_hand_mjcf_path": "source_hand.xml",
            })
            manifest_file = native_dir / "native_manifest.json"
            manifest_file.write_text(copied_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
            artifacts.append(NativeArtifactReference(role="provider_manifest", path=Path("native/native_manifest.json"), sha256=file_sha256(manifest_file)))
            proposal = GraspProposal(
                proposal_id=f"{request.request_id}_sample_{index:04d}",
                provider=ProviderProvenance(
                    id=self.spec.provider_id, config_hash=canonical_sha256(self.spec),
                    code_revision=SOURCE_COMMIT, adapter_code_revision=revision,
                    checkpoint_hash=canonical_sha256(report["checkpoint_sha256"]),
                ),
                generation=GenerationRecord(seed=request.seed + index, sample_index=index),
                object=ProposalObjectBinding(id=request.object.id, geometry_content_hash=request.object.geometry_content_hash),
                object_to_palm=translated.object_to_palm, joint_targets_rad=translated.joint_targets_rad,
                compatible_execution_protocols=self.spec.compatible_execution_protocols,
                input_binding=_binding(request, input_manifest_hash=file_sha256(binding_path or bundle), scene_contract_hash=None),
                provider_validation=ProviderValidation(status="passed", metrics=validation),
                provider_metrics={
                    "classification": "native_model_export", "native_inference_classification": report["classification"],
                    "pose_field": manifest.pose_field, "object_scale": manifest.object_scale,
                    "inference_input_manifest_sha256": report["input_manifest_sha256"],
                    "inference_point_cloud_sha256": report["point_cloud_sha256"],
                    "model_config_sha256": report["config_sha256"],
                    "actual_source_hand_sha256": manifest.source_hand_mjcf_sha256,
                    "kinematic_equivalence_certificate_sha256": KINEMATICS_CERTIFICATE_SHA256,
                    "translation_bridge": "certified_unidex_palm_subtree_to_registered_dexgraspnet_mapping",
                    "translation_strategy_id": self.spec.translation_strategy_id,
                },
                native_artifacts=tuple(artifacts),
            )
            target = sample / "proposal.json"
            write_proposal(target, proposal)
            result.append(target)
        return result
