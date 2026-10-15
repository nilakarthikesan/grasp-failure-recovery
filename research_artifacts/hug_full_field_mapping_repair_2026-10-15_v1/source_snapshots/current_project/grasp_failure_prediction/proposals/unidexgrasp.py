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


def read_unidexgrasp_input(path: Path, request: ProposalRequest):
    """Validate caller-prepared physical point coordinates, without sampling."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version", "object_id", "geometry_content_hash", "object_scale",
        "object_to_input", "point_cloud_path", "point_cloud_sha256", "point_cloud_units",
        "point_cloud_frame", "plane_input",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys or raw["schema_version"] != "unidexgrasp_object_input_v1":
        raise ValueError("worker requires a strict unidexgrasp_object_input_v1 manifest")
    if raw["object_id"] != request.object.id or raw["geometry_content_hash"] != request.object.geometry_content_hash:
        raise ValueError("UniDex object/geometry binding differs from request")
    if raw["point_cloud_units"] != "m" or raw["point_cloud_frame"] != "input_point_cloud":
        raise ValueError("UniDex point coordinates must be physical meters in the input point-cloud frame")
    scale = raw["object_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("object_scale must be finite and positive (geometry already scaled)")
    transform = RigidTransform.model_validate(raw["object_to_input"])
    plane = np.asarray(raw["plane_input"])
    if plane.shape != (4,) or plane.dtype.kind not in "fiu" or not np.isfinite(plane).all() or not np.isclose(np.linalg.norm(plane[:3]), 1., atol=1e-6):
        raise ValueError("plane_input must contain a unit normal and finite metric offset")
    points_file = _child(path.parent, raw["point_cloud_path"], "point cloud")
    _verify_file(points_file, raw["point_cloud_sha256"], "point cloud")
    points = np.load(points_file, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1024 or points.dtype.kind not in "fiu" or not np.isfinite(points).all():
        raise ValueError("caller-prepared point cloud must contain at least 1024 finite XYZ points")
    with np.errstate(over="ignore", invalid="ignore"):
        plane, points = plane.astype(np.float32), points.astype(np.float32)
    if not np.isfinite(plane).all() or not np.isfinite(points).all():
        raise ValueError("UniDex input exceeds finite model float32 coordinates")
    return raw, transform, plane, points


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


def _validate_tta_configuration(report: dict) -> None:
    """Bind resolved ContactNet and full eval configs for pinned upstream TTA.

    The eval config includes its TTA block and dataset surface-point counts;
    weights alone do not identify the adaptation objective or optimizer run.
    These declarations are provenance, not proof that inference occurred.
    """
    effective = report.get("effective_config")
    if not isinstance(effective, dict) or set(effective) != {"contact", "tta"}:
        raise TranslationError("native_tta_config_missing", "adapted poses require resolved ContactNet and full TTA eval configurations")
    def unresolved(value):
        if isinstance(value, dict):
            return any(unresolved(item) for item in value.values())
        if isinstance(value, list):
            return any(unresolved(item) for item in value)
        return isinstance(value, str) and ("${" in value or value == "???")

    for role, payload in effective.items():
        if not isinstance(payload, dict) or not payload:
            raise TranslationError("native_tta_config_missing", f"effective {role} configuration must be a nonempty mapping")
        if unresolved(payload):
            raise TranslationError("native_tta_config_invalid", f"effective {role} configuration must be fully resolved")
        try:
            digest = canonical_sha256(payload)
        except (TypeError, ValueError) as exc:
            raise TranslationError("native_tta_config_invalid", f"effective {role} configuration must be finite JSON") from exc
        if digest != report["config_sha256"][role]:
            raise TranslationError("native_tta_config_mismatch", f"effective {role} configuration differs from its hash")
    contact = effective["contact"]
    evaluation = effective["tta"]
    tta, dataset = evaluation.get("tta"), evaluation.get("dataset")
    required = {"contact_net", "batch_size", "iterations", "lr", "normalize_factor", "weight_cmap", "weight_dis", "weight_pen", "weight_spen", "weight_tpen", "thres_dis"}
    if not isinstance(tta, dict) or not required.issubset(tta) or not isinstance(dataset, dict):
        raise TranslationError("native_tta_config_missing", "TTA eval configuration lacks adaptation settings or dataset point counts")
    if contact.get("network_type") != "cm_net" or not isinstance(contact.get("model"), dict) or not isinstance(contact.get("dataset"), dict) or tta["contact_net"] != {"type": "cm_net"}:
        raise TranslationError("native_tta_config_invalid", "TTA ContactNet configuration must describe the pinned cm_net stage")
    network = contact["model"].get("network")
    if not isinstance(network, dict):
        raise TranslationError("native_tta_config_missing", "ContactNet configuration lacks its network settings")
    integer_settings = [(key, tta[key]) for key in ("batch_size", "iterations")]
    for role, point_config in (("tta", dataset), ("contact", contact["dataset"])):
        integer_settings.extend((f"{role}.{key}", point_config.get(key)) for key in ("num_obj_points", "num_hand_points"))
    integer_settings.append(("contact.out_channel", network.get("out_channel")))
    for key, value in integer_settings:
        if type(value) is not int or value <= 0:
            raise TranslationError("native_tta_config_invalid", f"TTA {key} must be a positive integer")
    for key in required - {"contact_net", "batch_size", "iterations"}:
        value = tta[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 or (key == "normalize_factor" and value == 0):
            raise TranslationError("native_tta_config_invalid", f"TTA {key} must be a finite nonnegative setting (normalization positive)")


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
    config_roles = checkpoint_roles | {"tta"} if manifest.pose_field == "tta_hand_pose" else checkpoint_roles
    if not isinstance(configs, dict) or set(configs) != config_roles:
        raise TranslationError("native_config_missing", "UniDex report must bind every selected-stage model and adaptation configuration")
    required_hashes.extend(configs.values())
    if any(not isinstance(value, str) or re.fullmatch(SHA256_PATTERN, value) is None for value in required_hashes):
        raise TranslationError("native_checkpoint_missing", "UniDex checkpoint hashes must be SHA-256 digests")
    tta_applied = manifest.pose_field == "tta_hand_pose"
    if report.get("tta_applied") is not tta_applied:
        raise TranslationError("native_report_mismatch", "selected pose field disagrees with reported TTA status")
    variant = "ipdf_plus_glow_with_tta" if tta_applied else "ipdf_plus_glow_without_tta"
    if report.get("sampling_variant") != variant:
        raise TranslationError("native_report_mismatch", "selected pose field disagrees with reported sampling variant")
    if tta_applied:
        _validate_tta_configuration(report)
    if request.seed + request.sample_count - 1 > 2**32 - 1:
        raise TranslationError("native_seed_invalid", "sample seeds exceed unsigned 32-bit range")
    with np.load(output, allow_pickle=False) as arrays:
        if set(arrays.files) - {"hand_pose", "tta_hand_pose"} or manifest.pose_field not in arrays.files:
            raise TranslationError("native_output_schema_invalid", "UniDex output must contain only declared numeric hand poses")
        poses = _numeric(arrays[manifest.pose_field], (request.sample_count, 28), "batch hand poses")
    return manifest, report, poses, output, report_file, source


def _validate_target_joint_limits(joints: dict[str, float], validator) -> None:
    """Reject unrepresentable targets before export using execution's tolerance."""
    names = validator.joint_names
    if set(joints) != set(names):
        raise TranslationError("shadow_joint_set_mismatch", "translated joint names differ from the target hand")
    values = _numeric([joints[name] for name in names], (len(names),), "target joints")
    limited = validator.model.jnt_limited.astype(bool)
    lower, upper = validator.model.jnt_range.T
    outside = limited & ((values < lower - 1e-9) | (values > upper + 1e-9))
    if np.any(outside):
        rejected = ", ".join(names[index] for index in np.flatnonzero(outside))
        raise TranslationError("shadow_joint_limit_violation", f"translated targets violate Shadow joint limits: {rejected}")


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
            try:
                original, _, _, _ = read_unidexgrasp_input(binding_path, request)
            except (ValueError, KeyError, TranslationError) as exc:
                raise TranslationError("native_input_binding_mismatch", "caller physical input contract is invalid") from exc
            for key, expected in {
                "schema_version": "unidexgrasp_object_input_v1",
                "point_cloud_sha256": report["point_cloud_sha256"],
                "object_to_input": manifest.object_to_input.model_dump(mode="json"),
                "object_scale": manifest.object_scale,
            }.items():
                if original.get(key) != expected:
                    raise TranslationError("native_input_binding_mismatch", f"caller input {key} differs from inference export")
        # Complete all decoding/translation/FK gates before writing proposals.
        decoded = []
        from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
        from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
        target_validator = ShadowPoseValidator(default_dex_urdf_root())
        with tempfile.TemporaryDirectory(prefix="unidexgrasp-kinematics-") as temp:
            for index, values in enumerate(poses):
                native = decode_unidexgrasp_hand_pose(
                    values, object_to_input=manifest.object_to_input,
                    source_hand_mjcf=source, source_hand_sha256=manifest.source_hand_mjcf_sha256,
                    kinematics_output=Path(temp) / f"source_{index}.xml",
                )
                translated = translate_native_pose(native, self.spec.translation_strategy_id)
                _validate_target_joint_limits(translated.joint_targets_rad, target_validator)
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
                    "sampling_variant": report["sampling_variant"], "tta_applied": report["tta_applied"],
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
