"""Pinned DexGraspNet saved-record import, decoding, and FK verification."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path
import shutil
import subprocess
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import yaml

from grasp_failure_prediction.evaluation.object_assets import load_object_geometry
from grasp_failure_prediction.evaluation.pose_validation import (
    ShadowPoseValidator,
    _body_id,
    _link_positions,
)
from grasp_failure_prediction.evaluation.retargeting import (
    RetargetedHandPose,
    default_dex_urdf_root,
)
from grasp_failure_prediction.evaluation.schema import canonical_json

from .schema import RigidTransform
from .translation import NativeHandPose, TranslatedHandPose, TranslationError


SOURCE_JOINT_NAMES = (
    "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
    "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
    "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
    "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
    "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0",
)
TRANSLATION_NAMES = ("WRJTx", "WRJTy", "WRJTz")
ROTATION_NAMES = ("WRJRx", "WRJRy", "WRJRz")
SOURCE_TIP_SITES = (
    "robot0:S_thtip", "robot0:S_fftip", "robot0:S_mftip",
    "robot0:S_rftip", "robot0:S_lftip",
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog() -> dict:
    resource = files(__package__).joinpath("configs", "dexgraspnet_samples.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload.get("schema_version") != "dexgraspnet_sample_catalog_v1":
        raise ValueError("DexGraspNet sample catalog has the wrong schema")
    return payload


def _relative_child(root: Path, relative: str, role: str) -> Path:
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts:
        raise ValueError(f"{role} must be a relative path")
    path = (root / item).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{role} escapes its bundle")
    return path


def _euler_xyz_matrix(angles: tuple[float, float, float]) -> np.ndarray:
    """Match transforms3d.euler.euler2mat(*angles), whose default is static XYZ."""

    x, y, z = angles
    sx, sy, sz = np.sin((x, y, z))
    cx, cy, cz = np.cos((x, y, z))
    return np.asarray([
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy, sx * cy, cx * cy],
    ])


def _quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quaternion_wxyz = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion_wxyz, rotation.reshape(9))
    quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
    if quaternion_xyzw[3] < 0:
        quaternion_xyzw *= -1
    return tuple(float(value) for value in quaternion_xyzw)


class DexGraspNetRecordDecoder:
    """Decode one hash-pinned record from the official bundled sample dataset."""

    schema_version = "dexgraspnet_replay_bundle_v1"

    def decode(self, manifest_path: Path) -> NativeHandPose:
        manifest_path = Path(manifest_path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.schema_version:
            raise TranslationError("native_schema_unsupported", "DexGraspNet replay bundle has the wrong schema")
        catalog = _catalog()
        key = payload.get("catalog_object_key")
        entry = catalog["objects"].get(key)
        if entry is None:
            raise TranslationError("dataset_object_unregistered", "DexGraspNet object is not in the pinned sample catalog")
        if payload.get("source_commit") != catalog["source_commit"]:
            raise TranslationError("dataset_revision_mismatch", "DexGraspNet source revision does not match the catalog")
        record_path = _relative_child(manifest_path.parent, payload["record_path"], "record path")
        if _sha256(record_path) != entry["dataset_sha256"] or payload.get("record_sha256") != entry["dataset_sha256"]:
            raise TranslationError("dataset_hash_mismatch", "DexGraspNet record does not match the pinned official sample")
        # The official format is an object-array pickle. It is opened only after its
        # complete file hash matches the pinned catalog entry above.
        records = np.load(record_path, allow_pickle=True)
        index = payload.get("record_index")
        if type(index) is not int or not 0 <= index < len(records):
            raise TranslationError("dataset_record_index_invalid", "DexGraspNet record index is outside the dataset")
        record = records[index]
        if hasattr(record, "item"):
            record = record.item()
        if not isinstance(record, dict) or set(record) != {"qpos", "scale"}:
            raise TranslationError("dataset_record_schema_invalid", "DexGraspNet record fields are unsupported")
        qpos = record["qpos"]
        expected = set(SOURCE_JOINT_NAMES + TRANSLATION_NAMES + ROTATION_NAMES)
        if not isinstance(qpos, dict) or set(qpos) != expected:
            raise TranslationError("dataset_joint_set_mismatch", "DexGraspNet qpos fields do not match the pinned format")
        numeric = {name: float(qpos[name]) for name in expected}
        if not all(np.isfinite(value) for value in numeric.values()):
            raise TranslationError("dataset_nonfinite_pose", "DexGraspNet pose contains non-finite values")
        scale = float(record["scale"])
        if not np.isfinite(scale) or scale <= 0 or not np.isclose(scale, payload.get("object_scale"), atol=1e-12):
            raise TranslationError("dataset_scale_mismatch", "DexGraspNet object scale does not match the imported geometry")
        rotation = _euler_xyz_matrix(tuple(numeric[name] for name in ROTATION_NAMES))
        return NativeHandPose(
            source_hand_model_id="dexgraspnet_shadow_mjcf_bd1b13d",
            source_root_link="robot0:palm",
            object_to_root=RigidTransform(
                translation_m=tuple(numeric[name] for name in TRANSLATION_NAMES),
                quaternion_xyzw=_quaternion_xyzw(rotation),
            ),
            joint_targets_rad={name: numeric[name] for name in SOURCE_JOINT_NAMES},
            payload={
                "record_index": index,
                "record_sha256": entry["dataset_sha256"],
                "object_scale": scale,
                "source_commit": catalog["source_commit"],
                "source_kinematics_path": str(_relative_child(
                    manifest_path.parent, payload["source_kinematics_path"], "source kinematics"
                )),
                "source_kinematics_sha256": payload["source_kinematics_sha256"],
            },
        )


def _source_tip_vectors(native: NativeHandPose) -> np.ndarray:
    path = Path(native.payload["source_kinematics_path"])
    if _sha256(path) != native.payload["source_kinematics_sha256"]:
        raise TranslationError("source_kinematics_hash_mismatch", "derived source kinematics changed after import")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index))
        for index in range(model.njnt)
    )
    if names != SOURCE_JOINT_NAMES:
        raise TranslationError("source_kinematics_joint_mismatch", "source kinematics has an unexpected joint order")
    data.qpos[:] = [native.joint_targets_rad[name] for name in names]
    mujoco.mj_forward(model, data)
    palm = _body_id(model, "robot0:palm")
    rotation = data.xmat[palm].reshape(3, 3)
    return np.asarray([
        rotation.T @ (data.site(name).xpos - data.xpos[palm])
        for name in SOURCE_TIP_SITES
    ])


def validate_fk_translation(
    native: NativeHandPose, translated: TranslatedHandPose, *, maximum_error_m: float
) -> dict[str, float | int | str | bool]:
    source_vectors = _source_tip_vectors(native)
    validator = ShadowPoseValidator(default_dex_urdf_root())
    pose = RetargetedHandPose(
        joint_names=tuple(translated.joint_targets_rad),
        qpos=np.asarray(tuple(translated.joint_targets_rad.values()), dtype=np.float64),
        landmarks_wrist_m=np.zeros((21, 3)),
        reference_vectors_m=np.zeros((10, 3)),
        T_camera_wrist=np.eye(4),
    )
    validator.set_pose(pose)
    palm = _body_id(validator.model, "palm")
    rotation = validator.data.xmat[palm].reshape(3, 3)
    target_vectors = np.asarray([
        rotation.T @ (position - validator.data.xpos[palm])
        for position in _link_positions(validator.model, validator.data, validator.source_urdf)[:5]
    ])
    errors = np.linalg.norm(source_vectors - target_vectors, axis=1)
    maximum = float(np.max(errors))
    if maximum > maximum_error_m:
        raise TranslationError(
            "forward_kinematics_residual_exceeded",
            f"DexGraspNet mapping fingertip residual {maximum:.6f} m exceeds {maximum_error_m:.6f} m",
        )
    return {
        "mean_fingertip_error_m": float(np.mean(errors)),
        "maximum_fingertip_error_m": maximum,
        "fk_comparison_joint_count": len(SOURCE_JOINT_NAMES),
    }


def _derive_kinematic_mjcf(source: Path, destination: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    option, size = root.find("option"), root.find("size")
    if option is not None:
        option.attrib.pop("apirate", None)
    if size is not None:
        size.attrib.pop("nstack", None)
    for tag in ("asset", "contact", "equality", "tendon", "actuator", "sensor"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom":
                parent.remove(child)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(destination))
    names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index))
        for index in range(model.njnt)
    )
    if names != SOURCE_JOINT_NAMES:
        raise ValueError("derived DexGraspNet kinematic model has an unexpected joint inventory")


def _object_mjcf(mesh: Path, destination: Path, scale: float, base_mass_kg: float) -> None:
    mesh_target = destination.parent / "decomposed.obj"
    shutil.copy2(mesh, mesh_target)
    provisional = f'''<mujoco model="dexgraspnet_imported_object">
  <compiler angle="radian" meshdir="." strippath="false"/>
  <asset><mesh name="object_mesh" file="decomposed.obj" scale="{scale:.17g} {scale:.17g} {scale:.17g}"/></asset>
  <worldbody><body name="object"><freejoint name="object_free"/><geom name="object_geom" type="mesh" mesh="object_mesh" density="1000"/></body></worldbody>
</mujoco>'''
    temp = destination.with_suffix(".provisional.xml")
    temp.write_text(provisional, encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(temp))
    finally:
        temp.unlink(missing_ok=True)
    source_mass = float(model.body_mass[1])
    inertia = model.body_inertia[1] * (base_mass_kg / source_mass)
    ipos, iquat = model.body_ipos[1], model.body_iquat[1]
    final = f'''<mujoco model="dexgraspnet_imported_object">
  <compiler angle="radian" meshdir="." strippath="false"/>
  <asset><mesh name="object_mesh" file="decomposed.obj" scale="{scale:.17g} {scale:.17g} {scale:.17g}"/></asset>
  <worldbody><body name="object">
    <inertial pos="{' '.join(map(str, ipos))}" quat="{' '.join(map(str, iquat))}" mass="{base_mass_kg:.17g}" diaginertia="{' '.join(map(str, inertia))}"/>
    <freejoint name="object_free"/>
    <geom name="object_geom" type="mesh" mesh="object_mesh" friction="1 0.005 0.0001"/>
  </body></worldbody>
</mujoco>'''
    destination.write_text(final, encoding="utf-8")


def import_official_sample(
    source_root: str | Path,
    output_directory: str | Path,
    *,
    project_root: str | Path,
    object_key: str = "core_mug_8570d9a8d24cb0acbebd3c0c0c70fb03",
    record_index: int = 0,
    base_mass_kg: float = 0.18,
) -> dict:
    source = Path(source_root).resolve()
    output = Path(output_directory).resolve()
    project = Path(project_root).resolve()
    if not output.is_relative_to(project):
        raise ValueError("DexGraspNet import output must remain inside the project root")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"DexGraspNet import output is not empty: {output}")
    catalog = _catalog()
    entry = catalog["objects"].get(object_key)
    if entry is None:
        raise ValueError(f"unknown pinned DexGraspNet sample: {object_key}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != catalog["source_commit"]:
        raise ValueError(f"DexGraspNet checkout must be pinned to {catalog['source_commit']}")
    record_source = source / entry["dataset_path"]
    mesh_source = source / entry["mesh_path"]
    hand_source = source / catalog["source_hand_mjcf"]["path"]
    for path, expected, role in (
        (record_source, entry["dataset_sha256"], "dataset"),
        (mesh_source, entry["mesh_sha256"], "object mesh"),
        (hand_source, catalog["source_hand_mjcf"]["sha256"], "source hand"),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"official DexGraspNet {role} hash mismatch: {path}")
    records = np.load(record_source, allow_pickle=True)
    if not 0 <= record_index < len(records):
        raise ValueError("DexGraspNet record index is outside the selected dataset")
    record = records[record_index]
    if hasattr(record, "item"):
        record = record.item()
    scale = float(record["scale"])
    output.mkdir(parents=True)
    native = output / "native"
    native.mkdir()
    record_path = native / "dataset.npy"
    shutil.copy2(record_source, record_path)
    kinematics_path = native / "shadow_hand_kinematics.xml"
    _derive_kinematic_mjcf(hand_source, kinematics_path)
    object_path = output / "object" / "object.xml"
    object_path.parent.mkdir()
    _object_mjcf(mesh_source, object_path, scale, base_mass_kg)
    geometry = load_object_geometry(object_path)
    bundle = {
        "schema_version": "dexgraspnet_replay_bundle_v1",
        "catalog_object_key": object_key,
        "object_id": entry["object_id"],
        "geometry_content_hash": geometry.content_hash,
        "source_repository": catalog["source_repository"],
        "source_commit": catalog["source_commit"],
        "license": catalog["license"],
        "record_path": "native/dataset.npy",
        "record_sha256": entry["dataset_sha256"],
        "record_index": record_index,
        "object_scale": scale,
        "source_mesh_sha256": entry["mesh_sha256"],
        "source_hand_mjcf_sha256": catalog["source_hand_mjcf"]["sha256"],
        "source_kinematics_path": "native/shadow_hand_kinematics.xml",
        "source_kinematics_sha256": _sha256(kinematics_path),
    }
    bundle_path = output / "replay_bundle.json"
    bundle_path.write_bytes(canonical_json(bundle) + b"\n")
    relative_bundle = bundle_path.relative_to(project)
    relative_object = object_path.relative_to(project)
    request = {
        "schema_version": "proposal_request_v1",
        "request_id": f"dexgraspnet_{object_key}_record_{record_index:04d}",
        "provider": {"id": "dexgraspnet_saved_record_v1"},
        "object": {"id": entry["object_id"], "geometry_content_hash": geometry.content_hash},
        "embodiment_id": "shadow_hand_right",
        "provider_input_path": relative_bundle.as_posix(),
        "seed": 0,
        "sample_count": 1,
    }
    request_path = output / "request.yaml"
    request_path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")
    object_z = max(0.001, -geometry.local_collision_bounds_m[0][2] + 0.001)
    experiment = {
        "schema_version": "grasp_eval_experiment_v1",
        "experiment_id": f"dexgraspnet_{object_key}_record_{record_index:04d}",
        "output_root": f"runs/dexgraspnet_{object_key}_record_{record_index:04d}",
        "record_video": True,
        "verify_native_artifacts": True,
        "jobs": [{
            "id": "dexgraspnet_replay",
            "request": request_path.relative_to(project).as_posix(),
            "evaluations": [{
                "case_id": f"dexgraspnet_{object_key}_record_{record_index:04d}",
                "proposal_sample_index": 0,
                "environment": {"id": "adroit_shadow_tabletop_v2"},
                "embodiment": {"id": "shadow_hand_right"},
                "execution_protocol": {"id": "fixed_grasp_lift_v2"},
                "object": {
                    "id": entry["object_id"], "mass_kg": base_mass_kg,
                    "friction_profile_id": "nominal_v1",
                    "geometry": {"mjcf_path": relative_object.as_posix(), "expected_content_hash": geometry.content_hash},
                },
                "initial_condition": {
                    "id": "dataset_object_frame_upright",
                    "object_position_m": [0.0, 0.0, object_z],
                    "object_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "motion_profile_id": "nominal_lift_v1", "seed": 0,
            }],
        }],
    }
    experiment_path = output / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
    return {
        "source_commit": revision,
        "license": catalog["license"],
        "record_index": record_index,
        "object_scale": scale,
        "geometry_content_hash": geometry.content_hash,
        "bundle": relative_bundle.as_posix(),
        "request": request_path.relative_to(project).as_posix(),
        "experiment": experiment_path.relative_to(project).as_posix(),
    }
