"""Validate and display a Dex-retargeted Shadow Hand pose in MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Sequence
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from .retargeting import RetargetedHandPose, RetargetingError, reorder_for_mujoco


TASK_LINK_NAMES = (
    "thtip",
    "fftip",
    "mftip",
    "rftip",
    "lftip",
    "thmiddle",
    "ffmiddle",
    "mfmiddle",
    "rfmiddle",
    "lfmiddle",
)


@dataclass(frozen=True)
class PoseValidation:
    link_names: tuple[str, ...]
    target_vectors_m: np.ndarray
    achieved_vectors_m: np.ndarray
    errors_m: np.ndarray
    fingertip_errors_m: np.ndarray
    mean_fingertip_error_m: float
    maximum_fingertip_error_m: float
    mean_error_m: float
    maximum_error_m: float
    contact_pairs: tuple[tuple[str, str], ...]


def _shadow_urdf_path(urdf_root: str | Path) -> Path:
    path = Path(urdf_root) / "shadow_hand" / "shadow_hand_right.urdf"
    if not path.is_file():
        raise RetargetingError(f"missing official Shadow Hand URDF: {path}")
    return path.resolve()


def _mujoco_compatible_urdf(source: Path) -> Path:
    """Write a temporary URDF whose mesh paths survive MuJoCo compilation."""

    tree = ET.parse(source)
    root = tree.getroot()
    mujoco_extension = ET.Element("mujoco")
    ET.SubElement(mujoco_extension, "compiler", {"strippath": "false"})
    root.insert(0, mujoco_extension)
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", str((source.parent / filename).resolve()))
    handle = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False)
    handle.close()
    output = Path(handle.name)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def load_shadow_model(urdf_root: str | Path) -> mujoco.MjModel:
    """Compile the same official Shadow URDF used by Dex into MuJoCo."""

    temporary = _mujoco_compatible_urdf(_shadow_urdf_path(urdf_root))
    try:
        return mujoco.MjModel.from_xml_path(str(temporary))
    finally:
        temporary.unlink(missing_ok=True)


def _parse_xyz(value: str | None) -> np.ndarray:
    if value is None:
        return np.zeros(3)
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.shape != (3,):
        raise RetargetingError(f"invalid URDF xyz vector: {value}")
    return result


def _fixed_tip_offsets(source: Path) -> dict[str, tuple[str, np.ndarray]]:
    """Return parent-body and local offset for fixed fingertip links."""

    root = ET.parse(source).getroot()
    result: dict[str, tuple[str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        child_name = child.get("link")
        if child_name not in TASK_LINK_NAMES[:5]:
            continue
        if joint.get("type") != "fixed":
            raise RetargetingError(f"expected {child_name} to use a fixed URDF joint")
        origin = joint.find("origin")
        rpy = _parse_xyz(origin.get("rpy") if origin is not None else None)
        if not np.allclose(rpy, 0.0, atol=1e-12):
            raise RetargetingError(f"unsupported rotated fixed tip frame: {child_name}")
        result[child_name] = (
            str(parent.get("link")),
            _parse_xyz(origin.get("xyz") if origin is not None else None),
        )
    missing = set(TASK_LINK_NAMES[:5]) - set(result)
    if missing:
        raise RetargetingError(f"missing fingertip frames in Shadow URDF: {sorted(missing)}")
    return result


def _body_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise RetargetingError(f"MuJoCo model is missing body {name!r}")
    return value


def _link_positions(
    model: mujoco.MjModel, data: mujoco.MjData, source_urdf: Path
) -> np.ndarray:
    tips = _fixed_tip_offsets(source_urdf)
    positions: list[np.ndarray] = []
    for name in TASK_LINK_NAMES:
        if name in tips:
            parent_name, offset = tips[name]
            body_id = _body_id(model, parent_name)
            rotation = data.xmat[body_id].reshape(3, 3)
            positions.append(data.xpos[body_id] + rotation @ offset)
        else:
            positions.append(data.xpos[_body_id(model, name)].copy())
    return np.asarray(positions)


def _joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index))
        for index in range(model.njnt)
    )


def _contact_pairs(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        result.append((first or f"geom:{contact.geom1}", second or f"geom:{contact.geom2}"))
    return tuple(result)


class ShadowPoseValidator:
    """Forward a retargeted pose through MuJoCo and measure task-link error."""

    def __init__(self, urdf_root: str | Path) -> None:
        self.urdf_root = Path(urdf_root)
        self.source_urdf = _shadow_urdf_path(self.urdf_root)
        self.model = load_shadow_model(self.urdf_root)
        self.data = mujoco.MjData(self.model)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return _joint_names(self.model)

    def set_pose(self, pose: RetargetedHandPose) -> None:
        self.data.qpos[:] = reorder_for_mujoco(
            pose.joint_names, pose.qpos, self.joint_names
        )
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def validate(self, pose: RetargetedHandPose, scaling_factor: float = 1.2) -> PoseValidation:
        self.set_pose(pose)
        palm_position = self.data.xpos[_body_id(self.model, "palm")]
        achieved = _link_positions(self.model, self.data, self.source_urdf) - palm_position
        target = pose.reference_vectors_m * scaling_factor
        errors = np.linalg.norm(achieved - target, axis=1)
        fingertip_errors = errors[:5].copy()
        return PoseValidation(
            link_names=TASK_LINK_NAMES,
            target_vectors_m=target,
            achieved_vectors_m=achieved,
            errors_m=errors,
            fingertip_errors_m=fingertip_errors,
            mean_fingertip_error_m=float(np.mean(fingertip_errors)),
            maximum_fingertip_error_m=float(np.max(fingertip_errors)),
            mean_error_m=float(np.mean(errors)),
            maximum_error_m=float(np.max(errors)),
            contact_pairs=_contact_pairs(self.model, self.data),
        )

    def launch_viewer(self, pose: RetargetedHandPose) -> None:
        """Open an interactive native MuJoCo viewer for a static pose."""

        self.set_pose(pose)
        import mujoco.viewer

        mujoco.viewer.launch(self.model, self.data)


def format_validation(validation: PoseValidation) -> str:
    lines = ["Retargeted Shadow Hand pose validation:"]
    for name, error in zip(
        validation.link_names, validation.errors_m, strict=True
    ):
        lines.append(f"  {name:<8} alignment error: {error * 1000:7.2f} mm")
    lines.append(
        "  mean fingertip error: "
        f"{validation.mean_fingertip_error_m * 1000:.2f} mm"
    )
    lines.append(
        "  maximum fingertip error: "
        f"{validation.maximum_fingertip_error_m * 1000:.2f} mm"
    )
    lines.append(f"  mean all-link error: {validation.mean_error_m * 1000:.2f} mm")
    lines.append(f"  MuJoCo contacts at pose: {len(validation.contact_pairs)}")
    return "\n".join(lines)
