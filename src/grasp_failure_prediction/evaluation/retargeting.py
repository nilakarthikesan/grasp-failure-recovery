"""Thin HUG-to-Shadow-Hand integration around Dex Retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from grasp_failure_prediction.integrations.hug import HugGraspPrediction


class RetargetingError(ValueError):
    """Raised when coordinate conversion or joint mapping is invalid."""


@dataclass(frozen=True)
class RetargetedHandPose:
    """One Dex Retargeting result with enough context for validation."""

    joint_names: tuple[str, ...]
    qpos: np.ndarray
    landmarks_wrist_m: np.ndarray
    reference_vectors_m: np.ndarray
    T_camera_wrist: np.ndarray


def _validate_rigid_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise RetargetingError("T_camera_wrist must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise RetargetingError("T_camera_wrist must have homogeneous last row [0,0,0,1]")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise RetargetingError("T_camera_wrist rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise RetargetingError("T_camera_wrist rotation must be right-handed")
    return value


def camera_landmarks_to_wrist(
    landmarks_camera_m: np.ndarray, T_camera_wrist: np.ndarray
) -> np.ndarray:
    """Convert HUG camera-frame landmarks into its local MANO wrist frame.

    HUG defines ``T_camera_wrist`` as the wrist-to-camera transform and exports
    metric landmarks in camera coordinates. Applying its inverse removes the
    predicted global wrist rotation and translation, which is the local MANO
    convention expected by Dex Retargeting's vector optimizer.
    """

    landmarks = np.asarray(landmarks_camera_m, dtype=np.float64)
    if landmarks.shape != (21, 3) or not np.all(np.isfinite(landmarks)):
        raise RetargetingError("HUG landmarks must be a finite 21x3 array")
    transform = _validate_rigid_transform(T_camera_wrist)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (landmarks - translation) @ rotation


def reference_vectors(
    landmarks_wrist_m: np.ndarray, human_indices: np.ndarray
) -> np.ndarray:
    """Build the origin-to-task vectors required by Dex vector retargeting."""

    indices = np.asarray(human_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[0] != 2:
        raise RetargetingError("Dex human landmark indices must have shape (2, N)")
    if np.any(indices < 0) or np.any(indices >= len(landmarks_wrist_m)):
        raise RetargetingError("Dex human landmark index is outside the HUG landmark array")
    return landmarks_wrist_m[indices[1]] - landmarks_wrist_m[indices[0]]


def reorder_for_mujoco(
    dex_joint_names: Sequence[str],
    dex_qpos: np.ndarray,
    mujoco_joint_names: Sequence[str],
) -> np.ndarray:
    """Reorder a Dex result using exact, explicit MuJoCo joint names."""

    values = np.asarray(dex_qpos, dtype=np.float64)
    if values.shape != (len(dex_joint_names),):
        raise RetargetingError("Dex qpos length does not match its joint-name list")
    by_mujoco_name = {
        name: float(value)
        for name, value in zip(dex_joint_names, values, strict=True)
    }
    missing = [name for name in mujoco_joint_names if name not in by_mujoco_name]
    if missing:
        raise RetargetingError(f"MuJoCo joints missing from Dex result: {missing}")
    return np.asarray([by_mujoco_name[name] for name in mujoco_joint_names])


def default_dex_urdf_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "third_party" / "dex-urdf" / "robots" / "hands"


class ShadowHandRetargeter:
    """Load the official Shadow config and retarget one HUG prediction."""

    def __init__(self, urdf_root: str | Path | None = None) -> None:
        try:
            from dex_retargeting.constants import (
                HandType,
                RetargetingType,
                RobotName,
                get_default_config_path,
            )
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ImportError as exc:  # pragma: no cover - dependency failure
            raise RetargetingError(
                'Dex Retargeting is unavailable; install the "eval" extra'
            ) from exc

        root = Path(urdf_root) if urdf_root is not None else default_dex_urdf_root()
        required_urdf = root / "shadow_hand" / "shadow_hand_right.urdf"
        if not required_urdf.is_file():
            raise RetargetingError(
                f"missing Shadow URDF at {required_urdf}; run "
                "`git submodule update --init --recursive`"
            )
        RetargetingConfig.set_default_urdf_dir(root.resolve())
        config_path = get_default_config_path(
            RobotName.shadow, RetargetingType.vector, HandType.right
        )
        self.config_path = Path(config_path)
        self._retargeting = RetargetingConfig.load_from_file(config_path).build()

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self._retargeting.joint_names)

    @property
    def joint_limits(self) -> np.ndarray:
        return np.asarray(self._retargeting.joint_limits, dtype=np.float64).copy()

    def retarget(self, grasp: HugGraspPrediction) -> RetargetedHandPose:
        landmarks = camera_landmarks_to_wrist(
            grasp.landmarks_3d, grasp.T_camera_wrist
        )
        indices = self._retargeting.optimizer.target_link_human_indices
        vectors = reference_vectors(landmarks, indices)
        qpos = np.asarray(self._retargeting.retarget(vectors), dtype=np.float64)
        if qpos.shape != (len(self.joint_names),) or not np.all(np.isfinite(qpos)):
            raise RetargetingError("Dex Retargeting returned an invalid joint vector")
        limits = self.joint_limits
        # Dex Retargeting deliberately relaxes optimizer bounds by 1e-3 rad.
        # Accept only that documented numerical margin, then clamp to the
        # physical URDF limits before exposing the pose to MuJoCo.
        optimizer_margin = 1.01e-3
        if np.any(qpos < limits[:, 0] - optimizer_margin) or np.any(
            qpos > limits[:, 1] + optimizer_margin
        ):
            raise RetargetingError("Dex Retargeting result violates Shadow joint limits")
        qpos = np.clip(qpos, limits[:, 0], limits[:, 1])
        return RetargetedHandPose(
            joint_names=self.joint_names,
            qpos=qpos,
            landmarks_wrist_m=landmarks,
            reference_vectors_m=vectors,
            T_camera_wrist=np.asarray(grasp.T_camera_wrist, dtype=np.float64).copy(),
        )
