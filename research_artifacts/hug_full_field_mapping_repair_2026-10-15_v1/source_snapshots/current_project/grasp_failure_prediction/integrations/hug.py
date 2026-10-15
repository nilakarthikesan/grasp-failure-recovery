"""Validate and normalize official HUG grasp-prediction outputs.

HUG (Human Universal Grasping) is a flow-matching model that predicts MANO hand
grasps from RGB-D input. Its released inference/visualization code writes one
prediction dictionary per input to ``grasp_pred/<stem>.pkl``. Training code and
the HUG-Bench simulation benchmark are not released as of this writing.

This module is the *only* place the standalone project touches HUG data, and it
does so without importing the CUDA-only HUG package. It takes a plain Python
dictionary (already unpickled by the caller), checks that the grasp fields we
need are present and shaped correctly, and returns a clean, validated
``HugGraspPrediction``.

The expected fields mirror HUG's ``Grasp`` and ``CameraIntrinsics`` data classes:

    pose            MANO axis-angle pose            (15, 3)   [batch dim optional]
    pose_6d         MANO 6D-rotation pose           (15, 6)   optional
    shape           MANO shape (betas)              (10,)
    R_6d            global orientation, 6D           (6,)     optional
    t               global translation              (3,)
    T_camera_wrist  wrist-to-camera transform       (4, 4)
    landmarks_3d    3D hand keypoints                (21, 3)
    landmarks_2d    2D hand keypoints                (21, 2)  optional
    mesh_vertices   MANO mesh vertices               (778, 3)
    mesh_faces      MANO mesh faces                  (1552, 3) optional
    camera_K        camera intrinsic matrix          (3, 3)   optional
    camera_width    image width in pixels            int      optional
    camera_height   image height in pixels           int      optional

A leading batch dimension of size 1 (e.g. ``(1, 15, 3)``) is accepted and
squeezed away so downstream code sees a single grasp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class HugPredictionError(ValueError):
    """Raised when a HUG prediction dictionary is missing or malformed."""


# name -> (expected trailing shape, required)
_ARRAY_FIELDS: dict[str, tuple[tuple[int, ...], bool]] = {
    "pose": ((15, 3), True),
    "pose_6d": ((15, 6), False),
    "shape": ((10,), True),
    "R_6d": ((6,), False),
    "t": ((3,), True),
    "T_camera_wrist": ((4, 4), True),
    "landmarks_3d": ((21, 3), True),
    "landmarks_2d": ((21, 2), False),
    "mesh_vertices": ((778, 3), True),
    "mesh_faces": ((1552, 3), False),
    "camera_K": ((3, 3), False),
}


@dataclass
class HugGraspPrediction:
    """A single validated HUG grasp prediction.

    Optional fields are ``None`` when the source dictionary did not include them.
    All array fields are ``float`` (or ``int`` for faces) numpy arrays with the
    batch dimension removed.
    """

    pose: np.ndarray
    shape: np.ndarray
    t: np.ndarray
    T_camera_wrist: np.ndarray
    landmarks_3d: np.ndarray
    mesh_vertices: np.ndarray
    object_name: str | None = None
    pose_6d: np.ndarray | None = None
    R_6d: np.ndarray | None = None
    landmarks_2d: np.ndarray | None = None
    mesh_faces: np.ndarray | None = None
    camera_K: np.ndarray | None = None
    camera_width: int | None = None
    camera_height: int | None = None
    extra_keys: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """Return a short human-readable description of the usable grasp fields."""
        lines: list[str] = []
        name = self.object_name if self.object_name is not None else "<unknown>"
        lines.append(f"object_name: {name}")
        lines.append(f"pose (MANO axis-angle): shape {self.pose.shape}")
        lines.append(f"shape (MANO betas): shape {self.shape.shape}")
        lines.append(
            f"t (global translation): {np.array2string(self.t, precision=4)}"
        )
        lines.append(f"T_camera_wrist: shape {self.T_camera_wrist.shape}")
        lines.append(f"landmarks_3d: shape {self.landmarks_3d.shape}")
        lines.append(f"mesh_vertices: shape {self.mesh_vertices.shape}")
        if self.pose_6d is not None:
            lines.append(f"pose_6d: shape {self.pose_6d.shape}")
        if self.R_6d is not None:
            lines.append(f"R_6d (global orientation): shape {self.R_6d.shape}")
        if self.landmarks_2d is not None:
            lines.append(f"landmarks_2d: shape {self.landmarks_2d.shape}")
        if self.mesh_faces is not None:
            lines.append(f"mesh_faces: shape {self.mesh_faces.shape}")
        if self.camera_K is not None:
            lines.append(f"camera_K: shape {self.camera_K.shape}")
        if self.camera_width is not None and self.camera_height is not None:
            lines.append(
                f"camera image size: {self.camera_width} x {self.camera_height}"
            )
        if self.extra_keys:
            lines.append(f"other keys present (ignored): {', '.join(self.extra_keys)}")
        return "\n".join(lines)


def _coerce_array(name: str, value: Any, trailing: tuple[int, ...]) -> np.ndarray:
    """Convert ``value`` to a numpy array and validate its trailing shape.

    Accepts an optional leading batch dimension of size 1, which is squeezed.
    """
    try:
        array = np.asarray(value, dtype=np.int64 if name == "mesh_faces" else np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise HugPredictionError(
            f"field {name!r} could not be converted to a numeric array: {exc}"
        ) from exc

    # Squeeze a single leading batch dimension of size 1, e.g. (1, 15, 3).
    if array.ndim == len(trailing) + 1 and array.shape[0] == 1:
        array = array[0]

    if array.shape != trailing:
        raise HugPredictionError(
            f"field {name!r} has shape {array.shape}, expected {trailing} "
            "(a single leading batch dimension of size 1 is allowed)"
        )
    if name != "mesh_faces" and not np.all(np.isfinite(array)):
        raise HugPredictionError(f"field {name!r} contains non-finite values")
    return array


def normalize_hug_prediction(prediction: Mapping[str, Any]) -> HugGraspPrediction:
    """Validate a HUG prediction mapping and return a normalized grasp.

    Parameters
    ----------
    prediction:
        A mapping already unpickled from an official HUG ``grasp_pred/*.pkl``
        file, or an equivalent dictionary. This function never unpickles data
        itself and never imports the HUG package.

    Raises
    ------
    HugPredictionError
        If required fields are missing or any field has the wrong shape.
    """
    if not isinstance(prediction, Mapping):
        raise HugPredictionError(
            f"expected a dict-like HUG prediction, got {type(prediction).__name__}"
        )

    arrays: dict[str, np.ndarray | None] = {}
    for name, (trailing, required) in _ARRAY_FIELDS.items():
        if name in prediction and prediction[name] is not None:
            arrays[name] = _coerce_array(name, prediction[name], trailing)
        elif required:
            raise HugPredictionError(f"missing required field {name!r}")
        else:
            arrays[name] = None

    camera_width = prediction.get("camera_width")
    camera_height = prediction.get("camera_height")
    object_name = prediction.get("object_name")

    known = set(_ARRAY_FIELDS) | {
        "camera_width",
        "camera_height",
        "object_name",
    }
    extra_keys = tuple(sorted(k for k in prediction if k not in known))

    return HugGraspPrediction(
        pose=arrays["pose"],
        shape=arrays["shape"],
        t=arrays["t"],
        T_camera_wrist=arrays["T_camera_wrist"],
        landmarks_3d=arrays["landmarks_3d"],
        mesh_vertices=arrays["mesh_vertices"],
        object_name=str(object_name) if object_name is not None else None,
        pose_6d=arrays["pose_6d"],
        R_6d=arrays["R_6d"],
        landmarks_2d=arrays["landmarks_2d"],
        mesh_faces=arrays["mesh_faces"],
        camera_K=arrays["camera_K"],
        camera_width=int(camera_width) if camera_width is not None else None,
        camera_height=int(camera_height) if camera_height is not None else None,
        extra_keys=extra_keys,
    )


def load_hug_prediction(path: str | Path) -> HugGraspPrediction:
    """Load and validate a HUG ``grasp_pred/*.pkl`` file from disk.

    The pickle is produced by the user's own HUG inference run. Only call this
    on files you trust: Python unpickling can execute arbitrary code, so we do
    not exercise this path in the test suite.
    """
    import pickle

    path = Path(path)
    if not path.is_file():
        raise HugPredictionError(f"no such HUG prediction file: {path}")

    with path.open("rb") as handle:
        raw = pickle.load(handle)  # noqa: S301 - user-supplied, trusted input

    # Some HUG exports wrap the grasp in a container or list of grasps.
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise HugPredictionError(f"HUG prediction file {path} is empty")
        raw = raw[0]

    return normalize_hug_prediction(raw)


def main() -> None:
    """CLI: print the usable grasp fields from a HUG prediction pickle."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Inspect an official HUG grasp_pred/*.pkl output. This does not "
            "import or run HUG; it only reads a prediction the user already "
            "generated in HUG's own CUDA environment."
        )
    )
    parser.add_argument(
        "prediction",
        type=Path,
        help="Path to a HUG grasp_pred/<stem>.pkl file.",
    )
    args = parser.parse_args()

    grasp = load_hug_prediction(args.prediction)
    print(f"Loaded HUG prediction: {args.prediction}")
    print(grasp.summary())


if __name__ == "__main__":
    main()
