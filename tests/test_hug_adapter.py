"""Tests for the HUG prediction adapter.

These tests build small *synthetic* HUG prediction dictionaries in memory. They
never load opaque or downloaded pickle data, since Python unpickling can execute
arbitrary code. The adapter under test also never imports the HUG package.
"""

from __future__ import annotations

import numpy as np
import pytest

from grasp_failure_prediction.integrations.hug import (
    HugGraspPrediction,
    HugPredictionError,
    normalize_hug_prediction,
)


def _synthetic_prediction(with_batch_dim: bool = False) -> dict:
    """Build a well-formed synthetic HUG prediction dictionary."""

    def shaped(*shape: int) -> np.ndarray:
        array = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        return array[None, ...] if with_batch_dim else array

    return {
        "object_name": "mug",
        "pose": shaped(15, 3),
        "pose_6d": shaped(15, 6),
        "shape": shaped(10),
        "R_6d": shaped(6),
        "t": shaped(3),
        "T_camera_wrist": np.eye(4) if not with_batch_dim else np.eye(4)[None, ...],
        "landmarks_3d": shaped(21, 3),
        "landmarks_2d": shaped(21, 2),
        "mesh_vertices": shaped(778, 3),
        "mesh_faces": np.zeros((1552, 3), dtype=np.int64),
        "camera_K": np.eye(3),
        "camera_width": 640,
        "camera_height": 480,
    }


def test_normalize_full_prediction() -> None:
    grasp = normalize_hug_prediction(_synthetic_prediction())

    assert isinstance(grasp, HugGraspPrediction)
    assert grasp.object_name == "mug"
    assert grasp.pose.shape == (15, 3)
    assert grasp.shape.shape == (10,)
    assert grasp.t.shape == (3,)
    assert grasp.T_camera_wrist.shape == (4, 4)
    assert grasp.landmarks_3d.shape == (21, 3)
    assert grasp.mesh_vertices.shape == (778, 3)
    assert grasp.pose_6d.shape == (15, 6)
    assert grasp.R_6d.shape == (6,)
    assert grasp.landmarks_2d.shape == (21, 2)
    assert grasp.mesh_faces.shape == (1552, 3)
    assert grasp.mesh_faces.dtype == np.int64
    assert grasp.camera_K.shape == (3, 3)
    assert grasp.camera_width == 640
    assert grasp.camera_height == 480
    assert grasp.extra_keys == ()


def test_batch_dimension_is_squeezed() -> None:
    grasp = normalize_hug_prediction(_synthetic_prediction(with_batch_dim=True))
    assert grasp.pose.shape == (15, 3)
    assert grasp.t.shape == (3,)
    assert grasp.T_camera_wrist.shape == (4, 4)


def test_optional_fields_default_to_none() -> None:
    prediction = _synthetic_prediction()
    for optional in ("pose_6d", "R_6d", "landmarks_2d", "mesh_faces", "camera_K"):
        del prediction[optional]
    del prediction["camera_width"]
    del prediction["camera_height"]

    grasp = normalize_hug_prediction(prediction)
    assert grasp.pose_6d is None
    assert grasp.R_6d is None
    assert grasp.landmarks_2d is None
    assert grasp.mesh_faces is None
    assert grasp.camera_K is None
    assert grasp.camera_width is None
    assert grasp.camera_height is None


def test_missing_required_field_raises() -> None:
    prediction = _synthetic_prediction()
    del prediction["pose"]
    with pytest.raises(HugPredictionError, match="pose"):
        normalize_hug_prediction(prediction)


def test_wrong_shape_raises() -> None:
    prediction = _synthetic_prediction()
    prediction["shape"] = np.zeros((7,))
    with pytest.raises(HugPredictionError, match="shape"):
        normalize_hug_prediction(prediction)


def test_non_finite_values_raise() -> None:
    prediction = _synthetic_prediction()
    prediction["t"] = np.array([np.nan, 0.0, 0.0])
    with pytest.raises(HugPredictionError, match="non-finite"):
        normalize_hug_prediction(prediction)


def test_non_mapping_input_raises() -> None:
    with pytest.raises(HugPredictionError, match="dict-like"):
        normalize_hug_prediction([1, 2, 3])  # type: ignore[arg-type]


def test_extra_keys_are_reported_not_rejected() -> None:
    prediction = _synthetic_prediction()
    prediction["confidence"] = 0.9
    prediction["frame_index"] = 3
    grasp = normalize_hug_prediction(prediction)
    assert grasp.extra_keys == ("confidence", "frame_index")


def test_summary_is_readable() -> None:
    grasp = normalize_hug_prediction(_synthetic_prediction())
    summary = grasp.summary()
    assert "object_name: mug" in summary
    assert "mesh_vertices" in summary
    assert "640 x 480" in summary
