"""Explicit frame conversion helpers for the simulation integration pilot."""
import numpy as np

# Dex's right-hand OPERATOR2MANO_RIGHT; A maps MANO column vectors to operator.
MANO_TO_OPERATOR_RIGHT = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])


def convert_hand_frame(vectors_mano, T_world_mano, A=MANO_TO_OPERATOR_RIGHT):
    """Convert local vectors and their pose together, preserving world geometry.

    A maps MANO coordinates to the retargeting base coordinates. Translation
    still denotes the human wrist; choosing a robot origin is a separate step.
    """
    vectors = np.asarray(vectors_mano, dtype=float)
    transform = np.asarray(T_world_mano, dtype=float)
    A = np.asarray(A, dtype=float)
    if transform.shape != (4, 4) or A.shape != (3, 3):
        raise ValueError("Expected 4x4 transform and 3x3 axis conversion")
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("Expected Nx3 vectors")
    if not all(np.isfinite(x).all() for x in (vectors, transform, A)):
        raise ValueError("Frame inputs must be finite")
    if not np.allclose(A.T @ A, np.eye(3)) or not np.isclose(np.linalg.det(A), 1):
        raise ValueError("Axis conversion must be a proper rotation")
    converted = vectors @ A.T
    pose = transform.copy()
    pose[:3, :3] = transform[:3, :3] @ A.T
    return converted, pose
