"""Static camera algebra for saved HUG exports; no inference or simulation."""
from __future__ import annotations

import numpy as np


def project_camera(points, intrinsic, *, denominator_epsilon=0.0, include_nonpositive=False):
    """OpenCV pinhole projection, optionally using HUG's z+1e-8 export rule.

    Positive depth is required: a finite projection behind the camera does not
    establish image support. Callers report such points separately.
    """
    p, k = np.asarray(points, float), np.asarray(intrinsic, float)
    if (p.ndim != 2 or p.shape[1] != 3 or not len(p) or k.shape != (3, 3)
            or not np.isfinite(p).all() or not np.isfinite(k).all()
            or not np.array_equal(k[2], [0., 0., 1.])
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.isfinite(denominator_epsilon) or denominator_epsilon < 0):
        raise ValueError("finite Nx3 points and canonical pinhole intrinsics required")
    positive = p[:, 2] > 0
    eligible = (p[:, 2] + denominator_epsilon != 0) if include_nonpositive else positive
    projected = np.full((len(p), 2), np.nan)
    homogeneous = np.einsum("ij,nj->ni", k, p[eligible], optimize=False)
    projected[eligible] = homogeneous[:, :2] / (homogeneous[:, 2:3] + denominator_epsilon)
    return projected, positive


def transform_points(points, transform):
    points, transform = np.asarray(points, float), np.asarray(transform, float)
    if (points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4)
            or not np.isfinite(points).all() or not np.isfinite(transform).all()
            or not np.array_equal(transform[3], [0., 0., 0., 1.])):
        raise ValueError("finite Nx3 points and affine 4x4 transform required")
    return np.einsum("ij,nj->ni", transform[:3, :3], points, optimize=False) + transform[:3, 3]


def inverse_transform_points(points, transform):
    """Solve the full affine system instead of assuming a rotation transpose."""
    points = np.asarray(points, float)
    transform_points(points, transform)  # Validate without changing input.
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return np.linalg.solve(transform, homogeneous.T).T[:, :3]


def rotation_from_saved_six_d(six_d):
    """Independent Gram-Schmidt on HUG's interleaved 3x2 column storage."""
    values = np.asarray(six_d, float)
    if values.size != 6 or not np.isfinite(values).all():
        raise ValueError("six finite orientation values required")
    columns = values.reshape(3, 2)
    a = columns[:, 0]
    if np.linalg.norm(a) < 1e-12:
        raise ValueError("degenerate first rotation column")
    a = a / np.linalg.norm(a)
    b = columns[:, 1] - a * np.dot(a, columns[:, 1])
    if np.linalg.norm(b) < 1e-12:
        raise ValueError("degenerate second rotation column")
    b = b / np.linalg.norm(b)
    return np.column_stack((a, b, np.cross(a, b)))


def image_support(points, intrinsic, width, height, *, depth=None, mask=None,
                  depth_comparison_allowance_m=0.002):
    """Describe center-sample support; depth differences are not visibility proof.

    RGB-D captures precede the generated hand. Their depth is scene depth,
    rather than observed hand depth. Nearest-pixel comparisons are descriptive.
    """
    if (isinstance(width, bool) or isinstance(height, bool)
            or int(width) != width or int(height) != height or min(width, height) < 1):
        raise ValueError("positive integer image dimensions required")
    uv, positive = project_camera(points, intrinsic)
    supported = positive & np.isfinite(uv).all(axis=1)
    supported &= (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
    supported &= (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
    result = {"point_count": len(uv), "positive_depth": int(positive.sum()),
              "nonpositive_depth": int((~positive).sum()),
              "within_pixel_center_window": int(supported.sum()),
              "outside_pixel_center_window": int((positive & ~supported).sum()),
              "window_definition": "0 <= u <= width-1 and 0 <= v <= height-1; no clipping"}
    if depth is not None:
        depth = np.asarray(depth)
        if depth.shape != (height, width) or not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError("finite nonnegative scene depth must match image")
        pixel = np.floor(uv[supported] + .5).astype(int)
        observed = depth[pixel[:, 1], pixel[:, 0]]
        valid = observed > 0
        delta = np.asarray(points)[supported, 2] - observed
        result.update(scene_depth_comparison="nearest pixel; scene without generated hand; not an occlusion certificate",
                      valid_scene_depth_at_supported_points=int(valid.sum()),
                      behind_scene_by_over_allowance=int(np.sum(valid & (delta > depth_comparison_allowance_m))),
                      in_front_of_scene_by_over_allowance=int(np.sum(valid & (delta < -depth_comparison_allowance_m))),
                      depth_comparison_allowance_m=depth_comparison_allowance_m)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (height, width):
            raise ValueError("object mask must match image")
        pixel = np.floor(uv[supported] + .5).astype(int)
        result["supported_points_inside_object_mask"] = int(np.count_nonzero(mask[pixel[:, 1], pixel[:, 0]]))
        result["mask_is_not_acceptance_gate"] = True
    return result
