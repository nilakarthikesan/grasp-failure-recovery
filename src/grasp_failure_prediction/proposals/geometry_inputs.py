"""Deterministic provider geometry inputs derived from registered object assets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from grasp_failure_prediction.evaluation.object_assets import ResolvedObjectGeometry
from grasp_failure_prediction.evaluation.schema import canonical_json, canonical_sha256


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _collision_triangles(geometry: ResolvedObjectGeometry) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(geometry.mjcf_path))
    triangles = []
    for geom_id in geometry.collision_geom_indices:
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
        faces = model.mesh_face[face_start : face_start + face_count]
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        transformed = vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom_id]
        triangles.append(transformed[faces])
    if not triangles:
        raise ValueError("registered object has no collision mesh triangles")
    result = np.concatenate(triangles).astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError("registered object triangles must be finite")
    return result


def prepare_geometry_inputs(
    geometry: ResolvedObjectGeometry,
    output_directory: Path,
    *,
    seed: int,
    point_count: int,
    include_normals: bool = False,
) -> dict:
    """Write a deterministic object-frame area-sampled point cloud and manifest."""

    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("geometry sampling seed must fit unsigned 32-bit range")
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"geometry input directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    triangles = _collision_triangles(geometry)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 0
    triangles, cross, double_area = triangles[valid], cross[valid], double_area[valid]
    if not len(triangles):
        raise ValueError("registered object collision mesh has no nondegenerate triangles")
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = rng.choice(len(triangles), size=point_count, p=double_area / double_area.sum())
    uv = rng.random((point_count, 2))
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    chosen = triangles[selected]
    points = chosen[:, 0] + uv[:, :1] * (chosen[:, 1] - chosen[:, 0]) + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
    points = np.asarray(points, dtype="<f8")
    points_path = output / "object_points.npy"
    np.save(points_path, points, allow_pickle=False)
    generated = {"object_point_cloud": {"path": points_path.name, "sha256": _sha256(points_path)}}
    if include_normals:
        normals = cross[selected] / double_area[selected, None]
        normals_path = output / "object_normals.npy"
        np.save(normals_path, np.asarray(normals, dtype="<f8"), allow_pickle=False)
        generated["object_normals"] = {"path": normals_path.name, "sha256": _sha256(normals_path)}
    identity = {
        "schema_version": "geometry_input_manifest_v1",
        "object_geometry_content_hash": geometry.content_hash,
        "dependency_files_sha256": geometry.files_sha256,
        "coordinate_frame": "mujoco_object_body",
        "length_unit": "meter",
        "object_to_points": {
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "sampling": {
            "method": "collision_mesh_area_uniform_v1",
            "rng": "numpy_pcg64",
            "seed": seed,
            "point_count": point_count,
            "include_normals": include_normals,
        },
        "generated_inputs": generated,
    }
    manifest = {**identity, "manifest_hash": canonical_sha256(identity)}
    (output / "geometry_input_manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    return manifest
