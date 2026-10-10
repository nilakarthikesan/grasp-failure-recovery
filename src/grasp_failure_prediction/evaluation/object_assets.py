"""Resolve and import a bounded, content-addressed rigid-object MJCF asset.

Native XML mesh and inertial frames are preserved. Compiled MuJoCo mesh
centering/principal-axis transforms must never be reapplied to the raw meshes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from .schema import canonical_sha256


@dataclass(frozen=True)
class ResolvedObjectGeometry:
    mjcf_path: Path
    content_hash: str
    files_sha256: dict[str, str]
    native_mass_kg: float
    source_model_name: str
    local_collision_bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    source_xml: bytes
    mesh_paths: tuple[tuple[str, Path], ...]
    collision_geom_indices: tuple[int, ...]


def _finite_attribute(element: ET.Element, name: str, size: int | None = None) -> None:
    if name not in element.attrib:
        return
    try:
        values = np.asarray([float(value) for value in element.attrib[name].split()])
    except ValueError as exc:
        raise ValueError(f"object {element.tag}.{name} must contain finite numbers") from exc
    if not len(values) or not np.all(np.isfinite(values)) or (size is not None and len(values) != size):
        raise ValueError(f"object {element.tag}.{name} must contain finite numbers of the required size")
    if name in {"quat", "refquat"} and np.linalg.norm(values) <= 1e-12:
        raise ValueError(f"object {element.tag}.{name} must be a nonzero quaternion")


def _validate_local_frames(element: ET.Element) -> None:
    for name, size in (("pos", 3), ("quat", 4), ("euler", 3), ("axisangle", 4),
                       ("xyaxes", 6), ("zaxis", 3), ("refpos", 3), ("refquat", 4)):
        _finite_attribute(element, name, size)


def _asset_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("object mesh paths must stay relative to the MJCF directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("object mesh paths must stay inside the MJCF directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"object mesh does not exist: {resolved}")
    return resolved


def load_object_geometry(
    mjcf_path: Path, *, expected_content_hash: str | None = None,
) -> ResolvedObjectGeometry:
    """Load exactly one free rigid mesh body and verify its complete mesh closure.

    ``content_hash`` is ``canonical_sha256(files_sha256)``. The mapping keys are
    filenames relative to the source MJCF directory; values are tagged SHA-256
    digests of the original MJCF and each referenced mesh. Location changes do
    not change identity. Extra files which the MJCF does not use are excluded.
    """
    path = Path(mjcf_path).resolve()
    source = path.read_bytes()
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise ValueError("object MJCF is not valid XML") from exc
    if root.tag != "mujoco" or not root.get("model", "").strip():
        raise ValueError("object MJCF must name its source model")
    if any(child.tag not in {"compiler", "asset", "worldbody"} for child in root):
        raise ValueError("object MJCF may contain only compiler, asset, and worldbody sections")
    if any(len(root.findall(tag)) != 1 for tag in ("compiler", "asset", "worldbody")):
        raise ValueError("object MJCF requires exactly one compiler, asset, and worldbody")
    compiler = root.find("compiler")
    assert compiler is not None
    if compiler.get("angle") != "radian":
        raise ValueError("object MJCF must declare radian angles for native-frame import")
    if any(name not in {"angle", "meshdir", "strippath"} for name in compiler.attrib):
        raise ValueError("unsupported object compiler setting")
    if compiler.get("strippath", "false") not in {"false", "0"}:
        raise ValueError("object MJCF must preserve mesh paths")
    meshdir = Path(compiler.get("meshdir", "."))
    if meshdir.is_absolute() or ".." in meshdir.parts:
        raise ValueError("object meshdir must stay relative to the MJCF directory")

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    assert asset is not None and worldbody is not None
    if any(child.tag != "mesh" for child in asset):
        raise ValueError("object assets may contain only explicit file-backed meshes")
    if len(worldbody) != 1 or worldbody[0].tag != "body" or len(root.findall(".//body")) != 1:
        raise ValueError("object MJCF must contain exactly one body without nested bodies")
    body = worldbody[0]
    if any(name not in {"name", "pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"}
           for name in body.attrib):
        raise ValueError("unsupported object body attribute")
    if any(child.tag not in {"inertial", "freejoint", "geom"} for child in body):
        raise ValueError("object body may contain only inertial, freejoint, and mesh geoms")
    if len(body.findall("freejoint")) != 1 or len(body.findall("inertial")) != 1:
        raise ValueError("object body requires exactly one freejoint and one explicit inertial")
    freejoint = body.find("freejoint")
    assert freejoint is not None
    if any(name != "name" for name in freejoint.attrib):
        raise ValueError("unsupported object freejoint setting")
    _validate_local_frames(body)
    inertial = body.find("inertial")
    assert inertial is not None
    _validate_local_frames(inertial)
    _finite_attribute(inertial, "mass", 1)
    if "mass" not in inertial.attrib or float(inertial.attrib["mass"]) <= 0:
        raise ValueError("object inertial mass must be finite and positive")
    if ("diaginertia" in inertial.attrib) == ("fullinertia" in inertial.attrib):
        raise ValueError("object inertial requires exactly one diagonal or full inertia")
    _finite_attribute(inertial, "diaginertia", 3)
    _finite_attribute(inertial, "fullinertia", 6)

    files = {path.name: "sha256:" + hashlib.sha256(source).hexdigest()}
    mesh_paths = []
    mesh_names = set()
    for mesh in asset:
        name, filename = mesh.get("name"), mesh.get("file")
        if not name or not filename or name in mesh_names:
            raise ValueError("object meshes require unique names and explicit files")
        mesh_names.add(name)
        _validate_local_frames(mesh)
        _finite_attribute(mesh, "scale", 3)
        if "scale" in mesh.attrib and any(float(value) <= 0 for value in mesh.attrib["scale"].split()):
            raise ValueError("object mesh scale must be positive")
        mesh_path = _asset_path(path.parent, str(meshdir / filename))
        files[str(mesh_path.relative_to(path.parent))] = "sha256:" + hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        mesh_paths.append((name, mesh_path))
    geoms = body.findall("geom")
    if not geoms:
        raise ValueError("object body must contain mesh geoms")
    for geom in geoms:
        if geom.get("type") != "mesh" or geom.get("mesh") not in mesh_names or geom.get("class"):
            raise ValueError("object geoms require explicit mesh references without defaults")
        if any(name in geom.attrib for name in ("material", "hfield")):
            raise ValueError("unsupported object geom asset reference")
        _validate_local_frames(geom)
        for name in ("friction", "solref", "solimp", "size", "mass", "density", "margin", "gap", "rgba"):
            _finite_attribute(geom, name)
    content_hash = canonical_sha256(files)
    if expected_content_hash is not None and content_hash != expected_content_hash:
        raise ValueError(f"object geometry content hash mismatch: expected {expected_content_hash}, resolved {content_hash}")
    try:
        native = mujoco.MjModel.from_xml_path(str(path))
    except (ValueError, mujoco.FatalError) as exc:
        raise ValueError(f"object MJCF failed physical validation: {exc}") from exc
    if native.nbody != 2 or native.njnt != 1 or native.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("object MJCF must compile to exactly one free rigid body")
    if not (np.all(np.isfinite(native.body_inertia)) and np.all(native.body_inertia[1] > 0)
            and np.isfinite(native.body_mass[1]) and native.body_mass[1] > 0):
        raise ValueError("object inertial properties must be finite and positive")
    collisions = tuple(int(index) for index in range(native.ngeom)
                       if native.geom_contype[index] or native.geom_conaffinity[index])
    if not collisions:
        raise ValueError("object requires at least one contact-active collision geom")
    vertices = []
    for geom_id in collisions:
        mesh_id = int(native.geom_dataid[geom_id])
        start, count = int(native.mesh_vertadr[mesh_id]), int(native.mesh_vertnum[mesh_id])
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, native.geom_quat[geom_id])
        vertices.append(native.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T
                        + native.geom_pos[geom_id])
    collision_vertices = np.concatenate(vertices)
    if not np.all(np.isfinite(collision_vertices)):
        raise ValueError("object collision geometry must be finite")
    bounds = (tuple(float(value) for value in collision_vertices.min(axis=0)),
              tuple(float(value) for value in collision_vertices.max(axis=0)))
    return ResolvedObjectGeometry(path, content_hash, files, float(native.body_mass[1]),
                                  root.attrib["model"], bounds, source, tuple(mesh_paths), collisions)


def append_object_geometry(
    scene: ET.Element, worldbody: ET.Element, geometry: ResolvedObjectGeometry,
) -> ET.Element:
    """Copy validated native frames into a scene without compiler-frame offsets."""
    for relative, expected in geometry.files_sha256.items():
        current = geometry.mjcf_path.parent / relative
        if "sha256:" + hashlib.sha256(current.read_bytes()).hexdigest() != expected:
            raise ValueError(f"object geometry source changed after resolution: {relative}")
    source = ET.fromstring(geometry.source_xml)
    source_asset, source_body = source.find("asset"), source.find("worldbody/body")
    assert source_asset is not None and source_body is not None
    asset = scene.find("asset")
    if asset is None:
        asset = ET.SubElement(scene, "asset")
    existing_mesh_names = {mesh.get("name") for mesh in asset.findall("mesh")}
    paths = dict(geometry.mesh_paths)
    mapping = {}
    for index, mesh in enumerate(source_asset):
        copied = deepcopy(mesh)
        name = f"gfp_object_mesh_{index:03d}"
        if name in existing_mesh_names:
            raise ValueError(f"object mesh name conflicts with scene: {name}")
        mapping[mesh.attrib["name"]] = name
        copied.set("name", name)
        copied.set("file", str(paths[mesh.attrib["name"]]))
        asset.append(copied)
    body = deepcopy(source_body)
    body.set("name", "object")
    freejoint = body.find("freejoint")
    assert freejoint is not None
    freejoint.set("name", "object_free")
    collision_number = visual_number = 0
    for index, geom in enumerate(body.findall("geom")):
        geom.set("mesh", mapping[geom.attrib["mesh"]])
        if index in geometry.collision_geom_indices:
            geom.set("name", "object_geom" if collision_number == 0 else f"object_collision_{collision_number:03d}")
            collision_number += 1
        else:
            geom.set("name", f"object_visual_{visual_number:03d}")
            visual_number += 1
    worldbody.append(body)
    return body
