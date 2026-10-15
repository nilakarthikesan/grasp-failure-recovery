"""Static, source-shaped MANO kinematics and an explicit collision approximation.

This is a HUG-inspired *diagnostic* model, not HUG's unpublished simulator or a
physics runner. Its free wrist and fifteen ball joints directly receive MANO
local rotations. It has no controller, actuators, object, floor, or rollout API.
Only ``mj_forward`` is used. Inertias are numerical placeholders, not calibrated
human-hand dynamics. Collision filtering is disabled for this isolated model.

The collision approximation is deliberately fixed from neutral source geometry:
argmax skinning-weight ownership (lowest joint index breaks ties), one capsule
per non-root bone, and a convex palm hull of wrist-owned vertices. Capsule radius
is the RMS distance of all owner vertices to the infinite bone axis. Capsule end
centers follow the next joint, or an explicitly supplied neutral mesh tip at the
terminal joint. No posed-case optimization, contact data, or outcome is used.
The terminal sites rigidly follow bones; they are *not* MANO's posed/skinned mesh
tips. MANO pose blend shapes and linear blend skinning are not approximated as
rigid motion here. Compare both against the original posed mesh separately.

Inputs are metres and a neutral wrist-centered shape. Callers must supply the
correct saved MANO shape and mean-adjusted literal local rotations; this module
does not decode model parameters, guess frame conventions, or retarget fingers.
SciPy is needed only to construct the neutral palm's convex hull.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


NATIVE_MANO_STATIC_VERSION = "native_mano_static_capsule_hull_v1"


def _array(value: object, shape: tuple[int, ...], name: str, *, integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if integer and raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    result = np.array(raw, dtype=np.int64 if integer else np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _rotation(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = _array(value, shape, name)
    product = np.einsum("...ji,...jk->...ik", result, result, optimize=False)
    if not np.allclose(product, np.eye(3), atol=1e-6, rtol=0) or not np.allclose(
        np.linalg.det(result), 1.0, atol=1e-6, rtol=0
    ):
        raise ValueError(f"{name} must contain proper orthonormal rotations")
    return result


def _numbers(values: object) -> str:
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).ravel())


def _quaternion(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).reshape(9))
    return quaternion


@dataclass(frozen=True)
class NativeManoGeometry:
    """Immutable neutral geometry, including explicit native finger-tip ordering."""

    neutral_joints: np.ndarray
    parents: np.ndarray
    neutral_vertices: np.ndarray
    faces: np.ndarray
    skinning_weights: np.ndarray
    neutral_tips: np.ndarray
    tip_joint_indices: np.ndarray

    def __post_init__(self) -> None:
        for name, shape, integer in (
            ("neutral_joints", (16, 3), False), ("parents", (16,), True),
            ("neutral_vertices", (778, 3), False), ("skinning_weights", (778, 16), False),
            ("neutral_tips", (5, 3), False), ("tip_joint_indices", (5,), True),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name, integer=integer))
        raw_faces = np.asarray(self.faces)
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or len(raw_faces) < 1:
            raise ValueError("faces must have shape (F, 3) with at least one triangle")
        object.__setattr__(self, "faces", _array(raw_faces, raw_faces.shape, "faces", integer=True))
        if np.any(self.faces < 0) or np.any(self.faces >= 778):
            raise ValueError("faces must index neutral_vertices")
        triangles = self.neutral_vertices[self.faces]
        area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        if np.any(np.linalg.norm(area_vectors, axis=1) <= 1e-16):
            raise ValueError("faces must not contain degenerate triangles")
        if np.linalg.norm(self.neutral_joints[0]) > 1e-12:
            raise ValueError("neutral_joints must be wrist-centered at joint 0")
        if self.parents[0] != -1 or any(not 0 <= self.parents[j] < j for j in range(1, 16)):
            raise ValueError("parents must be a topologically ordered tree rooted at joint 0")
        children = [np.flatnonzero(self.parents == j) for j in range(16)]
        if len(children[0]) != 5 or any(len(children[j]) > 1 for j in range(1, 16)):
            raise ValueError("MANO topology must have five unbranched fingers")
        leaves = {j for j in range(1, 16) if not len(children[j])}
        if set(self.tip_joint_indices) != leaves or len(set(self.tip_joint_indices)) != 5:
            raise ValueError("tip_joint_indices must identify the five distinct terminal joints")
        for leaf in leaves:
            depth, j = 0, leaf
            while j:
                depth, j = depth + 1, int(self.parents[j])
            if depth != 3:
                raise ValueError("each MANO finger must have exactly three articulated joints")
        if np.any(self.skinning_weights < 0) or not np.allclose(
            self.skinning_weights.sum(axis=1), 1.0, atol=1e-6, rtol=0
        ):
            raise ValueError("skinning_weights must be nonnegative and sum to one per vertex")
        owners = np.argmax(self.skinning_weights, axis=1)
        if any(np.count_nonzero(owners == j) < 1 for j in range(16)):
            raise ValueError("every joint must own at least one neutral vertex")


@dataclass(frozen=True)
class NativeManoReadback:
    """Copies of actual compiled-model world geometry after a static forward."""

    joint_positions_world: np.ndarray
    tip_positions_world: np.ndarray
    capsule_endpoints_world: np.ndarray
    capsule_radii: np.ndarray
    capsule_joint_indices: np.ndarray
    palm_vertices_world: np.ndarray
    palm_faces: np.ndarray
    wrist_position_world: np.ndarray
    wrist_rotation_world: np.ndarray


class NativeManoStaticModel:
    """Independent articulated MANO diagnostic with no physics execution method."""

    def __init__(self, geometry: NativeManoGeometry):
        if not isinstance(geometry, NativeManoGeometry):
            raise TypeError("geometry must be NativeManoGeometry")
        from scipy.spatial import ConvexHull, QhullError

        self.geometry = geometry
        owners = np.argmax(geometry.skinning_weights, axis=1)
        palm = geometry.neutral_vertices[owners == 0]
        if len(palm) < 4:
            raise ValueError("palm ownership needs at least four noncoplanar vertices")
        try:
            hull = ConvexHull(palm)
        except QhullError as exc:
            raise ValueError("neutral wrist-owned palm must have a nondegenerate convex hull") from exc
        if not np.isfinite(hull.volume) or hull.volume <= 1e-15:
            raise ValueError("neutral wrist-owned palm must have positive volume")
        # Preserve outward facet winding; Qhull simplices alone are unordered.
        faces = hull.simplices.copy()
        cross = np.cross(palm[faces[:, 1]] - palm[faces[:, 0]], palm[faces[:, 2]] - palm[faces[:, 0]])
        inward = np.einsum("ij,ij->i", cross, hull.equations[:, :3], optimize=False) < 0
        faces[inward] = faces[inward][:, [0, 2, 1]]
        children = [np.flatnonzero(geometry.parents == j) for j in range(16)]
        terminal_tips = dict(zip(map(int, geometry.tip_joint_indices), geometry.neutral_tips))
        endpoints, radii = [], []
        for j in range(1, 16):
            end = geometry.neutral_joints[children[j][0]] if len(children[j]) else terminal_tips[j]
            delta = end - geometry.neutral_joints[j]
            length = float(np.linalg.norm(delta))
            if length <= 1e-9:
                raise ValueError(f"bone {j} must have positive length")
            offsets = geometry.neutral_vertices[owners == j] - geometry.neutral_joints[j]
            axis = delta / length
            projections = np.einsum("ij,j->i", offsets, axis, optimize=False)
            radial = offsets - projections[:, None] * axis
            radius = float(np.sqrt(np.mean(np.einsum("ij,ij->i", radial, radial, optimize=False))))
            if not np.isfinite(radius) or radius <= 1e-9:
                raise ValueError(f"bone {j} must have positive source-derived radius")
            endpoints.append(delta)
            radii.append(radius)
        self.neutral_capsule_endpoints = np.stack(endpoints)
        self.neutral_capsule_radii = np.asarray(radii)
        for values in (self.neutral_capsule_endpoints, self.neutral_capsule_radii):
            values.setflags(write=False)

        root = ET.Element("mujoco", model=NATIVE_MANO_STATIC_VERSION)
        ET.SubElement(root, "compiler", angle="radian", autolimits="false")
        ET.SubElement(root, "option", gravity="0 0 0")
        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "mesh", name="native_mano_palm_hull", vertex=_numbers(palm),
                      face=" ".join(str(int(x)) for x in faces.ravel()))
        world = ET.SubElement(root, "worldbody")
        bodies = [ET.SubElement(world, "body", name="native_mano_bone_00")]
        ET.SubElement(bodies[0], "freejoint", name="native_mano_wrist")
        ET.SubElement(bodies[0], "inertial", pos="0 0 0", mass="0.1", diaginertia="0.0001 0.0001 0.0001")
        ET.SubElement(bodies[0], "geom", name="native_mano_palm", type="mesh", mesh="native_mano_palm_hull",
                      contype="0", conaffinity="0", rgba="0.6 0.75 0.9 1")
        for j in range(1, 16):
            parent = int(geometry.parents[j])
            body = ET.SubElement(bodies[parent], "body", name=f"native_mano_bone_{j:02d}",
                                 pos=_numbers(geometry.neutral_joints[j] - geometry.neutral_joints[parent]))
            bodies.append(body)
            ET.SubElement(body, "joint", name=f"native_mano_joint_{j:02d}", type="ball", limited="false")
            ET.SubElement(body, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-8 1e-8 1e-8")
            ET.SubElement(body, "geom", name=f"native_mano_capsule_{j:02d}", type="capsule",
                          fromto=_numbers(np.concatenate((np.zeros(3), endpoints[j - 1]))),
                          size=_numbers([radii[j - 1]]), contype="0", conaffinity="0", rgba="0.8 0.65 0.55 1")
        for tip, j in enumerate(map(int, geometry.tip_joint_indices)):
            ET.SubElement(bodies[j], "site", name=f"native_mano_tip_{tip}", size="0.001",
                          pos=_numbers(geometry.neutral_tips[tip] - geometry.neutral_joints[j]))
        self.xml = ET.tostring(root, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self._body_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_BODY, f"native_mano_bone_{j:02d}") for j in range(16)])
        self._joint_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, "native_mano_wrist")] + [
            self._id(mujoco.mjtObj.mjOBJ_JOINT, f"native_mano_joint_{j:02d}") for j in range(1, 16)])
        self._site_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_SITE, f"native_mano_tip_{j}") for j in range(5)])
        self._capsule_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_GEOM, f"native_mano_capsule_{j:02d}") for j in range(1, 16)])
        self._palm_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "native_mano_palm")
        self.set_pose(np.zeros(3), np.eye(3), np.repeat(np.eye(3)[None], 15, axis=0))

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise RuntimeError(f"compiled MANO model missing {name}")
        return result

    def set_pose(self, wrist_position: object, wrist_rotation: object, local_rotations: object) -> None:
        """Set supplied pose after validation; never integrate physics.

        Float32 SO(3) noise up to 1e-6 in orthogonality/determinant is allowed.
        Nearest-rotation SVD projection precedes the quaternion conversion; all
        projection and quaternion roundtrip errors are in ``pose_diagnostics``.
        This is numerical normalization, not a pose-fitting degree of freedom.
        """
        position = _array(wrist_position, (3,), "wrist_position")
        rotation = _rotation(wrist_rotation, (3, 3), "wrist_rotation")
        locals_ = _rotation(local_rotations, (15, 3, 3), "local_rotations")
        if float(self.data.time) != 0.0:
            raise RuntimeError("static MANO data has been advanced; create a fresh static model")
        inputs = np.concatenate((rotation[None], locals_))
        u, _, vh = np.linalg.svd(inputs)
        canonical = np.einsum("kij,kjl->kil", u, vh, optimize=False)
        quaternions = np.stack([_quaternion(matrix) for matrix in canonical])
        roundtrip = np.empty_like(canonical)
        for matrix, quaternion in zip(roundtrip, quaternions):
            mujoco.mju_quat2Mat(matrix.reshape(9), quaternion)
        diagnostics = {
            "input_rotation_orthogonality_max_abs": float(np.max(np.abs(
                np.einsum("kji,kjl->kil", inputs, inputs, optimize=False) - np.eye(3)))),
            "input_rotation_determinant_max_abs": float(np.max(np.abs(np.linalg.det(inputs) - 1.0))),
            "rotation_projection_max_abs": float(np.max(np.abs(canonical - inputs))),
            "rotation_quaternion_roundtrip_max_abs": float(np.max(np.abs(roundtrip - canonical))),
            "input_rotation_tolerance": 1e-6,
        }
        qpos = self.model.qpos0.copy()
        address = int(self.model.jnt_qposadr[self._joint_ids[0]])
        qpos[address:address + 3] = position
        qpos[address + 3:address + 7] = quaternions[0]
        for j, local in enumerate(locals_, start=1):
            address = int(self.model.jnt_qposadr[self._joint_ids[j]])
            qpos[address:address + 4] = quaternions[j]
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.pose_diagnostics = diagnostics

    def readback(self) -> NativeManoReadback:
        """Read compiled body/site frames and actual compiled mesh coordinates.

        MuJoCo recenters and rotates meshes during compilation. Recovering palm
        vertices through its geom world frame, once, handles that transform.
        Capsule endpoint order follows source joint -> child/tip.
        """
        if float(self.data.time) != 0.0:
            raise RuntimeError("static MANO data has been advanced")
        ids = self._capsule_ids
        rotations = self.data.geom_xmat[ids].reshape(-1, 3, 3)
        # MuJoCo's fromto convention places the first endpoint along local +z.
        deltas = rotations[:, :, 2] * self.model.geom_size[ids, 1, None]
        centers = self.data.geom_xpos[ids]
        endpoints = np.stack((centers + deltas, centers - deltas), axis=1)
        mesh_id = int(self.model.geom_dataid[self._palm_id])
        begin, count = int(self.model.mesh_vertadr[mesh_id]), int(self.model.mesh_vertnum[mesh_id])
        vertices = self.model.mesh_vert[begin:begin + count].astype(np.float64)
        rotation = self.data.geom_xmat[self._palm_id].reshape(3, 3)
        vertices_world = np.einsum("ij,kj->ki", rotation, vertices, optimize=False) + self.data.geom_xpos[self._palm_id]
        begin, count = int(self.model.mesh_faceadr[mesh_id]), int(self.model.mesh_facenum[mesh_id])
        return NativeManoReadback(
            joint_positions_world=self.data.xpos[self._body_ids].copy(),
            tip_positions_world=self.data.site_xpos[self._site_ids].copy(),
            capsule_endpoints_world=endpoints,
            capsule_radii=self.model.geom_size[ids, 0].copy(),
            capsule_joint_indices=np.arange(1, 16),
            palm_vertices_world=vertices_world,
            palm_faces=self.model.mesh_face[begin:begin + count].copy(),
            wrist_position_world=self.data.xpos[self._body_ids[0]].copy(),
            wrist_rotation_world=self.data.xmat[self._body_ids[0]].reshape(3, 3).copy(),
        )
