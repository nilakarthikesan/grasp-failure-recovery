"""Complete posed MANO surface, rigidly attached to the saved palm frame.

Pure array/XML preparation and readback checks only. Importing this module
does not import MuJoCo, construct a model, run FK or evaluate a distance field.
This is a fixed-articulation reference, not a skin deformation implementation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

OWNER_BODY = "mano_body_0"
FLEX_NAME = "literal_native_closed_skin_locked_v1"
VERTICES = 778
FACES = 1552
PARTS = ("palm", "index", "middle", "ring", "little", "thumb")
REPLAY_LIMIT_M = 1e-12
NORMAL_COMPONENT_LIMIT = 1e-10
MIN_DOUBLE_AREA_M2 = 1e-15
FLAGS = dict(accepted_mapping=False, native_fidelity_accepted=False,
             production_promotion=False, new_collection_or_training_allowed=False)
INPUT_NAMES = ("decoded_reference", "kinematic_readback", "membership_provenance",
               "shape_asset", "literal_pose")
FINGER_NAMES = tuple(f"mano_joint_{i}" for i in range(1, 16))
SLIDE_NAMES = ("wrist_tx", "wrist_ty", "wrist_tz")
JOINT_NAMES = SLIDE_NAMES + ("wrist_rot",) + FINGER_NAMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key].copy() for key in value.files}


def binding(path, label):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, label=label)


def verify_binding(row):
    if not isinstance(row, dict) or not {"path", "sha256", "bytes"} <= row.keys():
        raise ValueError("Explicit file hash and byte-size binding required")
    path = Path(row["path"]).resolve()
    if (not path.is_file() or path.stat().st_size != row["bytes"]
            or sha(path) != row["sha256"]):
        raise ValueError("Bound input bytes changed: " + str(path))
    return path


def rigid_transform(value):
    value = np.asarray(value, np.float64)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.array_equal(value[3], [0., 0., 0., 1.])):
        raise ValueError("Literal finite homogeneous rigid transform required")
    rotation = value[:3, :3]
    if (np.max(np.abs(np.einsum('ji,jk->ik', rotation, rotation, optimize=False)
                      - np.eye(3))) > 1e-12
            or abs(np.linalg.det(rotation) - 1) > 1e-12):
        raise ValueError("Proper owner rotation required; no normalization or fit")
    return value.copy()


def world_coordinates(local, owner):
    owner = rigid_transform(owner)
    return np.einsum('ij,nj->ni', owner[:3, :3], local, optimize=False) + owner[:3, 3]


def oriented_normals(vertices, faces):
    triangle = np.asarray(vertices, np.float64)[faces]
    cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    if not np.isfinite(area2).all() or np.any(area2 <= MIN_DOUBLE_AREA_M2):
        raise ValueError("Degenerate surface face; no repair, deletion or inference")
    return cross / area2[:, None], area2


def validate_surface(vertices, faces, labels):
    """Closed oriented triangle manifold checks, independent of an object."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    labels = np.asarray(labels)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or vertices.dtype.kind != 'f' or not np.isfinite(vertices).all()):
        raise ValueError("Finite floating surface vertices required")
    if (faces.ndim != 2 or faces.shape[1:] != (3,) or faces.dtype.kind not in 'iu'
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Literal integer triangle indices required")
    if labels.shape != (len(faces),) or labels.dtype.kind not in 'iu' or np.any((labels < 0) | (labels > 5)):
        raise ValueError("One frozen native anatomical label per face required")
    if len(np.unique(faces)) != len(vertices):
        raise ValueError("All source vertices must be referenced; no silent dropping")
    if len({tuple(sorted(map(int, face))) for face in faces}) != len(faces):
        raise ValueError("Duplicate triangle support")
    edges = {}
    for face in faces:
        if len(set(map(int, face))) != 3:
            raise ValueError("Repeated triangle vertex")
        for a, b in zip(face, np.roll(face, -1)):
            a, b = int(a), int(b)
            key = (min(a, b), max(a, b))
            edges.setdefault(key, []).append(1 if a < b else -1)
    if any(len(uses) != 2 or sum(uses) != 0 for uses in edges.values()):
        raise ValueError("Closed consistently wound two-manifold required")
    normals, area2 = oriented_normals(vertices, faces)
    centered = np.asarray(vertices, np.float64) - np.asarray(vertices, np.float64).mean(axis=0)
    triangle = centered[faces]
    volume = float(np.einsum('ij,ij->', triangle[:, 0],
                            np.cross(triangle[:, 1], triangle[:, 2]), optimize=False) / 6)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Positive outward signed volume required; no automatic face flip")
    return dict(vertices=len(vertices), faces=len(faces), edges=len(edges),
                signed_volume_m3=volume, minimum_double_area_m2=float(area2.min()),
                face_inventory={name: int((labels == i).sum()) for i, name in enumerate(PARTS)}), normals


def make_candidate(vertices_world, faces, labels, owner_world_transform, *, radius_m):
    """Pure complete-surface construction; no qualification implied by return."""
    if not np.isfinite(radius_m) or radius_m < 0:
        raise ValueError("Predeclared finite nonnegative flex radius required")
    owner = rigid_transform(owner_world_transform)
    world = np.asarray(vertices_world, np.float64).copy()
    faces = np.asarray(faces).copy()
    labels = np.asarray(labels).copy()
    topology, normals = validate_surface(world, faces, labels)
    local = np.einsum('ni,ij->nj', world - owner[:3, 3], owner[:3, :3], optimize=False)
    reconstructed = world_coordinates(local, owner)
    error = float(np.max(np.abs(reconstructed - world)))
    if error > REPLAY_LIMIT_M:
        raise ValueError("Saved-owner inverse transform failed its numerical replay check")
    local_normals, _ = oriented_normals(local, faces)
    return dict(native_vertices_world_m=world, local_vertices_m=local,
                native_faces=faces, native_face_labels=labels,
                native_vertex_ids=np.arange(len(world), dtype=np.int64),
                native_face_ids=np.arange(len(faces), dtype=np.int64),
                saved_owner_world_transform=owner, native_oriented_face_normals_world=normals,
                local_oriented_face_normals=local_normals), dict(
                    topology=topology, inverse_transform_max_error_m=error, radius_m=float(radius_m),
                    radius_changes_collision_boundary=bool(radius_m > 0), **FLAGS)


def xml_include(candidate, *, radius_m):
    """Inline float64 round-trip decimals; no OBJ quantization or hull asset."""
    if not np.isfinite(radius_m) or radius_m < 0:
        raise ValueError("Finite nonnegative declared radius required")
    validate_surface(candidate['local_vertices_m'], candidate['native_faces'], candidate['native_face_labels'])
    root = ET.Element('mujocoinclude')
    deformable = ET.SubElement(root, 'deformable')
    flex = ET.SubElement(deformable, 'flex', name=FLEX_NAME, dim='2', body=OWNER_BODY,
                         radius=format(float(radius_m), '.17g'),
                         vertex=' '.join(format(float(x), '.17g') for x in candidate['local_vertices_m'].flat),
                         element=' '.join(str(int(x)) for x in candidate['native_faces'].flat))
    ET.SubElement(flex, 'edge', damping='0')
    ET.SubElement(flex, 'elasticity', young='0', damping='0', elastic2d='none')
    ET.SubElement(flex, 'contact', contype='1', conaffinity='1', condim='3', margin='0', gap='0',
                  internal='false', selfcollide='none')
    return ET.tostring(root, encoding='unicode') + '\n'


def assert_locked_articulation(saved_named, actual_named):
    if set(saved_named) != set(JOINT_NAMES) or set(actual_named) != set(JOINT_NAMES):
        raise ValueError("Exact nineteen-joint native inventory required")
    for name in JOINT_NAMES:
        saved, actual = np.asarray(saved_named[name], float), np.asarray(actual_named[name], float)
        shape = () if name in SLIDE_NAMES else (4,)
        if saved.shape != shape or actual.shape != shape or not np.isfinite(saved).all() or not np.isfinite(actual).all():
            raise ValueError("Finite literal named native pose required")
        if name not in SLIDE_NAMES and (abs(np.linalg.norm(saved) - 1) > 1e-12 or abs(np.linalg.norm(actual) - 1) > 1e-12):
            raise ValueError("Literal unit quaternion required; no normalization")
        if name in FINGER_NAMES and not np.array_equal(saved, actual):
            raise ValueError("Finger articulation changed in locked-surface condition: " + name)


def prepare_saved_candidate(index, spec, output, *, radius_policy):
    """Prepare bound saved data only; caller separately freezes/runs any model.

    The run wrapper owns validation of prior completed native/ablation protocols.
    Every supplied prior receipt is nevertheless hash-checked before and after
    this preparation. Radius policy is validated by the primitive run wrapper.
    """
    if index not in (20, 28) or spec.get('source_inventory_index') != index:
        raise ValueError("Only original saved development cases20/28")
    if spec.get('schema_version') != 'manosim_full_surface_saved_case_spec_v1':
        raise ValueError("Unknown saved surface input schema")
    if set(spec.get('inputs', {})) != set(INPUT_NAMES):
        raise ValueError("Exact saved surface/pose/shape/membership input inventory required")
    rows = list(spec['inputs'].values()) + list(spec['prior_evidence'].values())
    paths = {name: verify_binding(row) for name, row in spec['inputs'].items()}
    for row in rows:
        verify_binding(row)
    if not spec['prior_evidence']:
        raise ValueError("Prior source-bound evidence required; not a new native input")
    if radius_policy.get('schema_version') != 'manosim_qualified_flex_radius_selection_v1':
        raise ValueError("Separately qualified radius selection required")
    verify_binding(radius_policy['qualification_receipt'])
    radius = radius_policy['radius_m']
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Finite selected radius required")
    decoded, kinematic, membership = (arrays(paths[name]) for name in INPUT_NAMES[:3])
    shape = np.load(paths['shape_asset'], allow_pickle=False)
    if (shape.dtype != np.float32 or shape.shape != (10,)
            or decoded['shape'].dtype != np.float32 or decoded['shape'].shape != (1, 10)
            or not np.isfinite(shape).all() or not np.array_equal(shape, decoded['shape'][0])
            or sha(paths['shape_asset']) != spec['expected_shape_sha256']):
        raise ValueError("Exact literal native float32 shape identity required")
    world, faces, labels = (decoded['native_vertices_world_m'], decoded['original_native_faces'],
                            membership['native_face_labels'])
    if world.shape != (VERTICES, 3) or faces.shape != (FACES, 3) or labels.shape != (FACES,) or set(labels) != set(range(6)):
        raise ValueError("All778 native vertices/1552 closed faces and six fixed parts required")
    owner = rigid_transform(kinematic['root_body_world_transform'])
    if (not np.array_equal(owner[:3, :3], kinematic['compiled_world_body_rotations'][0])
            or not np.array_equal(owner[:3, 3], kinematic['compiled_body_origins_world_m'][0])):
        raise ValueError("Saved actual palm body frame differs from its readback")
    literal = read(paths['literal_pose'])
    assert_locked_articulation(literal['named_qpos'], literal['named_qpos'])
    candidate, receipt = make_candidate(world, faces, labels, owner, radius_m=radius)
    candidate['original_shape_float32'] = shape.copy()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    array_path, xml_path = output/'candidate_surface.npz', output/'candidate_flex.xml'
    with array_path.open('xb') as f:
        np.savez_compressed(f, **candidate)
    with xml_path.open('x') as f:
        f.write(xml_include(candidate, radius_m=radius))
    artifacts = [binding(array_path, 'complete_original_native_surface_and_membership'),
                 binding(xml_path, 'single_palm_complete_face_flex_include')]
    receipt.update(schema_version='manosim_locked_native_full_surface_candidate_v1',
                   status='PREPARED_STATIC_REFERENCE_ONLY', source_inventory_index=index,
                   native_sha256=spec['native_sha256'], shape_asset_sha256=spec['expected_shape_sha256'],
                   owner_body_name=OWNER_BODY, flex_name=FLEX_NAME, ownership='single_saved_actual_palm_body',
                   coordinate_arithmetic='float64 inverse SAVED actual body transform; nonoptimized einsum;17-digit XML',
                   array_hashes={key: array_hash(value) for key, value in candidate.items()},
                   inputs=spec['inputs'], prior_evidence=spec['prior_evidence'], radius_policy=radius_policy,
                   artifacts=artifacts, full_orientation_status='NOT_RUN', physical_outcome=None,
                   scientific_acceptance_changed=False, full_face_deletion_count=0, convexification=False,
                   finger_articulation_allowed=False, root_rigid_motion_only=True,
                   compiled_readback_status='NOT_RUN', distance_fields_status='NOT_RUN',
                   limitations=['Reference geometry only; no physical hand qualification',
                                'Finger changes invalidate this fixed-articulation condition',
                                'Corresponding-face normals do not certify object-query orientation',
                                'No continuous surface/contact or dynamics acceptance is issued'])
    for row in rows + [radius_policy['qualification_receipt']]:
        verify_binding(row)
    path = output/'candidate_receipt.json'
    with path.open('x') as f:
        json.dump(receipt, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    return receipt, binding(path, 'locked_native_full_surface_preparation_receipt')


def compiled_surface_readback(model, data, expected, *, owner_body_id, radius_m, saved_named, actual_named):
    """Inspect provided actual arrays only; caller owns counted static forwards."""
    assert_locked_articulation(saved_named, actual_named)
    if (model.nflex != 1 or int(model.flex_dim[0]) != 2 or not bool(model.flex_rigid[0])
            or int(model.flex_vertnum[0]) != len(expected['local_vertices_m'])
            or int(model.flex_elemnum[0]) != len(expected['native_faces'])):
        raise ValueError("Exactly one complete rigid2D native flex required")
    if (float(model.flex_radius[0]) != radius_m or float(model.flex_margin[0]) != 0
            or float(model.flex_gap[0]) != 0):
        raise ValueError("Actual flex radius/margin/gap differs from declared boundary")
    vbase, ebase = int(model.flex_vertadr[0]), int(model.flex_elemdataadr[0])
    nvert, nface = len(expected['local_vertices_m']), len(expected['native_faces'])
    local = np.asarray(model.flex_vert[vbase:vbase+nvert], float).reshape(nvert, 3).copy()
    world = np.asarray(data.flexvert_xpos[vbase:vbase+nvert], float).reshape(nvert, 3).copy()
    faces = np.asarray(model.flex_elem[ebase:ebase+3*nface]).reshape(nface, 3).copy()
    owners = np.asarray(model.flex_vertbodyid[vbase:vbase+nvert]).copy()
    if not np.array_equal(owners, np.full(nvert, owner_body_id)):
        raise ValueError("Compiled vertex owner differs from single named palm body")
    if not np.array_equal(faces, expected['native_faces']) or not np.array_equal(local, expected['local_vertices_m']):
        raise ValueError("Compiler changed complete face order/indices or literal local vertices")
    owner = np.eye(4)
    owner[:3, :3] = np.asarray(data.xmat[owner_body_id]).reshape(3, 3)
    owner[:3, 3] = np.asarray(data.xpos[owner_body_id])
    owner = rigid_transform(owner)
    saved_owner = rigid_transform(expected['saved_owner_world_transform'])
    source_normals, _ = oriented_normals(expected['native_vertices_world_m'], expected['native_faces'])
    if not np.array_equal(source_normals, expected['native_oriented_face_normals_world']):
        raise ValueError('Original native oriented normal provenance differs')
    relative_rotation = np.einsum('ij,kj->ik', owner[:3, :3], saved_owner[:3, :3], optimize=False)
    target = (np.einsum('ij,nj->ni', relative_rotation,
                       expected['native_vertices_world_m'] - saved_owner[:3, 3], optimize=False)
              + owner[:3, 3])
    topology, normals = validate_surface(world, faces, expected['native_face_labels'])
    target_normals = np.einsum('ij,nj->ni', relative_rotation,
                              expected['native_oriented_face_normals_world'], optimize=False)
    if not all(np.isfinite(value).all() for value in (world, target, normals, target_normals)):
        raise ValueError('Finite actual/reference compiled surface and oriented normals required')
    vertex_error = float(np.max(np.abs(world - target)))
    vertex_l2_error = float(np.max(np.linalg.norm(world - target, axis=1)))
    normal_error = float(np.max(np.abs(normals - target_normals)))
    if vertex_error > REPLAY_LIMIT_M or normal_error > NORMAL_COMPONENT_LIMIT:
        raise ValueError("Compiled vertex/ORIENTED normal representation readback differs")
    if data.time != 0:
        raise ValueError("Static reference time advanced")
    values = dict(compiled_vertices_world_m=world, compiled_local_vertices_m=local,
                  compiled_faces=faces, compiled_vertex_body_ids=owners,
                  compiled_oriented_face_normals_world=normals,
                  expected_oriented_face_normals_world=target_normals,
                  actual_owner_world_transform=owner, reference_vertices_world_m=target,
                  native_face_labels=expected['native_face_labels'].copy(),
                  native_face_ids=expected['native_face_ids'].copy())
    receipt = dict(schema_version='manosim_locked_surface_compiled_readback_v1',
                   status='NUMERICAL_GEOMETRY_READBACK_ONLY', topology=topology,
                   maximum_vertex_error_m=vertex_error, maximum_normal_component_error=normal_error,
                   maximum_vertex_l2_error_m=vertex_l2_error,
                   corresponding_triangle_barycentric_discrepancy_bound_m=vertex_l2_error,
                   barycentric_bound_scope='fixed corresponding triangles at this pose only; no joint-configuration or contact-search coverage',
                   representation_limit_m=REPLAY_LIMIT_M, normal_component_limit=NORMAL_COMPONENT_LIMIT,
                   limits_are_contact_allowances=False, corresponding_face_normals_only=True,
                   object_query_orientation_status='NOT_RUN', radius_m=float(radius_m),
                   radius_changes_collision_boundary=bool(radius_m > 0),
                   finger_articulation_allowed=False, physical_outcome=None, **FLAGS)
    return values, receipt
