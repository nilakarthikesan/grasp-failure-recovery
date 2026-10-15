"""Pure matrix-input MANO skin and a51-dimensional local tangent derivative.

No asset loader, MANO constructor, MuJoCo import or engine callback is present.
Buffers and precision are supplied explicitly. Native/source parity remains a
separate qualification: a float64 derivative is not a savedfloat32 skin target.
"""
from __future__ import annotations
import math

import torch

PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)
ARRAY_KEYS = ('template', 'shapedirs', 'posedirs', 'joint_regressor', 'weights')
TANGENT_SCHEMA = 'mano_named_tangent_3slides_16local_angular_velocities_v1'
TANGENT_NAMES = tuple(['wrist_t'+axis for axis in 'xyz']
    + ['wrist_rot_'+axis for axis in 'xyz']
    + [f'mano_joint_{j}_{axis}' for j in range(1, 16) for axis in 'xyz'])


def finite_tensor(value, shape, name, *, reference=None):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.dtype not in (torch.float32, torch.float64) or value.device.type != 'cpu'
            or not bool(torch.isfinite(value).all())):
        raise ValueError('Finite explicit CPUfloat32/64 '+name+' '+str(shape)+' required')
    if reference is not None and (value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError('No implicit mixed precision/device conversion: '+name)
    return value


def validate_arrays(arrays, beta):
    if set(arrays) != set(ARRAY_KEYS):
        raise ValueError('Exact template/shape/pose/regressor/weight buffer inventory required')
    template = arrays['template']
    if not isinstance(template, torch.Tensor) or template.ndim != 2 or template.shape[1] != 3:
        raise ValueError('Explicit vertex template required')
    n = len(template)
    for name, shape in dict(template=(n, 3), shapedirs=(n, 3, 10), posedirs=(n, 3, 135),
                            joint_regressor=(16, n), weights=(n, 16)).items():
        finite_tensor(arrays[name], shape, name, reference=template)
    finite_tensor(beta, (1, 10), 'literal beta row', reference=template)
    if bool((arrays['weights'] < 0).any()):
        raise ValueError('Nonnegative literal MANO weights required; never renormalized')
    return n


def quaternion_product(left, right):
    """Hamilton wxyz multiplication; arbitrary broadcast batch dimensions."""
    a, b, c, d = left.unbind(-1); e, f, g, h = right.unbind(-1)
    return torch.stack((a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g,
                        a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e), dim=-1)


def rotation_increment(omega):
    theta = torch.linalg.vector_norm(omega, dim=-1)
    # torch.sinc is sin(pi*x)/(pi*x), including its differentiable zero limit.
    scale = .5*torch.sinc(theta/(2*math.pi))
    return torch.cat((torch.cos(theta/2)[..., None], omega*scale[..., None]), dim=-1)


def quaternion_matrices(quaternions):
    """Same homogeneous quaternion matrix formula as source manotorch.

    Unit input is checked. Division by squared norm is explicitly part of this
    representation formula; arbitrary nonunit input is refused, not fitted.
    """
    finite_tensor(quaternions, (16, 4), 'sixteen local unit quaternions')
    norm2 = (quaternions*quaternions).sum(-1)
    limit = 1e-6 if quaternions.dtype == torch.float32 else 1e-12
    if bool((torch.abs(norm2-1) > limit).any()):
        raise ValueError('Literal unit quaternion representation required')
    w, x, y, z = quaternions.unbind(-1); two = 2/norm2
    return torch.stack((1-two*(y*y+z*z), two*(x*y-z*w), two*(x*z+y*w),
                        two*(x*y+z*w), 1-two*(x*x+z*z), two*(y*z-x*w),
                        two*(x*z-y*w), two*(y*z+x*w), 1-two*(x*x+y*y)), dim=-1).reshape(16, 3, 3)


def skin_state(full_local_rotations, translation_world, beta, arrays):
    """Full B_S+B_P and16-bone LBS; flat_hand_mean=True/center_idx=0.

    Root rotation is already in the requested world frame. Child rotations are
    parent-to-child BODY rotations. The wrist translation is added once after
    source root centering. No skin weights/pose terms are omitted or normalized.
    """
    n = validate_arrays(arrays, beta); template = arrays['template']
    finite_tensor(full_local_rotations, (16, 3, 3), 'full local rotations', reference=template)
    finite_tensor(translation_world, (3,), 'world wrist translation', reference=template)
    tolerance = 1e-6 if template.dtype == torch.float32 else 1e-12
    eye = torch.eye(3, dtype=template.dtype, device=template.device)
    orthogonal = torch.einsum('kji,kjl->kil', full_local_rotations, full_local_rotations)
    if (bool((torch.abs(orthogonal-eye) > tolerance).any())
            or bool((torch.abs(torch.linalg.det(full_local_rotations)-1) > tolerance).any())):
        raise ValueError('Proper source-precision rotations required; no projection or fit')
    shaped = template + torch.einsum('vci,i->vc', arrays['shapedirs'], beta[0])
    rest = arrays['joint_regressor'] @ shaped
    corrective = torch.einsum('vcp,p->vc', arrays['posedirs'], (full_local_rotations[1:]-eye).reshape(135))
    posed = shaped + corrective
    rotations, positions = [full_local_rotations[0]], [rest[0]]
    for j in range(1, 16):
        parent = PARENTS[j]
        rotations.append(rotations[parent] @ full_local_rotations[j])
        positions.append(positions[parent] + rotations[parent] @ (rest[j]-rest[parent]))
    absolute_rotations, absolute_positions = torch.stack(rotations), torch.stack(positions)
    transformed = torch.einsum('kij,vkj->vki', absolute_rotations, posed[:, None, :]-rest[None, :, :])
    transformed = transformed + absolute_positions[None, :, :]
    vertices = torch.einsum('vk,vki->vi', arrays['weights'], transformed)-rest[0]+translation_world
    return dict(vertices=vertices, joint_anchors_world=absolute_positions-rest[0]+translation_world,
                absolute_rotations=absolute_rotations, neutral_anchors=rest,
                shaped_vertices=shaped, posed_vertices=posed, pose_corrective=corrective,
                literal_weight_row_sum=arrays['weights'].sum(-1))


def skin_vertices(full_local_rotations, translation_world, beta, arrays):
    return skin_state(full_local_rotations, translation_world, beta, arrays)['vertices']


def tangent_vertices(translation_world, local_quaternions, delta51, beta, arrays):
    finite_tensor(translation_world, (3,), 'world wrist translation', reference=arrays['template'])
    finite_tensor(local_quaternions, (16, 4), 'local quaternions', reference=translation_world)
    finite_tensor(delta51, (51,), 'named local tangent', reference=translation_world)
    updated = quaternion_product(local_quaternions, rotation_increment(delta51[3:].reshape(16, 3)))
    return skin_vertices(quaternion_matrices(updated), translation_world+delta51[:3], beta, arrays)


def skin_jacobian(translation_world, local_quaternions, beta, arrays, *, evaluation_callback=None):
    """d(vertex_world)/d(local named51 tangent),51 bounded forward-AD calls.

    Forward columns avoid batching2334 output gradients for a778-vertex hand.
    There is no MuJoCo call. Each column is one complete skin evaluation.
    """
    zero = torch.zeros(51, dtype=translation_world.dtype, device=translation_world.device)
    columns = []
    for column in range(51):
        if evaluation_callback is not None:
            evaluation_callback()
        direction = torch.zeros_like(zero); direction[column] = 1
        with torch.autograd.forward_ad.dual_level():
            delta = torch.autograd.forward_ad.make_dual(zero, direction)
            output = tangent_vertices(translation_world, local_quaternions, delta, beta, arrays)
            _, tangent = torch.autograd.forward_ad.unpack_dual(output)
            columns.append(tangent.clone())
    return torch.stack(columns, dim=-1)


def skin_force_vjp(translation_world, local_quaternions, beta, arrays, vertex_forces_world):
    """Instantaneous virtual-work transpose. Forces are held fixed for the VJP."""
    finite_tensor(vertex_forces_world, (len(arrays['template']), 3), 'world vertex forces', reference=translation_world)
    zero = torch.zeros(51, dtype=translation_world.dtype, device=translation_world.device, requires_grad=True)
    vertices = tangent_vertices(translation_world, local_quaternions, zero, beta, arrays)
    return torch.autograd.grad((vertices*vertex_forces_world.detach()).sum(), zero)[0]


def material_point(vertices, vertex_jacobian, face_vertex_ids, barycentric):
    if (tuple(face_vertex_ids.shape) != (3,) or face_vertex_ids.dtype != torch.int64
            or bool((face_vertex_ids < 0).any()) or bool((face_vertex_ids >= len(vertices)).any())):
        raise ValueError('Literal valid triangle vertex IDs required')
    finite_tensor(vertices, (len(vertices), 3), 'vertices')
    finite_tensor(vertex_jacobian, (len(vertices), 3, 51), 'vertex tangent Jacobian', reference=vertices)
    finite_tensor(barycentric, (3,), 'material barycentric coordinates', reference=vertices)
    limit = 1e-6 if vertices.dtype == torch.float32 else 1e-12
    if bool((barycentric < 0).any()) or abs(float(barycentric.sum())-1) > limit:
        raise ValueError('Nonnegative sum-one material barycentric coordinates required; no normalization')
    return (torch.einsum('i,ij->j', barycentric, vertices[face_vertex_ids]),
            torch.einsum('i,ijk->jk', barycentric, vertex_jacobian[face_vertex_ids]))


def motion_grid_spec():
    """109 fixed perturbations per saved input; no actual pose is evaluated here."""
    rows = [dict(pose_id='saved_native_unchanged', delta51=[0.]*51)]
    for j in range(15):
        for axis in range(3):
            for sign in (-1, 1):
                delta = [0.]*51; delta[6+3*j+axis] = sign*math.radians(2)
                rows.append(dict(pose_id=f'finger_{j+1}_axis_{axis}_{sign:+d}_2deg', delta51=delta))
    direction = [1., .25, -.1]; length = math.sqrt(sum(x*x for x in direction))
    for degrees in (.5, 2., 5.):
        for sign in (-1, 1):
            delta = [0.]*6 + [sign*math.radians(degrees)*x/length for _ in range(15) for x in direction]
            rows.append(dict(pose_id=f'coupled_{sign:+d}_{degrees:g}deg', delta51=delta))
    for axis in range(3):
        for sign in (-1, 1):
            delta = [0.]*51; delta[3+axis] = sign*math.radians(2)
            rows.append(dict(pose_id=f'wrist_axis_{axis}_{sign:+d}_2deg', delta51=delta))
            delta = [0.]*51; delta[axis] = sign*.002
            rows.append(dict(pose_id=f'translation_axis_{axis}_{sign:+d}_2mm', delta51=delta))
    assert len(rows) == 109
    return dict(schema_version='mano_articulated_development_motion_grid_v1',
                tangent_schema=TANGENT_SCHEMA, tangent_names=list(TANGENT_NAMES), poses=rows,
                source_inventory_indices=[20, 28], target_inputs_unchanged=True,
                interpretation='finite development perturbations only; no continuous motion-envelope or physical acceptance',
                accepted_mapping=False, collection_or_training_allowed=False)
