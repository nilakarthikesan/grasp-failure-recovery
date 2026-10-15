"""Manufactured complete-surface controls; no model construction or engine call."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('manufactured_locked_native_surface', HERE/'surface_candidate.py')
surface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surface)


def tetra(scale=1.):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) * scale
    return vertices, np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]), np.arange(4)


def owner():
    t = np.eye(4)
    t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    t[:3, 3] = [.123, -.456, .789]
    return t


def named():
    return {name: 0. if name in surface.SLIDE_NAMES else [1., 0., 0., 0.] for name in surface.JOINT_NAMES}


def candidate(scale=1., radius=0.):
    vertices, faces, labels = tetra(scale)
    return surface.make_candidate(vertices, faces, labels, owner(), radius_m=radius)[0]


def compiled(value, radius=0.):
    n, f = len(value['local_vertices_m']), len(value['native_faces'])
    model = SimpleNamespace(nflex=1, flex_dim=np.array([2]), flex_rigid=np.array([True]),
        flex_vertnum=np.array([n]), flex_elemnum=np.array([f]), flex_radius=np.array([radius]),
        flex_margin=np.zeros(1), flex_gap=np.zeros(1), flex_vertadr=np.zeros(1, int),
        flex_elemdataadr=np.zeros(1, int), flex_vert=value['local_vertices_m'].copy(),
        flex_elem=value['native_faces'].ravel().copy(), flex_vertbodyid=np.full(n, 1))
    data = SimpleNamespace(flexvert_xpos=surface.world_coordinates(value['local_vertices_m'], owner()),
        xpos=np.array([[0., 0., 0.], owner()[:3, 3]]),
        xmat=np.array([np.eye(3).ravel(), owner()[:3, :3].ravel()]), time=0.)
    return model, data


def test_complete_surface_inverse_transform_and_oriented_normal_readback():
    value = candidate()
    np.testing.assert_allclose(surface.world_coordinates(value['local_vertices_m'], owner()), value['native_vertices_world_m'], atol=1e-15)
    model, data = compiled(value)
    arrays, receipt = surface.compiled_surface_readback(model, data, value, owner_body_id=1,
        radius_m=0., saved_named=named(), actual_named=named())
    np.testing.assert_allclose(arrays['compiled_oriented_face_normals_world'], value['native_oriented_face_normals_world'], atol=1e-14)
    assert receipt['maximum_vertex_error_m'] < 1e-15
    assert receipt['maximum_normal_component_error'] < 1e-14
    assert receipt['limits_are_contact_allowances'] is False
    assert all(receipt[key] is False for key in surface.FLAGS)


def test_inline_xml_roundtrips_float64_and_keeps_every_face_and_single_owner():
    value = candidate()
    text = surface.xml_include(value, radius_m=0.)
    root = ET.fromstring(text)
    flex = root.find('./deformable/flex')
    assert flex.get('body') == 'mano_body_0' and flex.get('radius') == '0'
    np.testing.assert_array_equal(np.fromstring(flex.get('vertex'), sep=' ').reshape(-1, 3), value['local_vertices_m'])
    np.testing.assert_array_equal(np.fromstring(flex.get('element'), sep=' ', dtype=int).reshape(-1, 3), value['native_faces'])
    assert root.findall('.//mesh') == [] and root.findall('.//body') == []
    assert flex.find('contact').get('selfcollide') == 'none'


@pytest.mark.parametrize('change', ['open', 'flip_one', 'flip_all', 'duplicate', 'repeated', 'index', 'fractional', 'unused', 'degenerate'])
def test_bad_source_topology_or_winding_refused_without_automatic_repair(change):
    vertices, faces, labels = tetra()
    if change == 'open': faces, labels = faces[:-1], labels[:-1]
    elif change == 'flip_one': faces[0] = faces[0, ::-1]
    elif change == 'flip_all': faces = faces[:, ::-1]
    elif change == 'duplicate': faces[0] = faces[1]
    elif change == 'repeated': faces[0] = [0, 0, 1]
    elif change == 'index': faces[0, 0] = 4
    elif change == 'fractional': faces = faces.astype(float)
    elif change == 'unused': vertices = np.vstack([vertices, [5., 5., 5.]])
    elif change == 'degenerate': vertices[1] = vertices[0]
    with pytest.raises(ValueError): surface.make_candidate(vertices, faces, labels, owner(), radius_m=0.)


@pytest.mark.parametrize('radius', [-1e-9, float('nan'), float('inf')])
def test_radius_cannot_be_negative_nonfinite_or_silently_clamped(radius):
    with pytest.raises(ValueError): candidate(radius=radius)


def test_positive_radius_is_explicit_new_boundary_and_never_gets_fidelity_pass():
    value = candidate(radius=1e-6)
    model, data = compiled(value, radius=1e-6)
    _, receipt = surface.compiled_surface_readback(model, data, value, owner_body_id=1,
        radius_m=1e-6, saved_named=named(), actual_named=named())
    assert receipt['radius_changes_collision_boundary'] is True
    assert all(receipt[key] is False for key in surface.FLAGS)


@pytest.mark.parametrize('change', ['face', 'local', 'owner', 'radius', 'margin', 'gap', 'dim', 'time', 'nan_owner', 'nan_source_normal'])
def test_actual_compiled_contract_tampering_is_refused(change):
    value = candidate()
    model, data = compiled(value)
    if change == 'face': model.flex_elem[:3] = model.flex_elem[:3][::-1]
    elif change == 'local': model.flex_vert[0, 0] += 1e-16
    elif change == 'owner': model.flex_vertbodyid[0] = 0
    elif change == 'radius': model.flex_radius[0] = 1e-6
    elif change == 'margin': model.flex_margin[0] = 1e-10
    elif change == 'gap': model.flex_gap[0] = 1e-10
    elif change == 'dim': model.flex_dim[0] = 3
    elif change == 'time': data.time = .002
    elif change == 'nan_owner': data.xpos[1, 0] = np.nan
    elif change == 'nan_source_normal': value['native_oriented_face_normals_world'][0, 0] = np.nan
    with pytest.raises(ValueError): surface.compiled_surface_readback(model, data, value, owner_body_id=1,
        radius_m=0., saved_named=named(), actual_named=named())


def test_normal_guard_is_independent_of_small_vertex_error_and_reports_source_orientation():
    value = candidate(scale=1e-6)
    model, data = compiled(value)
    data.flexvert_xpos[0, 2] += 5e-13
    with pytest.raises(ValueError, match='ORIENTED normal'):
        surface.compiled_surface_readback(model, data, value, owner_body_id=1,
            radius_m=0., saved_named=named(), actual_named=named())


def test_rigid_root_motion_transforms_original_face_normals_without_finger_change():
    value = candidate()
    model, data = compiled(value)
    moved = np.eye(4); moved[:3, :3] = np.diag([-1., 1., -1.]); moved[:3, 3] = [2., 3., 4.]
    data.xpos[1] = moved[:3, 3]; data.xmat[1] = moved[:3, :3].ravel()
    data.flexvert_xpos = surface.world_coordinates(value['local_vertices_m'], moved)
    actual = named(); actual['wrist_tx'] = 2.; actual['wrist_rot'] = [0., 0., 1., 0.]
    _, receipt = surface.compiled_surface_readback(model, data, value, owner_body_id=1,
        radius_m=0., saved_named=named(), actual_named=actual)
    assert receipt['maximum_normal_component_error'] < 1e-14
    actual['mano_joint_3'] = [-1., 0., 0., 0.]
    with pytest.raises(ValueError, match='articulation changed'):
        surface.compiled_surface_readback(model, data, value, owner_body_id=1,
            radius_m=0., saved_named=named(), actual_named=actual)


def test_sheared_owner_is_not_silently_fitted_to_rigid_transform():
    bad = owner(); bad[0, 0] += 1e-7
    v, f, l = tetra()
    with pytest.raises(ValueError, match='Proper owner'):
        surface.make_candidate(v, f, l, bad, radius_m=0.)


def full_manufactured_inputs(tmp_path):
    # A closed bipyramid with the exact production counts, entirely synthetic.
    phase = np.arange(776) * (2*np.pi/776)
    world = np.vstack([[0., 0., .03], [0., 0., -.03],
                       np.column_stack([.02*np.cos(phase), .02*np.sin(phase), np.zeros(776)])])
    faces = np.array([[0, 2+i, 2+(i+1)%776] for i in range(776)]
                     + [[1, 2+(i+1)%776, 2+i] for i in range(776)])
    shape = np.arange(10, dtype=np.float32)
    paths = {}
    contents = dict(decoded_reference=dict(native_vertices_world_m=world, original_native_faces=faces, shape=shape[None]),
                    kinematic_readback=dict(root_body_world_transform=owner(),
                        compiled_world_body_rotations=np.tile(owner()[:3, :3], (16, 1, 1)),
                        compiled_body_origins_world_m=np.tile(owner()[:3, 3], (16, 1))),
                    membership_provenance=dict(native_face_labels=np.arange(1552)%6))
    for name, values in contents.items():
        path = tmp_path/(name+'.npz'); np.savez_compressed(path, **values); paths[name] = path
    paths['shape_asset'] = tmp_path/'shape.npy'; np.save(paths['shape_asset'], shape)
    paths['literal_pose'] = tmp_path/'literal.json'; paths['literal_pose'].write_text(json.dumps(dict(named_qpos=named())))
    prior = tmp_path/'prior.json'; prior.write_text('{"manufactured":true}\n')
    qualified = tmp_path/'primitive.json'; qualified.write_text('{"manufactured":true}\n')
    spec = dict(schema_version='manosim_full_surface_saved_case_spec_mj340_v1', source_inventory_index=20,
                native_sha256='a'*64, expected_shape_sha256=surface.sha(paths['shape_asset']),
                inputs={name: surface.binding(path, name) for name, path in paths.items()},
                prior_evidence=dict(manufactured=surface.binding(prior, 'prior')))
    policy = dict(schema_version='manosim_qualified_rigid_flex_radius_selection_mj340_v1', radius_m=0.,
                  qualification_receipt=surface.binding(qualified, 'manufactured_radius_receipt'))
    return spec, policy, paths


def test_saved_preparation_preserves_complete_native_support_and_creation_time_hashes(tmp_path):
    spec, policy, _ = full_manufactured_inputs(tmp_path)
    receipt, row = surface.prepare_saved_candidate(20, spec, tmp_path/'output', radius_policy=policy)
    assert receipt['topology']['vertices'] == 778 and receipt['topology']['faces'] == 1552
    assert receipt['full_face_deletion_count'] == 0 and receipt['convexification'] is False
    assert receipt['compiled_readback_status'] == 'NOT_RUN' and receipt['finger_articulation_allowed'] is False
    assert all(receipt[key] is False for key in surface.FLAGS)
    for artifact in receipt['artifacts'] + [row]: surface.verify_binding(artifact)
    values = surface.arrays(tmp_path/'output/candidate_surface.npz')
    np.testing.assert_array_equal(values['native_face_ids'], np.arange(1552))
    assert receipt['array_hashes']['native_faces'] == surface.array_hash(values['native_faces'])
    with pytest.raises(FileExistsError): surface.prepare_saved_candidate(20, spec, tmp_path/'output', radius_policy=policy)


@pytest.mark.parametrize('change', ['shape_dtype', 'shape_value', 'shape_file_hash', 'bound_byte', 'missing_part', 'palm_frame'])
def test_saved_shape_membership_and_provenance_refusals_precede_output(tmp_path, change):
    spec, policy, paths = full_manufactured_inputs(tmp_path)
    if change in ('shape_dtype', 'shape_value'):
        shape = np.load(paths['shape_asset'])
        if change == 'shape_dtype': shape = shape.astype(np.float64)
        else: shape[0] += .125
        np.save(paths['shape_asset'], shape)
        spec['inputs']['shape_asset'] = surface.binding(paths['shape_asset'], 'shape_asset')
        spec['expected_shape_sha256'] = surface.sha(paths['shape_asset'])
    elif change == 'shape_file_hash': spec['expected_shape_sha256'] = '0'*64
    elif change == 'bound_byte': paths['literal_pose'].write_text('{}')
    elif change == 'missing_part':
        np.savez_compressed(paths['membership_provenance'], native_face_labels=np.zeros(1552, int))
        spec['inputs']['membership_provenance'] = surface.binding(paths['membership_provenance'], 'membership')
    elif change == 'palm_frame':
        values = surface.arrays(paths['kinematic_readback']); values['compiled_body_origins_world_m'][0, 0] += 1e-15
        np.savez_compressed(paths['kinematic_readback'], **values)
        spec['inputs']['kinematic_readback'] = surface.binding(paths['kinematic_readback'], 'kinematic')
    with pytest.raises(ValueError): surface.prepare_saved_candidate(20, spec, tmp_path/'output', radius_policy=policy)
    assert not (tmp_path/'output').exists()
