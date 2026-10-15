"""Manufactured geometry controls, including buried-interface counterexamples."""
import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('trimesh')
from grasp_failure_prediction.evaluation.static_boundary_audit import (
    Capsule, ConvexSolid, SolidUnion, nearest_triangles, oriented_skin, skin_query, winding,
)


def cube():
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return ConvexSolid(vertices)


def tetra():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


def test_capsule_signed_closest_and_axis_normal_refusal():
    c = Capsule([0, 0, -1], [0, 0, 1], 1)
    signed, closest, normal, valid = c.nearest([[3, 0, 0], [0, 0, 3], [0, 0, 0]])
    np.testing.assert_allclose(signed, [2, 1, -1])
    np.testing.assert_allclose(closest[:2], [[1, 0, 0], [0, 0, 2]])
    np.testing.assert_allclose(normal[:2], [[1, 0, 0], [0, 0, 1]])
    assert valid.tolist() == [True, True, False]


def test_convex_exact_signed_inside_outside_corner():
    c = cube()
    signed, closest, _, _ = c.nearest([[0, 0, 0], [2, 2, 2], [1, .2, -.4]])
    np.testing.assert_allclose(signed, [-1, np.sqrt(3), 0], atol=1e-14)
    np.testing.assert_allclose(closest[1], [1, 1, 1])


def test_union_exterior_exact_exposed_projection():
    c = Capsule([0, 0, -1], [0, 0, 1], 1)
    d = Capsule([1, 0, -1], [1, 0, 1], 1)
    r = SolidUnion([c, d]).query([[3, 0, 0], [-2, 0, 0]])
    np.testing.assert_allclose(r['exterior_distance'], [1, 1])
    assert r['normal_valid'].all()
    assert not r['closest_component_projection_buried'].any()


def test_union_inside_buried_component_not_exposed_distance():
    # x=-.25 is on neither outer exit; the nearer individual surface lies inside
    # the second capsule. The shortest union exit is sqrt(1-.25**2) sideways.
    a = Capsule([-.5, 0, -2], [-.5, 0, 2], 1)
    b = Capsule([0, 0, 0], [0, 0, 4], 1)  # different axial extent
    r = SolidUnion([a, b]).query([[-.25, 0, 1]])
    assert r['inside'][0]
    assert np.isnan(r['exterior_distance'][0])
    assert r['closest_component_projection_buried'][0]
    exact = np.sqrt(1-.25**2)
    assert r['penetration_lower'][0] <= exact <= r['penetration_upper'][0]+1e-9
    assert not r['normal_valid'][0]


def test_single_convex_penetration_interval_is_tight():
    r = SolidUnion([cube()]).query([[.3, .2, -.4]])
    np.testing.assert_allclose(r['penetration_lower'], [.6])
    np.testing.assert_allclose(r['penetration_upper'], [.6], atol=1e-9)


def test_tangent_component_intersection_refuses_unique_normal():
    a = Capsule([0, 0, -1], [0, 0, 1], 1)
    b = Capsule([2, 0, -1], [2, 0, 1], 1)
    r = SolidUnion([a, b]).query([[1, 0, 0]])
    assert r['boundary'][0]
    assert not r['normal_valid'][0]


def test_convex_edge_normal_refused():
    r = SolidUnion([cube()]).query([[2, 2, .3]])
    assert not r['normal_valid'][0]
    np.testing.assert_allclose(r['exterior_distance'], [np.sqrt(2)])


def test_closed_winding_and_reversed_orientation():
    v, f = tetra()
    t, m = oriented_skin(v, f)
    assert m['winding_eligible'] and not m['global_winding_flipped']
    np.testing.assert_allclose(winding([[.1, .1, .1], [2, 2, 2]], t), [1, 0], atol=1e-14)
    t2, m2 = oriented_skin(v, f[:, [0, 2, 1]])
    assert m2['global_winding_flipped']
    np.testing.assert_allclose(winding([[.1, .1, .1]], t2), [1], atol=1e-14)


def test_open_or_inconsistently_oriented_skin_refused():
    v, f = tetra()
    assert not oriented_skin(v, f[:-1])[1]['winding_eligible']
    f[0] = f[0, [0, 2, 1]]
    assert not oriented_skin(v, f)[1]['winding_eligible']


def test_skin_two_sided_local_normal_and_surface_refusal():
    v, f = tetra()
    t, m = oriented_skin(v, f)
    r = skin_query([[.2, .2, -.001], [.2, .2, .001], [.2, .2, 0]], t, eligible=m['winding_eligible'])
    assert r['normal_valid'].tolist() == [True, True, False]
    assert r['inside'].tolist() == [False, True, False]
    np.testing.assert_allclose(r['normals'][:2], [[0, 0, -1], [0, 0, -1]])


def test_ineligible_skin_has_no_occupancy_or_normal_claim():
    v, f = tetra()
    t, _ = oriented_skin(v, f)
    r = skin_query([[.2, .2, -.001]], t, eligible=False)
    assert not r['occupancy_resolved'].any()
    assert not r['normal_valid'].any()


def test_nearest_triangle_edge_vertex_and_face():
    t = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
    distance, closest, face, bary = nearest_triangles([[.2, .2, 1], [2, 0, 0]], t)
    np.testing.assert_allclose(distance, [1, 1])
    np.testing.assert_allclose(closest, [[.2, .2, 0], [1, 0, 0]])
    np.testing.assert_allclose(bary.sum(axis=1), [1, 1])
    assert (face == 0).all()


@pytest.mark.parametrize('start,end,radius', [([0, 0, 0], [0, 0, 0], 1), ([0, 0, 0], [1, 0, 0], -1), ([0, 0, 0], [1, 0, 0], float('nan'))])
def test_invalid_capsule_refused(start, end, radius):
    with pytest.raises(ValueError):
        Capsule(start, end, radius)


def test_rigid_covariance_of_union_exterior_and_normals():
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_rotvec([.2, .3, -.7]).as_matrix()
    pos = np.array([.3, -2., .5])
    a, b = np.array([0., 0., -1]), np.array([0., 0., 1])
    points = np.array([[2., .3, 0], [0, 0, 3]])
    r = SolidUnion([Capsule(a, b, 1)]).query(points)
    s = SolidUnion([Capsule(a@rot.T+pos, b@rot.T+pos, 1)]).query(points@rot.T+pos)
    np.testing.assert_allclose(r['exterior_distance'], s['exterior_distance'], atol=1e-14)
    np.testing.assert_allclose(r['normals']@rot.T, s['normals'], atol=1e-14)
