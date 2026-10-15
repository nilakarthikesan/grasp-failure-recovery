"""Finite static diagnostics for a capsule/convex solid union and a saved skin.

Exterior union distances are exact to the represented primitives. Interior
distance is bounded, not confused with the minimum component-boundary distance.
Skin occupancy uses oriented winding, not a claim of a non-self-intersecting
physical solid. No integration, fitting, or acceptance policy lives here.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def _points(value):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError("points must be finite (N,3)")
    return result


def nearest_triangles(points, triangles):
    """Exact finite triangle search; return distance, point, face and barycentrics."""
    import trimesh
    points = _points(points)
    triangles = np.asarray(triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("nonempty (F,3,3) triangles required")
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    if not np.isfinite(triangles).all() or np.any(np.linalg.norm(cross, axis=1) <= 1e-16):
        raise ValueError("finite nondegenerate triangles required")
    distances, closest, faceids = [], [], []
    for start in range(0, len(points), 32):
        query = points[start:start+32]
        tiled = np.tile(triangles, (len(query), 1, 1))
        candidates = trimesh.triangles.closest_point(tiled, np.repeat(query, len(triangles), axis=0))
        candidates = candidates.reshape(len(query), len(triangles), 3)
        ds = np.linalg.norm(candidates-query[:, None], axis=2)
        indices = ds.argmin(axis=1)
        closest.extend(candidates[np.arange(len(query)), indices])
        distances.extend(ds[np.arange(len(query)), indices])
        faceids.extend(indices)
    closest = np.asarray(closest).reshape(-1, 3)
    faceids = np.asarray(faceids, dtype=int)
    bary = trimesh.triangles.points_to_barycentric(triangles[faceids], closest)
    return np.asarray(distances), closest, faceids, bary


class Capsule:
    def __init__(self, start, end, radius):
        self.start, self.end = _points([start, end])
        self.radius = float(radius)
        self.delta = self.end-self.start
        self.length2 = float(self.delta@self.delta)
        if self.length2 <= 1e-18 or not np.isfinite(radius) or radius <= 0:
            raise ValueError("positive capsule length and radius required")

    def signed(self, points):
        points = _points(points)
        along = np.clip((points-self.start)@self.delta/self.length2, 0, 1)
        centers = self.start+along[:, None]*self.delta
        return np.linalg.norm(points-centers, axis=1)-self.radius

    def nearest(self, points):
        points = _points(points)
        along = np.clip((points-self.start)@self.delta/self.length2, 0, 1)
        centers = self.start+along[:, None]*self.delta
        delta = points-centers
        length = np.linalg.norm(delta, axis=1)
        normals = delta/np.maximum(length[:, None], 1e-300)
        valid = length > 1e-12
        # Axis projections have many equally close normals; this placeholder is
        # excluded by valid, even when used in a component-distance diagnostic.
        axis = self.delta/np.sqrt(self.length2)
        seed = np.eye(3)[np.argmin(abs(axis))]
        perpendicular = np.cross(axis, seed)
        normals[~valid] = perpendicular/np.linalg.norm(perpendicular)
        return length-self.radius, centers+self.radius*normals, normals, valid


class ConvexSolid:
    def __init__(self, vertices):
        self.vertices = _points(vertices)
        hull = ConvexHull(self.vertices)
        self.equations = hull.equations
        faces = hull.simplices.copy()
        tri = self.vertices[faces]
        inward = np.einsum('ij,ij->i', np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), hull.equations[:, :3]) < 0
        faces[inward] = faces[inward][:, [0, 2, 1]]
        self.triangles = self.vertices[faces]
        self.normals = hull.equations[:, :3]

    def contains(self, points):
        return np.max(_points(points)@self.equations[:, :3].T+self.equations[:, 3], axis=1) <= 0

    def signed(self, points):
        distance, _, _, _ = nearest_triangles(points, self.triangles)
        return np.where(self.contains(points), -distance, distance)

    def nearest(self, points):
        distance, closest, face, bary = nearest_triangles(points, self.triangles)
        signed = np.where(self.contains(points), -distance, distance)
        return signed, closest, self.normals[face], bary.min(axis=1) > 1e-7


class SolidUnion:
    def __init__(self, components):
        self.components = tuple(components)
        if not self.components:
            raise ValueError("nonempty union required")

    def contains(self, points):
        points = _points(points)
        return np.logical_or.reduce([c.contains(points) if isinstance(c, ConvexSolid) else c.signed(points) <= 0 for c in self.components])

    def query(self, points, *, tolerance=1e-9):
        """Exterior exact distance; interior lower/upper penetration bounds.

        Lower = maximum depth in any containing component. Upper = nearest of
        six exterior axis witnesses, refined by 36 bisections. Those segments
        may cross disconnected components: the witness remains an upper bound,
        but is not asserted to be the first ray exit or the shortest exit.
        """
        points = _points(points)
        rows = [c.nearest(points) for c in self.components]
        signed = np.stack([r[0] for r in rows], axis=1)
        inside = signed.min(axis=1) < -tolerance
        boundary = (~inside) & (signed.min(axis=1) <= tolerance)
        which = abs(signed).argmin(axis=1)
        closest = np.stack([r[1] for r in rows], axis=1)[np.arange(len(points)), which]
        normals = np.stack([r[2] for r in rows], axis=1)[np.arange(len(points)), which]
        regular = np.stack([r[3] for r in rows], axis=1)[np.arange(len(points)), which]
        at_closest = np.stack([c.signed(closest) for c in self.components], axis=1)
        exposed = at_closest.min(axis=1) >= -tolerance
        active = (abs(at_closest) <= tolerance).sum(axis=1)
        normal_valid = (~inside) & (~boundary) & exposed & regular & (active == 1)
        exterior_distance = np.min(abs(signed), axis=1)
        exterior_distance[inside] = np.nan
        lower = np.where(inside, np.maximum(-signed.min(axis=1), 0), 0)
        upper = np.zeros(len(points))
        if inside.any():
            origins = points[inside]
            candidates = []
            for direction in np.concatenate((np.eye(3), -np.eye(3))):
                low = np.zeros(len(origins))
                high = np.maximum(lower[inside]*2, 1e-6)
                for _ in range(40):
                    enclosed = self.contains(origins+high[:, None]*direction)
                    if not enclosed.any():
                        break
                    high[enclosed] *= 2
                else:
                    raise RuntimeError("failed to find exterior witness")
                for _ in range(36):
                    mid = (low+high)/2
                    enclosed = self.contains(origins+mid[:, None]*direction)
                    low = np.where(enclosed, mid, low)
                    high = np.where(enclosed, high, mid)
                if self.contains(origins+high[:, None]*direction).any():
                    raise AssertionError("upper witnesses must be exterior")
                candidates.append(high)
            upper[inside] = np.min(candidates, axis=0)
            if np.any(upper[inside]+tolerance < lower[inside]):
                raise AssertionError("invalid penetration interval")
        return dict(inside=inside, boundary=boundary, exterior_distance=exterior_distance,
                    closest=closest, normals=normals, normal_valid=normal_valid,
                    closest_component_projection_buried=~exposed,
                    penetration_lower=lower, penetration_upper=upper)


def oriented_skin(vertices, faces):
    """Check closed oriented edge topology; globally orient by signed volume.

    Does not certify absence of nonadjacent triangle intersections.
    """
    vertices = _points(vertices)
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError("integer (F,3) faces required")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces outside vertex range")
    faces = faces.copy()
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    keys, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    orientation = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1), minlength=len(keys))
    closed = bool(np.all(counts == 2) and np.all(orientation == 0))
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    nondegenerate = bool(np.all(np.linalg.norm(cross, axis=1) > 1e-16))
    volume = float(np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()/6)
    flipped = volume < 0
    if flipped:
        faces = faces[:, [0, 2, 1]]
    return vertices[faces], dict(closed_oriented_edges=closed, nondegenerate=nondegenerate,
                                 absolute_signed_volume_m3=abs(volume), global_winding_flipped=flipped,
                                 self_intersection_certified_absent=False,
                                 winding_eligible=closed and nondegenerate and abs(volume) > 1e-15)


def winding(points, triangles):
    """Generalized oriented winding using exact triangle solid angles."""
    points = _points(points)
    output = []
    for start in range(0, len(points), 24):
        q = triangles[None]-points[start:start+24, None, None]
        a, b, c = q[:, :, 0], q[:, :, 1], q[:, :, 2]
        lengths = np.linalg.norm(q, axis=3)
        numerator = np.einsum('nfi,nfi->nf', a, np.cross(b, c))
        denominator = (lengths.prod(axis=2)+np.einsum('nfi,nfi->nf', a, b)*lengths[:, :, 2]
                       +np.einsum('nfi,nfi->nf', b, c)*lengths[:, :, 0]
                       +np.einsum('nfi,nfi->nf', c, a)*lengths[:, :, 1])
        output.extend(np.arctan2(numerator, denominator).sum(axis=1)/(2*np.pi))
    return np.asarray(output)


def skin_query(points, triangles, *, eligible, side_offset=1e-6):
    distance, closest, face, bary = nearest_triangles(points, triangles)
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals = cross[face]/np.linalg.norm(cross[face], axis=1)[:, None]
    wn = winding(points, triangles) if eligible else np.full(len(points), np.nan)
    binary = ((abs(wn) < 1e-5) | (abs(wn-1) < 1e-5)) & (distance > side_offset)
    normal_valid = np.zeros(len(points), dtype=bool)
    candidates = np.flatnonzero((bary.min(axis=1) > 1e-5) & (distance <= .005) & binary)
    if eligible and len(candidates):
        outer = winding(closest[candidates]+side_offset*normals[candidates], triangles)
        inner = winding(closest[candidates]-side_offset*normals[candidates], triangles)
        # This witnesses local outward orientation at the nearest patch, and
        # refuses ambiguous overlapping skin. It is not global injectivity.
        normal_valid[candidates] = (abs(outer) < 1e-5) & (abs(inner-1) < 1e-5)
    return dict(distance=distance, closest=closest, normals=normals,
                winding=wn, occupancy_resolved=binary, inside=binary & (wn > .5),
                normal_valid=normal_valid)
