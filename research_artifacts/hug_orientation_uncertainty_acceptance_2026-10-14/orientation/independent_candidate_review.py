"""Analytic mesh candidate regression only; no grasp assets or physics."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    folder = Path(__file__).resolve().parent
    script = folder / 'check_orientation.py'
    spec = importlib.util.spec_from_file_location('orientation_review_target', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = module.a
    radius = np.array([.008, .011, .013])
    vertices = np.array([[x, y, z] for x in (-radius[0], radius[0])
                         for y in (-radius[1], radius[1])
                         for z in (-radius[2], radius[2])])
    local_tri, equations, _ = a.hull(vertices)
    rotation = Rotation.from_euler('xyz', [.4, -.3, .2]).as_matrix()
    center = np.array([.060, -.020, .010])
    geom = {'id': 10, 'kind': a.mujoco.mjtGeom.mjGEOM_MESH,
            'pos': center, 'R': rotation, 'size': radius, 'part': 'palm',
            'tri': local_tri, 'tri_object': local_tri @ rotation.T + center,
            'equations': equations}
    query = np.array([[0., 0., 0.], [.001, -.002, .003], [-.003, .001, .004]])
    native_distance = np.array([.001, .004, .002])
    signed, _, _ = a.geom_query(geom, query)
    mapped_distance = abs(signed)
    assert np.all(mapped_distance > .005)
    selected = (native_distance <= .005) | (mapped_distance <= .005)
    world_vertices = geom['tri_object'].reshape(-1, 3)
    gap = np.maximum(np.maximum(world_vertices.min(0) - query,
                               query - world_vertices.max(0)), 0)
    lower = np.linalg.norm(gap, axis=1)
    tol = 1e-8
    old_candidates = selected & (lower <= .005 + tol)
    corrected_candidates = selected & (lower <= mapped_distance + tol)
    assert np.all(selected) and not np.any(old_candidates)
    assert np.all(corrected_candidates)
    assert np.all(lower <= mapped_distance + tol)
    report = {
        'schema': 'independent_orientation_candidate_review_v1',
        'review_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'orientation_script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
        'query_count': len(query),
        'query_object_m': query.tolist(),
        'declared_manufactured_native_near_distances_m': native_distance.tolist(),
        'actual_rotated_target_mesh_distances_m': mapped_distance.tolist(),
        'target_mesh_AABB_distance_lowerbounds_m': lower.tolist(),
        'fixed_5mm_filter_retained_count': int(old_candidates.sum()),
        'full_minimum_plus_tie_filter_retained_count': int(corrected_candidates.sum()),
        'tie_tolerance_m': tol,
        'passed': True,
        'integration_attempts': module.COUNTS['actual'],
        'no_grasp_assets_or_inference_or_eval_data': True,
        'scope': 'Manufactured candidate completeness regression for native-near/mapped-mesh-far queries. Native distances are declared diagnostic values, not reconstructed grasps. No claim about continuum normal bounds or actual hand mapping.'
    }
    assert report['integration_attempts'] == 0
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
