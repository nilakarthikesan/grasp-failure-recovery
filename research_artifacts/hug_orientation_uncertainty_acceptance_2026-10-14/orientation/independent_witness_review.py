"""Recompute two saved orientation witnesses only; static time-zero FK."""
from pathlib import Path
import hashlib
import importlib.util
import json
import pickle
import warnings
import numpy as np


def main():
    folder = Path(__file__).resolve().parent
    script = folder / 'check_orientation.py'
    spec = importlib.util.spec_from_file_location('orientation_witness_review', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = module.a
    p = module.read(folder / 'protocol.json')
    old_report_path = folder / 'hug_apple_scene_002_seed_0_orientation.json'
    old = module.read(old_report_path)
    prior = module.ROOT / 'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals'
    inventory = module.read(prior / 'existing_hug_inventory.json')
    fk = module.read(prior / 'static_saved_hug_fk.json')
    fkmap = {c['native_sha256']: c for c in fk['cases']}
    case = inventory['modern_cases'][old['source_inventory_index']]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        asset = pickle.loads(a.ASSET.read_bytes(), encoding='latin1')
    weights = np.asarray(asset['weights'])
    groupweights = np.column_stack([weights[:, ids].sum(1) for ids in a.GROUPS.values()])
    meta, tri, labels, normals, hand, _, model, data, initial, objadr, pos, rotation = a.load_case(case, groupweights, fkmap)
    qpos = data.qpos.copy()
    visual_tri, _, _ = module.visual_mesh(meta, model, data, pos, rotation)
    witnesses = old['resolutions']['1']['bands']['2']['common_same_label']['pointwise_sign_check']['witnesses']
    reviewed = []
    for direction, listed in witnesses.items():
        assert listed
        witness = listed[0]
        q = np.array([witness['query_object_m']])
        n_distance, nq, indices = a.nearest_parts(q, tri, labels, len(a.PARTS))
        native_normal = normals[indices[0]]
        native_part = a.PARTS[labels[indices[0]]]
        nmin = n_distance.min(1)
        target = []
        for geom in hand:
            signed, closest, normal = a.geom_query(geom, q)
            target.append((abs(signed[0]), geom, closest[0], normal[0], signed[0]))
        target.sort(key=lambda item: item[0])
        mapped_distance, chosen_geom, sq, mapped_normal, _ = target[0]
        object_normal = np.asarray(witness['object_outward_normal'])
        native_score = float(-np.dot(native_normal, object_normal))
        mapped_score = float(-np.dot(mapped_normal, object_normal))
        assert np.max(abs(nq[0] - witness['native_closest_point_object_m'])) < 1e-12
        assert np.max(abs(sq - witness['mapped_closest_point_object_m'])) < 1e-12
        assert abs(nmin[0] - witness['native_unsigned_distance_m']) < 1e-12
        assert abs(mapped_distance - witness['mapped_unsigned_distance_m']) < 1e-12
        assert np.max(abs(native_normal - witness['native_outward_normal'])) < 1e-12
        assert np.max(abs(mapped_normal - witness['mapped_outward_normal'])) < 1e-12
        assert abs(native_score - witness['native_facing_compatibility']) < 1e-12
        assert abs(mapped_score - witness['mapped_facing_compatibility']) < 1e-12
        assert native_part == chosen_geom['part'] == witness['native_part'] == witness['mapped_part']
        for normal in (native_normal, mapped_normal, object_normal):
            assert abs(np.linalg.norm(normal) - 1) < 1e-12
        native_inside, native_winding = a.winding_inside(q, tri)
        assert not native_inside[0]
        assert all(item[4] > 0 for item in target)
        point_classes, point_diagnostic = module.visual_membership(np.array([nq[0], sq]), visual_tri, p)
        assert np.array_equal(point_classes, [1, 1])
        other_depths = []
        for geom in hand:
            if geom['id'] == chosen_geom['id']:
                continue
            signed, _, _ = a.geom_query(geom, sq[None])
            other_depths.append(float(signed[0]))
        assert min(other_depths) >= -p['tie_distance_tolerance_m']
        count, bits, ambiguity, point_ambiguity = module.triangle_candidates(
            q, tri, labels, normals, nmin, nq, native_normal[None],
            p['tie_distance_tolerance_m'], p['normal_vector_difference_tolerance'])
        assert not ambiguity[0] and not point_ambiguity[0]
        assert (bits[0] & (bits[0] - 1)) == 0
        tied_target = [item for item in target if item[0] <= mapped_distance + p['tie_distance_tolerance_m']]
        for _, geom, closest, normal, _ in tied_target:
            assert geom['part'] == chosen_geom['part']
            assert np.linalg.norm(closest - sq) <= p['tie_distance_tolerance_m']
            assert np.linalg.norm(normal - mapped_normal) <= p['normal_vector_difference_tolerance']
        if chosen_geom['tri_object'] is not None:
            norm = a.tri_info(chosen_geom['tri_object'])[0]
            _, _, ambiguous, points_ambiguous = module.triangle_candidates(
                q, chosen_geom['tri_object'], np.zeros(len(norm), np.int64), norm,
                np.array([mapped_distance]), sq[None], mapped_normal[None],
                p['tie_distance_tolerance_m'], p['normal_vector_difference_tolerance'])
            assert not ambiguous[0] and not points_ambiguous[0]
        else:
            assert not module.primitive_ambiguous(chosen_geom, q, sq[None], p['tie_distance_tolerance_m'])[0]
        eps = p['facing_dot_numerical_deadzone']
        assert native_score * mapped_score < 0 and min(abs(native_score), abs(mapped_score)) > eps
        reviewed.append({'direction': direction, 'query_index': witness['query_index'],
                         'part': native_part, 'native_score': native_score,
                         'mapped_score': mapped_score,
                         'native_query_winding': float(native_winding[0]),
                         'mapped_query_min_positive_exterior_distance_m': min(item[4] for item in target),
                         'closest_visual_classes': point_classes.tolist(),
                         'target_closest_min_other_geom_signed_distance_m': min(other_depths),
                         'native_unique_label_point_normal': True,
                         'mapped_unique_label_point_normal_and_unburied': True,
                         'matches_published_witness_within_m': 1e-12})
    assert data.time == 0 and np.array_equal(data.qpos, qpos)
    assert np.array_equal(data.qpos[objadr:objadr + 7], initial)
    assert module.COUNTS['actual'] == 0
    for source, digest in meta['input_sha256'].items():
        assert a.sha(source) == digest
    report = {'schema': 'independent_orientation_witness_review_v1',
              'script_sha256': a.sha(__file__),
              'operational_orientation_script_sha256': a.sha(script),
              'orientation_protocol_sha256': a.sha(folder / 'protocol.json'),
              'source_case_report_sha256': a.sha(old_report_path),
              'native_sha256': case['native']['sha256'], 'witnesses': reviewed,
              'passed': True, 'physics_steps': 0, 'time_zero': True,
              'qpos_and_object_unchanged': True, 'input_hashes_unchanged': True,
              'new_inference': False, 'new_eval_data': False,
              'limitations': 'Independent rerun of two stored point witnesses using frozen source-geometry query implementations; not an independent algorithm or continuum normal/error certificate. Closed/winding/topology and numerical assumptions retained.'}
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
