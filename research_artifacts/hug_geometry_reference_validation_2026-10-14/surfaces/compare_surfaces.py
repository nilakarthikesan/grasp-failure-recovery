"""Static reference retargeter surfaces; no inference or physics integration."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import pickle
import time
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

COUNTS = {'actual': 0, 'selftests': 0}
TESTING = False
def forbidden(*args, **kwargs):
    COUNTS['selftests' if TESTING else 'actual'] += 1
    raise RuntimeError('Static reference surface audit forbids physics integration')
for api in ('mj_step', 'mj_step1', 'mj_step2'):
    setattr(mujoco, api, forbidden)

ROOT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
PREV = ROOT / 'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces'
spec = importlib.util.spec_from_file_location('frozen_surface_geometry', PREV / 'audit_other_surfaces.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
for api in ('mj_step', 'mj_step1', 'mj_step2'):
    setattr(mujoco, api, forbidden)
TESTING = True
for api in ('mj_step', 'mj_step1', 'mj_step2'):
    try:
        getattr(mujoco, api)(None, None)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Guard failed')
TESTING = False

def read(path):
    return json.loads(Path(path).read_text())

def calculate(q, area, nd, ninside, npart, tri, labels, nnorm, hand, objects):
    rows = np.arange(len(q))
    ndist = nd[rows, npart]
    sd = np.full_like(nd, np.inf)
    sq = np.full((len(q), len(a.PARTS), 3), np.nan)
    sn = np.full_like(sq, np.nan)
    sinside = np.zeros(len(q), dtype=bool)
    for g in hand:
        # Evaluate all points to keep continuous-distance comparisons finite.
        signed, closest, normal = a.geom_query(g, q)
        sinside |= signed < 0
        part = a.PARTS.index(g['part'])
        distance = np.abs(signed)
        update = distance < sd[:, part]
        sd[update, part] = distance[update]
        sq[update, part] = closest[update]
        sn[update, part] = normal[update]
    assert np.isfinite(sd).all()
    spart = sd.argmin(axis=1)
    sdist = sd[rows, spart]
    sc = sq[rows, spart]
    snormal = sn[rows, spart]
    exterior = (~ninside) & (~sinside) & (ndist > 1e-8) & (sdist > 1e-8)
    paired5 = exterior & (ndist <= .005) & (sdist <= .005)
    ni = np.flatnonzero(paired5)
    # Only this conditional subset needs exact native nearest points/normals.
    nq = np.full_like(q, np.nan)
    nnormal = np.full_like(q, np.nan)
    if len(ni):
        native_d, native_q, native_ix = a.nearest_parts(q[ni], tri, labels, len(a.PARTS))
        assert np.max(np.abs(native_d[:, :6] - nd[ni, :6])) < 1e-12
        assert np.array_equal(labels[native_ix], npart[ni])
        nq[ni] = native_q
        nnormal[ni] = nnorm[native_ix]
    angles = np.full(len(q), np.nan)
    angles[ni] = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', nnormal[ni], snormal[ni]), -1, 1)))
    separation = np.linalg.norm(nq - sc, axis=1) * 1000
    nobj = np.zeros(len(q), dtype=bool)
    sobj = np.zeros(len(q), dtype=bool)
    nobj[ni] = a.object_membership(nq[ni], objects)
    sobj[ni] = a.object_membership(sc[ni], objects)
    def pair(sel):
        return {
            'query_count': int(sel.sum()), 'area_mm2': float(area[sel].sum()),
            'normal_angle_deg': a.stats(angles[sel], area[sel]),
            'closest_point_separation_mm': a.stats(separation[sel], area[sel]),
            'both_closest_inside_collision_object_count': int((sel & nobj & sobj).sum()),
            'one_closest_inside_collision_object_count': int((sel & (nobj ^ sobj)).sum()),
            'neither_closest_inside_collision_object_count': int((sel & ~nobj & ~sobj).sum()),
            'neither_inside_object_normal_angle_deg': a.stats(angles[sel & ~nobj & ~sobj], area[sel & ~nobj & ~sobj]),
        }
    result = {
        'query_count': len(q), 'query_area_mm2': float(area.sum()),
        'native_occupied_area_mm2': float(area[ninside].sum()),
        'mapped_occupied_area_mm2': float(area[sinside].sum()), 'bands': {},
        'paired_unsigned_distance_change_mm': {},
    }
    for part_ix, part in enumerate(a.PARTS):
        available = np.isfinite(nd[:, part_ix])
        near = available & (nd[:, part_ix] <= .005)
        result['paired_unsigned_distance_change_mm'][part] = {
            'native_reference_available_query_count': int(available.sum()),
            'all_visual_queries': a.stats((sd[available, part_ix] - nd[available, part_ix]) * 1000, area[available]),
            'native_within_5mm_query_count': int(near.sum()),
            'native_within_5mm': a.stats((sd[near, part_ix] - nd[near, part_ix]) * 1000, area[near]),
            'meaning': 'mapped minus native unsigned distance; negative can be introduced/repair proximity, not automatically fidelity improvement',
        }
    for band in (1, 2, 5):
        b = band / 1000
        nnear, snear = ndist <= b, sdist <= b
        native_exterior = (~ninside) & (ndist > 1e-8) & nnear
        mapped_exterior = (~sinside) & (sdist > 1e-8) & snear
        common = exterior & nnear & snear
        same = common & (npart == spart)
        result['bands'][str(band)] = {
            'whole_hand_unsigned_near': a.coverage(nnear, snear, area),
            'occupied_or_near': a.coverage(ninside | nnear, sinside | snear, area),
            'independent_per_part_unsigned': {part: a.coverage(nd[:, i] <= b, sd[:, i] <= b, area) for i, part in enumerate(a.PARTS)},
            'exterior_nearest_labels': {
                'common_same_label': pair(same),
                'common_different_label': pair(common & (npart != spart)),
                'native_exterior_near_count': int(native_exterior.sum()),
                'mapped_exterior_near_count': int(mapped_exterior.sum()),
                'native_near_mapped_not_exterior_near_area_mm2': float(area[native_exterior & ~mapped_exterior].sum()),
                'mapped_near_native_not_exterior_near_area_mm2': float(area[mapped_exterior & ~native_exterior].sum()),
            },
        }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-report', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=Path('/private/tmp/hug-other-surfaces-2026-10-14'))
    parser.add_argument('--out', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    started = time.perf_counter()
    protocol_path = args.out / 'protocol_v3.json'
    protocol_hash = a.sha(protocol_path)
    protocol = read(protocol_path)
    assert a.sha(PREV / 'audit_other_surfaces.py') == protocol['previous_query_library_sha256']
    assert a.sha(PREV / 'visual_reference_surfaces/hug_visual_reference_surfaces.json') == protocol['previous_visual_report_sha256']
    assert a.sha(args.reference_report.parent / 'protocol.json') == protocol['reference_protocol_sha256']
    ref_hash = a.sha(args.reference_report)
    reference = read(args.reference_report)
    assert reference['protocol_sha256'] == protocol['reference_protocol_sha256']
    assert a.sha(a.URDF) == reference['urdf_sha256']
    urdf_xml = ET.parse(a.URDF).getroot()
    physical_ranges = {
        joint.get('name'): (float(joint.find('limit').get('lower')), float(joint.find('limit').get('upper')))
        for joint in urdf_xml.findall('joint') if joint.find('limit') is not None
    }
    refs = {c['native_sha256']: c for c in reference['cases']}
    prior = ROOT / 'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals'
    assert a.sha(prior / 'existing_hug_inventory.json') == protocol['previous_inventory_sha256']
    assert a.sha(prior / 'static_saved_hug_fk.json') == protocol['previous_fk_sha256']
    inventory, fk = read(prior / 'existing_hug_inventory.json'), read(prior / 'static_saved_hug_fk.json')
    fkmap = {c['native_sha256']: c for c in fk['cases']}
    old = read(PREV / 'visual_reference_surfaces/hug_visual_reference_surfaces.json')
    oldmap = {c['native_sha256']: c for c in old['cases']}
    expected = old['input_provenance']['licensed_mano_asset_sha256']
    assert a.sha(a.ASSET) == expected
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        asset = pickle.loads(a.ASSET.read_bytes(), encoding='latin1')
    weights = np.asarray(asset['weights'])
    grouped = np.column_stack([weights[:, ids].sum(axis=1) for ids in a.GROUPS.values()])
    protected_path = Path('/private/tmp/hug_current_stage_protected_2026-10-14.json')
    protected = read(protected_path)
    for path, h in protected.items():
        assert a.sha(path) == h
    caches = {r['path']: r for r in protocol['cached_query_sha256']}
    records = []
    first = set()
    maximum_regression_error = 0.0
    for selected in protocol['case_selection']:
        case = inventory['modern_cases'][selected['source_inventory_index']]
        assert case['native']['sha256'] == selected['native_sha256']
        h = case['native']['sha256']
        meta, tri, labels, normals, hand, objects, model, data, initial, objadr, object_pos, object_R = a.load_case(case, grouped, fkmap)
        object_id = a.ident(model, mujoco.mjtObj.mjOBJ_BODY, 'object')
        rootadr = int(model.jnt_qposadr[a.ident(model, mujoco.mjtObj.mjOBJ_JOINT, 'world_joint')])
        degrees = [2, 1] if case['object_id'] not in first else [2]
        first.add(case['object_id'])
        caseid = case['object_id'] + '_' + Path(case['native']['path']).parent.name
        rec = {'object_id': case['object_id'], 'native_sha256': h, 'source_case': Path(case['native']['path']).parent.name, 'input_sha256': meta['input_sha256'], 'resolutions': {}}
        loaded = {}
        for density in degrees:
            cache_name = f'visual_reference_surfaces/{caseid}_density{density}mm_queries.npz'
            cache_path = args.cache_root / cache_name
            assert a.sha(cache_path) == caches[cache_name]['sha256']
            with np.load(cache_path, allow_pickle=False) as f:
                loaded[density] = {key: f[key].copy() for key in ('query_object_m', 'area_mm2', 'native_distance_by_part_m', 'native_inside', 'native_nearest_part')}
            rec['resolutions'][str(density)] = {'query_cache_sha256': caches[cache_name]['sha256'], 'methods': {}}
        for method in protocol['methods']:
            target = refs[h]['methods'][method]
            assert target['status'] in ('computed', 'stored'), target['status']
            qpos = np.asarray(target['qpos_rad'], float)
            assert len(qpos) == len(target['joint_names']) == 24
            bound_checks = []
            for name, value in zip(target['joint_names'], qpos, strict=True):
                jid = a.ident(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                low, high = physical_ranges[name]
                assert low - 1e-9 <= value <= high + 1e-9
                scene_low, scene_high = model.jnt_range[jid]
                bound_difference = max(abs(scene_low - low), abs(scene_high - high))
                assert scene_low - bound_difference - 1e-9 <= value <= scene_high + bound_difference + 1e-9
                overflow = max(scene_low - value, value - scene_high, 0.0)
                bound_checks.append({'joint': name, 'source_URDF_range_rad': [low, high], 'scene_range_rad': [float(scene_low), float(scene_high)], 'bound_representation_difference_rad': float(bound_difference), 'strict_scene_exceedance_rad': float(overflow), 'strict_scene_1e9_tolerance_pass': bool(overflow <= 1e-9), 'qpos_unchanged': True})
                data.qpos[int(model.jnt_qposadr[jid])] = value
            data.qpos[rootadr:rootadr+3] = target['root_target_position_m']
            data.qpos[rootadr+3:rootadr+7] = target['root_target_quaternion_wxyz']
            if model.nmocap:
                data.mocap_pos[0] = target['root_target_position_m']
                data.mocap_quat[0] = target['root_target_quaternion_wxyz']
            mujoco.mj_forward(model, data)
            assert data.time == 0 and np.array_equal(data.qpos[objadr:objadr+7], initial)
            assert np.array_equal(data.xpos[object_id], object_pos)
            assert np.array_equal(data.xmat[object_id].reshape(3, 3), object_R)
            palm_id = a.ident(model, mujoco.mjtObj.mjOBJ_BODY, 'palm')
            assert np.linalg.norm(data.xpos[palm_id] - target['target_palm_world_position_m']) < 1e-7
            assert np.max(np.abs(data.xmat[palm_id].reshape(3, 3) - target['target_palm_world_rotation'])) < 1e-6
            newhand = []
            for g in hand:
                gg = a.geom_rep(model, data, g['id'], object_R, object_pos)
                gg['part'] = g['part']
                newhand.append(gg)
            nominal = data.qpos.copy()
            for density, f in loaded.items():
                tick = time.perf_counter()
                result = calculate(f['query_object_m'], f['area_mm2'], f['native_distance_by_part_m'], f['native_inside'], f['native_nearest_part'], tri, labels, normals, newhand, objects)
                result['seconds'] = time.perf_counter() - tick
                result['range_checks'] = {'full_source_URDF_physical_limits_pass': True, 'strict_scene_range_violation_count': sum(not check['strict_scene_1e9_tolerance_pass'] for check in bound_checks), 'joint_checks': bound_checks, 'policy': protocol['static_range_policy'], 'evaluator_acceptance_claimed': False}
                assert np.array_equal(data.qpos, nominal) and data.time == 0
                rec['resolutions'][str(density)]['methods'][method] = result
                print(json.dumps({'progress': caseid, 'method': method, 'density': density, 'seconds': result['seconds'], 'unsigned_iou_2mm': result['bands']['2']['whole_hand_unsigned_near']['jaccard']}), flush=True)
                if method == 'stored_current':
                    before = oldmap[h]['resolutions'][str(density)]
                    for band in (1, 2, 5):
                        for metric in ('whole_hand_unsigned_near', 'occupied_or_near'):
                            for key in ('native_area_mm2', 'shadow_area_mm2', 'intersection_area_mm2', 'union_area_mm2', 'jaccard'):
                                delta = abs(result['bands'][str(band)][metric][key] - before['bands'][str(band)][metric][key])
                                assert delta < 1e-8, (caseid, density, band, metric, key, delta)
                                maximum_regression_error = max(maximum_regression_error, delta)
                        for part in a.PARTS:
                            now = result['bands'][str(band)]['independent_per_part_unsigned'][part]
                            then = before['bands'][str(band)]['independent_per_part_unsigned'][part]
                            for key in ('native_area_mm2', 'shadow_area_mm2', 'intersection_area_mm2'):
                                assert abs(now[key] - then[key]) < 1e-8
            assert np.array_equal(data.qpos[objadr:objadr+7], initial)
        for path, expected in meta['input_sha256'].items():
            assert a.sha(path) == expected
        rec['static_invariants'] = {'physics_steps': 0, 'all_time_zero': True, 'initial_object_qpos_and_world_pose_unchanged': True, 'nominal_qpos_fixed_during_measurements': True}
        records.append(rec)
    assert a.sha(protocol_path) == protocol_hash and a.sha(args.reference_report) == ref_hash
    for path, expected in protected.items():
        assert a.sha(path) == expected
    assert COUNTS['actual'] == 0
    report = {'schema': 'hug_existing_static_reference_surfaces_v3', 'protocol_sha256': protocol_hash, 'script_sha256': a.sha(__file__), 'reference_report_sha256': ref_hash, 'query_library_sha256': a.sha(a.__file__), 'physics_steps': 0, 'guards': COUNTS, 'new_eval_data': False, 'new_inference': False, 'predictor_training': False, 'protected_hashes_unchanged': True, 'previous_current_surface_regression_max_difference': maximum_regression_error, 'runtime_seconds': time.perf_counter() - started, 'cases': records, 'limitations': protocol['limitations']}
    a.write(args.out / 'surface_reference_comparison.json', report)
    print(json.dumps({'completed': True, 'runtime_seconds': report['runtime_seconds'], 'report': str(args.out / 'surface_reference_comparison.json')}), flush=True)

if __name__ == '__main__':
    main()
