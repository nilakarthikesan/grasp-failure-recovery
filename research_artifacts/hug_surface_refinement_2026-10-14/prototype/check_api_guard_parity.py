"""Replay stored engineering assessments through API derivative; no geometry run.

This verifies selection parity, not the accuracy of the stored assessments.
Their geometry is independently audited in the validation sibling bundle.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np

BASE = Path(__file__).parent

def read(path):
    return json.loads(Path(path).read_bytes())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    frozen = BASE / 'guarded_refinement.py'
    derivative = BASE / 'production_candidate/surface_refinement_guard.py'
    old = load('guard_frozen_parity', frozen)
    new = load('guard_api_parity', derivative)
    protocol = read(BASE / 'protocol.json')
    urdf, = [Path(p) for p in protocol['source_sha256'] if p.endswith('.urdf')]
    assert sha(urdf) == protocol['source_sha256'][str(urdf)]
    limits = {
        j.get('name'): (float(j.find('limit').get('lower')), float(j.find('limit').get('upper')))
        for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None
    }
    raw_path = BASE / 'candidate_targets.json'
    guarded_path = BASE / 'guarded_candidate_targets.json'
    raw = read(raw_path)
    guarded = read(guarded_path)
    assert len(raw['cases']) == len(guarded['cases']) == 6
    rows = []
    for r, g in zip(raw['cases'], guarded['cases'], strict=True):
        assert r['native_sha256'] == g['native_sha256']
        q0 = np.asarray(r['original_qpos_rad'])
        qr = np.asarray(r['qpos_rad'])
        baseline = g['engineering_guard']['baseline']
        attempts = g['engineering_guard']['attempted_alphas']
        recorded = [(q0, baseline)] + [
            (q0 + row['alpha'] * (qr - q0), row['assessment']) for row in attempts
        ]
        def assess(q):
            matches = [value for qq, value in recorded if np.array_equal(q, qq)]
            assert len(matches) == 1, 'API queried an unrecorded or ambiguous interpolant'
            return matches[0]
        bounds = np.asarray([limits[n] for n in r['joint_names']])
        old_q, old_result = old.guarded_joint_selection(q0, qr, bounds, assess)
        new_q, new_result = new.guarded_joint_selection(q0, qr, bounds, assess)
        assert np.array_equal(new_q, old_q)
        assert np.array_equal(new_q, np.asarray(g['qpos_rad']))
        assert new_result == old_result == g['engineering_guard']
        rows.append({'native_sha256': r['native_sha256'], 'source_inventory_index': r['source_inventory_index'],
                     'selected_alpha': new_result['selected_alpha'], 'exact_target_and_diagnostic_parity': True})
    result = {'schema': 'hug_surface_api_guard_replay_parity_v1', 'cases': rows,
              'input_sha256': {str(p): sha(p) for p in (frozen, derivative, raw_path, guarded_path, urdf)},
              'script_sha256': sha(__file__), 'geometry_recomputed': False,
              'physics_steps': 0, 'new_inference': False, 'new_eval_data': False,
              'meaning': 'API selection parity only; does not validate geometry or scientific acceptance.'}
    (BASE / 'api_guard_parity.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'case_count': len(rows), 'exact_target_and_diagnostic_parity': True, 'geometry_recomputed': False}))

if __name__ == '__main__':
    main()
