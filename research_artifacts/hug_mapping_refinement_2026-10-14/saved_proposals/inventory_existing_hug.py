"""Inventory existing HUG sources and stored translations; no model/physics calls.

Writes only this audit directory. Reads trusted local JSON, NumPy and HUG pickle
artifacts. The selected command is the pre-force-close grasp endpoint, not final
actuated qpos, not the last nominal command, and not a newly optimized pose.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import pickle
import subprocess
import sys

import numpy as np

PRIMARY = Path('/Users/nilakarthikesan/grasp-failure-prediction')
WORK = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OUT = Path('/private/tmp/grasp-hug-existing-inventory-2026-10-14')
GROUP = PRIMARY / 'runs/fresh_object_scene_eval_2026-10-11'
MAPPING = {
    'retargeting.py': 'evaluation/retargeting.py',
    'pose_validation.py': 'evaluation/pose_validation.py',
    'runner.py': 'evaluation/runner.py',
    'case_runner.py': 'evaluation/case_runner.py',
    '../integrations/hug.py': 'integrations/hug.py',
    '../integrations/hug_frames.py': 'integrations/hug_frames.py',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def named_files(root, pattern):
    result = subprocess.run(['rg', '--files', '--hidden', '--no-ignore',
                             str(root / 'runs'), '-g', pattern],
                            check=True, capture_output=True, text=True)
    return sorted(Path(p) for p in result.stdout.splitlines())


def native_entry(path):
    entry = {'path': str(path), 'sha256': sha(path)}
    report = path.parent / 'inference_report.json'
    if report.is_file():
        d = load(report)
        entry.update(inference_report_path=str(report),
                     inference_report_sha256=sha(report),
                     real_hug_inference_passed=d.get('real_hug_inference_passed'),
                     recorded_proposal_hash_matches=d.get('proposal_sha256', '').removeprefix('sha256:') == entry['sha256'],
                     object_name=d.get('object_name'), seed=d.get('seed'),
                     input_tensor_hash=d.get('input_tensor_hash'),
                     hug_code_commit=d.get('hug_code_commit'),
                     checkpoint_sha256=d.get('checkpoint_sha256'))
    else:
        aggregate = path.parent.parent / 'report.json'
        entry.update(inference_report_path=None, real_hug_inference_passed=None,
                     aggregate_report_path=str(aggregate) if aggregate.is_file() else None)
    return entry


def main():
    primary_entries = [native_entry(p) for p in named_files(PRIMARY, 'proposal.pkl')]
    work_entries = [native_entry(p) for p in named_files(WORK, 'proposal.pkl')]
    hashes = defaultdict(list)
    for d in primary_entries + work_entries:
        hashes[d['sha256']].append(d['path'])
    episodes = defaultdict(list)
    for path in sorted((GROUP / 'collection_2080').glob('episode_*/collection_metadata.json')):
        d = load(path)
        episodes[d['proposal_sha256'].removeprefix('sha256:')].append((path, d))
    current_hashes = {k: sha(WORK / 'src/grasp_failure_prediction' / v)
                      for k, v in MAPPING.items()}
    cases = []
    for subgroup in ['inputs', 'inputs_new_objects']:
        for p in sorted((GROUP / subgroup / 'proposals').glob('scene_*/proposal.pkl')):
            native = native_entry(p)
            report = load(p.parent / 'inference_report.json')
            contract_path = p.parent / 'scene_contract.json'
            contract = load(contract_path)
            entries = episodes[native['sha256']]
            metadata_path, meta = next((path, d) for path, d in entries if d['noise_amplitude_rad'] == 0)
            episode = metadata_path.parent
            resolved = load(episode / 'resolved_case.json')
            params = resolved['execution_protocol']['parameters']
            dt = resolved['environment']['simulator']['control_timestep_s']
            with np.load(episode / 'trajectory.npz', allow_pickle=False) as trajectory:
                phases = trajectory['state']
                first = int(np.flatnonzero(phases == 'close_fingers')[0])
                row = first + int(round(params['close_duration_s'] / dt)) - 1
                assert phases[row] == 'close_fingers'
            with np.load(episode / 'actions.npz', allow_pickle=False) as actions:
                names = actions['joint_names'].tolist()
                target = actions['nominal_target_rad'][row]
                assert len(names) == len(set(names)) == len(target) == 24
                assert np.isfinite(target).all()
                command = {
                    'path': str(episode / 'actions.npz'), 'sha256': sha(episode / 'actions.npz'),
                    'target_field': 'nominal_target_rad', 'joint_names_field': 'joint_names',
                    'row_index': row, 'time_s': float(actions['time_s'][row]),
                    'joint_names': names, 'target_shape': list(target.shape),
                    'root_position_field': 'root_target_position_m',
                    'root_quaternion_field': 'root_target_quaternion_wxyz',
                    'position_and_rotation_frame': 'world robot free-root carrier target, not palm transform',
                    'selection_rule': 'first close_fingers index + round(close_duration/control_dt) - 1',
                    'meaning': 'stored original grasp endpoint before settle/force_close; grip_command=1.0',
                    'not_final_qpos_or_post_force_close_target': True,
                    'zero_applied_noise_at_selected_row': bool(np.all(actions['applied_noise_rad'][row] == 0)),
                }
            native_payload = pickle.loads(p.read_bytes())
            native_shapes = {k: list(np.asarray(v).shape) for k, v in native_payload.items()}
            mismatches = {
                k: {'recorded': meta['source_file_sha256'].get(k), 'current': 'sha256:' + expected}
                for k, expected in current_hashes.items()
                if meta['source_file_sha256'].get(k) != 'sha256:' + expected
            }
            episode_source_matches = meta['proposal_sha256'] == 'sha256:' + native['sha256']
            inference_report_matches = meta['inference_report_sha256'] == 'sha256:' + sha(p.parent / 'inference_report.json')
            files = {}
            for name in ['T_world_camera.npy', 'scene_state.npz', 'scene_contract.json']:
                path = p.parent / name
                files[name] = {'path': str(path), 'exists': path.is_file(), 'sha256': sha(path) if path.is_file() else None}
            with np.load(p.parent / 'scene_state.npz', allow_pickle=False) as state:
                state_shapes = {k: list(state[k].shape) for k in state.files}
            cases.append({
                'native': native, 'native_shapes': native_shapes,
                'object_id': contract['object_id'], 'geometry_content_hash': contract['geometry_content_hash'],
                'observation_group_id': meta['observation_group_id'],
                'object_position_m': contract['object_position_m'],
                'object_orientation_xyzw': contract['object_orientation_xyzw'],
                'frame_files': files, 'scene_state_shapes': state_shapes,
                'existing_episode_count_for_same_source': len(entries),
                'representative_zero_noise_episode': str(episode),
                'collection_metadata_path': str(metadata_path), 'collection_metadata_sha256': sha(metadata_path),
                'resolved_case_path': str(episode / 'resolved_case.json'),
                'scene_mjb': {'path': str(episode / 'scene.mjb'), 'sha256': sha(episode / 'scene.mjb')},
                'stored_shadow_command': command,
                'binding_checks': {
                    'native_report_real': report['real_hug_inference_passed'] is True,
                    'native_report_hash_matches': native['recorded_proposal_hash_matches'],
                    'episode_native_hash_matches': episode_source_matches,
                    'episode_inference_report_hash_matches': inference_report_matches,
                    'contract_camera_hash_matches': contract['camera_sha256'].removeprefix('sha256:') == sha(p.parent / 'T_world_camera.npy'),
                    'initial_pose_matches_episode': contract['object_position_m'] == meta['object_position_m'] and contract['object_orientation_xyzw'] == meta['object_orientation_xyzw'],
                    'same_current_mapping_sources': not mismatches,
                },
                'current_mapping_source_mismatches': mismatches,
                'recorded_runtime_versions': meta['runtime_versions'],
                'standalone_translated_proposal_json_available': False,
                'duplicate_native_copies': hashes[native['sha256']],
            })
    standalone = WORK / 'runs/hug_grasp_eval_2026-10-13_fresh_cube/output/proposals/hug_cube/sample_0000/proposal.json'
    canonical = load(standalone)
    reference = next(c for c in cases if c['native']['sha256'] == '573cb003458d9d7cfe85b5b4a7877e414257b3e67d5abda04a09bd6d687fa0c2')
    cmd = reference['stored_shadow_command']
    with np.load(cmd['path'], allow_pickle=False) as actions:
        stored = dict(zip(actions['joint_names'].tolist(), actions['nominal_target_rad'][cmd['row_index']].tolist()))
    fresh = canonical['joint_targets_rad']
    if isinstance(fresh, list):
        raise ValueError('Inspect canonical joint schema before comparing')
    max_delta = max(abs(stored[name] - fresh[name]) for name in stored)
    checks = [check for c in cases for check in c['binding_checks'].values()]
    report = {
        'schema': 'existing_hug_static_validation_inventory_v1',
        'scope': 'HUG only; inventory of existing source/translation artifacts; no inference, optimization, FK, mj_forward, mj_step, rollout or eval generation',
        'reproducible_command': 'PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python /private/tmp/grasp-hug-existing-inventory-2026-10-14/inventory_existing_hug.py',
        'runtime': {'python': sys.version, 'numpy': np.__version__},
        'script_sha256': sha(Path(__file__)),
        'summary': {
            'primary_proposal_files': len(primary_entries),
            'primary_unique_native_pickle_hashes': len({d['sha256'] for d in primary_entries}),
            'primary_proposals_with_true_sibling_inference_report': sum(d.get('real_hug_inference_passed') is True for d in primary_entries),
            'worktree_native_files': len(work_entries), 'worktree_unique_native_hashes': len({d['sha256'] for d in work_entries}),
            'source_bound_modern_candidates': len(cases),
            'unique_modern_native_hashes': len({c['native']['sha256'] for c in cases}),
            'unique_modern_observation_groups': len({c['observation_group_id'] for c in cases}),
            'unique_modern_geometry_hashes': len({c['geometry_content_hash'] for c in cases}),
            'candidates_by_object': dict(Counter(c['object_id'] for c in cases)),
            'source_bound_saved_nominal_shadow_commands': len(cases),
            'all_candidate_binding_and_mapping_hash_checks_pass': all(checks),
            'mapping_hash_mismatched_candidates': sum(bool(c['current_mapping_source_mismatches']) for c in cases),
            'existing_episodes_in_collection_2080': sum(len(v) for v in episodes.values()),
            'each_source_existing_episodes': sorted({c['existing_episode_count_for_same_source'] for c in cases}),
        },
        'current_worktree_mapping_hashes': current_hashes,
        'modern_cases': cases,
        'all_primary_native_proposal_entries': primary_entries,
        'all_worktree_native_proposal_entries': work_entries,
        'duplicate_native_hash_groups': [{'sha256': h, 'paths': ps} for h, ps in hashes.items() if len(ps) > 1],
        'direct_current_cube_confirmation': {
            'standalone_translation_json': str(standalone), 'file_sha256': sha(standalone),
            'native_hash': reference['native']['sha256'],
            'stored_pre_force_close_vs_current_canonical_max_joint_delta_rad': max_delta,
            'joint_names_match': set(stored) == set(fresh),
        },
        'limitations': [
            '32 source poses form eight observed scenes from four geometries. Seeds from one observation are not independent object/scene holdouts.',
            '2,080 existing rollout episodes are 65 variants per native source, not 2,080 different native proposals.',
            '75 unique primary pickle byte hashes includes older single-observation pilots with weaker scene/provenance bindings; it is not 75 independent validation cases.',
            'Only one source has a current canonical proposal JSON; the 32 reusable targets are existing trajectory command records.',
            'scene_state.npz holds captured initial object state; scene.mjb is the stored compiled collision model. This inventory does not load either into a simulator.',
            'Exact source hash equality does not itself prove fidelity or contact preservation. No such claims are made here.',
        ],
    }
    destination = OUT / 'existing_hug_inventory.json'
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'path': str(destination), 'summary': report['summary'],
                      'direct_current_cube_confirmation': report['direct_current_cube_confirmation']}, indent=2))


if __name__ == '__main__':
    main()
