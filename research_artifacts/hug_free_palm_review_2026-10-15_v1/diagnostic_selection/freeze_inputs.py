"""Freeze immutable selection inputs, without ranking or geometric measurement."""
from pathlib import Path
import datetime, gzip, hashlib, json

OUT = Path(__file__).parent
ROOT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
policy = read(OUT / 'policy.json')
assert sha(OUT / 'policy.json') == '103bf94793a6b41b6095ef3247098e2bb3b65749535e3ebacc6bfac9dc11c1c3'
assert sha(OUT / 'select_and_verify.py') == policy['script_sha256']
assert sha(OUT.parent / 'verify_outputs.py') == policy['independent_verifier_script_sha256']
assert sha(OUT / 'synthetic_selection_checks.json') == policy['synthetic_policy_controls_sha256']
assert not (OUT / 'execution_started.json').exists()
assert not (OUT / 'protocol.json').exists()
assert not (OUT / 'candidate_targets.json').exists()
inputs = {}
methods = []

def bind(path, expected=None):
    path = str(path)
    actual = sha(path)
    if expected is not None:
        assert actual == expected, path
    assert path not in inputs or inputs[path] == actual, path
    inputs[path] = actual

for order, directory in enumerate(policy['method_directories']):
    directory = Path(directory)
    protocol = read(directory / 'protocol.json')
    compact = read(directory / 'candidate_targets.json')
    summary = read(directory / 'fit_summary.json')
    assert compact['protocol_sha256'] == sha(directory / 'protocol.json')
    assert not compact['new_eval_data'] and compact['physics_steps'] == 0
    for table in ('source_sha256', 'implementation_sha256', 'query_cache_sha256', 'full_cache_sha256'):
        for path, expected in protocol[table].items():
            bind(path, expected)
    for name in ('protocol.json', 'candidate_targets.json', 'fit_summary.json', 'exploration_summary.json'):
        bind(directory / name)
    indices = []
    for provenance in compact['cohort_provenance']:
        path = Path(provenance['gzip_path'])
        bind(path, provenance['gzip_sha256'])
        raw = gzip.decompress(path.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == provenance['sha256']
        cohort = json.loads(raw)
        for case in cohort['cases']:
            indices.append(case['source_inventory_index'])
            for callback in case['keyed_direction_evidence']['callbacks']:
                bind(callback['trace_path'], callback['trace_compressed_sha256'])
    assert sorted(indices) == policy['case_indices']
    methods.append({'name': policy['method_order'][order], 'protocol_path': str(directory / 'protocol.json'), 'candidate_path': str(directory / 'candidate_targets.json')})

historical = ROOT / 'research_artifacts/hug_canonical_mcp_calibration_2026-10-14_v1/runtime_v3/candidate_targets.json'
inventory = ROOT / 'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json'
urdf = ROOT / 'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf'
bind(historical, '84496804c00267024af56f3364457943696db8481b9c1853bca2c93c6c824b9d')
for path in (inventory, urdf, OUT/'policy.json', OUT/'select_and_verify.py', OUT/'test_selection_policy.py', OUT/'synthetic_selection_checks.json', OUT.parent/'verify_outputs.py', Path(__file__)):
    bind(path)
protocol = {
    'schema': 'hug_free_palm_diagnostic_selection_inputs_v1',
    'freeze_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'policy_path': str(OUT/'policy.json'),
    'policy_sha256': sha(OUT/'policy.json'),
    'script_sha256': policy['script_sha256'],
    'input_sha256': inputs,
    'methods': methods,
    'case_indices': policy['case_indices'],
    'inventory_path': str(inventory),
    'historical_candidate_path': str(historical),
    'source_urdf': str(urdf),
    'input_closure_verified_before_first_ranking': True,
    'policy_frozen_before_final_fine_results_and_before_ranking': True,
    'scientific_acceptance_rule_sha256': policy['scientific_acceptance_rule_sha256'],
    'scientific_mapping_acceptance': False,
    'new_eval_data': False,
    'physics_steps': 0,
    'new_inference': False,
    'production_changed': False,
    'note': 'Hash/source inventory assembly only; no scores compared, no selection/ranking and no candidate FK before this freeze.'
}
with (OUT/'protocol.json').open('x') as stream:
    json.dump(protocol, stream, indent=2)
    stream.write('\n')
print(json.dumps({'protocol_sha256': sha(OUT/'protocol.json'), 'input_count': len(inputs), 'methods': [m['name'] for m in methods]}))
