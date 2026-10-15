#!/usr/bin/env python3
"""Frozen, bounded static DGA development experiment; no provider activation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path('/private/tmp/dga_component_constraints_20261015_v1')


def sha(path):
    return 'sha256:' + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write(path, value):
    def default(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False, default=default)
        f.write('\n')


def main():
    frozen = json.loads((HERE/'band_method_frozen_v1.json').read_text())
    for path, digest in frozen['dependency_sha256'].items():
        if sha(path) != digest: raise ValueError('Frozen dependency changed: '+path)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        if os.environ.get(name) != '1': raise ValueError('Single CPU thread required')
    p = json.loads((OLD/'native_mapping_implementation_frozen_v1.json').read_text())
    report = json.loads((OLD/'native_mapping_results_v1/native_component_fit_report.json').read_text())
    ledger = json.loads((OLD/'native_validation/results_v1/native_validation_report.json').read_text())
    g = frozen['development_global_index']
    row = ledger['records'][g]
    previous = next(r for r in report['records'] if r['global_sample_index'] == g)
    mods = {k: module('band_'+k, p['code_paths'][k]) for k in
            ['core','context','legacy','audit','geometry','mesh_helper','profile']}
    proposer = module('band_proposer', HERE/'band_corrector.py')
    profile, core = mods['profile'], mods['core']
    base = profile.original
    manifest, raws, _ = profile.load_multidex_raw_ledger(row['native_manifest']['path'],
                    expected_object_id='object01', expected_geometry_hash=p['geometry_content_hash'])
    raw = raws[row['within_batch_sample_index']].copy()
    if 'sha256:'+hashlib.sha256(raw.tobytes()).hexdigest() != row['raw33_row_sha256']:
        raise ValueError('Native row identity changed')
    reference = json.loads(Path(row['mapping_reference']['path']).read_text())
    refnpz = Path(row['mapping_reference']['path']).parent/reference['arrays']['path']
    if sha(refnpz) != reference['arrays']['sha256']: raise ValueError('Reference changed')
    with np.load(p['shared_fitting_queries']['path'], allow_pickle=False) as z:
        queries = z['queries_object_m']
    ctx = mods['context'].MappingContext(raw33=raw, object_from_input=manifest.object_from_input,
        source_urdf=p['source_hand_urdf'], target_urdf=p['target_hand_urdf'],
        target_assets_root=p['target_assets_root'], source_runtime_npz=refnpz,
        source_sample_index=row['within_batch_sample_index'], queries_object_m=queries,
        base=base, audit=mods['audit'], geometry=mods['geometry'], legacy=mods['legacy'],
        mesh_helper=mods['mesh_helper'])
    state = dict(previous['selected_state'])
    for key in ['variables','palm','root','forearm','delta']:
        state[key] = np.asarray(state[key], dtype=np.float64)
    state = ctx.certify_payload(state)
    source = ctx.initialize_source_fields()
    state['fields'] = ctx.evaluate_fields(state)
    state['component_metrics'] = core.component_metrics(source, state['fields'])
    if state['component_metrics'] != previous['component_metrics']:
        raise ValueError('Previous field metrics failed exact replay')
    caps = previous['landmark_limits']
    out = HERE.parent/'band_results_v1'
    out.mkdir(exist_ok=False)
    before = ctx.source_identity()
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        chosen, trace = proposer.propose(ctx, core, state, source, caps,
                                        deadline=start+600, max_states=2400)
    ctx.certify_payload(chosen)
    ctx.verify_immutable_source()
    if ctx.source_identity() != before: raise ValueError('Native source changed')
    if not core.accepted_feasible(chosen, caps, state['component_metrics']):
        raise ValueError('True final guard refusal')
    payload = ctx.export_pose_payload(chosen)
    np.savez(out/'candidate_pose.npz', global_sample_indices=np.array([g]),
             **{k:np.asarray(v)[None] for k,v in payload.items()})
    np.savez_compressed(out/'fields.npz', queries_object_m=queries,
                        source=source, initial=state['fields'], selected=chosen['fields'])
    write(out/'trial_trace.json', trace)
    summary = {k:v for k,v in trace.items() if k != 'trials'}
    summary.update(schema_version='dga_static_band_development_report_v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        development_global_index=g, elapsed_seconds=time.monotonic()-start,
        method_sha256=sha(HERE/'band_method_frozen_v1.json'),
        initial_state=core.state_summary(state), selected_state=core.state_summary(chosen),
        landmark_caps=caps, warnings=[{'category':w.category.__name__,'message':str(w.message),
                                      'filename':w.filename,'lineno':w.lineno} for w in observed],
        zero_execution={'physics_steps':0,'inference':0,'evaluation_episodes':0,'training':0},
        outputs={f.name:sha(f) for f in out.iterdir() if f.is_file()})
    write(out/'report.json', summary)
    if {'mujoco','torch','isaacgym','pybullet'} & set(sys.modules):
        raise ValueError('Forbidden simulation/inference imports')
    for path,digest in frozen['dependency_sha256'].items():
        if sha(path)!=digest: raise ValueError('Frozen dependency changed during run: '+path)
    print(json.dumps({k:summary[k] for k in ['development_global_index','trial_count',
        'eligible_improved_count','adopted_changed_state','baseline_proxy','chosen_proxy','elapsed_seconds']}))


if __name__ == '__main__':
    main()
