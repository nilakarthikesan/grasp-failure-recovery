"""Write-once bounded geometry-only articulated-skin engine bridge. Prepare reads only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib, importlib.util, json, os
from pathlib import Path
import signal, subprocess, sys, time, traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT/'research_artifacts/manosim_articulated_skin_2026-10-15_v1'
PRIOR_HASHES = ('166a1d040dcb41eaf8008f737f50a6c753ac9d64aa2bec6b432926df4c25d58f', '44453ca547658fff0e6d4478d9f497cb6c952478b87d93cd0b97d48025ec6cd5', '3bf9dbed86cfd4a7df8780509fe31d6f4ab36b7128cc472388b549c03fa56963')
SCHEMA = 'manosim_read_q_complete_skin_engine_bridge_v2'
FLAGS = dict(accepted_mapping=False, native_contact_qualified=False,
             new_collection_or_training_allowed=False, physical_outcome=None)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def git(*args): return subprocess.check_output(['git', '-C', str(ROOT), *args])


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def load_case():
    spec = importlib.util.spec_from_file_location('frozen_read_q_skin_engine_case_v1', HERE/'bridge_case.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected: raise ValueError('Frozen input/source changed: '+path)


def check_commit(p, freeze):
    if git('rev-parse', 'HEAD').decode().strip() != freeze: raise ValueError('HEAD changed from freeze')
    for relative, expected in p['tracked_sources'].items():
        if hashlib.sha256(git('show', freeze+':'+relative)).hexdigest() != expected:
            raise ValueError('Committed source differs: '+relative)
    path = Path(p['protocol_path'])
    if git('show', freeze+':'+str(path.relative_to(ROOT))) != path.read_bytes():
        raise ValueError('Protocol is not committed at freeze')


def prior_closure():
    paths = (PRIOR/'protocol.json', PRIOR/'run_v1/report.json', PRIOR/'run_v1/independent_readback.json')
    if tuple(sha(path) for path in paths) != PRIOR_HASHES: raise ValueError('Prior lineage changed')
    p, report, receipt = map(read, paths)
    if (receipt.get('all_bindings_verified_before') is not True
            or receipt.get('all_bindings_verified_after') is not True
            or receipt.get('expected_bindings') != 5235
            or receipt.get('status') != 'VERIFIED_COMPLETED_SAVED_SKIN_COMPARISONS_TANGENT_ARITHMETIC_AND_VIRTUAL_WORK'
            or report.get('status') != 'COMPLETED'
            or receipt.get('active_protocol_sha256') != PRIOR_HASHES[0]
            or report.get('protocol_sha256') != PRIOR_HASHES[0]
            or receipt.get('freeze_commit') != report.get('freeze_commit')
            or report.get('freeze_commit') != 'eac84d987201739944b615ee14a7cb5779c7d876'):
        raise ValueError('Prior completed articulated-skin evidence incomplete')
    bindings = dict(p['bindings']); verify(bindings)
    for path, row in receipt['artifact_inventory'].items():
        if sha(path) != row['sha256'] or Path(path).stat().st_size != row['bytes']:
            raise ValueError('Creation-time artifact changed: '+path)
        bindings[path] = row['sha256']
    for path in paths: bindings[str(path.resolve())] = sha(path)
    return p, bindings


def failed_v1_closure():
    failure = ROOT/'research_artifacts/manosim_skin_engine_bridge_2026-10-15_v1'
    paths = (failure/'protocol.json',failure/'run_v1/report.json',failure/'run_v1/independent_readback.json')
    expected = ('123408f6bbc7cfc07c07fefdae01bdc0884b295dcae963b2001802d193755054', '9cccf32c6c0fc7f17e09a4d1d99fcf4d7d50efdabadd377002fbbfb218c21070', '6f49b4ecb0a9c143057228a19868b2524ee50bb6c4f538676644ab7748c2d3e2')
    if tuple(sha(path) for path in paths) != expected: raise ValueError('Immutable failedv1 evidence changed')
    p,report,receipt=map(read,paths)
    if (report.get('status')!='HARD_STOP' or report.get('protocol_sha256')!=expected[0]
        or report.get('freeze_commit')!='5410cf8380f944a07fa7e62669fc4b6f5578ed55'
        or receipt.get('status')!='VERIFIED_HARD_STOP_DEFAULT_GEOM_TEMPLATE_INVENTORY_BEFORE_ALL_ENGINE_OPERATIONS'
        or receipt.get('active_protocol_sha256')!=expected[0] or receipt.get('freeze_commit')!=report['freeze_commit']
        or receipt.get('expected_bindings')!=5262 or receipt.get('all_bindings_verified_before') is not True
        or receipt.get('all_bindings_verified_after') is not True
        or report.get('cases',[{},{}])[1]!={'index':28,'status':'NOT_RUN'}):
        raise ValueError('Failedv1 lineage must preserve original pre-engine refusal')
    bindings=dict(p['bindings']);verify(bindings)
    for path,row in receipt['artifact_inventory'].items():
        if sha(path)!=row['sha256'] or Path(path).stat().st_size!=row['bytes']:raise ValueError('Failedv1 artifact changed')
        bindings[path]=row['sha256']
    for path in paths:bindings[str(path.resolve())]=sha(path)
    return p,bindings

def prepare(protocol_path, output_root):
    prior, bindings = prior_closure(); failed, old_bindings=failed_v1_closure()
    for path,digest in old_bindings.items():
        if path in bindings and bindings[path]!=digest:raise ValueError('Prior closure disagreement')
        bindings[path]=digest
    declaration = load_case().declaration()
    if set(declaration['cases']) != {'20', '28'} or declaration['budget'].get('physics_steps') != 0:
        raise ValueError('Only two declared geometry-only articulated cases; zero physics integration')
    sources = sorted(HERE.glob('*.py'))+sorted(HERE.glob('README*.md'))
    sources += sorted(HERE.glob('controls*.json'))+[HERE/'manufactured_control_lineage.json']
    controls = read(HERE/'controls.json')
    if controls.get('exit_code') != 0 or controls.get('integration_calls') != 0:
        raise ValueError('Successful manufactured controls required before native declaration')
    for path in sources+[Path(x) for x in declaration['source_inputs']]:
        if not path.is_absolute(): path = ROOT/path
        bindings[str(path.resolve())] = sha(path)
    tracked = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    for name, digest in {**prior['tracked_sources'],**failed['tracked_sources']}.items():
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        if bindings[str(path)] != digest: raise ValueError('Prior selected source binding differs')
        tracked[str(path.relative_to(ROOT))] = digest
    output_root = Path(output_root).resolve(); raw = ROOT/'runs'/HERE.name/output_root.name/'raw'
    if output_root.exists() or raw.exists(): raise ValueError('Fresh output/raw paths required')
    p = dict(schema_version=SCHEMA, prepared_utc=now(),
        protocol_path=str(Path(protocol_path).resolve()), output_root=str(output_root), raw_root=str(raw),
        bindings=bindings, tracked_sources=tracked, prior_protocol_sha256=PRIOR_HASHES[0],
        prior_independent_sha256=PRIOR_HASHES[2],
        prior_failed_v1_protocol_sha256='123408f6bbc7cfc07c07fefdae01bdc0884b295dcae963b2001802d193755054',
        prior_failed_v1_independent_sha256='6f49b4ecb0a9c143057228a19868b2524ee50bb6c4f538676644ab7748c2d3e2', **declaration, **FLAGS,
        scientific_acceptance_changed=False)
    write(protocol_path, p)
    return dict(protocol_sha256=sha(protocol_path), bindings=len(bindings), budget=p['budget'])


def verify_protocol(path, freeze):
    p = read(path)
    if p.get('schema_version') != SCHEMA or Path(p['protocol_path']) != Path(path).resolve():
        raise ValueError('Unexpected protocol')
    check_commit(p, freeze); verify(p['bindings'])
    for key, value in FLAGS.items():
        if p.get(key) is not value: raise ValueError('Qualification cannot grant admission')
    return p


def validate_measured(value):
    if not isinstance(value, dict) or value.get('status') != 'COMPLETED':
        raise ValueError('Expected complete bounded case receipt')
    reserved = {'index', 'schema_version', 'protocol_sha256', 'freeze_commit', 'started_utc'}
    if reserved.intersection(value): raise ValueError('Case cannot replace transaction identity')
    for key, expected in FLAGS.items():
        if key in value and value[key] is not expected: raise ValueError('Case cannot grant admission')
    if not isinstance(value.get('forbidden_counts'), dict) or any(value['forbidden_counts'].values()):
        raise ValueError('Explicit zero forbidden operation counts required')
    return value


def verify_case_output(row, p, freeze):
    raw, tracked = Path(row['raw_result']), Path(row['result'])
    if sha(raw) != row['raw_result_sha256'] or sha(tracked) != row['result_sha256'] or raw.read_bytes() != tracked.read_bytes():
        raise ValueError('Raw/tracked case receipt changed')
    value = read(raw)
    expected = dict(index=row['index'], protocol_sha256=sha(p['protocol_path']),
        freeze_commit=freeze, schema_version=SCHEMA+'_case')
    if any(value.get(k) != v for k, v in expected.items()): raise ValueError('Child transaction identity changed')
    for key, flag in FLAGS.items():
        if value.get(key) is not flag: raise ValueError('Child issued admission/outcome')
    root = (Path(p['raw_root'])/str(row['index'])).resolve()
    seen = set()
    for artifact in value['partial_artifacts']:
        path = Path(artifact['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(root) or path.resolve() in seen:
            raise ValueError('Child artifact escaped or duplicated')
        if sha(path) != artifact['sha256'] or path.stat().st_size != artifact['bytes']:
            raise ValueError('Child artifact changed: '+str(path))
        seen.add(path.resolve())
    if {x.resolve() for x in root.rglob('*.npz')} != {x for x in seen if x.suffix == '.npz'}:
        raise ValueError('Unindexed numeric artifact')
    # Check write-time events as well as the final partial-file inventory.
    progress = root/'progress.jsonl'
    for line in progress.read_text().splitlines() if progress.exists() else ():
        event = json.loads(line)
        if event.get('event') == 'artifact':
            artifact = event['artifact']; path = Path(artifact['path']).resolve()
            if path not in seen or sha(path) != artifact['sha256'] or path.stat().st_size != artifact['bytes']:
                raise ValueError('Creation-time artifact changed')
    return value


def child(protocol_path, freeze, index):
    p = verify_protocol(protocol_path, freeze)
    output = Path(p['raw_root'])/str(index); output.mkdir(parents=True, exist_ok=False)
    def checkpoint(value):
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(utc=now(), **value), sort_keys=True, allow_nan=False)+'\n')
            stream.flush()
    result = dict(schema_version=SCHEMA+'_case', index=index, status='HARD_STOP',
        protocol_sha256=sha(protocol_path), freeze_commit=freeze, started_utc=now(), **FLAGS)
    attempts = {'physics_integration_attempts': 0,'forbidden_direct_query_or_pipeline_attempts':0}
    try:
        sys.path.insert(0, str(ROOT/'runs/manosim_mujoco_3_4_0_runtime_v1'))
        import mujoco
        if mujoco.__version__ != '3.4.0': raise ValueError('Isolated MuJoCo3.4.0 required')
        def blocked(*a, **kw):
            attempts['physics_integration_attempts'] += 1
            raise RuntimeError('Physics integration prohibited by skin protocol')
        for name in ('mj_step', 'mj_step1', 'mj_step2'): setattr(mujoco, name, blocked)
        allowed_calls = dict(model_compilations=0,data_constructions=0,mujoco_forward_calls=0,direct_mujoco_flex_calls=0)
        def allowed(name, function):
            def call(*args, **kwargs):
                if allowed_calls[name] >= p['budget'][name]: raise RuntimeError('Engine-call cap exhausted '+name)
                allowed_calls[name] += 1
                return function(*args, **kwargs)
            return call
        class CountedModel:
            from_xml_path = staticmethod(allowed('model_compilations', mujoco.MjModel.from_xml_path))
        mujoco.MjModel = CountedModel
        mujoco.MjData = allowed('data_constructions', mujoco.MjData)
        mujoco.mj_forward = allowed('mujoco_forward_calls', mujoco.mj_forward)
        mujoco.mj_flex = allowed('direct_mujoco_flex_calls', mujoco.mj_flex)
        def blocked_query(*a, **kw):
            attempts['forbidden_direct_query_or_pipeline_attempts'] += 1
            raise RuntimeError('Direct query or undeclared pipeline prohibited by geometry bridge')
        for name in ('mj_forwardSkip','mj_kinematics','mj_fwdPosition','mj_fwdVelocity','mj_fwdAcceleration','mj_collision','mj_geomDistance','mj_integratePos'):
            if hasattr(mujoco,name):setattr(mujoco,name,blocked_query)
        def timeout(*a): raise TimeoutError('Declared skin-case wall budget exceeded')
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, p['budget']['maximum_seconds_per_case'])
        measured = load_case().run_case(p['cases'][str(index)], output, p, checkpoint)
        result.update(validate_measured(measured))
        if measured.get('counts') != {k:v for k,v in p['budget'].items() if k != 'maximum_seconds_per_case'}:
            raise ValueError('Exact complete per-case operation inventory required')
        if allowed_calls != {k:p['budget'][k] for k in allowed_calls}:
            raise ValueError('Independent engine-call counter differs')
        if any(result.get('forbidden_counts', {}).values()) or any(attempts.values()):
            raise ValueError('Forbidden operation attempted')
        for key, value in FLAGS.items():
            if result.get(key) is not value: raise ValueError('Case cannot grant admission')
        verify(p['bindings']); check_commit(p, freeze); result['failure'] = None
    except BaseException as error:
        result.update(status='HARD_STOP', failure=dict(type=type(error).__name__, message=str(error),
                                                    traceback=traceback.format_exc()))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result.update(finished_utc=now(), integration_guards=attempts, allowed_engine_calls=locals().get('allowed_calls',{}))
    result['partial_artifacts'] = [dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        for path in sorted(output.rglob('*')) if path.is_file()]
    write(output/'result.json', result)
    return 1 if result['status'] == 'HARD_STOP' else 0


def run(protocol_path, freeze):
    p = verify_protocol(protocol_path, freeze)
    out = Path(p['output_root']); out.mkdir(parents=True, exist_ok=False)
    write(out/'STARTED.json', dict(protocol_sha256=sha(protocol_path), freeze_commit=freeze, started_utc=now()))
    rows, stopped = [], False
    for index in (20, 28):
        row = dict(index=index, status='NOT_RUN')
        if not stopped:
            started = time.monotonic()
            with (out/f'case_{index}.log').open('x') as log:
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_case',
                    '--protocol', str(Path(protocol_path).resolve()), '--freeze-commit', freeze,
                    '--index', str(index)], stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
                try: code = proc.wait(timeout=p['budget']['maximum_seconds_per_case']+30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait(); code = -signal.SIGKILL
            row.update(returncode=code, parent_wall_seconds=time.monotonic()-started)
            raw = Path(p['raw_root'])/str(index)/'result.json'
            if raw.exists():
                value = read(raw); write(out/str(index)/'result.json', value)
                row.update(status=value['status'], raw_result=str(raw), raw_result_sha256=sha(raw),
                           result=str(out/str(index)/'result.json'), result_sha256=sha(out/str(index)/'result.json'))
            else: row['status'] = 'HARD_STOP_NO_RESULT'
            stopped = code != 0 or row['status'].startswith('HARD_STOP')
        rows.append(row)
    report = dict(schema_version=SCHEMA+'_report', status='HARD_STOP' if stopped else 'COMPLETED',
        protocol_sha256=sha(protocol_path), freeze_commit=freeze, cases=rows, finished_utc=now(), **FLAGS)
    try:
        verify(p['bindings']); check_commit(p, freeze)
        for row in rows:
            if 'raw_result' in row: verify_case_output(row, p, freeze)
    except BaseException as error: report.update(status='HARD_STOP', final_binding_failure=str(error))
    write(out/'report.json', report)
    return dict(status=report['status'], cases=rows, report_sha256=sha(out/'report.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run', '_case'))
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--output', type=Path, default=HERE/'run_v1')
    parser.add_argument('--freeze-commit')
    parser.add_argument('--index', type=int, choices=(20,28))
    args = parser.parse_args()
    if args.mode == 'prepare': print(json.dumps(prepare(args.protocol, args.output)))
    elif args.mode == 'run': print(json.dumps(run(args.protocol, args.freeze_commit)))
    else: raise SystemExit(child(args.protocol, args.freeze_commit, args.index))


if __name__ == '__main__': main()
