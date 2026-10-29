"""Run the approved three-proposal Phase 1 engineering demonstration.

Every stage consumes the original scene/proposal. A native mapping rejection
blocks the corresponding physical trial; invalid runs never become failures.
This demonstration neither trains a predictor nor estimates general performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_scene(scene):
    report = read_json(scene / 'validation_report.json')
    if report.get('scene_preparation_checks_passed') is not True:
        raise ValueError('scene preparation has not passed')
    for name, digest in report['artifact_sha256'].items():
        if Path(name).name != name or file_hash(scene / name) != digest:
            raise ValueError('scene input changed: ' + name)
    return report


def _stage(command, logfile, environment):
    started = time.perf_counter()
    print('Running: ' + ' '.join(map(str, command)), flush=True)
    with logfile.open('x') as stream:
        result = subprocess.run(list(map(str, command)), cwd=PROJECT, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    print(f'Stage exit={result.returncode}; log={logfile}', flush=True)
    return {'returncode': result.returncode,
            'elapsed_s': time.perf_counter() - started, 'log': str(logfile)}


def run_demo(scene, output, *, python, hug_root, checkpoint, wuji_assets,
             dependency_dirs=(), native_assets=None):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    verify_scene(scene)
    if output.exists():
        raise FileExistsError('demo output must be a fresh directory')
    output.mkdir(parents=True)
    logs = output / 'logs'
    logs.mkdir()
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(PROJECT / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'egl')
    # Load base numerical libraries before appending the existing optional
    # dependency depot, which also contains a different NumPy installation.
    bootstrap = ('import numpy,mujoco,sys,runpy; sys.path.extend(' +
                 repr([str(Path(p).resolve()) for p in dependency_dirs]) +
                 '); module=sys.argv[1]; sys.argv=sys.argv[1:]; '
                 'runpy.run_module(module,run_name="__main__")')
    python_module = [python, '-c', bootstrap]
    protocol = {
        'schema_version': 'phase1_demo_protocol_v1',
        'scene': str(scene), 'scene_report_sha256': file_hash(scene / 'validation_report.json'),
        'seeds': [0, 1, 2], 'sampling_steps': 50, 'point_cloud_seed': 42,
        'hand': 'official_wuji_hand_2_1_right',
        'hand_revision': '5c4dfa16917355d073e2e1d7e3d03897b6cf9508',
        'success_definition': {'minimum_lift_m': 0.05, 'minimum_hold_s': 1.0},
        'wrist_abstraction': 'idealized moving positioning fixture; no robot arm',
        'finger_actuation': 'official force-limited position actuators',
        'invalid_mapping_policy': 'null task outcome; no physical trial',
        'selection': 'all three fixed seeds retained, no selection by task outcome',
        'scope': 'single synthetic scene engineering demo; no training or generalization claim',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
    }
    write_json(output / 'protocol.json', protocol)
    hug_command = [*python_module, 'grasp_failure_prediction.integrations.phase1_hug',
                   '--scene', scene, '--output', output / 'hug', '--hug-root', hug_root,
                   '--checkpoint', checkpoint, '--seeds', 0, 1, 2, '--steps', 50]
    for directory in dependency_dirs:
        hug_command += ['--dependency-dir', directory]
    generation = _stage(hug_command, logs / 'hug.log', environment)
    rows = []
    for seed in protocol['seeds']:
        name = f'seed_{seed:03d}'
        proposal = output / 'hug' / name
        row = {'seed': seed, 'proposal_directory': str(proposal), 'success': None}
        if not (proposal / 'proposal.npz').is_file():
            row.update(status='generation_failed', reason='fresh proposal not produced')
            rows.append(row)
            continue
        native_command = [*python_module, 'grasp_failure_prediction.integrations.phase1_manosim',
                          '--scene', scene, '--proposal', proposal, '--output', output / 'native' / name]
        if native_assets is not None:
            native_command += ['--assets-root', native_assets]
        row['native_stage'] = _stage(native_command, logs / f'{name}_native.log', environment)
        native_report = output / 'native' / name / 'report.json'
        if row['native_stage']['returncode'] != 0 or not native_report.is_file() or read_json(native_report).get('mapping_checks_passed') is not True:
            row.update(status='native_mapping_rejected', reason='no passed native mapping report')
            rows.append(row)
            continue
        trial_command = [*python_module, 'grasp_failure_prediction.evaluation.phase1_wuji',
                         '--scene', scene, '--proposal', proposal, '--assets', wuji_assets,
                         '--output', output / 'wuji' / name]
        row['trial_stage'] = _stage(trial_command, logs / f'{name}_wuji.log', environment)
        result_path = output / 'wuji' / name / 'result.json'
        if not result_path.is_file():
            row.update(status='runtime_failed', reason='no trial result')
        else:
            result = read_json(result_path)
            row.update(status=result.get('status', 'runtime_failed'), result_path=str(result_path),
                       failure_type=result.get('failure_type'), metrics=result.get('metrics', {}))
            if row['status'] == 'completed' and type(result.get('success')) is bool:
                row['success'] = result['success']
        rows.append(row)
        write_json(output / 'progress.json', rows)
    completed = [row for row in rows if row['status'] == 'completed' and type(row['success']) is bool]
    summary = {
        'schema_version': 'phase1_demo_summary_v1', 'protocol': protocol, 'generation': generation,
        'attempts': rows, 'proposals_requested': 3, 'completed_trials': len(completed),
        'successes': sum(row['success'] is True for row in completed),
        'failures': sum(row['success'] is False for row in completed),
        'invalid_or_unexecuted': len(rows) - len(completed),
        'pipeline_execution_finished': True,
        'all_required_stages_completed': len(completed) == 3,
        'training_executed': False,
    }
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'output', 'hug-root', 'checkpoint', 'wuji-assets'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--dependency-dir', type=Path, action='append', default=[])
    parser.add_argument('--native-assets-root', type=Path)
    args = parser.parse_args(argv)
    summary = run_demo(args.scene, args.output, python=args.python, hug_root=args.hug_root,
                       checkpoint=args.checkpoint, wuji_assets=args.wuji_assets,
                       dependency_dirs=args.dependency_dir, native_assets=args.native_assets_root)
    return 0 if summary['all_required_stages_completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
