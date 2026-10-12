"""Capture declared scenes and generate fresh HUG proposals before collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from grasp_failure_prediction.evaluation.case_runner import load_case


def prepare_scenes(specs_path, output, *, project_root, hug_root, checkpoint, proposal_seeds,
                   renderer_python=None):
    project, output = Path(project_root).resolve(), Path(output).resolve()
    output.relative_to(project)
    if output.exists():
        raise FileExistsError('preparation output must be a new directory')
    specs = json.loads(Path(specs_path).read_text())
    if not isinstance(specs, list) or not specs:
        raise ValueError('scene specs must be a nonempty list')
    if len(set(proposal_seeds)) != len(proposal_seeds) or not proposal_seeds:
        raise ValueError('proposal seeds must be nonempty and unique')
    parsed = []
    for spec in specs:
        relative = Path(spec['case_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('scene case paths must be project-relative')
        parsed.append((load_case(project / relative), project / relative,
                       spec.get('camera_position_m', [.25,-.35,.35])))
    output.mkdir(parents=True)
    cases_dir = output / 'cases'
    cases_dir.mkdir()
    manifest = []
    source_root = Path(__file__).resolve().parents[1]
    renderer_python = renderer_python or sys.executable
    for index, (case, case_path, camera) in enumerate(parsed):
        observation = output / 'observations' / f'scene_{index:03d}'
        print(f'Capturing scene {index}: {case.object.id}', flush=True)
        subprocess.run([renderer_python, '-m', 'grasp_failure_prediction.integrations.sim_observations',
                        str(case_path), '--project-root', str(project), '--output', str(observation),
                        '--camera-position', *map(str,camera)], check=True, cwd=source_root)
        contract = json.loads((observation / 'scene_contract.json').read_text())
        for seed in proposal_seeds:
            proposal_dir = output / 'proposals' / f'scene_{index:03d}_seed_{seed}'
            shutil.copytree(observation, proposal_dir)
            print(f'Generating fresh HUG proposal for scene {index}, seed {seed}', flush=True)
            subprocess.run([sys.executable, str(source_root / 'scripts/infer_sim_observation.py'),
                            '--hug-root', str(hug_root), '--checkpoint', str(checkpoint),
                            '--observation', str(proposal_dir), '--seed', str(seed), '--steps', '50'],
                           check=True, cwd=source_root)
            payload = case.model_dump(mode='json')
            payload['case_id'] = f'scene{index:03d}_hug_seed{seed}'
            payload['object']['mass_kg'] = contract['mass_kg']
            payload['initial_condition'] = {
                'id': f'captured_scene{index:03d}',
                'object_position_m': contract['object_position_m'],
                'object_orientation_xyzw': contract['object_orientation_xyzw'],
            }
            payload['grasp'].update(id=f'hug_scene{index:03d}_seed{seed}', inference_seed=seed,
                                    observation_path=str(observation.relative_to(project)),
                                    prediction_path=str((proposal_dir/'proposal.pkl').relative_to(project)))
            prepared_case = cases_dir / f'scene_{index:03d}_seed_{seed}.yaml'
            prepared_case.write_text(yaml.safe_dump(payload, sort_keys=False))
            # Parse the final authored case instead of trusting model_copy updates.
            load_case(prepared_case)
            manifest.append(str(prepared_case.relative_to(project)))
            (output/'cases.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'scenes':len(parsed), 'proposals':len(manifest),
                      'manifest':str(output/'cases.json')}, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene_specs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--hug-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--proposal-seeds', type=int, nargs='+', default=[0,1])
    parser.add_argument('--renderer-python', help='optional renderer interpreter, such as mjpython')
    args = parser.parse_args()
    prepare_scenes(args.scene_specs, args.output, project_root=args.project_root,
                   hug_root=args.hug_root.resolve(), checkpoint=args.checkpoint.resolve(),
                   proposal_seeds=args.proposal_seeds, renderer_python=args.renderer_python)
