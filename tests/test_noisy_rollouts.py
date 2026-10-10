from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import mujoco
import numpy as np
import pinocchio
import pytest
import yaml

from grasp_failure_prediction.evaluation import noisy_rollouts
from grasp_failure_prediction.evaluation.noisy_rollouts import (
    NOISY_STATES, NoisyShadowRunner, collect_noisy_rollouts,
)
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter, default_dex_urdf_root
from grasp_failure_prediction.evaluation.runner import ActuatedShadowRunner, ExecutionState, ExecutionTrace
from test_case_runner import write_test_case
from test_evaluation_schema import valid_case
from test_retargeting import synthetic_hug_prediction


TRUSTED_SOURCE = (Path(os.environ['GRASP_PILOT_TRUSTED_SOURCE'])
                  if os.environ.get('GRASP_PILOT_TRUSTED_SOURCE') else None)


@pytest.fixture(scope='module')
def pose():
    return ShadowHandRetargeter().retarget(synthetic_hug_prediction()[0])


def make_runner(amplitude=0.0, seed=17):
    protocol = load_protocol_registry().resolve('fixed_grasp_lift_v2')
    return NoisyShadowRunner(protocol, default_dex_urdf_root(),
                             noise_amplitude_rad=amplitude, noise_seed=seed)


def execute(runner, pose):
    runner.reset(seed=9000, object_mass_kg=0.18,
                 object_position_m=np.array([0.0, 0.0, 0.03]),
                 object_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))
    return runner.execute(pose, grasp_palm_position_m=np.array([0.0, 0.0, 0.09]),
                          grasp_palm_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))


def test_zero_noise_is_identical_to_original_fixed_controller(pose):
    protocol = load_protocol_registry().resolve('fixed_grasp_lift_v2')
    baseline = ActuatedShadowRunner(protocol, default_dex_urdf_root())
    noisy = make_runner()
    first, second = execute(baseline, pose), execute(noisy, pose)
    assert first.states == second.states
    for old, new in zip(first.steps, second.steps, strict=True):
        assert old.time_s == new.time_s
        np.testing.assert_array_equal(old.hand_qpos, new.hand_qpos)
        np.testing.assert_array_equal(old.object_position_m, new.object_position_m)
        assert old.hand_object_contact == new.hand_object_contact
    np.testing.assert_array_equal(baseline.data.qpos, noisy.data.qpos)
    np.testing.assert_array_equal(baseline.data.qvel, noisy.data.qvel)


def test_rng_is_local_reproducible_and_paired_across_magnitudes(pose):
    first, second = make_runner(.15, 72), make_runner(.15, 72)
    np.random.seed(3)
    execute(first, pose)
    np.random.seed(840)
    execute(second, pose)
    np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
    np.testing.assert_array_equal(first.episode_bias_rad, second.episode_bias_rad)
    different = make_runner(.15, 73)
    assert not np.array_equal(first.episode_bias_rad, different.episode_bias_rad)
    smaller = make_runner(.05, 72)
    np.testing.assert_allclose(first.episode_bias_rad, 3 * smaller.episode_bias_rad)


def test_noise_bounds_phases_clipping_and_no_input_mutation():
    runner = make_runner(.15, 19)
    limits = runner.model.jnt_range[runner._joint_ids]
    nominal = limits[:, 1].copy() - 1e-6
    original = nominal.copy()
    wrist = np.array([name.startswith('WRJ') for name in runner._hand_joint_names])
    for state in ExecutionState:
        command, requested = runner._noisy_command(state, nominal)
        np.testing.assert_array_equal(nominal, original)
        np.testing.assert_array_equal(command[wrist], nominal[wrist])
        np.testing.assert_array_equal(requested[wrist], 0.0)
        assert np.max(np.abs(requested)) <= .15
        if state in NOISY_STATES:
            np.testing.assert_allclose(command, np.clip(nominal + requested, limits[:, 0], limits[:, 1]))
            assert np.all(command >= limits[:, 0]) and np.all(command <= limits[:, 1])
            assert np.any(np.abs(command - nominal - requested) > 1e-8)
        else:
            np.testing.assert_array_equal(command, nominal)
            np.testing.assert_array_equal(requested, 0.0)


@pytest.mark.parametrize(('amplitude', 'seed'), [(-.1, 1), (float('nan'), 1),
                                                 (float('inf'), 1), (.1, -1), (.1, 2**32)])
def test_invalid_noise_is_rejected_before_model_creation(amplitude, seed):
    with pytest.raises(ValueError):
        make_runner(amplitude, seed)


def test_action_artifacts_align_commands_states_and_exact_replay(tmp_path, pose):
    runner = make_runner(.15)
    trace = execute(runner, pose)
    runner.write_action_artifacts(tmp_path)
    with np.load(tmp_path / 'actions.npz', allow_pickle=False) as actions:
        np.testing.assert_array_equal(actions['time_s'], [step.time_s for step in trace.steps])
        np.testing.assert_allclose(actions['time_s'] - actions['action_interval_start_s'], .04)
        np.testing.assert_allclose(actions['time_s'] - actions['diagnostic_time_s'], .002)
        np.testing.assert_array_equal(actions['previous_command_rad'][1:], actions['commanded_target_rad'][:-1])
        np.testing.assert_allclose(actions['nominal_target_rad'] + actions['applied_noise_rad'],
                                   actions['commanded_target_rad'])
        assert actions['full_qpos'].shape == (len(trace.steps), runner.model.nq)
        assert actions['full_qvel'].shape == (len(trace.steps), runner.model.nv)
        assert len(actions['joint_names']) == 24
        model = mujoco.MjModel.from_binary_path(str(tmp_path / 'scene.mjb'))
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, actions['integration_state'][-1],
                          mujoco.mjtState.mjSTATE_INTEGRATION)
        np.testing.assert_array_equal(data.qpos, runner.data.qpos)
        np.testing.assert_array_equal(data.qvel, runner.data.qvel)
        np.testing.assert_array_equal(data.ctrl, runner.data.ctrl)
        assert data.time == runner.data.time
    assert (tmp_path / 'scene.xml').is_file()


def test_saved_proposal_missing_camera_is_an_unlabeled_validation_error(tmp_path):
    write_test_case(tmp_path)
    manifest = tmp_path / 'cases.json'
    manifest.write_text(json.dumps(['case.yaml']))
    output = tmp_path / 'output'
    summary = collect_noisy_rollouts(manifest, output, project_root=tmp_path,
                                     noise_amplitudes_rad=[0], repetitions=1)
    assert summary['errors'] == 1 and summary['completed'] == 0
    row = json.loads((output / 'index.jsonl').read_text())
    assert row['status'] == 'validation_failed' and row['label'] is None
    assert 'world camera transform' in row['error']
    assert not (output / 'episode_000000/result.json').exists()


def test_alignment_gate_is_enforced_for_saved_proposal(tmp_path):
    write_test_case(tmp_path)
    np.save(tmp_path / 'grasps/object01/T_world_camera.npy', np.eye(4))
    manifest = tmp_path / 'cases.json'
    manifest.write_text(json.dumps(['case.yaml']))
    output = tmp_path / 'output'
    summary = collect_noisy_rollouts(manifest, output, project_root=tmp_path,
                                     noise_amplitudes_rad=[0], repetitions=1)
    assert summary['errors'] == 1
    row = json.loads((output / 'index.jsonl').read_text())
    assert row['label'] is None and 'alignment' in row['error']


@pytest.mark.parametrize('invalid_part', ['scale', 'shear', 'reflection', 'homogeneous_row'])
def test_nonrigid_camera_is_an_unlabeled_preflight_error(tmp_path, invalid_part):
    write_test_case(tmp_path)
    camera = np.eye(4)
    if invalid_part == 'scale':
        camera[0, 0] = 2
    elif invalid_part == 'shear':
        camera[0, 1] = .1
    elif invalid_part == 'reflection':
        camera[0, 0] = -1
    else:
        camera[3, 0] = .1
    np.save(tmp_path / 'grasps/object01/T_world_camera.npy', camera)
    manifest = tmp_path / 'cases.json'
    manifest.write_text(json.dumps(['case.yaml']))
    output = tmp_path / 'output'
    summary = collect_noisy_rollouts(manifest, output, project_root=tmp_path,
                                     noise_amplitudes_rad=[0], repetitions=1)
    row = json.loads((output / 'index.jsonl').read_text())
    assert summary['errors'] == 1 and summary['failures'] == 0
    assert row['label'] is None and 'rigid' in row['error']
    assert not (output / 'episode_000000/result.json').exists()


def test_runtime_failure_removes_completed_result_and_cannot_be_label_zero(tmp_path, monkeypatch):
    manifest = tmp_path / 'cases.json'
    manifest.write_text(json.dumps(['case.yaml']))
    monkeypatch.setattr(noisy_rollouts, '_prepare_case',
                        lambda *args: SimpleNamespace(source={'group_id': 'fixed_group',
                                                             'observation_group_id': 'scene_group',
                                                             'object_id': 'object01'}))

    def fail(prepared, output, entry):
        output.mkdir()
        (output / 'result.json').write_text('{"status":"completed"}')
        raise RuntimeError('artifact writing failed')

    monkeypatch.setattr(noisy_rollouts, '_run_episode', fail)
    output = tmp_path / 'output'
    summary = collect_noisy_rollouts(manifest, output, project_root=tmp_path,
                                     noise_amplitudes_rad=[0], repetitions=1)
    assert summary['failures'] == 0 and summary['errors'] == 1
    row = json.loads((output / 'index.jsonl').read_text())
    assert row['label'] is None and row['status'] == 'runtime_failed'
    assert not (output / 'episode_000000/result.json').exists()


def test_invalid_manifest_and_nonempty_output_are_rejected(tmp_path):
    manifest = tmp_path / 'cases.json'
    manifest.write_text(json.dumps(['../case.yaml']))
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='project-relative'):
        collect_noisy_rollouts(manifest, output, project_root=tmp_path)
    assert not output.exists()
    manifest.write_text(json.dumps(['case.yaml']))
    output.mkdir()
    (output / 'keep.txt').write_text('preserve')
    with pytest.raises(FileExistsError):
        collect_noisy_rollouts(manifest, output, project_root=tmp_path)
    assert (output / 'keep.txt').read_text() == 'preserve'


@pytest.mark.skipif(TRUSTED_SOURCE is None or not (TRUSTED_SOURCE / 'proposal.pkl').is_file(),
                    reason='local trusted HUG proposal is unavailable')
def test_trusted_case_collects_labeled_bundles_and_location_independent_groups(tmp_path):
    # Copy an existing locally generated HUG proposal; never deserialize a new external pickle.
    source = tmp_path / 'source'
    source.mkdir()
    shutil.copyfile(TRUSTED_SOURCE / 'proposal.pkl', source / 'proposal.pkl')
    shutil.copyfile(TRUSTED_SOURCE / 'T_world_camera.npy', source / 'T_world_camera.npy')
    observation = tmp_path / 'observation'
    observation.mkdir()
    np.save(observation / 'depth.npy', np.ones((2, 2)))
    payload = valid_case()
    payload['grasp']['prediction_path'] = 'source/proposal.pkl'
    payload['grasp']['observation_path'] = 'observation'
    case_path = tmp_path / 'case.yaml'
    case_path.write_text(yaml.safe_dump(payload))
    first = noisy_rollouts._prepare_case(case_path, tmp_path)
    shutil.copytree(observation, tmp_path / 'observation_copy')
    payload['grasp']['observation_path'] = 'observation_copy'
    case_path.write_text(yaml.safe_dump(payload))
    second = noisy_rollouts._prepare_case(case_path, tmp_path)
    assert first.source['group_id'] == second.source['group_id']
    assert first.source['observation_group_id'] == second.source['observation_group_id']
    # Trailing bytes leave this trusted pickle's decoded proposal unchanged but
    # deliberately change its file identity. Scene grouping must ignore that.
    changed_proposal = source / 'proposal_copy.pkl'
    changed_proposal.write_bytes((source / 'proposal.pkl').read_bytes() + b'\n')
    payload['grasp']['prediction_path'] = 'source/proposal_copy.pkl'
    case_path.write_text(yaml.safe_dump(payload))
    third = noisy_rollouts._prepare_case(case_path, tmp_path)
    assert third.source['group_id'] != first.source['group_id']
    assert third.source['observation_group_id'] == first.source['observation_group_id']
    payload['grasp']['prediction_path'] = 'source/proposal.pkl'
    case_path.write_text(yaml.safe_dump(payload))
    manifest = tmp_path / 'cases.json'
    manifest.write_text(json.dumps(['case.yaml']))
    output = tmp_path / 'output'
    summary = collect_noisy_rollouts(manifest, output, project_root=tmp_path,
                                     noise_amplitudes_rad=[0, .05], repetitions=1)
    assert summary['completed'] == 2 and summary['errors'] == 0
    plan = json.loads((output / 'plan.json').read_text())
    assert plan['episodes'][0]['noise_seed'] == plan['episodes'][1]['noise_seed']
    assert 'noisy_rollouts.py' in plan['source_file_sha256']
    for episode in plan['episodes']:
        path = output / episode['episode_id']
        for name in ('result.json', 'trajectory.npz', 'resolved_case.json',
                     'actions.npz', 'scene.xml', 'scene.mjb', 'collection_metadata.json', 'events.json'):
            assert (path / name).is_file()
        result = json.loads((path / 'result.json').read_text())
        metadata = json.loads((path / 'collection_metadata.json').read_text())
        assert metadata['label'] == int(result['success'])
        events = json.loads((path / 'events.json').read_text())
        assert events['binary_outcome']['label'] == metadata['label']
        assert metadata['event_annotations']['schema_version'] == events['schema_version']
        assert metadata['runtime_versions']['numpy'] == np.__version__
        assert metadata['runtime_versions']['pinocchio'] == pinocchio.__version__
        assert metadata['diagnostic_timing'] == events['observations']['diagnostic_timing']
        assert metadata['diagnostic_timing']['diagnostic_lag_s'] == .002
        assert metadata['noise_application']['nonzero_requested_bias'] == bool(episode['noise_amplitude_rad'])
        assert metadata['group_id'] == first.source['group_id']
        assert metadata['source_file_sha256'] == plan['source_file_sha256']
        for name in ('pose_validation.py', '../integrations/hug.py', '../integrations/hug_frames.py',
                     'rollout_events.py'):
            assert name in metadata['source_file_sha256']
        assert metadata['privileged_replay_only_fields'] == ['full_qpos', 'full_qvel', 'integration_state']
        assert 'whitelist' in metadata['training_input_policy']
        assert result['artifacts']['video'] is None


def test_numerical_instability_cannot_be_published_as_a_grasp_failure(pose):
    runner = make_runner()
    trace = execute(runner, pose)
    noisy_rollouts._validate_execution(runner, trace)
    original = runner.action_history[-1]['full_qvel'][0]
    runner.action_history[-1]['full_qvel'][0] = np.nan
    with pytest.raises(RuntimeError, match='nonfinite'):
        noisy_rollouts._validate_execution(runner, trace)
    runner.action_history[-1]['full_qvel'][0] = original
    steps = list(trace.steps)
    steps[1] = replace(steps[1], time_s=steps[0].time_s)
    with pytest.raises(RuntimeError, match='strictly increasing'):
        noisy_rollouts._validate_execution(runner, ExecutionTrace(tuple(steps)))
    runner.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
    with pytest.raises(RuntimeError, match='numerical instability'):
        noisy_rollouts._validate_execution(runner, trace)
