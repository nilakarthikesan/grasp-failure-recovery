from pathlib import Path
import json,hashlib
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction');R=Path('/private/tmp/compiled_hand_audit_20261015_v1/shadow/export_integration_fix');L=R.parent/'legacy_callsite_fix'
paths=['src/grasp_failure_prediction/evaluation/runner.py','src/grasp_failure_prediction/evaluation/recording.py','src/grasp_failure_prediction/evaluation/render_rollout.py','src/grasp_failure_prediction/evaluation/noisy_rollouts.py','src/grasp_failure_prediction/integrations/sim_observations.py','scripts/capture_sim_observation.py','scripts/sample_hug_sim_proposals.py','scripts/audit_hug_execution.py','scripts/validate_shadow_baseline.py','scripts/actuated_shadow_pilot.py','tests/test_noisy_rollouts.py']
inputs={};outputs={}
for rel in paths:
 p=W/rel;raw=p.read_bytes();inputs[rel]=hashlib.sha256(raw).hexdigest();before=R/'before'/rel;before.parent.mkdir(parents=True,exist_ok=True);before.write_bytes(raw)
 s=raw.decode()
 if rel.endswith('/runner.py'):
  assert inputs[rel]=='35ec1ba83262027154e62a86aba66379152d550cb70add6f03909c9e7630fdb6',inputs[rel]
  s=s.replace('def add_shadow_actuation(scene_spec: mujoco.MjSpec) -> mujoco.MjModel:','def _add_shadow_actuation_spec(scene_spec: mujoco.MjSpec) -> mujoco.MjSpec:')
  s=s.replace('''    return spec.compile()


def _joint_qpos_address''','''    return spec


def add_shadow_actuation(scene_spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Compile the existing Shadow servos from a live scene specification."""

    return _add_shadow_actuation_spec(scene_spec).compile()


def _joint_qpos_address''')
  s=s.replace('''        self.model = add_shadow_actuation(self._scene_spec)''','''        self._scene_spec = _add_shadow_actuation_spec(self._scene_spec)
        self.model = self._scene_spec.compile()''')
 elif rel.endswith('/recording.py'):
  s=s.replace('".video_scene.xml"','".video_scene.mjb"').replace('mujoco.mj_saveLastXML(str(scene_path), self.runner.model)','mujoco.mj_saveModel(self.runner.model, str(scene_path))')
 elif rel.endswith('/render_rollout.py'):
  needle='\ndef render_rollout('
  s=s.replace(needle,'''\ndef _load_scene_model(scene_path: Path) -> mujoco.MjModel:
    """Load exact compiled snapshots, with support for historical XML scenes."""

    if scene_path.suffix.lower() == ".mjb":
        return mujoco.MjModel.from_binary_path(str(scene_path))
    return mujoco.MjModel.from_xml_path(str(scene_path))


def render_rollout(''')
  s=s.replace('''    model = mujoco.MjModel.from_xml_path(str(scene_path))\n    data''','''    model = _load_scene_model(scene_path)\n    data''')
 elif rel.endswith('/noisy_rollouts.py'):
  s=s.replace('''        # XML aids inspection; binary includes meshes for self-contained state replay.
        mujoco.mj_saveLastXML(str(output / "scene.xml"), self.model)''','''        # Preserve exact compiled parameters and meshes for state replay.''')
 elif rel.endswith('/sim_observations.py'):
  s=s.replace("    mujoco.mj_saveLastXML(str(output / 'scene.xml'), runner.model)\n",'')
 elif rel.endswith('/test_noisy_rollouts.py'):
  s=s.replace("assert (tmp_path / 'scene.xml').is_file()","assert (tmp_path / 'scene.mjb').is_file()")
  s=s.replace("'actions.npz', 'scene.xml', 'scene.mjb',","'actions.npz', 'scene.mjb',")
 elif rel.endswith('/actuated_shadow_pilot.py'):
  s=(L/'actuated_shadow_pilot.py').read_text()
  s=s.replace('AdroitShadowRunner, ExecutionState, add_shadow_actuation,','AdroitShadowRunner, ExecutionState, _add_shadow_actuation_spec,')
  s=s.replace('def add_actuation(scene_spec: mujoco.MjSpec) -> mujoco.MjModel:','def _actuation_spec(scene_spec: mujoco.MjSpec) -> mujoco.MjSpec:')
  s=s.replace('''    return add_shadow_actuation(spec)


class ActuatedShadowRunner''','''    return _add_shadow_actuation_spec(spec)


def add_actuation(scene_spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Compile the historical pilot servos from a live scene specification."""
    return _actuation_spec(scene_spec).compile()


class ActuatedShadowRunner''')
  s=s.replace('''        self.model = add_actuation(self._scene_spec)''','''        self._scene_spec = _actuation_spec(self._scene_spec)
        self.model = self._scene_spec.compile()''')
  s=s.replace('mujoco.mj_saveLastXML(str(args.output / "scene.xml"), runner.model)','mujoco.mj_saveModel(runner.model, str(args.output / "scene.mjb"))')
 else:
  repl={
   'scripts/capture_sim_observation.py':('mujoco.mj_saveLastXML(str(output / "scene.xml"), runner.model)','mujoco.mj_saveModel(runner.model, str(output / "scene.mjb"))'),
   'scripts/sample_hug_sim_proposals.py':('mujoco.mj_saveLastXML(str(args.output / "scene.xml"), runner.model)','mujoco.mj_saveModel(runner.model, str(args.output / "scene.mjb"))'),
   'scripts/audit_hug_execution.py':('mujoco.mj_saveLastXML(str(args.output / "scene.xml"), trial_runner.model)','mujoco.mj_saveModel(trial_runner.model, str(args.output / "scene.mjb"))'),
   'scripts/validate_shadow_baseline.py':("mujoco.mj_saveLastXML(str(args.output/'scene.xml'),probe.model)","mujoco.mj_saveModel(probe.model,str(args.output/'scene.mjb'))"),
  };old,new=repl[rel];assert old in s;s=s.replace(old,new)
 compile(s,rel,'exec');out=R/'after'/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s);outputs[rel]=hashlib.sha256(out.read_bytes()).hexdigest()
receipt={'scope':'Exact compiled MJB export instead of process-global precision-limited XML, preserving legacy XML render reader. Retain actual actuated live spec for static query builders. No physics, inference, evaluation outputs or training generated.','inputs':inputs,'outputs':outputs}
(R/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
