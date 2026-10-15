"""Static/mocked integration only: no simulator integration or HUG inference."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

from grasp_failure_prediction.evaluation.prediction_recording import PredictionRecorder, load_attempt
from grasp_failure_prediction.evaluation.registry import load_environment_registry
from grasp_failure_prediction.evaluation.runner import ExecutionState, ExecutionTrace, RunnerStep
from grasp_failure_prediction.evaluation.scene_contracts import build_scene_contract
from grasp_failure_prediction.evaluation.scoring import FailureType, ScoredOutcome
from grasp_failure_prediction.evaluation.schema import canonical_sha256
from grasp_failure_prediction.prediction.recorded_dataset import PrefixSpec, export_recorded_dataset
from grasp_failure_prediction.proposals import execution
from grasp_failure_prediction.proposals import prediction_recording as recording
from grasp_failure_prediction.proposals.artifacts import file_sha256, proposal_content_hash, write_proposal
from grasp_failure_prediction.proposals.registry import load_provider_registry
from grasp_failure_prediction.proposals.schema import GraspProposal, ProposalExecutionValidation


SCHEMA = "hug_prediction_recording_v1"
HASH = "sha256:" + "a" * 64
GEOMETRY = "sha256:45c1162a4f815d65c43254f2f499f53bde1eecea4d39912d8ce36d2c5cd92a81"
JOINTS = list(json.loads((Path(__file__).parent / "fixtures/provider_saved_cube.json").read_text())["joint_targets_rad"])


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("all physics APIs are forbidden in recorder integration tests")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)


class FakeRecordedRunner:
    def __init__(self, protocol, *args, **kwargs):
        self.protocol = protocol
        self.control_timestep_s = .04
        self.physics_timestep_s = .002
        self.model = SimpleNamespace()
        self.data = SimpleNamespace(qpos=np.zeros(38), qvel=np.zeros(36), qacc=np.zeros(36), ctrl=np.zeros(24))
        self.prediction_recorder = PredictionRecorder(JOINTS, np.tile([-1., 1.], (24, 1)))

    def reset(self, **kwargs):
        pass


def fake_execute(runner, *args):
    steps = []
    for index in range(80):
        phase = ExecutionState.CLOSE_FINGERS if index < 74 else ExecutionState.LIFT
        time = round((index + 1) * .04, 10)
        runner.prediction_recorder.append(time_s=time, action_interval_start_s=round(index * .04, 10),
            phase=phase.value, hand_qpos=np.full(24, .2), hand_qvel=np.full(24, .3),
            requested_target_rad=np.full(24, 2.), applied_target_rad=np.ones(24),
            diagnostics={"full_qpos": np.zeros(38), "full_qvel": np.zeros(36), "contact_count": np.asarray(1)})
        steps.append(RunnerStep(index, time, phase, np.zeros(3), np.full(24, .2), np.array([0., 0., .03]),
                                1, True, False, True))
    return ExecutionTrace(tuple(steps))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    # Transaction tests start after scientific admission and use a manufactured
    # runner. Separate admission tests retain the real fidelity refusal gate.
    monkeypatch.setattr(recording.HugPredictionRecording,"require_fidelity_for_execution",lambda self:None)
    native = tmp_path / "proposals/sample/native"
    native.mkdir(parents=True)
    for name in ("rgb.png", "depth.png", "object_mask.png"):
        (native / name).write_bytes(b"synthetic-test-image")
    np.save(native / "depth_m.npy", np.ones((1, 1)))
    np.save(native / "intrinsics.npy", np.eye(3))
    np.save(native / "T_world_camera.npy", np.eye(4))
    (native / "proposal.pkl").write_bytes(b"synthetic-native-hash-fixture-not-unpickled")
    environment_hash = load_environment_registry().config_hash("adroit_shadow_tabletop_v2")
    images = {name: file_sha256(native / name) for name in (
        "rgb.png", "depth.png", "object_mask.png", "depth_m.npy", "intrinsics.npy", "T_world_camera.npy")}
    contract = build_scene_contract(object_id="object01", geometry_content_hash=GEOMETRY, mass_kg=.18,
        object_position_m=[0., 0., .03], object_orientation_xyzw=[0., 0., 0., 1.],
        environment_config_hash=environment_hash, camera_sha256=images["T_world_camera.npy"], observation_files_sha256=images)
    (native / "scene_contract.json").write_text(json.dumps(contract))
    (native / "observation_manifest.json").write_text(json.dumps({"selection_uv": [0, 0], "source": {"kind": "synthetic_test"}}))
    contract_hash, manifest_hash = [file_sha256(native / name) for name in ("scene_contract.json", "observation_manifest.json")]
    report = {"real_hug_inference_passed": True, "proposal_sha256": file_sha256(native / "proposal.pkl").removeprefix("sha256:"),
        "scene_contract_sha256": contract_hash.removeprefix("sha256:"), "seed": 7,
        "manifest_sha256": manifest_hash.removeprefix("sha256:"),
        "checkpoint_sha256": HASH.removeprefix("sha256:"), "hug_code_commit": "synthetic-hug-revision",
        "hug_depth_png_sha256": images["depth.png"].removeprefix("sha256:"),
        "observation_hashes": {key: images[name].removeprefix("sha256:") for key, name in (
            ("rgb_path", "rgb.png"), ("depth_m_path", "depth_m.npy"), ("intrinsics_path", "intrinsics.npy"), ("mask_path", "object_mask.png"))}}
    (native / "inference_report.json").write_text(json.dumps(report))
    provider = load_provider_registry().resolve(recording.HUG_PROVIDER_ID)
    observation_id = canonical_sha256({"input_manifest_hash": manifest_hash, "scene_contract_hash": contract_hash, "object_geometry": GEOMETRY})
    proposal = GraspProposal.model_validate({
        "proposal_id": "hug_fixture_sample_0000", "provider": {"id": provider.provider_id,
            "config_hash": canonical_sha256(provider), "code_revision": "synthetic-hug-revision", "adapter_code_revision": "synthetic", "checkpoint_hash": HASH},
        "generation": {"seed": 7, "sample_index": 0}, "object": {"id": "object01", "geometry_content_hash": GEOMETRY},
        "object_to_palm": {"translation_m": [0., .045, .04], "quaternion_xyzw": [0., 0., 0., 1.]},
        "joint_targets_rad": {name: .1 for name in JOINTS}, "compatible_execution_protocols": ["fixed_grasp_lift_v2"],
        "input_binding": {"geometry_group_id": canonical_sha256({"geometry_content_hash": GEOMETRY}),
            "observation_group_id": observation_id, "generation_request_group_id": HASH,
            "input_manifest_hash": manifest_hash, "scene_contract_hash": contract_hash},
        "provider_validation": {"status": "passed", "metrics": {"mean_fingertip_error_m": .005}},
        "native_artifacts": [{"role": role, "path": "native/" + name, "sha256": file_sha256(native / name)}
            for role, name in (("provider_output", "proposal.pkl"), ("provider_report", "inference_report.json"))]})
    proposal_path = native.parent / "proposal.json"
    write_proposal(proposal_path, proposal)
    case = {"schema_version": "eval_case_v2", "case_id": "recorded_hug_smoke",
        "environment": {"id": "adroit_shadow_tabletop_v2"}, "embodiment": {"id": "shadow_hand_right"},
        "execution_protocol": {"id": "fixed_grasp_lift_v2"},
        "object": {"id": "object01", "mass_kg": .18, "friction_profile_id": "nominal_v1"},
        "grasp": {"id": proposal.proposal_id, "proposal_path": str(proposal_path.relative_to(tmp_path)), "expected_proposal_hash": proposal_content_hash(proposal)},
        "initial_condition": {"id": "synthetic_capture", "object_position_m": [0., 0., .03], "object_orientation_xyzw": [0., 0., 0., 1.]},
        "motion_profile_id": "nominal_lift_v1", "seed": 19}
    case_path = tmp_path / "case.yaml"
    case_path.write_text(yaml.safe_dump(case))
    asset = tmp_path / "synthetic_hand_asset.bin"
    asset.write_bytes(b"unit-test-only")
    monkeypatch.setattr(recording, "_hand_files", lambda: {asset: file_sha256(asset)})
    monkeypatch.setattr(recording, "CheckedRecordedShadowRunner", FakeRecordedRunner)
    monkeypatch.setattr(execution, "_validate_pose", lambda p: (
        execution.NamedJointPose(tuple(JOINTS), np.full(24, .1)),
        ProposalExecutionValidation(joint_count=24, minimum_joint_limit_margin_rad=.1, self_contact_count=0,
                                    provider_validation_status=p.provider_validation.status)))
    monkeypatch.setattr(execution, "_execute", fake_execute)
    monkeypatch.setattr(execution, "score_trace", lambda *a, **k: ScoredOutcome(True, None, True, .15, 2., .18, False))
    monkeypatch.setattr(mujoco, "mj_saveModel", lambda model, path: Path(path).write_bytes(b"synthetic-compiled-model"))
    return SimpleNamespace(root=tmp_path, case=case, case_path=case_path, proposal=proposal, proposal_path=proposal_path, native=native)


def run(bundle, output="run", **options):
    return execution.run_proposal_case(bundle.case_path, bundle.root / output, project_root=bundle.root,
                                       record_video=False, recording_schema=SCHEMA, **options)


def test_opt_in_records_aligned_observables_hashes_labels_and_export(bundle):
    result = run(bundle)
    assert result.status == "completed"
    root = bundle.root / "run"
    metadata, payload, arrays = load_attempt(root / "prediction_recording")
    assert metadata.mapping.status.value == "ok" and metadata.outcome.success is True
    assert arrays["hand_qvel"].shape == (80, 24)
    np.testing.assert_array_equal(arrays["requested_target_rad"], 2.)
    np.testing.assert_array_equal(arrays["applied_target_rad"], 1.)
    assert set(metadata.provenance_hashes) >= {"code", "configuration", "inputs", "model", "assets", "native_pose", "mapped_pose", "label_rule"}
    assert not (root / "_pending_execution").exists()
    assert (root / "result.json").is_file() and (root / "scene.mjb").is_file()
    receipt = json.loads((root / "recording_manifest.json").read_text())
    assert receipt["source_generator_verification"] == "verified_hug"
    report = export_recorded_dataset([root / "prediction_recording"], bundle.root / "derived", prefix_spec=PrefixSpec(
        protocol_id="synthetic_test", prediction_time_s=2.96, history_steps=20, control_timestep_s=.04))
    assert report["counts"]["included"] == 1 and report["counts"]["success"] == 1


def test_default_execution_keeps_legacy_layout_and_does_not_enable_recorder(bundle, monkeypatch):
    from grasp_failure_prediction.evaluation import mapping_fidelity
    # Exercise legacy layout after a manufactured admission; real HUG default
    # admission refuses independently in test_hug_fidelity_admission.py.
    monkeypatch.setattr(mapping_fidelity,"require_fidelity_pass",lambda *args,**kwargs:None)
    monkeypatch.setattr(execution, "_runner_class", lambda resolved: FakeRecordedRunner)
    monkeypatch.setattr(recording, "HugPredictionRecording", lambda *a, **k: pytest.fail("default mode enabled recorder"))
    execution.run_proposal_case(bundle.case_path, bundle.root / "legacy", project_root=bundle.root, record_video=False)
    root = bundle.root / "legacy"
    assert (root / "result.json").is_file() and (root / "trajectory.npz").is_file()
    assert not (root / "recording_manifest.json").exists() and not (root / "prediction_recording").exists()


def test_native_verification_is_forced_even_when_optional_flag_false(bundle):
    (bundle.native / "proposal.pkl").write_bytes(b"tampered native")
    with pytest.raises(ValueError, match="native proposal artifact hash mismatch"):
        run(bundle, verify_native_artifacts=False)
    metadata, _, arrays = load_attempt(bundle.root / "run/prediction_recording")
    assert metadata.mapping.status.value == "setup_error" and metadata.outcome is None and arrays is None
    assert not (bundle.root / "run/result.json").exists()


def test_default_hug_execution_cannot_omit_recorder_to_bypass_fidelity(bundle,monkeypatch):
    from grasp_failure_prediction.evaluation.mapping_fidelity import MappingFidelityRefused
    monkeypatch.setattr(execution,"_execute",lambda *args:pytest.fail("unqualified HUG task executed"))
    with pytest.raises(MappingFidelityRefused):
        execution.run_proposal_case(bundle.case_path,bundle.root/"legacy_refused",project_root=bundle.root,record_video=False)
    assert not (bundle.root/"legacy_refused/result.json").exists()


def test_structured_rejection_is_preserved_and_never_labelled(bundle):
    bundle.proposal = bundle.proposal.model_copy(update={"object_to_palm": bundle.proposal.object_to_palm.model_copy(update={"translation_m": (1., 0., 0.)})})
    write_proposal(bundle.proposal_path, bundle.proposal)
    bundle.case["grasp"]["expected_proposal_hash"] = proposal_content_hash(bundle.proposal)
    bundle.case_path.write_text(yaml.safe_dump(bundle.case))
    with pytest.raises(execution.ProposalRejected):
        run(bundle)
    metadata, _, arrays = load_attempt(bundle.root / "run/prediction_recording")
    assert metadata.mapping.status.value == "refused_guard" and metadata.outcome is None and arrays is None
    assert (bundle.root / "run/rejection.json").is_file()
    assert not (bundle.root / "run/result.json").exists()


def test_refusal_stays_in_export_denominator_without_negative_label(bundle):
    run(bundle, "good")
    bundle.proposal = bundle.proposal.model_copy(update={"object_to_palm": bundle.proposal.object_to_palm.model_copy(update={"translation_m": (1., 0., 0.)})})
    write_proposal(bundle.proposal_path, bundle.proposal)
    bundle.case["case_id"] = "refused_hug_attempt"
    bundle.case["grasp"]["expected_proposal_hash"] = proposal_content_hash(bundle.proposal)
    bundle.case_path.write_text(yaml.safe_dump(bundle.case))
    with pytest.raises(execution.ProposalRejected):
        run(bundle, "refused")
    manifest = export_recorded_dataset([bundle.root / "good/prediction_recording", bundle.root / "refused/prediction_recording"],
        bundle.root / "export", prefix_spec=PrefixSpec(protocol_id="synthetic", prediction_time_s=2.96, history_steps=20, control_timestep_s=.04))
    assert manifest["counts"]["attempted"] == 2 and manifest["counts"]["included"] == 1
    assert manifest["counts"]["excluded"] == 1 and manifest["counts"]["failure"] == 0


@pytest.mark.parametrize("condition,message", [
    ("guard", "engineering guard"), ("observation", "group identities"),
    ("checkpoint", "checkpoint differs"), ("revision", "code revision differs"),
])
def test_invalid_native_provenance_or_admission_stops_before_execution(bundle, monkeypatch, condition, message):
    proposal = bundle.proposal
    if condition == "guard":
        proposal = proposal.model_copy(update={"provider_validation": proposal.provider_validation.model_copy(update={"metrics": {"mean_fingertip_error_m": .02}})})
    elif condition == "observation":
        proposal = proposal.model_copy(update={"input_binding": proposal.input_binding.model_copy(update={"observation_group_id": HASH})})
    elif condition == "checkpoint":
        proposal = proposal.model_copy(update={"provider": proposal.provider.model_copy(update={"checkpoint_hash": "sha256:" + "b" * 64})})
    else:
        proposal = proposal.model_copy(update={"provider": proposal.provider.model_copy(update={"code_revision": "wrong"})})
    write_proposal(bundle.proposal_path, proposal)
    bundle.case["grasp"]["expected_proposal_hash"] = proposal_content_hash(proposal)
    bundle.case_path.write_text(yaml.safe_dump(bundle.case))
    monkeypatch.setattr(execution, "_execute", lambda *a, **k: pytest.fail("invalid native input reached execution"))
    with pytest.raises(ValueError, match=message):
        run(bundle)
    attempt = load_attempt(bundle.root / "run/prediction_recording")[0]
    assert attempt.outcome is None
    assert attempt.mapping.status.value == ("refused_guard" if condition == "guard" else "setup_error")


def test_capture_mass_mismatch_is_not_a_physical_failure(bundle, monkeypatch):
    bundle.case["object"]["mass_kg"] = .2
    bundle.case_path.write_text(yaml.safe_dump(bundle.case))
    monkeypatch.setattr(execution, "_execute", lambda *a, **k: pytest.fail("unbound mass reached execution"))
    with pytest.raises(ValueError, match="mass_kg mismatch"):
        run(bundle)
    attempt = load_attempt(bundle.root / "run/prediction_recording")[0]
    assert attempt.mapping.status.value == "setup_error" and attempt.outcome is None


def test_runtime_failure_retains_partial_arrays_and_original_exception(bundle, monkeypatch):
    original = RuntimeError("synthetic runtime failure")
    def fail(*args):
        fake_execute(*args)
        raise original
    monkeypatch.setattr(execution, "_execute", fail)
    with pytest.raises(RuntimeError) as caught:
        run(bundle)
    assert caught.value is original
    metadata, _, arrays = load_attempt(bundle.root / "run/prediction_recording")
    assert metadata.execution_status == "runtime_error" and metadata.outcome is None and arrays is None
    assert (bundle.root / "run/_incomplete_execution/partial_diagnostics.npz").exists()


def test_post_execution_input_change_quarantines_completed_result(bundle, monkeypatch):
    def mutate(*args):
        trace = fake_execute(*args)
        (bundle.native / "depth_m.npy").write_bytes(b"changed after collection started")
        return trace
    monkeypatch.setattr(execution, "_execute", mutate)
    with pytest.raises(ValueError, match="binding changed"):
        run(bundle)
    metadata, _, _ = load_attempt(bundle.root / "run/prediction_recording")
    assert metadata.outcome is None and metadata.execution_status == "runtime_error"
    assert not (bundle.root / "run/result.json").exists()
    assert (bundle.root / "run/_incomplete_execution/result.json").is_file()


def test_recording_validation_error_still_writes_null_runtime_attempt(bundle, monkeypatch):
    def malformed(*args):
        trace = fake_execute(*args)
        args[0].prediction_recorder._rows[-1]["hand_qvel"] = np.zeros(23)
        return trace
    monkeypatch.setattr(execution, "_execute", malformed)
    with pytest.raises(ValueError):
        run(bundle)
    metadata, _, _ = load_attempt(bundle.root / "run/prediction_recording")
    assert metadata.outcome is None and metadata.execution_status == "runtime_error"
    assert (bundle.root / "run/_incomplete_execution/partial_recording_error.json").exists()


def test_early_schema_error_retains_unresolved_setup_attempt(bundle):
    bundle.case_path.write_text("schema_version: invalid\n")
    with pytest.raises(ValueError):
        run(bundle)
    metadata, _, _ = load_attempt(bundle.root / "run/prediction_recording")
    assert metadata.mapping.status.value == "setup_error" and metadata.outcome is None
    assert metadata.attempt_id.startswith("unresolved_")


def test_nonempty_output_and_unknown_schema_do_not_overwrite(bundle):
    output = bundle.root / "run"
    output.mkdir(); (output / "keep.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        run(bundle)
    assert (output / "keep.txt").read_text() == "preserve"
    with pytest.raises(ValueError, match="unsupported prediction recording schema"):
        execution.run_proposal_case(bundle.case_path, bundle.root / "other", recording_schema="unknown")


def test_runner_selection_rejects_controller_substitution(bundle):
    context = recording.HugPredictionRecording(bundle.case_path, bundle.root / "unused", bundle.root)
    with pytest.raises(ValueError, match="unchanged registered"):
        context.runner_class(FakeRecordedRunner)


@pytest.mark.parametrize("condition", ["warning", "time_reset", "nonfinite"])
def test_checked_runner_refuses_control_boundary_numerical_failures(monkeypatch, condition):
    runner = object.__new__(recording.CheckedRecordedShadowRunner)
    runner.control_timestep_s = .04
    runner.data = SimpleNamespace(time=1., qpos=np.zeros(2), qvel=np.zeros(2), qacc=np.zeros(2), ctrl=np.zeros(2),
                                  warning=[SimpleNamespace(number=0) for _ in range(8)])
    def parent_advance(self, *args, **kwargs):
        self.data.time += .04
        if condition == "warning":
            self.data.warning[int(mujoco.mjtWarning.mjWARN_BADQPOS)].number = 1
        elif condition == "time_reset":
            self.data.time = .04
        else:
            self.data.qacc[0] = np.nan
    monkeypatch.setattr(recording.RecordedActuatedShadowRunner, "_advance", parent_advance)
    with pytest.raises(RuntimeError, match="warning|reset|nonfinite"):
        runner._advance(None, None, None, None)


def test_cli_exposes_explicit_recording_schema(monkeypatch):
    seen = {}
    monkeypatch.setattr(sys, "argv", ["runner", "case.yaml", "--output", "out", "--recording-schema", SCHEMA, "--no-video"])
    monkeypatch.setattr(execution, "run_proposal_case", lambda *a, **kwargs: seen.update(kwargs))
    execution.main()
    assert seen["recording_schema"] == SCHEMA and seen["record_video"] is False
