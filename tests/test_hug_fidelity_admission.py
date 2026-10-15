"""Covered HUG entry points must not confuse numerical PASS with fidelity."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from grasp_failure_prediction.proposals.prediction_recording import HugPredictionRecording, RecordingAdmissionError
from test_proposal_prediction_recording import bundle


# The imported transaction fixture mocks admission. This one test restores the
# real method, leaving only its manufactured runner/inputs/scene checks mocked.
REAL_RECORDING_FIDELITY = HugPredictionRecording.require_fidelity_for_execution


def driver_module():
    path=Path(__file__).parents[1]/"scripts/run_hug_build_collection.py"
    spec=importlib.util.spec_from_file_location("fidelity_collection_driver",path)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    return driver


def mapped_row(tmp_path):
    return dict(mapping_status="ok",proposal_path=str(tmp_path/"proposal.json"),
                mapping_validation=dict(static_input_gate="PASS",native_surface_fidelity_accepted=False))


def test_collection_numerical_pass_does_not_start_filters_or_create_output(tmp_path,monkeypatch):
    from grasp_failure_prediction.evaluation.mapping_fidelity import MappingFidelityRefused
    driver=driver_module();output=tmp_path/"evidence";output.mkdir()
    driver.write(output/"generation.json",dict(status="COMPLETED_BOUNDED_GENERATION",artifact_bindings={},
                                             conditions={},proposals=[mapped_row(tmp_path)]))
    protocol=dict(raw_output=str(tmp_path/"raw"),quotas=dict(pilot_attempts=20))
    monkeypatch.setattr(driver,"read_protocol",lambda output:protocol)
    monkeypatch.setattr(driver,"_freeze_schedules",lambda *args:{})
    monkeypatch.setattr(driver,"authorized",lambda *args,**kwargs:None)
    monkeypatch.setattr(driver,"_run_filters",lambda *args:pytest.fail("filters started before fidelity admission"))
    with pytest.raises(MappingFidelityRefused):driver.collect(output,tmp_path/"auth.json","pilot")
    assert not (tmp_path/"raw").exists()


def test_direct_filter_and_attempt_entrypoints_cannot_bypass_gate(tmp_path):
    from grasp_failure_prediction.evaluation.mapping_fidelity import MappingFidelityRefused
    driver=driver_module();row=mapped_row(tmp_path)
    with pytest.raises(MappingFidelityRefused):driver._run_filters({},dict(proposals=[row]),tmp_path/"filter")
    with pytest.raises(MappingFidelityRefused):driver._collect_attempt({},row,{},tmp_path/"task",{})
    assert not (tmp_path/"filter").exists() and not (tmp_path/"task").exists()
    with pytest.raises(ValueError,match="cannot enter physical collection"):
        driver._collect_attempt({},dict(mapping_status="refused_guard"),{},tmp_path/"refused",{})
    assert not (tmp_path/"refused").exists()


def test_recording_fidelity_refusal_remains_unexecuted_without_negative_label(tmp_path):
    from grasp_failure_prediction.evaluation.runner import ActuatedShadowRunner
    context=HugPredictionRecording(tmp_path/"missing-case.yaml",tmp_path/"attempt",tmp_path)
    context.proposal=SimpleNamespace(native_artifacts=[])
    with pytest.raises(RecordingAdmissionError) as refused:context.runner_class(ActuatedShadowRunner)
    assert refused.value.mapping_status.value=="refused_guard"
    assert refused.value.code=="native_surface_fidelity_unqualified"
    assert context.runner is None and not context.execution_started


def test_real_recording_fidelity_gate_persists_null_refusal_before_runner_creation(bundle,monkeypatch):
    from grasp_failure_prediction.proposals import execution,prediction_recording as recording
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    monkeypatch.setattr(HugPredictionRecording,"require_fidelity_for_execution",REAL_RECORDING_FIDELITY)
    monkeypatch.setattr(recording,"CheckedRecordedShadowRunner",lambda *a,**k:pytest.fail("runner constructed before fidelity admission"))
    output=bundle.root/"fidelity_refusal"
    with pytest.raises(RecordingAdmissionError) as refusal:
        execution.run_proposal_case(bundle.case_path,output,project_root=bundle.root,
                                    record_video=False,recording_schema=recording.RECORDING_SCHEMA)
    assert refusal.value.code=="native_surface_fidelity_unqualified"
    metadata,payload,arrays=load_attempt(output/"prediction_recording")
    assert metadata.mapping.status.value=="refused_guard"
    assert metadata.mapping.code=="native_surface_fidelity_unqualified"
    assert metadata.execution_status=="not_executed" and metadata.outcome is None and arrays is None
    receipt=json.loads((output/"recording_manifest.json").read_text())
    assert receipt["outcome"] is None and receipt["execution_status"]=="not_executed"
    assert receipt["source_generator_verification"]=="verified_hug"
    assert (output/"_incomplete_execution/exception_traceback.txt").is_file()
    assert not any((output/name).exists() for name in ("result.json","trajectory.npz","scene.mjb","_pending_execution"))


def test_refused_mappings_do_not_require_success_certificate(tmp_path):
    driver=driver_module()
    driver._require_collection_fidelity(dict(proposals=[dict(mapping_status="refused_guard")]))


def test_legacy_case_runner_refuses_before_inference_scene_or_output(tmp_path,monkeypatch):
    from grasp_failure_prediction.evaluation import case_runner
    from grasp_failure_prediction.evaluation.mapping_fidelity import MappingFidelityRefused
    monkeypatch.setattr(case_runner,"load_case",lambda path:object())
    monkeypatch.setattr(case_runner,"resolve_case",lambda case:object())
    monkeypatch.setattr(case_runner,"_validate_case_configuration",lambda case:None)
    monkeypatch.setattr(case_runner,"_resolve_object_geometry",lambda *args:pytest.fail("scene work before admission"))
    monkeypatch.setattr(case_runner,"_run_case_inference",lambda *args,**kw:pytest.fail("inference before admission"))
    with pytest.raises(MappingFidelityRefused):case_runner.run_case(tmp_path/"case",tmp_path/"output")
    assert not (tmp_path/"output").exists()


def test_saved_pilot_refuses_before_executor_or_output(tmp_path,monkeypatch):
    from grasp_failure_prediction.evaluation.mapping_fidelity import MappingFidelityRefused
    spec=importlib.util.spec_from_file_location("fidelity_saved_pilot",Path(__file__).parents[1]/"scripts/run_hug_saved_recording_pilot.py")
    pilot=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,pilot);spec.loader.exec_module(pilot)
    output=tmp_path/"evidence";output.mkdir()
    pilot.write(output/"protocol.json",dict(project=str(tmp_path),dependency_dirs=[],raw_output=str(tmp_path/"raw")))
    pilot.write(tmp_path/"authorization.json",{})
    monkeypatch.setattr(pilot,"verify_authorization",lambda *args:None)
    with pytest.raises(MappingFidelityRefused):pilot.run(output,tmp_path/"authorization.json",executor=lambda *args:pytest.fail("task started"))
    assert not (tmp_path/"raw").exists() and not (output/"summary.json").exists()
