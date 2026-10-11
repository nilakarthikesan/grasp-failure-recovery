"""Fresh source novelty, frozen weights and independently regenerated labels.

Synthetic source files below are test fixtures, not generated HUG predictions.
The evaluator hashes these proposal bytes but never unpickles them.
"""

import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from grasp_failure_prediction.evaluation.case_runner import _geometry_content_hash, _source_scene_binding
from grasp_failure_prediction.evaluation.object_assets import load_object_geometry
from grasp_failure_prediction.evaluation.registry import resolve_case
from grasp_failure_prediction.evaluation.scene_contracts import build_scene_contract
from grasp_failure_prediction.evaluation.schema import EvaluationCase, canonical_sha256
from grasp_failure_prediction.prediction.dataset import _sha256, load_dataset
from grasp_failure_prediction.prediction.fresh_evaluation import REGISTRY_FILES, evaluate_fresh_collection
from grasp_failure_prediction.prediction.model import load_model, summarize_prefix
from test_evaluation_schema import valid_case
from test_object_assets import write_box_mesh


SOURCE = Path(__file__).resolve().parents[1]
ORIGINAL = SOURCE / "research_artifacts/object_scene_pilot_v1"
MODELS = SOURCE / "research_artifacts/task_outcome_baseline_v1"


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def identity(metadata):
    source = {key: metadata[key] for key in (
        "observation_files_sha256", "proposal_sha256", "camera_sha256", "geometry_content_hash",
        "object_position_m", "object_orientation_xyzw", "observation_manifest_sha256")}
    metadata.update(group_id=canonical_sha256(source),
                    observation_group_id=canonical_sha256({key: value for key, value in source.items() if key != "proposal_sha256"}),
                    geometry_group_id=canonical_sha256({"geometry_content_hash": metadata["geometry_content_hash"]}))


def rewrite_index(root):
    records = [json.loads((episode / "collection_metadata.json").read_text())
               for episode in sorted(root.glob("episode_*"))]
    (root / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        lookup = {row["episode_id"]: row for row in records}
        plan["episodes"] = [{key: lookup[row["episode_id"]][key] for key in row}
                            for row in plan["episodes"]]
        write_json(plan_path, plan)
        write_summary(root, records)


def write_summary(root, records):
    completed = [row for row in records if row["status"] == "completed"]
    successes = sum(row["label"] for row in completed)
    write_json(root / "summary.json", {"planned": len(records), "completed": len(completed),
                                       "successes": successes, "failures": len(completed) - successes,
                                       "errors": len(records) - len(completed)})


@pytest.fixture
def fresh(tmp_path):
    original = load_dataset(ORIGINAL)
    source_hashes = next(iter(original.manifest["provenance_sources"].values()))["source_file_sha256"]
    root = tmp_path / "collection"
    root.mkdir()
    for scene in range(2):
        observation = tmp_path / f"observation_{scene}"
        observation.mkdir()
        for filename in ("rgb.png", "depth.png", "depth_m.npy", "intrinsics.npy", "object_mask.png", "T_world_camera.npy"):
            (observation / filename).write_bytes(f"fixture {scene} {filename}".encode())
        hashes = {path.name: _sha256(path) for path in sorted(observation.iterdir())}
        manifest = {"selection_uv": [0, 0], "source": {"kind": "test_fixture", "seed": 10100 + scene}}
        write_json(observation / "observation_manifest.json", manifest)
        payload = valid_case()
        payload["seed"] = 10100 + scene
        payload["grasp"].update(prediction_path=f"proposal_{scene}/proposal.pkl",
                                 observation_path=f"observation_{scene}", inference_seed=0)
        geometry = None
        if scene == 1:
            geometry_xml = tmp_path / "new_geometry.xml"
            write_box_mesh(tmp_path / "fixture.obj", [0, 0, 0], [.025, .025, .04])
            geometry_xml.write_text('<mujoco model="fixture"><compiler angle="radian" meshdir="."/>'
                                    '<asset><mesh name="fixture" file="fixture.obj"/></asset>'
                                    '<worldbody><body name="object"><freejoint/>'
                                    '<inertial pos="0 0 0" mass="0.18" diaginertia="0.0002 0.0002 0.0002"/>'
                                    '<geom type="mesh" mesh="fixture"/>'
                                    '</body></worldbody></mujoco>')
            geometry = load_object_geometry(geometry_xml)
            payload["object"]["geometry"] = {"mjcf_path": "new_geometry.xml", "expected_content_hash": geometry.content_hash}
        case = EvaluationCase.model_validate(payload)
        resolved = resolve_case(case)
        contract = build_scene_contract(object_id=case.object.id, geometry_content_hash=_geometry_content_hash(geometry),
                                        mass_kg=case.object.mass_kg,
                                        object_position_m=case.initial_condition.object_position_m,
                                        object_orientation_xyzw=case.initial_condition.object_orientation_xyzw,
                                        environment_config_hash=resolved.environment_config_hash,
                                        camera_sha256=hashes["T_world_camera.npy"], observation_files_sha256=hashes)
        write_json(observation / "scene_contract.json", contract)
        proposal_dir = tmp_path / f"proposal_{scene}"
        shutil.copytree(observation, proposal_dir)
        proposal = proposal_dir / "proposal.pkl"
        proposal.write_bytes(f"test fixture proposal {scene}".encode())
        report = {"real_hug_inference_passed": True, "seed": 0, "sampling_steps": 50,
                  "device": "test_fixture", "dtype": "float32", "hug_code_commit": "c" * 40,
                  "checkpoint_sha256": "d" * 64, "inference_script_sha256": "e" * 64,
                  "proposal_sha256": _sha256(proposal).removeprefix("sha256:"),
                  "scene_contract_sha256": _sha256(proposal_dir / "scene_contract.json").removeprefix("sha256:"),
                  "manifest_sha256": _sha256(proposal_dir / "observation_manifest.json").removeprefix("sha256:"),
                  "hug_depth_png_sha256": hashes["depth.png"].removeprefix("sha256:"),
                  "observation_hashes": {key: hashes[value].removeprefix("sha256:") for key, value in
                                         {"rgb_path": "rgb.png", "depth_m_path": "depth_m.npy",
                                          "intrinsics_path": "intrinsics.npy", "mask_path": "object_mask.png"}.items()}}
        write_json(proposal_dir / "inference_report.json", report)
        binding = _source_scene_binding(case, resolved, tmp_path, proposal, geometry)
        for label in (0, 1):
            episode = root / f"episode_{scene * 2 + label:06d}"
            episode.mkdir()
            metadata = {key: value for key, value in binding.items() if key != "scene_contract"}
            metadata.update(schema_version="noisy_rollout_pilot_v3", episode_id=episode.name,
                            status="completed", label=label, failure_type=None if label else "failed_acquisition",
                            proposal_path=str(case.grasp.prediction_path), proposal_sha256=_sha256(proposal),
                            geometry_content_hash=_geometry_content_hash(geometry),
                            geometry_asset_files_sha256=geometry.files_sha256 if geometry else {},
                            object_position_m=contract["object_position_m"], object_orientation_xyzw=contract["object_orientation_xyzw"],
                            object_id=case.object.id, noise_amplitude_rad=label * .15,
                            case_index=scene, case_path=f"case_{scene}.yaml", repetition=label,
                            noise_seed=100 + label,
                            source_file_sha256=source_hashes.copy(),
                            runtime_versions={"test_fixture": "v1"})
            identity(metadata)
            write_json(episode / "collection_metadata.json", metadata)
            write_json(episode / "resolved_case.json", {
                "case": case.model_dump(mode="json"), "environment": resolved.environment.model_dump(mode="json"),
                "execution_protocol": resolved.execution_protocol.model_dump(mode="json"),
                "environment_config_hash": resolved.environment_config_hash,
                "execution_protocol_config_hash": resolved.execution_protocol_config_hash})
            write_json(episode / "result.json", {"status": "completed", "success": bool(label),
                                                "failure_type": metadata["failure_type"],
                                                "resolved_execution": {"code_commit": "f" * 40}})
            times = np.arange(1, 165) * .04
            states = np.full(len(times), "close_fingers", dtype="U20")
            states[:29], states[74:112], states[112:163], states[163:] = "approach", "lift", "hold", "score"
            qpos = np.tile(np.linspace(.01, .3, 24), (len(times), 1)) + scene * .02 + label * .04
            positions = np.zeros((len(times), 3))
            positions[:, 2] = .025
            if label:
                positions[74:, 2] = .175
            np.savez_compressed(episode / "trajectory.npz", time_s=times, state=states,
                                hand_qpos=qpos, object_position_m=positions,
                                hand_object_contact=np.ones(len(times), dtype=bool),
                                hand_table_contact=np.zeros(len(times), dtype=bool))
            np.savez_compressed(episode / "actions.npz", time_s=times,
                                joint_names=np.asarray(original.manifest["input_contract"]["joint_names"]),
                                hand_qvel=qpos * .1, commanded_target_rad=qpos + .1)
    rewrite_index(root)
    records = [json.loads((episode / "collection_metadata.json").read_text()) for episode in sorted(root.glob("episode_*"))]
    plan_fields = ("episode_id", "case_index", "case_path", "repetition", "noise_seed", "noise_amplitude_rad",
                   "group_id", "observation_group_id", "geometry_group_id", "object_id")
    write_json(root / "plan.json", {"schema_version": "noisy_rollout_plan_v1",
                                    "episodes": [{key: row[key] for key in plan_fields} for row in records],
                                    "source_file_sha256": source_hashes})
    write_summary(root, records)
    original_resolved = json.loads((root / "episode_000000/resolved_case.json").read_text())
    reference = {key: original_resolved[key] for key in
                 ("environment", "execution_protocol", "environment_config_hash", "execution_protocol_config_hash")}
    reference.update(schema_version="frozen_execution_reference_v1", embodiment_id="shadow_hand_right",
                     source_resolved_case_sha256=original.manifest["episodes"][0]["raw_files_sha256"]["resolved_case.json"].removeprefix("sha256:"),
                     registry_yaml_sha256={name: _sha256(SOURCE / name).removeprefix("sha256:") for name in REGISTRY_FILES})
    write_json(tmp_path / "execution_reference.json", reference)
    return root, tmp_path


def evaluate(fresh, name="evaluation"):
    root, project = fresh
    output = project / name
    return output, evaluate_fresh_collection(root, ORIGINAL, MODELS, output, project_root=project,
                                              execution_reference=project / "execution_reference.json")


def test_fresh_evaluation_uses_unchanged_models_and_reports_geometry_novelty(fresh):
    before = {path.name: _sha256(path) for path in MODELS.iterdir() if path.is_file()}
    output, report = evaluate(fresh)
    assert report["counts"] == {"indexed": 4, "included": 4, "excluded": 0, "success": 2, "failure": 2,
                                "observation_groups": 2, "proposal_groups": 2, "geometry_groups": 2}
    manifest = json.loads((output / "manifest.json").read_text())
    assert [row["geometry_novelty"] for row in manifest["episodes"]] == ["new_scene_seen_geometry"] * 2 + ["new_geometry"] * 2
    with np.load(output / "dataset.npz", allow_pickle=False) as data:
        assert data["X"].shape == (4, 1440)
        assert data["X"].dtype == np.float32
        expected = load_model(MODELS / "proprioception_and_commands.npz").predict_success(summarize_prefix(data["X"]))
    with (output / "predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    actual = [float(row["p_success"]) for row in rows if row["model"] == "proprioception_and_commands"]
    np.testing.assert_array_equal(expected, actual)
    assert before == {path.name: _sha256(path) for path in MODELS.iterdir() if path.is_file()}
    assert set(report["models"]) == {"training_prevalence", "proprioception_and_commands", "commands_only"}
    assert report["predictions_sha256"] == _sha256(output / "predictions.csv")
    assert not (output / "splits.json").exists()


@pytest.mark.parametrize("content", ["observation", "proposal"])
def test_renaming_identity_cannot_hide_original_source_overlap(fresh, content):
    root, project = fresh
    path = root / "episode_000000/collection_metadata.json"
    metadata = json.loads(path.read_text())
    original = load_dataset(ORIGINAL)
    provenance = next(iter(original.manifest["provenance_sources"].values()))
    if content == "observation":
        metadata["observation_files_sha256"] = provenance["observation_files_sha256"]
    else:
        metadata["proposal_sha256"] = provenance["proposal_sha256"]
    identity(metadata)
    write_json(path, metadata)
    rewrite_index(root)
    with pytest.raises(ValueError, match="overlaps original"):
        evaluate(fresh)
    assert not (project / "evaluation").exists()


@pytest.mark.parametrize("corruption", ["label", "timestamp", "order", "binding", "scorer", "controller", "execution_source", "prefix_nan"])
def test_corrupt_contract_or_label_is_rejected_before_output(fresh, corruption):
    root, project = fresh
    episode = root / "episode_000000"
    if corruption in ("label", "scorer", "execution_source"):
        metadata = json.loads((episode / "collection_metadata.json").read_text())
        if corruption in ("scorer", "execution_source"):
            name = "scoring.py" if corruption == "scorer" else "runner.py"
            metadata["source_file_sha256"][name] = "sha256:" + "0" * 64
        else:
            metadata.update(label=1, failure_type=None)
            result = json.loads((episode / "result.json").read_text())
            result.update(success=True, failure_type=None)
            write_json(episode / "result.json", result)
        write_json(episode / "collection_metadata.json", metadata)
        rewrite_index(root)
    elif corruption == "binding":
        (project / "observation_0/depth.png").write_bytes(b"changed source")
    elif corruption == "controller":
        path = episode / "resolved_case.json"
        resolved = json.loads(path.read_text())
        resolved["execution_protocol"]["parameters"]["force_close_delta_rad"] = .3
        write_json(path, resolved)
    else:
        path = episode / "actions.npz"
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        if corruption == "timestamp":
            values["time_s"][70] += .01
        elif corruption == "prefix_nan":
            values["hand_qvel"][70, 0] = np.nan
        else:
            values["joint_names"] = values["joint_names"][::-1]
        np.savez_compressed(path, **values)
    with pytest.raises(ValueError):
        evaluate(fresh)
    assert not (project / "evaluation").exists()


def test_future_and_privileged_mutations_cannot_change_frozen_predictions(fresh):
    before, _ = evaluate(fresh, "before")
    root, _ = fresh
    for episode in root.glob("episode_*"):
        for filename in ("trajectory.npz", "actions.npz"):
            path = episode / filename
            with np.load(path, allow_pickle=False) as data:
                values = {key: data[key].copy() for key in data.files}
            field = "hand_qpos" if filename == "trajectory.npz" else "hand_qvel"
            values[field][74:] = np.nan
            if filename == "actions.npz":
                values["commanded_target_rad"][74:] = np.inf
                values["full_qpos"] = np.full((len(values["time_s"]), 38), np.nan)
                values["requested_noise_rad"] = np.full((len(values["time_s"]), 24), 999.)
            np.savez_compressed(path, **values)
    after, _ = evaluate(fresh, "after")
    with np.load(before / "dataset.npz", allow_pickle=False) as a, \
            np.load(after / "dataset.npz", allow_pickle=False) as b:
        np.testing.assert_array_equal(a["X"], b["X"])
        np.testing.assert_array_equal(a["y"], b["y"])
    assert (before / "predictions.csv").read_bytes() == (after / "predictions.csv").read_bytes()


def test_errors_remain_excluded_and_unlabeled(fresh):
    root, project = fresh
    plan = json.loads((root / "plan.json").read_text())
    planned = dict(plan["episodes"][-1], episode_id="episode_000004")
    plan["episodes"].append(planned)
    write_json(root / "plan.json", plan)
    error = dict(planned, status="runtime_failed", label=None, error_type="FixtureError", error="test fixture")
    episode = root / error["episode_id"]
    episode.mkdir()
    write_json(episode / "collection_metadata.json", error)
    with (root / "index.jsonl").open("a") as handle:
        handle.write(json.dumps(error) + "\n")
    records = [json.loads(line) for line in (root / "index.jsonl").read_text().splitlines()]
    write_summary(root, records)
    output, report = evaluate(fresh)
    assert report["counts"]["excluded"] == 1
    assert report["counts"]["failure"] == 2
    assert json.loads((output / "manifest.json").read_text())["excluded"][0]["episode_id"] == "episode_000004"


def test_changed_saved_weights_and_overwrite_are_rejected(fresh):
    root, project = fresh
    output, _ = evaluate(fresh)
    with pytest.raises(FileExistsError):
        evaluate(fresh)
    copied = project / "changed_models"
    shutil.copytree(MODELS, copied)
    with (copied / "proprioception_and_commands.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="model hash mismatch"):
        evaluate_fresh_collection(root, ORIGINAL, copied, project / "second", project_root=project,
                                  execution_reference=project / "execution_reference.json")


@pytest.mark.parametrize("corruption", ["truncated", "reordered", "unplanned", "plan_field", "summary", "incomplete_summary"])
def test_incomplete_or_inconsistent_frozen_batches_are_rejected(fresh, corruption):
    root, project = fresh
    records = [json.loads(line) for line in (root / "index.jsonl").read_text().splitlines()]
    if corruption == "truncated":
        records = records[:-1]
    elif corruption == "reordered":
        records = records[::-1]
    elif corruption == "unplanned":
        records[-1]["episode_id"] = "episode_000999"
    elif corruption == "plan_field":
        records[0]["noise_seed"] += 1
    else:
        path = root / "summary.json"
        summary = json.loads(path.read_text())
        if corruption == "summary":
            summary["successes"] += 1
        else:
            summary["completed"] -= 1
        write_json(path, summary)
    (root / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="plan|summary"):
        evaluate(fresh)
    assert not (project / "evaluation").exists()


def test_consistent_new_config_cannot_replace_original_execution_reference(fresh, monkeypatch):
    from grasp_failure_prediction.evaluation import registry

    root, project = fresh
    original_resolve = registry.resolve_case

    def changed_registry(case):
        resolved = original_resolve(case)
        parameters = resolved.execution_protocol.parameters.model_copy(update={"force_close_delta_rad": .3})
        protocol = resolved.execution_protocol.model_copy(update={"parameters": parameters})
        return replace(resolved, execution_protocol=protocol,
                       execution_protocol_config_hash=canonical_sha256(protocol))

    monkeypatch.setattr(registry, "resolve_case", changed_registry)
    for path in root.glob("episode_*/resolved_case.json"):
        recorded = json.loads(path.read_text())
        changed = changed_registry(EvaluationCase.model_validate(recorded["case"]))
        recorded["execution_protocol"] = changed.execution_protocol.model_dump(mode="json")
        recorded["execution_protocol_config_hash"] = changed.execution_protocol_config_hash
        write_json(path, recorded)
    with pytest.raises(ValueError, match="original frozen specification"):
        evaluate(fresh)
    assert not (project / "evaluation").exists()


@pytest.mark.parametrize("corruption", ["source_hash", "registry_hash", "reference_spec"])
def test_reference_must_bind_original_source_and_registry(fresh, corruption):
    root, project = fresh
    path = project / "execution_reference.json"
    reference = json.loads(path.read_text())
    if corruption == "source_hash":
        reference["source_resolved_case_sha256"] = "0" * 64
    elif corruption == "registry_hash":
        reference["registry_yaml_sha256"][REGISTRY_FILES[0]] = "0" * 64
    else:
        reference["execution_protocol"]["parameters"]["force_close_delta_rad"] = .3
    write_json(path, reference)
    with pytest.raises(ValueError):
        evaluate(fresh)
    assert not (project / "evaluation").exists()
