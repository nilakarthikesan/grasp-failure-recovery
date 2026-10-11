"""Causal feature/export integrity and label-independent grouped holdouts."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from grasp_failure_prediction.prediction.dataset import (
    LABEL_RULE,
    export_dataset,
    load_dataset,
    main,
    split_observation_groups,
)


def write_json(path, value):
    path.write_text(json.dumps(value))


def read_json(path):
    return json.loads(path.read_text())


def rewrite_npz(path, **overrides):
    with np.load(path, allow_pickle=False) as original:
        arrays = {key: original[key].copy() for key in original.files}
    arrays.update(overrides)
    np.savez_compressed(path, **arrays)


@pytest.fixture
def collection(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    rows = []
    times = np.arange(1, 82) * 0.04
    states = np.full(len(times), "close_fingers", dtype="U16")
    states[:29] = "approach"
    states[74:] = "lift"
    names = np.asarray([f"j{i}" for i in range(24)])
    measured = np.arange(len(times) * 24).reshape(len(times), 24) / 1000
    for scene in range(4):
        for repetition in range(2):
            episode_id = f"episode_{len(rows):06d}"
            episode = root / episode_id
            episode.mkdir()
            label = repetition
            metadata = {
                "schema_version": "noisy_rollout_pilot_v3", "episode_id": episode_id,
                "status": "completed", "label": label,
                "failure_type": None if label else "failed_acquisition",
                "group_id": f"proposal_{scene}",
                "observation_group_id": f"observation_{scene}",
                "geometry_group_id": "cube" if scene < 2 else "apple",
                "geometry_content_hash": "sha256:" + str(scene // 2) * 64,
                "source_file_sha256": {"scoring.py": "sha256:" + "a" * 64},
                "hug_inference_provenance": {"hug_code_commit": "b" * 40,
                    "checkpoint_sha256": "c" * 64, "seed": repetition},
                "noise_amplitude_rad": repetition * 0.6, "noise_seed": 123 + repetition,
                "object_id": "excluded_object_identifier", "object_position_m": [999, 999, 999],
            }
            write_json(episode / "collection_metadata.json", metadata)
            write_json(episode / "resolved_case.json", {
                "case": {"seed": 9100 + scene}, "execution_protocol": {"success": LABEL_RULE},
            })
            write_json(episode / "result.json", {
                "status": "completed", "success": bool(label),
                "failure_type": metadata["failure_type"],
                "resolved_execution": {"code_commit": "d" * 40},
                "outcome": {"future_loss_time_s": 5.0},
            })
            np.savez_compressed(episode / "trajectory.npz", time_s=times, state=states,
                                hand_qpos=measured + scene,
                                object_position_m=np.full((len(times), 3), 999),
                                hand_object_contact=np.ones(len(times), dtype=bool))
            np.savez_compressed(episode / "actions.npz", time_s=times, joint_names=names,
                                hand_qvel=measured + 10, commanded_target_rad=measured + 20,
                                root_target_position_m=np.full((len(times), 3), 999),
                                full_qpos=np.full((len(times), 38), 999),
                                requested_noise_rad=np.full((len(times), 24), 999))
            rows.append(metadata)
    (root / "index.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    write_json(root / "plan.json", {"planned": 8})
    write_json(root / "summary.json", {"completed": 8, "successes": 4, "failures": 4})
    return root


def export(collection, tmp_path, name="derived"):
    output = tmp_path / name
    manifest = export_dataset(collection, output)
    return output, manifest


def test_export_contains_only_named_time_major_prefix_columns(collection, tmp_path):
    output, manifest = export(collection, tmp_path)
    bundle = load_dataset(output)
    assert bundle.X.shape == (8, 1440)
    assert bundle.X.dtype == np.float32
    assert bundle.y.dtype == np.uint8
    assert bundle.y.tolist() == [0, 1] * 4
    assert set(path.name for path in output.iterdir()) == {"dataset.npz", "manifest.json", "splits.json"}
    with np.load(collection / "episode_000000/trajectory.npz", allow_pickle=False) as tr, \
            np.load(collection / "episode_000000/actions.npz", allow_pickle=False) as ac:
        expected = np.concatenate([tr["hand_qpos"][54:74], ac["hand_qvel"][54:74],
                                   ac["commanded_target_rad"][54:74]], axis=1)
    np.testing.assert_array_equal(bundle.X[0], expected.astype(np.float32).ravel())
    assert manifest["input_contract"]["fields"] == ["hand_qpos", "hand_qvel", "commanded_target_rad"]
    columns = manifest["feature_columns"]
    assert len(columns) == 1440
    assert columns[0]["joint_name"] == "j0"
    assert columns[0]["field"] == "hand_qpos"
    assert columns[0]["frame_offset_s"] == pytest.approx(-0.76)
    assert columns[24]["field"] == "hand_qvel"
    assert columns[48]["field"] == "commanded_target_rad"
    assert columns[-1]["frame_offset_s"] == 0
    assert all(row["observed_at_s"] <= 2.96 + 1e-9 for row in manifest["episodes"])
    assert manifest["counts"]["observation_groups"] == 4
    assert manifest["counts"]["geometry_groups"] == 2
    assert manifest["label_scorer_sha256"] == "sha256:" + "a" * 64
    assert manifest["episodes"][0]["raw_files_sha256"]["actions.npz"].startswith("sha256:")
    provenance = manifest["provenance_sources"][manifest["episodes"][0]["source_provenance_id"]]
    assert provenance["hug_inference_provenance"]["checkpoint_sha256"] == "c" * 64


def test_future_and_privileged_mutations_cannot_change_features(collection, tmp_path):
    first, _ = export(collection, tmp_path, "first")
    before = load_dataset(first)
    for episode in collection.glob("episode_*"):
        with np.load(episode / "trajectory.npz", allow_pickle=False) as tr:
            qpos = tr["hand_qpos"].copy()
        qpos[74:] = np.nan
        rewrite_npz(episode / "trajectory.npz", hand_qpos=qpos,
                    object_position_m=np.full((81, 3), -12345),
                    hand_object_contact=np.zeros(81, dtype=bool))
        with np.load(episode / "actions.npz", allow_pickle=False) as ac:
            qvel, commands = ac["hand_qvel"].copy(), ac["commanded_target_rad"].copy()
        qvel[74:], commands[74:] = np.inf, np.nan
        rewrite_npz(episode / "actions.npz", hand_qvel=qvel, commanded_target_rad=commands,
                    root_target_position_m=np.full((81, 3), -999),
                    full_qpos=np.full((81, 38), np.nan),
                    requested_noise_rad=np.full((81, 24), -777))
        metadata = read_json(episode / "collection_metadata.json")
        metadata.update(noise_seed=777, noise_amplitude_rad=123, object_id="changed")
        write_json(episode / "collection_metadata.json", metadata)
    second, _ = export(collection, tmp_path, "second")
    after = load_dataset(second)
    np.testing.assert_array_equal(before.X, after.X)
    np.testing.assert_array_equal(before.y, after.y)
    assert (first / "dataset.npz").read_bytes() == (second / "dataset.npz").read_bytes()
    assert before.splits == after.splits


def test_declared_scene_split_preserves_pairs_and_is_independent_of_labels(collection, tmp_path):
    output, _ = export(collection, tmp_path)
    bundle = load_dataset(output)
    splits = bundle.splits["partitions"]
    assert splits["train"]["episode_ids"] == ["episode_000000", "episode_000001", "episode_000004", "episode_000005"]
    assert splits["validation"]["episode_ids"] == ["episode_000002", "episode_000003"]
    assert splits["test"]["episode_ids"] == ["episode_000006", "episode_000007"]
    for key in ("episode_ids", "observation_group_ids", "group_ids"):
        roles = [set(splits[role][key]) for role in ("train", "validation", "test")]
        assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
    assert splits["train"]["geometry_group_ids"] == ["apple", "cube"]
    rows = deepcopy(bundle.manifest["episodes"])
    baseline = split_observation_groups(rows)
    for row in rows:
        row.update(label=999, failure_type="arbitrary", success=False)
    assert split_observation_groups(rows) == baseline


@pytest.mark.parametrize("key", ["observation_group_id", "group_id"])
def test_split_rejects_shared_source_across_roles(collection, tmp_path, key):
    _, manifest = export(collection, tmp_path)
    rows = deepcopy(manifest["episodes"])
    rows[2][key] = rows[0][key]
    with pytest.raises(ValueError, match="conflicting|crosses"):
        split_observation_groups(rows)


@pytest.mark.parametrize("corruption", ["timestamp", "shape", "prefix_nan", "phase", "joints", "short"])
def test_malformed_observable_prefix_is_rejected_before_output(collection, tmp_path, corruption):
    episode = collection / "episode_000000"
    if corruption == "timestamp":
        with np.load(episode / "actions.npz", allow_pickle=False) as a:
            times = a["time_s"].copy()
        times[70] += 0.001
        rewrite_npz(episode / "actions.npz", time_s=times)
    elif corruption == "shape":
        rewrite_npz(episode / "actions.npz", hand_qvel=np.zeros((81, 23)))
    elif corruption == "prefix_nan":
        with np.load(episode / "trajectory.npz", allow_pickle=False) as a:
            qpos = a["hand_qpos"].copy()
        qpos[70, 0] = np.nan
        rewrite_npz(episode / "trajectory.npz", hand_qpos=qpos)
    elif corruption == "phase":
        with np.load(episode / "trajectory.npz", allow_pickle=False) as a:
            states = a["state"].copy()
        states[70] = "lift"
        rewrite_npz(episode / "trajectory.npz", state=states)
    elif corruption == "joints":
        rewrite_npz(collection / "episode_000002/actions.npz",
                    joint_names=np.asarray([f"j{i}" for i in reversed(range(24))]))
    else:
        times = np.asarray([2.92, 2.96, 3.0])
        np.savez_compressed(episode / "trajectory.npz", time_s=times,
                            state=np.asarray(["close_fingers", "close_fingers", "lift"]),
                            hand_qpos=np.zeros((3, 24)))
        np.savez_compressed(episode / "actions.npz", time_s=times,
                            joint_names=np.asarray([f"j{i}" for i in range(24)]),
                            hand_qvel=np.zeros((3, 24)), commanded_target_rad=np.zeros((3, 24)))
    with pytest.raises(ValueError):
        export_dataset(collection, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("corruption", ["result", "bool_label", "source_schema", "task_rule", "scorer"])
def test_label_or_source_inconsistency_is_rejected(collection, tmp_path, corruption):
    episode = collection / ("episode_000002" if corruption == "scorer" else "episode_000000")
    if corruption == "result":
        result = read_json(episode / "result.json")
        result["success"] = True
        write_json(episode / "result.json", result)
    elif corruption in ("bool_label", "source_schema", "scorer"):
        metadata = read_json(episode / "collection_metadata.json")
        if corruption == "bool_label":
            metadata["label"] = False
        elif corruption == "source_schema":
            metadata["schema_version"] = "old_source"
        else:
            metadata["source_file_sha256"]["scoring.py"] = "sha256:" + "e" * 64
        write_json(episode / "collection_metadata.json", metadata)
    else:
        resolved = read_json(episode / "resolved_case.json")
        resolved["execution_protocol"]["success"]["minimum_lift_m"] = 0.13
        write_json(episode / "resolved_case.json", resolved)
    with pytest.raises(ValueError):
        export_dataset(collection, tmp_path / "bad")


def test_unlabeled_errors_are_accounted_without_loading_as_failures(collection, tmp_path):
    with (collection / "index.jsonl").open("a") as f:
        f.write(json.dumps({"episode_id": "episode_000008", "status": "error", "label": None}) + "\n")
    output, manifest = export(collection, tmp_path)
    assert manifest["counts"]["indexed"] == 9
    assert manifest["counts"]["included"] == 8
    assert manifest["counts"]["excluded"] == 1
    assert manifest["counts"]["failure"] == 4
    assert "episode_000008" not in load_dataset(output).episode_ids
    assert manifest["excluded"] == [{"episode_id": "episode_000008", "reason": "error_or_unlabeled"}]


def test_labeled_error_cannot_become_training_failure(collection, tmp_path):
    with (collection / "index.jsonl").open("a") as f:
        f.write(json.dumps({"episode_id": "episode_000008", "status": "error", "label": 0}) + "\n")
    with pytest.raises(ValueError, match="unlabeled"):
        export_dataset(collection, tmp_path / "bad")


def test_completed_missing_file_is_an_error_and_existing_output_is_preserved(collection, tmp_path):
    output, _ = export(collection, tmp_path)
    original = (output / "dataset.npz").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        export_dataset(collection, output)
    assert (output / "dataset.npz").read_bytes() == original
    (collection / "episode_000000/actions.npz").unlink()
    with pytest.raises(FileNotFoundError):
        export_dataset(collection, tmp_path / "missing")


@pytest.mark.parametrize("file_name", ["dataset.npz", "splits.json"])
def test_loader_checks_payload_and_split_hashes(collection, tmp_path, file_name):
    output, _ = export(collection, tmp_path)
    with (output / file_name).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(output)


def test_loader_rejects_extra_privileged_array_even_with_updated_hash(collection, tmp_path):
    output, _ = export(collection, tmp_path)
    rewrite_npz(output / "dataset.npz", noise_amplitude_rad=np.ones(8))
    manifest = read_json(output / "manifest.json")
    manifest["dataset_sha256"] = "sha256:" + hashlib.sha256((output / "dataset.npz").read_bytes()).hexdigest()
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="unexpected exported arrays"):
        load_dataset(output)


@pytest.mark.parametrize("corruption", ["label_rule", "scorer", "counts", "provenance"])
def test_loader_validates_semantics_beyond_payload_hash(collection, tmp_path, corruption):
    output, _ = export(collection, tmp_path)
    manifest = read_json(output / "manifest.json")
    if corruption == "label_rule":
        manifest["label_rule"]["minimum_lift_m"] = 0.13
    elif corruption == "scorer":
        manifest["label_scorer_sha256"] = "sha256:" + "f" * 64
    elif corruption == "counts":
        manifest["counts"]["success"] += 1
    else:
        manifest["episodes"][0]["source_provenance_id"] = "unknown"
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError):
        load_dataset(output)


def test_loader_rejects_rehashed_label_mutation(collection, tmp_path):
    output, _ = export(collection, tmp_path)
    with np.load(output / "dataset.npz", allow_pickle=False) as data:
        labels = data["y"].copy()
    labels[0] = 1
    rewrite_npz(output / "dataset.npz", y=labels)
    manifest = read_json(output / "manifest.json")
    manifest["dataset_sha256"] = "sha256:" + hashlib.sha256((output / "dataset.npz").read_bytes()).hexdigest()
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="source task labels"):
        load_dataset(output)


def test_cli_uses_frozen_baseline_and_reports_counts(collection, tmp_path, capsys):
    main(["--runs-root", str(collection), "--output", str(tmp_path / "cli")])
    assert json.loads(capsys.readouterr().out)["included"] == 8
    with pytest.raises(ValueError, match="fixes"):
        export_dataset(collection, tmp_path / "other", history_steps=10)
