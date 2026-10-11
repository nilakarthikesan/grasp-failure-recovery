"""Evaluate genuinely new recorded scenes with unchanged task-outcome models.

No optimizer, calibration or threshold selection is invoked. Source diagnostics
are used to audit labels and identities, never as predictor columns.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import platform
import re
from types import SimpleNamespace

import numpy as np
import scipy

from .dataset import (LABEL_RULE, _canonical_sha256, _json, _prefix, _sha256, _write_json,
                      load_dataset)
from .model import MODEL_SCHEMA, load_model, outcome_metrics, summarize_prefix


MODEL_NAMES = ("training_prevalence", "proprioception_and_commands", "commands_only")
PROVENANCE_KEYS = (
    "geometry_content_hash", "geometry_asset_files_sha256", "proposal_sha256",
    "camera_sha256", "observation_files_sha256", "observation_manifest_sha256",
    "scene_contract_sha256", "scene_contract_status", "inference_report_sha256",
    "hug_inference_provenance", "source_file_sha256", "runtime_versions", "observation_source",
)
RAW_FILES = ("actions.npz", "trajectory.npz", "collection_metadata.json",
             "result.json", "resolved_case.json")
FROZEN_EXECUTION_FILES = ("runner.py", "noisy_rollouts.py", "retargeting.py", "registry.py",
                          "case_runner.py", "object_assets.py", "scene_contracts.py", "scoring.py")
REGISTRY_FILES = ("src/grasp_failure_prediction/evaluation/configs/environments.yaml",
                  "src/grasp_failure_prediction/evaluation/configs/execution_protocols.yaml")


def _hex_sha256(path: Path) -> str:
    return _sha256(path).removeprefix("sha256:")


def _freeze(original_dir: Path, models_dir: Path):
    original = load_dataset(original_dir)
    report = _json(models_dir / "report.json")
    if (report.get("schema_version") != "task_outcome_baseline_report_v1"
            or report.get("model_schema") != MODEL_SCHEMA
            or report.get("input_contract") != original.manifest["input_contract"]
            or report.get("summary_order") != ["last", "mean", "population_std", "last_minus_first"]
            or set(report.get("models", {})) != set(MODEL_NAMES)):
        raise ValueError("saved model report differs from the frozen input/model contract")
    snapshot = {}
    for filename, key in (("dataset.npz", "input_dataset_sha256"),
                          ("manifest.json", "input_manifest_sha256"),
                          ("splits.json", "input_splits_sha256")):
        digest = _hex_sha256(original_dir / filename)
        if digest != report.get(key):
            raise ValueError("original dataset differs from the saved model report")
        snapshot["original_" + filename] = digest
    implementation = Path(__file__).with_name("model.py")
    if report.get("training_source_sha256", {}).get("model.py") != _hex_sha256(implementation):
        raise ValueError("model implementation differs from the frozen training source")
    snapshot["model_implementation_sha256"] = _hex_sha256(implementation)
    snapshot["model_report_sha256"] = _hex_sha256(models_dir / "report.json")
    lookup = {str(episode): index for index, episode in enumerate(original.episode_ids)}
    train = [lookup[episode] for episode in original.splits["partitions"]["train"]["episode_ids"]]
    prevalence = report["models"]["training_prevalence"].get("p_success")
    if (isinstance(prevalence, bool) or not isinstance(prevalence, (int, float))
            or not np.isfinite(prevalence) or prevalence != float(original.y[train].mean())):
        raise ValueError("saved prevalence differs from original training labels")
    models = {}
    for name, width in (("proprioception_and_commands", 288), ("commands_only", 96)):
        info = report["models"][name]
        filename = info.get("model_file")
        if filename != name + ".npz":
            raise ValueError("unexpected saved model filename")
        path = models_dir / filename
        digest = _hex_sha256(path)
        if digest != info.get("model_sha256"):
            raise ValueError("saved model hash mismatch")
        model = load_model(path)
        if len(model.normalizer.mean) != width or not model.converged:
            raise ValueError("saved model dimensions/convergence differ from frozen baseline")
        models[name] = model
        snapshot[filename] = digest
    return original, report, models, snapshot


def _source_identity(metadata: dict) -> dict:
    """Recreate content identities used by the collector, ignoring human IDs."""
    identity = {key: metadata[key] for key in (
        "observation_files_sha256", "proposal_sha256", "camera_sha256",
        "geometry_content_hash", "object_position_m", "object_orientation_xyzw",
        "observation_manifest_sha256",
    )}
    observation = {key: value for key, value in identity.items() if key != "proposal_sha256"}
    expected = {
        "group_id": _canonical_sha256(identity),
        "observation_group_id": _canonical_sha256(observation),
        "geometry_group_id": _canonical_sha256({"geometry_content_hash": metadata["geometry_content_hash"]}),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("recorded grouping identities do not match their source content")
    return expected


def _execution_reference(path: Path, original) -> dict:
    reference = _json(path)
    if (reference.get("schema_version") != "frozen_execution_reference_v1"
            or reference.get("embodiment_id") != "shadow_hand_right"
            or reference.get("execution_protocol", {}).get("success") != LABEL_RULE):
        raise ValueError("execution reference must preserve the original robot and task labels")
    original_resolved_hashes = {row["raw_files_sha256"]["resolved_case.json"].removeprefix("sha256:")
                                for row in original.manifest["episodes"]}
    if reference.get("source_resolved_case_sha256") not in original_resolved_hashes:
        raise ValueError("execution reference is not bound to an original resolved case")
    for spec, hash_key in (("environment", "environment_config_hash"),
                           ("execution_protocol", "execution_protocol_config_hash")):
        if _canonical_sha256(reference[spec]) != reference.get(hash_key):
            raise ValueError("execution reference specification/hash mismatch")
    hashes = reference.get("registry_yaml_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(REGISTRY_FILES):
        raise ValueError("execution reference must bind both original registry YAML files")
    source_root = Path(__file__).resolve().parents[3]
    if any(_hex_sha256(source_root / name) != digest for name, digest in hashes.items()):
        raise ValueError("registry YAML differs from the frozen execution reference")
    return reference


def _validate_collection(indexed: list, plan: dict, summary: dict) -> None:
    planned = plan.get("episodes")
    if (plan.get("schema_version") != "noisy_rollout_plan_v1"
            or not isinstance(planned, list) or not planned
            or not all(isinstance(row, dict) for row in planned)):
        raise ValueError("collection requires a nonempty frozen episode plan")
    planned_ids = [row.get("episode_id") for row in planned]
    if (any(not isinstance(episode, str) or not re.fullmatch(r"episode_[0-9]{6}", episode)
            for episode in planned_ids) or len(set(planned_ids)) != len(planned_ids)):
        raise ValueError("frozen plan episode identities must be safe and unique")
    if [row.get("episode_id") for row in indexed] != planned_ids:
        raise ValueError("collection index must completely match frozen plan order and membership")
    for expected, recorded in zip(planned, indexed, strict=True):
        if any(key not in recorded or recorded[key] != value for key, value in expected.items()):
            raise ValueError("indexed attempt differs from its frozen plan")
    completed, successes, failures, errors = 0, 0, 0, 0
    for entry in indexed:
        if entry.get("status") == "completed":
            label = entry.get("label")
            if type(label) is not int or label not in (0, 1):
                raise ValueError("completed attempts require integer labels")
            completed += 1
            successes += label
            failures += 1 - label
        elif entry.get("status") in ("validation_failed", "runtime_failed") and entry.get("label") is None:
            errors += 1
        else:
            raise ValueError("only completed labeled attempts or explicit unlabeled errors are valid")
    expected_summary = {"planned": len(planned), "completed": completed, "successes": successes,
                        "failures": failures, "errors": errors}
    if (set(summary) != set(expected_summary)
            or any(type(value) is not int or value < 0 for value in summary.values())
            or summary != expected_summary):
        raise ValueError("collection summary does not match complete frozen plan/index counts")


def _verify_binding(metadata: dict, resolved: dict, project_root: Path, reference: dict) -> None:
    from ..evaluation.case_runner import (_resolve_object_geometry, _source_scene_binding)
    from ..evaluation.registry import resolve_case
    from ..evaluation.schema import EvaluationCase

    case = EvaluationCase.model_validate(resolved["case"])
    if metadata.get("object_id") != case.object.id:
        raise ValueError("recorded object identity differs from the resolved case")
    actual = resolve_case(case)
    for key in ("environment", "execution_protocol", "environment_config_hash", "execution_protocol_config_hash"):
        if resolved.get(key) != reference[key]:
            raise ValueError("fresh execution differs from the original frozen specification: " + key)
    if (actual.environment.model_dump(mode="json") != resolved.get("environment")
            or actual.execution_protocol.model_dump(mode="json") != resolved.get("execution_protocol")
            or actual.environment_config_hash != resolved.get("environment_config_hash")
            or actual.execution_protocol_config_hash != resolved.get("execution_protocol_config_hash")):
        raise ValueError("recorded execution differs from the versioned environment/protocol")
    if (case.embodiment.id != "shadow_hand_right"
            or actual.environment.environment_id != "adroit_shadow_tabletop_v2"
            or actual.execution_protocol.execution_protocol_id != "fixed_grasp_lift_v2"
            or actual.environment.simulator.control_timestep_s != 0.04):
        raise ValueError("fresh execution must retain the original robot/controller/timing")
    if str(case.grasp.prediction_path) != metadata.get("proposal_path"):
        raise ValueError("resolved proposal reference differs from collection metadata")
    proposal = project_root / case.grasp.prediction_path
    geometry = _resolve_object_geometry(case, project_root)
    binding = _source_scene_binding(case, actual, project_root, proposal, geometry)
    if binding["scene_contract_status"] != "verified":
        raise ValueError("fresh evaluation requires verified capture and inference binding")
    for key in ("camera_sha256", "observation_files_sha256", "scene_contract_sha256",
                "scene_contract_status", "observation_manifest_sha256", "observation_source",
                "inference_report_sha256", "hug_inference_provenance"):
        if metadata.get(key) != binding[key]:
            raise ValueError("recorded source binding differs from actual captured inputs: " + key)
    contract = binding["scene_contract"]
    for key in ("geometry_content_hash", "object_position_m", "object_orientation_xyzw"):
        if metadata.get(key) != contract[key]:
            raise ValueError("recorded scene content differs from verified contract: " + key)
    if metadata.get("proposal_sha256") != _sha256(proposal):
        raise ValueError("proposal hash mismatch")
    expected_assets = geometry.files_sha256 if geometry is not None else {}
    if metadata.get("geometry_asset_files_sha256") != expected_assets:
        raise ValueError("geometry asset binding mismatch")


def _rescore(episode: Path, resolved: dict, label: int, failure_type) -> None:
    from ..evaluation.runner import ExecutionState
    from ..evaluation.scoring import score_trace

    with np.load(episode / "trajectory.npz", allow_pickle=False) as data:
        times, states = data["time_s"], data["state"]
        positions = data["object_position_m"]
        contacts, table = data["hand_object_contact"], data["hand_table_contact"]
        if (positions.shape != (len(times), 3) or not np.isfinite(positions).all()
                or contacts.shape != times.shape or table.shape != times.shape
                or contacts.dtype != np.bool_ or table.dtype != np.bool_):
            raise ValueError("label supervision arrays must be finite and aligned")
        steps = tuple(SimpleNamespace(time_s=float(time), state=ExecutionState(str(state)),
                                      object_position_m=position, hand_object_contact=bool(contact),
                                      hand_table_contact=bool(collision))
                      for time, state, position, contact, collision in
                      zip(times, states, positions, contacts, table, strict=True))
    protocol = resolved["execution_protocol"]
    spec = SimpleNamespace(success=SimpleNamespace(**protocol["success"]),
                           parameters=SimpleNamespace(**protocol["parameters"]))
    outcome = score_trace(SimpleNamespace(steps=steps), spec, control_timestep_s=0.04)
    category = outcome.failure_type.value if outcome.failure_type is not None else None
    if int(outcome.success) != label or category != failure_type:
        raise ValueError("source label/category differs from regenerated frozen task score")


def evaluate_fresh_collection(runs_root: str | Path, original_dataset: str | Path,
                              models_dir: str | Path, output_dir: str | Path, *,
                              project_root: str | Path, execution_reference: str | Path) -> dict:
    """Export and score new trajectories without refitting any saved parameter."""
    root, original_dir, models_dir, output, project = map(
        lambda path: Path(path).resolve(),
        (runs_root, original_dataset, models_dir, output_dir, project_root))
    if output.exists():
        raise FileExistsError("fresh evaluation output must be a new directory")
    if output == root or root in output.parents:
        raise ValueError("derived evaluation must be separate from the raw collection")
    original, model_report, models, snapshot = _freeze(original_dir, models_dir)
    reference_path = Path(execution_reference).resolve()
    reference = _execution_reference(reference_path, original)
    snapshot["execution_reference_sha256"] = _hex_sha256(reference_path)
    scorer = original.manifest["label_scorer_sha256"]
    scorer_file = Path(__file__).parents[1] / "evaluation/scoring.py"
    if _sha256(scorer_file) != scorer:
        raise ValueError("current label scorer differs from the original collection")
    original_provenance = list(original.manifest["provenance_sources"].values())
    execution_hashes = {}
    execution_dir = scorer_file.parent
    for filename in FROZEN_EXECUTION_FILES:
        recorded = {source.get("source_file_sha256", {}).get(filename) for source in original_provenance}
        if len(recorded) != 1 or None in recorded:
            raise ValueError("original collection must bind one frozen execution source: " + filename)
        expected = next(iter(recorded))
        if _sha256(execution_dir / filename) != expected:
            raise ValueError("current execution implementation differs from original: " + filename)
        execution_hashes[filename] = expected
    snapshot["execution_source_files_sha256"] = execution_hashes
    source_hashes = {name: _sha256(root / name) for name in ("index.jsonl", "plan.json", "summary.json")}
    old_observations = set(original.observation_group_ids.tolist())
    old_proposals = set(original.group_ids.tolist())
    old_observation_bytes = {_canonical_sha256(p["observation_files_sha256"]) for p in original_provenance}
    old_proposal_bytes = {p["proposal_sha256"] for p in original_provenance}
    old_geometry = set(original.geometry_group_ids.tolist())
    indexed = [json.loads(line) for line in (root / "index.jsonl").read_text().splitlines()]
    if not indexed or not all(isinstance(row, dict) for row in indexed):
        raise ValueError("collection index must contain episode objects")
    plan, summary = _json(root / "plan.json"), _json(root / "summary.json")
    _validate_collection(indexed, plan, summary)
    if any(plan.get("source_file_sha256", {}).get(name) != digest
           for name, digest in execution_hashes.items()):
        raise ValueError("frozen plan source differs from the original controller")
    rows, features, labels, exclusions, provenance = [], [], [], [], {}
    seen, binding_cache, observation_sources, proposal_sources = set(), set(), {}, {}
    for entry in indexed:
        episode_id = entry.get("episode_id")
        if (not isinstance(episode_id, str) or not re.fullmatch(r"episode_[0-9]{6}", episode_id)
                or episode_id in seen):
            raise ValueError("unsafe or duplicate indexed episode identity")
        seen.add(episode_id)
        if entry.get("status") != "completed" or entry.get("label") is None:
            if entry.get("label") is not None:
                raise ValueError("errors must be unlabeled, not failures")
            error_metadata = root / episode_id / "collection_metadata.json"
            if _json(error_metadata) != entry:
                raise ValueError("error index and persisted metadata must agree")
            exclusions.append({"episode_id": episode_id, "reason": entry["status"],
                               "raw_metadata_sha256": _sha256(error_metadata)})
            continue
        episode = root / episode_id
        metadata, result, resolved = (_json(episode / name) for name in
                                      ("collection_metadata.json", "result.json", "resolved_case.json"))
        label = metadata.get("label")
        if metadata != entry:
            raise ValueError("completed index and persisted metadata must agree")
        if (metadata.get("schema_version") != "noisy_rollout_pilot_v3"
                or metadata.get("episode_id") != episode_id or metadata.get("status") != "completed"
                or type(label) is not int or label not in (0, 1)
                or type(entry.get("label")) is not int or entry["label"] != label
                or result.get("status") != "completed" or type(result.get("success")) is not bool
                or int(result["success"]) != label
                or result.get("failure_type") != metadata.get("failure_type")):
            raise ValueError("index, metadata and result labels/status must agree")
        if (metadata.get("source_file_sha256", {}).get("scoring.py") != scorer
                or resolved["execution_protocol"]["success"] != LABEL_RULE):
            raise ValueError("label scorer/thresholds differ from the frozen task rule")
        if any(metadata.get("source_file_sha256", {}).get(name) != digest
               for name, digest in execution_hashes.items()):
            raise ValueError("recorded execution source differs from the frozen original controller")
        identity = _source_identity(metadata)
        if any(entry.get(key) != value for key, value in identity.items()):
            raise ValueError("index/source grouping identities differ")
        obs, proposal, geometry = (identity[key] for key in
                                  ("observation_group_id", "group_id", "geometry_group_id"))
        if (obs in old_observations or proposal in old_proposals
                or _canonical_sha256(metadata["observation_files_sha256"]) in old_observation_bytes
                or metadata["proposal_sha256"] in old_proposal_bytes):
            raise ValueError("fresh evaluation overlaps original observation/proposal source content")
        scene_seed = resolved["case"]["seed"]
        if type(scene_seed) is not int:
            raise ValueError("scene seed must be an integer")
        if observation_sources.setdefault(obs, (scene_seed, geometry)) != (scene_seed, geometry):
            raise ValueError("one observation group has conflicting scene/geometry sources")
        if proposal_sources.setdefault(proposal, obs) != obs:
            raise ValueError("one proposal group crosses observations")
        source = {key: metadata.get(key) for key in PROVENANCE_KEYS}
        commit = result.get("resolved_execution", {}).get("code_commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ValueError("invalid recorded source commit")
        source["code_commit"] = commit
        source_id = _canonical_sha256(source)
        binding_key = (source_id, _canonical_sha256(resolved))
        if binding_key not in binding_cache:
            _verify_binding(metadata, resolved, project, reference)
            binding_cache.add(binding_key)
        prefix, names, observed_at = _prefix(episode)
        if names != model_report["input_contract"]["joint_names"]:
            raise ValueError("fresh joint ordering differs from the frozen model")
        _rescore(episode, resolved, label, metadata.get("failure_type"))
        provenance[source_id] = source
        rows.append(dict(identity, episode_id=episode_id, scene_seed=scene_seed,
                         observed_at_s=observed_at, source_task_label=label,
                         source_provenance_id=source_id, object_id=metadata.get("object_id"),
                         geometry_novelty="new_geometry" if geometry not in old_geometry else "new_scene_seen_geometry",
                         failure_type=metadata.get("failure_type"),
                         noise_amplitude_rad=metadata.get("noise_amplitude_rad"),
                         raw_files_sha256={name: _sha256(episode / name) for name in RAW_FILES}))
        features.append(prefix)
        labels.append(label)
    if not rows:
        raise ValueError("no completed labeled episodes to evaluate")
    X, y = np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.uint8)
    probabilities = {"training_prevalence": np.full(len(rows), model_report["models"]["training_prevalence"]["p_success"])}
    for name, model in models.items():
        probabilities[name] = model.predict_success(summarize_prefix(X, commands_only=name == "commands_only"))
    groups = defaultdict(list)
    for row_index, row in enumerate(rows):
        for dimension in ("object_id", "geometry_novelty", "geometry_group_id", "observation_group_id", "scene_seed", "noise_amplitude_rad"):
            groups[(dimension, str(row[dimension]))].append(row_index)
    report = {
        "schema_version": "frozen_fresh_task_outcome_evaluation_v1",
        "scope": "new source observations/proposals, unchanged weights/scaling and fixed 0.5 failure alert",
        "frozen_artifacts": snapshot,
        "execution_reference": reference,
        "prediction_time_s": 2.96, "history_steps": 20,
        "input_contract": model_report["input_contract"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "counts": {"indexed": len(indexed), "included": len(rows), "excluded": len(exclusions),
                   "success": int(y.sum()), "failure": int(len(y) - y.sum()),
                   "observation_groups": len(observation_sources), "proposal_groups": len(proposal_sources),
                   "geometry_groups": len({row["geometry_group_id"] for row in rows})},
        "models": {name: {"metrics": outcome_metrics(y, probability)} for name, probability in probabilities.items()},
        "group_metrics": [dict(dimension=dimension, value=value, model=name,
                               metrics=outcome_metrics(y[indices], probability[indices]))
                          for (dimension, value), indices in sorted(groups.items())
                          for name, probability in probabilities.items()],
        "limitations": ["paired attempts share observation and proposal families; counts are not independent population samples",
                        "simulated final task outcomes, not physical slip onset, warning lead time or real robot reliability",
                        "this evaluation is not available for training, threshold selection or calibration"],
    }
    output.mkdir(parents=True)
    np.savez_compressed(output / "dataset.npz", X=X, y=y,
                        **{key: np.asarray([row[field] for row in rows]) for key, field in (
                            ("episode_ids", "episode_id"), ("group_ids", "group_id"),
                            ("observation_group_ids", "observation_group_id"), ("geometry_group_ids", "geometry_group_id"))})
    manifest = {"schema_version": "fresh_task_prefix_dataset_v1", "partition": "fresh_evaluation_only",
                "source_collection_name": root.name, "frozen_artifacts": snapshot,
                "execution_reference": reference,
                "label_rule": original.manifest["label_rule"], "label_scorer_sha256": scorer,
                "input_contract": model_report["input_contract"],
                "feature_columns": original.manifest["feature_columns"], "episodes": rows,
                "provenance_sources": provenance, "excluded": exclusions, "counts": report["counts"],
                "source_files_sha256": source_hashes,
                "dataset_sha256": _sha256(output / "dataset.npz"), "evaluator_sha256": _sha256(Path(__file__))}
    _write_json(output / "manifest.json", manifest)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["episode_id", "partition", "label_success", "model", "p_success", "p_failure", "failure_alert"])
        for name, probability in probabilities.items():
            for row, label, value in zip(rows, y, probability, strict=True):
                p = float(value)
                writer.writerow([row["episode_id"], "fresh_evaluation", int(label), name,
                                 format(p, ".17g"), format(1 - p, ".17g"), int(1 - p >= 0.5)])
    report.update(dataset_sha256=_sha256(output / "dataset.npz"),
                  manifest_sha256=_sha256(output / "manifest.json"),
                  predictions_sha256=_sha256(output / "predictions.csv"))
    _write_json(output / "report.json", report)
    rechecked_snapshot = _freeze(original_dir, models_dir)[3]
    rechecked_snapshot["execution_reference_sha256"] = _hex_sha256(reference_path)
    rechecked_snapshot["execution_source_files_sha256"] = {
        name: _sha256(execution_dir / name) for name in execution_hashes}
    if (rechecked_snapshot != snapshot
            or source_hashes != {name: _sha256(root / name) for name in source_hashes}):
        raise RuntimeError("frozen artifacts changed during evaluation")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--original-dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--execution-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_fresh_collection(args.runs_root, args.original_dataset, args.models,
                                      args.output, project_root=args.project_root,
                                      execution_reference=args.execution_reference)
    print(json.dumps({"counts": report["counts"], "models": report["models"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
