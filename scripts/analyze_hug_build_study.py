#!/usr/bin/env python3
"""Read-only Phase 3 accounting for the completed frozen HUG study.

No schedules, splits, labels or datasets are rewritten. No predictors are fit,
and all three MuJoCo integration APIs are forbidden throughout analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import mujoco

import run_hug_build_collection as collector


AXES = ("mass_multiplier", "friction_coefficient", "table_offset_m", "approach_mode", "sensor_noise_std_rad")


@contextmanager
def no_physics():
    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2")}
    counts = {"integration_api_attempts": 0}
    def forbidden(*args, **kwargs):
        counts["integration_api_attempts"] += 1
        raise RuntimeError("study analysis forbids physics integration")
    for name in originals:
        setattr(mujoco, name, forbidden)
    try:
        yield counts
    finally:
        for name, value in originals.items():
            setattr(mujoco, name, value)


def coverage(rows):
    return dict(attempts=len(rows), objects=len({r["object_id"] for r in rows}),
                categories=len({r["category_id"] for r in rows}),
                geometries=len({r["geometry_group_id"] for r in rows}),
                captures=len({r["observation_group_id"] for r in rows}),
                proposal_families=len({r["proposal_group_id"] for r in rows}),
                axis_values={key: sorted({row[key] for row in rows}) for key in AXES})


def fold_eligibility(data):
    """Mirror the frozen trainer's eligibility; single-class test ranking is null."""
    result = []
    for fold in data.folds:
        partitions = {}
        reasons = []
        for role in ("train", "validation", "test"):
            labels = data.dataset.y[fold["indices"][role]]
            partitions[role] = dict(included=len(labels), successes=int(labels.sum()),
                                    failures=int(len(labels)-labels.sum()), two_classes=len(np.unique(labels)) == 2)
            if not len(labels):
                reasons.append("empty_"+role)
            elif role != "test" and len(np.unique(labels)) != 2:
                reasons.append("single_class_"+role)
        test = partitions["test"]
        ranking = ("ELIGIBLE_NOT_MEASURED" if test["two_classes"] else
                   "UNDEFINED_SINGLE_CLASS_TEST" if test["included"] else "UNDEFINED_EMPTY_TEST")
        result.append(dict(fold_id=fold["fold_id"], test_category=fold["test_category"],
                           validation_category=fold["validation_category"], partitions=partitions,
                           eligible_for_predeclared_comparison=not reasons, skip_reasons=reasons,
                           test_ranking=dict(status=ranking, failure_roc_auc=None, failure_average_precision=None)))
    return result


def _analyze(root):
    protocol = collector.read_protocol(root)
    generation_path = root / "generation.json"
    summary_path = root / "study_summary.json"
    generation, summary = (json.loads(path.read_text()) for path in (generation_path, summary_path))
    if (summary.get("schema_version") != "hug_build_collection_result_v1"
            or summary.get("condition") != "study" or summary.get("status") != "COMPLETED_BOUNDED_QUOTA"
            or summary.get("error") is not None):
        raise ValueError("require the completed study quota without integrity errors")
    if (generation.get("status") != "COMPLETED_BOUNDED_GENERATION"
            or summary["preparation_sha256"] != collector.sha(root / "preparation.json")
            or summary["generation_sha256"] != collector.sha(generation_path)):
        raise ValueError("study is not bound to this frozen preparation/generation")
    collector.verify({**protocol["source_bindings"], **protocol["input_bindings"], **protocol["asset_bindings"],
                      **generation["artifact_bindings"], **summary["raw_artifact_bindings"],
                      **summary["filters"]["artifact_bindings"]})
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    from grasp_failure_prediction.prediction.recorded_training import load_training_data
    from grasp_failure_prediction.prediction.failure_analysis import summarize_attempts

    schedule_path, folds_path, binding_path = (root / name for name in
        ("study_schedule.json", "study_folds.json", "study_attempt_scenario_binding.json"))
    schedule = json.loads(schedule_path.read_text())
    condition = generation["conditions"]["study"]
    if (schedule != condition["schedule"] or json.loads(folds_path.read_text()) != condition["folds"]
            or json.loads(binding_path.read_text()) != condition["attempt_binding"]
            or summary["schedule_sha256"] != collector.digest(schedule)):
        raise ValueError("study schedule/folds/binding differ from frozen generation")
    rows = summary["attempts"]
    scenarios = {row["attempt_id"]: row for row in schedule["scenarios"]}
    if (len(rows) != protocol["quotas"]["study_attempts"] or len(scenarios) != len(rows)
            or len({row["attempt_id"] for row in rows}) != len(rows)
            or {row["attempt_id"] for row in rows} != set(scenarios)
            or summary["quota"] != len(rows)):
        raise ValueError("study ledger does not retain every frozen attempt exactly once")
    if dict(Counter(row["status"] for row in rows)) != summary["counts"]:
        raise ValueError("study status counts differ from its rows")
    if sum(row["physics_steps"] for row in rows) != summary["physics_steps"]:
        raise ValueError("study physics counter differs from its rows")
    raw_dirs, metadata, raw_bindings, realized = [], {}, {}, []
    for row in rows:
        directory = Path(row["recording_path"])
        actual, payload, _ = load_attempt(directory)
        if actual.attempt_id != row["attempt_id"] or collector.sha(directory/"attempt.json") != row["attempt_sha256"]:
            raise ValueError("study raw attempt identity/hash differs from ledger")
        scenario = scenarios[actual.attempt_id]
        if (row["scenario_id"] != scenario["scenario_id"] or actual.provenance_hashes.get("scenario") != collector.digest(scenario)
                or (actual.object_id, actual.geometry_group_id, actual.observation_group_id, actual.proposal_group_id)
                != tuple(scenario[key] for key in ("object_id", "geometry_group_id", "observation_group_id", "proposal_group_id"))):
            raise ValueError("raw attempt does not bind its exact frozen scenario/groups")
        outcome = None if actual.outcome is None else actual.outcome.model_dump(mode="json")
        if outcome != row["outcome"] or (outcome is not None) != (row["status"] == "COMPLETED"):
            raise ValueError("raw physical/null outcome differs from study ledger")
        if outcome is None and (actual.execution_status != "not_executed" or row["physics_steps"] != 0):
            raise ValueError("completed-quota null attempt must remain unexecuted and unlabeled")
        if outcome is not None:
            if row["physics_steps"] != protocol["quotas"]["maximum_task_steps_per_attempt"]:
                raise ValueError("completed task differs from declared exact step quota")
            realization_path = directory.parent/"scenario_realization.json"
            expected = actual.provenance_hashes.get("scenario_realization")
            if expected is None or collector.sha(realization_path) != expected or row.get("scenario_realization_sha256") != expected:
                raise ValueError("completed task lacks its hash-bound scenario realization")
            if json.loads(realization_path.read_text())["scenario"] != scenario:
                raise ValueError("runtime scenario realization differs from frozen scenario")
            realized.append(scenario)
        raw_dirs.append(directory); metadata[actual.attempt_id] = actual
        raw_bindings[actual.attempt_id] = dict(attempt_sha256=row["attempt_sha256"], artifacts_sha256=payload["artifacts_sha256"])
    taxonomy = summarize_attempts(raw_dirs)
    data = load_training_data(root/"study_prefixes", schedule_path, folds_path, binding_path)
    if data.dataset.manifest != summary["dataset"]:
        raise ValueError("strict exported dataset differs from study summary")
    for row in data.dataset.manifest["episodes"] + data.dataset.manifest["excluded"]:
        actual = metadata[row["attempt_id"]]
        if (row["attempt_sha256"] != raw_bindings[actual.attempt_id]["attempt_sha256"]
                or row["mapping_status"] != actual.mapping.status.value or row["execution_status"] != actual.execution_status):
            raise ValueError("dataset attempt denominator differs from raw recording")
        if "source_task_label" in row and row["source_task_label"] != int(actual.outcome.success):
            raise ValueError("exported task label differs from raw physical label")
    included = data.dataset.manifest["episodes"]
    category_counts = {}
    for category in sorted({row["category_id"] for row in scenarios.values()}):
        index = [i for i, row in enumerate(included) if scenarios[row["attempt_id"]]["category_id"] == category]
        y = data.dataset.y[index]
        category_counts[category] = dict(scheduled=sum(r["category_id"] == category for r in scenarios.values()),
                                         included=len(y), successes=int(y.sum()), failures=int(len(y)-y.sum()),
                                         two_classes=len(np.unique(y)) == 2)
    source_paths = (Path(__file__), Path(summarize_attempts.__code__.co_filename), Path(load_training_data.__code__.co_filename))
    inputs = ("preparation.json", "generation.json", "study_summary.json", "study_schedule.json", "study_folds.json",
              "study_attempt_scenario_binding.json", "study_prefixes/manifest.json", "study_prefixes/dataset.npz")
    return dict(schema_version="hug_build_study_analysis_v1", status="COMPLETED_READ_ONLY_ANALYSIS",
                analyzed_utc=datetime.now(timezone.utc).isoformat(),
                inspected_files_sha256={name: collector.sha(root/name) for name in inputs},
                analyzer_sources_sha256={str(path.resolve()): collector.sha(path) for path in source_paths},
                strict_training_adapter_bindings=data.source_bindings, raw_attempt_bindings=raw_bindings,
                failure_taxonomy=taxonomy, collection_status_counts=summary["counts"], task_physics_steps=summary["physics_steps"],
                filter_status_counts=dict(Counter(row["filter_status"] for row in summary["filters"]["proposals"].values())),
                coverage=dict(scheduled=coverage(list(scenarios.values())), executed=coverage(realized),
                              included_prefixes=coverage([scenarios[row["attempt_id"]] for row in included])),
                coverage_scope="Scheduled identities and hash-verified runtime scenario assignments; variants/seeds are not independent geometries or captures",
                per_category_included_classes=category_counts, frozen_fold_eligibility=fold_eligibility(data),
                dataset_counts=data.dataset.manifest["counts"],
                schedule_changed=False, folds_changed=False, model_fits=0, inference_calls=0,
                scope="Cause-based reporting and frozen-fold eligibility only; no model metrics, slip onset or generalization acceptance")


def analyze(collection_output, output):
    root, output = Path(collection_output).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("study analysis requires a fresh output file")
    with no_physics() as counts:
        report = _analyze(root)
    if counts["integration_api_attempts"]:
        raise RuntimeError("study analysis attempted forbidden integration")
    report["read_only_guard"] = counts
    collector.write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.collection_output, args.output)
    print(json.dumps(dict(status=report["status"], dataset_counts=report["dataset_counts"],
                          eligible_folds=sum(row["eligible_for_predeclared_comparison"] for row in report["frozen_fold_eligibility"]))))


if __name__ == "__main__":
    main()
