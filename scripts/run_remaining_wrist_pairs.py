"""Execute the seven remaining frozen A/B pairs once, with no noise or model use.

The ordered inventory and source hashes are saved before any physics step.
The earlier first pair and every historical artifact remain read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from grasp_failure_prediction.evaluation.paired_wrist_execution import (
    execute_frozen_wrist_case, load_frozen_manifest,
)


EXPECTED_POSE_SHA256 = "354f245ba6663974c92a132895709ea8d41dd5578942dc7af8e526f75dbb8ee1"
MODEL_SHA256 = "16c9941a51130daea818c6ad227ef1e3a02e4be9ed1c7aa5a8e8f3e023428beb"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(project_root: Path, manifest_path: Path, first_bundle: Path, output: Path):
    repository = Path(__file__).resolve().parents[1]
    if output.exists():
        raise FileExistsError("Choose a new batch directory; do not repeat or overwrite attempts")
    poses = load_frozen_manifest(manifest_path, EXPECTED_POSE_SHA256,
                                repository_root=repository, project_root=project_root)
    first_manifest = json.loads((first_bundle / "manifest.json").read_text())
    for relative, digest in first_manifest["files_sha256"].items():
        if sha(first_bundle / relative) != digest:
            raise ValueError(f"First-pair package changed: {relative}")
    first_report = json.loads((first_bundle / "report.json").read_text())
    if (first_report["pair_id"] != poses["first_pair_id"]
            or first_report["completed_scored"] != 2
            or first_report["pose_manifest_sha256"] != EXPECTED_POSE_SHA256):
        raise ValueError("First-pair reference differs from the completed diagnostic")
    cases = [case for case in poses["cases"] if case["pair_id"] != poses["first_pair_id"]]
    if len(cases) != 7 or not all(case["eligible"] for case in cases):
        raise ValueError("Expected exactly the seven remaining eligible declared pairs")
    frozen_names = [
        "research_artifacts/task_outcome_baseline_v1/proprioception_and_commands.npz",
        "research_artifacts/task_outcome_baseline_v1/report.json",
        "src/grasp_failure_prediction/prediction/model.py",
        "src/grasp_failure_prediction/prediction/dataset.py",
        "research_artifacts/object_scene_pilot_v1/dataset.npz",
        "research_artifacts/object_scene_pilot_v1/splits.json",
    ]
    execution_names = [
        "scripts/run_remaining_wrist_pairs.py",
        "scripts/select_wrist_comparison_poses.py",
        "src/grasp_failure_prediction/evaluation/paired_wrist_execution.py",
        "src/grasp_failure_prediction/evaluation/table_safe_execution.py",
        "src/grasp_failure_prediction/evaluation/table_clearance.py",
        "src/grasp_failure_prediction/evaluation/noisy_rollouts.py",
        "src/grasp_failure_prediction/evaluation/runner.py",
    ]
    frozen = {name: sha(repository / name) for name in frozen_names}
    if frozen[frozen_names[0]] != MODEL_SHA256:
        raise ValueError("Original predictor file changed")
    execution = {name: sha(repository / name) for name in execution_names}
    plan = dict(
        schema_version="remaining_frozen_wrist_pairs_execution_plan_v1",
        declared_at_utc=datetime.now(timezone.utc).isoformat(),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        pair_ids=[case["pair_id"] for case in cases], postures=["A", "B"],
        planned_pairs=7, planned_new_attempts=14, pose_manifest_sha256=EXPECTED_POSE_SHA256,
        first_report_sha256=sha(first_bundle / "report.json"),
        first_bundle_manifest_sha256=sha(first_bundle / "manifest.json"),
        first_pair_id=poses["first_pair_id"],
        model_contract=poses["model_contract"],
        prediction_time_s=2.96, history_steps=20, failure_alert_threshold=0.5,
        noise_amplitude_rad=0.0, noise_seed=9000,
        execution_protocol_id="fixed_grasp_lift_v4_frozen_wrist_pair",
        frozen_files_sha256=frozen, paired_source_sha256=execution,
        selection="All seven remaining frozen pairs in existing order, once per posture; no replacements based on scores or outcomes.",
        scope="Development wrist sensitivity between two unfamiliar settings per grasp. The first pair is a read-only reference, not re-executed.",
        outcome_rule="Completed grasp/lift/hold labels unchanged. Guard or setup stops remain null-labeled and unscored.",
        analysis_freeze="Final scoring and independent audit sources are bound separately before analysis, after code is finished.",
    )
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    parent_sha = sha(output / "plan.json")
    for index, case in enumerate(cases):
        pair_root = output / case["pair_id"]
        pair_root.mkdir()
        write(pair_root / "plan.json", dict(plan, pair_id=case["pair_id"],
              pair_index=index, case_path=case["case_path"], parent_plan_sha256=parent_sha,
              schema_version="remaining_frozen_wrist_pair_execution_plan_v1"))
    print(f"Frozen seven-pair inventory, plan SHA256 {parent_sha}", flush=True)
    status_rows = []
    for case in cases:
        for posture in ("A", "B"):
            print(f"Executing {case['pair_id']} wrist {posture}, noise 0", flush=True)
            record = execute_frozen_wrist_case(
                case["case_path"], output / case["pair_id"] / posture,
                project_root=project_root, manifest_path=manifest_path,
                expected_manifest_sha256=EXPECTED_POSE_SHA256,
                pair_id=case["pair_id"], posture=posture,
                noise_amplitude_rad=0.0, noise_seed=9000, repository_root=repository)
            status = {key: record.get(key) for key in (
                "pair_id", "posture", "status", "label", "failure_type", "runtime_guard_event")}
            status_rows.append(status)
            print(json.dumps(status), flush=True)
    for relative, digest in (frozen | execution).items():
        if sha(repository / relative) != digest:
            raise ValueError(f"Experiment source changed during execution: {relative}")
    for relative, digest in first_manifest["files_sha256"].items():
        if sha(first_bundle / relative) != digest:
            raise ValueError(f"Historical first-pair package changed: {relative}")
    write(output / "execution_summary.json", dict(
        planned_pairs=7, planned_attempts=14, attempts=status_rows,
        frozen_files_unchanged=True, first_pair_payloads_unchanged=True,
        no_model_evaluation_or_training_by_executor=True))
    print("All seven declared pairs executed; model and first-pair artifacts unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pose-manifest", type=Path, required=True)
    parser.add_argument("--first-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.project_root.resolve(), args.pose_manifest.resolve(),
        args.first_bundle.resolve(), args.output.resolve())
