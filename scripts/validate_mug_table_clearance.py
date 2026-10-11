#!/usr/bin/env python3
"""Run 24 mug checks and three unchanged-object controls, not a data campaign."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from grasp_failure_prediction.evaluation.table_safe_execution import execute_table_safe_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new validation output directory")
    output.mkdir(parents=True, exist_ok=True)
    prefix = Path("runs/fresh_object_scene_eval_2026-10-11")
    entries = []
    for scene in (2, 3):
        for seed in range(4):
            for amplitude in (0.0, 0.15, 0.60):
                entries.append(dict(
                    episode_id=f"mug_scene{scene:03d}_seed{seed}_noise{amplitude:.2f}",
                    case_path=str(prefix / "inputs_new_objects/cases" / f"scene_{scene:03d}_seed_{seed}.yaml"),
                    noise_amplitude_rad=amplitude,
                    noise_seed=int(np.random.SeedSequence([9000, scene, seed]).generate_state(1)[0]),
                    original_episode_id=f"episode_{1560 + (scene - 2) * 260 + seed * 65:06d}",
                    object_role="mug",
                ))
    for name, location, scene, original in (
        ("cube", "inputs", 0, 0), ("apple", "inputs", 2, 520),
        ("milk_carton", "inputs_new_objects", 0, 1040),
    ):
        entries.append(dict(
            episode_id=f"control_{name}",
            case_path=str(prefix / location / "cases" / f"scene_{scene:03d}_seed_0.yaml"),
            noise_amplitude_rad=0.0, noise_seed=9000,
            original_episode_id=f"episode_{original:06d}", object_role="control",
        ))
    plan = dict(schema_version="mug_clearance_validation_plan_v1", episodes=entries,
                scope="27 deterministic controller checks; no predictor training/evaluation")
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    for entry in entries:
        record = execute_table_safe_case(
            entry["case_path"], output / entry["episode_id"], project_root=project,
            noise_amplitude_rad=entry["noise_amplitude_rad"], noise_seed=entry["noise_seed"],
        )
        rows.append(dict(entry, status=record["status"], label=record["label"],
                         failure_type=record.get("failure_type"),
                         sampled_approach_table_contact=record.get("sampled_approach_table_contact"),
                         sampled_hand_table_contact=record.get("sampled_hand_table_contact")))
        print(f'{entry["episode_id"]}: {record["status"]} label={record["label"]} '
              f'table_contact={record.get("sampled_hand_table_contact")}', flush=True)
    summary = dict(planned=len(entries), status_counts=dict(Counter(r["status"] for r in rows)),
                   outcome_counts=dict(Counter(r["failure_type"] or "success" for r in rows
                                               if r["status"] == "completed")),
                   sampled_approach_contacts=sum(bool(r["sampled_approach_table_contact"]) for r in rows),
                   sampled_any_phase_contacts=sum(bool(r["sampled_hand_table_contact"]) for r in rows),
                   episodes=rows,
                   caveat="40 ms samples only; independent 2 ms contact audit required")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
