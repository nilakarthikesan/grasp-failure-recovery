"""Replay the separately versioned conservative guard on saved HUG arrays.

No forward kinematics, geometry call, fitting, proposal or physics is run.
This is a selection-safeguard replay, not mapping validation or correction.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    p = read(args.protocol)
    if sha(__file__) != p["replay_script_sha256"]:
        raise ValueError("replay script differs from frozen protocol")
    for path, expected in p["input_sha256"].items():
        if sha(args.root / path) != expected:
            raise ValueError("frozen input changed: " + path)
    guard_path = args.root / p["guard_module_path"]
    spec = importlib.util.spec_from_file_location("frozen_direction_guard_replay", guard_path)
    guard_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard_module
    spec.loader.exec_module(guard_module)
    candidates = read(args.root / p["candidate_report_path"])["cases"]
    if [c["source_inventory_index"] for c in candidates] != p["source_inventory_indices"]:
        raise ValueError("six original cases or order changed")
    results = []
    for c in candidates:
        if (len(c["selected_tip_norms_sqrt_einsum_m"]) != 5 or
                len(c["authoritative_original_tip_caps_sqrt_einsum_m"]) != 5):
            raise ValueError("all five frozen fingertip norms/caps are required")
        baseline = c["baseline_fresh_diagnostics"]["normal_diagnostics"]
        selected = c["selected_fresh_diagnostics"]["normal_diagnostics"]
        guard = guard_module.freeze_direction_component_guard(
            c["direction_reference_by_part"], baseline)
        identity_result = guard.assess(baseline)
        selected_result = guard.assess(selected)
        if identity_result["accepted"] is not True:
            raise AssertionError("original identity control was refused")
        if selected_result["accepted"] is not False:
            raise AssertionError("known stored direction-regressing selected fit was allowed")
        results.append({"source_inventory_index": c["source_inventory_index"],
                        "case_id": c["object_id"] + "_" + c["source_case"],
                        "native_sha256": c["native_sha256"],
                        "original_identity_control": identity_result,
                        "stored_selected_pose_guard": selected_result,
                        "all_five_selected_tip_caps_pass_saved_arithmetic": all(
                            a <= b for a, b in zip(c["selected_tip_norms_sqrt_einsum_m"],
                                                  c["authoritative_original_tip_caps_sqrt_einsum_m"])),
                        "mapping_acceptance": "FAIL in all original recorded resolutions; not recomputed or changed"})
    report = {"schema": "hug_direction_component_guard_saved_replay_v1",
              "protocol_sha256": sha(args.protocol),
              "replay_script_sha256": sha(__file__),
              "guard_module_sha256": sha(guard_path),
              "candidate_count": len(results),
              "original_identity_controls_accepted": sum(r["original_identity_control"]["accepted"] for r in results),
              "stored_selected_poses_refused": sum(not r["stored_selected_pose_guard"]["accepted"] for r in results),
              "records": results, "interpretation": p["interpretation"],
              "new_generator_proposals": False, "physics_steps": 0,
              "new_eval_data": False, "training": False, "new_fits": False}
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
