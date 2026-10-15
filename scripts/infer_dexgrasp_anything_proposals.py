#!/usr/bin/env python3
"""Inspect saved DGA exports and write unregistered static mapping candidates.

This is the provider-specific offline worker boundary. Fresh native inference
is not implemented or invoked. Output is intentionally candidate.json rather
than proposal.json: shared registration and scientific acceptance are pending.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grasp_failure_prediction.proposals.dexgrasp_anything import (
    DGAContractError, file_sha256, load_dga_export, map_dga_candidate,
)


def inspect_saved_export(manifest_path, *, object_id, geometry_hash, target_urdf, output,
                         expected_seed=None, expected_count=None):
    manifest, poses = load_dga_export(
        manifest_path, expected_object_id=object_id, expected_geometry_hash=geometry_hash)
    if expected_seed is not None and manifest.seed != expected_seed:
        raise DGAContractError("native_seed_mismatch", "saved native seed differs from request")
    if expected_count is not None and manifest.sample_count != expected_count:
        raise DGAContractError("native_count_mismatch", "saved native count differs from request")
    # Fail closed on corrupt assets; retain valid source proposals outside the
    # target domain as explicit adapter rejects, never physical failure labels.
    records = []
    for pose in poses:
        try:
            records.append(("candidate.json", map_dga_candidate(pose, target_hand_urdf=target_urdf)))
        except DGAContractError as error:
            if error.code != "joint_limit_violation":
                raise
            records.append(("rejection.json", {
                "schema_version": "dga_static_mapping_rejection_v1", "execution_eligible": False,
                "code": error.code, "message": str(error), "physical_failure_label": None,
                "joint_targets_rad": {name: float(value) for name, value in pose.joint_targets_rad.items()},
                "object_from_forearm": pose.object_from_forearm.tolist(),
                "object_from_palm": pose.object_from_palm.tolist(),
            }))
    if output.exists():
        raise DGAContractError("output_exists", "choose a new output directory; existing evidence is preserved")
    output.mkdir(parents=True)
    for index, (filename, candidate) in enumerate(records):
        candidate.update({"evidence_origin": manifest.evidence_origin,
                          "native_manifest_sha256": file_sha256(manifest_path),
                          "sample_index": index, "native_inference_executed": False})
        sample = output / f"sample_{index:04d}"
        sample.mkdir()
        (sample / filename).write_text(json.dumps(candidate, indent=2, allow_nan=False) + "\n")
    report = {
        "schema_version": "dga_offline_worker_report_v1", "sample_count": len(records),
        "candidate_count": sum(name == "candidate.json" for name, _ in records),
        "rejection_count": sum(name == "rejection.json" for name, _ in records),
        "evidence_origin": manifest.evidence_origin, "execution_eligible": False,
        "actual_native_validation": "NOT_TESTED", "native_inference_executed": False,
        "physics_steps": 0, "native_manifest_sha256": file_sha256(manifest_path),
    }
    (output / "inspection_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--geometry-hash", required=True)
    parser.add_argument("--target-hand-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_saved_export(
        args.native_manifest, object_id=args.object_id, geometry_hash=args.geometry_hash,
        target_urdf=args.target_hand_urdf, output=args.output), indent=2))


if __name__ == "__main__":
    main()
