#!/usr/bin/env python3
"""All-32 saved HUG export/camera consistency; static, no inference or rollout.

prepare freezes source, native proposals and original observation contracts.
run never overwrites measurements. Saved 2D predictions are not observed labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grasp_failure_prediction.integrations.hug_projection import (
    image_support, inverse_transform_points, project_camera,
    rotation_from_saved_six_d, transform_points,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__}


def prepare(output):
    if output.exists():
        raise FileExistsError("fresh protocol directory required")
    inventory = ROOT / "research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json"
    cases = json.loads(inventory.read_text())["modern_cases"]
    if len(cases) != 32:
        raise ValueError("quota is exactly original32; no new proposals")
    upstream = Path("/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug/src/models/mano.py")
    sources = [Path(__file__).resolve(), ROOT / "src/grasp_failure_prediction/integrations/hug_projection.py",
               ROOT / "tests/test_hug_projection.py", ROOT / "src/grasp_failure_prediction/integrations/hug.py",
               ROOT / "src/grasp_failure_prediction/integrations/hug_frames.py",
               ROOT / "src/grasp_failure_prediction/integrations/sim_observations.py", upstream, inventory]
    bound_cases = []
    for index, case in enumerate(cases):
        native = Path(case["native"]["path"])
        if sha(native) != case["native"]["sha256"]:
            raise ValueError("native inventory changed")
        observation = native.parent
        contract_path = Path(case["frame_files"]["scene_contract.json"]["path"])
        contract = json.loads(contract_path.read_text())
        capture = json.loads((observation / "capture_report.json").read_text())
        inference = json.loads(Path(case["native"]["inference_report_path"]).read_text())
        for filename, expected in contract["observation_files_sha256"].items():
            if sha(observation / filename) != expected.removeprefix("sha256:"):
                raise ValueError("captured observation contract changed: " + filename)
            sources.append(observation / filename)
        if (sha(contract_path) != case["frame_files"]["scene_contract.json"]["sha256"]
                or sha(contract_path) != inference["scene_contract_sha256"]
                or sha(observation / "observation_manifest.json") != inference["manifest_sha256"]
                or sha(observation / "scene.mjb") != capture["scene_binary_sha256"].removeprefix("sha256:")
                or sha(native) != inference["proposal_sha256"]):
            raise ValueError("capture/inference/native bindings disagree")
        sources.extend([native, contract_path, Path(case["native"]["inference_report_path"]),
                        observation / "observation_manifest.json", observation / "capture_report.json",
                        observation / "scene.mjb", observation / "scene_state.npz"])
        state_entry = case["frame_files"]["scene_state.npz"]
        if sha(observation / "scene_state.npz") != state_entry["sha256"]:
            raise ValueError("saved capture state changed")
        bound_cases.append({"index": index, "native": str(native), "observation": str(observation),
                            "object_id": case["object_id"], "seed": case["native"]["seed"],
                            "observation_group_id": case["observation_group_id"]})
    protocol = {"schema": "hug_saved_camera_projection_v1", "cases": bound_cases,
                "runtime": runtime(), "bindings": {str(p): sha(p) for p in sources},
                "selection": "all32 original saved proposals; no successful-lift or image-support filter",
                "tolerances": {"stored_2d_max_abs_px": 1e-8, "world_roundtrip_max_abs_m": 1e-12,
                    "projection_world_roundtrip_max_abs_px": 1e-8,
                    "compiled_camera_max_abs": 1e-10, "compiled_intrinsics_max_abs_px": 1e-9,
                    "native_wrist_rotation_max_abs": 1e-6, "native_wrist_translation_max_abs_m": 1e-12},
                "tolerance_basis": "float64 projection/frame arithmetic allowances; 1e-6 rotation allowance for stored float32 Gram-Schmidt; numerical engineering checks, not physical fidelity budgets",
                "hug_export_denominator_epsilon": 1e-8, "scene_depth_descriptive_allowance_m": .002,
                "projection_reference": str(upstream),
                "camera_convention": "OpenCV x right, y down, z forward; captured MuJoCo OpenGL camera uses diag(1,-1,-1)",
                "native_geometry_frame": "saved mesh_vertices/landmarks_3d already include wrist rotation and translation; no second wrist transform",
                "independent_observed_2d_ground_truth": "not available; stored landmarks_2d are projections of predicted landmarks_3d",
                "image_support_policy": "positive depth/window/mask/depth counts descriptive; preserve poor/outside/occluded proposals",
                "physics_integration_allowed": False, "new_inference_allowed": False,
                "new_retargeting_allowed": False, "outcome_labels_or_training_allowed": False}
    output.mkdir(parents=True)
    write(output / "protocol.json", protocol)
    print(json.dumps({"protocol": str(output / "protocol.json"), "sha256": sha(output / "protocol.json"), "case_count": 32}))


def run(output):
    path = output / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol_hash = sha(path)
    if (output / "results.json").exists():
        raise FileExistsError("do not overwrite measured evidence")
    def verify():
        if sha(path) != protocol_hash or runtime() != protocol["runtime"]:
            raise ValueError("protocol/runtime changed")
        for filename, expected in protocol["bindings"].items():
            if sha(filename) != expected:
                raise ValueError("source/input binding changed: " + filename)
    verify()
    counters = {"physics_steps": 0, "static_forwards": 0, "new_inference": 0, "new_retargeting": 0}
    def no_integration(*args, **kwargs):
        counters["physics_steps"] += 1
        raise RuntimeError("static audit forbids physics integration")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, no_integration)
    tol = protocol["tolerances"]
    records = []
    for case in protocol["cases"]:
        observation = Path(case["observation"])
        # Inputs are original user-generated, hash-verified NumPy dictionaries.
        raw = pickle.loads(Path(case["native"]).read_bytes())
        k = np.load(observation / "intrinsics.npy", allow_pickle=False)
        transform = np.load(observation / "T_world_camera.npy", allow_pickle=False)
        with Image.open(observation / "rgb.png") as image:
            width, height = image.size
        depth = np.load(observation / "depth_m.npy", allow_pickle=False)
        mask = np.asarray(Image.open(observation / "object_mask.png"))
        model = mujoco.MjModel.from_binary_path(str(observation / "scene.mjb"))
        data = mujoco.MjData(model)
        with np.load(observation / "scene_state.npz", allow_pickle=False) as state:
            data.qpos[:] = state["qpos"]
            data.qvel[:] = state["qvel"]
        mujoco.mj_forward(model, data)
        counters["static_forwards"] += 1
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        if camera_id < 0:
            raise ValueError("missing captured front camera")
        compiled_transform = np.eye(4)
        compiled_transform[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) * np.array([1., -1., -1.])
        compiled_transform[:3, 3] = data.cam_xpos[camera_id]
        focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2))
        compiled_k = np.array([[focal, 0., (width - 1) / 2], [0., focal, (height - 1) / 2], [0., 0., 1.]])
        errors = {"compiled_camera_max_abs": float(np.max(np.abs(compiled_transform - transform))),
                  "compiled_intrinsics_max_abs_px": float(np.max(np.abs(compiled_k - k))),
                  "native_wrist_translation_max_abs_m": float(np.max(np.abs(np.asarray(raw["T_camera_wrist"])[:3, 3] - np.asarray(raw["t"]).reshape(3)))),
                  "native_wrist_rotation_max_abs": float(np.max(np.abs(np.asarray(raw["T_camera_wrist"])[:3, :3] - rotation_from_saved_six_d(raw["R_6d"]))))}
        parts = {}
        for field in ("landmarks_3d", "mesh_vertices"):
            points = np.asarray(raw[field], float)
            world = transform_points(points, transform)
            restored = inverse_transform_points(world, transform)
            export_uv, positive = project_camera(points, k, denominator_epsilon=1e-8, include_nonpositive=True)
            plain_uv, _ = project_camera(points, k)
            via_world, _ = project_camera(restored, k, denominator_epsilon=1e-8, include_nonpositive=True)
            values = {"world_roundtrip_max_abs_m": float(np.max(np.abs(restored - points))),
                      "projection_world_roundtrip_max_abs_px": float(np.max(np.abs(via_world[positive] - export_uv[positive]))) if positive.any() else None,
                      "hug_epsilon_vs_plain_pinhole_max_abs_px": float(np.max(np.abs(export_uv[positive] - plain_uv[positive]))) if positive.any() else None,
                      "image_support": image_support(points, k, width, height, depth=depth, mask=mask)}
            if field == "landmarks_3d":
                stored = raw.get("landmarks_2d")
                values["stored_2d_available"] = stored is not None
                values["stored_2d_max_abs_px"] = float(np.max(np.abs(np.asarray(stored) - export_uv))) if stored is not None and np.isfinite(export_uv).all() else None
            parts[field] = values
        checks = {key: value <= tol[key] for key, value in errors.items()}
        checks["world_roundtrip"] = all(v["world_roundtrip_max_abs_m"] <= tol["world_roundtrip_max_abs_m"] for v in parts.values())
        checks["projection_world_roundtrip"] = all(v["projection_world_roundtrip_max_abs_px"] is not None and v["projection_world_roundtrip_max_abs_px"] <= tol["projection_world_roundtrip_max_abs_px"] for v in parts.values())
        uv_error = parts["landmarks_3d"]["stored_2d_max_abs_px"]
        checks["stored_2d_export_consistency"] = uv_error is not None and uv_error <= tol["stored_2d_max_abs_px"]
        records.append({**case, "image_width": width, "image_height": height, "errors": errors,
                        "parts": parts, "checks": checks, "numerical_consistency_pass": all(checks.values())})
    verify()
    report = {"schema": "hug_saved_camera_projection_v1", "protocol_sha256": protocol_hash,
              "case_count": len(records), "counts": counters, "cases": records,
              "all_numerical_consistency_pass": all(x["numerical_consistency_pass"] for x in records),
              "independent_observed_2d_ground_truth_validated": False,
              "geometric_or_dynamic_contact_fidelity_validated": False,
              "new_eval_data": False, "source_bindings_unchanged": True}
    write(output / "results.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else run)(args.output.resolve())
