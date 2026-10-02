"""Audit saved simulator observation, HUG encoding, and final collision geometry.

Requires the combined evaluation/observation package and trusted local pickle
artifacts. This audits one attempt, not general grasp-model correctness.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys
import types

import cv2
import mujoco
import numpy as np
from PIL import Image
import torch

from grasp_failure_prediction.integrations.observations import validate_depth_encoding
from grasp_failure_prediction.integrations.hug import load_hug_prediction
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.runner import AdroitShadowRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--urdf-root", type=Path, required=True)
    parser.add_argument("--hug-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.observation
    metric = np.load(root / "depth_m.npy")
    png = np.asarray(Image.open(root / "depth.png"))
    depth_report = validate_depth_encoding(metric, png)
    prepared = pickle.loads((root / "inputs/cube.pkl").read_bytes())
    decoded_depth = cv2.imdecode(np.frombuffer(prepared["depth"], np.uint8), cv2.IMREAD_UNCHANGED)
    K = np.load(root / "intrinsics.npy")
    if not np.array_equal(decoded_depth, png) or not np.array_equal(prepared["camera"]["K"], K):
        raise ValueError("Prepared HUG input does not match the captured depth/calibration")
    package = types.ModuleType("hug")
    package.__path__ = [str(args.hug_root.resolve() / "src")]
    sys.modules["hug"] = package
    from hug.dataloader.grasp_dataset import GraspDataset
    dataset = GraspDataset(str(root / "inputs"), split="eval")
    item = dataset[0]
    interactive = dataset.get_inference_data("cube")
    u, v = prepared["condition_point"]
    rgb_agrees = bool(torch.equal(item["rgb"], interactive["rgb"]))
    expected_point = np.array([u, v, png[int(v), int(u)] / 1000], dtype=np.float32)
    np.testing.assert_allclose(item["point_uv"].numpy(), expected_point)
    if not rgb_agrees:
        raise ValueError("Dataset RGB differs from official interactive inference RGB")
    runner = AdroitShadowRunner(load_protocol_registry().resolve("fixed_grasp_lift_v1"), args.urdf_root)
    trace = np.load(args.trace)
    parameters = runner.protocol.parameters
    # Four initial recorded states, followed by approach and closing.
    index = 4 + runner._control_steps(parameters.approach_duration_s) + runner._control_steps(parameters.close_duration_s) - 1
    runner.data.qpos[:] = trace["qpos"][index]
    mujoco.mj_forward(runner.model, runner.data)
    distances = {}
    for i in range(runner.model.ngeom):
        body = int(runner.model.geom_bodyid[i])
        if body not in runner._hand_body_ids:
            continue
        name = mujoco.mj_id2name(runner.model, mujoco.mjtObj.mjOBJ_BODY, body)
        distance = float(mujoco.mj_geomDistance(runner.model, runner.data, i, runner._object_geom_id, 1., np.zeros(6)))
        distances[name] = min(distances.get(name, 1.), distance)
    grasp = load_hug_prediction(root / "proposal.pkl")
    T_wc = np.load(root / "T_world_camera.npy")
    vertices_world = grasp.mesh_vertices @ T_wc[:3, :3].T + T_wc[:3, 3]
    report = {
        "observation_source": "virtual camera in the partner MuJoCo cube scene",
        "cube_full_dimensions_m": (runner.model.geom_size[runner._object_geom_id] * 2).tolist(),
        "cube_mass_kg": float(runner.model.body_mass[runner._object_body_id]),
        "simulator_actuators": runner.model.nu,
        "depth_encoding": depth_report,
        "official_prepared_depth_matches_capture": True,
        "official_prepared_intrinsics_match_capture": True,
        "official_dataset_and_interactive_rgb_match": rgb_agrees,
        "selected_pixel_and_depth_match_interactive_contract": True,
        "selected_uv_and_depth_m": expected_point.tolist(),
        "human_wrist_landmark_matches_transform_translation_m": float(np.linalg.norm(grasp.landmarks_3d[0] - grasp.T_camera_wrist[:3, 3])),
        "human_mesh_minimum_world_height_m": float(vertices_world[:, 2].min()),
        "robot_collision_gap_at_end_of_close_m": distances,
        "thumb_nearest_gap_m": min(d for name, d in distances.items() if name.startswith("th")),
        "stable_shadow_grasp_validated": False,
        "limitations": ["Checks one saved observation and one proposal", "Synthetic-image domain shift remains untested", "Landmark fit does not establish collision-surface fit or force closure"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
