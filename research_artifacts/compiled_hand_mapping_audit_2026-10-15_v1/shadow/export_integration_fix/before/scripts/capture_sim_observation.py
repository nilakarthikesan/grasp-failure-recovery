"""Capture a calibrated cube observation from the partner's MuJoCo scene.

Run with the partner checkout's src on PYTHONPATH alongside this checkout's src.
This is a simulation integration experiment, not a physical camera calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.runner import AdroitShadowRunner
from grasp_failure_prediction.integrations.observations import check_manifest


def capture(urdf_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    runner = AdroitShadowRunner(
        load_protocol_registry().resolve("fixed_grasp_lift_v1"), urdf_root
    )
    runner.reset(
        seed=42, object_mass_kg=0.18,
        object_position_m=np.array([0.0, 0.0, 0.03]),
        object_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    # Park the hand above the scene before settling the object on the table.
    runner.data.qpos[runner._root_qpos_address:runner._root_qpos_address + 3] = [0, 0, 1]
    mujoco.mj_forward(runner.model, runner.data)
    for _ in range(100):
        mujoco.mj_step(runner.model, runner.data)
    camera_id = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    position = np.array([0.25, -0.35, 0.35])
    target = runner.data.xpos[runner._object_body_id].copy()
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation_gl = np.column_stack([right, up, -forward])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation_gl.reshape(9))
    runner.model.cam_pos[camera_id] = position
    runner.model.cam_quat[camera_id] = quaternion
    runner.model.cam_fovy[camera_id] = 45
    mujoco.mj_forward(runner.model, runner.data)
    size = 224
    focal = size / (2 * np.tan(np.deg2rad(45) / 2))
    K = np.array([[focal, 0, (size - 1) / 2],
                  [0, focal, (size - 1) / 2], [0, 0, 1]])
    T_world_camera = np.eye(4)
    T_world_camera[:3, :3] = rotation_gl @ np.diag([1, -1, -1])
    T_world_camera[:3, 3] = position
    with mujoco.Renderer(runner.model, height=size, width=size) as renderer:
        renderer.update_scene(runner.data, camera="front")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(runner.data, camera="front")
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(runner.data, camera="front")
        segmentation = renderer.render().copy()
    mask = ((segmentation[:, :, 0] == runner._object_geom_id) &
            (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))).astype(np.uint8)
    valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
    depth = np.where(valid, depth, 0).astype(np.float32)
    yy, xx = np.where(mask & valid)
    if not len(xx):
        raise ValueError("Cube is not visible with valid depth")
    index = np.argmin((xx - xx.mean())**2 + (yy - yy.mean())**2)
    selection = [int(xx[index]), int(yy[index])]
    Image.fromarray(rgb).save(output / "rgb.png")
    Image.fromarray(mask * 255).save(output / "object_mask.png")
    Image.fromarray(np.rint(depth * 1000).astype(np.uint16)).save(output / "depth.png")
    np.save(output / "depth_m.npy", depth)
    np.save(output / "intrinsics.npy", K)
    np.save(output / "T_world_camera.npy", T_world_camera)
    np.savez(output / "scene_state.npz", qpos=runner.data.qpos, qvel=runner.data.qvel)
    runner.model.cam_pos[camera_id] = position
    mujoco.mj_saveLastXML(str(output / "scene.xml"), runner.model)
    # Verify backprojected visible points lie on the known cube's surface.
    pixels = np.column_stack([xx, yy, np.ones(len(xx))])
    camera_points = (pixels @ np.linalg.inv(K).T) * depth[yy, xx, None]
    world_points = camera_points @ T_world_camera[:3, :3].T + position
    local = (world_points - target) @ runner.data.xmat[runner._object_body_id].reshape(3, 3)
    surface_error = np.abs(np.max(np.abs(local), axis=1) - 0.025)
    if surface_error.max() > 0.002:
        raise ValueError("Camera projection does not match the simulated cube surface")
    manifest = dict(
        rgb_path="rgb.png", depth_m_path="depth_m.npy", intrinsics_path="intrinsics.npy",
        mask_path="object_mask.png", selection_uv=selection, depth_units="meters",
        registered_to_rgb=True, intrinsics_at_rgb_resolution=True,
        source={"kind": "mujoco_render", "object": "cube", "seed": 42,
                "mujoco_version": mujoco.__version__},
    )
    (output / "observation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checks = check_manifest(output / "observation_manifest.json", output / "checks")
    report = {"input_checks_passed": checks["preflight_passed"],
              "cube_pixels": len(xx), "selection_uv": selection,
              "maximum_cube_surface_error_m": float(surface_error.max()),
              "object_world_position_m": target.tolist(),
              "T_world_camera": T_world_camera.tolist(),
              "camera_convention": "x right, y down, z forward; transform maps camera to world",
              "inference_tested": False,
              "rgb_sha256": hashlib.sha256((output / "rgb.png").read_bytes()).hexdigest()}
    (output / "capture_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not checks["preflight_passed"]:
        raise ValueError(checks["errors"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.urdf_root.resolve(), args.output.resolve()), indent=2))
