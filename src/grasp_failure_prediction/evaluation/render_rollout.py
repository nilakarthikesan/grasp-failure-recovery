"""Render saved MuJoCo control-step states to a browser-compatible MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np


def render_rollout(
    scene_path: Path,
    states_path: Path,
    output_path: Path,
    *,
    fps: float,
    width: int = 640,
    height: int = 480,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    states = np.load(states_path)
    qpos = states["qpos"]
    phases = states["state"]
    times = states["time_s"]
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) == 0:
        raise ValueError("recorded rollout qpos does not match the MuJoCo scene")

    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    camera = mujoco.MjvCamera()
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    camera.lookat[:] = data.xpos[object_id]
    camera.lookat[2] = max(camera.lookat[2], 0.08)
    camera.distance = 0.7
    camera.azimuth = 130.0
    camera.elevation = -25.0
    option = mujoco.MjvOption()
    # Match RGB-D capture: show the source visual mesh, hide collision-only hulls.
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open H.264 video writer: {output_path}")
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for position, phase, time_s in zip(qpos, phases, times, strict=True):
                data.qpos[:] = position
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = np.ascontiguousarray(renderer.render()[:, :, ::-1])
                cv2.rectangle(frame, (0, 0), (width, 38), (12, 16, 24), -1)
                cv2.putText(
                    frame,
                    f"{phase}  {float(time_s):6.2f}s",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args()
    render_rollout(args.scene, args.states, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
