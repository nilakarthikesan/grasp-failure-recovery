#!/usr/bin/env python3
"""Render identical HUG proposal executions side by side from saved states."""

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = [mujoco.MjModel.from_binary_path(str(p / "scene.mjb"))
              for p in (args.before, args.after)]
    data = [mujoco.MjData(m) for m in models]
    actions = [np.load(p / "actions.npz") for p in (args.before, args.after)]
    trajectories = [np.load(p / "trajectory.npz") for p in (args.before, args.after)]
    if not np.array_equal(trajectories[0]["time_s"], trajectories[1]["time_s"]):
        raise ValueError("comparison requires identical saved time grids")
    width, height = 640, 480
    camera = mujoco.MjvCamera()
    data[0].qpos[:] = actions[0]["full_qpos"][0]
    mujoco.mj_forward(models[0], data[0])
    camera.lookat[:] = data[0].xpos[models[0].body("object").id]
    camera.lookat[2] = 0.08
    camera.distance = 0.7
    camera.azimuth = 130
    camera.elevation = -25
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"avc1"),
                             25, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError("H.264 writer unavailable")
    renderers = [mujoco.Renderer(m, height=height, width=width) for m in models]
    try:
        for index, time in enumerate(trajectories[0]["time_s"]):
            frames = []
            for side in (0, 1):
                data[side].qpos[:] = actions[side]["full_qpos"][index]
                mujoco.mj_forward(models[side], data[side])
                renderers[side].update_scene(data[side], camera=camera, scene_option=option)
                frame = np.ascontiguousarray(renderers[side].render()[:, :, ::-1])
                cv2.rectangle(frame, (0, 0), (width, 62), (12, 16, 24), -1)
                title = ("Original: forearm intersects table" if side == 0
                         else "Corrected: wrist adjusted, same grasp")
                cv2.putText(frame, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.61, (225, 225, 245) if side == 0 else (175, 245, 175),
                            1, cv2.LINE_AA)
                text = f'{trajectories[side]["state"][index]} | {float(time):.2f} s | noise 0 rad'
                cv2.putText(frame, text, (12, 49), cv2.FONT_HERSHEY_SIMPLEX,
                            0.56, (220, 220, 220), 1, cv2.LINE_AA)
                frames.append(frame)
            combined = np.hstack(frames)
            writer.write(combined)
            if index == 24:
                cv2.imwrite(str(args.output.with_suffix(".png")), combined)
    finally:
        for renderer in renderers:
            renderer.close()
        writer.release()


if __name__ == "__main__":
    main()
