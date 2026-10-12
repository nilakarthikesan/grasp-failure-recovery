"""Render the saved A/B diagnostic, with offline predictions at their cutoff.

Uses saved simulator geometry for replay only. Does not collect new attempts.
"""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np


def render(runs_root, scores, output):
    report = json.loads((scores / "report.json").read_text())
    if report["completed_scored"] != 2:
        raise ValueError("Side-by-side video requires two completed saved attempts")
    rows = {row["posture"]: row for row in report["records"]}
    models = [mujoco.MjModel.from_binary_path(str(runs_root / posture / "scene.mjb"))
              for posture in ("A", "B")]
    data = [mujoco.MjData(model) for model in models]
    actions = [np.load(runs_root / posture / "actions.npz", allow_pickle=False)
               for posture in ("A", "B")]
    trajectories = [np.load(runs_root / posture / "trajectory.npz", allow_pickle=False)
                    for posture in ("A", "B")]
    times = trajectories[0]["time_s"]
    if not np.array_equal(times, trajectories[1]["time_s"]):
        raise ValueError("Replay requires the same saved time grid")
    width, height, heading = 640, 480, 112
    camera = mujoco.MjvCamera()
    data[0].qpos[:] = actions[0]["full_qpos"][0]
    mujoco.mj_forward(models[0], data[0])
    camera.lookat[:] = data[0].xpos[models[0].body("object").id]
    camera.lookat[2] = 0.08
    camera.distance, camera.azimuth, camera.elevation = 0.7, 130, -25
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Choose a new replay output path")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"avc1"), 25,
                             (width * 2, height + heading))
    if not writer.isOpened():
        raise RuntimeError("H.264 video writer unavailable")
    renderers = [mujoco.Renderer(model, height=height, width=width) for model in models]
    try:
        for index, time in enumerate(times):
            frames = []
            for side, posture in enumerate(("A", "B")):
                row = rows[posture]
                data[side].qpos[:] = actions[side]["full_qpos"][index]
                mujoco.mj_forward(models[side], data[side])
                renderers[side].update_scene(data[side], camera=camera, scene_option=option)
                frame = np.ascontiguousarray(renderers[side].render()[:, :, ::-1])
                canvas = np.full((height + heading, width, 3), (24, 20, 16), np.uint8)
                canvas[heading:] = frame
                title = f"Wrist {posture} | same intended grasp | zero noise"
                state = str(trajectories[side]["state"][index])
                prediction = ("Prediction input still being observed" if time < 2.96 - 1e-9 else
                              f"Offline score at 2.96 s: {row['p_failure']:.6f} | "
                              f"{'WARNING' if row['failure_alert'] else 'CLEAR'}")
                outcome = ("Task outcome still unfolding" if index < len(times) - 1 else
                           f"Actual final task: {'SUCCESS' if row['label_success'] else 'FAILURE'}")
                for line, text in enumerate((title, f"{time:.2f} s | {state}", prediction, outcome)):
                    cv2.putText(canvas, text, (12, 23 + 25 * line), cv2.FONT_HERSHEY_SIMPLEX,
                                0.56, (235, 235, 235), 1, cv2.LINE_AA)
                frames.append(canvas)
            combined = np.hstack(frames)
            writer.write(combined)
            if abs(time - 2.96) < 1e-9:
                cv2.imwrite(str(output.with_suffix(".png")), combined)
            if index == len(times) - 1:
                for _ in range(50):
                    writer.write(combined)
    finally:
        for renderer in renderers:
            renderer.close()
        writer.release()
        for file in actions + trajectories:
            file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.runs_root.resolve(), args.scores.resolve(), args.output.resolve())
