"""Open the actual HUG attempt in MuJoCo's interactive viewer using mjpython."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.runner import AdroitShadowRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--urdf-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, help="Full qpos trace from alignment ablations")
    args = parser.parse_args()
    runner = AdroitShadowRunner(load_protocol_registry().resolve("fixed_grasp_lift_v1"), args.urdf_root)
    saved = np.load(args.observation / "scene_state.npz")
    trace = np.load(args.trace or args.observation / "execution_trace.npz")
    transform = np.asarray(json.loads((args.observation / "execution_report.json").read_text())["T_world_wrist"])
    orientation = np.empty(4)
    mujoco.mju_mat2Quat(orientation, transform[:3, :3].reshape(9))
    runner.data.qpos[:] = saved["qpos"]
    runner.data.qvel[:] = 0
    mujoco.mj_forward(runner.model, runner.data)
    print("Replaying the recorded HUG attempt; this is not fresh inference. Close the viewer to exit.", flush=True)
    with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, .08]
        viewer.cam.distance = .7
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -25
        index = 0
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                if "qpos" in trace:
                    runner.data.qpos[:] = trace["qpos"][index]
                else:
                    hand = trace["hand_qpos"][index]
                    root, quaternion = runner._root_pose_for_palm(
                        trace["palm_position_m"][index], orientation, hand,
                    )
                    address = runner._root_qpos_address
                    runner.data.qpos[address:address + 3] = root
                    runner.data.qpos[address + 3:address + 7] = quaternion
                    runner.data.qpos[runner._hand_qpos_addresses] = hand
                    address = runner._object_qpos_address
                    runner.data.qpos[address:address + 3] = trace["object_position_m"][index]
                runner.data.qvel[:] = 0
                mujoco.mj_forward(runner.model, runner.data)
            viewer.sync()
            index = (index + 1) % len(trace["time_s"])
            time.sleep(max(0, .04 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
