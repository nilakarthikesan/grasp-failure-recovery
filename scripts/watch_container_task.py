#!/usr/bin/env python
"""Watch the scripted Panda complete one container task in real time.

On macOS, launch this with MuJoCo's ``mjpython`` executable::

    MUJOCO_GL=glfw .venv-part1/bin/mjpython -u scripts/watch_container_task.py
"""

from __future__ import annotations

import time

from grasp_failure_prediction.part1.environment import WeightedContainerTask
from grasp_failure_prediction.part1.scripted import ScriptedDemonstrator


def main() -> int:
    env = WeightedContainerTask(
        render_images=False,
        capture_snapshots=False,
        onscreen=True,
    )
    try:
        obs = env.reset(seed=1000)
        teacher = ScriptedDemonstrator(env.table_top, env.cube_half_z)
        cube_pos = env.container_position()

        while True:
            started = time.monotonic()
            action = teacher.act(obs["eef_pose"][:3], cube_pos, env.target_xy)
            obs, step = env.step(action)
            cube_pos = step.privileged["object_pose"][:3]

            # Keep the viewer near the environment's 20 Hz control rate.
            time.sleep(max(0.0, env.dt - (time.monotonic() - started)))
            if step.success or step.failure:
                outcome = "SUCCESS" if step.success else "FAILURE"
                print(f"episode finished after {step.step_index} steps: {outcome}")
                time.sleep(2.0)
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
