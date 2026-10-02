"""Positive-control physics check, NOT a HUG or Shadow Hand grasp.

Two position-actuated pads pinch the same 180g/5cm cube. A dynamically welded
carrier follows a mocap target. This tests gravity/friction/actuator execution
independently of observation, retargeting, and the Shadow executor.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81" noslip_iterations="10" cone="elliptic"/>
  <default><geom friction="1 .005 .0001" condim="3"/></default>
  <worldbody>
    <geom name="table" type="box" size=".4 .4 .025" pos="0 0 -.025"/>
    <body name="cube" pos="0 0 .025"><freejoint/>
      <geom name="cube_geom" type="box" size=".025 .025 .025" mass=".18"/>
    </body>
    <body name="target" mocap="true" pos="0 0 .025"/>
    <body name="carrier" pos="0 0 .025"><freejoint/>
      <inertial pos="0 0 .06" mass=".2" diaginertia=".001 .001 .001"/>
      <body name="left_pad" pos="-.04 0 0">
        <joint name="left_close" type="slide" axis="1 0 0" range="0 .025" damping="1"/>
        <geom type="box" size=".005 .02 .025" mass=".03"/>
      </body>
      <body name="right_pad" pos=".04 0 0">
        <joint name="right_close" type="slide" axis="-1 0 0" range="0 .025" damping="1"/>
        <geom type="box" size=".005 .02 .025" mass=".03"/>
      </body>
    </body>
  </worldbody>
  <equality><weld body1="target" body2="carrier" solref=".005 1"/></equality>
  <actuator>
    <position joint="left_close" kp="5000" kv="30" ctrllimited="true" ctrlrange="0 .025" forcelimited="true" forcerange="-8 8"/>
    <position joint="right_close" kp="5000" kv="30" ctrllimited="true" ctrlrange="0 .025" forcelimited="true" forcerange="-8 8"/>
  </actuator>
</mujoco>
"""


def trial(close):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    qposes, times, heights, hold_heights = [], [], [], []
    maximum_normal_force = 0
    for _ in range(2250):
        time = float(data.time)
        data.ctrl[:] = min(time / .8, 1) * .020 if close else 0
        fraction = np.clip((time - 1.) / 1.5, 0, 1)
        data.mocap_pos[0] = [0, 0, .025 + .15 * fraction]
        mujoco.mj_step(model, data)
        height = float(data.xpos[cube, 2])
        heights.append(height)
        if data.time >= 2.5:
            hold_heights.append(height)
        for ci in range(data.ncon):
            c = data.contact[ci]
            pair = {int(c.geom1), int(c.geom2)}
            if cube_geom in pair and 0 not in pair:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, force)
                maximum_normal_force = max(maximum_normal_force, float(force[0]))
        if len(heights) % 20 == 0:
            qposes.append(data.qpos.copy())
            times.append(float(data.time))
    result = {"close_pads": close, "cube_mass_kg": .18, "gravity_m_s2": 9.81,
              "cube_weight_n": .18 * 9.81,
              "maximum_lift_m": max(heights) - .025,
              "minimum_hold_lift_m": min(hold_heights) - .025,
              "final_lift_m": heights[-1] - .025,
              "maximum_individual_normal_contact_force_n": maximum_normal_force,
              "success": min(hold_heights) - .025 >= .14,
              "raw_hug_proposal": False, "shadow_hand_tested": False}
    return result, np.stack(qposes), np.asarray(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "pinch.xml").write_text(XML)
    reports = []
    for close in [False, True]:
        report, qpos, time = trial(close)
        np.savez(args.output / f"pinch_{close}_trace.npz", qpos=qpos, time_s=time)
        reports.append(report)
        print(json.dumps(report), flush=True)
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
