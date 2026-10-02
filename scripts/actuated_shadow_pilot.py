"""Actuated Shadow Hand diagnostic using unchanged collision shapes.

Each hinge has a force-limited position servo. A dynamic free forearm is welded
to a commanded mocap carrier (ideal arm); fingers move through physics, not qpos
overwrites. Gains are experimental, not calibrated to a physical Shadow robot.
"""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import tempfile
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from check_contact_execution import setup
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.runner import AdroitShadowRunner, ExecutionState
from grasp_failure_prediction.evaluation.retargeting import reorder_for_mujoco
from grasp_failure_prediction.evaluation.scoring import score_trace


# Positive flexion joints from the Shadow URDF. Abduction/circumduction and
# wrist commands remain unchanged. This is an experimental execution rule,
# not the authors' released per-joint force-close implementation.
FLEXION_JOINTS = frozenset(
    f"{finger}J{joint}" for finger in ("FF", "MF", "RF", "LF") for joint in (1, 2, 3)
) | {"THJ1", "THJ2", "THJ4"}


def force_close_targets(joint_names, target, limits, delta_rad):
    """Apply one fixed, bounded closure rule without inspecting the object."""
    if not np.isfinite(delta_rad) or not 0 <= delta_rad <= .3:
        raise ValueError("Experimental force-close offset must be in [0, 0.3] radians")
    command = np.asarray(target, dtype=float).copy()
    for i, name in enumerate(joint_names):
        if name in FLEXION_JOINTS:
            command[i] += delta_rad
    return np.clip(command, limits[:, 0], limits[:, 1])


def add_actuation(model):
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "scene.xml"
        mujoco.mj_saveLastXML(str(xml), model)
        tree = ET.parse(xml)
        root = tree.getroot()
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("cone", "elliptic")
        option.set("noslip_iterations", "10")
        option.set("timestep", ".002")
        option.set("integrator", "implicitfast")
        world = root.find("worldbody")
        ET.SubElement(world, "body", {"name": "arm_target", "mocap": "true",
            "pos": " ".join(map(str, model.qpos0[:3])),
            "quat": " ".join(map(str, model.qpos0[3:7]))})
        equality = ET.SubElement(root, "equality")
        ET.SubElement(equality, "weld", {"name": "ideal_arm", "body1": "arm_target", "body2": "forearm",
            "relpose": "0 0 0 1 0 0 0", "solref": ".01 1"})
        actuator = ET.SubElement(root, "actuator")
        for joint in root.findall(".//joint"):
            name = joint.get("name")
            if not name or joint.get("type") == "free":
                continue
            limits = joint.get("range")
            effort = abs(float(joint.get("actuatorfrcrange", "-1 1").split()[1]))
            if not name.startswith("WR"):
                effort = min(effort, 1.)
            ET.SubElement(actuator, "position", {"name": f"servo_{name}", "joint": name,
                "kp": "50" if name.startswith("WR") else "10", "kv": ".3",
                "ctrllimited": "true", "ctrlrange": limits,
                "forcelimited": "true", "forcerange": f"{-effort} {effort}"})
        tree.write(xml)
        return mujoco.MjModel.from_xml_path(str(xml))


class ActuatedShadowRunner(AdroitShadowRunner):
    def __init__(self, protocol, urdf_root, *, force_close_delta_rad=0.):
        super().__init__(protocol, urdf_root)
        if not np.isfinite(force_close_delta_rad) or not 0 <= force_close_delta_rad <= .3:
            raise ValueError("Experimental force-close offset must be in [0, 0.3] radians")
        self.force_close_delta_rad = force_close_delta_rad
        self.model = add_actuation(self.model)
        self.data = mujoco.MjData(self.model)
        self._initialized = False
        self._last_root = self.model.qpos0[:7].copy()
        self._last_ctrl = np.zeros(self.model.nu)
        self.saved_qpos = []
        self.contact_samples = []
        self.peak_force_fraction = 0.

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._initialized = False
        self.saved_qpos = []
        self.contact_samples = []
        self.peak_force_fraction = 0.

    def execute(self, pose, *, grasp_palm_position_m, grasp_palm_quaternion_wxyz):
        params = self.protocol.parameters
        target = reorder_for_mujoco(pose.joint_names, pose.qpos, self._hand_joint_names)
        closing = force_close_targets(self._hand_joint_names, target,
                                      self.model.jnt_range[1:25], self.force_close_delta_rad)
        # Preserve wrist and abduction; open by a small bounded flexion change.
        near_open = target.copy()
        for i,name in enumerate(self._hand_joint_names):
            if name in {"FFJ1","FFJ2","FFJ3","MFJ1","MFJ2","MFJ3","RFJ1","RFJ2","RFJ3","LFJ1","LFJ2","LFJ3","THJ1"}:
                near_open[i] -= .2
        near_open = np.clip(near_open,self.model.jnt_range[1:25,0],self.model.jnt_range[1:25,1])
        root,quat = self._root_pose_for_palm(grasp_palm_position_m,grasp_palm_quaternion_wxyz,target)
        pre = root + [0,0,params.pregrasp_distance_m]
        for state in [ExecutionState.RESET,ExecutionState.LOAD_OBJECT,ExecutionState.RETARGET,ExecutionState.MOVE_TO_PREGRASP]:
            self._advance(state,pre,quat,near_open)
        count = self._control_steps(params.approach_duration_s)
        for step in range(1,count+1):
            alpha=step/count
            self._advance(ExecutionState.APPROACH,(1-alpha)*pre+alpha*root,quat,near_open)
        count=self._control_steps(params.close_duration_s)
        for step in range(1,count+1):
            alpha=step/count
            self._advance(ExecutionState.CLOSE_FINGERS,root,quat,(1-alpha)*near_open+alpha*target)
        settling_steps = self._control_steps(.8)
        for step in range(1, settling_steps + 1):
            # Build grip force only after reaching the original retargeted pose.
            command = target + (closing - target) * step / settling_steps
            self._advance(ExecutionState.CLOSE_FINGERS,root,quat,command)
        count=self._control_steps(params.lift_duration_s)
        for step in range(1,count+1):
            self._advance(ExecutionState.LIFT,root+[0,0,params.lift_height_m*step/count],quat,closing)
        for _ in range(self._control_steps(params.hold_duration_s)):
            self._advance(ExecutionState.HOLD,root+[0,0,params.lift_height_m],quat,closing)
        self._advance(ExecutionState.SCORE,root+[0,0,params.lift_height_m],quat,closing)
        from grasp_failure_prediction.evaluation.runner import ExecutionTrace
        return ExecutionTrace(tuple(self._trace))

    def _advance(self, state, root_position, root_quaternion, hand_qpos):
        if not self._initialized:
            # Initialization only; subsequent motion never overwrites joints.
            self.data.qpos[:3] = root_position
            self.data.qpos[3:7] = root_quaternion
            self.data.qpos[self._hand_qpos_addresses] = hand_qpos
            self.data.qvel[:] = 0
            self._last_root = np.r_[root_position, root_quaternion]
            self._last_ctrl = hand_qpos.copy()
            self._initialized = True
        start = self.data.qpos.copy()
        start[:7] = self._last_root
        goal = start.copy()
        goal[:3] = root_position
        goal[3:7] = root_quaternion
        velocity = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, velocity, self.control_timestep_s, start, goal)
        for step in range(self.physics_steps_per_control):
            alpha = (step + 1) / self.physics_steps_per_control
            command = start.copy()
            mujoco.mj_integratePos(self.model, command, velocity, alpha * self.control_timestep_s)
            self.data.mocap_pos[0] = command[:3]
            self.data.mocap_quat[0] = command[3:7]
            self.data.ctrl[:] = (1-alpha) * self._last_ctrl + alpha * hand_qpos
            mujoco.mj_step(self.model, self.data)
            limits = np.max(np.abs(self.model.actuator_forcerange),axis=1)
            self.peak_force_fraction = max(self.peak_force_fraction,
                float(np.max(np.abs(self.data.actuator_force)/limits)))
            bodies, normals = [], []
            for ci in range(self.data.ncon):
                c = self.data.contact[ci]
                if self._object_geom_id not in {int(c.geom1), int(c.geom2)}:
                    continue
                other = int(c.geom1) if int(c.geom2) == self._object_geom_id else int(c.geom2)
                body = int(self.model.geom_bodyid[other])
                if body not in self._hand_body_ids:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, ci, force)
                if force[0] > .01:
                    bodies.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body))
                    normals.append(c.frame[:3] * (1 if int(c.geom2) == self._object_geom_id else -1))
            opposing = any(np.dot(a,b) < -.5 for i,a in enumerate(normals) for b in normals[i+1:])
            self.contact_samples.append((state.value, opposing, bodies))
        self._last_root = goal[:7].copy()
        self._last_ctrl = hand_qpos.copy()
        self.saved_qpos.append(self.data.qpos.copy())
        self._record(state)


def run_candidate(runner, pose, position, quaternion, snapshot):
    runner.reset(seed=42, object_mass_kg=.18, object_position_m=np.array([0,0,.03]),
                 object_orientation_wxyz=np.array([1,0,0,0]))
    runner.data.qpos[:] = snapshot["qpos"]
    runner.data.qvel[:] = snapshot["qvel"]
    mujoco.mj_forward(runner.model, runner.data)
    trace = runner.execute(pose, grasp_palm_position_m=position, grasp_palm_quaternion_wxyz=quaternion)
    result = score_trace(trace, runner.protocol, control_timestep_s=runner.control_timestep_s)
    report = {"score": asdict(result), "actuators": runner.model.nu,
              "force_close_delta_rad": runner.force_close_delta_rad,
              "force_bearing_bodies": sorted({b for _,_,bs in runner.contact_samples for b in bs}),
              "opposition_time_s": {p: sum(opp for phase,opp,_ in runner.contact_samples if phase==p)*runner.physics_timestep_s
                                    for p in ["close_fingers","lift","hold"]},
              "final_maximum_absolute_actuator_force_nm": float(np.max(np.abs(runner.data.actuator_force))),
              "peak_force_fraction_of_configured_limit": runner.peak_force_fraction,
              "finger_target_error_rad": float(np.max(np.abs(runner.data.qpos[runner._hand_qpos_addresses]-runner.data.ctrl)))}
    return report, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--urdf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-close-delta-rad", type=float, default=0.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pose, position, quaternion = setup(args.observation, args.urdf_root)
    protocol = load_protocol_registry().resolve("fixed_grasp_lift_v1")
    protocol = protocol.model_copy(update={"parameters": protocol.parameters.model_copy(update={"grip_command":1.})})
    runner = ActuatedShadowRunner(protocol, args.urdf_root,
                                  force_close_delta_rad=args.force_close_delta_rad)
    snapshot = np.load(args.observation / "scene_state.npz")
    report, trace = run_candidate(runner, pose, position, quaternion, snapshot)
    report.update({"proposal": "unchanged HUG with corrected retargeting", "hardware_calibrated": False,
                   "collision_shapes_changed": False, "ideal_arm_constraint": True})
    np.savez(args.output / "trace.npz", qpos=np.stack(runner.saved_qpos), time_s=[s.time_s for s in trace.steps])
    mujoco.mj_saveLastXML(str(args.output / "scene.xml"), runner.model)
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
