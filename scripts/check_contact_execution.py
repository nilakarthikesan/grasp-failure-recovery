"""Compare original and velocity-consistent prescribed-hand execution.

This remains a kinematic diagnostic, not an actuator-calibrated controller.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import mujoco
import numpy as np

from grasp_failure_prediction.integrations.hug import load_hug_prediction
from grasp_failure_prediction.integrations.hug_frames import convert_hand_frame
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator, _body_id
from grasp_failure_prediction.evaluation.runner import AdroitShadowRunner
from grasp_failure_prediction.evaluation.scoring import score_trace


class VelocityConsistentRunner(AdroitShadowRunner):
    """Move prescribed joints smoothly with velocities consistent with qpos."""
    def _advance(self, state, root_position, root_quaternion, hand_qpos):
        start = self.data.qpos.copy()
        self._set_kinematic_targets(root_position, root_quaternion, hand_qpos)
        goal = self.data.qpos.copy()
        self.data.qpos[:] = start
        velocity = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, velocity, self.control_timestep_s, start, goal)
        # Root (6 velocities) + 24 hand joints; preserve the free object's state.
        velocity[30:] = 0
        for step in range(self.physics_steps_per_control):
            commanded = start.copy()
            mujoco.mj_integratePos(self.model, commanded, velocity, step * self.physics_timestep_s)
            self.data.qpos[:31] = commanded[:31]
            self.data.qvel[:30] = velocity[:30]
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
        self.data.qpos[:31] = goal[:31]
        self.data.qvel[:30] = velocity[:30]
        mujoco.mj_forward(self.model, self.data)
        self._record(state)


def setup(observation, urdf_root):
    grasp = load_hug_prediction(observation / "proposal.pkl")
    retargeter = ShadowHandRetargeter(urdf_root)
    pose = retargeter.retarget(grasp)
    vectors, base_pose = convert_hand_frame(pose.reference_vectors_m,
        np.load(observation / "T_world_camera.npy") @ grasp.T_camera_wrist)
    retargeter._retargeting.filter = None
    retargeter._retargeting.optimizer.scaling = 1.0
    for _ in range(20):
        joints = retargeter._retargeting.retarget(vectors.astype(np.float32))
    limits = retargeter._retargeting.optimizer.robot.joint_limits
    pose = replace(pose, qpos=np.clip(joints, limits[:, 0], limits[:, 1]), reference_vectors_m=vectors)
    validator = ShadowPoseValidator(urdf_root)
    validator.set_pose(pose)
    base_to_palm = validator.data.xmat[_body_id(validator.model, "palm")].reshape(3, 3)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (base_pose[:3, :3] @ base_to_palm).reshape(9))
    return pose, base_pose[:3, 3], quat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--urdf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pose, position, quaternion = setup(args.observation, args.urdf_root)
    protocol = load_protocol_registry().resolve("fixed_grasp_lift_v1")
    protocol = protocol.model_copy(update={"parameters": protocol.parameters.model_copy(update={"grip_command": 1.})})
    snapshot = np.load(args.observation / "scene_state.npz")
    reports = []
    for name, cls in [("zero_hand_velocity", AdroitShadowRunner),
                      ("consistent_hand_velocity", VelocityConsistentRunner),
                      ("consistent_velocity_noslip", VelocityConsistentRunner)]:
        runner = cls(protocol, args.urdf_root)
        if name == "consistent_velocity_noslip":
            runner.model.opt.noslip_iterations = 10
            runner.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        runner.reset(seed=42, object_mass_kg=.18, object_position_m=np.array([0, 0, .03]),
                     object_orientation_wxyz=np.array([1, 0, 0, 0]))
        runner.data.qpos[:] = snapshot["qpos"]
        runner.data.qvel[:] = snapshot["qvel"]
        mujoco.mj_forward(runner.model, runner.data)
        states, contact_diagnostics = [], []
        original = runner._record

        def record(state):
            original(state)
            states.append(runner.data.qpos.copy())
            normals, bodies = [], []
            for ci in range(runner.data.ncon):
                c = runner.data.contact[ci]
                pair = {int(c.geom1), int(c.geom2)}
                if runner._object_geom_id not in pair:
                    continue
                other_geom = int(c.geom1) if int(c.geom2) == runner._object_geom_id else int(c.geom2)
                body = int(runner.model.geom_bodyid[other_geom])
                if body not in runner._hand_body_ids:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(runner.model, runner.data, ci, force)
                if force[0] <= .01:
                    continue
                normals.append(c.frame[:3].copy() * (1 if int(c.geom2) == runner._object_geom_id else -1))
                bodies.append(mujoco.mj_id2name(runner.model, mujoco.mjtObj.mjOBJ_BODY, body))
            opposing = any(np.dot(a, b) < -.5 for i, a in enumerate(normals) for b in normals[i + 1:])
            contact_diagnostics.append({"phase": state.value, "opposing_normals": opposing, "bodies": bodies})

        runner._record = record
        trace = runner.execute(pose, grasp_palm_position_m=position,
                               grasp_palm_quaternion_wxyz=quaternion)
        result = score_trace(trace, protocol, control_timestep_s=runner.control_timestep_s)
        np.savez(args.output / f"{name}_trace.npz", qpos=np.stack(states), time_s=[s.time_s for s in trace.steps])
        reports.append({"variant": name, "score": asdict(result),
                        "force_bearing_hand_bodies": sorted({b for d in contact_diagnostics for b in d["bodies"]}),
                        "opposing_contact_frames_by_phase": {p: sum(d["opposing_normals"] for d in contact_diagnostics if d["phase"] == p)
                            for p in ["close_fingers", "lift", "hold"]},
                        "contact_frames_by_phase": {p: sum(s.hand_object_contact for s in trace.steps if s.state.value == p)
                            for p in ["close_fingers", "lift", "hold"]}})
        print(json.dumps(reports[-1]), flush=True)
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
