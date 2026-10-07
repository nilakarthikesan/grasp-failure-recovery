"""Controlled alignment ablations on the same saved HUG proposal."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from grasp_failure_prediction.integrations.hug import load_hug_prediction
from grasp_failure_prediction.integrations.hug_frames import convert_hand_frame, MANO_TO_OPERATOR_RIGHT
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator, _body_id
from grasp_failure_prediction.evaluation.runner import AdroitShadowRunner
from grasp_failure_prediction.evaluation.scoring import score_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--urdf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    grasp = load_hug_prediction(args.observation / "proposal.pkl")
    T_wc = np.load(args.observation / "T_world_camera.npy")
    T_wm = T_wc @ grasp.T_camera_wrist
    snapshot = np.load(args.observation / "scene_state.npz")
    reports = []
    # Vary axes, optimizer convergence, palm orientation, and closure separately.
    variants = [("original", False, 1, False, .6, 1.2),
                ("converged_only", False, 20, False, .6, 1.2),
                ("axes_only", True, 20, False, .6, 1.2),
                ("axes_and_palm_frame", True, 20, True, .6, 1.2),
                ("full_target_closure", True, 20, True, 1., 1.2),
                ("unscaled_target", True, 20, True, 1., 1.)]
    for name, axes, iterations, palm_frame, closure, scale in variants:
        retargeter = ShadowHandRetargeter(args.urdf_root)
        pose = retargeter.retarget(grasp)
        vectors, base_pose = convert_hand_frame(pose.reference_vectors_m, T_wm,
                                                MANO_TO_OPERATOR_RIGHT if axes else np.eye(3))
        if iterations > 1:
            # Single static proposal: disable teleoperation smoothing and solve
            # the same target repeatedly to separate convergence from axes.
            retargeter._retargeting.filter = None
            retargeter._retargeting.optimizer.scaling = scale
            for _ in range(iterations):
                qpos = retargeter._retargeting.retarget(vectors.astype(np.float32))
            limits = retargeter._retargeting.optimizer.robot.joint_limits
            qpos = np.clip(qpos, limits[:, 0], limits[:, 1])
            pose = replace(pose, qpos=qpos, reference_vectors_m=vectors)
        validator = ShadowPoseValidator(args.urdf_root)
        alignment = validator.validate(pose, scaling_factor=scale)
        # The optimizer's vectors are expressed in the URDF base frame, whereas
        # runner.execute accepts a palm frame. Compose its actual local FK pose.
        base_to_palm = validator.data.xmat[_body_id(validator.model, "palm")].reshape(3, 3)
        rotation = base_pose[:3, :3] @ base_to_palm if palm_frame else T_wm[:3, :3]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
        protocol = load_protocol_registry().resolve("fixed_grasp_lift_v1")
        protocol = protocol.model_copy(update={"parameters": protocol.parameters.model_copy(update={"grip_command": closure})})
        runner = AdroitShadowRunner(protocol, args.urdf_root)
        runner.reset(seed=42, object_mass_kg=.18, object_position_m=np.array([0, 0, .03]),
                     object_orientation_wxyz=np.array([1, 0, 0, 0]))
        runner.data.qpos[:] = snapshot["qpos"]
        runner.data.qvel[:] = snapshot["qvel"]
        mujoco.mj_forward(runner.model, runner.data)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, .08]
        camera.distance = .7
        camera.azimuth = 130
        camera.elevation = -25
        frames, qposes, contact_details = [], [], []
        with mujoco.Renderer(runner.model, height=480, width=640) as renderer:
            original = runner._record

            def record(state):
                original(state)
                qposes.append(runner.data.qpos.copy())
                for ci in range(runner.data.ncon):
                    contact = runner.data.contact[ci]
                    geoms = {int(contact.geom1), int(contact.geom2)}
                    bodies = {int(runner.model.geom_bodyid[g]) for g in geoms}
                    if runner._object_geom_id in geoms and bodies & runner._hand_body_ids:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(runner.model, runner.data, ci, force)
                        contact_details.append({"phase": state.value, "distance_m": float(contact.dist),
                                                "normal_force_n": float(force[0])})
                if runner._step_index % 2:
                    return
                renderer.update_scene(runner.data, camera=camera)
                im = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 640, 32), fill="black")
                draw.text((8, 8), f"{name} | {state.value}", fill="white")
                frames.append(im)

            runner._record = record
            trace = runner.execute(pose, grasp_palm_position_m=T_wm[:3, 3],
                                   grasp_palm_quaternion_wxyz=quaternion)
        score = score_trace(trace, protocol, control_timestep_s=runner.control_timestep_s)
        frames[0].save(args.output / f"{name}.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)
        np.savez(args.output / f"{name}_trace.npz", qpos=np.stack(qposes), time_s=[s.time_s for s in trace.steps])
        approach = [s for s in trace.steps if s.state.value == "approach"][-1]
        contact_by_phase = {
            state: sum(s.hand_object_contact for s in trace.steps if s.state.value == state)
            for state in ["approach", "close_fingers", "lift", "hold"]
        }
        reports.append({"variant": name, "grip_fraction": closure,
                        "retargeting_scale": scale,
                        "optimizer_iterations": iterations,
                        "mean_fingertip_error_m": alignment.mean_fingertip_error_m,
                        "maximum_fingertip_error_m": alignment.maximum_fingertip_error_m,
                        "approach_palm_target_error_m": float(np.linalg.norm(approach.palm_position_m - T_wm[:3, 3])),
                        "contact_frames_by_phase": contact_by_phase,
                        "minimum_hand_object_contact_distance_m": min((c["distance_m"] for c in contact_details), default=None),
                        "maximum_sampled_normal_force_n": max((c["normal_force_n"] for c in contact_details), default=0),
                        "score": asdict(score)})
        print(json.dumps(reports[-1]), flush=True)
    (args.output / "ablation_report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
