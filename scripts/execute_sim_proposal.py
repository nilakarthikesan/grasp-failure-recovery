"""Execute one actual HUG cube proposal using the partner's fixed protocol.

The initial placement uses the partner's wrist-to-palm correspondence (zero
translation offset). Record its alignment residual; this is a plumbing pilot,
not a calibrated robot grasp-success benchmark.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from grasp_failure_prediction.integrations.hug import load_hug_prediction
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
from grasp_failure_prediction.evaluation.runner import AdroitShadowRunner
from grasp_failure_prediction.evaluation.scoring import score_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--urdf-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.observation.resolve()
    if (root / "execution_report.json").exists():
        raise FileExistsError("Execution report already exists")
    grasp = load_hug_prediction(root / "proposal.pkl")
    retargeter = ShadowHandRetargeter(args.urdf_root.resolve())
    pose = retargeter.retarget(grasp)
    alignment = ShadowPoseValidator(args.urdf_root.resolve()).validate(pose)
    T_world_camera = np.load(root / "T_world_camera.npy")
    T_world_wrist = T_world_camera @ grasp.T_camera_wrist
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, T_world_wrist[:3, :3].reshape(9))
    protocol = load_protocol_registry().resolve("fixed_grasp_lift_v1")
    runner = AdroitShadowRunner(protocol, args.urdf_root.resolve())
    snapshot = np.load(root / "scene_state.npz")
    runner.reset(seed=42, object_mass_kg=0.18,
                 object_position_m=np.array([0, 0, 0.03]),
                 object_orientation_wxyz=np.array([1, 0, 0, 0]))
    runner.data.qpos[:] = snapshot["qpos"]
    runner.data.qvel[:] = snapshot["qvel"]
    mujoco.mj_forward(runner.model, runner.data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.08]
    camera.distance = 0.7
    camera.azimuth = 130
    camera.elevation = -25
    frames = []
    with mujoco.Renderer(runner.model, height=480, width=640) as renderer:
        original_record = runner._record

        def capture(state):
            original_record(state)
            if runner._step_index % 2:
                return
            renderer.update_scene(runner.data, camera=camera)
            frame = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 640, 48), fill="black")
            draw.text((12, 8), "REAL HUG OUTPUT - provisional wrist/palm mapping", fill="white")
            draw.text((12, 27), f"{state.value} | {runner.data.time:.2f}s", fill="white")
            frames.append(frame)

        runner._record = capture
        trace = runner.execute(pose, grasp_palm_position_m=T_world_wrist[:3, 3],
                               grasp_palm_quaternion_wxyz=quaternion)
    score = score_trace(trace, protocol, control_timestep_s=runner.control_timestep_s)
    frames[0].save(root / "actual_hug_attempt.gif", save_all=True,
                   append_images=frames[1:], duration=80, loop=0)
    np.savez(root / "execution_trace.npz",
             object_position_m=np.stack([s.object_position_m for s in trace.steps]),
             palm_position_m=np.stack([s.palm_position_m for s in trace.steps]),
             hand_qpos=np.stack([s.hand_qpos for s in trace.steps]),
             time_s=np.array([s.time_s for s in trace.steps]))
    report = {"actual_hug_proposal_executed": True,
              "proposal_sha256": hashlib.sha256((root / "proposal.pkl").read_bytes()).hexdigest(),
              "T_world_wrist": T_world_wrist.tolist(),
              "score": asdict(score), "steps": len(trace.steps),
              "retargeting_mean_fingertip_error_m": alignment.mean_fingertip_error_m,
              "retargeting_maximum_fingertip_error_m": alignment.maximum_fingertip_error_m,
              "placement_assumption": "HUG wrist corresponds to Shadow palm with zero offset",
              "benchmark_ready": False,
              "limitations": ["Wrist-to-palm placement requires calibration and review",
                              "Partner runner prescribes hand motion kinematically",
                              "Single proposal, single cube, single camera view"]}
    (root / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
