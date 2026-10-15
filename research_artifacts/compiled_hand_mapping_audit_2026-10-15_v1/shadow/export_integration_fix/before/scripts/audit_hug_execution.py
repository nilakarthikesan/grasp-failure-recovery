"""Separate raw HUG geometry, robot conversion, and fixed execution behavior.

No proposal, object geometry, or target contact point is optimized. MANO skinning
weights are used only to identify mesh regions in this diagnostic. This is one
cube case, not a benchmark or a reproduction of the authors' robot controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import mujoco
import numpy as np

from actuated_shadow_pilot import ActuatedShadowRunner, run_candidate
from check_contact_execution import setup
from grasp_failure_prediction.evaluation.pose_validation import (
    ShadowPoseValidator, TASK_LINK_NAMES, _link_positions,
)
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import (
    ShadowHandRetargeter, reorder_for_mujoco,
)
from grasp_failure_prediction.integrations.hug import load_hug_prediction


def box_signed_distance(points, center, rotation, half_sizes):
    """Exact point SDF to an oriented box; negative values are inside."""
    local = (np.asarray(points) - center) @ rotation
    excess = np.abs(local) - half_sizes
    return (np.linalg.norm(np.maximum(excess, 0), axis=-1)
            + np.minimum(np.max(excess, axis=-1), 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--urdf-root", type=Path, required=True)
    parser.add_argument("--mano-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    proposal = args.observation / "proposal.pkl"
    before_hash = hashlib.sha256(proposal.read_bytes()).hexdigest()
    grasp = load_hug_prediction(proposal)
    transform = np.load(args.observation / "T_world_camera.npy")
    snapshot = np.load(args.observation / "scene_state.npz")
    pose, position, quaternion = setup(args.observation, args.urdf_root)
    protocol = load_protocol_registry().resolve("fixed_grasp_lift_v1")
    runner = ActuatedShadowRunner(protocol, args.urdf_root)
    runner.data.qpos[:] = snapshot["qpos"]
    mujoco.mj_forward(runner.model, runner.data)
    object_id = runner._object_body_id
    center = runner.data.xpos[object_id].copy()
    rotation = runner.data.xmat[object_id].reshape(3, 3).copy()
    sizes = runner.model.geom_size[runner._object_geom_id].copy()
    if runner.model.geom_type[runner._object_geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("This diagnostic requires a box object")
    landmarks_world = grasp.landmarks_3d @ transform[:3, :3].T + transform[:3, 3]
    vertices_world = grasp.mesh_vertices @ transform[:3, :3].T + transform[:3, 3]

    # Trusted model archive downloaded by the user; never include it in Git.
    with args.mano_model.open("rb") as stream:
        mano = pickle.load(stream, encoding="latin1")
    labels = np.asarray(mano["weights"]).argmax(axis=1)
    groups = {"thumb": [13, 14, 15], "index": [1, 2, 3],
              "middle": [4, 5, 6], "ring": [10, 11, 12], "little": [7, 8, 9]}
    human_gaps = {
        name: float(box_signed_distance(vertices_world[np.isin(labels, ids)],
                                       center, rotation, sizes).min())
        for name, ids in groups.items()
    }

    # Confirm that Pinocchio's robot kinematics and the MuJoCo importer agree.
    retargeter = ShadowHandRetargeter(args.urdf_root)
    robot = retargeter._retargeting.optimizer.robot
    q_pin = reorder_for_mujoco(pose.joint_names, pose.qpos, tuple(robot.dof_joint_names))
    robot.compute_forward_kinematics(q_pin)
    pin_positions = np.array([robot.get_link_pose(robot.get_link_index(name))[:3, 3]
                              for name in TASK_LINK_NAMES])
    validator = ShadowPoseValidator(args.urdf_root)
    validator.set_pose(pose)
    mj_positions = _link_positions(validator.model, validator.data, validator.source_urdf)
    cross_engine_error = float(np.linalg.norm(pin_positions - mj_positions, axis=1).max())
    if cross_engine_error > 1e-5:
        raise ValueError(f"Robot FK differs across engines by {cross_engine_error} m")

    q = reorder_for_mujoco(pose.joint_names, pose.qpos, runner._hand_joint_names)
    root, root_quat = runner._root_pose_for_palm(position, quaternion, q)
    runner.data.qpos[:] = snapshot["qpos"]
    runner.data.qpos[:3] = root
    runner.data.qpos[3:7] = root_quat
    runner.data.qpos[runner._hand_qpos_addresses] = q
    mujoco.mj_forward(runner.model, runner.data)
    robot_positions = _link_positions(runner.model, runner.data, validator.source_urdf)
    human_indices = [4, 8, 12, 16, 20, 2, 6, 10, 14, 18]
    report = {
        "proposal_sha256": before_hash,
        "raw_human_finger_minimum_signed_vertex_gap_m": human_gaps,
        "pinocchio_mujoco_maximum_link_difference_m": cross_engine_error,
        "robot_minus_human_tip_vectors_m":
            (robot_positions[:5] - landmarks_world[human_indices[:5]]).tolist(),
        "robot_human_mean_tip_error_m": float(np.linalg.norm(
            robot_positions[:5] - landmarks_world[human_indices[:5]], axis=1).mean()),
        "executions": [],
        "limitations": [
            "Mesh-vertex distances are static diagnostics, not force-closure certificates",
            "Fixed force-close rule is experimental, not the authors' controller",
            "One proposal on one synthetic observation is not a success-rate estimate",
        ],
    }
    for delta in (0., .05, .1, .2, .3):
        trial_runner = ActuatedShadowRunner(protocol, args.urdf_root,
                                            force_close_delta_rad=delta)
        result, trace = run_candidate(trial_runner, pose, position, quaternion, snapshot)
        stem = f"closure_{delta:.2f}"
        np.savez(args.output / f"{stem}_trace.npz",
                 qpos=np.stack(trial_runner.saved_qpos),
                 time_s=[s.time_s for s in trace.steps],
                 phase=[s.state.value for s in trace.steps])
        result["proposal_modified"] = False
        result["object_aware_contact_fit"] = False
        report["executions"].append(result)
        print(json.dumps(result), flush=True)
    mujoco.mj_saveLastXML(str(args.output / "scene.xml"), trial_runner.model)
    report["proposal_file_unchanged"] = before_hash == hashlib.sha256(proposal.read_bytes()).hexdigest()
    if not report["proposal_file_unchanged"]:
        raise ValueError("Immutable proposal changed during the audit")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
