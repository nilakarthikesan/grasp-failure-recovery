"""Search bounded engineered grasp corrections; never label these raw HUG output."""
import argparse
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from check_contact_execution import setup, VelocityConsistentRunner
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.scoring import score_trace


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
    runner = VelocityConsistentRunner(protocol, args.urdf_root)
    snapshot = np.load(args.observation / "scene_state.npz")
    # Dex and MuJoCo use different finger orderings; limits must follow names.
    ranges = {name: runner.model.jnt_range[i+1] for i,name in enumerate(runner._hand_joint_names)}
    lower = np.array([ranges[name][0] for name in pose.joint_names])
    upper = np.array([ranges[name][1] for name in pose.joint_names])
    curl_joints = {"FFJ2", "FFJ3", "MFJ2", "MFJ3", "RFJ2", "RFJ3", "LFJ2", "LFJ3", "THJ1", "THJ2"}
    forward = -position.copy()
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    results = []
    best_height = -1
    # This is same-object engineering, not training/evaluation data or a proof
    # that no feasible grasp exists outside this finite candidate set.
    for curl, dz, advance in itertools.product([0., .1, .2, .4, .6], [-.01, 0., .01], [-.01, 0., .01]):
        joints = pose.qpos.copy()
        for i, name in enumerate(pose.joint_names):
            if name in curl_joints:
                joints[i] += curl
        joints = np.clip(joints, lower, upper)
        corrected = replace(pose, qpos=joints)
        target = position + forward * advance + [0, 0, dz]
        runner.reset(seed=42, object_mass_kg=.18, object_position_m=np.array([0, 0, .03]),
                     object_orientation_wxyz=np.array([1, 0, 0, 0]))
        runner.data.qpos[:] = snapshot["qpos"]
        runner.data.qvel[:] = snapshot["qvel"]
        mujoco.mj_forward(runner.model, runner.data)
        states = []
        original = runner._record

        def record(state):
            original(state)
            states.append(runner.data.qpos.copy())

        runner._record = record
        trace = runner.execute(corrected, grasp_palm_position_m=target,
                               grasp_palm_quaternion_wxyz=quaternion)
        runner._record = original
        score = score_trace(trace, protocol, control_timestep_s=runner.control_timestep_s)
        result = {"curl_rad": curl, "vertical_offset_m": dz, "advance_m": advance,
                  "raw_hug_proposal": False, "score": asdict(score)}
        results.append(result)
        if score.maximum_lift_m > best_height:
            best_height = score.maximum_lift_m
            np.savez(args.output / "best_trace.npz", qpos=np.stack(states), time_s=[s.time_s for s in trace.steps])
            (args.output / "best_candidate.json").write_text(json.dumps(result, indent=2) + "\n")
        if len(results) % 15 == 0:
            print(f"Checked {len(results)}/45 engineered candidates; best lift {best_height:.4f} m", flush=True)
    summary = {"candidates": len(results), "successes": sum(r["score"]["success"] for r in results),
               "maximum_lift_m": best_height, "results": results}
    (args.output / "search_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
