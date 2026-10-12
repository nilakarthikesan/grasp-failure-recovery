"""Independently reproduce a guard rejection using matching saved commands.

The rejected bundle has no completed trajectory or scene model. Supply the
matching unguarded rollout as --reference; its geometry and command provenance
must agree. Every 2 ms step is checked until the first runtime margin crossing.
The tool writes its report and exits 2 when any promised validation fails.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def validate(rejected, reference):
    meta = json.loads((rejected / "collection_metadata.json").read_text())
    old = json.loads((reference / "collection_metadata.json").read_text())
    guard = meta["runtime_guard_event"]
    snapshot = np.load(rejected / "runtime_guard_state.npz")
    a = np.load(reference / "actions.npz")
    t = np.load(reference / "trajectory.npz")
    case = json.loads((reference / "resolved_case.json").read_text())["case"]
    keys = [
        "geometry_content_hash",
        "proposal_sha256",
        "camera_sha256",
        "object_position_m",
        "object_orientation_xyzw",
        "noise_seed",
        "noise_amplitude_rad",
        "intended_world_palm_position_m",
        "intended_world_palm_quaternion_wxyz",
        "clearance_plan",
    ]
    mismatches = [k for k in keys if meta.get(k) != old.get(k)]
    if mismatches:
        raise ValueError("Rejection reference differs: " + ",".join(mismatches))
    m = mujoco.MjModel.from_binary_path(str(reference / "scene.mjb"))
    d = mujoco.MjData(m)
    scratch = mujoco.MjData(m)

    def jid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    r = int(m.jnt_qposadr[jid("world_joint")])
    obj = int(m.jnt_qposadr[jid("object_free")])
    ha = np.array([m.jnt_qposadr[jid(str(n))] for n in a["joint_names"]])
    act = np.array(
        [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_" + str(n))
            for n in a["joint_names"]
        ]
    )
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
    fore = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "forearm")
    if table < 0 or fore < 0:
        raise ValueError("Reference model requires named table and forearm")
    hand = set()
    for b in range(m.nbody):
        cur = b
        while cur > 0:
            if cur == fore:
                hand.add(b)
                break
            cur = int(m.body_parentid[cur])
    active = [
        int(g)
        for g in range(m.ngeom)
        if int(m.geom_bodyid[g]) in hand
        and (m.geom_contype[g] or m.geom_conaffinity[g])
    ]

    def table_top(data):
        vertical_axis = data.geom_xmat[table].reshape(3, 3)[2]
        if not np.allclose(np.abs(vertical_axis), [0.0, 0.0, 1.0], atol=1e-10):
            raise ValueError("This audit supports a horizontal table top")
        if m.geom_type[table] == mujoco.mjtGeom.mjGEOM_BOX:
            return float(data.geom_xpos[table, 2] + m.geom_size[table, 2])
        if m.geom_type[table] == mujoco.mjtGeom.mjGEOM_PLANE:
            return float(data.geom_xpos[table, 2])
        raise ValueError("Reference table must be a horizontal box or plane")

    def clearance(data):
        top = table_top(data)
        rows = []
        for g in active:
            z = float(data.geom_xpos[g, 2])
            rot = data.geom_xmat[g].reshape(3, 3)[2]
            size = m.geom_size[g]
            typ = int(m.geom_type[g])
            if typ == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[g])
                v = m.mesh_vert[
                    m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]
                ]
                low = float(np.min(np.sum(v * rot, axis=1) + z))
            elif typ == mujoco.mjtGeom.mjGEOM_BOX:
                low = z - float(np.abs(rot) @ size)
            elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
                low = z - size[0]
            elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
                low = z - size[0] - size[1] * abs(rot[2])
            elif typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
                low = (
                    z
                    - size[0] * np.sqrt(rot[0] ** 2 + rot[1] ** 2)
                    - size[1] * abs(rot[2])
                )
            elif typ == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                low = z - np.sqrt(np.sum((rot * size) ** 2))
            else:
                raise ValueError("Unsupported hand geom " + str(typ))
            if not np.isfinite(low):
                raise ValueError("Reference geometry has nonfinite support height")
            rows.append((float(low) - top, g))
        return min(rows)

    def contacts(data):
        rows = []
        for i in range(data.ncon):
            c = data.contact[i]
            pair = [int(c.geom1), int(c.geom2)]
            if table not in pair:
                continue
            g = pair[1] if pair[0] == table else pair[0]
            if int(m.geom_bodyid[g]) in hand:
                rows.append(dict(geom_id=g, distance_m=float(c.dist)))
        return rows

    reset = case["initial_condition"]
    d.qpos[obj : obj + 3] = reset["object_position_m"]
    d.qpos[obj + 3 : obj + 7] = np.asarray(reset["object_orientation_xyzw"])[
        [3, 0, 1, 2]
    ]
    d.qpos[r : r + 7] = np.r_[
        a["root_target_position_m"][0], a["root_target_quaternion_wxyz"][0]
    ]
    d.qpos[ha] = a["commanded_target_rad"][0]
    d.qvel[:] = 0
    d.mocap_pos[0] = a["root_target_position_m"][0]
    d.mocap_quat[0] = a["root_target_quaternion_wxyz"][0]
    d.ctrl[act] = a["commanded_target_rad"][0]
    mujoco.mj_forward(m, d)
    previous_root = d.qpos[r : r + 7].copy()
    previous_hand = a["commanded_target_rad"][0].copy()
    dt = float(a["control_timestep_s"])
    substeps = int(round(dt / m.opt.timestep))
    if substeps < 1 or not np.isclose(substeps * m.opt.timestep, dt):
        raise ValueError("Control interval must contain whole physics substeps")
    minimum_before, _ = clearance(d)
    prior = dict(time_s=float(d.time), clearance_m=minimum_before)
    first = None
    contact_count = 0
    count = 0
    for ix in range(len(t["time_s"])):
        start = d.qpos.copy()
        start[r : r + 7] = previous_root
        goal = start.copy()
        goal[r : r + 7] = np.r_[
            a["root_target_position_m"][ix], a["root_target_quaternion_wxyz"][ix]
        ]
        vel = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, vel, dt, start, goal)
        for sub in range(1, substeps + 1):
            alpha = sub / substeps
            command = start.copy()
            mujoco.mj_integratePos(m, command, vel, alpha * dt)
            d.mocap_pos[0] = command[r : r + 3]
            d.mocap_quat[0] = command[r + 3 : r + 7]
            d.ctrl[act] = (1 - alpha) * previous_hand + alpha * a[
                "commanded_target_rad"
            ][ix]
            mujoco.mj_step(m, d)
            count += 1
            contact_count += len(contacts(d))
            scratch.qpos[:] = d.qpos
            mujoco.mj_kinematics(m, scratch)
            low, g = clearance(scratch)
            if low < float(guard["required_table_clearance_m"]):
                first = dict(
                    time_s=float(d.time),
                    clearance_m=low,
                    geom_id=g,
                    body=mujoco.mj_id2name(
                        m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])
                    ),
                    phase=str(t["state"][ix]),
                    qpos_bitwise_equal_to_rejection=d.qpos.tobytes()
                    == snapshot["qpos"].tobytes(),
                    qvel_bitwise_equal_to_rejection=d.qvel.tobytes()
                    == snapshot["qvel"].tobytes(),
                    time_equal_to_rejection=float(d.time) == float(snapshot["time_s"]),
                    preceding_substep=prior,
                    minimum_prior_clearance_m=float(minimum_before),
                )
                break
            minimum_before = min(minimum_before, low)
            prior = dict(time_s=float(d.time), clearance_m=low)
        if first:
            break
        previous_root = goal[r : r + 7].copy()
        previous_hand = a["commanded_target_rad"][ix].copy()
    scratch.qpos[:] = snapshot["qpos"]
    scratch.qvel[:] = snapshot["qvel"]
    scratch.time = float(snapshot["time_s"])
    mujoco.mj_forward(m, scratch)
    low, g = clearance(scratch)
    forward_contacts = contacts(scratch)
    report = dict(
        rejected_episode=str(rejected),
        geometry_model_reference=str(reference / "scene.mjb"),
        geometry_model_sha256=hashlib.sha256(
            (reference / "scene.mjb").read_bytes()
        ).hexdigest(),
        matching_geometry_and_command_metadata=not mismatches,
        status=meta["status"],
        label_is_null="label" in meta and meta["label"] is None,
        result_json_absent=not (rejected / "result.json").exists(),
        trajectory_npz_absent=not (rejected / "trajectory.npz").exists(),
        physics_substeps_to_guard=count,
        hand_table_contact_constraints_through_guard=contact_count,
        first_margin_violation=first,
        forward_geometry_clearance_m=low,
        forward_geometry_clearance_matches_metadata=low
        == guard["actual_table_clearance_m"],
        forward_geometry_geom_id=g,
        forward_hand_table_contacts=forward_contacts,
        table_top_world_z_m=table_top(scratch),
        guard_event=guard,
    )
    checks = dict(
        matching_reference=report["matching_geometry_and_command_metadata"],
        runtime_rejection_status=meta["status"] == "runtime_clearance_rejected",
        no_label=report["label_is_null"],
        no_completed_result=report["result_json_absent"],
        no_completed_trajectory=report["trajectory_npz_absent"],
        no_contact_before_guard=contact_count == 0,
        no_contact_at_saved_state=not forward_contacts,
        saved_geometry_above_table=low > 0.0,
        saved_clearance_matches_metadata=report[
            "forward_geometry_clearance_matches_metadata"
        ],
        saved_geom_matches_metadata=g == guard["geom_id"],
        crossing_found=first is not None,
    )
    if first is not None:
        checks.update(
            first_margin_crossing=first["minimum_prior_clearance_m"]
            >= guard["required_table_clearance_m"]
            and 0.0 < first["clearance_m"] < guard["required_table_clearance_m"],
            saved_qpos_matches_replay=first["qpos_bitwise_equal_to_rejection"],
            saved_qvel_matches_replay=first["qvel_bitwise_equal_to_rejection"],
            saved_time_matches_replay=first["time_equal_to_rejection"],
            replay_time_matches_metadata=first["time_s"] == guard["time_s"],
            replay_clearance_matches_metadata=first["clearance_m"]
            == guard["actual_table_clearance_m"],
            replay_geom_matches_metadata=first["geom_id"] == guard["geom_id"],
            replay_body_matches_metadata=first["body"] == guard["body"],
            replay_phase_matches_metadata=first["phase"] == guard["phase"],
        )
    report["validation_checks"] = checks
    report["all_validation_checks_passed"] = all(checks.values())
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejected", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()
    try:
        report = validate(args.rejected, args.reference)
    except (ValueError, KeyError, OSError) as exc:
        report = dict(all_validation_checks_passed=False, error=str(exc))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["all_validation_checks_passed"]:
        raise SystemExit(2)
