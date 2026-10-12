"""Independent 2ms replay audit of existing saved rollout bundles; creates no rollouts.
Usage: python scripts/audit_mug_table_clearance.py --run DIR --report FILE [--original DIR]
The replay uses only each scene.mjb, saved actuator/root commands, and reset contract.
All physics substeps are examined, rather than only 40ms trajectory samples.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def identical(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def key(p):
    meta = json.loads((p / "collection_metadata.json").read_text())
    a = np.load(p / "actions.npz")
    bias = np.asarray(a["episode_bias_rad"], dtype=np.float64).copy()
    # Multiplying random signs by zero produces signed zero bytes. A disabled
    # disturbance is the same control condition for any seed; normalize signs
    # only when pairing references, then compare original array bytes below.
    bias[bias == 0.0] = 0.0
    return (
        meta.get("proposal_sha256"),
        meta.get("geometry_content_hash"),
        hashlib.sha256(bias.tobytes()).hexdigest(),
    )


def compare(old, new):
    out = {}
    for filename in ["actions.npz", "trajectory.npz", "replay_states.npz"]:
        if not (old / filename).exists() or not (new / filename).exists():
            continue
        a = np.load(old / filename)
        b = np.load(new / filename)
        same = set(a.files) == set(b.files)
        bad = [k for k in set(a.files) & set(b.files) if not identical(a[k], b[k])]
        out[filename] = dict(
            file_bytes_identical=sha(old / filename) == sha(new / filename),
            all_arrays_bitwise_identical=same and not bad,
            mismatched_arrays=sorted(bad),
            field_names_identical=same,
        )
    return out


def table_contacts(m, d, hand_ids, table):
    rows = []
    for i in range(d.ncon):
        c = d.contact[i]
        pair = [int(c.geom1), int(c.geom2)]
        if table not in pair:
            continue
        g = pair[1] if pair[0] == table else pair[0]
        body = int(m.geom_bodyid[g])
        if body not in hand_ids:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body)
        part = (
            "forearm"
            if name == "forearm"
            else "wrist"
            if name == "wrist"
            else "palm"
            if name == "palm"
            else "finger"
        )
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        rows.append(
            dict(
                body=name,
                part=part,
                geom_id=g,
                distance_m=float(c.dist),
                normal_force_n=float(f[0]),
                position_m=c.pos.tolist(),
            )
        )
    return rows


def audit(p):
    a = np.load(p / "actions.npz")
    t = np.load(p / "trajectory.npz")
    case = json.loads((p / "resolved_case.json").read_text())["case"]
    result = json.loads((p / "result.json").read_text())
    m = mujoco.MjModel.from_binary_path(str(p / "scene.mjb"))
    d = mujoco.MjData(m)

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
    fore = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "forearm")
    hand_ids = set()
    for b in range(m.nbody):
        cur = b
        while cur > 0:
            if cur == fore:
                hand_ids.add(b)
                break
            cur = int(m.body_parentid[cur])
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
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
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    first = None
    first_approach = None
    contact_counts = {}
    parts = {}
    minimum_dist = {}
    mismatches = []
    state_mismatches = []
    max_qpos = 0
    max_qvel = 0
    all_count = 0
    dt = float(a["control_timestep_s"])
    substeps = int(round(dt / m.opt.timestep))
    expected = substeps * m.opt.timestep
    if abs(expected - dt) > 1e-12:
        raise ValueError("control/physics time mismatch")
    for ix in range(len(t["time_s"])):
        start = d.qpos.copy()
        start[r : r + 7] = previous_root
        goal = start.copy()
        goal[r : r + 7] = np.r_[
            a["root_target_position_m"][ix], a["root_target_quaternion_wxyz"][ix]
        ]
        vel = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, vel, dt, start, goal)
        state = str(t["state"][ix])
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
            all_count += 1
            c = table_contacts(m, d, hand_ids, table)
            if c:
                contact_counts[state] = contact_counts.get(state, 0) + 1
                for x in c:
                    parts[x["part"]] = parts.get(x["part"], 0) + 1
                    minimum_dist[x["part"]] = min(
                        minimum_dist.get(x["part"], 0), x["distance_m"]
                    )
                event = dict(
                    index=ix,
                    substep=sub,
                    state=state,
                    diagnostic_time_s=float(d.time) - m.opt.timestep,
                    endpoint_time_s=float(d.time),
                    contacts=c,
                )
                if first is None:
                    first = event
                if (
                    state
                    in {
                        "reset",
                        "load_object",
                        "retarget",
                        "move_to_pregrasp",
                        "approach",
                    }
                    and first_approach is None
                ):
                    first_approach = event
        qp = identical(d.qpos, a["full_qpos"][ix])
        qv = identical(d.qvel, a["full_qvel"][ix])
        max_qpos = max(max_qpos, float(np.max(np.abs(d.qpos - a["full_qpos"][ix]))))
        max_qvel = max(max_qvel, float(np.max(np.abs(d.qvel - a["full_qvel"][ix]))))
        if not qp or not qv:
            mismatches.append(
                dict(index=ix, state=state, qpos_bitwise=qp, qvel_bitwise=qv)
            )
        integ = np.empty(mujoco.mj_stateSize(m, state_spec))
        mujoco.mj_getState(m, d, integ, state_spec)
        if not identical(integ, a["integration_state"][ix]):
            state_mismatches.append(ix)
        previous_root = goal[r : r + 7].copy()
        previous_hand = a["commanded_target_rad"][ix].copy()
    return dict(
        episode=str(p),
        case_id=case["case_id"],
        object_id=case["object"]["id"],
        noise_amplitude_rad=json.loads(
            (p / "collection_metadata.json").read_text()
        ).get("noise_amplitude_rad"),
        recorded_outcome=dict(
            success=result["success"], failure_type=result["failure_type"]
        ),
        physics_substeps_checked=all_count,
        all_qpos_qvel_bitwise_identical=not mismatches,
        all_integration_states_bitwise_identical=not state_mismatches,
        max_qpos_error=max_qpos,
        max_qvel_error=max_qvel,
        state_mismatches=state_mismatches,
        qpos_qvel_mismatches=mismatches,
        first_any_hand_table_contact=first,
        first_early_hand_table_contact=first_approach,
        table_contact_substeps_by_phase=contact_counts,
        table_contact_constraints_by_part=parts,
        minimum_signed_contact_distance_m_by_part=minimum_dist,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--original", type=Path)
    ap.add_argument("--episodes", nargs="*", help="Optional episode directory names")
    args = ap.parse_args()
    if not args.run.is_dir():
        raise SystemExit("Run directory absent; no audit performed")
    candidates = sorted(
        p.parent
        for p in args.run.rglob("actions.npz")
        if (p.parent / "trajectory.npz").exists()
        and (p.parent / "scene.mjb").exists()
        and (p.parent / "resolved_case.json").exists()
        and (p.parent / "result.json").exists()
    )
    if args.episodes:
        candidates = [p for p in candidates if p.name in args.episodes]
    refs = {}
    if args.original:
        for p in args.original.glob("episode_*"):
            if (p / "actions.npz").exists():
                refs[key(p)] = p
    rows = []
    for p in candidates:
        row = audit(p)
        old = refs.get(key(p))
        if old:
            row["original_reference"] = str(old)
            row["original_comparison"] = compare(old, p)
        rows.append(row)
        print(
            json.dumps(
                dict(
                    episode=p.name,
                    case=row["case_id"],
                    object=row["object_id"],
                    replay_bitwise=row["all_qpos_qvel_bitwise_identical"],
                    full_state_bitwise=row["all_integration_states_bitwise_identical"],
                    early_collision=row["first_early_hand_table_contact"] is not None,
                    contact_phases=row["table_contact_substeps_by_phase"],
                ),
                sort_keys=True,
            ),
            flush=True,
        )
    payload = dict(
        mujoco_version=mujoco.__version__,
        run=str(args.run),
        episodes_checked=len(rows),
        physics_substeps_checked=sum(r["physics_substeps_checked"] for r in rows),
        all_replays_bitwise=bool(rows)
        and all(
            r["all_qpos_qvel_bitwise_identical"]
            and r["all_integration_states_bitwise_identical"]
            for r in rows
        ),
        episodes_with_early_table_contact=sum(
            r["first_early_hand_table_contact"] is not None for r in rows
        ),
        episodes_with_any_table_contact=sum(
            r["first_any_hand_table_contact"] is not None for r in rows
        ),
        episodes=rows,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2))
    print("REPORT", args.report)
    if not payload["all_replays_bitwise"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
