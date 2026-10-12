"""Independently check a saved first wrist pair; never collect or fit a model.

The prediction reconstruction uses only the declared closing input whitelist.
Full-state forward kinematics and command replay are retrospective diagnostics,
kept separate from those predictor inputs. Existing simulator states may be
replayed to check reproducibility; no additional grasp attempt is generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np


MODEL_SHA256 = "16c9941a51130daea818c6ad227ef1e3a02e4be9ed1c7aa5a8e8f3e023428beb"
ARM_NAMES = ("A", "B")
TRAJECTORY_INPUT_KEYS = {"time_s", "state", "hand_qpos"}
ACTION_INPUT_KEYS = {"time_s", "joint_names", "hand_qvel", "commanded_target_rad"}
EARLY_PHASES = {"reset", "load_object", "retarget", "move_to_pregrasp", "approach"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: str) -> str:
    value = value.removeprefix("sha256:")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid SHA256 binding")
    return value


def json_read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def bound_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("source paths must remain relative to their declared root")
    result = (root / path).resolve()
    if root != result and root not in result.parents:
        raise ValueError("source binding escapes its root")
    return result


def assert_close(actual, expected, name: str, tolerance=1e-12) -> None:
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=tolerance):
        raise ValueError(f"independent check differs: {name}")


def prefix_arrays(episode: Path) -> tuple[np.ndarray, dict, dict]:
    """Build the float32 model input without prediction-library imports."""
    with np.load(episode / "trajectory.npz", allow_pickle=False) as t, \
            np.load(episode / "actions.npz", allow_pickle=False) as a:
        times = t["time_s"]
        if (times.ndim != 1 or not np.all(np.isfinite(times))
                or np.any(np.diff(times) <= 0)
                or not np.array_equal(times, a["time_s"])):
            raise ValueError("raw action/trajectory time grids differ")
        eligible = np.flatnonzero(times <= 2.96 + 1e-9)
        if len(eligible) < 20:
            raise ValueError("complete recording lacks the closing history")
        selected = eligible[-20:]
        assert_close(times[selected], np.arange(20) * 0.04 + 2.20,
                     "closing prediction times", 1e-9)
        if (not np.all(t["state"][selected] == "close_fingers")
                or np.any(np.isin(t["state"][eligible], ["lift", "hold", "score"]))):
            raise ValueError("model prefix includes an incorrect or future phase")
        if np.any(t["time_s"][t["state"] == "lift"] < 3.0 - 1e-9):
            raise ValueError("lift began before the frozen contract")
        names = a["joint_names"]
        if names.shape != (24,) or len(set(names.tolist())) != 24:
            raise ValueError("invalid joint identity/order")
        raw_blocks = (t["hand_qpos"], a["hand_qvel"], a["commanded_target_rad"])
        if any(x.shape != (len(times), 24) for x in raw_blocks):
            raise ValueError("raw predictor fields lack aligned 24-joint samples")
        blocks = [x[selected].astype(np.float32) for x in raw_blocks]
        if not all(np.all(np.isfinite(x)) for x in blocks):
            raise ValueError("nonfinite observable model input")
        trajectory = {key: t[key][selected].copy() for key in TRAJECTORY_INPUT_KEYS}
        actions = {key: (a[key].copy() if key == "joint_names" else a[key][selected].copy())
                   for key in ACTION_INPUT_KEYS}
    return np.concatenate(blocks, axis=1).reshape(-1), trajectory, actions


def summarize(X: np.ndarray) -> np.ndarray:
    history = np.asarray(X, dtype=np.float64).reshape(-1, 20, 72)
    return np.concatenate([history[:, -1], np.mean(history, axis=1),
                           np.std(history, axis=1, ddof=0),
                           history[:, -1] - history[:, 0]], axis=1)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def descendant(model, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def rotation_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Geodesic angle between same-shaped arrays of 3x3 rotation matrices."""
    relative = np.matmul(first, np.swapaxes(second, -1, -2))
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def morphology(episode: Path) -> dict:
    """Forward geometry from saved actual endpoint states, without mj_step."""
    model = mujoco.MjModel.from_binary_path(str(episode / "scene.mjb"))
    data = mujoco.MjData(model)
    palm = model.body("palm").id
    fingers = [g for g in range(model.ngeom)
               if int(model.geom_bodyid[g]) != palm
               and descendant(model, int(model.geom_bodyid[g]), palm)]
    geometry_identity = [dict(geom_id=g,
                             geom_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
                             body_name=model.body(int(model.geom_bodyid[g])).name)
                         for g in fingers]
    with np.load(episode / "actions.npz", allow_pickle=False) as actions, \
            np.load(episode / "trajectory.npz", allow_pickle=False) as trajectory:
        hand = [model.jnt_qposadr[model.joint(str(name)).id] for name in actions["joint_names"]]
        palm_positions, palm_rotations, finger_positions, finger_rotations = [], [], [], []
        for qpos in actions["full_qpos"]:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            palm_positions.append(data.xpos[palm].copy())
            palm_rotations.append(data.xmat[palm].reshape(3, 3).copy())
            finger_positions.append(data.geom_xpos[fingers].copy())
            finger_rotations.append(data.geom_xmat[fingers].reshape(-1, 3, 3).copy())
        actual_joint = actions["full_qpos"][:, hand].copy()
        assert_close(actual_joint, trajectory["hand_qpos"], "raw hand encoder reconstruction", 0.0)
        values = dict(time_s=trajectory["time_s"].copy(), state=trajectory["state"].copy(),
                      palm_position_m=np.asarray(palm_positions),
                      palm_rotation=np.asarray(palm_rotations),
                      finger_geom_position_m=np.asarray(finger_positions),
                      finger_geom_rotation=np.asarray(finger_rotations),
                      hand_qpos=actual_joint, commanded_target_rad=actions["commanded_target_rad"].copy())
    values["finger_geometry_identity"] = geometry_identity
    return values


def stopped_geometry(episode: Path, metadata: dict) -> dict | None:
    """Check the saved guard endpoint without assigning a task outcome."""
    state_path = episode / "runtime_guard_state.npz"
    scene_path = episode / "scene.mjb"
    if not state_path.exists() or not scene_path.exists():
        return None
    model = mujoco.MjModel.from_binary_path(str(scene_path))
    data = mujoco.MjData(model)
    with np.load(state_path, allow_pickle=False) as state:
        data.qpos[:] = state["qpos"]
        endpoint_time = float(state["time_s"])
    mujoco.mj_forward(model, data)
    table, forearm = model.geom("table").id, model.body("forearm").id
    table_top = data.geom_xpos[table, 2] + model.geom_size[table, 2]
    clearances = []
    for geom in range(model.ngeom):
        if (not descendant(model, int(model.geom_bodyid[geom]), forearm)
                or not (model.geom_contype[geom] or model.geom_conaffinity[geom])):
            continue
        kind, size = int(model.geom_type[geom]), model.geom_size[geom]
        zrow = data.geom_xmat[geom].reshape(3, 3)[2]
        center = float(data.geom_xpos[geom, 2])
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            lowest = center + float(np.min(model.mesh_vert[start:start + count] @ zrow))
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            lowest = center - float(np.abs(zrow) @ size)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            lowest = center - float(size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            lowest = center - float(size[0] + abs(zrow[2]) * size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            lowest = center - float(size[0] * np.linalg.norm(zrow[:2]) + abs(zrow[2]) * size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            lowest = center - float(np.linalg.norm(zrow * size))
        else:
            raise ValueError("unsupported active hand geometry in stopped-state audit")
        clearances.append((lowest - table_top, geom))
    minimum, geom = min(clearances)
    event = metadata["runtime_guard_event"]
    assert_close(endpoint_time, event["time_s"], "stopped guard time")
    assert_close(minimum, event["actual_table_clearance_m"], "stopped guard clearance")
    if minimum >= event["required_table_clearance_m"] or event["geom_id"] != geom:
        raise ValueError("saved guard endpoint does not reproduce the declared violation")
    return dict(time_s=endpoint_time, phase=event["phase"], geom_id=geom,
                actual_table_clearance_m=float(minimum), label_success=None, p_failure=None,
                scope="Stopped-state geometry only; no completed outcome or predictor score")


def tracking_comparison(first: dict, second: dict) -> tuple[dict, dict]:
    if (first["finger_geometry_identity"] != second["finger_geometry_identity"]
            or not np.array_equal(first["time_s"], second["time_s"])
            or not np.array_equal(first["state"], second["state"])):
        raise ValueError("paired geometry/phase/time grids differ")
    palm_position = np.linalg.norm(first["palm_position_m"] - second["palm_position_m"], axis=-1)
    palm_rotation = rotation_distance(first["palm_rotation"], second["palm_rotation"])
    fingers_position = np.linalg.norm(first["finger_geom_position_m"] - second["finger_geom_position_m"], axis=-1)
    fingers_rotation = rotation_distance(first["finger_geom_rotation"], second["finger_geom_rotation"])
    errors = dict(palm_position_difference_m=palm_position,
                  palm_orientation_difference_rad=palm_rotation,
                  finger_geom_position_difference_m=fingers_position,
                  finger_geom_orientation_difference_rad=fingers_rotation)
    phases = list(dict.fromkeys(first["state"].tolist()))
    summaries = []
    for phase in ["all_phases", "prediction_prefix", *phases]:
        selected = (np.ones(len(palm_position), dtype=bool) if phase == "all_phases"
                    else ((first["time_s"] >= 2.20 - 1e-9) & (first["time_s"] <= 2.96 + 1e-9))
                    if phase == "prediction_prefix" else first["state"] == phase)
        row = dict(phase=phase, saved_endpoints=int(selected.sum()))
        for field, values in errors.items():
            subset = values[selected]
            row[field] = dict(maximum=float(np.max(subset)),
                              rms=float(np.sqrt(np.mean(subset ** 2))))
        for arm, data in zip(ARM_NAMES, (first, second), strict=True):
            tracking = data["hand_qpos"][selected, :2] - data["commanded_target_rad"][selected, :2]
            row[f"{arm}_wrist_qpos_minus_command_rad"] = {
                name: dict(maximum_absolute=float(np.max(np.abs(tracking[:, j]))),
                           rms=float(np.sqrt(np.mean(tracking[:, j] ** 2))))
                for j, name in enumerate(("WRJ2", "WRJ1"))}
        summaries.append(row)
    saved = dict(time_s=first["time_s"], state=first["state"], **errors)
    for arm, data in zip(ARM_NAMES, (first, second), strict=True):
        saved.update({f"{arm}_{key}": value for key, value in data.items()
                      if isinstance(value, np.ndarray)})
    return dict(comparisons_by_phase=summaries,
                finger_geometry_identity=first["finger_geometry_identity"],
                scope="Fresh forward kinematics at saved actual 40 ms endpoints; not identical dynamics or a geometry-only causal intervention"), saved


def audit(runs_root: Path, pose_manifest: Path, scores: Path, output: Path) -> dict:
    repository = Path(__file__).resolve().parents[1]
    runs_root, pose_manifest, scores, output = [p.resolve() for p in (runs_root, pose_manifest, scores, output)]
    if output.exists():
        raise FileExistsError("independent audit output must be a new directory")
    runtime_assets = next((p for p in runs_root.parents if (p / "runs").is_dir()), None)
    if runtime_assets is None:
        raise ValueError("cannot identify the declared runtime-assets root")
    report = json_read(scores / "report.json")
    plan = json_read(runs_root / "plan.json")
    manifest = json_read(pose_manifest)
    if (report["schema_version"] != "frozen_wrist_pair_model_diagnostic_v1"
            or manifest["schema_version"] != "explicit_wrist_pair_pose_manifest_v1"):
        raise ValueError("unsupported diagnostic/pose manifest schema")
    manifest_sha = sha(pose_manifest)
    if (digest(report["pose_manifest_sha256"]) != manifest_sha
            or sha(scores / "pose_manifest.json") != manifest_sha
            or sha(scores / "plan.json") != sha(runs_root / "plan.json")
            or report["prediction_time_s"] != 2.96
            or report["failure_alert_threshold"] != 0.5):
        raise ValueError("scoring plan, cutoff or manifest binding changed")
    contract = manifest["model_contract"]
    if (contract["history_steps"] != 20 or contract["prediction_time_s"] != 2.96
            or contract["failure_alert_threshold"] != 0.5
            or contract["fields"] != ["hand_qpos", "hand_qvel", "commanded_target_rad"]
            or not contract["weights_normalization_features_unchanged"]
            or not contract["outcomes_not_inputs"]):
        raise ValueError("frozen model contract differs")
    pair_id = report["pair_id"]
    pair = [row for row in manifest["cases"] if row["pair_id"] == pair_id]
    if len(pair) != 1 or not pair[0]["eligible"]:
        raise ValueError("scored pair was not eligible in the frozen manifest")
    pair = pair[0]
    sources = {pose_manifest: manifest_sha, runs_root / "plan.json": sha(runs_root / "plan.json"),
               Path(__file__).resolve(): sha(Path(__file__).resolve())}
    for key, expected in manifest["source_sha256"].items():
        kind, relative = key.split(":", 1)
        root = {"repository": repository, "runtime_assets": runtime_assets}[kind]
        path = bound_path(root, relative)
        if sha(path) != digest(expected):
            raise ValueError(f"pose-manifest source changed: {key}")
        sources[path] = sha(path)
    for key, expected in report["frozen_file_sha256"].items():
        path = bound_path(repository, key)
        if sha(path) != digest(expected):
            raise ValueError(f"frozen scoring source changed: {key}")
        sources[path] = sha(path)
    for key, expected in report["raw_source_sha256"].items():
        path = bound_path(runs_root, key)
        if sha(path) != digest(expected):
            raise ValueError(f"recording changed after scoring: {key}")
        sources[path] = sha(path)
    for field in ("frozen_files_sha256", "paired_source_sha256"):
        for key, expected in plan[field].items():
            path = bound_path(repository, key)
            if sha(path) != digest(expected):
                raise ValueError(f"pre-execution source changed: {key}")
            sources[path] = sha(path)
    for path in sorted(runs_root.rglob("*")) + sorted(scores.rglob("*")):
        if path.is_file():
            sources[path] = sha(path)
    model_path = bound_path(repository, contract["model_file"])
    if sha(model_path) != MODEL_SHA256 or digest(contract["model_sha256"]) != MODEL_SHA256:
        raise ValueError("original model bytes changed")
    sources[model_path] = sha(model_path)
    with np.load(model_path, allow_pickle=False) as model:
        mean, scale = model["mean"].copy(), model["scale"].copy()
        active, weights = model["active"].copy(), model["weights"].copy()
        intercept = float(model["intercept"])
    wrist_mask = np.arange(288) % 24 < 2
    if (mean.shape != (288,) or scale.shape != (288,) or active.shape != (288,)
            or active.dtype != np.bool_ or active.sum() != 269 or weights.shape != (269,)
            or np.sum(active & wrist_mask) != 20 or np.any(scale <= 0)):
        raise ValueError("frozen model summary dimensions changed")
    pilot = repository / "research_artifacts/object_scene_pilot_v1"
    split_path, dataset_path = pilot / "splits.json", pilot / "dataset.npz"
    train_ids = json_read(split_path)["partitions"]["train"]["episode_ids"]
    with np.load(dataset_path, allow_pickle=False) as dataset:
        lookup = {str(value): i for i, value in enumerate(dataset["episode_ids"])}
        train_features = summarize(dataset["X"][[lookup[value] for value in train_ids]])
    if len(train_features) != 520 or len(set(train_ids)) != 520:
        raise ValueError("training-only normalization inventory changed")
    manual_mean, manual_std = train_features.mean(axis=0), train_features.std(axis=0)
    manual_active = manual_std > 1e-8
    if (not np.array_equal(manual_mean, mean) or not np.array_equal(manual_active, active)
            or not np.array_equal(np.where(manual_active, manual_std, 1.0), scale)):
        raise ValueError("independent original training normalization differs")
    sources.update({split_path: sha(split_path), dataset_path: sha(dataset_path)})
    training_min, training_max = train_features.min(axis=0), train_features.max(axis=0)
    scored = {r["posture"]: r for r in report["records"]}
    excluded = {r["posture"]: r for r in report["excluded"]}
    if (len(scored) != len(report["records"]) or len(excluded) != len(report["excluded"])
            or set(scored) & set(excluded) or set(scored) | set(excluded) != set(ARM_NAMES)):
        raise ValueError("every first-pair arm must occur once, including stops")
    independent_rows, geometries, dynamic, stopped_checks = [], {}, [], []
    raw_commands = {}
    for arm in ARM_NAMES:
        episode = runs_root / arm
        metadata = json_read(episode / "collection_metadata.json")
        if (metadata["pair_id"] != pair_id or metadata["posture"] != arm
                or digest(metadata["frozen_pose_manifest_sha256"]) != manifest_sha
                or metadata["noise_amplitude_rad"] != 0.0
                or metadata["noise_mode"] != "disabled"
                or metadata["predictor_evaluated_by_executor"] is not False
                or metadata["original_case_sha256"].removeprefix("sha256:") != digest(pair["case_sha256"])):
            raise ValueError("execution identity, model isolation or zero-noise condition differs")
        for key, expected in pair["source_binding"].items():
            if metadata.get(key) != expected:
                raise ValueError(f"recorded source differs from the manifest: {key}")
        assert_close(metadata["intended_world_palm_position_m"], pair["intended_world_palm_position_m"], "palm intention")
        assert_close(metadata["intended_world_palm_quaternion_wxyz"], pair["intended_world_palm_quaternion_wxyz"], "palm orientation intention")
        wrist = pair["poses"][arm]["wrist_rad"]
        assert_close(metadata["execution_protocol"]["explicit_wrist_target_rad"], wrist, "declared treatment", 0.0)
        certificate = metadata.get("frozen_pose_certificate")
        if certificate is not None and (certificate["explicit_angles_unchanged"] is not True
                                       or certificate["automatic_substitution"] is not False):
            raise ValueError("executor substituted the declared treatment")
        actions_path = episode / "actions.npz"
        if actions_path.exists():
            with np.load(actions_path, allow_pickle=False) as actions:
                if any(np.any(actions[key] != 0.0) for key in ("episode_bias_rad", "requested_noise_rad", "applied_noise_rad")):
                    raise ValueError("nonzero applied disturbance in a zero-noise pair")
                assert_close(actions["commanded_target_rad"][:, :2], np.tile(wrist, (len(actions["time_s"]), 1)), "exact wrist commands", 0.0)
                raw_commands[arm] = actions["commanded_target_rad"].copy()
        if metadata["status"] != "completed":
            row = excluded.get(arm)
            if (metadata["label"] is not None or row is None or row["label_success"] is not None
                    or row["p_failure"] is not None or (episode / "result.json").exists()
                    or (scores / "prediction_inputs" / arm).exists()):
                raise ValueError("stopped execution received a label or model score")
            independent_rows.append(dict(posture=arm, status=metadata["status"], label_success=None,
                                         p_failure=None, scope="Stopped arm excluded from model metrics and paired completed-motion comparison"))
            stopped_checks.append(dict(posture=arm, saved_guard_endpoint=stopped_geometry(episode, metadata)))
            continue
        if arm not in scored or not certificate:
            raise ValueError("completed arm lacks a score or geometry certificate")
        prefix, trajectory_input, action_input = prefix_arrays(episode)
        if action_input["joint_names"].tolist() != contract["joint_names"]:
            raise ValueError("prediction joint order changed")
        portable = scores / "prediction_inputs" / arm
        for filename, expected, whitelist in (("trajectory.npz", trajectory_input, TRAJECTORY_INPUT_KEYS),
                                              ("actions.npz", action_input, ACTION_INPUT_KEYS)):
            with np.load(portable / filename, allow_pickle=False) as values:
                if set(values.files) != whitelist:
                    raise ValueError("portable prefix includes a privileged field")
                if any(not np.array_equal(values[key], array) for key, array in expected.items()):
                    raise ValueError("portable prediction inputs differ from raw pre-lift samples")
        features = summarize(prefix[None])[0]
        standardized = (features - mean) / scale
        contributions = np.zeros(288)
        contributions[active] = standardized[active] * weights
        wrist_term = float(contributions[wrist_mask].sum())
        other_term = float(contributions[~wrist_mask].sum())
        logit = float(np.dot(standardized[active], weights) + intercept)
        p_success = sigmoid(logit)
        calculated = dict(success_logit=logit, wrist_contribution=wrist_term,
                          nonwrist_contribution=other_term, intercept=intercept,
                          p_success=p_success, p_failure=1.0 - p_success)
        assert_close(logit, wrist_term + other_term + intercept, "independent attribution sum")
        for key, value in calculated.items():
            assert_close(value, scored[arm][key], f"{arm} independently reconstructed {key}")
        result = json_read(episode / "result.json")
        label = int(result["success"])
        if type(result["success"]) is not bool or metadata["label"] != label or scored[arm]["label_success"] != label:
            raise ValueError("recorded task label differs")
        alert = calculated["p_failure"] >= 0.5
        if scored[arm]["failure_alert"] != alert:
            raise ValueError("threshold alert differs")
        independent_rows.append(dict(posture=arm, status="completed", label_success=label,
                                     failure_alert=alert, **calculated,
                                     active_wrist_features_outside_training_range=int(np.sum(
                                         active & wrist_mask & ((features < training_min) | (features > training_max)))),
                                     prefix_start_s=2.20, observed_through_s=2.96))
        geometries[arm] = morphology(episode)
        # This module predates the paired executor and reproduces dynamics from
        # raw scene/reset/commands, rather than invoking a rollout runner.
        from audit_mug_table_clearance import audit as replay_audit
        replay = replay_audit(episode)
        if (not replay["all_qpos_qvel_bitwise_identical"]
                or not replay["all_integration_states_bitwise_identical"]
                or replay["first_any_hand_table_contact"] is not None):
            raise ValueError("independent saved-command replay differs or touches the table")
        dynamic.append(dict(posture=arm, **replay))
    tracking, arrays = None, None
    finger_commands_equal = None
    if set(raw_commands) == set(ARM_NAMES):
        common_endpoints = min(len(raw_commands["A"]), len(raw_commands["B"]))
        finger_commands_equal = np.array_equal(raw_commands["A"][:common_endpoints, 2:], raw_commands["B"][:common_endpoints, 2:])
        if not finger_commands_equal:
            raise ValueError("the 22 finger commands differ between treatments")
    if set(geometries) == set(ARM_NAMES):
        if not np.array_equal(raw_commands["A"][:, 2:], raw_commands["B"][:, 2:]):
            raise ValueError("the 22 finger commands differ between treatments")
        tracking, arrays = tracking_comparison(geometries["A"], geometries["B"])
    completed = [r for r in independent_rows if r["status"] == "completed"]
    confusion = dict(true_positive=0, false_positive=0, true_negative=0, false_negative=0)
    for row in completed:
        key = ("false_positive" if row["label_success"] and row["failure_alert"] else
               "true_negative" if row["label_success"] else
               "true_positive" if row["failure_alert"] else "false_negative")
        confusion[key] += 1
    if any(sha(path) != before for path, before in sources.items()):
        raise ValueError("an original source, score or recording changed during independent audit")
    output.mkdir(parents=True)
    if arrays is not None:
        np.savez_compressed(output / "morphology_tracking.npz", **arrays)
    result = dict(schema_version="independent_wrist_pair_audit_v1", pair_id=pair_id,
                  pose_manifest_sha256=manifest_sha, independent_records=independent_rows,
                  failure_confusion=confusion, completed=len(completed), stopped_unscored=len(excluded),
                  training_rows=520, active_features=269, active_wrist_features=20,
                  training_only_normalization_bitwise_reproduced=True,
                  portable_prefix_whitelist_and_raw_equivalence_verified=True,
                  unchanged_model_sha256=MODEL_SHA256, all_available_saved_commands_zero_noise_verified=True,
                  identical_22_applied_finger_commands_verified=finger_commands_equal,
                  actual_motion_tracking=tracking, independent_dynamic_replays=dynamic,
                  stopped_guard_endpoint_checks=stopped_checks,
                  source_sha256={str(path): value for path, value in sorted(sources.items())},
                  original_sources_unchanged=True,
                  scope="One predeclared development pair; no additional collection, fitting, metric tuning, familiarity claim or generalization estimate")
    (output / "independent_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pose-manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.runs_root, args.pose_manifest, args.scores, args.output)
    print(json.dumps({key: result[key] for key in ("pair_id", "completed", "stopped_unscored", "failure_confusion", "independent_records")}, indent=2))


if __name__ == "__main__":
    main()
