"""Static clearance audit of observed training wrists on the eight mug proposals.

No simulator time is advanced, predictor evaluated, label generated or model
fitted. MuJoCo calls are restricted to model setup, forward geometry and rigid
transform utilities. Every other exported mj_/mju_ function is blocked while
the audit runs, including mj_step, mj_step1, mj_step2 and integration routines.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from grasp_failure_prediction.evaluation.noisy_rollouts import _prepare_case, _runtime_versions, NoisyShadowRunner
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root, reorder_for_mujoco
from grasp_failure_prediction.evaluation.runner import OPENING_JOINTS
from grasp_failure_prediction.evaluation.table_clearance import (
    _descendant, _lowest_world_z, _rotation,
)


ALLOWED_MUJOCO_CALLS = frozenset({
    "mj_forward", "mj_geomDistance", "mj_name2id", "mj_id2name",
    "mj_resetData", "mj_setConst", "mj_saveLastXML",
    "mju_quat2Mat", "mju_mat2Quat",
})
TABLE_GATE_M = 0.010
PROXIMAL_GATE_M = 0.001
INVARIANCE_TOLERANCE = 1e-10
REPLAY_TOLERANCE = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@contextmanager
def static_mujoco_only():
    originals, counts = {}, {}
    for name in dir(mujoco):
        value = getattr(mujoco, name)
        if not name.startswith(("mj_", "mju_")) or not callable(value):
            continue
        originals[name] = value
        if name in ALLOWED_MUJOCO_CALLS:
            def allowed(*args, _name=name, _function=value, **kwargs):
                counts[_name] = counts.get(_name, 0) + 1
                return _function(*args, **kwargs)
            setattr(mujoco, name, allowed)
        else:
            def blocked(*args, _name=name, **kwargs):
                raise RuntimeError(f"{_name} is forbidden in this static audit")
            setattr(mujoco, name, blocked)
    try:
        # The protection is exercised, rather than asserted from source alone.
        try:
            mujoco.mj_step(None, None)
        except RuntimeError as error:
            if "forbidden" not in str(error):
                raise
        else:
            raise AssertionError("mj_step blocking failed")
        yield counts
    finally:
        for name, function in originals.items():
            setattr(mujoco, name, function)


def static_context(runner, prepared):
    model = runner.model
    names = tuple(runner._hand_joint_names)
    ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                      for name in names])
    target = reorder_for_mujoco(prepared.pose.joint_names, prepared.pose.qpos, names)
    palm = int(runner._palm_body_id)
    hand = tuple(g for g in range(model.ngeom)
                 if int(model.geom_bodyid[g]) in runner._hand_body_ids
                 and (model.geom_contype[g] or model.geom_conaffinity[g]))
    finger_geoms = tuple(g for g in range(model.ngeom)
                         if int(model.geom_bodyid[g]) != palm
                         and _descendant(model, int(model.geom_bodyid[g]), palm))
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = runner.data.qpos
    mujoco.mj_forward(model, scratch)
    table = int(runner._table_geom_id)
    if not np.allclose(scratch.geom_xmat[table].reshape(3, 3), np.eye(3),
                       atol=1e-10, rtol=0):
        raise ValueError("The table must be horizontal and axis aligned")
    return dict(model=model, scratch=scratch, names=names, target=target,
                addresses=model.jnt_qposadr[ids], limits=model.jnt_range[ids],
                wrist_indices=np.asarray([names.index("WRJ2"), names.index("WRJ1")]),
                root=int(runner._root_qpos_address), palm=palm, hand_geoms=hand,
                proximal_geoms=tuple(g for g in hand
                                     if not _descendant(model, int(model.geom_bodyid[g]), palm)),
                object_geoms=tuple(sorted(runner.object_collision_geom_ids)),
                finger_geoms=finger_geoms,
                table_top=float(scratch.geom_xpos[table, 2] + model.geom_size[table, 2]),
                desired_position=prepared.palm_position,
                desired_rotation=_rotation(prepared.palm_quaternion))


def evaluate_pose(context, wrist):
    """Evaluate this exact wrist pair, with no search or automatic replacement."""
    c = context
    model, scratch = c["model"], c["scratch"]
    target = c["target"].copy()
    target[c["wrist_indices"]] = wrist
    if np.any(target < c["limits"][:, 0] - 1e-6) or np.any(target > c["limits"][:, 1] + 1e-6):
        raise ValueError("Candidate is outside physical joint limits")
    scratch.qpos[c["addresses"]] = target
    scratch.qpos[c["root"]:c["root"] + 7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_forward(model, scratch)
    relative_rotation = scratch.xmat[c["palm"]].reshape(3, 3).copy()
    relative_position = scratch.xpos[c["palm"]].copy()
    root_rotation = c["desired_rotation"] @ relative_rotation.T
    root_position = c["desired_position"] - root_rotation @ relative_position
    root_quaternion = np.empty(4)
    mujoco.mju_mat2Quat(root_quaternion, root_rotation.reshape(9))
    scratch.qpos[c["root"]:c["root"] + 7] = np.r_[root_position, root_quaternion]
    finger_indices = [i for i in range(len(target)) if i not in c["wrist_indices"]]
    if not np.array_equal(target[finger_indices], c["target"][finger_indices]):
        raise ValueError("The original 22 finger targets changed")
    # Record both nominal target and the existing near-open approach state.
    states, geometry = {}, {}
    for state_name in ("nominal_target", "near_open_approach"):
        state = target.copy()
        if state_name == "near_open_approach":
            for i, name in enumerate(c["names"]):
                if name in OPENING_JOINTS:
                    state[i] -= 0.2
            state = np.clip(state, c["limits"][:, 0], c["limits"][:, 1])
        scratch.qpos[c["addresses"]] = state
        mujoco.mj_forward(model, scratch)
        geometry[state_name] = {
            "palm_position": scratch.xpos[c["palm"]].copy(),
            "palm_rotation": scratch.xmat[c["palm"]].reshape(3, 3).copy(),
            "finger_positions": scratch.geom_xpos[list(c["finger_geoms"])].copy(),
            "finger_rotations": scratch.geom_xmat[list(c["finger_geoms"])].copy(),
            "finger_commands": state[finger_indices].copy(),
        }
        states[state_name] = {
            "maximum_world_palm_position_error_m": float(np.max(np.abs(
                scratch.xpos[c["palm"]] - c["desired_position"]))),
            "maximum_world_palm_rotation_matrix_error": float(np.max(np.abs(
                scratch.xmat[c["palm"]].reshape(3, 3) - c["desired_rotation"]))),
        }
    # The frozen v3 planner has check_closing_targets=false. Its geometric gate
    # is the near-open approach endpoint, not a dynamic or closure certificate.
    clearance = min(_lowest_world_z(model, scratch, g) for g in c["hand_geoms"]) - c["table_top"]
    proximal = min(float(mujoco.mj_geomDistance(model, scratch, hand, obj, 10.0, None))
                   for hand in c["proximal_geoms"] for obj in c["object_geoms"])
    failed = []
    if clearance < TABLE_GATE_M:
        failed.append("nominal_table_clearance_below_10_mm")
    if proximal < PROXIMAL_GATE_M:
        failed.append("nominal_proximal_object_distance_below_1_mm")
    for state in states.values():
        if (state["maximum_world_palm_position_error_m"] > INVARIANCE_TOLERANCE
                or state["maximum_world_palm_rotation_matrix_error"] > INVARIANCE_TOLERANCE):
            failed.append("world_palm_invariance_failed")
    return dict(wrist_rad=np.asarray(wrist).tolist(),
                nominal_table_clearance_m=float(clearance),
                nominal_proximal_object_distance_m=float(proximal),
                passes_geometry_gate=not failed, failed_gates=sorted(set(failed)),
                root_position_m=root_position.tolist(),
                root_quaternion_wxyz=root_quaternion.tolist(),
                intended_22_finger_targets_unchanged=True,
                world_palm_invariance=states), geometry


def compare_invariance(row, geometry, baseline):
    row["finger_geometry_invariance"] = {}
    for state_name, values in geometry.items():
        other = baseline[state_name]
        entry = dict(
            maximum_world_finger_geom_position_difference_m=float(np.max(np.abs(
                values["finger_positions"] - other["finger_positions"]))),
            maximum_world_finger_geom_rotation_matrix_difference=float(np.max(np.abs(
                values["finger_rotations"] - other["finger_rotations"]))),
            applied_finger_commands_equal=np.array_equal(values["finger_commands"], other["finger_commands"]),
        )
        entry["passes"] = bool(entry["applied_finger_commands_equal"]
                               and entry["maximum_world_finger_geom_position_difference_m"] <= INVARIANCE_TOLERANCE
                               and entry["maximum_world_finger_geom_rotation_matrix_difference"] <= INVARIANCE_TOLERANCE)
        if not entry["passes"]:
            raise ValueError("Carrier compensation did not preserve nominal finger geometry")
        row["finger_geometry_invariance"][state_name] = entry


def analyze(repository: Path, assets: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Output already exists; choose a new output directory")
    pilot = repository / "research_artifacts/object_scene_pilot_v1"
    baseline = repository / "research_artifacts/task_outcome_baseline_v1"
    protocol_path = repository / "research_artifacts/wrist_comparison_plan_v1/protocol.json"
    plan = json.loads(protocol_path.read_text())
    training_report = json.loads((baseline / "report.json").read_text())
    sources = {}

    def add_source(path):
        path = Path(path).resolve()
        try:
            key = "repository:" + str(path.relative_to(repository))
        except ValueError:
            key = "runtime_assets:" + str(path.relative_to(assets))
        sources[key] = (path, sha(path))

    # The source protocol is frozen before this audit; its path-only link to
    # this output avoids circular hashes. Also record the numerical contract.
    protocol_contract = {key: plan[key] for key in (
        "development_cases", "model_contract", "geometry_and_motion",
    )}
    protocol_contract_sha = hashlib.sha256(json.dumps(
        protocol_contract, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    if (plan["geometry_and_motion"]["nominal_table_clearance_m"] != TABLE_GATE_M
            or plan["geometry_and_motion"]["nominal_proximal_object_clearance_m"] != PROXIMAL_GATE_M):
        raise ValueError("The declared numerical geometry gates changed")
    for path in (protocol_path, pilot / "dataset.npz", pilot / "splits.json",
                 pilot / "manifest.json", baseline / "report.json",
                 baseline / "proprioception_and_commands.npz", Path(__file__)):
        add_source(path)
    for path in (repository / "src/grasp_failure_prediction").rglob("*.py"):
        add_source(path)
    for path in (repository / "src/grasp_failure_prediction/evaluation/configs").rglob("*"):
        if path.is_file():
            add_source(path)
    for path in (default_dex_urdf_root() / "shadow_hand").rglob("*"):
        if path.is_file():
            add_source(path)
    for field, path in (("input_dataset_sha256", pilot / "dataset.npz"),
                        ("input_splits_sha256", pilot / "splits.json"),
                        ("input_manifest_sha256", pilot / "manifest.json")):
        if sha(path) != training_report[field]:
            raise ValueError("Original training evidence no longer matches its report")
    if sha(baseline / "proprioception_and_commands.npz") != training_report["models"]["proprioception_and_commands"]["model_sha256"]:
        raise ValueError("Frozen model file changed")
    if sha(baseline / "proprioception_and_commands.npz") != plan["model_contract"]["model_sha256"]:
        raise ValueError("The declared frozen model changed")
    splits = json.loads((pilot / "splits.json").read_text())
    input_contract = json.loads((pilot / "manifest.json").read_text())["input_contract"]
    if input_contract != training_report["input_contract"]:
        raise ValueError("Original input contract changed")
    with np.load(pilot / "dataset.npz", allow_pickle=False) as data:
        mask = np.isin(data["episode_ids"], splits["partitions"]["train"]["episode_ids"])
        if int(mask.sum()) != 520:
            raise ValueError("Expected the original 520 training rows")
        if data["X"].shape[1] != 1440 or input_contract["joint_names"][:2] != ["WRJ2", "WRJ1"]:
            raise ValueError("Original input dimensions or wrist ordering changed")
        pairs = np.unique(data["X"][mask].reshape(-1, 20, 72)[:, -1, 48:50], axis=0).astype(float)
    if len(pairs) != 8 or len(plan["development_cases"]) != 8:
        raise ValueError("This audit freezes exactly 8 training pairs and 8 mug proposals")
    cases = []
    with static_mujoco_only() as call_counts:
        for case_entry in plan["development_cases"]:
            case_path = assets / case_entry["case_path"]
            stored_path = assets / "runs/mug_table_clearance_validation_2026-10-11/final_checks_27" / case_entry["episode_id"] / "collection_metadata.json"
            add_source(case_path)
            for path in stored_path.parent.iterdir():
                if path.is_file():
                    add_source(path)
            prepared = _prepare_case(case_path, assets)
            add_source(assets / prepared.source["proposal_path"])
            add_source((assets / prepared.source["proposal_path"]).parent / "T_world_camera.npy")
            for path in prepared.source["geometry_asset_files_sha256"]:
                add_source(prepared.object_geometry.mjcf_path.parent / path)
            resolved = prepared.resolved
            runner = NoisyShadowRunner(
                resolved.execution_protocol, default_dex_urdf_root(),
                physics_timestep_s=resolved.environment.simulator.physics_timestep_s,
                control_timestep_s=resolved.environment.simulator.control_timestep_s,
                noise_amplitude_rad=0.0, object_geometry=prepared.object_geometry,
            )
            case = resolved.case
            runner.reset(seed=case.seed, object_mass_kg=case.object.mass_kg,
                         object_position_m=np.asarray(case.initial_condition.object_position_m),
                         object_orientation_wxyz=np.asarray(case.initial_condition.object_orientation_xyzw)[[3, 0, 1, 2]])
            context = static_context(runner, prepared)
            original_wrist = context["target"][context["wrist_indices"]]
            original, reference_geometry = evaluate_pose(context, original_wrist)
            candidates = []
            for candidate_index, pair in enumerate(pairs):
                row, geometry = evaluate_pose(context, pair)
                compare_invariance(row, geometry, reference_geometry)
                row["training_candidate_index"] = candidate_index
                candidates.append(row)
            stored = json.loads(stored_path.read_text())
            if stored["noise_amplitude_rad"] != 0.0:
                raise ValueError("Expected stored zero-noise B")
            corrected, geometry = evaluate_pose(context, stored["clearance_plan"]["adjusted_wrist_rad"])
            compare_invariance(corrected, geometry, reference_geometry)
            saved = stored["clearance_plan"]
            if (saved["configuration"]["clearance_m"] != TABLE_GATE_M
                    or saved["configuration"]["proximal_object_clearance_m"] != PROXIMAL_GATE_M
                    or saved["configuration"]["check_closing_targets"] is not False):
                raise ValueError("Stored v3 geometric gate differs from the frozen audit")
            repeat_checks = {
                "table_clearance_matches_saved": abs(corrected["nominal_table_clearance_m"] - saved["minimum_table_clearance_m"]) <= REPLAY_TOLERANCE,
                "proximal_distance_matches_saved": abs(corrected["nominal_proximal_object_distance_m"] - saved["minimum_proximal_object_distance_m"]) <= REPLAY_TOLERANCE,
                "carrier_position_matches_saved": np.allclose(corrected["root_position_m"], saved["root_position_m"], atol=REPLAY_TOLERANCE, rtol=0),
                "carrier_quaternion_matches_saved": np.allclose(corrected["root_quaternion_wxyz"], saved["root_quaternion_wxyz"], atol=REPLAY_TOLERANCE, rtol=0),
            }
            corrected["matches_stored_v3_plan"] = {key: bool(value) for key, value in repeat_checks.items()}
            if not all(repeat_checks.values()):
                raise ValueError("Static B geometry differs from its original v3 certificate")
            if float(runner.data.time) != 0.0 or runner.action_history:
                raise ValueError("Audit advanced time or generated action history")
            cases.append(dict(episode_id=case_entry["episode_id"], case_path=case_entry["case_path"],
                              original_wrist_rad=original["wrist_rad"],
                              intended_world_palm_position_m=prepared.palm_position.tolist(),
                              intended_world_palm_quaternion_wxyz=prepared.palm_quaternion.tolist(),
                              resolved_environment=resolved.environment.model_dump(mode="json"),
                              environment_config_hash=resolved.environment_config_hash,
                              resolved_base_execution_protocol=resolved.execution_protocol.model_dump(mode="json"),
                              base_execution_protocol_config_hash=resolved.execution_protocol_config_hash,
                              stored_v3_clearance_configuration=saved["configuration"],
                              source_binding=prepared.source, training_candidates=candidates,
                              corrected_B=corrected, simulator_time_s=float(runner.data.time),
                              generated_action_history_rows=0))
    after = {key: sha(path) for key, (path, _) in sources.items()}
    before = {key: digest for key, (_, digest) in sources.items()}
    if before != after:
        raise ValueError("An audited source changed while checking geometry")
    rows = [row for case in cases for row in case["training_candidates"]]
    b_rows = [case["corrected_B"] for case in cases]
    report = dict(
        schema_version="observed_wrist_pose_static_feasibility_v1", date="2026-10-12",
        scope="Finite geometric audit only; no continuous-range proof, new rollout, prediction, label, training or model improvement",
        training_rows=520, training_wrist_joint_names=["WRJ2", "WRJ1"],
        candidate_source="Float32 last wrist commands actually recorded in original training rows; no inferred continuous-range coverage",
        unique_training_wrist_pairs_rad=pairs.tolist(), mug_proposals=8,
        observed_training_candidate_checks=len(rows),
        observed_training_candidates_passing_geometry=sum(row["passes_geometry_gate"] for row in rows),
        corrected_B_checks=len(b_rows), corrected_B_passing_geometry=sum(row["passes_geometry_gate"] for row in b_rows),
        observed_training_table_clearance_range_m=[min(row["nominal_table_clearance_m"] for row in rows), max(row["nominal_table_clearance_m"] for row in rows)],
        corrected_B_table_clearance_range_m=[min(row["nominal_table_clearance_m"] for row in b_rows), max(row["nominal_table_clearance_m"] for row in b_rows)],
        gates=dict(nominal_table_clearance_m=TABLE_GATE_M, nominal_proximal_object_distance_m=PROXIMAL_GATE_M,
                   check_closing_targets=False, invariance_absolute_tolerance=INVARIANCE_TOLERANCE,
                   saved_plan_reproduction_absolute_tolerance=REPLAY_TOLERANCE),
        cases=cases, runtime_versions=_runtime_versions(),
        permitted_exported_mujoco_calls=sorted(ALLOWED_MUJOCO_CALLS),
        actual_exported_mujoco_call_counts=call_counts, mj_step_block_exercised=True,
        source_sha256=before, original_sources_unchanged=True,
        source_protocol_path=str(protocol_path.relative_to(repository)),
        source_protocol_numerical_contract=protocol_contract,
        source_protocol_numerical_contract_sha256=protocol_contract_sha,
        full_source_protocol_sha256=sha(protocol_path),
        valid_A_B_pose_manifest_created=False,
        interpretation="No actually observed training wrist pair passed the fixed approach geometry gates for these mug proposals. All existing corrected B plans repeat statically. A new safe A must still be selected and frozen, and its relation to training reported; no familiar-versus-unfamiliar or dynamic success comparison has been run.",
    )
    output.mkdir(parents=True)
    write_json(output / "report.json", report)
    write_json(output / "manifest.json", dict(
        schema_version="wrist_pose_static_feasibility_manifest_v1",
        files_sha256={"report.json": sha(output / "report.json")},
        source_sha256=before, no_new_rollouts_or_predictions=True,
        frozen_model_unchanged=True,
    ))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True,
                        help="Checkout containing the existing ignored runs and robot runtime assets")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; overwrites refused")
    args = parser.parse_args()
    report = analyze(Path(__file__).resolve().parents[1], args.project_root.resolve(), args.output.resolve())
    print(json.dumps({key: report[key] for key in (
        "observed_training_candidate_checks", "observed_training_candidates_passing_geometry",
        "corrected_B_checks", "corrected_B_passing_geometry", "observed_training_table_clearance_range_m",
        "corrected_B_table_clearance_range_m", "original_sources_unchanged",
    )}, indent=2))


if __name__ == "__main__":
    main()
