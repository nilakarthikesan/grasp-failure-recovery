"""Freeze geometry-selected wrist pairs before any rollout or prediction.

Selection advances no simulator time and reads no model weights. It uses
original training wrist commands and normalizer constants, static geometry,
physical limits and the stored corrected B poses only. The existing static
audit blocks every exported MuJoCo integration/step routine during selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from analyze_wrist_pose_feasibility import (
    ALLOWED_MUJOCO_CALLS, INVARIANCE_TOLERANCE, PROXIMAL_GATE_M,
    REPLAY_TOLERANCE, TABLE_GATE_M, compare_invariance, evaluate_pose, sha,
    static_context, static_mujoco_only, write_json,
)
from grasp_failure_prediction.evaluation.noisy_rollouts import (
    NoisyShadowRunner, _prepare_case, _runtime_versions,
)
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root


GRID_POINTS = 13
REFINEMENT_ROUNDS = 3
REFINEMENT_POINTS = 5
# A modest command separation declared before any new score or outcome. This
# is about one degree in combined wrist coordinates, rather than a numerical
# near-copy. Report the actual physical difference and do not equate this
# design gate with useful motion or model improvement.
MINIMUM_WRIST_SEPARATION_RAD = 0.02
COMMAND_FEATURE_INDICES = np.asarray([48, 49])
FROZEN_MODEL_SHA256 = "16c9941a51130daea818c6ad227ef1e3a02e4be9ed1c7aa5a8e8f3e023428beb"


def _canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def select(repository: Path, assets: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Output already exists; choose a new output directory")
    protocol_path = repository / "research_artifacts/wrist_comparison_plan_v1/protocol.json"
    static_report_path = repository / "research_artifacts/wrist_pose_feasibility_v1/report.json"
    plan = json.loads(protocol_path.read_text())
    previous = json.loads(static_report_path.read_text())
    sources: dict[str, tuple[Path, str]] = {}

    def add_source(path):
        path = Path(path).resolve()
        try:
            key = "repository:" + str(path.relative_to(repository))
        except ValueError:
            key = "runtime_assets:" + str(path.relative_to(assets))
        sources[key] = (path, sha(path))

    # Re-use the original finite audit's complete frozen bindings. Hashing raw
    # recordings is an integrity operation; their labels and predictions are
    # never inspected to rank candidates or choose the first case.
    for key, digest in previous["source_sha256"].items():
        prefix, relative = key.split(":", 1)
        root = {"repository": repository, "runtime_assets": assets}[prefix]
        path = (root / relative).resolve()
        if sha(path) != digest:
            raise ValueError(f"Original audit input changed: {key}")
        sources[key] = (path, digest)
    for path in (protocol_path, static_report_path,
                 static_report_path.parent / "manifest.json", Path(__file__)):
        add_source(path)
    model_path = repository / plan["model_contract"]["model_file"]
    if (sha(model_path) != FROZEN_MODEL_SHA256
            or plan["model_contract"]["model_sha256"] != FROZEN_MODEL_SHA256):
        raise ValueError("Expected the unchanged original model")
    if (plan["geometry_and_motion"]["nominal_table_clearance_m"] != TABLE_GATE_M
            or plan["geometry_and_motion"]["nominal_proximal_object_clearance_m"] != PROXIMAL_GATE_M
            or plan["noise"] != {"finger_command_bias_rad": 0.0, "wrist_random_noise_rad": 0.0}):
        raise ValueError("The frozen geometry or zero-noise contract changed")
    pilot = repository / "research_artifacts/object_scene_pilot_v1"
    splits = json.loads((pilot / "splits.json").read_text())
    with np.load(pilot / "dataset.npz", allow_pickle=False) as data:
        train = np.isin(data["episode_ids"], splits["partitions"]["train"]["episode_ids"])
        if int(train.sum()) != 520 or data["X"].shape[1] != 1440:
            raise ValueError("Unexpected original training partition/input shape")
        # Only training X and episode identities are read, never y or test X.
        commands = data["X"][train].astype(np.float64).reshape(-1, 20, 72)[:, -1, 48:50]
    training_pairs = np.unique(commands, axis=0)
    with np.load(model_path, allow_pickle=False) as constants:
        # Deliberately do not access weights, intercept or a prediction API.
        mean = constants["mean"][COMMAND_FEATURE_INDICES]
        scale = constants["scale"][COMMAND_FEATURE_INDICES]
        active = constants["active"][COMMAND_FEATURE_INDICES]
    if (not active.all() or not np.allclose(commands.mean(axis=0), mean, atol=1e-12, rtol=0)
            or not np.allclose(commands.std(axis=0), scale, atol=1e-12, rtol=0)):
        raise ValueError("Wrist normalizer does not match the original training commands")
    if len(training_pairs) != 8 or len(plan["development_cases"]) != 8:
        raise ValueError("Expected the eight declared training pairs and eight mug cases")
    range_min, range_max = commands.min(axis=0), commands.max(axis=0)

    def annotate(row, b_wrist):
        wrist = np.asarray(row["wrist_rad"])
        row["training_wrist_scaled_distance"] = float(np.linalg.norm((wrist - mean) / scale))
        b_distance = float(np.linalg.norm((b_wrist - mean) / scale))
        row["training_distance_lower_than_B"] = bool(row["training_wrist_scaled_distance"] < b_distance)
        row["in_training_command_range_per_joint"] = ((wrist >= range_min) & (wrist <= range_max)).tolist()
        row["is_observed_training_pair"] = bool(np.any(np.all(training_pairs == wrist, axis=1)))
        row["wrist_distance_from_B_rad"] = float(np.linalg.norm(wrist - b_wrist))
        row["eligible_as_A"] = bool(row["passes_geometry_gate"]
                                    and row["wrist_distance_from_B_rad"] >= MINIMUM_WRIST_SEPARATION_RAD)

    cases, candidate_reports = [], []
    with static_mujoco_only() as call_counts:
        for case_entry in plan["development_cases"]:
            case_path = assets / case_entry["case_path"]
            stored_path = assets / "runs/mug_table_clearance_validation_2026-10-11/final_checks_27" / case_entry["episode_id"] / "collection_metadata.json"
            add_source(case_path)
            add_source(stored_path)
            # Only the old B geometric certificate and execution configuration
            # are consumed from metadata; old label/failure fields are unused.
            stored = json.loads(stored_path.read_text())
            saved = stored["clearance_plan"]
            if (stored["noise_amplitude_rad"] != 0.0
                    or saved["configuration"]["clearance_m"] != TABLE_GATE_M
                    or saved["configuration"]["proximal_object_clearance_m"] != PROXIMAL_GATE_M
                    or saved["configuration"]["check_closing_targets"] is not False):
                raise ValueError("Stored corrected B does not match the frozen gates")
            prepared = _prepare_case(case_path, assets)
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
            _, reference = evaluate_pose(context, context["target"][context["wrist_indices"]])
            b_wrist = np.asarray(saved["adjusted_wrist_rad"], dtype=np.float64)
            b, geometry = evaluate_pose(context, b_wrist)
            compare_invariance(b, geometry, reference)
            annotate(b, b_wrist)
            repeat_checks = dict(
                table_clearance_matches_saved=abs(b["nominal_table_clearance_m"] - saved["minimum_table_clearance_m"]) <= REPLAY_TOLERANCE,
                proximal_distance_matches_saved=abs(b["nominal_proximal_object_distance_m"] - saved["minimum_proximal_object_distance_m"]) <= REPLAY_TOLERANCE,
                carrier_position_matches_saved=bool(np.allclose(b["root_position_m"], saved["root_position_m"], atol=REPLAY_TOLERANCE, rtol=0)),
                carrier_quaternion_matches_saved=bool(np.allclose(b["root_quaternion_wxyz"], saved["root_quaternion_wxyz"], atol=REPLAY_TOLERANCE, rtol=0)),
            )
            b["matches_stored_v3_plan"] = repeat_checks
            if not all(repeat_checks.values()):
                raise ValueError("Stored B geometric certificate did not reproduce")
            limits = context["limits"][context["wrist_indices"]]
            grid_axes = [np.linspace(lower, upper, GRID_POINTS) for lower, upper in limits]
            grid_step = (limits[:, 1] - limits[:, 0]) / (GRID_POINTS - 1)
            candidates = {}

            def evaluate_candidate(wrist, origin):
                pair = tuple(float(value) for value in wrist)
                if pair in candidates:
                    if origin not in candidates[pair]["candidate_origins"]:
                        candidates[pair]["candidate_origins"].append(origin)
                    return
                row, geometry = evaluate_pose(context, pair)
                compare_invariance(row, geometry, reference)
                annotate(row, b_wrist)
                row["candidate_origins"] = [origin]
                candidates[pair] = row

            for i in grid_axes[0]:
                for j in grid_axes[1]:
                    evaluate_candidate([i, j], "physical_limits_13_by_13_grid")
            for pair in training_pairs:
                evaluate_candidate(pair, "actually_observed_training_pair")
            evaluate_candidate(b_wrist, "stored_corrected_B")

            def ranked_candidates():
                # Stable numeric tie break: WRJ2 then WRJ1. Neither predictions
                # nor new/old grasp outcomes enter the objective.
                return sorted((row for row in candidates.values() if row["eligible_as_A"]),
                              key=lambda row: (row["training_wrist_scaled_distance"], *row["wrist_rad"]))

            for round_index in range(REFINEMENT_ROUNDS):
                # B cannot be selected as A, but excluding it as a search
                # center would miss nearby distinct candidates simply because
                # B has already reached a better boundary than the coarse grid.
                passing = sorted((row for row in candidates.values() if row["passes_geometry_gate"]),
                                 key=lambda row: (row["training_wrist_scaled_distance"], *row["wrist_rad"]))
                center = np.asarray(passing[0]["wrist_rad"] if passing else b["wrist_rad"])
                radius = grid_step / (2 ** round_index)
                axes = [np.linspace(max(limits[i, 0], center[i] - radius[i]),
                                    min(limits[i, 1], center[i] + radius[i]), REFINEMENT_POINTS)
                        for i in range(2)]
                for i in axes[0]:
                    for j in axes[1]:
                        evaluate_candidate([i, j], f"nearest_training_geometry_refinement_{round_index + 1}")
            ranked = ranked_candidates()
            a = dict(ranked[0]) if ranked else None
            if a is not None:
                # Fresh re-evaluation catches scratch-state/order mistakes in
                # the selected certificate before any manifest is exported.
                fresh, geometry = evaluate_pose(context, a["wrist_rad"])
                compare_invariance(fresh, geometry, reference)
                for key in ("nominal_table_clearance_m", "nominal_proximal_object_distance_m"):
                    if abs(a[key] - fresh[key]) > REPLAY_TOLERANCE:
                        raise ValueError("Selected A did not reproduce on re-evaluation")
            if float(runner.data.time) != 0.0 or runner.action_history:
                raise ValueError("Static selection advanced time or produced actions")
            b_valid = bool(b["passes_geometry_gate"])
            eligible = bool(a is not None and b_valid)
            finger_indices = [i for i in range(len(context["names"])) if i not in context["wrist_indices"]]
            qualification = (
                "no_feasible_pair" if not eligible else
                "familiar_command_range_A_versus_unfamiliar_B_geometry_only" if all(a["in_training_command_range_per_joint"]) and not all(b["in_training_command_range_per_joint"]) else
                "two_out_of_range_wrists_sensitivity_only" if not all(a["in_training_command_range_per_joint"]) and not all(b["in_training_command_range_per_joint"]) else
                "wrist_sensitivity_geometry_only"
            )
            entry = dict(
                pair_id=case_entry["episode_id"], case_path=case_entry["case_path"],
                case_sha256=sha(case_path), original_episode_id=case_entry["original_episode_id"],
                eligible=eligible, status="frozen_eligible_static_pair" if eligible else "no_feasible_pair",
                reason=None if eligible else "No distinct passing A or stored B fails revalidation",
                comparison_qualification=qualification,
                intended_world_palm_position_m=prepared.palm_position.tolist(),
                intended_world_palm_quaternion_wxyz=prepared.palm_quaternion.tolist(),
                intended_finger_joint_names=[context["names"][i] for i in finger_indices],
                intended_finger_target_rad=context["target"][finger_indices].tolist(),
                physical_wrist_limits_rad=limits.tolist(),
                resolved_environment=resolved.environment.model_dump(mode="json"),
                environment_config_hash=resolved.environment_config_hash,
                resolved_base_execution_protocol=resolved.execution_protocol.model_dump(mode="json"),
                base_execution_protocol_config_hash=resolved.execution_protocol_config_hash,
                stored_v3_execution_protocol=stored["execution_protocol"],
                stored_v3_execution_protocol_config_hash=stored["execution_protocol_config_hash"],
                source_binding=prepared.source,
                poses={"A": a, "B": b},
                wrist_delta_A_minus_B_rad=None if a is None else (np.asarray(a["wrist_rad"]) - b_wrist).tolist(),
                simulator_time_s=float(runner.data.time), generated_action_history_rows=0,
            )
            cases.append(entry)
            candidate_reports.append(dict(
                pair_id=entry["pair_id"], candidate_count=len(candidates),
                geometry_passing_count=sum(row["passes_geometry_gate"] for row in candidates.values()),
                distinct_A_passing_count=len(ranked),
                selected_A_training_distance=None if a is None else a["training_wrist_scaled_distance"],
                stored_B_training_distance=b["training_wrist_scaled_distance"],
                candidates=list(candidates.values()),
            ))
    before = {key: digest for key, (_, digest) in sources.items()}
    if before != {key: sha(path) for key, (path, _) in sources.items()}:
        raise ValueError("Selection input changed during the static search")
    first = next((case["pair_id"] for case in cases if case["eligible"]), None)
    selection_rule = dict(
        objective="Minimum Euclidean distance of the two last wrist commands to original training mean after original train-only scaling; static geometry gates and distinctness first",
        priority="Passing actually observed training pairs are preferred; none passes in these declared cases. Otherwise rank all declared grid/refinement candidates by training distance.",
        tie_break="Ascending WRJ2 then WRJ1 for exactly equal distance",
        physical_grid_points_per_joint=GRID_POINTS,
        observed_training_pairs_always_included=True, stored_B_always_included=True,
        refinement_rounds=REFINEMENT_ROUNDS, refinement_points_per_joint=REFINEMENT_POINTS,
        refinement_radius="Initial physical grid spacing, then half and quarter; centered on nearest-training passing candidate including B as a center only, or B if none; physical limits truncate each axis",
        minimum_wrist_separation_from_B_rad=MINIMUM_WRIST_SEPARATION_RAD,
        separation_rationale="Declared before new predictions/outcomes: about one degree in combined wrist command coordinates and comparable to the narrow original WRJ1 command range; avoids numerical near-copy contrasts, without claiming a minimum dynamic or prediction effect",
        first_pair_selection="First eligible case in the previously declared eight-case order, independent of predictions and outcomes",
        original_training_rows=520, command_feature_indices=COMMAND_FEATURE_INDICES.tolist(),
        wrist_joint_names=["WRJ2", "WRJ1"], training_mean_rad=mean.tolist(),
        training_scale_rad=scale.tolist(),
        training_command_min_rad=range_min.tolist(), training_command_max_rad=range_max.tolist(),
        no_model_weights_intercept_or_prediction_read=True, no_training_labels_read=True,
        no_new_or_old_outcome_based_selection=True,
    )
    pose_manifest = dict(
        schema_version="explicit_wrist_pair_pose_manifest_v1", date="2026-10-12",
        status="frozen_before_new_rollouts_and_predictions", first_pair_id=first,
        execution_scope="Run only the first eligible pair initially; review actual outcomes before any further collection, training or model change",
        model_contract=plan["model_contract"], noise=plan["noise"],
        outcome_rule=plan["outcome_rule"], held_constant=plan["held_constant"],
        gates=dict(nominal_table_clearance_m=TABLE_GATE_M,
                   nominal_proximal_object_clearance_m=PROXIMAL_GATE_M,
                   runtime_table_margin_m=0.001, runtime_check_interval_s=0.002,
                   check_closing_targets=False,
                   invariance_absolute_tolerance=INVARIANCE_TOLERANCE,
                   pose_reproduction_absolute_tolerance=REPLAY_TOLERANCE),
        selection_rule=selection_rule, cases=cases, source_sha256=before,
        source_protocol_numerical_contract_sha256=previous["source_protocol_numerical_contract_sha256"],
        full_source_protocol_sha256=sha(protocol_path), runtime_versions=_runtime_versions(),
        no_simulator_time_advanced=True, no_predictions_generated=True,
        no_labels_generated=True, frozen_model_unchanged=True,
        interpretation="Static selection preserves intended geometry but does not certify dynamic safety, equal motion/contact, success or a causal warning mechanism. Both out-of-training wrist settings establish sensitivity only, not familiar versus unfamiliar transfer.",
    )
    report = dict(
        schema_version="explicit_wrist_pair_static_selection_report_v1", date="2026-10-12",
        declared_cases=len(cases), eligible_pairs=sum(case["eligible"] for case in cases),
        no_feasible_pairs=sum(not case["eligible"] for case in cases), first_pair_id=first,
        selection_rule=selection_rule, candidate_searches=candidate_reports,
        candidate_evaluations=sum(case["candidate_count"] for case in candidate_reports),
        permitted_exported_mujoco_calls=sorted(ALLOWED_MUJOCO_CALLS),
        actual_exported_mujoco_call_counts=call_counts, mj_step_block_exercised=True,
        source_sha256=before, original_sources_unchanged=True,
        no_predictions_labels_or_rollouts=True, simulator_time_s=0.0,
        scope="Finite deterministic geometry search, not continuous feasibility proof or dynamic validation",
    )
    output.mkdir(parents=True)
    write_json(output / "pose_manifest.json", pose_manifest)
    write_json(output / "selection_report.json", report)
    first_case = next((case for case in cases if case["pair_id"] == first), None)
    rows = []
    for case in cases:
        if not case["eligible"]:
            rows.append(f"* `{case['pair_id']}`: no feasible pair.")
            continue
        a, b = case["poses"]["A"], case["poses"]["B"]
        rows.append(f"* `{case['pair_id']}`: A [{a['wrist_rad'][0]:.9f}, {a['wrist_rad'][1]:.9f}] rad; "
                    f"B [{b['wrist_rad'][0]:.9f}, {b['wrist_rad'][1]:.9f}] rad. "
                    f"Training distances {a['training_wrist_scaled_distance']:.3f} and {b['training_wrist_scaled_distance']:.3f}; "
                    f"separation {a['wrist_distance_from_B_rad']:.6f} rad. A closer than B: {a['training_distance_lower_than_B']}. {case['comparison_qualification']}.")
    first_detail = ("No pair is eligible." if first_case is None else
                    f"The first eligible pair is `{first}`, chosen by the existing case order. "
                    f"Its A/B comparison is **{first_case['comparison_qualification'].replace('_', ' ')}**.")
    readme = f"""# Frozen wrist pair selection

12 October 2026. This package freezes exact A/B wrist angles and compensated
carrier coordinates before any new rollout or predictor evaluation.

## What was selected

The previous finite audit found no passing combination among the eight
actually observed training wrist pairs and eight mug proposals. This search
therefore uses the declared fallback: the geometrically feasible candidate
nearest the original training wrist-command distribution. B remains the
stored zero-noise corrected pose for the same case.

{sum(case['eligible'] for case in cases)}/{len(cases)} cases have a distinct
passing A/B pair. {first_detail}

Both wrists must pass the 10 mm nominal table gate and 1 mm proximal-object
gate. Carrier compensation preserves the intended world palm and all 22
finger targets and their nominal world geometry. This only certifies the
near-open approach endpoint used by the existing v3 planner. It does not
certify closure, dynamics or grasp success. The later executor must retain
the 1 mm runtime table guard at every 2 ms physics endpoint and stop without
a label if the margin fails.

## Selection rule

A fixed 13 by 13 grid spans both physical wrist limits. All eight observed
training command pairs and B are also evaluated. Three local 5 by 5
refinement rounds use initial grid spacing, then half and quarter, centered
on the currently closest-training passing candidate. B may be a refinement
center but cannot be selected as A. Candidates must be at least
{MINIMUM_WRIST_SEPARATION_RAD} rad from B in Euclidean wrist distance.
Every evaluated candidate is recorded, including rejected ones. This is a
finite search, not a continuous global optimum or infeasibility proof.

Training distance is the Euclidean norm of the two wrist commands after
scaling with the original training-only mean and scale at feature indices
48 and 49. No model weights or intercept are read. No prediction, new grasp
outcome, old label or failure type is used to rank candidates or choose the
first pair. Old B metadata contains those historical fields, but only its
geometric certificate and execution settings are consumed. The original
training partition contributes X and episode identities only, never y.

All retained candidates are rechecked for intended palm/finger invariance
within 1e-10. Selected A is re-evaluated before export. Every stored B
certificate repeats within 1e-12. Original source, model, data and asset
hashes agree before and after selection. Simulator time stays zero and the
existing integration blocker is explicitly exercised.

## Frozen pairs

{chr(10).join(rows)}

Whether A lies within each original command range is recorded separately.
Being closer to training is not the same as being in the training range,
and command familiarity does not imply familiar measured joint histories.
The selected distinct A can have greater training distance than B; this
comparison is reported rather than forcing a favorable direction.
If both settings are out of range, this is a wrist-sensitivity comparison,
not a familiar versus unfamiliar test. A difference in eventual predictions
would still need interpretation alongside actual dynamics and outcomes.

## Files and reproduction

* [pose_manifest.json](pose_manifest.json) is the exact, ordered pose contract
  for an explicit executor. The first eligible pair should be run first with
  zero noise and the unchanged original model.
* [selection_report.json](selection_report.json) contains every candidate,
  rejected gate, distance, invariance check, source binding and allowed-call
  counter. It reports no generated predictions, labels or rollouts.
* [manifest.json](manifest.json) hashes all package payloads and input sources.
* [select_wrist_comparison_poses.py](../../scripts/select_wrist_comparison_poses.py)
  reproduces selection, refusing to overwrite an output directory.

```bash
python -B scripts/select_wrist_comparison_poses.py \\
  --project-root /path/to/checkout/with/existing/runs \\
  --output /tmp/wrist_pair_selection_reproduction
```

The next experiment is one explicit zero-noise pair. These static results do
not improve the model or answer the later matched-budget data-diversity
question. No additional collection or fitting is justified by this geometry
certificate alone.
"""
    (output / "README.md").write_text(readme)
    write_json(output / "manifest.json", dict(
        schema_version="wrist_pair_selection_bundle_manifest_v1",
        files_sha256={name: sha(output / name) for name in (
            "pose_manifest.json", "selection_report.json", "README.md")},
        source_sha256=before, original_sources_unchanged=True,
        pose_manifest_canonical_sha256=_canonical_sha(pose_manifest),
        no_rollouts_predictions_labels_or_model_changes=True,
    ))
    return pose_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; overwrites refused")
    args = parser.parse_args()
    result = select(Path(__file__).resolve().parents[1], args.project_root.resolve(), args.output.resolve())
    print(json.dumps(dict(eligible_pairs=sum(case["eligible"] for case in result["cases"]),
                         first_pair_id=result["first_pair_id"],
                         pair_summary=[dict(pair_id=case["pair_id"], eligible=case["eligible"],
                                            qualification=case["comparison_qualification"],
                                            A=None if case["poses"]["A"] is None else case["poses"]["A"]["wrist_rad"],
                                            B=case["poses"]["B"]["wrist_rad"])
                                       for case in result["cases"]]), indent=2))


if __name__ == "__main__":
    main()
