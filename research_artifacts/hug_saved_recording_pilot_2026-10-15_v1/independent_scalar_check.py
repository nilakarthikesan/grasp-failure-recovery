"""Independent saved-artifact checks; no dynamics, forwards, fitting or imports of the driver."""
from pathlib import Path
import hashlib
import json

import numpy as np
import mujoco


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "independent_scalar_receipt.json"
if OUT.exists():
    raise FileExistsError(OUT)


def sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def forbidden(*args, **kwargs):
    raise AssertionError("independent receipt forbids simulation integration/forwards")


for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_forwardSkip"):
    setattr(mujoco, name, forbidden)

protocol = read(ROOT / "protocol.json")
authorization = read(ROOT / "authorization.json")
summary = read(ROOT / "summary.json")
assert summary["status"] == "COMPLETED_BOUNDED_QUOTA"
assert authorization["authorized"] and authorization["protocol_sha256"] == sha(ROOT / "protocol.json")
assert summary["protocol_sha256"] == sha(ROOT / "protocol.json")
assert summary["authorization_sha256"] == sha(ROOT / "authorization.json")
for group in ("source_bindings", "input_bindings", "saved_model_bindings"):
    for path, expected in protocol[group].items():
        assert sha(path) == expected, path
for field in ("static_profile_report", "camera_report", "current_selected_preflight"):
    assert sha(authorization[field]["path"]) == authorization[field]["sha256"]

schedule = protocol["schedule"]
assert len(schedule) == 20 and len(summary["schedule"]) == 20
assert [x["source_inventory_index"] for x in schedule] == list(range(0, 32, 2)) + [0, 8, 16, 24]
assert len({digest(v["command"]) for v in protocol["cases"].values()}) == 16
inventory = read(protocol["inventory"])["modern_cases"]
records, labels, per_case, repeats = {}, {}, [], []
raw = Path(protocol["raw_output"])
assert {p.name for p in (raw / "attempts").iterdir()} == {s["attempt_id"] for s in schedule}

for planned, row in zip(schedule, summary["schedule"], strict=True):
    assert all(row[key] == planned[key] for key in planned)
    assert row["status"] == "COMPLETED" and row["outcome"] is not None
    aid = row["attempt_id"]
    path = Path(row["recording"])
    assert path == raw / "attempts" / aid / "recording"
    payload = read(path / "attempt.json")
    assert sha(path / "attempt.json") == row["raw_record_sha256"]
    assert payload["artifacts_sha256"] == row["raw_artifacts_sha256"]
    for name, expected in payload["artifacts_sha256"].items():
        assert sha(path / name) == expected
    meta = payload["attempt"]
    frozen = protocol["cases"][str(row["source_inventory_index"])]
    assert meta["attempt_id"] == aid and meta["mapping"]["status"] == "ok"
    assert meta["execution_status"] == "completed" and meta["outcome"] == row["outcome"]
    assert meta["source_generator"] == "hug" and meta["hand_id"] == "shadow_hand_right"
    assert meta["execution_protocol_id"] == "fixed_grasp_lift_v2"
    assert meta["proposal_group_id"] == frozen["proposal_group_id"]
    assert meta["observation_group_id"] == frozen["observation_group_id"]
    assert meta["geometry_group_id"] == frozen["geometry_group_id"]
    assert meta["provenance_hashes"]["mapped_pose"] == digest(frozen["command"])
    assert meta["provenance_hashes"]["configuration"] == sha(ROOT / "protocol.json")
    model_path = path.parent / "scene.mjb"
    assert sha(model_path) == meta["provenance_hashes"]["model"]
    obs, diag = archive(path / "observables.npz"), archive(path / "diagnostics.npz")
    times, phase = obs["time_s"], obs["phase"]
    assert len(times) == 163 and np.max(np.abs(np.diff(times) - .04)) < 1e-12
    assert np.max(np.abs(times - np.arange(1, 164) * .04)) < 1e-10
    assert np.array_equal(diag["time_s"], times)
    assert np.max(np.abs(diag["diagnostic_contact_time_s"] - (times-.002))) < 1e-12
    assert np.array_equal(obs["applied_target_rad"], np.clip(obs["requested_target_rad"], obs["target_limits_rad"][:, 0], obs["target_limits_rad"][:, 1]))
    # Read addresses from the saved compiled artifact; no data/forward/step call.
    model = mujoco.MjModel.from_binary_path(str(model_path))
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    object_address = int(model.jnt_qposadr[joint])
    for i, name in enumerate(obs["joint_names"]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        np.testing.assert_array_equal(obs["hand_qpos"][:, i], diag["full_qpos"][:, model.jnt_qposadr[jid]])
        np.testing.assert_array_equal(obs["hand_qvel"][:, i], diag["full_qvel"][:, model.jnt_dofadr[jid]])
    spec = read(inventory[row["source_inventory_index"]]["resolved_case_path"])["execution_protocol"]
    heights = diag["full_qpos"][:, object_address+2]
    lift = max(0., float(max(heights)-heights[0]))
    acquired = bool(np.any((diag["hand_object_contact"] != 0) & np.isin(phase, ["close_fingers", "lift", "hold"])))
    approach = (diag["hand_table_contact"] != 0) & np.isin(phase, ["move_to_pregrasp", "approach"])
    limit = spec["success"]
    hold_good = heights[phase == "hold"] >= heights[0] + limit["minimum_lift_m"] - limit["maximum_object_drop_m"]
    current = longest = 0
    for value in hold_good:
        current = current + 1 if value else 0
        longest = max(longest, current)
    hold = longest*.04
    drop = float(max(heights)-heights[-1])
    timeout = bool(times[-1] > spec["parameters"]["total_timeout_s"] + 1e-9)
    success = acquired and lift >= limit["minimum_lift_m"] and hold >= limit["required_hold_s"] and drop <= limit["maximum_object_drop_m"] and not np.any(approach) and not timeout
    failure = None if success else ("timeout" if timeout else "approach_collision" if np.any(approach) else "no_contact" if not acquired else "failed_acquisition" if lift < limit["minimum_lift_m"] else "drop" if drop > limit["maximum_object_drop_m"] else "slip_during_hold")
    manual = dict(success=bool(success), failure_type=failure, acquired_object=acquired, maximum_lift_m=lift,
                  hold_duration_s=hold, final_object_height_m=float(heights[-1]), approach_collision=bool(np.any(approach)))
    scored = read(ROOT / "cases" / (aid + "_scored_outcome.json"))
    assert set(manual) == set(scored)
    for key, value in manual.items():
        if isinstance(value, float):
            assert abs(value-scored[key]) <= 1e-12
        else:
            assert value == scored[key]
    first = float(times[np.flatnonzero(approach)[0]] - .002) if failure == "approach_collision" else None
    assert row["outcome"]["success"] == success and row["outcome"]["failure_type"] == failure
    assert row["outcome"]["first_failure_time_s"] == first
    eligible = failure != "approach_collision" and (first is None or first > 2.96 + 1e-9)
    labels[aid] = (int(success), eligible)
    records[aid] = (obs, diag, scored, meta)
    if row["repeat_of"]:
        first_record = records[row["repeat_of"]]
        assert first_record[2] == scored
        errors = {}
        for group, first_group, second_group in (("observables", first_record[0], obs), ("diagnostics", first_record[1], diag)):
            assert set(first_group) == set(second_group)
            for name, values in first_group.items():
                other = second_group[name]
                if values.dtype.kind in "fiu":
                    assert values.shape == other.shape and np.isfinite(values).all() and np.isfinite(other).all()
                    error = float(np.max(np.abs(values-other)))
                    assert error <= 1e-10
                    errors[group + "/" + name] = error
                else:
                    assert np.array_equal(values, other)
        repeats.append({"attempt_id": aid, "repeat_of": row["repeat_of"], "scored_outcome_exact": True,
                        "maximum_recorded_endpoint_errors": errors})
    per_case.append({"attempt_id": aid, "label": int(success), "failure_type": failure,
                     "eligible_prefix": eligible, "recording_hashes_verified": True,
                     "manual_outcome_matches": True, "joint_qpos_qvel_order_matches_saved_model": True})

assert len(repeats) == 4
manifest = read(ROOT / "derived_prefixes/manifest.json")
data = archive(ROOT / "derived_prefixes/dataset.npz")
assert sha(ROOT / "derived_prefixes/dataset.npz") == manifest["dataset_sha256"]
assert set(data) == {"X", "y", "episode_ids", "group_ids", "observation_group_ids", "geometry_group_ids"}
assert data["X"].dtype == np.float32 and data["X"].shape == (15, 1440)
included = [x["attempt_id"] for x in manifest["episodes"]]
excluded = [x["attempt_id"] for x in manifest["excluded"]]
assert set(included) == {aid for aid, (_, eligible) in labels.items() if eligible}
assert set(excluded) == {aid for aid, (_, eligible) in labels.items() if not eligible}
assert all(x["reason"] == "failure_at_or_before_cutoff" for x in manifest["excluded"])
assert included == data["episode_ids"].tolist()
assert manifest["input_contract"]["fields"] == ["hand_qpos", "hand_qvel", "applied_target_rad"]
assert manifest["input_contract"]["history_steps"] == 20 and manifest["input_contract"]["prediction_time_s"] == 2.96
for index, aid in enumerate(included):
    obs, diag, scored, meta = records[aid]
    selected = np.flatnonzero(obs["time_s"] <= 2.96 + 1e-9)[-20:]
    assert len(selected) == 20 and abs(obs["time_s"][selected[-1]]-2.96) < 1e-9
    assert np.all(obs["phase"][selected] == "close_fingers")
    assert not np.any(np.isin(obs["phase"][obs["time_s"] <= 2.96 + 1e-9], ["lift", "hold", "score"]))
    manual = np.concatenate([obs[field][selected] for field in ("hand_qpos", "hand_qvel", "applied_target_rad")], axis=1).astype(np.float32).reshape(-1)
    np.testing.assert_array_equal(manual, data["X"][index])
    assert int(data["y"][index]) == labels[aid][0]
    assert data["group_ids"][index] == meta["proposal_group_id"]
    assert data["observation_group_ids"][index] == meta["observation_group_id"]
    assert data["geometry_group_ids"][index] == meta["geometry_group_id"]
    assert manifest["episodes"][index]["provenance_hashes"] == meta["provenance_hashes"]
    assert manifest["input_contract"]["joint_names"] == obs["joint_names"].tolist()

smoke = read(ROOT / "model_input_smoke.json")
assert smoke["model_training_performed"] is False and "compatibility" in smoke["scope"]
model_checks = {}
for name, saved in smoke["models"].items():
    model_path = Path(protocol["saved_models_dir"]) / (name + ".npz")
    assert sha(model_path) == saved["model_sha256"] == protocol["saved_model_bindings"][str(model_path)]
    weights = archive(model_path)
    history = data["X"].astype(float).reshape(len(included), 20, 72)
    if name == "commands_only":
        history = history[:, :, 48:]
    features = np.concatenate((history[:, -1], history.mean(axis=1), history.std(axis=1), history[:, -1]-history[:, 0]), axis=1)
    scaled = ((features-weights["mean"])/weights["scale"])[:, weights["active"]]
    logits = np.sum(scaled*weights["weights"], axis=1)+float(weights["intercept"])
    probabilities = 1/(1+np.exp(-logits))
    error = float(np.max(np.abs(probabilities-np.asarray(saved["p_success"]))))
    assert error < 1e-12
    y = data["y"].astype(bool)
    alert = 1-probabilities >= .5
    confusion = dict(true_positive=int(np.sum(alert & ~y)), false_positive=int(np.sum(alert & y)), true_negative=int(np.sum(~alert & y)), false_negative=int(np.sum(~alert & ~y)))
    assert confusion == saved["smoke_metrics"]["failure_confusion"]
    pos, neg = (1-probabilities)[~y], (1-probabilities)[y]
    auc = float(np.mean((pos[:, None] > neg).astype(float) + .5*(pos[:, None] == neg)))
    bounded = np.clip(probabilities, 1e-12, 1-1e-12)
    truth = y.astype(float)
    loss = float(np.mean(-truth*np.log(bounded)-(1-truth)*np.log1p(-bounded)))
    brier = float(np.mean((probabilities-y)**2))
    assert abs(auc-saved["smoke_metrics"]["failure_roc_auc"]) < 1e-12
    assert abs(loss-saved["smoke_metrics"]["log_loss"]) < 1e-12
    assert abs(brier-saved["smoke_metrics"]["brier_score"]) < 1e-12
    model_checks[name] = dict(max_probability_error=error, independent_auc=auc,
                             independent_log_loss=loss, independent_brier=brier,
                             failure_confusion=confusion, scope="compatibility smoke only; reused development inputs, no fit")

assert summary["physics_steps"] == 20*3260 == protocol["quota"]["maximum_total_physics_steps"]
assert summary["new_inference_calls"] == summary["new_retarget_solves"] == 0
for group in ("source_bindings", "input_bindings", "saved_model_bindings"):
    for path, expected in protocol[group].items():
        assert sha(path) == expected
receipt = {"schema": "hug_saved_pilot_independent_scalar_receipt_v1", "status": "PASS",
           "source_sha256": sha(__file__), "protocol_sha256": sha(ROOT/"protocol.json"),
           "authorization_sha256": sha(ROOT/"authorization.json"), "summary_sha256": sha(ROOT/"summary.json"),
           "attempts_accounted": 20, "completed": 20, "null_outcomes": 0,
           "unique_saved_commands": 16, "repeats": repeats, "successes": sum(v[0] for v in labels.values()),
           "physical_failures": sum(1-v[0] for v in labels.values()), "excluded_pre_cutoff_approach": len(excluded),
           "included_prefixes": len(included), "feature_shape": list(data["X"].shape),
           "all_manual_task_outcomes_match": True, "all_features_match_allowed_causal_channels_exactly": True,
           "all_source_input_model_hashes_unchanged": True, "model_checks": model_checks,
           "cases": per_case, "receipt_physics_steps": 0, "receipt_static_forwards": 0,
           "compiled_model_reads_only": 20,
           "limits": ["repeat comparison covers recorded163 control endpoints and diagnostic full qpos/qvel, not every physics substep", "first_failure_time_s is first sampled approach contact diagnostic time, not exact continuous onset", "20attempts reuse16development proposals with4repeats; smoke metrics are not independent generalization results"]}
OUT.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
print(json.dumps({key: value for key, value in receipt.items() if key not in ("cases", "repeats")}, indent=2))
