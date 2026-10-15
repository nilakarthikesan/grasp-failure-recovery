"""Manufactured array controls only: no model, native FK or integration."""
from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from grasp_failure_prediction.evaluation.manosim_recording import (
    BALLS, CTRL_NAMES, NATIVE_DOMAIN, OBSERVABLE_KEYS, QPOS_NAMES, QVEL_NAMES,
    NativeAttemptMetadata, NativeRecorder, NativeRecordingContract,
    canonical_sha256, file_sha256, load_native_attempt, native_admission,
    validate_native_observables,
)


HASH = "sha256:" + "a" * 64
OTHER_HASH = "sha256:" + "b" * 64


@pytest.fixture(autouse=True)
def no_native_operations(monkeypatch):
    import mujoco
    def blocked(*args, **kwargs):
        raise AssertionError("manufactured recording controls prohibit native model/FK/physics")
    for name in ("MjModel", "MjData", "mj_forward", "mj_kinematics", "mj_fwdPosition", "mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, blocked)


def contract(**changes):
    payload = dict(assets_sha256=HASH, shape_sha256=HASH, collider_sha256=HASH,
        controller_sha256=HASH, compiled_model_sha256=HASH,
        controller_protocol_id="manufactured_native_controller_v1",
        sampling_protocol_id="manufactured_endpoints_v1", control_interval_s=.04)
    payload.update(changes)
    return NativeRecordingContract(**payload)


def named_qpos(sign=1.):
    result = dict.fromkeys(QPOS_NAMES, 0.)
    result.update({f"{joint}:quat_w": sign for joint in BALLS})
    return result


def append(recorder, *, index=1, phase="close_fingers", **changes):
    values = dict(time_s=round(index * .04, 10), action_interval_start_s=round((index - 1) * .04, 10),
        phase=phase, qpos_by_name=named_qpos(), qvel_by_name=dict.fromkeys(QVEL_NAMES, .125),
        requested_ctrl_by_name=dict.fromkeys(CTRL_NAMES, 2.), effective_ctrl_by_name=dict.fromkeys(CTRL_NAMES, .75))
    values.update(changes)
    recorder.append(**values)


def metadata(c=None, *, status="runtime_error", mapping="ok", **changes):
    c = c or contract()
    decision = {"status": mapping}
    if mapping != "ok":
        decision.update(code="manufactured_refusal", message="manufactured native refusal")
    payload = dict(attempt_id="new_native_attempt", contract_sha256=c.contract_sha256,
        object_id="manufactured_object", geometry_group_id="geometry_family", observation_group_id="capture_family",
        proposal_group_id="native_family", mapping=decision, execution_status=status,
        provenance_hashes={key: HASH for key in ("code", "configuration", "inputs", "native_pose", "label_rule")})
    if status == "completed":
        payload["outcome"] = dict(success=True, label_rule_id="manufactured_native_rule_v1")
    payload.update(changes)
    return NativeAttemptMetadata(**payload)


def test_contract_has_independent_named_layouts_and_mixed_units():
    c = contract()
    assert (len(c.qpos_channels), len(c.qvel_channels), len(c.ctrl_channels)) == (67, 51, 51)
    assert tuple(x.name for x in c.qpos_channels) == QPOS_NAMES
    assert tuple(x.name for x in c.qvel_channels) == QVEL_NAMES
    assert tuple(x.name for x in c.ctrl_channels) == CTRL_NAMES
    assert [x.unit for x in c.qpos_channels[:4]] == ["m", "m", "m", "1"]
    assert [x.unit for x in c.qvel_channels[:4]] == ["m/s", "m/s", "m/s", "rad/s"]
    assert [x.unit for x in c.ctrl_channels[:4]] == ["m", "m", "m", "rad"]
    assert c.contract_sha256 == canonical_sha256(c.model_dump(mode="json"))
    with pytest.raises(AttributeError):
        NativeRecorder(c).contract = contract(shape_sha256=OTHER_HASH)


@pytest.mark.parametrize("field", ["assets_sha256", "shape_sha256", "collider_sha256", "controller_sha256", "compiled_model_sha256",
                                  "controller_protocol_id", "sampling_protocol_id", "control_interval_s"])
def test_every_identity_or_sampling_change_changes_contract_digest(field):
    replacement = OTHER_HASH if field.endswith("sha256") else .02 if field == "control_interval_s" else "different_v2"
    assert contract(**{field: replacement}).contract_sha256 != contract().contract_sha256


@pytest.mark.parametrize("changes", [
    {"domain": "shadow_hand_right"}, {"hand_id": "shadow_hand_right"}, {"handedness": "left"},
    {"shape_sha256": "a" * 64}, {"assets_sha256": "sha256:" + "A" * 64},
    {"control_interval_s": float("inf")}, {"control_interval_s": True}, {"control_interval_s": 0.},
    {"controller_protocol_id": ""}, {"upstream_revision": "unknown"}, {"control_policy": "clip_shadow_targets"},
    {"quaternion_norm_atol": .1}, {"clock_atol_s": .1},
])
def test_unknown_domains_hashes_units_and_sampling_refused(changes):
    with pytest.raises(ValidationError):
        contract(**changes)


@pytest.mark.parametrize("field,edit", [
    ("qpos_channels", lambda a: a[:51]), ("qvel_channels", lambda a: a + a[:16]),
    ("ctrl_channels", lambda a: list(reversed(a))),
    ("qpos_channels", lambda a: [dict(a[0], unit="mm")] + a[1:]),
    ("qvel_channels", lambda a: [dict(a[0], frame="world")] + a[1:]),
    ("ctrl_channels", lambda a: [dict(a[0], name="wrist_tx")] + a[1:]),
])
def test_equal_width_coercion_aliases_and_channel_semantics_refused(field, edit):
    values = contract().model_dump(mode="json")[field]
    with pytest.raises(ValidationError, match="67/51/51"):
        contract(**{field: edit(values)})


def test_admission_remains_closed_even_with_asserted_exact_contract_pass():
    c = contract()
    assert not native_admission(c)["admitted"]
    result = native_admission(c, {"contract_sha256": c.contract_sha256, "status": "PASS", "admitted": True})
    assert not result["admitted"] and result["evidence_contract_matches"]
    assert not native_admission(c, {"contract_sha256": OTHER_HASH})["evidence_contract_matches"]
    assert not native_admission(c, {"contract_sha256": np.asarray([c.contract_sha256])})["admitted"]


def test_named_maps_reordered_and_copied_without_quaternion_sign_or_control_coercion():
    recorder = NativeRecorder(contract())
    qpos = named_qpos(-1.)
    append(recorder, qpos_by_name=dict(reversed(list(qpos.items()))))
    qpos[QPOS_NAMES[0]] = 99.
    arrays = recorder.arrays()
    assert arrays["hand_qpos"].shape == (1, 67)
    assert arrays["hand_qvel"].shape == arrays["requested_ctrl"].shape == arrays["effective_ctrl"].shape == (1, 51)
    assert arrays["hand_qpos"][0, 0] == 0. and arrays["hand_qpos"][0, 3] == -1.
    assert np.all(arrays["requested_ctrl"] == 2.) and np.all(arrays["effective_ctrl"] == .75)
    arrays["hand_qpos"][0, 0] = 55.
    assert recorder.arrays()["hand_qpos"][0, 0] == 0.


@pytest.mark.parametrize("field,value", [
    ("qpos_by_name", np.zeros(67)), ("qvel_by_name", np.zeros(51)),
    ("requested_ctrl_by_name", dict.fromkeys(QVEL_NAMES, 0.)),
    ("effective_ctrl_by_name", dict.fromkeys(QPOS_NAMES, 0.)),
    ("qpos_by_name", dict.fromkeys(QPOS_NAMES[:24], 0.)),
    ("qvel_by_name", dict(dict.fromkeys(QVEL_NAMES, 0.), joint_alias=0.)),
])
def test_flat_vectors_swapped_equal_width_names_and_shadow_width_refused(field, value):
    recorder = NativeRecorder(contract())
    with pytest.raises(ValueError, match="exact native named"):
        append(recorder, **{field: value})
    assert recorder.sample_count == 0


@pytest.mark.parametrize("field,key,value", [
    ("qpos_by_name", QPOS_NAMES[0], True), ("qpos_by_name", QPOS_NAMES[0], [0.]),
    ("qvel_by_name", QVEL_NAMES[0], float("nan")), ("requested_ctrl_by_name", CTRL_NAMES[0], float("inf")),
    ("effective_ctrl_by_name", CTRL_NAMES[0], "1"), ("qpos_by_name", QPOS_NAMES[3], 0.),
    ("qpos_by_name", QPOS_NAMES[3], 1.0001),
])
def test_nonfinite_nonscalar_and_nonunit_raw_coordinates_refused(field, key, value):
    maps = {"qpos_by_name": named_qpos(), "qvel_by_name": dict.fromkeys(QVEL_NAMES, 0.),
            "requested_ctrl_by_name": dict.fromkeys(CTRL_NAMES, 0.), "effective_ctrl_by_name": dict.fromkeys(CTRL_NAMES, 0.)}
    maps[field][key] = value
    with pytest.raises(ValueError):
        append(NativeRecorder(contract()), **maps)


def test_phase_clock_and_fixed_endpoint_alignment():
    r = NativeRecorder(contract())
    append(r); append(r, index=2); append(r, index=3, phase="lift")
    a = r.arrays()
    np.testing.assert_array_equal(a["phase_start_s"], [0., 0., .08])
    np.testing.assert_allclose(a["phase_elapsed_s"], [.04, .08, .04], rtol=0, atol=1e-15)
    for changes in ({"time_s": .17}, {"action_interval_start_s": .121}, {"phase": "unknown"}, {"time_s": True}):
        with pytest.raises(ValueError):
            append(r, index=4, **changes)
    assert r.sample_count == 3


@pytest.mark.parametrize("key,edit", [
    ("hand_qpos", lambda a: a[:, :51]), ("hand_qvel", lambda a: np.zeros((1, 67))),
    ("ctrl_names", lambda a: np.asarray(QVEL_NAMES)), ("domain", lambda a: np.asarray("shadow_hand_right")),
    ("contract_sha256", lambda a: np.asarray(OTHER_HASH)), ("phase_start_s", lambda a: np.asarray([.01])),
    ("phase_elapsed_s", lambda a: np.asarray([0.])), ("hand_qvel", lambda a: a * np.inf),
])
def test_loaded_array_validation_cannot_bypass_named_contract(key, edit):
    r = NativeRecorder(contract()); append(r)
    arrays = r.arrays(); arrays[key] = edit(arrays[key])
    with pytest.raises(ValueError):
        validate_native_observables(arrays, contract())


def test_diagnostics_independent_clock_and_no_observable_features(tmp_path):
    c = contract(); r = NativeRecorder(c); append(r)
    r.append_diagnostics(time_s=.038, values={"contacts": np.array([1., 2.]), "full_scene_qpos": np.arange(80.)})
    r.append_diagnostics(time_s=.039, values={"contacts": np.array([3., 4.]), "full_scene_qpos": np.arange(80.)})
    assert set(r.arrays()) == OBSERVABLE_KEYS
    payload = r.write(tmp_path / "native", metadata(c))
    meta, loaded_contract, _, observables = load_native_attempt(tmp_path / "native")
    assert loaded_contract == c and meta.outcome is None and not payload["native_admission"]["admitted"]
    assert "contacts" not in observables and "full_scene_qpos" not in observables
    with np.load(tmp_path / "native" / "diagnostics.npz", allow_pickle=False) as diagnostic:
        np.testing.assert_array_equal(diagnostic["time_s"], [.038, .039])
    arrays = r.arrays(); arrays["contact_count"] = np.array([2.])
    with pytest.raises(ValueError, match="diagnostics are forbidden"):
        validate_native_observables(arrays, c)


def test_diagnostic_fields_shapes_time_and_finalization_are_strict(tmp_path):
    r = NativeRecorder(contract())
    r.append_diagnostics(time_s=0., values={"x": np.zeros(2)})
    for time, values in ((0., {"x": np.zeros(2)}), (.01, {"y": np.zeros(2)}),
                          (.01, {"x": np.zeros(3)}), (.01, {"time_s": 0.}), (.01, {"x": [np.nan]})):
        with pytest.raises(ValueError):
            r.append_diagnostics(time_s=time, values=values)
    r.write(tmp_path / "runtime", metadata())
    assert load_native_attempt(tmp_path / "runtime")[3] is None
    with pytest.raises(ValueError, match="finalized"):
        append(r)
    with pytest.raises(ValueError, match="finalized"):
        r.append_diagnostics(time_s=.01, values={"x": np.zeros(2)})


@pytest.mark.parametrize("mapping", ["refused_invalid_retarget", "refused_guard", "unreachable_pregrasp", "setup_error"])
def test_mapping_setup_refusals_persist_null_without_fake_sample(tmp_path, mapping):
    c = contract(); m = metadata(c, mapping=mapping, status="not_executed")
    p = NativeRecorder(c).write(tmp_path / "refusal", m)
    assert p["attempt"]["outcome"] is None and p["recording"] is None and p["artifacts_sha256"] == {}
    assert load_native_attempt(tmp_path / "refusal")[3] is None
    with pytest.raises(ValidationError, match="outcome null"):
        metadata(c, mapping=mapping, status="completed")


@pytest.mark.parametrize("status", ["not_executed", "runtime_error"])
def test_unexecuted_and_runtime_cannot_be_physical_failure_labels(status):
    with pytest.raises(ValidationError, match="must be null"):
        metadata(status=status, outcome={"success": False, "failure_type": "drop", "label_rule_id": "rule_v1"})


def test_refusal_with_samples_completed_without_samples_wrong_contract_and_mutated_metadata_refused(tmp_path):
    c = contract(); r = NativeRecorder(c); append(r)
    with pytest.raises(ValueError, match="unexecuted native refusal"):
        r.write(tmp_path / "bad1", metadata(c, status="not_executed"))
    with pytest.raises(ValueError, match="requires samples"):
        NativeRecorder(c).write(tmp_path / "bad2", metadata(c, status="completed"))
    with pytest.raises(ValueError, match="contract mismatch"):
        r.write(tmp_path / "bad3", metadata(c, contract_sha256=OTHER_HASH))
    m = metadata(c); m.provenance_hashes["inputs"] = "invalid"
    with pytest.raises(ValidationError):
        r.write(tmp_path / "bad4", m)
    assert not any(tmp_path.iterdir())


def test_completed_metadata_roundtrip_has_separate_label_without_native_certification(tmp_path):
    c = contract(); r = NativeRecorder(c); append(r)
    p = r.write(tmp_path / "completed", metadata(c, status="completed"))
    m, _, _, a = load_native_attempt(tmp_path / "completed")
    assert m.outcome.success and a["hand_qpos"].shape == (1, 67)
    assert not p["native_admission"]["admitted"]
    assert "outcome" not in a
    with pytest.raises(ValueError, match="finalized"):
        r.write(tmp_path / "second", m)
    with pytest.raises(ValueError, match="overwrite"):
        NativeRecorder(c).write(tmp_path / "completed", metadata(c))


@pytest.mark.parametrize("edit", [
    lambda p: p.update(schema_version="hug_prediction_recording_v1"),
    lambda p: p["contract"].update(shape_sha256=OTHER_HASH),
    lambda p: p["attempt"].update(hand_id="shadow_hand_right"),
    lambda p: p["native_admission"].update(admitted=True),
    lambda p: p.update(recording={"samples": 999}),
    lambda p: p.update(recording={"samples": True}),
    lambda p: p["artifacts_sha256"].update(unknown_npz=HASH),
    lambda p: p.update(diagnostics_policy="predictor features allowed"),
    lambda p: p.update(shadow_joint_count=24),
])
def test_attempt_reader_rejects_schema_contract_admission_or_count_tamper(tmp_path, edit):
    r = NativeRecorder(contract()); append(r); r.write(tmp_path, metadata())
    path = tmp_path / "attempt.json"; p = json.loads(path.read_text()); edit(p); path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        load_native_attempt(tmp_path)


@pytest.mark.parametrize("artifact", ["observables.npz", "diagnostics.npz"])
def test_all_artifact_bytes_verified_including_unopened_diagnostics(tmp_path, artifact):
    r = NativeRecorder(contract()); append(r); r.append_diagnostics(time_s=.038, values={"privileged": [1.]})
    r.write(tmp_path, metadata())
    with (tmp_path / artifact).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_native_attempt(tmp_path)


@pytest.mark.parametrize("kind", ["extra_npz", "omitted_diagnostic", "symlink_observable", "symlink_marker"])
def test_unindexed_or_symlink_artifacts_cannot_be_silently_dropped(tmp_path, kind):
    root = tmp_path / "native"; r = NativeRecorder(contract()); append(r)
    r.append_diagnostics(time_s=.038, values={"contact": [1.]}); r.write(root, metadata())
    if kind == "extra_npz":
        np.savez_compressed(root / "extra.npz", extra=[1.])
    elif kind == "omitted_diagnostic":
        p = json.loads((root / "attempt.json").read_text()); del p["artifacts_sha256"]["diagnostics.npz"]
        (root / "attempt.json").write_text(json.dumps(p))
    else:
        name = "observables.npz" if kind == "symlink_observable" else "attempt.json"
        source = root / name; moved = tmp_path / name; source.rename(moved); source.symlink_to(moved)
    with pytest.raises(ValueError, match="unindexed|symlink|hash mismatch"):
        load_native_attempt(root)


def test_rehashed_observables_cannot_smuggle_diagnostics_or_equal_width_swap(tmp_path):
    r = NativeRecorder(contract()); append(r); r.write(tmp_path, metadata())
    path = tmp_path / "observables.npz"
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files}
    values["privileged_contacts"] = np.zeros(1)
    np.savez_compressed(path, **values)
    p = json.loads((tmp_path / "attempt.json").read_text()); p["artifacts_sha256"][path.name] = file_sha256(path)
    (tmp_path / "attempt.json").write_text(json.dumps(p))
    with pytest.raises(ValueError, match="allowlist"):
        load_native_attempt(tmp_path)
