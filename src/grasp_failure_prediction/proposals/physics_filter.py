"""Prepared six-axis, closed-state screening protocol ``shadow_physics_filter_v1``.

This is metadata about a proposal, never a task outcome or a predictor feature.
No approach, closing, refitting, recovery or task rollout is performed here. Each
axis gets a separate model and MjData restored from the SAME mjSTATE_INTEGRATION
snapshot (including controls, mocap, applied forces and solver warm start).

Predeclared engineering defaults, not thresholds fitted to outcomes:
* Six world-axis gravities of 9.81 m/s², held independently for 1 s.
* At most 20 mm object-origin translation relative to the moving palm frame;
  this uses the existing task's 20 mm final-loss scale, without borrowing labels.
* Two distinct distal fingers above 0.01 N normal force in >=90% of EACH 200 ms
  window, with no unsupported interval >40 ms. The force floor follows the
  runner's contact significance convention; 40 ms is one controller interval.
  Palm support is reported separately and cannot substitute for two fingers.
* At most 5 mm negative MuJoCo hand/object contact distance. Soft-contact models
  permit overlap: this is a conservative screening cap (10% of the 50 mm test
  cube), not exact surface penetration or a universal contact-fidelity bound.
* An optional initial static table-plane clearance gate reuses compiled-geometry
  support from table_clearance.py. Declared table collisions are disabled ONLY
  on filter copies, so six-axis retention cannot be supplied by the table.

No test split or later rollout outcome enters this module. A pass is not proof
of force closure, physical success, observed surface fidelity or sim-to-real
transfer. All rejected/error proposals must be retained by the caller.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import hashlib
from importlib.resources import files
import math
import time
from typing import Any, Mapping

import mujoco
import numpy as np
import yaml


AXES = (("pos_x", (1, 0, 0)), ("neg_x", (-1, 0, 0)),
        ("pos_y", (0, 1, 0)), ("neg_y", (0, -1, 0)),
        ("pos_z", (0, 0, 1)), ("neg_z", (0, 0, -1)))
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass(frozen=True)
class PhysicsFilterSpec:
    schema_version: str = "physics_filter_spec_v1"
    filter_id: str = "shadow_physics_filter_v1"
    gravity_m_s2: float = 9.81
    hold_s: float = 1.0
    maximum_displacement_m: float = 0.020
    minimum_normal_force_n: float = 0.01
    minimum_distal_fingers: int = 2
    contact_window_s: float = 0.200
    minimum_supported_fraction: float = 0.90
    maximum_unsupported_s: float = 0.040
    maximum_penetration_m: float = 0.005
    maximum_steps_per_axis: int = 1000
    maximum_total_steps: int = 6000
    maximum_wall_s: float = 60.0
    time_increment_atol_s: float = 1e-12
    disable_declared_table_collisions: bool = True
    minimum_table_clearance_m: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "physics_filter_spec_v1":
            raise ValueError("unsupported physics filter schema")
        if not isinstance(self.filter_id, str) or not self.filter_id.strip():
            raise ValueError("filter_id must be a nonempty versioned protocol identity")
        for name in ("gravity_m_s2", "hold_s", "maximum_displacement_m",
                     "minimum_normal_force_n", "contact_window_s",
                     "maximum_penetration_m", "maximum_wall_s", "time_increment_atol_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("minimum_distal_fingers", "maximum_steps_per_axis", "maximum_total_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 1 <= self.minimum_distal_fingers <= 5:
            raise ValueError("minimum_distal_fingers must be in [1, 5]")
        if (isinstance(self.minimum_supported_fraction, bool)
                or not np.isfinite(self.minimum_supported_fraction)
                or not 0 <= self.minimum_supported_fraction <= 1):
            raise ValueError("minimum_supported_fraction must be in [0, 1]")
        if (isinstance(self.maximum_unsupported_s, bool)
                or not np.isfinite(self.maximum_unsupported_s) or self.maximum_unsupported_s < 0):
            raise ValueError("maximum_unsupported_s must be finite and nonnegative")
        if type(self.disable_declared_table_collisions) is not bool:
            raise ValueError("disable_declared_table_collisions must be boolean")
        if self.minimum_table_clearance_m is not None and (
            isinstance(self.minimum_table_clearance_m, bool)
            or not np.isfinite(self.minimum_table_clearance_m) or self.minimum_table_clearance_m < 0
        ):
            raise ValueError("minimum_table_clearance_m must be finite and nonnegative")


def load_physics_filter_spec() -> PhysicsFilterSpec:
    resource = files(__package__).joinpath("configs", "physics_filter.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"shadow_physics_filter_v1"}:
        raise ValueError("physics filter registry must contain the declared v1 identity")
    return PhysicsFilterSpec(**raw["shadow_physics_filter_v1"])


@dataclass(frozen=True)
class PhysicsFilterGeometry:
    object_body_id: int
    reference_body_id: int
    object_geom_ids: tuple[int, ...]
    hand_geom_ids: tuple[int, ...]
    distal_finger_geom_ids: Mapping[str, tuple[int, ...]]
    palm_geom_ids: tuple[int, ...]
    table_geom_ids: tuple[int, ...] = ()
    table_top_z_m: float | None = None

    @classmethod
    def from_shadow_model(cls, model: mujoco.MjModel) -> "PhysicsFilterGeometry":
        """Explicit current runner naming contract; no positional joint guessing."""
        def body(name: str) -> int:
            value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if value < 0:
                raise ValueError(f"missing named Shadow body: {name}")
            return int(value)

        def geoms(ancestor: int) -> tuple[int, ...]:
            return tuple(i for i in range(model.ngeom)
                         if _descendant(model, int(model.geom_bodyid[i]), ancestor)
                         and (model.geom_contype[i] or model.geom_conaffinity[i]))

        object_id, palm, root = body("object"), body("palm"), body("forearm")
        fingers = {label: geoms(body(name)) for label, name in
                   (("thumb", "thdistal"), ("index", "ffdistal"),
                    ("middle", "mfdistal"), ("ring", "rfdistal"), ("little", "lfdistal"))}
        palm_geoms = tuple(i for i in geoms(palm) if int(model.geom_bodyid[i]) == palm)
        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        # Caller must supply table_top_z_m to enable the optional plane gate.
        return cls(object_id, palm, geoms(object_id), geoms(root), fingers,
                   palm_geoms, () if table < 0 else (int(table),))


def _descendant(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


@dataclass
class PhysicsFilterResult:
    filter_status: str
    configuration: dict[str, Any]
    provenance: dict[str, Any]
    axes: list[dict[str, Any]] = field(default_factory=list)
    rejection_codes: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "physics_filter_result_v1"

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Sample:
    relative_position: np.ndarray
    relative_rotation: np.ndarray
    supported_regions: frozenset[str]
    penetration_m: float
    external_support_n: float


def _integration_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    result = np.empty(mujoco.mj_stateSize(model, STATE), dtype=np.float64)
    mujoco.mj_getState(model, data, result, STATE)
    return result


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _warnings(data: mujoco.MjData) -> np.ndarray:
    return np.asarray(data.warning.number).copy()


def _guard(data: mujoco.MjData, previous_warnings: np.ndarray | None = None) -> None:
    for name in ("qpos", "qvel", "qacc", "ctrl", "act", "mocap_pos", "mocap_quat"):
        if not np.all(np.isfinite(getattr(data, name))):
            raise RuntimeError(f"nonfinite {name}")
    if not np.isfinite(data.time):
        raise RuntimeError("nonfinite simulation time")
    warnings = _warnings(data)
    if np.any(warnings if previous_warnings is None else warnings != previous_warnings):
        raise RuntimeError(f"MuJoCo warning counters: {warnings.tolist()}")


def _sample(model: mujoco.MjModel, data: mujoco.MjData,
            geometry: PhysicsFilterGeometry, force_floor: float) -> _Sample:
    reference = data.xmat[geometry.reference_body_id].reshape(3, 3)
    relative = reference.T @ (data.xpos[geometry.object_body_id] - data.xpos[geometry.reference_body_id])
    rotation = reference.T @ data.xmat[geometry.object_body_id].reshape(3, 3)
    objects, hands = set(geometry.object_geom_ids), set(geometry.hand_geom_ids)
    regions = {i: name for name, ids in geometry.distal_finger_geom_ids.items() for i in ids}
    regions.update({i: "palm" for i in geometry.palm_geom_ids})
    supported, penetration, external = set(), 0.0, 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if not ({first, second} & objects) or {first, second} <= objects:
            continue
        other = second if first in objects else first
        distance = float(contact.dist)
        if not math.isfinite(distance):
            raise RuntimeError("nonfinite contact distance")
        if other in hands:
            penetration = max(penetration, -distance)
        # ncon can include inactive margin contacts; they do not supply support.
        if int(contact.efc_address) < 0:
            continue
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        if not np.all(np.isfinite(force)):
            raise RuntimeError("nonfinite contact force")
        normal = max(0.0, float(force[0]))
        if other in hands and normal > force_floor and other in regions:
            supported.add(regions[other])
        if other not in hands:
            external = max(external, normal)
    if not np.all(np.isfinite(relative)) or not np.all(np.isfinite(rotation)):
        raise RuntimeError("nonfinite body frame")
    return _Sample(relative, rotation, frozenset(supported), penetration, external)


def _validate_geometry(model: mujoco.MjModel, data: mujoco.MjData,
                       geometry: PhysicsFilterGeometry, spec: PhysicsFilterSpec) -> None:
    for value in (geometry.object_body_id, geometry.reference_body_id):
        if not 0 < value < model.nbody:
            raise ValueError("object/reference must be existing non-world bodies")
    groups = (geometry.object_geom_ids, geometry.hand_geom_ids, geometry.table_geom_ids)
    if not groups[0] or not groups[1]:
        raise ValueError("object and hand collision geometry must be declared")
    for group in groups:
        if len(set(group)) != len(group) or any(type(i) is not int or not 0 <= i < model.ngeom for i in group):
            raise ValueError("geometry IDs must be distinct valid integers")
    objects, hands, tables = map(set, groups)
    if objects & hands or objects & tables or hands & tables:
        raise ValueError("object, hand and table geometry must be disjoint")
    object_bodies = {i for i in range(model.nbody) if _descendant(model, i, geometry.object_body_id)}
    expected = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in object_bodies
                and (model.geom_contype[i] or model.geom_conaffinity[i])}
    if objects != expected:
        raise ValueError("all active object collision geometry must be declared exactly")
    if any(int(model.geom_bodyid[i]) in object_bodies for i in hands):
        raise ValueError("hand geometry cannot belong to the object")
    region_ids: list[int] = []
    if "palm" in geometry.distal_finger_geom_ids:
        raise ValueError("palm is a separate region")
    if len(geometry.distal_finger_geom_ids) < spec.minimum_distal_fingers:
        raise ValueError("not enough declared distal finger regions")
    for ids in (*geometry.distal_finger_geom_ids.values(), geometry.palm_geom_ids):
        if not ids or not set(ids) <= hands:
            raise ValueError("support region geometry must be a nonempty subset of the hand")
        region_ids.extend(ids)
    if len(set(region_ids)) != len(region_ids):
        raise ValueError("a geom cannot supply multiple support regions")
    start, count = int(model.body_jntadr[geometry.object_body_id]), int(model.body_jntnum[geometry.object_body_id])
    joints = range(start, start + count)
    if count != 1 or model.jnt_type[start] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("object root must own one unconstrained free joint")
    dof = int(model.jnt_dofadr[start])
    if np.any(data.qfrc_applied[dof:dof + 6]) or np.any(data.xfrc_applied[list(object_bodies)]):
        raise ValueError("object external applied forces would confound gravity retention")
    if np.any(model.body_gravcomp[list(object_bodies)]):
        raise ValueError("object gravity compensation is forbidden")
    if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY):
        raise ValueError("gravity must be enabled")
    for i in range(model.neq):
        if data.eq_active[i] and (int(model.eq_obj1id[i]) in object_bodies
                                 or int(model.eq_obj2id[i]) in object_bodies):
            # Conservative: object-attached equality constraints need a new protocol.
            raise ValueError("active equality involving object is forbidden")
    for i in range(model.nu):
        if (model.actuator_trntype[i] in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT)
                and int(model.actuator_trnid[i, 0]) in joints):
            raise ValueError("object joint actuation is forbidden")
    if spec.disable_declared_table_collisions:
        for a, b in zip(model.pair_geom1, model.pair_geom2):
            if int(a) in tables or int(b) in tables:
                raise ValueError("explicit table contact pairs cannot be disabled by collision masks")
    if spec.minimum_table_clearance_m is not None:
        if not tables or geometry.table_top_z_m is None or not np.isfinite(geometry.table_top_z_m):
            raise ValueError("table plane gate needs declared table geoms and finite top Z")


def _window_records(samples: list[frozenset[str]], dt: float, window_steps: int,
                    minimum_fingers: int, region_names: tuple[str, ...] | None = None
                    ) -> tuple[list[dict[str, Any]], float]:
    supported = [len(regions - {"palm"}) >= minimum_fingers for regions in samples]
    records = []
    names = region_names or tuple(sorted(set().union(*samples) | {"palm"}))
    for start in range(0, len(samples), window_steps):
        end = min(start + window_steps, len(samples))
        finger_counts = [len(regions - {"palm"}) for regions in samples[start:end]]
        records.append({"start_s": start * dt, "end_s": end * dt,
                        "sample_count": end - start,
                        "supported_fraction": sum(supported[start:end]) / (end - start),
                        "minimum_distal_finger_count": min(finger_counts),
                        "maximum_distal_finger_count": max(finger_counts),
                        "mean_distal_finger_count": sum(finger_counts) / len(finger_counts),
                        "region_contact_fraction": {name: sum(name in s for s in samples[start:end]) / (end - start)
                                                    for name in names}})
    longest = current = 0
    for ok in supported:
        current = 0 if ok else current + 1
        longest = max(longest, current)
    return records, longest * dt


def filter_closed_grasp(model: mujoco.MjModel, closed_data: mujoco.MjData,
                        geometry: PhysicsFilterGeometry, *,
                        configuration: PhysicsFilterSpec | None = None,
                        provenance: Mapping[str, str] | None = None) -> PhysicsFilterResult:
    """Return metadata for ONE already-closed proposal; preserve caller state.

    The caller records the model/asset/command hashes in ``provenance`` and keeps
    this record for accepted AND rejected proposals. API misuse raises ValueError
    before integration. Numerical/runtime faults produce ``filter_status=error``
    and unrun axes, never physical failure labels. No source MjData or model
    array is written. Do not call concurrently with mutation of the same source.
    """
    spec = configuration or load_physics_filter_spec()
    _validate_geometry(model, closed_data, geometry, spec)
    _guard(closed_data)
    dt = float(model.opt.timestep)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("positive finite timestep required")
    steps, window = round(spec.hold_s / dt), round(spec.contact_window_s / dt)
    if (steps < 1 or window < 1 or not math.isclose(steps * dt, spec.hold_s, abs_tol=1e-12, rel_tol=0)
            or not math.isclose(window * dt, spec.contact_window_s, abs_tol=1e-12, rel_tol=0)):
        raise ValueError("hold and contact windows must be integer timestep multiples")
    if steps > spec.maximum_steps_per_axis or 6 * steps > spec.maximum_total_steps:
        raise ValueError("requested trials exceed the finite declared step quota")
    snapshot = _integration_state(model, closed_data)
    if not np.all(np.isfinite(snapshot)):
        raise ValueError("initial integration state must be finite")
    source_qacc, source_warnings = closed_data.qacc.copy(), _warnings(closed_data)
    source_gravity = model.opt.gravity.copy()
    source_masks = (model.geom_contype.copy(), model.geom_conaffinity.copy())
    details = dict(provenance or {})
    details.update({"state_kind": "mjSTATE_INTEGRATION", "initial_state_sha256": _digest(snapshot),
                    "source_gravity_m_s2": source_gravity.tolist(), "mujoco_version": mujoco.__version__,
                    "timestep_s": dt, "steps_per_axis": steps, "requested_steps": 6 * steps,
                    "geometry": asdict(geometry), "table_collisions_disabled_on_copies":
                    spec.disable_declared_table_collisions, "task_outcome_label": None,
                    "predictor_feature_eligible": False})
    result = PhysicsFilterResult("accepted", asdict(spec), details)
    started, total_steps, hard_error = time.monotonic(), 0, False
    try:
        for axis, unit in AXES:
            gravity = np.asarray(unit, dtype=np.float64) * spec.gravity_m_s2
            record: dict[str, Any] = {"axis": axis, "status": "not_run", "steps": 0,
                                      "gravity_m_s2": gravity.tolist()}
            result.axes.append(record)
            if hard_error:
                continue
            try:
                trial_model = copy.copy(model)
                if (np.shares_memory(trial_model.opt.gravity, model.opt.gravity)
                        or np.shares_memory(trial_model.geom_contype, model.geom_contype)
                        or np.shares_memory(trial_model.geom_conaffinity, model.geom_conaffinity)):
                    raise RuntimeError("model copy shares mutable gravity storage")
                trial_model.opt.gravity[:] = source_gravity
                if spec.disable_declared_table_collisions:
                    ids = list(geometry.table_geom_ids)
                    trial_model.geom_contype[ids] = 0
                    trial_model.geom_conaffinity[ids] = 0
                trial_data = mujoco.MjData(trial_model)
                mujoco.mj_setState(trial_model, trial_data, snapshot, STATE)
                restored = _integration_state(trial_model, trial_data)
                if not np.array_equal(restored, snapshot):
                    raise RuntimeError("full integration-state restoration was not exact")
                record["initial_state_sha256"] = _digest(restored)
                record["initial_state_restored_exactly"] = True
                trial_model.opt.gravity[:] = gravity
                mujoco.mj_forward(trial_model, trial_data)
                _guard(trial_data)
                initial = _sample(trial_model, trial_data, geometry, spec.minimum_normal_force_n)
                max_displacement, max_rotation = 0.0, 0.0
                max_penetration, max_external = initial.penetration_m, initial.external_support_n
                contact_samples: list[frozenset[str]] = []
                if spec.minimum_table_clearance_m is not None:
                    from grasp_failure_prediction.evaluation.table_clearance import _lowest_world_z
                    clearance = min(_lowest_world_z(trial_model, trial_data, i)
                                    for i in geometry.hand_geom_ids) - float(geometry.table_top_z_m)
                    record["initial_table_clearance_m"] = clearance
                for _ in range(steps):
                    if (record["steps"] >= spec.maximum_steps_per_axis
                            or total_steps >= spec.maximum_total_steps
                            or time.monotonic() - started >= spec.maximum_wall_s):
                        raise RuntimeError("finite physics filter quota exceeded")
                    _guard(trial_data)
                    before_time, warnings = float(trial_data.time), _warnings(trial_data)
                    # Count an attempted step even if MuJoCo raises or resets.
                    record["steps"] += 1
                    total_steps += 1
                    mujoco.mj_step(trial_model, trial_data)
                    _guard(trial_data, warnings)
                    if not math.isclose(float(trial_data.time) - before_time, dt,
                                        abs_tol=spec.time_increment_atol_s, rel_tol=0):
                        raise RuntimeError("simulation time failed to advance by one timestep")
                    # mj_step integrates qpos after its forward pass. Refresh
                    # geometry/forces so the final integrated endpoint is sampled.
                    mujoco.mj_forward(trial_model, trial_data)
                    _guard(trial_data, warnings)
                    sample = _sample(trial_model, trial_data, geometry, spec.minimum_normal_force_n)
                    max_displacement = max(max_displacement, float(np.linalg.norm(
                        sample.relative_position - initial.relative_position)))
                    cosine = (np.trace(initial.relative_rotation.T @ sample.relative_rotation) - 1) / 2
                    max_rotation = max(max_rotation, float(np.arccos(np.clip(cosine, -1, 1))))
                    max_penetration = max(max_penetration, sample.penetration_m)
                    max_external = max(max_external, sample.external_support_n)
                    contact_samples.append(sample.supported_regions)
                region_names = (*geometry.distal_finger_geom_ids, "palm")
                windows, unsupported_s = _window_records(contact_samples, dt, window,
                                                         spec.minimum_distal_fingers, region_names)
                record.update({"status": "accepted", "maximum_relative_displacement_m": max_displacement,
                               "maximum_relative_rotation_rad_report_only": max_rotation,
                               "maximum_hand_object_penetration_m": max_penetration,
                               "maximum_external_object_support_n": max_external,
                               "contact_windows": windows, "longest_unsupported_s": unsupported_s,
                               "region_contact_fraction": {name: sum(name in s for s in contact_samples) / steps
                                                           for name in region_names}})
                checks = [(f"GRAVITY_FAIL_{axis}", max_displacement, spec.maximum_displacement_m, ">"),
                          ("PENETRATION_EXCEEDED", max_penetration, spec.maximum_penetration_m, ">"),
                          ("EXTERNAL_OBJECT_SUPPORT", max_external, spec.minimum_normal_force_n, ">"),
                          ("INSUFFICIENT_CONTACT", min(w["supported_fraction"] for w in windows),
                           spec.minimum_supported_fraction, "<"),
                          ("CONTACT_DROPOUT", unsupported_s, spec.maximum_unsupported_s, ">")]
                if spec.minimum_table_clearance_m is not None:
                    checks.append(("TABLE_COLLISION_PREDICTED", record["initial_table_clearance_m"],
                                   spec.minimum_table_clearance_m, "<"))
                for code, measured, threshold, comparison in checks:
                    if (measured < threshold - 1e-12 if comparison == "<" else measured > threshold + 1e-12):
                        result.rejection_codes.append({"code": code, "axis": axis, "measured": measured,
                                                       "threshold": threshold, "comparison": comparison,
                                                       "units": ("fraction" if code == "INSUFFICIENT_CONTACT"
                                                                 else "s" if code == "CONTACT_DROPOUT"
                                                                 else "N" if code == "EXTERNAL_OBJECT_SUPPORT"
                                                                 else "m")})
                        record["status"] = "rejected"
                        result.filter_status = "rejected"
            except Exception as exc:
                hard_error = True
                record.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
                result.rejection_codes.append({"code": "FILTER_RUNTIME_ERROR", "axis": axis,
                                               "measured": record["steps"],
                                               "threshold": spec.maximum_steps_per_axis,
                                               "comparison": "runtime_guard", "detail": str(exc)})
                result.filter_status = "error"
    finally:
        result.provenance["attempted_steps"] = total_steps
        result.provenance["elapsed_wall_s"] = time.monotonic() - started
        unchanged = (np.array_equal(snapshot, _integration_state(model, closed_data))
                     and np.array_equal(source_qacc, closed_data.qacc)
                     and np.array_equal(source_warnings, _warnings(closed_data))
                     and np.array_equal(source_gravity, model.opt.gravity)
                     and np.array_equal(source_masks[0], model.geom_contype)
                     and np.array_equal(source_masks[1], model.geom_conaffinity))
        result.provenance["source_state_and_gravity_unchanged"] = unchanged
        if not unchanged:
            raise RuntimeError("source state/model changed during isolated filter call")
    return result
