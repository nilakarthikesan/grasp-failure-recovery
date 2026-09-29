"""Complete simulator state capture and restore.

Part III branches counterfactual recovery attempts from recorded states, so a
snapshot must restore *identical forward dynamics*. MuJoCo's
``mjSTATE_INTEGRATION`` captures the full integration state (time, qpos, qvel,
act, plugin/warmstart), but it does **not** capture model parameters. Because
this task overrides the container's mass, inertia, and friction, those changed
``mjModel`` fields are stored alongside the integration buffer and reapplied on
restore.

Additional non-physics state that a faithful replay needs (environment step
counter, RNG, and any controller/action-queue state) is carried in
``extra`` and restored by the caller. Stock robosuite state helpers only save
time/qpos/qvel, which is insufficient here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

try:  # mujoco is only present in the robot extra.
    import mujoco
except Exception:  # pragma: no cover - import guarded for the baseline env.
    mujoco = None  # type: ignore


@dataclass
class SimSnapshot:
    """A restorable simulator state."""

    integration_state: np.ndarray
    # Changed model parameters (not part of mjSTATE_INTEGRATION).
    body_mass: Dict[int, float] = field(default_factory=dict)
    body_inertia: Dict[int, np.ndarray] = field(default_factory=dict)
    geom_friction: Dict[int, np.ndarray] = field(default_factory=dict)
    # Non-physics replay state (env counters, RNG, controller/action queue).
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Flatten to plain arrays for HDF5 storage."""
        out: Dict[str, np.ndarray] = {"integration_state": self.integration_state}
        if self.body_mass:
            ids = np.array(sorted(self.body_mass), dtype=np.int64)
            out["mass_ids"] = ids
            out["mass_vals"] = np.array([self.body_mass[i] for i in ids], dtype=np.float64)
        if self.body_inertia:
            ids = np.array(sorted(self.body_inertia), dtype=np.int64)
            out["inertia_ids"] = ids
            out["inertia_vals"] = np.stack([self.body_inertia[i] for i in ids])
        if self.geom_friction:
            ids = np.array(sorted(self.geom_friction), dtype=np.int64)
            out["friction_ids"] = ids
            out["friction_vals"] = np.stack([self.geom_friction[i] for i in ids])
        return out

    @classmethod
    def from_arrays(cls, arrays: Dict[str, np.ndarray]) -> "SimSnapshot":
        snap = cls(integration_state=np.asarray(arrays["integration_state"]))
        if "mass_ids" in arrays:
            snap.body_mass = {
                int(i): float(v)
                for i, v in zip(arrays["mass_ids"], arrays["mass_vals"])
            }
        if "inertia_ids" in arrays:
            snap.body_inertia = {
                int(i): np.asarray(v)
                for i, v in zip(arrays["inertia_ids"], arrays["inertia_vals"])
            }
        if "friction_ids" in arrays:
            snap.geom_friction = {
                int(i): np.asarray(v)
                for i, v in zip(arrays["friction_ids"], arrays["friction_vals"])
            }
        return snap


def _raw_model(model):
    """Return the raw mjModel from a robosuite model wrapper or raw handle."""
    return getattr(model, "_model", model)


def _raw_data(data):
    """Return the raw mjData from a robosuite data wrapper or raw handle."""
    return getattr(data, "_data", data)


def capture(
    sim,
    tracked_body_ids=None,
    tracked_geom_ids=None,
    extra: Optional[Dict[str, Any]] = None,
) -> SimSnapshot:
    """Capture a full snapshot from a robosuite ``sim`` (``env.sim``)."""
    if mujoco is None:  # pragma: no cover
        raise RuntimeError("mujoco is required for snapshot capture")
    m = _raw_model(sim.model)
    d = _raw_data(sim.data)
    size = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
    buf = np.zeros(size, dtype=np.float64)
    mujoco.mj_getState(m, d, buf, mujoco.mjtState.mjSTATE_INTEGRATION)

    snap = SimSnapshot(integration_state=buf, extra=dict(extra or {}))
    for bid in tracked_body_ids or []:
        snap.body_mass[int(bid)] = float(m.body_mass[bid])
        snap.body_inertia[int(bid)] = np.array(m.body_inertia[bid], dtype=np.float64)
    for gid in tracked_geom_ids or []:
        snap.geom_friction[int(gid)] = np.array(m.geom_friction[gid], dtype=np.float64)
    return snap


def restore(sim, snapshot: SimSnapshot) -> None:
    """Restore a snapshot into a robosuite ``sim`` and recompute derived state."""
    if mujoco is None:  # pragma: no cover
        raise RuntimeError("mujoco is required for snapshot restore")
    m = _raw_model(sim.model)
    d = _raw_data(sim.data)
    # Reapply changed model parameters first (they are not in the state buffer).
    for bid, mass in snapshot.body_mass.items():
        m.body_mass[bid] = mass
    for bid, inertia in snapshot.body_inertia.items():
        m.body_inertia[bid] = inertia
    for gid, fric in snapshot.geom_friction.items():
        m.geom_friction[gid] = fric
    mujoco.mj_setState(
        m, d, snapshot.integration_state, mujoco.mjtState.mjSTATE_INTEGRATION
    )
    mujoco.mj_forward(m, d)
