#!/usr/bin/env python
"""Inspect the weighted-container task end to end.

Runs one scripted demonstration, prints the phase transitions and outcome,
verifies the canonical HDF5 record round-trips, and checks that a per-step
snapshot restores the container pose exactly.

Usage:
    MUJOCO_GL=cgl python scripts/inspect_container_task.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from grasp_failure_prediction.part1.collect import run_episode
from grasp_failure_prediction.part1.config import Phase
from grasp_failure_prediction.part1.environment import WeightedContainerTask
from grasp_failure_prediction.part1.record import load_episodes, save_episodes


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "cgl")
    env = WeightedContainerTask(mass_kg=0.30, render_images=True, capture_snapshots=True)
    episode = run_episode(env, seed=1000)

    print("Weighted-container task inspection")
    print("=" * 50)
    print(f"  steps:            {episode.length}")
    print(f"  terminal success: {episode.terminal_success}")
    print(f"  terminal failure: {episode.terminal_failure}")
    print(f"  container mass:   {episode.object_mass} kg")
    print(f"  container fric:   {episode.object_friction}")

    # Phase transitions.
    transitions = []
    prev = None
    for step in episode.steps:
        if step.phase != prev:
            transitions.append((step.step_index, Phase(step.phase).label))
            prev = step.phase
    print("  phase transitions:")
    for idx, label in transitions:
        print(f"    step {idx:>3}: {label}")

    # Observation contract check.
    episode.assert_no_privileged_leak()
    print("  policy obs keys:  ", sorted(episode.steps[0].obs.keys()))
    print("  privileged keys:  ", sorted(episode.steps[0].privileged.keys()))

    # HDF5 round-trip.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "demos.hdf5")
        save_episodes(path, [episode])
        reloaded = load_episodes(path)[0]
        assert reloaded.length == episode.length
        assert np.allclose(reloaded.steps[0].action, episode.steps[0].action)
        print(f"  HDF5 round-trip:   OK ({reloaded.length} steps reloaded)")

    # Snapshot exactness (the Part III branching primitive): capture the current
    # state, perturb the sim, restore, and confirm the raw container pose is
    # reproduced bit-for-bit.
    try:
        raw_data = getattr(env._data, "_data", env._data)
        snap = env.snapshot()
        qpos_before = raw_data.qpos.copy()
        for _ in range(10):
            env.step(np.array([0.7, 0.5, 0.4, 0.0, 0.0, 0.0, -1.0]))
        perturb = float(np.max(np.abs(raw_data.qpos - qpos_before)))
        env.restore(snap)
        err = float(np.max(np.abs(raw_data.qpos - qpos_before)))
        print(
            f"  snapshot restore:  {'EXACT' if err == 0.0 else f'{err:.2e}'} "
            f"(perturbed qpos by {perturb:.4f}, restored qpos error {err:.2e})"
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
