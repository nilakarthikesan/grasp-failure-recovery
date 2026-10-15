"""Frozen Part I task configuration.

These dataclasses are the single machine-readable source of the contract in
``docs/PART_I_TRAINING_SPEC.md``. Collection, training, and evaluation all read
the same :func:`default_config` so demonstrations and closed-loop rollouts stay
comparable. Changing a value here is a deliberate change to the experiment.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Tuple


class Phase(enum.IntEnum):
    """Discrete task phase logged per timestep (spec section 2)."""

    REACH = 0
    GRASP = 1
    LIFT = 2
    TRANSPORT = 3
    PLACE = 4
    DONE = 5

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class ObservationContract:
    """What the learned policy is allowed to see (spec section 4).

    ``policy_keys`` are the only inputs a policy may consume. ``privileged_keys``
    are logged for labels/analysis and must never enter the policy.
    """

    policy_keys: Tuple[str, ...] = (
        "rgb_front",
        "rgb_wrist",
        "joint_pos",
        "joint_vel",
        "eef_pose",
        "gripper_state",
    )
    privileged_keys: Tuple[str, ...] = (
        "object_pose",
        "object_mass",
        "object_friction",
        "contacts",
    )

    def assert_no_leak(self, policy_input_keys) -> None:
        """Raise if any privileged key is present in a policy input set."""
        leaked = set(self.privileged_keys) & set(policy_input_keys)
        if leaked:
            raise ValueError(
                f"privileged keys leaked into policy input: {sorted(leaked)}"
            )


@dataclass(frozen=True)
class TaskConfig:
    """Task geometry and timing (spec sections 2-3)."""

    env_name: str = "PickPlaceCan"
    robot: str = "Panda"
    # Minimum height above the table (metres) that counts as a successful lift.
    lift_height: float = 0.04
    # Placement tolerance (metres) around the target region centre.
    place_tolerance: float = 0.05
    # Control loop rate (Hz) and episode horizon in control steps.
    control_freq: int = 20
    horizon: int = 200
    # Camera streams the policy receives.
    camera_names: Tuple[str, ...] = ("frontview", "robot0_eye_in_hand")
    camera_height: int = 84
    camera_width: int = 84
    # Excess-force flag threshold (N), logged separately from success/failure.
    excess_force_newtons: float = 60.0


@dataclass(frozen=True)
class PhysicsConfig:
    """Single nominal training condition (spec section 6).

    Mass and friction are *fixed* during Part I training. Their variation is
    reserved for the frozen-policy Part II evaluation, so they must not be
    randomised here.
    """

    nominal_mass_kg: float = 0.18
    nominal_sliding_friction: float = 1.0
    nominal_torsional_friction: float = 0.005
    nominal_rolling_friction: float = 0.0001
    # Fixed container half-extent (metres). Pinning the size makes every episode
    # use an identical-looking container (Lift randomizes size by default) and
    # makes model parameters deterministic for cross-instance snapshot restore.
    nominal_half_size: float = 0.021
    # Only the object's starting pose may be randomised in Part I.
    randomize_object_xy: bool = True
    randomize_object_yaw: bool = True


@dataclass(frozen=True)
class SplitConfig:
    """Episode-level split by starting condition (spec section 8)."""

    n_train_episodes: int = 40
    n_val_episodes: int = 10
    train_seed_base: int = 1000
    val_seed_base: int = 9000

    def train_seeds(self):
        return [self.train_seed_base + i for i in range(self.n_train_episodes)]

    def val_seeds(self):
        # Disjoint from training seeds: held-out starting conditions.
        return [self.val_seed_base + i for i in range(self.n_val_episodes)]


@dataclass(frozen=True)
class Part1Config:
    """Top-level Part I configuration bundle."""

    task: TaskConfig = field(default_factory=TaskConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    observation: ObservationContract = field(default_factory=ObservationContract)

    def with_overrides(self, **task_overrides) -> "Part1Config":
        """Return a copy with a few task fields overridden (e.g. shorter horizon)."""
        return replace(self, task=replace(self.task, **task_overrides))


def default_config() -> Part1Config:
    """The frozen Part I configuration used across the pipeline."""
    return Part1Config()
