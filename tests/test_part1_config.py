"""Tests for the frozen Part I configuration contract.

These tests exercise pure configuration logic and do not import the heavy
simulation/learning stack, so they run in the lightweight baseline environment.
"""

from grasp_failure_prediction.part1.config import (
    ObservationContract,
    Part1Config,
    Phase,
    default_config,
)


def test_phases_are_ordered_and_labelled():
    order = [Phase.REACH, Phase.GRASP, Phase.LIFT, Phase.TRANSPORT, Phase.PLACE, Phase.DONE]
    assert [int(p) for p in order] == [0, 1, 2, 3, 4, 5]
    assert Phase.TRANSPORT.label == "transport"


def test_observation_contract_separates_privileged_metadata():
    contract = ObservationContract()
    # No overlap between what the policy sees and privileged labels.
    assert not (set(contract.policy_keys) & set(contract.privileged_keys))
    # Mass/friction/contacts are privileged, never policy inputs.
    for key in ("object_mass", "object_friction", "contacts", "object_pose"):
        assert key in contract.privileged_keys
        assert key not in contract.policy_keys


def test_observation_contract_detects_leak():
    contract = ObservationContract()
    # A clean policy input set passes.
    contract.assert_no_leak(["rgb_front", "joint_pos"])
    # Leaking a privileged key raises.
    try:
        contract.assert_no_leak(["rgb_front", "object_mass"])
    except ValueError as exc:
        assert "object_mass" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a leak to be detected")


def test_train_and_val_seeds_are_disjoint():
    cfg = default_config()
    train = set(cfg.split.train_seeds())
    val = set(cfg.split.val_seeds())
    assert train and val
    assert not (train & val), "held-out validation seeds must be disjoint"
    assert len(train) == cfg.split.n_train_episodes
    assert len(val) == cfg.split.n_val_episodes


def test_physics_is_single_nominal_condition():
    cfg = default_config()
    # Part I trains under one nominal mass/friction; only pose may randomise.
    assert cfg.physics.nominal_mass_kg > 0
    assert cfg.physics.nominal_sliding_friction > 0
    assert cfg.physics.randomize_object_xy is True


def test_with_overrides_returns_modified_copy():
    cfg = default_config()
    short = cfg.with_overrides(horizon=10)
    assert short.task.horizon == 10
    assert cfg.task.horizon != 10  # original unchanged (frozen dataclass copy)
    assert isinstance(short, Part1Config)
