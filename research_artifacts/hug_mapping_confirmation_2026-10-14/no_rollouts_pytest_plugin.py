"""Test-run guard: the user deferred all simulation rollouts."""
import pytest


@pytest.fixture(autouse=True)
def prohibit_physics_steps(monkeypatch):
    import mujoco

    def prohibited(*args, **kwargs):
        pytest.fail("Simulation stepping is forbidden during this implementation")

    monkeypatch.setattr(mujoco, "mj_step", prohibited)
    monkeypatch.setattr(mujoco, "mj_step1", prohibited)
    monkeypatch.setattr(mujoco, "mj_step2", prohibited)
