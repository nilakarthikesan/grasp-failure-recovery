"""Inspect AdroitHandRelocate-v1 without training or acting.

This module starts one new episode with ``reset()``, reads the simulator's
initial state, and closes it. It deliberately never calls ``env.step(...)``:
the hand does not move and no policy or neural network is involved.

The logic was migrated verbatim from the original playground diagnostic script
into the standalone package so it can be reused and imported. The thin entry
point lives in ``scripts/inspect_adroit.py``.

Run from the repository root:

    python scripts/inspect_adroit.py

or, once the package is installed:

    inspect-adroit
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np


ENVIRONMENT_ID = "AdroitHandRelocate-v1"
RESET_SEED = 0


def heading(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    """Return a readable MuJoCo name, including a fallback for unnamed objects."""
    return mujoco.mj_id2name(model, object_type, index) or f"<unnamed:{index}>"


def geom_label(model: mujoco.MjModel, geom_id: int) -> str:
    """Describe a collision geometry and the body that owns it."""
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    body_id = int(model.geom_bodyid[geom_id])
    body_name = object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return f"{geom_name or '<unnamed geom>'} on body {body_name!r}"


def print_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    print(f"contacts present immediately after reset: {data.ncon}")
    if data.ncon == 0:
        print("  none")
        return

    for index in range(data.ncon):
        contact = data.contact[index]
        first = geom_label(model, contact.geom1)
        second = geom_label(model, contact.geom2)
        print(f"  {index}: {first} <-> {second} (distance={contact.dist:.6f} m)")


def inspect_adroit() -> None:
    """Reset one Adroit episode and print its interface and physical properties."""
    # Importing a package does not automatically register all third-party Gymnasium
    # environments. This explicit call makes the Adroit environment IDs available.
    gym.register_envs(gymnasium_robotics)

    heading("1. Start and reset the environment")
    env = gym.make(ENVIRONMENT_ID)
    observation, info = env.reset(seed=RESET_SEED)
    adroit = env.unwrapped

    print(f"environment: {ENVIRONMENT_ID}")
    print(f"reset seed: {RESET_SEED}")
    print(f"reset returned observation shape: {observation.shape}")
    print(f"reset returned info: {info}")
    print("No action was sent: env.step(...) is never called.")

    heading("2. Observations: information available to a future model")
    print(f"observation space: {env.observation_space}")
    print("  values  0:30  arm, wrist, and finger joint positions")
    print("  values 30:33  palm position minus object position")
    print("  values 33:36  palm position minus target position")
    print("  values 36:39  object position minus target position")
    print(f"initial observation:\n{np.array2string(observation, precision=4)}")
    print(
        "Important: the default 39-value observation contains relative positions, "
        "but no tactile or contact measurements."
    )

    heading("3. Actions: controls available to a future policy")
    print(f"action space: {env.action_space}")
    print(
        "Each action has 30 normalized values in [-1, 1]: 6 move the arm, "
        "2 move the wrist, and 22 move finger joints."
    )
    print("actuator names:")
    for index in range(adroit.model.nu):
        name = object_name(adroit.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        low, high = adroit.model.actuator_ctrlrange[index]
        print(f"  {index:>2}: {name:<8} physical control range [{low: .3f}, {high: .3f}]")

    heading("4. Timing and physical properties")
    physics_timestep = float(adroit.model.opt.timestep)
    control_timestep = float(adroit.dt)
    max_steps = int(env.spec.max_episode_steps)
    print(f"MuJoCo physics timestep: {physics_timestep:.4f} s")
    print(f"physics steps per action: {adroit.frame_skip}")
    print(f"control timestep: {control_timestep:.4f} s")
    print(f"maximum actions per episode: {max_steps}")
    print(f"maximum simulated episode duration: {max_steps * control_timestep:.2f} s")

    object_body_id = adroit.obj_body_id
    object_position = adroit.data.xpos[object_body_id].copy()
    object_mass = float(adroit.model.body_mass[object_body_id])
    object_geom_ids = np.flatnonzero(adroit.model.geom_bodyid == object_body_id)

    print(f"\nobject body: {adroit.model.body(object_body_id).name}")
    print(f"object position after reset (x, y, z): {object_position}")
    print(f"object mass: {object_mass:.6f} kg")
    for geom_id in object_geom_ids:
        geom_name = object_name(adroit.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        friction = adroit.model.geom_friction[geom_id]
        print(
            f"object geom {geom_name!r} friction "
            f"(sliding, torsional, rolling): {friction}"
        )

    print()
    print_contacts(adroit.model, adroit.data)
    print(
        "\nMuJoCo knows about these contacts internally. They are not included in "
        "the environment's default observation, so adding contact-aware inputs "
        "will require an explicit observation wrapper or custom environment."
    )

    heading("5. Demonstrations")
    print(
        "No demonstration was downloaded or replayed during this inspection. "
        "That is the next separate step after we review this environment interface."
    )

    env.close()


def main() -> None:
    inspect_adroit()


if __name__ == "__main__":
    main()
