"""Part I: learn the nominal grasp-and-transport policy.

This package implements the Part I contract documented in
``docs/PART_I_TRAINING_SPEC.md``: a simulated Panda arm with a two-finger
gripper (robosuite/MuJoCo) that grasps a rigid container, lifts it, transports
it a short distance, and places it in a target region.

Design rules that the whole package must respect:

* The learned policy only ever sees the sensor observations declared in
  :class:`~grasp_failure_prediction.part1.config.ObservationContract`.
* Object mass, friction, object pose, and simulator contact truth are
  *privileged metadata* used for labels and analysis only. They are never fed
  to the policy.
* Every logged episode must be restorable, so Parts II and III can branch
  matched interventions from recorded states.
"""

from grasp_failure_prediction.part1.config import (
    ObservationContract,
    PhysicsConfig,
    Part1Config,
    Phase,
    SplitConfig,
    TaskConfig,
    default_config,
)

__all__ = [
    "ObservationContract",
    "PhysicsConfig",
    "Part1Config",
    "Phase",
    "SplitConfig",
    "TaskConfig",
    "default_config",
]
