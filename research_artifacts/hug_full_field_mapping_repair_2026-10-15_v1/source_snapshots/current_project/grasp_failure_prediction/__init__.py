"""Learn, anticipate, and recover from grasp failures under physical variation.

The current package contains two early infrastructure paths for a three-part
research program spanning policy learning, future-failure prediction, and
recovery:

* ``grasp_failure_prediction.environments`` — a runnable Mac baseline built on
  Gymnasium-Robotics + MuJoCo (AdroitHandRelocate-v1) for inspection and, later,
  rollout collection.
* ``grasp_failure_prediction.integrations`` — an optional boundary that consumes
  official HUG ``grasp_pred/*.pkl`` outputs without requiring HUG's Ubuntu/CUDA
  runtime on this machine.
"""

__version__ = "0.1.0"
