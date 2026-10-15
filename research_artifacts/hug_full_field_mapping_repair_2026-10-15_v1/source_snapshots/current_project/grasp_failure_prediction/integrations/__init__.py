"""Optional integration boundary for official HUG prediction outputs.

Nothing in this subpackage imports the upstream HUG package. HUG's full
inference runs in its own Ubuntu/CUDA/Python 3.10 environment; here we only
consume the ``grasp_pred/*.pkl`` dictionaries it writes.
"""

from grasp_failure_prediction.integrations.hug import (
    HugGraspPrediction,
    HugPredictionError,
    load_hug_prediction,
    normalize_hug_prediction,
)

__all__ = [
    "HugGraspPrediction",
    "HugPredictionError",
    "load_hug_prediction",
    "normalize_hug_prediction",
]
