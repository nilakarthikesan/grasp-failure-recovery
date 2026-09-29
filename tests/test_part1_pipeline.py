"""Pure-logic tests for the export/eval pipeline (no heavy stack imported).

The evaluation metrics (``evaluate.EvalResult``) are held in the design-review
PR because *how results are calculated and communicated* is still open for
discussion. Those tests skip automatically when the module is not present.
"""

import numpy as np
import pytest

from grasp_failure_prediction.part1.lerobot_export import STATE_DIM, state_vector


def _eval_result_cls():
    module = pytest.importorskip(
        "grasp_failure_prediction.part1.evaluate",
        reason="evaluation metrics live in the design-review PR",
    )
    return module.EvalResult


def test_state_vector_layout():
    obs = {
        "joint_pos": np.arange(7, dtype=np.float32),
        "joint_vel": np.arange(7, dtype=np.float32) + 10,
        "eef_pose": np.arange(7, dtype=np.float32) + 20,
        "gripper_state": np.array([100.0, 200.0], dtype=np.float32),
    }
    vec = state_vector(obs)
    assert vec.shape == (STATE_DIM,)
    assert vec.dtype == np.float32
    # Order is joint_pos, joint_vel, eef_pose, gripper_state.
    assert vec[0] == 0 and vec[7] == 10 and vec[14] == 20
    assert vec[-1] == 200.0


def test_eval_result_summary_rates():
    EvalResult = _eval_result_cls()
    res = EvalResult(
        n_episodes=4,
        successes=2,
        object_losses=1,
        timeouts=1,
        collisions=0,
        excess_force=1,
        max_phase=[5, 4, 1, 0],  # reached place, place, grasp, reach
    )
    s = res.summary()
    assert s["success_rate"] == 0.5
    assert s["object_loss_rate"] == 0.25
    assert s["timeout_rate"] == 0.25
    assert s["excess_force_rate"] == 0.25
    # Phase-reach uses >= thresholds; DONE(5) also counts as >= place here since
    # eval feeds only non-DONE phases, but this unit test checks the arithmetic.
    assert s["reached_grasp_rate"] == 0.75  # phases 5,4,1 >= GRASP(1)
    assert s["reached_place_rate"] == 0.5   # phases 5,4 >= PLACE(4)


def test_eval_result_empty_is_safe():
    EvalResult = _eval_result_cls()
    res = EvalResult()
    s = res.summary()
    assert s["episodes"] == 0
    assert s["success_rate"] == 0.0
