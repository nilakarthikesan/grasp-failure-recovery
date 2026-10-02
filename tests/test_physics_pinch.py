"""Positive and negative controls for simulated frictional pickup."""
import importlib.util
from pathlib import Path


def test_force_limited_opposing_pads_lift_but_open_pads_do_not():
    path = Path(__file__).resolve().parents[1] / "scripts/check_physics_pinch.py"
    spec = importlib.util.spec_from_file_location("physics_pinch_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    open_result, _, _ = module.trial(False)
    closed_result, _, _ = module.trial(True)
    assert not open_result["success"]
    assert open_result["maximum_lift_m"] < .001
    assert closed_result["success"]
    assert closed_result["minimum_hold_lift_m"] >= .14
