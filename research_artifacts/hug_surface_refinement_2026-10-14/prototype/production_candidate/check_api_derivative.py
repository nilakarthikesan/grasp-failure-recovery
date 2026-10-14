"""Reproducible pure-API checks; no physics, source inference or eval data."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import scipy


ROOT = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen_surface = load("frozen_surface_api", ROOT / "surface_refinement.py")
    frozen_guard = load("frozen_guard_api", ROOT / "guarded_refinement.py")
    hardened_surface = load(
        "grasp_failure_prediction.evaluation.surface_refinement",
        ROOT / "production_candidate/surface_refinement.py",
    )
    hardened_guard = load(
        "grasp_failure_prediction.evaluation.surface_refinement_guard",
        ROOT / "production_candidate/surface_refinement_guard.py",
    )
    functional = load("manufactured_proposal_tests", ROOT / "test_surface_refinement.py")
    tests = [ROOT / "test_surface_refinement.py", ROOT / "test_guarded_refinement.py"]
    source_paths = [
        ROOT / "surface_refinement.py", ROOT / "guarded_refinement.py",
        ROOT / "production_candidate/surface_refinement.py",
        ROOT / "production_candidate/surface_refinement_guard.py", *tests,
    ]
    hashes_before = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    parity = []
    cases = [
        ("default_local_side_basin", functional.proposal_case(), {}),
        ("represented_side_change", functional.proposal_case(),
         {"closest_vector_weight": 3.0, "anchor_weight": 0.0, "joint_weight": 0.0}),
        ("physical_bound", functional.proposal_case(
            physical_bounds=np.array([[-0.005, 0.04]]), native_distances=np.array([0.02]),
            native_closest_points=np.array([[0.0, 0.0, -0.02]])),
         {"closest_vector_weight": 3.0, "anchor_weight": 0.0, "joint_weight": 0.0}),
        ("local_trust_bound", functional.proposal_case(),
         {"closest_vector_weight": 3.0, "joint_delta_limit_rad": 0.002}),
        ("baseline_no_improvement", functional.proposal_case(initial_q=np.array([-0.01])), {}),
        ("native_away_normal", functional.proposal_case(), {"normal_weight": 1.0}),
    ]
    for name, args, options in cases:
        before = frozen_surface.refine_part(
            **args, evaluate=functional.plane_evaluator,
            config=frozen_surface.RefinementConfig(**options),
        )
        after = hardened_surface.refine_part(
            **args, evaluate=functional.plane_evaluator,
            config=hardened_surface.RefinementConfig(**options),
        )
        assert np.array_equal(before[0], after[0]), name
        assert before[1] == after[1], name
        parity.append({"case": name, "returned_joint_targets_exact": True,
                       "all_diagnostic_fields_exact": True})

    q0 = np.array([0.0, 0.0])
    qr = np.array([0.2, -0.1])
    limits = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    for kind in ("full", "half", "all_tip_bad", "all_objective_bad", "baseline_tip_bad"):
        def assess(q):
            alpha = q[0] / qr[0]
            objective = 1.0 if kind in ("all_objective_bad", "baseline_tip_bad") else 1.0 - alpha
            tip = 0.006
            if kind == "half":
                tip += 0.008 * alpha
            elif kind == "all_tip_bad" and alpha:
                tip = 0.011
            elif kind == "baseline_tip_bad":
                tip = 0.020
            return {"objective": objective, "mean_fingertip_error_m": tip}
        before = frozen_guard.guarded_joint_selection(q0, qr, limits, assess)
        after = hardened_guard.guarded_joint_selection(q0, qr, limits, assess)
        assert np.array_equal(before[0], after[0]), kind
        assert before[1] == after[1], kind
        parity.append({"case": "guard_" + kind, "returned_joint_targets_exact": True,
                       "all_diagnostic_fields_exact": True})

    for path in source_paths:
        ast.parse(path.read_text())
        compile(path.read_text(), str(path), "exec")
    status = pytest.main(["-q", "-p", "no:cacheprovider", *map(str, tests)])
    assert status == 0, status
    hashes_after = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    assert hashes_after == hashes_before
    report = {
        "schema": "hug_static_api_validation_derivative_v1",
        "status": "PASS", "pytest_exit_code": int(status),
        "manufactured_test_case_count": 70,
        "manufactured_parity_case_count": len(parity),
        "parity": parity,
        "scope": "Manufactured pure-API inputs only; this derivative was not rerun on all six saved grasp geometries.",
        "new_eval_data": False, "physics_steps": 0,
        "source_hashes_before_and_after_unchanged": True,
        "source_hashes": hashes_after,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "pytest": pytest.__version__},
        "changes": [
            "Explicit experimental/static-only docstrings and acceptance caveats.",
            "Reject boolean numeric settings and metrics; freeze mutable alpha input as a tuple.",
            "Validate nonempty unsigned site domains and reject negative evaluator distances.",
            "Reject out-of-bound input targets without clipping, including tiny infeasible excess.",
            "Copy source reference arrays and provide read-only copied callback joint vectors.",
            "Snapshot returned guard assessments so reused dictionaries cannot rewrite history.",
            "Remove unused asdict import; objective, default values and solver settings unchanged.",
        ],
        "limitations": [
            "A local objective improvement can leave an opposite-side geometric relationship unchanged.",
            "Fingertip guard and objective improvement are engineering checks, not geometric fidelity acceptance.",
            "No production grasp-eval execution path imports these modules automatically.",
        ],
    }
    (ROOT / "production_candidate/api_derivative_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({"status": report["status"], "tests": 70,
                      "exact_manufactured_parity_cases": len(parity)}))


if __name__ == "__main__":
    main()
