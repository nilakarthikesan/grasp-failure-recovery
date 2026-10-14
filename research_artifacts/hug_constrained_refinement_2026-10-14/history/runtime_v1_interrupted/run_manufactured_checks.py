"""Pure mathematical regression runner; never inference or physics."""
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import scipy


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestCount:
    count = 0

    def pytest_sessionfinish(self, session, exitstatus):
        self.count = session.testscollected


def main():
    protocol_path = ROOT / "manufactured_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    for file, expected in protocol["source_sha256"].items():
        assert sha(ROOT / file) == expected
        if file.endswith(".py"):
            source = (ROOT / file).read_text()
            ast.parse(source)
            compile(source, str(ROOT / file), "exec")
    name = "grasp_failure_prediction.evaluation.constrained_surface_refinement"
    spec = importlib.util.spec_from_file_location(name, ROOT / "constrained_refinement.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert asdict(module.ConstrainedRefinementConfig()) == protocol["fixed_settings"]
    assert module.DIRECTION_SCORE_ENDPOINT_ALLOWANCE == protocol["direction_cosine_endpoint_allowance"]
    counter = TestCount()
    status = pytest.main(["-q", "-p", "no:cacheprovider",
                          str(ROOT / "test_constrained_surface_refinement.py")],
                         plugins=[counter])
    assert status == 0
    for file, expected in protocol["source_sha256"].items():
        assert sha(ROOT / file) == expected
    report = {
        "schema": "hug_joint_constrained_manufactured_verification_v1",
        "status": "PASS",
        "protocol_sha256": sha(protocol_path),
        "source_sha256": protocol["source_sha256"],
        "test_case_count": counter.count,
        "pytest_exit_code": int(status),
        "source_hashes_unchanged_before_after": True,
        "source_parse_and_in_memory_compile": "PASS",
        "new_inference": False, "new_eval_data": False, "physics_steps": 0,
        "scope": "Manufactured mathematical callbacks only; no saved grasp geometry or mapping sign-off.",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "pytest": pytest.__version__},
    }
    (ROOT / "manufactured_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "test_cases": counter.count,
                      "physics_steps": 0, "new_eval_data": False}))


if __name__ == "__main__":
    main()
