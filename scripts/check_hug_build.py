"""Run focused HUG checks in the pinned local runtime with integration blocked."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy
import mujoco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-protocol", type=Path,
        default=Path("research_artifacts/hug_fresh_mapping_qualification_2026-10-15_v1/protocol.json"))
    parser.add_argument("--report", type=Path)
    args, pytest_args = parser.parse_known_args()
    if args.report and args.report.exists():
        raise FileExistsError("never overwrite a verification receipt")
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(args.dependency_protocol.read_text())
    sys.path.insert(0, str(root / "src"))
    sys.path.extend(protocol["dependency_dirs"])
    import pytest
    import pydantic
    calls = []
    def blocked(*a, **kw):
        calls.append(True)
        raise RuntimeError("focused static checks prohibit physics integration")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, blocked)
    started = time.perf_counter()
    code = pytest.main(pytest_args + ["-p", "no:cacheprovider"])
    receipt = dict(exit_code=int(code), integration_calls=len(calls),
                   pytest_arguments=pytest_args, elapsed_s=time.perf_counter()-started,
                   runtime=dict(python=platform.python_version(), numpy=numpy.__version__,
                                mujoco=mujoco.__version__, pydantic=pydantic.__version__))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt))
    raise SystemExit(int(code) or int(bool(calls)))


if __name__ == "__main__":
    main()
