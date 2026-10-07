#!/usr/bin/env python
"""Verify the container runtime and execute one saved-grasp MuJoCo case."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import numpy as np


EXPECTED_CHECKPOINT_SHA256 = (
    "515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")


def find_mano_right(root: Path) -> Path:
    candidates = (root / "models" / "MANO_RIGHT.pkl", root / "MANO_RIGHT.pkl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"MANO right-hand model is missing: expected {expected}")


def verify_runtime(require_hug_assets: bool) -> dict:
    modules = {
        name: importlib.import_module(name)
        for name in ("cv2", "dex_retargeting", "mujoco", "numpy", "torch")
    }
    mujoco_version = modules["mujoco"].__version__
    if not mujoco_version.startswith("3.3."):
        raise RuntimeError(f"expected MuJoCo 3.3.x, got {mujoco_version}")

    hug_root = Path(os.environ.get("HUG_ROOT", "/opt/hug"))
    dex_root = Path(os.environ.get("DEX_URDF_ROOT", "/opt/dex-urdf/robots/hands"))
    require_file(hug_root / "src" / "inference.py", "pinned HUG source")
    require_file(
        dex_root / "shadow_hand" / "shadow_hand_right.urdf",
        "pinned Shadow Hand URDF",
    )

    report = {
        "runtime_passed": True,
        "python_version": sys.version.split()[0],
        "mujoco_version": mujoco_version,
        "torch_version": modules["torch"].__version__,
        "cuda_available": modules["torch"].cuda.is_available(),
        "hug_commit": os.environ.get("HUG_COMMIT", "unknown"),
        "dex_urdf_commit": os.environ.get("DEX_URDF_COMMIT", "unknown"),
        "code_commit": os.environ.get("CODE_COMMIT", "unknown"),
        "hug_assets_required": require_hug_assets,
    }

    if require_hug_assets:
        checkpoint = Path(
            os.environ.get("HUG_CHECKPOINT", "/models/hug_full.safetensors")
        )
        mano_root = Path(
            os.environ.get("MANO_MODEL_ROOT", "/opt/hug/assets/mano_models")
        )
        require_file(checkpoint, "HUG checkpoint")
        mano_right = find_mano_right(mano_root)
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(
                "HUG checkpoint hash mismatch: "
                f"expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_hash}"
            )
        importlib.import_module("hug.dataloader.grasp_dataset")
        importlib.import_module("hug.inference")
        importlib.import_module("hug.models.mano")
        report["hug_checkpoint_sha256"] = checkpoint_hash
        report["mano_right_path"] = str(mano_right)
        report["mano_right_sha256"] = sha256(mano_right)
        report["hug_imports_passed"] = True

    return report


def write_synthetic_hug_fixture(root: Path) -> None:
    """Write a deterministic trusted fixture for the runner plumbing check."""

    landmarks = np.zeros((21, 3), dtype=np.float64)
    bases = {
        1: (-0.030, -0.015, 0.010),
        5: (-0.020, 0.015, 0.000),
        9: (0.000, 0.020, 0.000),
        13: (0.020, 0.015, 0.000),
        17: (0.038, 0.005, 0.000),
    }
    lengths = {1: 0.025, 5: 0.030, 9: 0.034, 13: 0.031, 17: 0.026}
    for base_index, base in bases.items():
        landmarks[base_index] = base
        direction = (
            np.array([-0.7, 0.7, 0.1])
            if base_index == 1
            else np.array([0.0, 1.0, 0.0])
        )
        for offset in range(1, 4):
            landmarks[base_index + offset] = (
                landmarks[base_index]
                + direction * lengths[base_index] * offset
            )

    prediction = {
        "pose": np.zeros((15, 3)),
        "shape": np.zeros(10),
        "t": np.zeros(3),
        "T_camera_wrist": np.eye(4),
        "landmarks_3d": landmarks,
        "mesh_vertices": np.zeros((778, 3)),
    }
    path = root / "grasps" / "object01" / "grasp_003.pkl"
    path.parent.mkdir(parents=True)
    path.write_bytes(pickle.dumps(prediction))


def run_saved_grasp_case(project_root: Path) -> dict:
    with (
        tempfile.TemporaryDirectory(prefix="container-input-") as input_root,
        tempfile.TemporaryDirectory(prefix="container-eval-") as output,
    ):
        portable_input_root = Path(input_root)
        write_synthetic_hug_fixture(portable_input_root)
        command = [
            sys.executable,
            "-m",
            "grasp_failure_prediction.evaluation.case_runner",
            str(project_root / "eval_cases" / "hug_case_001" / "case.yaml"),
            "--project-root",
            str(portable_input_root),
            "--output",
            output,
        ]
        completed = subprocess.run(
            command,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        output_root = Path(output)
        required = (
            "resolved_case.json",
            "trajectory.npz",
            "final_state.npz",
            "result.json",
        )
        for name in required:
            require_file(output_root / name, f"evaluation artifact {name}")
        result = json.loads((output_root / "result.json").read_text())
        if result.get("status") != "completed":
            raise RuntimeError(f"evaluation did not complete: {result}")
        return {
            "saved_grasp_evaluation_passed": True,
            "case_id": result["case_id"],
            "success": result["success"],
            "failure_type": result["failure_type"],
            "artifacts": list(required),
            "runner_stdout": completed.stdout,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-hug-assets",
        action="store_true",
        help="also require and hash the mounted HUG checkpoint and MANO model",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    report = verify_runtime(args.require_hug_assets)
    if not args.skip_evaluation:
        report.update(run_saved_grasp_case(project_root))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
