"""Validate the Part I robot-training stack.

Reports the installed versions of the simulation and learning packages, checks
the compute backend, and (optionally) runs two live smoke checks:

* a short headless robosuite Panda ``PickPlaceCan`` rollout, and
* a tiny LeRobot-dataset + ACT round-trip that proves recorded frames can flow
  into an ACT-consumable batch.

Run it with::

    MUJOCO_GL=cgl python scripts/validate_stack.py

On macOS offscreen rendering uses CGL, so ``MUJOCO_GL=cgl`` is set for you if
the variable is unset.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Packages that make up the Part I stack. (import_name, distribution_name)
_STACK = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("mujoco", "mujoco"),
    ("robosuite", "robosuite"),
    ("robomimic", "robomimic"),
    ("lerobot", "lerobot"),
    ("gymnasium", "gymnasium"),
    ("gymnasium_robotics", "gymnasium-robotics"),
    ("h5py", "h5py"),
    ("datasets", "datasets"),
]


@dataclass
class StackReport:
    versions: Dict[str, str] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)
    backend: str = "cpu"
    notes: List[str] = field(default_factory=list)
    sim_rollout_ok: Optional[bool] = None
    lerobot_roundtrip_ok: Optional[bool] = None

    @property
    def ok(self) -> bool:
        return not self.failures and self.sim_rollout_ok is not False \
            and self.lerobot_roundtrip_ok is not False


def _ensure_mujoco_gl() -> None:
    """macOS renders offscreen via CGL; set a sane default if unset."""
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "cgl" if sys.platform == "darwin" else "egl"


def check_imports() -> StackReport:
    report = StackReport()
    for import_name, dist_name in _STACK:
        try:
            mod = importlib.import_module(import_name)
            try:
                version = md.version(dist_name)
            except md.PackageNotFoundError:
                version = getattr(mod, "__version__", "?")
            report.versions[dist_name] = version
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            report.failures[dist_name] = f"{type(exc).__name__}: {exc}"

    # Compute backend.
    try:
        import torch

        if torch.backends.mps.is_available():
            report.backend = "mps"
        elif torch.cuda.is_available():
            report.backend = "cuda"
        else:
            report.backend = "cpu"
    except Exception:  # noqa: BLE001
        report.backend = "unknown"

    # robomimic intentionally omits egl_probe on macOS.
    if "robomimic" in report.versions:
        report.notes.append(
            "robomimic imported; egl_probe (Linux-EGL helper) is intentionally "
            "absent on macOS."
        )
    return report


def check_sim_rollout(report: StackReport, steps: int = 5) -> None:
    """Run a short headless robosuite Panda PickPlaceCan rollout."""
    _ensure_mujoco_gl()
    try:
        import numpy as np
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        cfg = load_composite_controller_config(robot="Panda")
        env = suite.make(
            env_name="PickPlaceCan",
            robots="Panda",
            controller_configs=cfg,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["frontview", "robot0_eye_in_hand"],
            camera_heights=84,
            camera_widths=84,
            control_freq=20,
            horizon=steps + 1,
        )
        obs = env.reset()
        action = np.zeros(env.action_dim)
        for _ in range(steps):
            obs, _reward, _done, _info = env.step(action)
        assert obs["frontview_image"].shape == (84, 84, 3)
        env.close()
        report.sim_rollout_ok = True
        report.notes.append(
            f"robosuite Panda PickPlaceCan rollout ok (action_dim={env.action_dim})."
        )
    except Exception as exc:  # noqa: BLE001
        report.sim_rollout_ok = False
        report.failures["sim_rollout"] = f"{type(exc).__name__}: {exc}"


def check_lerobot_roundtrip(report: StackReport) -> None:
    """Build a tiny LeRobot dataset and confirm ACT can consume a batch."""
    import tempfile

    try:
        import numpy as np
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as tmp:
            features = {
                "observation.images.front": {
                    "dtype": "image",
                    "shape": (84, 84, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"],
                },
            }
            ds = LeRobotDataset.create(
                repo_id="grasp_failure_prediction/stack_check",
                fps=20,
                # LeRobot creates this directory itself, so point at a
                # not-yet-existing subpath of the temp dir.
                root=os.path.join(tmp, "ds"),
                features=features,
                use_videos=False,
            )
            rng = np.random.default_rng(0)
            for _ in range(6):
                frame = {
                    "observation.images.front": rng.integers(
                        0, 255, (84, 84, 3), dtype=np.uint8
                    ),
                    "observation.state": rng.standard_normal(7).astype(np.float32),
                    "action": rng.standard_normal(7).astype(np.float32),
                    # In LeRobot 0.6.x the task string travels inside the frame.
                    "task": "stack check",
                }
                ds.add_frame(frame)
            ds.save_episode()
            # LeRobot 0.6.x requires finalize() before the dataset is readable.
            if hasattr(ds, "finalize"):
                ds.finalize()

            sample = ds[0]
            assert "action" in sample
            assert "observation.state" in sample
            report.lerobot_roundtrip_ok = True
            report.notes.append(
                f"LeRobot dataset round-trip ok ({ds.num_frames} frames, "
                f"{ds.num_episodes} episode)."
            )
    except Exception as exc:  # noqa: BLE001
        report.lerobot_roundtrip_ok = False
        report.failures["lerobot_roundtrip"] = f"{type(exc).__name__}: {exc}"


def format_report(report: StackReport) -> str:
    lines = ["Part I stack validation", "=" * 40]
    for _import_name, dist_name in _STACK:
        if dist_name in report.versions:
            lines.append(f"  OK   {dist_name:22s} {report.versions[dist_name]}")
        else:
            lines.append(f"  FAIL {dist_name:22s} {report.failures.get(dist_name, '?')}")
    lines.append(f"  compute backend: {report.backend}")
    if report.sim_rollout_ok is not None:
        status = "OK" if report.sim_rollout_ok else "FAIL"
        lines.append(f"  sim rollout:        {status}")
    if report.lerobot_roundtrip_ok is not None:
        status = "OK" if report.lerobot_roundtrip_ok else "FAIL"
        lines.append(f"  lerobot round-trip: {status}")
    if report.notes:
        lines.append("-" * 40)
        for note in report.notes:
            lines.append(f"  note: {note}")
    if report.failures:
        lines.append("-" * 40)
        for name, msg in report.failures.items():
            lines.append(f"  failure[{name}]: {msg}")
    lines.append("-" * 40)
    lines.append("  RESULT: " + ("PASS" if report.ok else "FAIL"))
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate the Part I training stack.")
    parser.add_argument(
        "--no-sim", action="store_true", help="skip the live robosuite rollout"
    )
    parser.add_argument(
        "--no-lerobot", action="store_true", help="skip the LeRobot round-trip"
    )
    args = parser.parse_args(argv)

    report = check_imports()
    if not args.no_sim:
        check_sim_rollout(report)
    if not args.no_lerobot:
        check_lerobot_roundtrip(report)
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
