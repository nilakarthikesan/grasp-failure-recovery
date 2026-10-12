"""Record evaluation states and render them to H.264 in a clean subprocess."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np

from .runner import AdroitShadowRunner, RunnerStep


class EvaluationVideoRecorder:
    """Collect control-step states and render them after physics completes."""

    def __init__(self, runner: AdroitShadowRunner, path: str | Path) -> None:
        self.runner = runner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._qpos: list[np.ndarray] = []
        self._states: list[str] = []
        self._times: list[float] = []

    def capture(self, step: RunnerStep) -> None:
        self._qpos.append(self.runner.data.qpos.copy())
        self._states.append(step.state.value)
        self._times.append(step.time_s)

    def close(self) -> None:
        if not self._qpos:
            raise RuntimeError("evaluation video contains no states")
        scene_path = self.path.parent / ".video_scene.xml"
        states_path = self.path.parent / ".video_states.npz"
        mujoco.mj_saveLastXML(str(scene_path), self.runner.model)
        np.savez_compressed(
            states_path,
            qpos=np.stack(self._qpos),
            state=np.asarray(self._states),
            time_s=np.asarray(self._times),
        )
        executable = sys.executable
        if sys.platform == "darwin":
            executable = shutil.which("mjpython") or str(
                Path(sys.executable).with_name("mjpython")
            )
        command = [
            executable,
            str(Path(__file__).with_name("render_rollout.py")),
            "--scene",
            str(scene_path),
            "--states",
            str(states_path),
            "--output",
            str(self.path),
            "--fps",
            str(1.0 / self.runner.control_timestep_s),
        ]
        completed = subprocess.run(command)
        if completed.returncode == 0:
            scene_path.unlink(missing_ok=True)
            states_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                f"video renderer exited with status {completed.returncode}; "
                f"debug inputs retained at {scene_path} and {states_path}"
            )
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError(f"evaluation video contains no frames: {self.path}")
