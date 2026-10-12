"""Plot saved wrist commands and physical outcomes; no inference or new physics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def plot(runs_root, scores, output):
    if output.exists():
        raise FileExistsError("Choose a new figure path")
    report = json.loads((scores / "report.json").read_text())
    poses = json.loads((scores / "pose_manifest.json").read_text())
    case = next(case for case in poses["cases"] if case["pair_id"] == report["pair_id"])
    repo = Path(__file__).resolve().parents[1]
    splits = json.loads((repo / "research_artifacts/object_scene_pilot_v1/splits.json").read_text())
    with np.load(repo / "research_artifacts/object_scene_pilot_v1/dataset.npz", allow_pickle=False) as pilot:
        mask = np.isin(pilot["episode_ids"], splits["partitions"]["train"]["episode_ids"])
        training = np.unique(pilot["X"][mask].reshape(-1, 20, 72)[:, -1, 48:50], axis=0)
    colors = {"A": "#247ba0", "B": "#df7c26"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for posture in ("A", "B"):
        episode = runs_root / posture
        if not (episode / "trajectory.npz").exists():
            continue
        with np.load(episode / "trajectory.npz", allow_pickle=False) as trajectory:
            height = (trajectory["object_position_m"][:, 2] - trajectory["object_position_m"][0, 2]) * 100
            axes[0].plot(trajectory["time_s"], height, color=colors[posture], label=f"Wrist {posture}", lw=2)
    axes[0].axvline(2.96, color="#914343", ls="--", lw=1.5)
    axes[0].text(2.96, 16.7, "Prediction at 2.96 s", ha="center", color="#914343", fontsize=10)
    axes[0].axhline(14, color="#888888", ls=":", lw=1, label="14 cm peak-lift requirement")
    axes[0].set(xlabel="Simulation time (seconds)", ylabel="Object lift from its initial center (cm)",
                title="What the simulator actually did", ylim=(-0.5, 18.5))
    axes[0].legend(loc="lower right", fontsize=9)
    degrees = np.rad2deg(training)
    lower, upper = degrees.min(axis=0), degrees.max(axis=0)
    axes[1].add_patch(Rectangle(lower, *(upper - lower), edgecolor="#777777", facecolor="#eeeeee", lw=1))
    axes[1].scatter(degrees[:, 0], degrees[:, 1], c="#777777", s=25, label="8 original training command pairs")
    for posture in ("A", "B"):
        wrist = np.rad2deg(case["poses"][posture]["wrist_rad"])
        axes[1].scatter(*wrist, c=colors[posture], s=70, label=f"Selected wrist {posture}", zorder=3)
        axes[1].annotate(posture, wrist, xytext=(7, 0), textcoords="offset points", color=colors[posture], weight="bold")
    axes[1].set(xlabel="WRJ2 commanded angle (degrees)", ylabel="WRJ1 commanded angle (degrees)",
                title="Both wrist settings are outside training coverage")
    axes[1].legend(loc="upper right", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    labels = [row["label_success"] for row in report["records"]]
    alerts = [row["failure_alert"] for row in report["records"]]
    finding = ("Both grasps succeeded. The unchanged model warned on both." if labels == [1, 1] and alerts == [True, True]
               else "One frozen development pair: inspect the recorded labels and alerts.")
    fig.suptitle(finding, fontsize=14, weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.runs_root.resolve(), args.scores.resolve(), args.output.resolve())
