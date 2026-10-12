"""Scientific figure of saved paired warnings and a task-label boundary.

Reads recorded outcomes and predictions only; no new simulation or inference.
"""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot(scores, first_bundle, output):
    if output.exists():
        raise FileExistsError("Choose a new figure output path")
    report = json.loads((scores / "report.json").read_text())
    first = json.loads((first_bundle / "report.json").read_text())
    pairs = [first, *report["paired_reports"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.8),
                             gridspec_kw={"width_ratios": [1.8, 1]}, constrained_layout=True)
    matrix = axes[0]
    matrix.set(xlim=(-0.02, 2.02), ylim=(8, -0.5))
    matrix.set_xticks([0.5, 1.5], ["Wrist A", "Wrist B"])
    matrix.xaxis.tick_top()
    labels = []
    for index, pair in enumerate(pairs):
        scene, seed = re.search(r"scene(\d+)_seed(\d+)", pair["pair_id"]).groups()
        labels.append(f"Scene {int(scene)}, grasp {seed}" + (" *" if index == 0 else ""))
        lookup = {row["posture"]: row for row in pair["records"]}
        for column, posture in enumerate(("A", "B")):
            row = lookup.get(posture)
            if row is None:
                text, color = "STOPPED\nUnscored", "#eeeeee"
            else:
                success, warning = row["label_success"], row["failure_alert"]
                outcome = "SUCCESS" if success else "FAILS LIFT RULE"
                prediction = ("WARNING: false alarm" if success and warning else
                              "WARNING: correct" if warning else
                              "CLEAR: correct" if success else "CLEAR: missed failure")
                text = outcome + "\n" + prediction
                color = "#e2f0e9" if success else "#ffe9b8"
            matrix.add_patch(Rectangle((column + 0.02, index - 0.43), 0.96, 0.86,
                                       facecolor=color, edgecolor="white", lw=2))
            matrix.text(column + 0.5, index, text, ha="center", va="center", fontsize=9.2)
    matrix.set_yticks(range(8), labels)
    matrix.set_title("Warning versus actual task outcome", pad=34)
    matrix.tick_params(length=0)
    for spine in matrix.spines.values():
        spine.set_visible(False)
    matrix.text(0, 7.85, "* Earlier pair retained as a reference; seven other pairs are new.",
                fontsize=9, ha="left")
    boundary = next(pair for pair in report["paired_reports"]
                    if pair["pair_id"] == "mug_scene003_seed2_noise0.00")
    values = [row["measured_task_outcome"]["outcome"]["maximum_lift_m"] * 100
              for row in boundary["records"]]
    ax = axes[1]
    ax.bar([0, 1], values, color=["#d99c2b", "#68a58a"], width=0.52)
    ax.axhline(14, color="#555555", ls="--", lw=1.3)
    for index, value in enumerate(values):
        ax.text(index, value + 0.008, f"{value:.6f} cm", ha="center", fontsize=10)
    ax.text(-0.42, 14.0015, "14 cm rule", fontsize=9, color="#555555")
    ax.set(xticks=[0, 1], xticklabels=["A: label 0", "B: label 1"],
           ylabel="Peak lift (cm), zoomed around the cutoff", ylim=(13.94, 14.06),
           title="Scene 3, grasp 2: a small label boundary")
    ax.text(0.5, 13.945, "A misses the rule by 0.036 mm.\nBoth satisfy the 2 s HOLD measure.",
            ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The unchanged model warned on every attempt", fontsize=15, weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--first-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.scores.resolve(), args.first_bundle.resolve(), args.output.resolve())
