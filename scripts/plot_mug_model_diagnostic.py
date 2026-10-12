"""Plot the saved diagnostic; does not load weights or compute predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    root = args.diagnostic
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report = json.loads((root / "report.json").read_text())
    rows = list(csv.DictReader((root / "predictions.csv").open()))
    excluded = json.loads((root / "excluded.json").read_text())
    groups = report["mug_by_noise"]
    false_alarms = [g["metrics"]["failure_confusion"]["false_positive"] for g in groups]
    successful = [g["metrics"]["successes"] for g in groups]
    rates = [100 * a / b for a, b in zip(false_alarms, successful, strict=True)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.suptitle("The frozen model flagged every completed mug grasp", fontsize=16, x=0.07, ha="left")
    axes[0].bar(range(3), rates, color="#b24b45", width=0.62)
    for i, (a, b) in enumerate(zip(false_alarms, successful, strict=True)):
        axes[0].text(i, 103, f"{a}/{b}\nfalse alarms", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylim(0, 122)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel("False alarms among completed successful grasps (%)")
    axes[0].set_xticks(range(3), ["0.00", "0.15", "0.60"])
    axes[0].set_xlabel("Requested finger target noise bound (rad)")
    axes[0].set_title("False alarms repeat at all three bounds", fontsize=11, loc="left", pad=12)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="y", alpha=0.15)
    axes[0].set_axisbelow(True)

    matrix = np.zeros((8, 3), dtype=int)
    cell_labels = [["" for _ in range(3)] for _ in range(8)]
    mug_rows = [r for r in rows if r["role"] == "mug"]
    for row in mug_rows + excluded:
        episode = row["episode_id"]
        if not episode.startswith("mug_"):
            continue
        scene = int(episode.split("_scene")[1].split("_")[0])
        seed = int(episode.split("_seed")[1].split("_")[0])
        i = (scene - 2) * 4 + seed
        j = [0.0, 0.15, 0.6].index(float(row["noise_amplitude_rad"]))
        if row in excluded:
            matrix[i, j], cell_labels[i][j] = 2, "Stopped\nUnscored"
        else:
            if row["failure_alert"] != "True":
                raise ValueError("figure expects the verified all-alert mug result")
            if row["label_success"] == "1":
                matrix[i, j], cell_labels[i][j] = 0, "Success\nFalse alarm"
            else:
                matrix[i, j], cell_labels[i][j] = 1, "Failure\nFlagged"
    cmap = ListedColormap(["#f1d8d4", "#dce4eb", "#efefef"])
    axes[1].imshow(matrix, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    for i in range(8):
        for j in range(3):
            if not cell_labels[i][j]:
                raise ValueError("missing planned mug check")
            axes[1].text(j, i, cell_labels[i][j], ha="center", va="center", fontsize=9)
    axes[1].set_xticks(range(3), ["0.00", "0.15", "0.60"])
    axes[1].set_yticks(range(8), [f"Scene {scene}, seed {seed}" for scene in (2, 3) for seed in range(4)])
    axes[1].set_xlabel("Requested finger target noise bound (rad)")
    axes[1].set_ylabel("Saved mug proposal")
    axes[1].set_title("Same proposals, paired noise conditions", fontsize=11, loc="left", pad=12)
    axes[1].set_xticks(np.arange(-0.5, 3, 1), minor=True)
    axes[1].set_yticks(np.arange(-0.5, 8, 1), minor=True)
    axes[1].grid(which="minor", color="white", linewidth=2)
    axes[1].tick_params(which="minor", bottom=False, left=False)
    axes[1].spines[:].set_visible(False)
    fig.text(0.07, 0.055, "Source: saved v3 development checks, scored 12 October 2026. Original model, 2.96 s cutoff and 0.5 alert threshold.", fontsize=9)
    fig.text(0.07, 0.025, "20 mug successes, 3 acquisition failures, 1 stopped attempt. Bounds are requested offsets; joint limits clip applied commands.", fontsize=9)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.07, right=0.97, wspace=0.50)
    fig.savefig(root / "noise_false_alarms.png", dpi=170)
    fig.savefig(root / "noise_false_alarms.svg")
    plt.close(fig)
    manifest["files_sha256"] = {
        str(path.relative_to(root)): "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(root / "noise_false_alarms.png")


if __name__ == "__main__":
    main()
