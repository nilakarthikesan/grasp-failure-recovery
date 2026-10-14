"""Scientific diagnostics from completed independent reports only."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

B = Path(__file__).parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    report = json.loads((B / "summary.json").read_text())
    assert report["case_count"] == 6 and report["resolution_count"] == 9
    assert report["all_engineering_source_preflights_pass"]
    assert report["all_protected_hashes_unchanged"]
    cases = sorted(report["cases"], key=lambda c: c["source_inventory_index"])
    assert [c["source_inventory_index"] for c in cases] == [8, 12, 16, 20, 24, 28]
    labels = ["Apple A", "Apple B", "Milk A", "Milk B", "Mug A", "Mug B"]
    series = {k: [] for k in ("iou0", "iou1", "lost0", "lost1", "added0", "added1", "tip0", "tip1", "max0", "max1", "flip0", "flip1", "eligible0", "eligible1")}
    for c in cases:
        band = c["resolutions"]["2"]["bands"]["2"]
        for suffix, prefix in (("0", "original"), ("1", "constrained")):
            series["iou" + suffix].append(band[prefix + "_whole_hand_unsigned_proximity"]["centroid_estimates"]["iou"])
            series["lost" + suffix].append(band["six_part_area_sums"][prefix + "_lost_lower_mm2"])
            series["added" + suffix].append(band["six_part_area_sums"][prefix + "_added_lower_mm2"])
            landmarks = c[prefix + "_landmarks"]
            series["tip" + suffix].append(landmarks["mean_fingertip_error_m"] * 1000)
            series["max" + suffix].append(landmarks["maximum_fingertip_error_m"] * 1000)
            domain = band[prefix + "_orientation_domain"]
            series["flip" + suffix].append(domain["sampled_sign_reversal_count"])
            series["eligible" + suffix].append(domain["outside_visual_eligible_denominator"]["query_count"])
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "figure.dpi": 180})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))
    ax = axes.ravel()
    x = np.arange(6)
    width = .35
    colors = ("#2b6cb0", "#c65d1d")
    assert report["selection_source_counts"]["original_fallback"] == 6
    names = ("Original mapping", "Returned original pose")
    for z in ax:
        z.set_xticks(x, labels, rotation=35, ha="right")
        z.spines[["top", "right"]].set_visible(False)
        z.grid(axis="y", alpha=.18)
        z.set_axisbelow(True)
    for i, (color, name) in enumerate(zip(colors, names)):
        suffix = str(i)
        positions = x + (i - .5) * width
        ax[0].bar(positions, series["tip" + suffix], width, color=color, label=name)
        ax[0].scatter(positions, series["max" + suffix], color=color, marker="_", s=90, zorder=3)
        ax[1].bar(positions, series["iou" + suffix], width, color=color)
        ax[2].bar(positions, series["lost" + suffix], width, color=color)
        ax[2].bar(positions, series["added" + suffix], width, bottom=series["lost" + suffix], color=color, alpha=.55, hatch="///")
        ax[3].bar(positions, series["flip" + suffix], width, color=color)
    for j in range(6):
        label = "N=" + str(series["eligible0"][j]) + "/" + str(series["eligible1"][j])
        ax[3].text(x[j], max(series["flip0"][j], series["flip1"][j]) + .15, label, ha="center", va="bottom", fontsize=7.5, color="#333")
    ax[0].set_title("Native fingertip alignment")
    ax[0].set_ylabel("Mean error (bars); maximum (marks), mm")
    ax[0].axhline(10, color="#555", ls="--", lw=1)
    ax[0].text(.02, .95, "10 mm guard applies to mean; per-tip cap is original error", transform=ax[0].transAxes, fontsize=8, va="top")
    ax[0].set_ylim(0, max(12, max(series["max0"] + series["max1"]) * 1.22))
    ax[1].set_title("Whole-hand unsigned proximity")
    ax[1].set_ylabel("2 mm centroid IoU (estimate)")
    ax[1].set_ylim(0, max(series["iou0"] + series["iou1"]) * 1.23)
    ax[2].set_title("Definite lost / added proximity")
    ax[2].set_ylabel("Sum of six part-area lower bounds, mm²")
    ax[3].set_title("Object-facing direction reversals")
    ax[3].set_ylabel("Reliable sampled reversals (queries)")
    ax[3].set_ylim(0, max(series["flip0"] + series["flip1"]) * 1.3 + 2)
    ax[3].text(.02, .96, "N = original / returned eligible queries", transform=ax[3].transAxes, fontsize=8, va="top")
    handles, legend = ax[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#888", hatch="///", alpha=.55))
    legend.append("Added proximity (hatched)")
    fig.legend(handles, legend, ncol=3, loc="upper center", bbox_to_anchor=(.5, .944), frameon=False)
    fig.suptitle(f"HUG constrained fit: {report['returned_surface_correction_count']}/6 surface corrections returned", fontsize=17, y=.99)
    counts = report["strict_status_counts"]
    fig.text(.5, .071, f"Unchanged strict rule, nine resolutions: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['UNRESOLVED']} UNRESOLVED. Engineering checks are separate.", ha="center", fontsize=10)
    fig.text(.5, .047, "Original cells: maximum triangle area 4 mm²; proximity band 2 mm. Part-area sums may overlap and are not physical contact area.", ha="center", fontsize=9, color="#444")
    fig.text(.5, .025, "Direction counts use each mapping's actual eligible domain. Counts alone cannot establish better direction preservation; cell bounds remain in the report.", ha="center", fontsize=9, color="#444")
    fig.subplots_adjust(top=.88, bottom=.18, wspace=.29, hspace=.58)
    figure = B / "constrained_candidate_tradeoffs.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.7), sharex=True)
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Little"]
    finger_x = np.arange(5)
    for z, c, label in zip(axs.ravel(), cases, labels):
        original = np.array(c["original_landmarks"]["fingertip_errors_m"]) * 1000
        candidate = np.array(c["constrained_landmarks"]["fingertip_errors_m"]) * 1000
        z.plot(finger_x, original, "o-", color=colors[0], label=names[0])
        z.plot(finger_x, candidate, "s--", color=colors[1], label=names[1])
        z.set_title(label)
        z.set_xticks(finger_x, finger_names, rotation=30, ha="right")
        z.set_ylabel("Native fingertip error, mm")
        z.set_ylim(0, max(1.0, max(original.max(), candidate.max()) * 1.15))
        z.grid(alpha=.18)
        z.spines[["top", "right"]].set_visible(False)
    fig.legend(*axs.ravel()[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
    fig.suptitle("Each fingertip retains its original geometric error cap", fontsize=17, y=.99)
    fig.text(.5, .024, "Actual independent FK errors on the same saved native targets; final error ≤ original error + 10 nm numerical allowance. No quality or lifting filter.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.86, bottom=.14, wspace=.27, hspace=.48)
    tips = B / "constrained_fingertip_errors.png"
    fig.savefig(tips, dpi=180)
    plt.close(fig)
    metadata = {"schema": "hug_constrained_scientific_figures_v1", "script_sha256": sha(__file__), "summary_sha256": sha(B / "summary.json"), "figure_sha256": {str(p.name): sha(p) for p in (figure, tips)}, "case_order": [c["case_id"] for c in cases], "maximum_triangle_area_mm2": 4, "proximity_band_mm": 2, "no_physical_contact_area_or_label_claim": True, "direction_denominators_change": True, "all_fingertips_retained": True, "plot_series": series}
    (B / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"figures": [str(figure), str(tips)], "metadata": str(B / "figure_metadata.json")}, indent=2))

if __name__ == "__main__":
    main()
