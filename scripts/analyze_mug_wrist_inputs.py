"""Explain saved mug scores against original training wrist features; no fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from grasp_failure_prediction.prediction.model import load_model, summarize_prefix


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def analyze(repository: Path, output: Path) -> dict:
    pilot = repository / "research_artifacts/object_scene_pilot_v1"
    diagnostic = repository / "research_artifacts/corrected_mug_model_diagnostic_v1"
    baseline = repository / "research_artifacts/task_outcome_baseline_v1"
    model_path = baseline / "proprioception_and_commands.npz"
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    sources = [model_path, baseline / "report.json", pilot / "dataset.npz",
               pilot / "splits.json", pilot / "manifest.json", diagnostic / "dataset.npz",
               diagnostic / "predictions.csv", repository / "src/grasp_failure_prediction/prediction/model.py"]
    before = {str(p.relative_to(repository)): sha(p) for p in sources}
    training_report = json.loads((baseline / "report.json").read_text())
    for key, path in (("input_dataset_sha256", pilot / "dataset.npz"),
                      ("input_splits_sha256", pilot / "splits.json"),
                      ("input_manifest_sha256", pilot / "manifest.json")):
        if sha(path) != training_report[key]:
            raise ValueError(f"Original training source changed: {path.name}")
    if (sha(model_path) != training_report["models"]["proprioception_and_commands"]["model_sha256"]
            or sha(sources[-1]) != training_report["training_source_sha256"]["model.py"]):
        raise ValueError("Frozen model or prediction implementation changed")
    split = json.loads((pilot / "splits.json").read_text())
    train_ids = split["partitions"]["train"]["episode_ids"]
    with np.load(pilot / "dataset.npz", allow_pickle=False) as values:
        train_mask = np.isin(values["episode_ids"], train_ids)
        if train_mask.sum() != 520 or len(set(train_ids)) != 520:
            raise ValueError("Expected the original 520 training episodes only")
        train_features = summarize_prefix(values["X"][train_mask])
    with np.load(diagnostic / "dataset.npz", allow_pickle=False) as values:
        features = summarize_prefix(values["X"])
        episode_ids = values["episode_ids"].copy()
        labels = values["y"].copy()
        names = values["joint_names"].tolist()
    if names != training_report["input_contract"]["joint_names"] or names[:2] != ["WRJ2", "WRJ1"]:
        raise ValueError("Joint ordering changed")
    model = load_model(model_path)
    mean, std = train_features.mean(axis=0), train_features.std(axis=0)
    active = std > 1e-8
    if (not np.array_equal(mean, model.normalizer.mean)
            or not np.array_equal(np.where(active, std, 1.0), model.normalizer.scale)
            or not np.array_equal(active, model.normalizer.active)):
        raise ValueError("Training statistics do not reproduce saved normalization")
    coefficients = np.zeros(features.shape[1])
    coefficients[active] = model.weights
    standardized = (features - mean) / model.normalizer.scale
    contribution = standardized * coefficients
    minimum, maximum = train_features.min(axis=0), train_features.max(axis=0)
    outside = (features < minimum) | (features > maximum)
    wrist_mask = np.arange(features.shape[1]) % 24 < 2
    wrist_active = wrist_mask & active
    stats = training_report["summary_order"]
    fields = training_report["field_order"]
    with (diagnostic / "predictions.csv").open() as stream:
        predictions = {row["episode_id"]: row for row in csv.DictReader(stream)}
    if set(predictions) != set(episode_ids.tolist()):
        raise ValueError("Saved prediction inventory changed")
    records, feature_rows = [], []
    for i, episode_id in enumerate(episode_ids):
        original = predictions[str(episode_id)]
        if int(original["label_success"]) != int(labels[i]):
            raise ValueError("Saved labels disagree")
        wrist = float(contribution[i, wrist_mask].sum())
        other = float(contribution[i, ~wrist_mask].sum())
        logit = wrist + other + model.intercept
        # Reconstruct the exact existing linear score, not an altered prediction.
        expected_logit = float(np.einsum("j,j->", standardized[i, active], model.weights) + model.intercept)
        probability = float(model.predict_success(features[i:i + 1])[0])
        if not np.isclose(logit, expected_logit, rtol=0, atol=1e-12):
            raise ValueError("Attribution does not reconstruct saved model score")
        if not np.isclose(probability, float(original["p_success"]), rtol=0, atol=1e-14):
            raise ValueError("Prediction differs from frozen scoring result")
        record = dict(episode_id=str(episode_id), role=original["role"],
                      label_success=int(labels[i]), noise_amplitude_rad=float(original["noise_amplitude_rad"]),
                      p_failure=float(original["p_failure"]), success_logit=logit,
                      wrist_net_success_logit_contribution=wrist,
                      nonwrist_net_success_logit_contribution=other,
                      intercept=model.intercept,
                      wrist_absolute_contribution_fraction=float(np.abs(contribution[i, wrist_mask]).sum() / np.abs(contribution[i]).sum()),
                      active_wrist_features_outside_training_range=int(outside[i, wrist_active].sum()),
                      active_wrist_features_abs_z_above_5=int((np.abs(standardized[i, wrist_active]) > 5).sum()))
        records.append(record)
        for j in np.flatnonzero(wrist_mask):
            feature_rows.append(dict(episode_id=str(episode_id), feature_index=int(j),
                                     statistic=stats[j // 72], field=fields[(j % 72) // 24],
                                     joint=names[j % 24], active=bool(active[j]), value=float(features[i, j]),
                                     training_min=float(minimum[j]), training_max=float(maximum[j]),
                                     training_mean=float(mean[j]), training_scale=float(model.normalizer.scale[j]),
                                     standardized_value=float(standardized[i, j]),
                                     outside_training_range=bool(outside[i, j]), coefficient=float(coefficients[j]),
                                     success_logit_contribution=float(contribution[i, j])))
    successful = np.char.startswith(episode_ids, "mug_") & (labels == 1)
    successes = [r for r in records if r["role"] == "mug" and r["label_success"] == 1]
    comparisons = []
    for field, offset in (("hand_qpos", 0), ("commanded_target_rad", 48)):
        for joint in (0, 1):
            j = offset + joint
            comparisons.append(dict(statistic="last", field=field, joint=names[joint],
                                    unit="rad", training_min=float(minimum[j]), training_max=float(maximum[j]),
                                    successful_mug_min=float(features[successful, j].min()),
                                    successful_mug_max=float(features[successful, j].max()),
                                    successful_mugs_outside_training_range=int(outside[successful, j].sum())))
    after = {str(p.relative_to(repository)): sha(p) for p in sources}
    if before != after:
        raise ValueError("Original data or model changed during analysis")
    report = dict(schema_version="frozen_mug_wrist_input_diagnostic_v1", date="2026-10-12",
                  training_episodes=520, scored_episodes=len(records), successful_mugs=len(successes),
                  active_wrist_features=int(wrist_active.sum()),
                  normalization_reproduced_exactly=True, original_sources_unchanged=True,
                  source_sha256=before, analysis_script_sha256=sha(Path(__file__)),
                  successful_mug_last_wrist_comparisons=comparisons,
                  successful_mugs_with_negative_wrist_net_contribution=sum(r["wrist_net_success_logit_contribution"] < 0 for r in successes),
                  successful_mug_wrist_net_contribution_range=[min(r["wrist_net_success_logit_contribution"] for r in successes), max(r["wrist_net_success_logit_contribution"] for r in successes)],
                  successful_mug_wrist_absolute_contribution_fraction_range=[min(r["wrist_absolute_contribution_fraction"] for r in successes), max(r["wrist_absolute_contribution_fraction"] for r in successes)],
                  prediction_time_s=2.96, threshold_unchanged=0.5,
                  interpretation="Both wrist commands are outside training range for all 20 successful mugs; the net wrist terms push the saved linear score toward failure in all 20",
                  scope="Observational explanation of the exact unchanged model score; no model change, counterfactual prediction, new simulation, causal isolation, or improved model result")
    output.mkdir(parents=True)
    write_json(output / "report.json", report)
    for filename, rows in (("episode_attribution.csv", records), ("wrist_features.csv", feature_rows)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(Path(__file__).resolve().parents[1], args.output.resolve())
    print(json.dumps({k: report[k] for k in ("training_episodes", "successful_mugs", "successful_mug_last_wrist_comparisons", "successful_mugs_with_negative_wrist_net_contribution")}, indent=2))


if __name__ == "__main__":
    main()
