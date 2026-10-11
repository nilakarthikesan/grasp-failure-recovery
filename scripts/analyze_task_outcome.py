"""Describe frozen predictions by group and perturbation; never fit a model."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from grasp_failure_prediction.prediction.dataset import load_dataset
from grasp_failure_prediction.prediction.model import outcome_metrics


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(dataset_dir, model_dir, runs_root, output):
    dataset_dir, model_dir, runs_root, output = map(Path, (dataset_dir, model_dir, runs_root, output))
    if output.exists():
        raise FileExistsError("analysis output must be new")
    bundle = load_dataset(dataset_dir)
    report = json.loads((model_dir / "report.json").read_text())
    for filename, key in (("dataset.npz", "input_dataset_sha256"),
                          ("manifest.json", "input_manifest_sha256"),
                          ("splits.json", "input_splits_sha256")):
        if sha256(dataset_dir / filename) != report[key]:
            raise ValueError("dataset differs from the frozen training report")
    if sha256(model_dir / "predictions.csv") != report["predictions_sha256"]:
        raise ValueError("prediction file differs from the frozen report")
    source = {row["episode_id"]: row for row in bundle.manifest["episodes"]}
    partition = {episode: role for role, data in bundle.splits["partitions"].items()
                 for episode in data["episode_ids"]}
    metadata = {}
    for episode, row in source.items():
        path = runs_root / episode / "collection_metadata.json"
        if "sha256:" + sha256(path) != row["raw_files_sha256"][path.name]:
            raise ValueError("analysis metadata differs from the exported source")
        metadata[episode] = json.loads(path.read_text())

    groups = defaultdict(list)
    seen = set()
    with (model_dir / "predictions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            episode, model = row["episode_id"], row["model"]
            key = (episode, model)
            if key in seen or episode not in source or model not in report["models"]:
                raise ValueError("invalid or repeated prediction identity")
            seen.add(key)
            label, p = int(row["label_success"]), float(row["p_success"])
            if (label != source[episode]["source_task_label"] or row["partition"] != partition[episode]
                    or abs(1 - p - float(row["p_failure"])) > 1e-15):
                raise ValueError("prediction labels or membership differ")
            m, role = metadata[episode], row["partition"]
            for dimension, value in (("all", "all"), ("scene_seed", str(source[episode]["scene_seed"])),
                                     ("proposal_group", source[episode]["group_id"]),
                                     ("noise_amplitude_rad", str(m["noise_amplitude_rad"])),
                                     ("outcome_category", m["failure_type"] or "success")):
                groups[(role, model, dimension, value)].append((label, p))
    if len(seen) != len(source) * len(report["models"]):
        raise ValueError("predictions do not cover all episodes and models")
    records = []
    for (role, model, dimension, value), rows in sorted(groups.items()):
        labels, probabilities = np.array(rows).T
        metrics = outcome_metrics(labels, probabilities)
        if dimension == "all" and metrics != report["models"][model]["metrics"][role]:
            raise ValueError("recomputed metrics differ from the frozen report")
        records.append({"partition": role, "model": model, "dimension": dimension,
                        "value": value, "metrics": metrics})
    result = {"schema_version": "task_outcome_descriptive_analysis_v1",
              "scope": "post-hoc descriptions of unchanged predictions; no fitting, selection or threshold changes",
              "report_sha256": sha256(model_dir / "report.json"),
              "predictions_sha256": report["predictions_sha256"],
              "metadata_contract": "all metadata hashes verified against the export; used only for analysis",
              "limitations": "ROC is undefined without both classes; failure AP is undefined without failures; four drops cannot establish drop detection",
              "records": records}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.dataset, args.models, args.runs_root, args.output)
    print(json.dumps({"groups": len(result["records"]), "output": str(args.output)}))
