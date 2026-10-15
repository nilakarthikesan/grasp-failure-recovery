#!/usr/bin/env python3
"""Re-score portable historical prefixes with immutable saved logistic weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from grasp_failure_prediction.prediction.model import load_model, outcome_metrics, summarize_prefix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(project: Path) -> dict:
    root = project / "research_artifacts"
    original = root / "object_scene_pilot_v1"
    fresh = root / "fresh_object_scene_eval_v1"
    model_root = root / "task_outcome_baseline_v1"
    frozen = json.loads((model_root / "report.json").read_text())
    report = {"schema_version": "historical_prediction_readback_v1", "optimization_calls": 0,
              "source_sha256": sha(Path(__file__)), "bindings": {}, "metrics": {}}
    for path in (original, fresh):
        manifest = json.loads((path / "manifest.json").read_text())
        assert sha(path / "dataset.npz") == manifest["dataset_sha256"].removeprefix("sha256:")
        report["bindings"][path.name] = {name: sha(path / name) for name in ("manifest.json", "dataset.npz")}
    splits = json.loads((original / "splits.json").read_text())
    with np.load(original / "dataset.npz", allow_pickle=False) as a:
        old = {name: a[name] for name in a.files}
    with np.load(fresh / "dataset.npz", allow_pickle=False) as a:
        new = {name: a[name] for name in a.files}
    test_ids = set(splits["partitions"]["test"]["episode_ids"])
    test = np.array([str(v) in test_ids for v in old["episode_ids"]])
    seen = np.isin(new["geometry_group_ids"], np.unique(old["geometry_group_ids"]))
    for name, commands in (("proprioception_and_commands", False), ("commands_only", True)):
        model_path = model_root / (name + ".npz")
        assert sha(model_path) == frozen["models"][name]["model_sha256"]
        model = load_model(model_path)
        old_p = model.predict_success(summarize_prefix(old["X"], commands_only=commands))
        new_p = model.predict_success(summarize_prefix(new["X"], commands_only=commands))
        original_metrics = outcome_metrics(old["y"][test], old_p[test])
        expected = frozen["models"][name]["metrics"]["test"]
        for key in ("failure_roc_auc", "failure_average_precision", "brier_score", "log_loss"):
            np.testing.assert_allclose(original_metrics[key], expected[key], rtol=0, atol=1e-12)
        assert original_metrics["failure_confusion"] == expected["failure_confusion"]
        report["bindings"][model_path.name] = sha(model_path)
        report["metrics"][name] = {"original_test": original_metrics,
            "fresh_seen_geometry": outcome_metrics(new["y"][seen], new_p[seen]),
            "fresh_new_geometry": outcome_metrics(new["y"][~seen], new_p[~seen])}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify(args.project)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["metrics"], indent=2))
