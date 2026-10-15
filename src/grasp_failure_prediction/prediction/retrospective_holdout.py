"""Retrospective object holdouts of already inspected historical HUG prefixes.

This is a new development analysis, never a revision of the frozen-model test.
Fold membership is computed from source identities without consulting outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

from .model import outcome_metrics, select_model, summarize_prefix


SCHEMA = "hug_retrospective_object_holdout_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_folds(geometry, observation, proposal) -> list[dict]:
    """Hold out each geometry; validate on the last capture per other geometry.

    Lexicographic capture identity, fixed before fitting, chooses validation.
    All observations and proposal families are disjoint across partitions.
    """
    g, o, p = [np.asarray(v) for v in (geometry, observation, proposal)]
    if (g.ndim != 1 or not len(g) or o.shape != g.shape or p.shape != g.shape
            or any(v.dtype.kind not in "US" or not np.all(v != "") for v in (g, o, p))):
        raise ValueError("nonempty aligned string identities required")
    for key, sources in ((o, g), (p, o)):
        if any(len(set(sources[key == identity])) != 1 for identity in set(key)):
            raise ValueError("one observation/proposal has conflicting parent identities")
    objects = sorted(set(g))
    if len(objects) < 2:
        raise ValueError("object holdout needs at least two geometries")
    folds = []
    for held in objects:
        test = g == held
        validation = np.zeros(len(g), dtype=bool)
        for other in objects:
            if other == held:
                continue
            captures = sorted(set(o[g == other]))
            if len(captures) < 2:
                raise ValueError("each training geometry needs two captures")
            validation |= (g == other) & (o == captures[-1])
        train = ~(test | validation)
        indices = {"train": np.flatnonzero(train), "validation": np.flatnonzero(validation),
                   "test": np.flatnonzero(test)}
        for role, rows in indices.items():
            if not len(rows):
                raise ValueError("empty " + role)
        for identities in (o, p):
            sets = [set(identities[rows]) for rows in indices.values()]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
                raise ValueError("group leakage")
        folds.append({"held_geometry": held, "indices": indices})
    return folds


def calibration(y, probability, bins=10) -> dict:
    """Fixed equal-width success-probability bins; no test fitting."""
    y, probability = np.asarray(y), np.asarray(probability)
    outcome_metrics(y, probability)
    membership = np.minimum((probability * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        selected = membership == index
        if selected.any():
            rows.append({"lower": index / bins, "upper": (index + 1) / bins,
                         "n": int(selected.sum()), "mean_p_success": float(probability[selected].mean()),
                         "observed_success_fraction": float(y[selected].mean())})
    return {"bins": rows, "expected_calibration_error": sum(
        row["n"] * abs(row["mean_p_success"] - row["observed_success_fraction"])
        for row in rows) / len(y)}


def group_uncertainty(y, probability, groups, *, seed=20261015, repeats=1000) -> dict:
    """Resample proposal families, including all their correlated noise variants."""
    groups = np.asarray(groups)
    if groups.shape != np.asarray(y).shape or repeats < 1:
        raise ValueError("aligned groups and positive repeat count required")
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("failure_roc_auc", "success_false_alarm_rate", "failure_recall")}
    for _ in range(repeats):
        selected = np.concatenate([np.flatnonzero(groups == group)
                                   for group in rng.choice(unique, size=len(unique), replace=True)])
        result = outcome_metrics(np.asarray(y)[selected], np.asarray(probability)[selected])
        for name in values:
            if result[name] is not None:
                values[name].append(result[name])
    return {"unit": "proposal family; conditional on this one held-out geometry",
            "independent_groups": len(unique), "seed": seed, "repeats": repeats,
            "percentile_95": {name: {"valid_repeats": len(v),
                "interval": np.quantile(v, [0.025, 0.975]).tolist() if v else None}
                for name, v in values.items()}}


def load_sources(paths: list[Path]):
    manifests, arrays, identities, bindings = [], [], [], {}
    for path in paths:
        manifest_path = path / "manifest.json"
        m = json.loads(manifest_path.read_text())
        if m["dataset_sha256"].removeprefix("sha256:") != digest(path / "dataset.npz"):
            raise ValueError("dataset hash mismatch")
        with np.load(path / "dataset.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in ("X", "y", "episode_ids", "group_ids",
                                                "observation_group_ids", "geometry_group_ids")}
        n = len(data["y"])
        if (data["X"].shape != (n, 1440) or not np.isfinite(data["X"]).all()
                or any(data[key].shape != (n,) for key in data if key != "X")
                or not np.isin(data["y"], [0, 1]).all()):
            raise ValueError("invalid historical arrays")
        episodes = m["episodes"]
        if len(episodes) != n:
            raise ValueError("manifest row count mismatch")
        for i, row in enumerate(episodes):
            if (row["source_task_label"] != int(data["y"][i]) or any(
                    str(data[key][i]) != row[key[:-1]] for key in
                    ("episode_ids", "group_ids", "observation_group_ids", "geometry_group_ids"))):
                raise ValueError("manifest row identity/label mismatch")
        if manifests and any(m[key] != manifests[0][key] for key in
                             ("input_contract", "label_rule", "label_scorer_sha256", "feature_columns")):
            raise ValueError("historical contracts cannot be pooled")
        arrays.append(data)
        manifests.append(m)
        identities.extend(path.name + "/" + str(episode) for episode in data["episode_ids"])
        bindings[path.name] = {filename: digest(path / filename)
                               for filename in ("dataset.npz", "manifest.json")}
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate namespaced episode identities")
    joined = {key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
    return joined, identities, bindings


def run(paths: list[Path], output: Path, protocol: Path) -> dict:
    if output.exists():
        raise FileExistsError("analysis output must be fresh")
    data, ids, bindings = load_sources(paths)
    folds = object_folds(data["geometry_group_ids"], data["observation_group_ids"], data["group_ids"])
    y = data["y"]
    summaries = {name: summarize_prefix(data["X"], commands_only=commands)
                 for name, commands in (("proprioception_and_commands", False), ("commands_only", True))}
    report = {"schema_version": SCHEMA, "source_bindings": bindings,
              "protocol_sha256": digest(protocol), "source_sha256": digest(Path(__file__)),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "scope": "retrospective development on previously inspected historical data; no new simulation",
              "attempts": len(y), "successes": int(y.sum()), "failures": int(len(y)-y.sum()),
              "geometry_groups": len(np.unique(data["geometry_group_ids"])),
              "observation_groups": len(np.unique(data["observation_group_ids"])),
              "proposal_groups": len(np.unique(data["group_ids"])), "folds": []}
    output.mkdir(parents=True)
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["held_geometry", "model", "episode_id", "success", "p_success"])
        for fold in folds:
            indices = fold["indices"]
            train, validation, test = (indices[role] for role in ("train", "validation", "test"))
            row = {"held_geometry": fold["held_geometry"], "partitions": {
                role: {"episodes": [ids[i] for i in index], "n": len(index),
                       "observations": len(set(data["observation_group_ids"][index])),
                       "proposals": len(set(data["group_ids"][index]))}
                for role, index in indices.items()}, "models": {}}
            for name in ("training_prevalence", *summaries):
                if name == "training_prevalence":
                    probability = np.full(len(test), float(y[train].mean()))
                    selection = {"p_success": float(y[train].mean())}
                else:
                    features = summaries[name]
                    model, candidates = select_model(features[train], y[train], features[validation], y[validation])
                    probability = model.predict_success(features[test])
                    selection = {"regularization": model.regularization, "candidates": candidates}
                row["models"][name] = {"selection": selection,
                    "metrics": outcome_metrics(y[test], probability),
                    "calibration": calibration(y[test], probability),
                    "group_uncertainty": group_uncertainty(y[test], probability, data["group_ids"][test])}
                for i, p in zip(test, probability, strict=True):
                    writer.writerow([fold["held_geometry"], name, ids[i], int(y[i]), format(p, ".17g")])
            report["folds"].append(row)
    report["predictions_sha256"] = digest(output / "predictions.csv")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.sources, args.output, args.protocol)
    print(json.dumps({"attempts": report["attempts"], "folds": [{
        "held_geometry": f["held_geometry"], "models": {n: v["metrics"] for n, v in f["models"].items()}}
        for f in report["folds"]]}, indent=2))


if __name__ == "__main__":
    main()
