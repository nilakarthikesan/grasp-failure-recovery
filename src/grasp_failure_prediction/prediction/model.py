"""Causal prefix baselines for the frozen simulated task outcome.

These models predict final task success, not slip onset or a fixed-horizon loss.
Only encoder and command history enters the model; outcome data is supervision.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit


MODEL_SCHEMA = "task_outcome_logistic_v1"
REGULARIZATION_GRID = (0.001, 0.01, 0.1, 1.0)


def summarize_prefix(X: np.ndarray, *, history_steps: int = 20,
                     joint_count: int = 24, commands_only: bool = False) -> np.ndarray:
    """Summarize time-major [qpos, qvel, applied target] histories.

    The four statistic blocks are last, mean, population standard deviation,
    and last-minus-first. Columns within a block preserve field/joint order.
    """
    values = np.asarray(X, dtype=np.float64)
    if (values.ndim != 2 or history_steps < 2 or joint_count < 1
            or values.shape[1] != history_steps * 3 * joint_count
            or not np.isfinite(values).all()):
        raise ValueError("invalid observable prefix matrix")
    history = values.reshape(len(values), history_steps, 3 * joint_count)
    if commands_only:
        history = history[:, :, 2 * joint_count:]
    return np.concatenate((history[:, -1], history.mean(axis=1),
                           history.std(axis=1), history[:, -1] - history[:, 0]), axis=1)


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray
    active: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Normalizer":
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("normalization requires finite training rows")
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        active = std > 1e-8
        return cls(mean, np.where(active, std, 1.0), active)

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != len(self.mean)
                or not np.isfinite(values).all()):
            raise ValueError("model feature shape or values are invalid")
        return ((values - self.mean) / self.scale)[:, self.active]


@dataclass(frozen=True)
class LogisticModel:
    normalizer: Normalizer
    weights: np.ndarray
    intercept: float
    regularization: float
    iterations: int
    converged: bool

    def predict_success(self, X: np.ndarray) -> np.ndarray:
        return expit(np.einsum("ij,j->i", self.normalizer.transform(X), self.weights)
                     + self.intercept)


def _labels(y: np.ndarray, *, both_classes: bool = False) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim != 1 or not len(values) or not np.isin(values, (0, 1)).all():
        raise ValueError("labels must be a nonempty vector of 0=failure, 1=success")
    if both_classes and len(np.unique(values)) != 2:
        raise ValueError("training requires both outcome classes")
    return values.astype(np.float64)


def fit_logistic(X_train: np.ndarray, y_train: np.ndarray, *,
                 regularization: float, max_iterations: int = 400) -> LogisticModel:
    y = _labels(y_train, both_classes=True)
    if len(X_train) != len(y) or not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("invalid training length or regularization")
    normalizer = Normalizer.fit(X_train)
    X = normalizer.transform(X_train)
    initial = np.zeros(X.shape[1] + 1)
    prevalence = float(y.mean())
    initial[-1] = np.log(prevalence / (1 - prevalence))

    def objective(theta):
        weights, intercept = theta[:-1], theta[-1]
        # Explicit contractions also avoid spurious floating-point flags in
        # this macOS NumPy/BLAS runtime on small, finite matrix-vector products.
        z = np.einsum("ij,j->i", X, weights) + intercept
        residual = expit(z) - y
        loss = np.mean(np.logaddexp(0, z) - y * z)
        loss += 0.5 * regularization * float(np.sum(weights * weights))
        gradient = np.concatenate((np.einsum("ij,i->j", X, residual) / len(y) + regularization * weights,
                                   [residual.mean()]))
        return float(loss), gradient

    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iterations, "ftol": 1e-10, "gtol": 1e-6})
    return LogisticModel(normalizer, result.x[:-1], float(result.x[-1]),
                         float(regularization), int(result.nit), bool(result.success))


def _roc_auc(positive: np.ndarray, score: np.ndarray) -> float | None:
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    if not n_positive or not n_negative:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


def _average_precision(positive: np.ndarray, score: np.ndarray) -> float | None:
    if not positive.any():
        return None
    order = np.argsort(-score, kind="stable")
    truth, sorted_scores = positive[order], score[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, len(order)]
    tp = np.cumsum(truth)[ends - 1]
    recall = tp / positive.sum()
    precision = tp / ends
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def outcome_metrics(y: np.ndarray, p_success: np.ndarray) -> dict:
    truth = _labels(y).astype(bool)
    probability = np.asarray(p_success, dtype=np.float64)
    if (probability.shape != truth.shape or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()):
        raise ValueError("invalid success probabilities")
    failure, p_failure = ~truth, 1 - probability
    alert = p_failure >= 0.5
    tp, fp = int((alert & failure).sum()), int((alert & truth).sum())
    tn, fn = int((~alert & truth).sum()), int((~alert & failure).sum())
    recall_failure = tp / (tp + fn) if tp + fn else None
    recall_success = tn / (tn + fp) if tn + fp else None
    bounded = np.clip(probability, 1e-12, 1 - 1e-12)
    return {
        "episodes": len(truth), "successes": int(truth.sum()), "failures": int(failure.sum()),
        "accuracy": (tp + tn) / len(truth),
        "balanced_accuracy": ((recall_failure + recall_success) / 2
                              if recall_failure is not None and recall_success is not None else None),
        "log_loss": float(-np.mean(truth * np.log(bounded) + failure * np.log1p(-bounded))),
        "brier_score": float(np.mean((probability - truth) ** 2)),
        "failure_roc_auc": _roc_auc(failure, p_failure),
        "failure_average_precision": _average_precision(failure, p_failure),
        "failure_recall": recall_failure,
        "failure_precision": tp / (tp + fp) if tp + fp else 0.0,
        "success_false_alarm_rate": fp / (fp + tn) if fp + tn else None,
        "failure_confusion": {"true_positive": tp, "false_positive": fp,
                              "true_negative": tn, "false_negative": fn},
        "decision_rule": "alert if p_failure >= 0.5; p_failure = 1 - p_success",
    }


def select_model(X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray,
                 y_validation: np.ndarray, *, grid=REGULARIZATION_GRID) -> tuple[LogisticModel, list[dict]]:
    candidates = []
    models = []
    for regularization in grid:
        model = fit_logistic(X_train, y_train, regularization=regularization)
        metrics = outcome_metrics(y_validation, model.predict_success(X_validation))
        candidates.append({"regularization": float(regularization), "converged": model.converged,
                           "iterations": model.iterations, "validation": metrics})
        models.append(model)
    eligible = [i for i, model in enumerate(models) if model.converged]
    if not eligible:
        raise RuntimeError("no regularization candidate converged")
    best = min(eligible, key=lambda i: (candidates[i]["validation"]["log_loss"],
                                        -models[i].regularization))
    return models[best], candidates


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_model(model: LogisticModel, output: Path) -> None:
    np.savez_compressed(output, mean=model.normalizer.mean, scale=model.normalizer.scale,
                        active=model.normalizer.active, weights=model.weights,
                        intercept=np.array(model.intercept), regularization=np.array(model.regularization),
                        iterations=np.array(model.iterations), converged=np.array(model.converged))


def load_model(path: str | Path) -> LogisticModel:
    with np.load(path, allow_pickle=False) as values:
        model = LogisticModel(Normalizer(values["mean"], values["scale"], values["active"]),
                              values["weights"], float(values["intercept"]),
                              float(values["regularization"]), int(values["iterations"]),
                              bool(values["converged"]))
    if (model.normalizer.mean.ndim != 1 or model.normalizer.scale.shape != model.normalizer.mean.shape
            or model.normalizer.active.shape != model.normalizer.mean.shape
            or model.normalizer.active.dtype != np.bool_
            or model.weights.ndim != 1 or len(model.weights) != int(model.normalizer.active.sum())
            or not np.isfinite(model.normalizer.mean).all()
            or not np.isfinite(model.normalizer.scale).all() or (model.normalizer.scale <= 0).any()
            or not np.isfinite(model.weights).all() or not np.isfinite(model.intercept)
            or not np.isfinite(model.regularization) or model.regularization <= 0
            or model.iterations < 0):
        raise ValueError("invalid saved model")
    return model


def predict_episode(model_path: str | Path, episode_dir: str | Path) -> dict:
    """Predict from one recorded closing prefix without opening outcome files."""
    from .dataset import _prefix

    model_path = Path(model_path)
    report = json.loads((model_path.parent / "report.json").read_text())
    if report.get("model_schema") != MODEL_SCHEMA:
        raise ValueError("unsupported model contract")
    names = {value.get("model_file"): name for name, value in report["models"].items()
             if "model_file" in value}
    model_name = names.get(model_path.name)
    if model_name not in ("proprioception_and_commands", "commands_only"):
        raise ValueError("model file is not in its report")
    if _sha256(model_path) != report["models"][model_name]["model_sha256"]:
        raise ValueError("model hash mismatch")
    prefix, joint_names, observed_at = _prefix(Path(episode_dir))
    if joint_names != report["input_contract"]["joint_names"]:
        raise ValueError("episode joint ordering differs from the model")
    features = summarize_prefix(prefix[None, :], commands_only=model_name == "commands_only")
    probability = float(load_model(model_path).predict_success(features)[0])
    return {"schema_version": "task_outcome_prediction_v1", "model": model_name,
            "observed_through_s": observed_at, "p_success": probability,
            "p_failure": 1 - probability, "predicted_task_success": probability > 0.5,
            "scope": "frozen eventual task outcome; not slip onset or a recovery decision"}


def train_baselines(dataset_dir: str | Path, output_dir: str | Path) -> dict:
    from .dataset import load_dataset

    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    bundle = load_dataset(dataset_dir)
    contract = bundle.manifest["input_contract"]
    if (contract["prediction_time_s"] != 2.96 or contract["history_steps"] != 20
            or len(contract["joint_names"]) != 24
            or contract["fields"] != ["hand_qpos", "hand_qvel", "commanded_target_rad"]):
        raise ValueError("this model requires the frozen pre-lift observable contract")
    if output_dir.exists():
        raise FileExistsError("training output must be a fresh directory")
    ids = [str(value) for value in bundle.episode_ids]
    lookup = {value: i for i, value in enumerate(ids)}
    if len(lookup) != len(ids):
        raise ValueError("duplicate episode identities")
    partitions = bundle.splits["partitions"]
    indices = {role: np.array([lookup[value] for value in partitions[role]["episode_ids"]])
               for role in ("train", "validation", "test")}
    if any(not len(index) for index in indices.values()):
        raise ValueError("three nonempty partitions required")
    flattened = np.concatenate(list(indices.values()))
    if len(flattened) != len(ids) or len(np.unique(flattened)) != len(ids):
        raise ValueError("partitions must cover each row exactly once")
    train, validation = indices["train"], indices["validation"]
    y = bundle.y
    output_dir.mkdir(parents=True)
    report = {
        "schema_version": "task_outcome_baseline_report_v1",
        "model_schema": MODEL_SCHEMA, "target": "frozen final task outcome, 1=success, 0=failure",
        "input_dataset_sha256": _sha256(dataset_dir / "dataset.npz"),
        "input_manifest_sha256": _sha256(dataset_dir / "manifest.json"),
        "input_splits_sha256": _sha256(dataset_dir / "splits.json"),
        "prediction_time_s": 2.96, "history_steps": 20, "joint_count": 24,
        "summary_order": ["last", "mean", "population_std", "last_minus_first"],
        "field_order": ["hand_qpos", "hand_qvel", "commanded_target_rad"],
        "selection": "minimum validation log loss among converged fixed-grid models; test is not used",
        "normalization": "training-only mean/std; training std <=1e-8 excluded",
        "objective": "mean binary cross entropy + 0.5 * regularization * squared weight norm; unpenalized intercept",
        "regularization_grid": list(REGULARIZATION_GRID),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "input_contract": contract,
        "training_source_sha256": {"model.py": _sha256(Path(__file__))},
        "limitations": ["retrospective development holdouts from previously inspected simulation scenes",
                        "cube-only validation and apple-only test; both geometries appear in training",
                        "task-outcome probabilities, not fixed-horizon slip warnings or real-world failure rates"],
        "models": {},
    }
    predictions = {role: {} for role in indices}
    prevalence = float(y[train].mean())
    report["models"]["training_prevalence"] = {
        "p_success": prevalence,
        "metrics": {role: outcome_metrics(y[index], np.full(len(index), prevalence))
                    for role, index in indices.items()},
    }
    for role, index in indices.items():
        predictions[role]["training_prevalence"] = np.full(len(index), prevalence)

    for name, commands_only in (("proprioception_and_commands", False), ("commands_only", True)):
        features = summarize_prefix(bundle.X, commands_only=commands_only)
        model, candidates = select_model(features[train], y[train], features[validation], y[validation])
        path = output_dir / f"{name}.npz"
        save_model(model, path)
        # Persistent weights and normalization must reproduce all probabilities.
        restored = load_model(path)
        probability = restored.predict_success(features)
        np.testing.assert_array_equal(probability, model.predict_success(features))
        report["models"][name] = {
            "feature_count": features.shape[1], "active_feature_count": int(model.normalizer.active.sum()),
            "regularization": model.regularization, "iterations": model.iterations,
            "converged": model.converged, "selection_candidates": candidates,
            "model_file": path.name, "model_sha256": _sha256(path),
            "metrics": {role: outcome_metrics(y[index], probability[index]) for role, index in indices.items()},
        }
        for role, index in indices.items():
            predictions[role][name] = probability[index]

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["episode_id", "partition", "label_success", "model", "p_success", "p_failure"])
        for role, index in indices.items():
            for model_name, probability in predictions[role].items():
                for local, row in enumerate(index):
                    p = float(probability[local])
                    writer.writerow([ids[row], role, int(y[row]), model_name, format(p, ".17g"), format(1 - p, ".17g")])
    report["predictions_sha256"] = _sha256(output_dir / "predictions.csv")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episode", type=Path)
    args = parser.parse_args()
    if args.model is not None or args.episode is not None:
        if args.model is None or args.episode is None or args.dataset is not None or args.output is not None:
            parser.error("prediction requires --model and --episode only")
        print(json.dumps(predict_episode(args.model, args.episode), indent=2))
        return
    if args.dataset is None or args.output is None:
        parser.error("training requires dataset and --output")
    report = train_baselines(args.dataset, args.output)
    print(json.dumps({name: value["metrics"]["test"] for name, value in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
