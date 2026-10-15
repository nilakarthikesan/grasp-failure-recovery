"""Prepared comparison for NEW recorded prefixes and frozen category folds.

No historical pooling or automatic training occurs on import. Production fitting
must follow the separately frozen Phase 2/3 collection; synthetic tests exercise
software only. All five named models share rows, folds, seeds and a fixed 0.5
failure-warning threshold. Identities/filter outcomes are never feature columns.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any

import numpy as np
import scipy

from grasp_failure_prediction.evaluation.scenario_sampler import freeze_category_folds
from .model import (Normalizer, fit_logistic, load_model, outcome_metrics,
                    save_model, summarize_prefix)
from .recorded_dataset import RecordedDataset, PrefixSpec, _input_contract, load_recorded_dataset


SCHEMA = "hug_recorded_model_comparison_v1"
MODEL_NAMES = ("training_prevalence", "commands_only_logistic", "summary_logistic",
               "full_history_logistic", "small_gru")


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                  allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return file_sha256(path)


@dataclass(frozen=True)
class TrainingSpec:
    comparison_id: str = "hug_recorded_model_comparison_v1"
    seeds: tuple[int, ...] = (0, 1, 2)
    regularization_grid: tuple[float, ...] = (.001, .01, .1, 1.)
    max_logistic_iterations: int = 400
    include_gru: bool = True
    gru_candidates: tuple[tuple[int, float], ...] = ((16, .001), (16, .003), (32, .001), (32, .003))
    gru_epochs: int = 50
    gru_weight_decay: float = .0001
    warning_threshold: float = .5
    calibration_bins: int = 10

    def __post_init__(self):
        if not self.comparison_id.strip() or self.seeds != (0, 1, 2):
            raise ValueError("comparison identity and exactly seeds (0,1,2) are required")
        if self.warning_threshold != .5:
            raise ValueError("v1 warning threshold is fixed at 0.5")
        if not 1 <= len(self.regularization_grid) <= 4 or len(set(self.regularization_grid)) != len(self.regularization_grid):
            raise ValueError("declare one to four distinct regularization candidates")
        if any(not np.isfinite(x) or x <= 0 for x in self.regularization_grid):
            raise ValueError("regularization must be finite and positive")
        if not 1 <= len(self.gru_candidates) <= 4 or len(set(self.gru_candidates)) != len(self.gru_candidates):
            raise ValueError("declare one to four distinct GRU candidates")
        if any(type(h) is not int or not 1 <= h <= 64 or not np.isfinite(lr) or lr <= 0
               for h, lr in self.gru_candidates):
            raise ValueError("GRU candidates need small hidden size and positive learning rate")
        for name in ("max_logistic_iterations", "gru_epochs", "calibration_bins"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.include_gru) is not bool or not np.isfinite(self.gru_weight_decay) or self.gru_weight_decay < 0:
            raise ValueError("invalid GRU inclusion/weight decay")


@dataclass(frozen=True)
class TrainingData:
    dataset: RecordedDataset
    schedule: dict
    folds: tuple[dict, ...]
    source_bindings: dict


def _adapt_dataset(dataset: RecordedDataset, schedule: dict, folds: dict, binding: dict) -> TrainingData:
    """Validate external scenario linkage; no scenario IDs inferred from names."""
    if (schedule.get("schema_version") != "hug_scenario_schedule_v1"
            or schedule.get("outcome_selection") is not False):
        raise ValueError("require the declared outcome-independent scenario schedule")
    if folds != freeze_category_folds(schedule):
        raise ValueError("folds differ from the exact frozen schedule/category assignments")
    schedule_hash = canonical_sha256(schedule)
    if (binding.get("schema_version") != "hug_attempt_scenario_binding_v1"
            or binding.get("schedule_sha256") != schedule_hash):
        raise ValueError("attempt/scenario binding must identify the frozen schedule")
    scenarios = schedule["scenarios"]
    by_scenario = {row["scenario_id"]: row for row in scenarios}
    if len(by_scenario) != len(scenarios):
        raise ValueError("duplicate scenario IDs")
    mapping = binding.get("attempt_to_scenario")
    audit_rows = dataset.manifest["episodes"] + dataset.manifest["excluded"]
    attempts = [row["attempt_id"] for row in audit_rows]
    if (not isinstance(mapping, dict) or set(mapping) != set(attempts)
            or len(set(mapping.values())) != len(mapping)
            or set(mapping.values()) != set(by_scenario)):
        raise ValueError("binding must bijectively account for every scheduled and recorded attempt")
    # Enforce stable grouping identities, including captures and proposal variants.
    bindings: dict[str, dict] = {key: {} for key in
                               ("object_id", "geometry_group_id", "observation_group_id", "proposal_group_id")}
    for row in scenarios:
        for key in bindings:
            fields = ("category_id",) if key == "geometry_group_id" else (
                ("category_id", "geometry_group_id") if key == "object_id" else
                ("category_id", "geometry_group_id", "object_id") if key == "observation_group_id" else
                ("category_id", "geometry_group_id", "object_id", "observation_group_id"))
            value = tuple(row[field] for field in fields)
            if bindings[key].setdefault(row[key], value) != value:
                raise ValueError(f"conflicting {key} grouping identity")
    included = dataset.manifest["episodes"]
    for row in included:
        scenario = by_scenario[mapping[row["attempt_id"]]]
        for left, right in (("object_id", "object_id"), ("geometry_group_id", "geometry_group_id"),
                            ("observation_group_id", "observation_group_id"), ("group_id", "proposal_group_id")):
            if row[left] != scenario[right]:
                raise ValueError("recorded row identity conflicts with scheduled scenario")
        if row["provenance_hashes"].get("scenario") != canonical_sha256(scenario):
            raise ValueError("recorded row is not hash-bound to its scheduled physical scenario")
    adapted = []
    for fold in folds["folds"]:
        indices = {role: np.array([i for i, row in enumerate(included)
                                   if fold["assignments"][mapping[row["attempt_id"]]] == role], dtype=np.int64)
                   for role in ("train", "validation", "test")}
        for field in ("category_id", "geometry_group_id", "object_id", "observation_group_id", "proposal_group_id"):
            groups = [set(by_scenario[mapping[included[i]["attempt_id"]]][field] for i in index)
                      for index in indices.values()]
            if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
                raise ValueError(f"cross-partition leakage in {field}")
        if sorted(np.concatenate(list(indices.values())).tolist()) != list(range(len(included))):
            raise ValueError("fold rows are missing or duplicated")
        adapted.append({"fold_id": fold["fold_id"], "test_category": fold["test_category"],
                        "validation_category": fold["validation_category"], "indices": indices})
    return TrainingData(dataset, schedule, tuple(adapted), {
        "schedule_sha256": schedule_hash, "folds_sha256": canonical_sha256(folds),
        "attempt_binding_sha256": canonical_sha256(binding)})


def load_training_data(dataset_dir: str | Path, schedule_path: str | Path,
                       folds_path: str | Path, binding_path: str | Path) -> TrainingData:
    dataset_dir = Path(dataset_dir)
    paths = list(map(Path, (schedule_path, folds_path, binding_path)))
    data = _adapt_dataset(load_recorded_dataset(dataset_dir),
                           *(json.loads(path.read_text()) for path in paths))
    data.source_bindings.update({"dataset_sha256": file_sha256(dataset_dir / "dataset.npz"),
                                 "dataset_manifest_sha256": file_sha256(dataset_dir / "manifest.json"),
                                 **{name: file_sha256(path) for name, path in zip(
                                     ("schedule_file_sha256", "folds_file_sha256", "binding_file_sha256"), paths)}})
    data.source_bindings["input_files"] = {str(path.resolve()): file_sha256(path)
        for path in (dataset_dir / "dataset.npz", dataset_dir / "manifest.json", *paths)}
    data.source_bindings["loaded_data_sha256"] = _loaded_data_sha256(data)
    return data


def _loaded_data_sha256(data: TrainingData) -> str:
    arrays = {}
    for name in ("X", "y", "episode_ids", "group_ids", "observation_group_ids", "geometry_group_ids"):
        value = np.ascontiguousarray(getattr(data.dataset, name))
        arrays[name] = {"dtype": value.dtype.str, "shape": list(value.shape),
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    folds = [{**{key: fold[key] for key in ("fold_id", "test_category", "validation_category")},
              "indices": {role: rows.tolist() for role, rows in fold["indices"].items()}} for fold in data.folds]
    return canonical_sha256({"arrays": arrays, "manifest": data.dataset.manifest,
                             "schedule": data.schedule, "adapted_folds": folds})


def _check_loaded_inputs(data: TrainingData) -> None:
    required = {"dataset_sha256", "dataset_manifest_sha256", "schedule_file_sha256", "folds_file_sha256",
                "binding_file_sha256", "input_files", "loaded_data_sha256"}
    if not required <= set(data.source_bindings):
        raise ValueError("training requires the strict file-backed adapter")
    if data.source_bindings["loaded_data_sha256"] != _loaded_data_sha256(data):
        raise ValueError("loaded features/labels/fold memberships changed after adapter validation")
    for name, expected in data.source_bindings["input_files"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError("bound dataset/schedule/folds/binding file changed before fitting")


def _source_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parents[1]
    names = ("prediction/recorded_training.py", "prediction/model.py", "prediction/recorded_dataset.py",
             "evaluation/prediction_recording.py", "evaluation/scenario_sampler.py")
    return {"src/grasp_failure_prediction/" + name: package / name for name in names}


def _training_runtime(spec: TrainingSpec) -> dict:
    version = None
    if spec.include_gru:
        try:
            import torch
            version = torch.__version__
        except ImportError:
            pass
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": version, "device": "cpu"}


def _protocol_contract(data: TrainingData, spec: TrainingSpec) -> dict:
    return {"specification": json.loads(json.dumps(asdict(spec))),
            "source_bindings": data.source_bindings,
            "source_files": {name: file_sha256(path) for name, path in _source_paths().items()},
            "input_contract": data.dataset.manifest["input_contract"], "runtime": _training_runtime(spec)}


def prepare_training_protocol(data: TrainingData, output_path: str | Path, *,
                              specification: TrainingSpec | None = None) -> dict:
    """Declare exact inputs/settings without fitting; commit this before train."""
    path, spec = Path(output_path), specification or TrainingSpec()
    if path.exists():
        raise FileExistsError("training protocol declaration must be a fresh file")
    _check_loaded_inputs(data)
    payload = {"schema_version": "hug_recorded_training_protocol_v1", **_protocol_contract(data, spec),
               "prepared_utc": datetime.now(timezone.utc).isoformat(),
               "requires_commit_before_fit": True,
               "scope": "predeclared final-task-outcome model comparison; no training during preparation"}
    # Do not create arbitrary directories or modify an earlier declaration.
    _write(path, payload)
    return payload


def _committed_protocol_receipt(protocol: Path) -> dict:
    """Require the declaration AND bound source bytes in the current Git HEAD."""
    try:
        root = Path(subprocess.check_output(["git", "-C", str(protocol.parent), "rev-parse", "--show-toplevel"],
                                            text=True, stderr=subprocess.PIPE).strip()).resolve()
        paths = [protocol.resolve(), *_source_paths().values()]
        relative = [str(path.resolve().relative_to(root)) for path in paths]
        subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", "--", *relative],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        changed = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *relative],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if changed.returncode != 0:
            raise ValueError("protocol or bound training source differs from committed HEAD")
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        return {"commit": head, "protocol_and_bound_sources_clean_at_HEAD": True}
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise ValueError("commit the prepared training protocol and all bound sources before fitting") from exc


def _validate_training_protocol(data: TrainingData, spec: TrainingSpec, protocol: Path) -> dict:
    _check_loaded_inputs(data)
    declaration = json.loads(protocol.read_text())
    if (declaration.get("schema_version") != "hug_recorded_training_protocol_v1"
            or declaration.get("requires_commit_before_fit") is not True):
        raise ValueError("require a structured predeclared training protocol")
    actual = _protocol_contract(data, spec)
    for field, expected in actual.items():
        if declaration.get(field) != expected:
            raise ValueError(f"frozen training protocol mismatch: {field}")
    return _committed_protocol_receipt(protocol)


def _features(X: np.ndarray, kind: str, contract: dict) -> np.ndarray:
    if kind == "full_history_logistic":
        return np.asarray(X, dtype=np.float64)
    if kind not in ("commands_only_logistic", "summary_logistic"):
        raise ValueError("unknown logistic feature representation")
    return summarize_prefix(X, history_steps=contract["history_steps"],
                            joint_count=len(contract["joint_names"]),
                            commands_only=kind == "commands_only_logistic")


def _log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    return outcome_metrics(y, probability)["log_loss"]


def _calibration(y: np.ndarray, probability: np.ndarray, bins: int) -> dict:
    membership = np.minimum((probability * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        selected = membership == i
        if selected.any():
            rows.append({"lower": i / bins, "upper": (i + 1) / bins, "n": int(selected.sum()),
                         "mean_p_success": float(probability[selected].mean()),
                         "observed_success_fraction": float(y[selected].mean())})
    return {"bins": rows, "expected_calibration_error": sum(
        row["n"] * abs(row["mean_p_success"] - row["observed_success_fraction"]) for row in rows) / len(y)}


def _save_normalizer(path: Path, normalizer: Normalizer) -> str:
    np.savez_compressed(path, mean=normalizer.mean, scale=normalizer.scale, active=normalizer.active)
    return file_sha256(path)


def _load_normalizer(path: Path) -> Normalizer:
    with np.load(path, allow_pickle=False) as value:
        if set(value.files) != {"mean", "scale", "active"}:
            raise ValueError("normalizer array allowlist mismatch")
        normalizer = Normalizer(value["mean"], value["scale"], value["active"])
    if (normalizer.mean.ndim != 1 or normalizer.scale.shape != normalizer.mean.shape
            or normalizer.active.shape != normalizer.mean.shape or normalizer.active.dtype != np.bool_
            or not np.isfinite(normalizer.mean).all() or not np.isfinite(normalizer.scale).all()
            or (normalizer.scale <= 0).any()):
        raise ValueError("invalid normalizer")
    return normalizer


def _gru_module(torch, input_size: int, hidden_size: int):
    class SmallGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
            self.head = torch.nn.Linear(hidden_size, 1)

        def forward(self, values):
            _, hidden = self.gru(values)
            return self.head(hidden[-1]).squeeze(-1)
    return SmallGRU().cpu()


def _sequence(normalizer: Normalizer, X: np.ndarray, steps: int, channels: int) -> np.ndarray:
    values = normalizer.transform(X.reshape(-1, channels)).reshape(len(X), steps, -1)
    # An all-constant training prefix has no active channel. A declared zero
    # channel permits a bias-only network, without fitting held-out variability.
    if values.shape[2] == 0:
        values = np.zeros((len(X), steps, 1))
    return values.astype(np.float32)


@contextmanager
def _deterministic_torch(torch, seed: int):
    previous = torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(previous)


def _fit_gru(X_train, y_train, X_validation, y_validation, *, contract, spec, seed):
    import torch
    steps, channels = contract["history_steps"], 3 * len(contract["joint_names"])
    normalizer = Normalizer.fit(X_train.reshape(-1, channels))
    train = torch.from_numpy(_sequence(normalizer, X_train, steps, channels))
    validation = torch.from_numpy(_sequence(normalizer, X_validation, steps, channels))
    truth = torch.tensor(y_train, dtype=torch.float32)
    candidates, models = [], []
    for hidden, learning_rate in spec.gru_candidates:
        with _deterministic_torch(torch, seed):
            model = _gru_module(torch, train.shape[2], hidden)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=spec.gru_weight_decay)
            model.train()
            for _ in range(spec.gru_epochs):
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train), truth)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite GRU training loss")
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                probability = torch.sigmoid(model(validation)).numpy().astype(np.float64)
            candidates.append({"hidden_size": hidden, "learning_rate": learning_rate,
                               "epochs": spec.gru_epochs, "validation_log_loss": _log_loss(y_validation, probability)})
            models.append(model)
    selected = min(range(len(candidates)), key=lambda i: (candidates[i]["validation_log_loss"], i))
    return models[selected], normalizer, {"candidates": candidates, "selected_candidate": selected,
                                          "selection_metric": "validation_log_loss", "torch_version": torch.__version__,
                                          "input_size": train.shape[2]}


def _gru_predict(model, normalizer, X, contract):
    import torch
    values = torch.from_numpy(_sequence(normalizer, X, contract["history_steps"], 3 * len(contract["joint_names"])))
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(values)).numpy().astype(np.float64)


def predict_saved_model(directory: str | Path, X: np.ndarray) -> np.ndarray:
    """Inference accepts only an allowlisted feature matrix; no label/ID files."""
    root = Path(directory)
    metadata = json.loads((root / "model.json").read_text())
    if metadata.get("schema_version") != "hug_recorded_predictor_v1" or metadata["kind"] not in MODEL_NAMES:
        raise ValueError("unknown saved predictor schema/kind")
    contract = metadata["input_contract"]
    prefix = PrefixSpec.model_validate({key: contract[key] for key in PrefixSpec.model_fields})
    if contract != _input_contract(prefix, contract["joint_names"]):
        raise ValueError("saved predictor feature contract violates exporter allowlist")
    if metadata.get("selection_sha256") != canonical_sha256(metadata["selection"]):
        raise ValueError("saved predictor selection hash mismatch")
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[1] != contract["features"] or not np.isfinite(values).all():
        raise ValueError("prediction feature matrix violates saved contract")
    if set(metadata["artifacts"]) != {"model.npz", "normalizer.npz"}:
        raise ValueError("saved model artifact allowlist mismatch")
    for filename, expected in metadata["artifacts"].items():
        if filename not in {"model.npz", "normalizer.npz"} or file_sha256(root / filename) != expected:
            raise ValueError("saved model artifact hash mismatch")
    kind = metadata["kind"]
    if kind == "training_prevalence":
        with np.load(root / "model.npz", allow_pickle=False) as saved:
            if set(saved.files) != {"p_success"}:
                raise ValueError("prevalence model allowlist mismatch")
            prevalence = float(saved["p_success"])
        if not np.isfinite(prevalence) or not 0 <= prevalence <= 1:
            raise ValueError("invalid prevalence model probability")
        _load_normalizer(root / "normalizer.npz")
        return np.full(len(values), prevalence)
    if kind != "small_gru":
        model = load_model(root / "model.npz")
        # The separately saved normalizer is a visible, bound training artifact.
        normalizer = _load_normalizer(root / "normalizer.npz")
        if any(not np.array_equal(getattr(model.normalizer, name), getattr(normalizer, name))
               for name in ("mean", "scale", "active")):
            raise ValueError("embedded and separate normalizers differ")
        return model.predict_success(_features(values, kind, contract))
    import torch
    selection = metadata["selection"]
    chosen = selection["candidates"][selection["selected_candidate"]]
    with _deterministic_torch(torch, metadata["seed"]):
        model = _gru_module(torch, selection["input_size"], chosen["hidden_size"])
        with np.load(root / "model.npz", allow_pickle=False) as state:
            model.load_state_dict({key: torch.from_numpy(state[key].copy()) for key in state.files}, strict=True)
    return _gru_predict(model, _load_normalizer(root / "normalizer.npz"), values, contract)


def train_comparison(data: TrainingData, output_dir: str | Path, *,
                     specification: TrainingSpec | None = None, protocol_path: str | Path) -> dict:
    """Explicit entry point; caller freezes protocol and qualifies collection first."""
    spec, output, protocol = specification or TrainingSpec(), Path(output_dir), Path(protocol_path)
    if output.exists():
        raise FileExistsError("comparison output must be a fresh directory")
    if not protocol.is_file():
        raise ValueError("a saved frozen protocol is required before fitting")
    frozen_receipt = _validate_training_protocol(data, spec, protocol)
    bundle, contract = data.dataset, data.dataset.manifest["input_contract"]
    X, y = bundle.X, bundle.y
    representations = {kind: _features(X, kind, contract) for kind in MODEL_NAMES[1:4]}
    report = {"schema_version": SCHEMA, "specification": asdict(spec),
              "source_bindings": data.source_bindings, "protocol_sha256": file_sha256(protocol),
              "training_source_sha256": file_sha256(Path(__file__)),
              "frozen_protocol_receipt": frozen_receipt,
              "input_contract": contract, "dataset_counts": bundle.manifest["counts"],
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "target": "final task outcome from causal close prefix; not physical slip onset",
              "decision_rule": "failure warning if 1-p_success >= 0.5",
              "seed_scope": "three training seeds; logistic/prevalence deterministic; not three independent datasets",
              "folds": []}
    output.mkdir(parents=True)
    for fold_index, fold in enumerate(data.folds):
        indices = fold["indices"]
        train, validation, test = (indices[role] for role in ("train", "validation", "test"))
        record = {key: fold[key] for key in ("fold_id", "test_category", "validation_category")}
        record["partitions"] = {role: {"episode_ids": bundle.episode_ids[index].tolist(), "n": len(index),
                                       "row_indices_sha256": canonical_sha256(index.tolist())}
                                for role, index in indices.items()}
        record["models"] = []
        report["folds"].append(record)
        reason = next((f"empty_{role}" for role, rows in indices.items() if not len(rows)), None)
        if reason is None:
            reason = next((f"single_class_{role}" for role, rows in (("train", train), ("validation", validation))
                           if len(np.unique(y[rows])) != 2), None)
        for seed in spec.seeds:
            for kind in MODEL_NAMES:
                item = {"name": kind, "seed": seed, "status": "skipped", "metrics": None}
                record["models"].append(item)
                if reason:
                    item["reason"] = reason
                    continue
                if kind == "small_gru" and not spec.include_gru:
                    item["reason"] = "optional_gru_disabled_before_fitting"
                    continue
                if kind == "small_gru":
                    try:
                        import torch  # optional dependency; never installed here.
                    except ImportError:
                        item["reason"] = "optional_torch_unavailable"
                        continue
                model_dir = output / f"fold_{fold_index:03d}" / kind / f"seed_{seed}"
                model_dir.mkdir(parents=True)
                try:
                    if kind == "training_prevalence":
                        prevalence = float(y[train].mean())
                        selection = {"p_success": prevalence, "selection_metric": "none; training prevalence",
                                     "candidates": []}
                        probability = np.full(len(test), prevalence)
                        np.savez_compressed(model_dir / "model.npz", p_success=np.array(prevalence))
                        # Empty normalizer explicitly means this model has no inputs.
                        normalizer = Normalizer(np.empty(0), np.empty(0), np.empty(0, dtype=bool))
                    elif kind != "small_gru":
                        features = representations[kind]
                        candidates, models = [], []
                        for regularization in spec.regularization_grid:
                            model = fit_logistic(features[train], y[train], regularization=regularization,
                                                 max_iterations=spec.max_logistic_iterations)
                            candidates.append({"regularization": regularization, "converged": model.converged,
                                               "iterations": model.iterations, "validation_log_loss":
                                               _log_loss(y[validation], model.predict_success(features[validation]))})
                            models.append(model)
                        eligible = [i for i, model in enumerate(models) if model.converged]
                        if not eligible:
                            raise RuntimeError("no logistic candidate converged")
                        selected = min(eligible, key=lambda i: (candidates[i]["validation_log_loss"],
                                                                -spec.regularization_grid[i]))
                        model = models[selected]
                        selection = {"candidates": candidates, "selected_candidate": selected,
                                     "selection_metric": "validation_log_loss", "seed_effect": "deterministic fit"}
                        # Test rows are first passed to the selected model here.
                        probability = model.predict_success(features[test])
                        normalizer = model.normalizer
                        save_model(model, model_dir / "model.npz")
                    else:
                        model, normalizer, selection = _fit_gru(X[train], y[train], X[validation], y[validation],
                                                               contract=contract, spec=spec, seed=seed)
                        probability = _gru_predict(model, normalizer, X[test], contract)
                        np.savez_compressed(model_dir / "model.npz", **{
                            key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
                    normalizer_hash = _save_normalizer(model_dir / "normalizer.npz", normalizer)
                    metadata = {"schema_version": "hug_recorded_predictor_v1", "kind": kind, "seed": seed,
                                "input_contract": contract, "selection": selection,
                                "selection_sha256": canonical_sha256(selection),
                                "training_episode_ids_sha256": canonical_sha256(bundle.episode_ids[train].tolist()),
                                "validation_episode_ids_sha256": canonical_sha256(bundle.episode_ids[validation].tolist()),
                                "artifacts": {"model.npz": file_sha256(model_dir / "model.npz"),
                                              "normalizer.npz": normalizer_hash}}
                    metadata_hash = _write(model_dir / "model.json", metadata)
                    readback = predict_saved_model(model_dir, X[test])
                    if not np.allclose(probability, readback, atol=1e-7 if kind == "small_gru" else 1e-12, rtol=0):
                        raise RuntimeError("saved predictor readback differs")
                    predictions = {"episode_ids": bundle.episode_ids[test].tolist(), "y_success": y[test].tolist(),
                                   "p_success": probability.tolist(), "warning_threshold": .5}
                    prediction_hash = _write(model_dir / "predictions.json", predictions)
                    metrics = outcome_metrics(y[test], probability)
                    single_class_test = len(np.unique(y[test])) != 2
                    if single_class_test:
                        metrics["failure_roc_auc"] = None
                        metrics["failure_average_precision"] = None
                    item.update({"status": "completed", "model_directory": str(model_dir.relative_to(output)),
                                 "model_metadata_sha256": metadata_hash, "selection_sha256": canonical_sha256(selection),
                                 "predictions_sha256": prediction_hash,
                                 "metrics": metrics,
                                 "calibration": _calibration(y[test], probability, spec.calibration_bins),
                                 "ranking_status": "skipped_single_class_test" if single_class_test else "measured"})
                except Exception as exc:
                    item.update({"status": "error", "reason": str(exc), "error_type": type(exc).__name__})
                    _write(model_dir / "error.json", {"error": str(exc), "error_type": type(exc).__name__})
    report["status"] = ("completed_with_model_errors" if any(m["status"] == "error" for f in report["folds"] for m in f["models"])
                        else "completed_with_skips" if any(m["status"] == "skipped" for f in report["folds"] for m in f["models"])
                        else "completed")
    if _validate_training_protocol(data, spec, protocol) != frozen_receipt:
        raise RuntimeError("training protocol/source commit changed during fitting")
    _write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-training-protocol", "train"):
        command = commands.add_parser(name)
        for argument in ("dataset", "schedule", "folds", "attempt-binding", "protocol"):
            command.add_argument("--" + argument, type=Path, required=True)
        command.add_argument("--without-gru", action="store_true")
        if name == "train":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_training_data(args.dataset, args.schedule, args.folds, args.attempt_binding)
    spec = TrainingSpec(include_gru=not args.without_gru)
    if args.command == "prepare-training-protocol":
        prepare_training_protocol(data, args.protocol, specification=spec)
        print(json.dumps({"status": "prepared; commit protocol and bound sources before fitting",
                          "protocol": str(args.protocol)}))
        return
    report = train_comparison(data, args.output, protocol_path=args.protocol, specification=spec)
    print(json.dumps({"status": report["status"], "folds": len(report["folds"])}))


if __name__ == "__main__":
    main()
