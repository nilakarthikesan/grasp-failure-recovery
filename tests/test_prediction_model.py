import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("scipy")

from grasp_failure_prediction.prediction.model import (
    Normalizer, fit_logistic, load_model, outcome_metrics, save_model,
    select_model, summarize_prefix, train_baselines,
)


def test_prefix_order_and_command_ablation():
    history = np.arange(2 * 3 * 6, dtype=float).reshape(2, 3, 6)
    X = history.reshape(2, -1)
    actual = summarize_prefix(X, history_steps=3, joint_count=2)
    expected = np.concatenate([history[:, -1], history.mean(1), history.std(1),
                               history[:, -1] - history[:, 0]], axis=1)
    np.testing.assert_array_equal(actual, expected)
    commands = summarize_prefix(X, history_steps=3, joint_count=2, commands_only=True)
    np.testing.assert_array_equal(commands, expected.reshape(2, 4, 6)[:, :, 4:].reshape(2, -1))


def test_scaler_is_training_only_and_ignores_constant_columns():
    train = np.array([[1, 7], [3, 7]], dtype=float)
    normalizer = Normalizer.fit(train)
    np.testing.assert_array_equal(normalizer.mean, [2, 7])
    np.testing.assert_array_equal(normalizer.active, [True, False])
    np.testing.assert_array_equal(normalizer.transform([[5, 9000]]), [[3]])
    np.testing.assert_array_equal(normalizer.mean, [2, 7])


def test_probability_direction_saved_model_and_determinism(tmp_path):
    X = np.array([[-3, 2], [-2, 2], [-1, 2], [1, 2], [2, 2], [3, 2]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    model = fit_logistic(X, y, regularization=0.1)
    assert model.converged
    probability = model.predict_success(X)
    assert (probability[:3] < 0.5).all() and (probability[3:] > 0.5).all()
    path = tmp_path / "weights.npz"
    save_model(model, path)
    np.testing.assert_array_equal(load_model(path).predict_success(X), probability)
    np.testing.assert_array_equal(fit_logistic(X, y, regularization=0.1).predict_success(X), probability)


def test_metrics_use_failure_positive_and_handle_ties():
    y = np.array([0, 1, 0, 1])
    metrics = outcome_metrics(y, np.array([0.1, 0.9, 0.2, 0.8]))
    assert metrics["failure_roc_auc"] == 1
    assert metrics["failure_average_precision"] == 1
    assert metrics["failure_confusion"] == {
        "true_positive": 2, "false_positive": 0, "true_negative": 2, "false_negative": 0}
    tied = outcome_metrics(y, np.full(4, 0.5))
    assert tied["failure_roc_auc"] == 0.5
    assert tied["failure_average_precision"] == 0.5
    assert tied["balanced_accuracy"] == 0.5
    single = outcome_metrics(np.ones(3), np.full(3, 0.75))
    assert single["failure_roc_auc"] is None
    assert single["failure_average_precision"] is None


def test_known_rank_and_average_precision():
    # Failure scores are [0.1, 0.4, 0.35, 0.8], truth [0, 0, 1, 1].
    metrics = outcome_metrics(np.array([1, 1, 0, 0]), 1 - np.array([0.1, 0.4, 0.35, 0.8]))
    assert metrics["failure_roc_auc"] == 0.75
    assert metrics["failure_average_precision"] == pytest.approx(5 / 6)


def test_selection_uses_validation_only():
    X = np.array([[-2], [-1], [1], [2]], dtype=float)
    y = np.array([0, 0, 1, 1])
    selected, candidates = select_model(X, y, X, y, grid=(0.01, 1.0))
    assert selected.regularization == 0.01
    assert len(candidates) == 2
    np.testing.assert_array_equal(selected.normalizer.mean, X.mean(0))


@pytest.mark.parametrize("X", [np.ones((2, 1441)), np.full((2, 1440), np.nan)])
def test_invalid_prefix_rejected(X):
    with pytest.raises(ValueError):
        summarize_prefix(X)


def test_bad_labels_and_probabilities_rejected():
    with pytest.raises(ValueError):
        fit_logistic(np.ones((2, 2)), np.array([1, 1]), regularization=0.1)
    with pytest.raises(ValueError):
        outcome_metrics(np.array([0, 2]), np.array([0.1, 0.9]))
    with pytest.raises(ValueError):
        outcome_metrics(np.array([0, 1]), np.array([-0.1, 0.9]))


def test_malformed_saved_weights_rejected(tmp_path):
    model = fit_logistic(np.array([[-1.0], [1.0]]), np.array([0, 1]), regularization=0.1)
    path = tmp_path / "model.npz"
    save_model(model, path)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["weights"] = values["weights"][:, None]
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="invalid saved model"):
        load_model(path)


def test_test_mutation_cannot_change_fit_selection_or_scaler(tmp_path, monkeypatch):
    import grasp_failure_prediction.prediction.dataset as dataset

    rng = np.random.default_rng(52)
    X = rng.normal(size=(14, 1440)).astype(np.float32)
    y = np.tile([0, 1], 7).astype(np.uint8)
    X[y == 1] += 0.25
    ids = np.array([f"episode_{i:06d}" for i in range(14)])
    contract = {"prediction_time_s": 2.96, "history_steps": 20,
                "joint_names": [f"joint{i}" for i in range(24)],
                "fields": ["hand_qpos", "hand_qvel", "commanded_target_rad"]}
    partitions = {role: {"episode_ids": ids[index].tolist()}
                  for role, index in {"train": slice(0, 6), "validation": slice(6, 10),
                                      "test": slice(10, 14)}.items()}
    bundle = SimpleNamespace(X=X, y=y, episode_ids=ids,
                             manifest={"input_contract": contract}, splits={"partitions": partitions})
    directory = tmp_path / "dataset"
    directory.mkdir()
    for name in ("dataset.npz", "manifest.json", "splits.json"):
        (directory / name).write_bytes(b"isolated training integration fixture")
    monkeypatch.setattr(dataset, "load_dataset", lambda path: bundle)
    first = train_baselines(directory, tmp_path / "first")
    altered = X.copy()
    altered[10:] *= 1000
    changed_labels = y.copy()
    changed_labels[10:] = 1 - changed_labels[10:]
    bundle.X, bundle.y = altered, changed_labels
    second = train_baselines(directory, tmp_path / "second")
    for name in ("proprioception_and_commands", "commands_only"):
        assert first["models"][name]["regularization"] == second["models"][name]["regularization"]
        with np.load(tmp_path / "first" / f"{name}.npz") as one, \
                np.load(tmp_path / "second" / f"{name}.npz") as two:
            for key in one.files:
                np.testing.assert_array_equal(one[key], two[key])


def test_nonconverged_candidate_cannot_win(monkeypatch):
    import grasp_failure_prediction.prediction.model as module

    def fit(X, y, *, regularization):
        probability = np.array([0.0, 1.0]) if regularization == 0.01 else np.array([0.5, 0.5])
        return SimpleNamespace(regularization=regularization, iterations=400,
                               converged=regularization != 0.01,
                               predict_success=lambda values: probability)
    monkeypatch.setattr(module, "fit_logistic", fit)
    X, y = np.ones((2, 1)), np.array([0, 1])
    model, candidates = select_model(X, y, X, y, grid=(0.01, 1.0))
    assert model.regularization == 1.0 and not candidates[0]["converged"]
    with pytest.raises(RuntimeError, match="no regularization candidate converged"):
        select_model(X, y, X, y, grid=(0.01,))
