import numpy as np

from sklearn.linear_model import LogisticRegression

from ml.dataset_builder import _load_barrier_config
from ml.model_evaluator import ModelEvaluator


def test_model_metrics_include_n_samples():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(X, y)

    metrics = ModelEvaluator().evaluate(model, X, y, model_name="logreg")

    assert metrics.n_samples == len(y)
    assert metrics.to_dict()["n_samples"] == len(y)


def test_barrier_config_loads_without_package_shadowing():
    cfg = _load_barrier_config("EURUSD")

    assert isinstance(cfg, dict)
    assert cfg["_default"]["atr_multiplier"] == 1.5
    assert cfg["EURAUD"]["atr_multiplier"] == 2.5
