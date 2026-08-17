from __future__ import annotations

from pathlib import Path

from ml.data_bootstrap import bootstrap_feature_store_if_needed
from ml.feature_store import FeatureStore
from ml.model_store import ModelStore


def test_bootstrap_feature_store_if_needed_creates_rows(tmp_path: Path):
    store = FeatureStore(db_path=tmp_path / "ml_features_test.db")

    result = bootstrap_feature_store_if_needed(
        pair="EURUSD",
        timeframe="15m",
        min_samples=10,
        rows_per_pair=20,
        store=store,
    )

    assert result["bootstrapped"] is True
    assert result["rows_after"] >= 10

    rows = store.load_training_data(pair="EURUSD", timeframe="15m", min_samples=0)
    assert not rows.empty
    assert len(rows) >= 10


def test_model_store_normalizes_non_string_timeframe(tmp_path: Path):
    store = ModelStore(base_dir=tmp_path / "ml_models")
    model = {"ok": True}

    version = store.save_model(
        model=model,
        pair="EURUSD",
        timeframe=object(),
        model_type="lstm",
        metrics={"accuracy": 0.7},
        feature_names=["x1", "x2"],
    )

    assert version.startswith("v")
    assert any((tmp_path / "ml_models").rglob("lstm_v1.pkl"))
