import numpy as np

from training.features.feature_engineering import (
    FEATURE_COLUMNS,
    build_training_matrix,
    latest_features_from_history,
)


def _sample_rows(n: int = 30):
    rng = np.random.default_rng(0)
    price = 10 + np.cumsum(rng.normal(0, 0.2, size=n))
    rows = []
    for i in range(n):
        rows.append(
            {
                "price": float(price[i]),
                "rsi": float(40 + rng.random() * 30),
                "sma": float(np.mean(price[max(0, i - 4) : i + 1])),
                "volume": float(1e6),
                "timestamp": f"2026-01-{(i % 28) + 1:02d}",
            }
        )
    return rows


def test_build_training_matrix_shape_and_labels():
    rows = _sample_rows(40)
    X, y = build_training_matrix(rows)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)
    assert set(y.unique().tolist()).issubset({0, 1})


def test_latest_features_keys():
    rows = _sample_rows(20)
    f = latest_features_from_history(rows)
    assert f is not None
    for k in ("price", "price_change", "rsi", "sma", "volume", "volatility"):
        assert k in f
