import numpy as np
from sklearn.ensemble import RandomForestClassifier

from training.evaluation.evaluator import Evaluator
from training.features.feature_engineering import build_training_matrix


def test_evaluator_holdout_and_cv():
    rng = np.random.default_rng(1)
    n = 60
    price = 15 + np.cumsum(rng.normal(0, 0.15, size=n))
    rows = []
    for i in range(n):
        rows.append(
            {
                "price": float(price[i]),
                "rsi": float(45 + rng.random() * 20),
                "sma": float(np.mean(price[max(0, i - 4) : i + 1])),
                "volume": float(1e6),
                "timestamp": f"2026-02-{(i % 28) + 1:02d}",
            }
        )
    X, y = build_training_matrix(rows)
    assert len(X) > 10
    m = RandomForestClassifier(n_estimators=20, random_state=0)
    m.fit(X.values, y.values)
    X_train = X.values[:40]
    y_train = y.values[:40]
    X_test = X.values[40:]
    y_test = y.values[40:]
    ev = Evaluator()
    rep = ev.evaluate_holdout(m, X_test, y_test, X_train, y_train, cv_folds=3)
    assert 0 <= rep.accuracy <= 1
    assert rep.cv_accuracy_mean is not None
