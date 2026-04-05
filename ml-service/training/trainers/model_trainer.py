"""Train sklearn classifier; feature column order is fixed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass
class TrainArtifacts:
    model: object
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_columns: List[str]


class ModelTrainer:
    def __init__(
        self,
        algorithm: str = "random_forest",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.algorithm = algorithm
        self.test_size = test_size
        self.random_state = random_state

    def _build_estimator(self):
        if self.algorithm == "logistic_regression":
            return LogisticRegression(max_iter=2000, random_state=self.random_state)
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_columns: List[str],
    ) -> TrainArtifacts:
        X_arr = X[feature_columns].values.astype(float)
        y_arr = y.values.astype(int)

        if len(np.unique(y_arr)) < 2:
            raise ValueError("Need both classes in training data for binary classification.")

        X_train, X_test, y_train, y_test = train_test_split(
            X_arr,
            y_arr,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_arr,
        )

        model = self._build_estimator()
        model.fit(X_train, y_train)

        return TrainArtifacts(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_columns=list(feature_columns),
        )
