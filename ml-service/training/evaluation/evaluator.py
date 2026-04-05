"""Metrics, cross-validation, and simple overfitting signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score


@dataclass
class EvaluationReport:
    accuracy: float
    precision: float
    recall: float
    train_accuracy: Optional[float] = None
    cv_accuracy_mean: Optional[float] = None
    cv_accuracy_std: Optional[float] = None
    overfitting_warning: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "train_accuracy": self.train_accuracy,
            "cv_accuracy_mean": self.cv_accuracy_mean,
            "cv_accuracy_std": self.cv_accuracy_std,
            "overfitting_warning": self.overfitting_warning,
        }


class Evaluator:
    def __init__(self, positive_label: int = 1):
        self.positive_label = positive_label

    def evaluate_holdout(
        self,
        model: ClassifierMixin,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        cv_folds: int = 5,
    ) -> EvaluationReport:
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(
            precision_score(
                y_test,
                y_pred,
                pos_label=self.positive_label,
                zero_division=0,
            )
        )
        rec = float(
            recall_score(
                y_test,
                y_pred,
                pos_label=self.positive_label,
                zero_division=0,
            )
        )

        train_acc: Optional[float] = None
        overfit = False
        if X_train is not None and y_train is not None:
            train_acc = float(accuracy_score(y_train, model.predict(X_train)))
            if train_acc - acc > 0.15:
                overfit = True

        cv_mean: Optional[float] = None
        cv_std: Optional[float] = None
        if X_train is not None and y_train is not None and len(y_train) >= cv_folds:
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="accuracy",
            )
            cv_mean = float(np.mean(scores))
            cv_std = float(np.std(scores))

        return EvaluationReport(
            accuracy=acc,
            precision=prec,
            recall=rec,
            train_accuracy=train_acc,
            cv_accuracy_mean=cv_mean,
            cv_accuracy_std=cv_std,
            overfitting_warning=overfit,
        )
