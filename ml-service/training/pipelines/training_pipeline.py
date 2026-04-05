"""End-to-end training: fetch, clean, features, train, evaluate, version, persist."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from infrastructure.config.settings import Settings, get_settings
from training.evaluation.evaluator import Evaluator
from training.features.feature_engineering import FEATURE_COLUMNS, build_training_matrix
from training.trainers.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model_filename: str
    accuracy: float
    precision: float
    recall: float
    rolled_back: bool
    metadata_path: Path
    duration_seconds: float
    report: Dict[str, Any]


class TrainingPipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        data_fetcher: Optional[Any] = None,
    ):
        self.settings = settings or get_settings()
        self._data_fetcher = data_fetcher

    def _metadata_path(self) -> Path:
        return self.settings.models_dir / self.settings.metadata_filename

    def _load_metadata(self) -> Dict[str, Any]:
        path = self._metadata_path()
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _next_version_filename(self) -> str:
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        pattern = re.compile(r"^model_v(\d+)\.pkl$")
        max_v = 0
        for p in self.settings.models_dir.glob("model_v*.pkl"):
            m = pattern.match(p.name)
            if m:
                max_v = max(max_v, int(m.group(1)))
        return f"model_v{max_v + 1}.pkl"

    def _save_joblib(self, model: object, feature_columns: List[str], path: Path) -> None:
        joblib.dump({"model": model, "feature_columns": feature_columns}, path)

    def run(
        self,
        symbols: Optional[List[str]] = None,
        algorithm: str = "random_forest",
    ) -> TrainingResult:
        t0 = time.perf_counter()
        symbols = symbols or self.settings.train_symbol_list
        fetcher = self._data_fetcher
        if fetcher is None:
            from app.services.data_service_client import DataServiceClient

            fetcher = DataServiceClient(self.settings)

        frames_x: List[pd.DataFrame] = []
        frames_y: List[pd.Series] = []

        for sym in symbols:
            history = fetcher.get_market_history(sym)
            rows = history.get("data") or []
            X_part, y_part = build_training_matrix(rows)
            if len(X_part) > 0:
                frames_x.append(X_part)
                frames_y.append(y_part)

        if not frames_x:
            raise RuntimeError("No training rows after fetch/feature build.")

        X = pd.concat(frames_x, axis=0, ignore_index=True)
        y = pd.concat(frames_y, axis=0, ignore_index=True)

        trainer = ModelTrainer(algorithm=algorithm)
        artifacts = trainer.train(X, y, FEATURE_COLUMNS)
        evaluator = Evaluator()
        report = evaluator.evaluate_holdout(
            artifacts.model,
            artifacts.X_test,
            artifacts.y_test,
            artifacts.X_train,
            artifacts.y_train,
        )

        meta_prev = self._load_metadata()
        prev_acc = float(meta_prev.get("accuracy") or 0.0)
        prev_model = meta_prev.get("current_model")

        new_name = self._next_version_filename()
        new_path = self.settings.models_dir / new_name
        self._save_joblib(artifacts.model, artifacts.feature_columns, new_path)

        rolled_back = False
        reject = bool(prev_model) and report.accuracy < prev_acc - 0.02

        if reject:
            rolled_back = True
            try:
                new_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove rejected model file %s", new_path)
            payload = {
                **meta_prev,
                "last_rejected_model": new_name,
                "last_rejection_reason": "accuracy_below_threshold_vs_previous",
                "last_candidate_accuracy": round(report.accuracy, 4),
            }
            with open(self._metadata_path(), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            duration = time.perf_counter() - t0
            logger.info(
                "training_finished",
                extra={
                    "duration_seconds": round(duration, 3),
                    "accuracy": prev_acc,
                    "candidate_accuracy": report.accuracy,
                    "model_file": prev_model,
                    "rolled_back": True,
                },
            )
            return TrainingResult(
                model_filename=str(prev_model),
                accuracy=float(prev_acc),
                precision=float(meta_prev.get("precision") or 0.0),
                recall=float(meta_prev.get("recall") or 0.0),
                rolled_back=True,
                metadata_path=self._metadata_path(),
                duration_seconds=duration,
                report=report.to_dict(),
            )

        payload = {
            "current_model": new_name,
            "accuracy": round(report.accuracy, 4),
            "precision": round(report.precision, 4),
            "recall": round(report.recall, 4),
            "created_at": date.today().isoformat(),
            "previous_model": prev_model,
            "train_accuracy": report.train_accuracy,
            "cv_accuracy_mean": report.cv_accuracy_mean,
            "overfitting_warning": report.overfitting_warning,
        }
        with open(self._metadata_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        duration = time.perf_counter() - t0
        logger.info(
            "training_finished",
            extra={
                "duration_seconds": round(duration, 3),
                "accuracy": report.accuracy,
                "model_file": new_name,
                "rolled_back": False,
            },
        )

        return TrainingResult(
            model_filename=new_name,
            accuracy=report.accuracy,
            precision=report.precision,
            recall=report.recall,
            rolled_back=rolled_back,
            metadata_path=self._metadata_path(),
            duration_seconds=duration,
            report=report.to_dict(),
        )


def train_on_synthetic_data(models_dir: Path) -> None:
    """Bootstrap a minimal model when no Data Service is available (dev/tests)."""
    rng = np.random.default_rng(42)
    n = 80
    price = 30 + np.cumsum(rng.normal(0, 0.5, size=n))
    rows = []
    for i in range(n):
        rows.append(
            {
                "price": float(price[i]),
                "rsi": float(40 + rng.random() * 40),
                "sma": float(np.mean(price[max(0, i - 4) : i + 1])),
                "volume": float(1e6 + rng.random() * 5e5),
                "timestamp": f"2026-01-{1 + (i % 28):02d}",
            }
        )
    X, y = build_training_matrix(rows)
    trainer = ModelTrainer()
    art = trainer.train(X, y, FEATURE_COLUMNS)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "model_v1.pkl"
    joblib.dump({"model": art.model, "feature_columns": art.feature_columns}, path)
    ev = Evaluator()
    rep = ev.evaluate_holdout(art.model, art.X_test, art.y_test, art.X_train, art.y_train)
    meta = {
        "current_model": "model_v1.pkl",
        "accuracy": round(rep.accuracy, 4),
        "precision": round(rep.precision, 4),
        "recall": round(rep.recall, 4),
        "created_at": datetime.now(timezone.utc).date().isoformat(),
        "bootstrap": True,
    }
    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
