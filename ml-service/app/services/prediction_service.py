"""Serving layer: load model, pull data, features, predict (no training in controller)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.models.model_loader import ModelLoader
from app.services.data_service_client import DataServiceClient
from infrastructure.config.settings import Settings, get_settings
from infrastructure.kafka.producer import PredictionEventPublisher
from training.features.feature_engineering import latest_features_from_history

logger = logging.getLogger(__name__)

LABEL_UP = 1
LABEL_DOWN = 0


class PredictionService:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        loader: Optional[ModelLoader] = None,
        data_client: Optional[DataServiceClient] = None,
        publisher: Optional[PredictionEventPublisher] = None,
    ):
        self.settings = settings or get_settings()
        self.loader = loader or ModelLoader(self.settings)
        self.data_client = data_client or DataServiceClient(self.settings)
        self.publisher = publisher or PredictionEventPublisher(self.settings)

    def _vectorize(
        self, features: Dict[str, float], feature_columns: list
    ) -> np.ndarray:
        return np.array([[float(features[c]) for c in feature_columns]], dtype=float)

    def _confidence(self, model, X: np.ndarray) -> Tuple[str, float]:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            pred = int(model.predict(X)[0])
            conf = float(max(proba))
        else:
            pred = int(model.predict(X)[0])
            conf = 1.0
        label = "UP" if pred == LABEL_UP else "DOWN"
        return label, conf

    def predict(self, symbol: str, publish_event: bool = True) -> Dict[str, Any]:
        t0 = time.perf_counter()
        loaded = self.loader.load_current()
        history = self.data_client.get_market_history(symbol)
        rows = history.get("data") or []
        feats = latest_features_from_history(rows)
        if feats is None:
            raise ValueError("Insufficient market data to build features.")

        X = self._vectorize(feats, loaded.feature_columns)
        pred_label, confidence = self._confidence(loaded.model, X)

        features_used = {
            "rsi": feats["rsi"],
            "sma": feats["sma"],
            "volume": feats["volume"],
        }

        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "inference_complete",
            extra={
                "symbol": symbol,
                "prediction": pred_label,
                "confidence": round(confidence, 4),
                "inference_time_ms": round(duration_ms, 2),
            },
        )

        if publish_event and self.publisher.is_enabled():
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            self.publisher.publish(
                symbol=symbol.upper(),
                prediction=pred_label,
                confidence=confidence,
                timestamp_iso=ts,
            )

        return {
            "symbol": symbol.upper(),
            "prediction": pred_label,
            "confidence": confidence,
            "features_used": features_used,
        }
