"""Kafka producer for ml-prediction-generated (key=symbol)."""

from __future__ import annotations

import json
import logging
from typing import Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError

from infrastructure.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class PredictionEventPublisher:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._producer: Optional[KafkaProducer] = None

    def is_enabled(self) -> bool:
        return bool(self.settings.kafka_enabled)

    def _ensure(self) -> Optional[KafkaProducer]:
        if not self.is_enabled():
            return None
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=self.settings.kafka_bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
                retries=3,
            )
        return self._producer

    def publish(
        self,
        symbol: str,
        prediction: str,
        confidence: float,
        timestamp_iso: str,
    ) -> None:
        producer = self._ensure()
        if producer is None:
            return
        payload = {
            "symbol": symbol,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "timestamp": timestamp_iso,
        }
        try:
            future = producer.send(
                self.settings.kafka_topic_predictions,
                key=symbol,
                value=payload,
            )
            future.get(timeout=10)
        except KafkaError as e:
            logger.error("kafka_publish_failed", extra={"error": str(e), "symbol": symbol})

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush()
            self._producer.close()
            self._producer = None
