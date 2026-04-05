"""Application configuration (env + defaults)."""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_service_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the Data Service (REST pull).",
    )
    data_service_timeout_seconds: float = 30.0
    data_service_max_retries: int = 3
    data_service_retry_backoff_seconds: float = 1.0

    models_dir: Path = Field(default_factory=_default_models_dir)
    metadata_filename: str = "metadata.json"

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_predictions: str = "ml-prediction-generated"
    kafka_enabled: bool = True

    default_train_symbols: str = "PETR4,VALE3"
    batch_inference_symbols: str = "PETR4,VALE3"

    train_cron_hour: int = 18
    train_cron_minute: int = 20
    batch_cron_hour: int = 18
    batch_cron_minute: int = 30

    scheduler_enabled: bool = True
    scheduler_timezone: str = "America/Sao_Paulo"

    log_level: str = "INFO"

    synthetic_fallback: bool = Field(
        default=False,
        description="If true, return synthetic history when Data Service is down and cache is empty (dev only).",
    )

    @property
    def train_symbol_list(self) -> List[str]:
        return [s.strip() for s in self.default_train_symbols.split(",") if s.strip()]

    @property
    def batch_symbol_list(self) -> List[str]:
        return [s.strip() for s in self.batch_inference_symbols.split(",") if s.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
