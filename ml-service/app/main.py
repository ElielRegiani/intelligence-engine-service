"""FastAPI entrypoint: serving + scheduler; training stays in training layer."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pythonjsonlogger import jsonlogger

from app.api.routes import router
from infrastructure.config.settings import get_settings
from infrastructure.kafka.producer import PredictionEventPublisher
from infrastructure.scheduler.jobs import setup_scheduler, shutdown_scheduler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_logging() -> None:
    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    fmt = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    meta_path = settings.models_dir / settings.metadata_filename
    needs_bootstrap = True
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("current_model"):
                needs_bootstrap = False
        except json.JSONDecodeError:
            pass
    if needs_bootstrap:
        from training.pipelines.training_pipeline import train_on_synthetic_data

        train_on_synthetic_data(settings.models_dir)

    setup_scheduler(settings)
    yield
    shutdown_scheduler()
    PredictionEventPublisher(settings).close()


app = FastAPI(
    title="ML Service (Intelligence Engine)",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/")
def root() -> dict:
    return {"service": "ml-service", "docs": "/docs"}
