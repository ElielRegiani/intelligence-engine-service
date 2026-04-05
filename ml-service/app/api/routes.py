"""HTTP routes (inference only; training triggered via scheduler or admin endpoint)."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.schemas.prediction_schema import HealthResponse, PredictRequest, PredictResponse
from app.services.prediction_service import PredictionService
from infrastructure.config.settings import get_settings
from training.pipelines.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    from app.models.model_loader import ModelLoader

    loader = ModelLoader(settings)
    meta = loader.read_metadata()
    name = meta.get("current_model")
    return HealthResponse(status="ok", current_model=name)


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    try:
        svc = PredictionService()
        out = svc.predict(body.symbol.strip(), publish_event=True)
        return PredictResponse(**out)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/admin/train", response_model=Dict[str, Any])
def admin_train() -> Dict[str, Any]:
    """Manual training trigger (same pipeline as scheduled job)."""
    try:
        pipeline = TrainingPipeline()
        result = pipeline.run()
        return {
            "model_filename": result.model_filename,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "rolled_back": result.rolled_back,
            "duration_seconds": result.duration_seconds,
            "report": result.report,
        }
    except Exception as e:
        logger.exception("admin_train_failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
