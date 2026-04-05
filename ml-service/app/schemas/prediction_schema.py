from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    symbol: str = Field(..., min_length=1, description="Ticker symbol, e.g. PETR4")


class PredictResponse(BaseModel):
    symbol: str
    prediction: str
    confidence: float
    features_used: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    current_model: Optional[str] = None
