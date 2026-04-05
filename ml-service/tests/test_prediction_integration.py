from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.prediction_service import PredictionService
from infrastructure.config.settings import Settings


def _history():
    data = []
    for i in range(35):
        data.append(
            {
                "price": 28.0 + i * 0.05,
                "rsi": 48.0 + (i % 12),
                "sma": 27.5 + i * 0.04,
                "volume": 1_100_000.0,
                "timestamp": f"2026-04-{(i % 28) + 1:02d}",
            }
        )
    return {"symbol": "PETR4", "data": data}


@patch("app.services.prediction_service.DataServiceClient")
def test_predict_with_mock_data_service(mock_client_cls, tmp_path: Path):
    models = tmp_path / "models"
    settings = Settings(
        models_dir=models,
        kafka_enabled=False,
        scheduler_enabled=False,
    )
    from training.pipelines.training_pipeline import train_on_synthetic_data

    train_on_synthetic_data(models)

    instance = MagicMock()
    instance.get_market_history.return_value = _history()
    mock_client_cls.return_value = instance

    svc = PredictionService(settings=settings)
    out = svc.predict("PETR4", publish_event=False)
    assert out["symbol"] == "PETR4"
    assert out["prediction"] in ("UP", "DOWN")
    assert 0 <= out["confidence"] <= 1
    assert "rsi" in out["features_used"]
