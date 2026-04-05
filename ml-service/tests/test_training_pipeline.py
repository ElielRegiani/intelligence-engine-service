from pathlib import Path

from infrastructure.config.settings import Settings
from training.pipelines.training_pipeline import TrainingPipeline


def _rows():
    data = []
    for i in range(50):
        data.append(
            {
                "price": 25.0 + (i % 5) * 0.2,
                "rsi": 45.0 + (i % 15),
                "sma": 24.0 + i * 0.05,
                "volume": 900_000.0 + i * 100,
                "timestamp": f"2026-04-{(i % 28) + 1:02d}",
            }
        )
    return {"symbol": "X", "data": data}


class _FakeFetcher:
    def get_market_history(self, symbol: str):
        return _rows()


def test_training_pipeline_saves_model(tmp_path: Path):
    models = tmp_path / "models"
    settings = Settings(models_dir=models, kafka_enabled=False, scheduler_enabled=False)
    pipeline = TrainingPipeline(settings=settings, data_fetcher=_FakeFetcher())
    result = pipeline.run(symbols=["PETR4"], algorithm="logistic_regression")
    assert result.model_filename.startswith("model_v")
    assert (models / "metadata.json").exists()
    assert result.accuracy >= 0.0
