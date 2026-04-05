from unittest.mock import MagicMock, patch

from app.services.data_service_client import DataServiceClient
from infrastructure.config.settings import Settings


def _history_payload():
    data = []
    for i in range(25):
        data.append(
            {
                "price": 30.0 + i * 0.1,
                "rsi": 50 + (i % 10),
                "sma": 30.0 + i * 0.05,
                "volume": 1_000_000.0,
                "timestamp": f"2026-03-{(i % 28) + 1:02d}",
            }
        )
    return {"symbol": "PETR4", "data": data}


@patch("app.services.data_service_client.httpx.Client")
def test_data_service_fetch_and_cache(mock_client_cls):
    settings = Settings(
        data_service_base_url="http://test.local",
        data_service_max_retries=2,
        data_service_retry_backoff_seconds=0,
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = _history_payload()
    mock_resp.raise_for_status = MagicMock()
    instance = MagicMock()
    instance.get.return_value = mock_resp
    mock_client_cls.return_value.__enter__.return_value = instance

    c = DataServiceClient(settings)
    a = c.get_market_history("PETR4")
    b = c.get_market_history("PETR4")
    assert a["symbol"] == "PETR4"
    assert a == b
    assert instance.get.call_count == 1
