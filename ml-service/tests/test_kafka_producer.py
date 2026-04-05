from unittest.mock import MagicMock, patch

from infrastructure.config.settings import Settings
from infrastructure.kafka.producer import PredictionEventPublisher


@patch("infrastructure.kafka.producer.KafkaProducer")
def test_publish_sends_key_and_payload(mock_kp):
    settings = Settings(kafka_enabled=True, kafka_bootstrap_servers="localhost:9092")
    mock_kp.return_value.send.return_value.get = MagicMock(return_value=None)
    pub = PredictionEventPublisher(settings)
    pub.publish("PETR4", "UP", 0.8, "2026-04-01T18:30:00")
    mock_kp.return_value.send.assert_called_once()
    call_kw = mock_kp.return_value.send.call_args
    assert call_kw[1]["key"] == "PETR4"
