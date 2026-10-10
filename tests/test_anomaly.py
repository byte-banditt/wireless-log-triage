import pytest

from wireless_log_triage.modeling import RollingDetector


def test_rolling_detector_uses_past_of_same_protocol():
    detector = RollingDetector(window=4, threshold=3.0)
    baseline = {"protocol": "wifi", "rssi": -55, "latency_ms": 20, "throughput_mbps": 180}
    for index in range(4):
        detector.observe({**baseline, "latency_ms": 20 + index})
    assert detector.observe({**baseline, "latency_ms": 100})["anomalous"] is True
    assert detector.observe({"protocol": "bluetooth", "rssi": -75, "latency_ms": 100, "throughput_mbps": 2})["anomalous"] is False


def test_rolling_detector_rejects_out_of_order_log():
    detector = RollingDetector(window=4)
    row = {"protocol": "wifi", "rssi": -55, "latency_ms": 20, "throughput_mbps": 180}
    detector.observe({**row, "timestamp": "2025-01-01T00:01:00+00:00"})
    with pytest.raises(ValueError, match="timestamp order"):
        detector.observe({**row, "timestamp": "2025-01-01T00:00:00+00:00"})
