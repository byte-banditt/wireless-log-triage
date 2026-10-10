import json
from unittest.mock import Mock, patch

from wireless_log_triage.triage import grounded_triage


def _row(test_id):
    return {"timestamp": "2025-01-01T00:00:00+00:00", "test_id": test_id, "protocol": "wifi", "component": "MAC", "status": "timeout", "error_code": "E_TIMEOUT", "message": "handshake timed out", "cluster": 2, "rssi": -55, "latency_ms": 20, "throughput_mbps": 180, "retry_count": 2}


def test_triage_rejects_invented_citation():
    response = Mock()
    response.json.return_value = {"response": json.dumps({"summary": "handshake issue", "likely_root_cause": "handshake timeout", "affected_component": "MAC", "citations": ["INVENTED"]})}
    available = Mock()
    available.json.return_value = {"models": [{"name": "qwen3:8b"}]}
    with patch("wireless_log_triage.triage.requests.get", return_value=available), patch("wireless_log_triage.triage.requests.post", return_value=response):
        result = grounded_triage([_row("T1")])
    assert result["likely_root_cause"] == "insufficient evidence"
    assert result["citations"] == []


def test_triage_returns_supported_citation():
    response = Mock()
    response.json.return_value = {"response": json.dumps({"summary": "T1 timed out", "likely_root_cause": "handshake timeout", "affected_component": "MAC", "citations": ["T1"]})}
    available = Mock()
    available.json.return_value = {"models": [{"name": "qwen3:8b"}]}
    with patch("wireless_log_triage.triage.requests.get", return_value=available), patch("wireless_log_triage.triage.requests.post", return_value=response):
        result = grounded_triage([_row("T1")])
    assert result["citations"] == ["T1"]
    assert result["affected_component"] == "MAC"
