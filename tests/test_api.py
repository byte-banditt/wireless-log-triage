from fastapi.testclient import TestClient

from wireless_log_triage.api import app, get_service


class FakeService:
    def ingest(self, batch_id, logs):
        return [{**row, "batch_id": batch_id} for row in logs]

    def clusters(self):
        return [{"cluster": 1, "failure_pattern": "handshake_timeout", "count": 1, "test_ids": ["T1"]}]

    def anomalies(self):
        return []

    def triage(self, batch_id):
        return {"summary": batch_id, "citations": ["T1"]}


def test_endpoints_route_batch_and_results():
    app.dependency_overrides[get_service] = lambda: FakeService()
    try:
        client = TestClient(app)
        row = {"timestamp": "2025-01-01T00:00:00+00:00", "test_id": "T1", "protocol": "wifi", "component": "MAC", "status": "timeout", "error_code": "E_TIMEOUT", "message": "handshake timed out", "rssi": -55, "latency_ms": 20, "throughput_mbps": 180, "retry_count": 2}
        assert client.post("/ingest", json={"batch_id": "B1", "logs": [row]}).json()["logs"][0]["batch_id"] == "B1"
        assert client.get("/clusters").json()["clusters"][0]["count"] == 1
        assert client.get("/triage/B1").json()["citations"] == ["T1"]
        assert client.get("/anomalies").json()["anomalies"] == []
    finally:
        app.dependency_overrides.clear()
