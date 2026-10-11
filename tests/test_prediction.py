from copy import deepcopy

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from wireless_log_triage.api import app, get_failure_service
from wireless_log_triage.dataset import generate
from wireless_log_triage.prediction import (
    FailureRiskService, feature_row, fit_predictor, labels, probabilities,
    score_predictions, training_features, validate_split,
)


def test_history_excludes_own_labels_future_rows_and_simultaneous_records():
    train, _ = generate(train_count=80, eval_count=40)
    # Same pair and timestamp: neither record can see the other's outcome.
    train[1].update(protocol=train[0]["protocol"], component=train[0]["component"], timestamp=train[0]["timestamp"])
    features, rates = training_features(train)
    changed = deepcopy(train)
    changed[10]["status"] = "pass" if changed[10]["status"] != "pass" else "fail"
    changed_features, _ = training_features(changed)
    assert features[:11] == changed_features[:11]
    assert features[0]["historical_failure_rate"] == features[1]["historical_failure_rate"] == 0.5
    assert rates.total == len(train)
    assert rates.failures == int(labels(train).sum())


def test_eval_labels_and_outcome_fields_cannot_change_predictions():
    train, evaluation = generate(train_count=80, eval_count=40)
    model = fit_predictor(train)
    predicted, baseline = probabilities(model, evaluation)
    changed = deepcopy(evaluation)
    for row in changed:
        row.update(status="pass", message="label leak", failure_label="anything", error_code="E_FW_RESET", injected_anomaly=True)
    other_predicted, other_baseline = probabilities(model, changed)
    np.testing.assert_array_equal(predicted, other_predicted)
    np.testing.assert_array_equal(baseline, other_baseline)
    assert model["rates"].total == len(train)
    assert 0 <= predicted.min() <= predicted.max() <= 1
    assert set(feature_row(evaluation[0], 0.5)) == {
        "protocol", "component", "retry_count", "rssi", "latency_ms", "throughput_mbps", "historical_failure_rate",
    }


def test_unseen_pair_falls_back_to_training_global_rate():
    train, evaluation = generate(train_count=80, eval_count=40)
    model = fit_predictor(train)
    row = {**evaluation[0], "component": "NEW_COMPONENT"}
    assert model["rates"].probability(row) == float(labels(train).mean())
    assert np.isfinite(probabilities(model, [row])[0]).all()


def test_split_rejects_overlap_duplicate_ids_and_timezone_errors():
    train, evaluation = generate(train_count=80, eval_count=40)
    validate_split(train, evaluation)
    overlapping = deepcopy(evaluation)
    overlapping[0]["timestamp"] = train[-1]["timestamp"]
    with pytest.raises(ValueError, match="end before"):
        validate_split(train, overlapping)
    duplicate = deepcopy(evaluation)
    duplicate[0]["test_id"] = train[0]["test_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_split(train, duplicate)
    naive = deepcopy(train)
    for row in naive:
        row["timestamp"] = row["timestamp"].replace("+00:00", "")
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_split(naive, evaluation)


def test_calibration_and_fixed_threshold():
    truth = np.array([0, 1, 1, 0])
    metrics, bins = score_predictions(truth, np.array([0.1, 0.5, 1.0, 0.2]))
    assert metrics["precision"] == metrics["recall"] == 1
    assert metrics["brier_score"] == pytest.approx((0.01 + 0.25 + 0 + 0.04) / 4)
    assert sum(item["count"] for item in bins) == 4
    assert bins[-1]["count"] == 1


def test_serialization_and_api_use_real_model_without_clustering(tmp_path):
    train, evaluation = generate(train_count=80, eval_count=40)
    model = fit_predictor(train)
    model["run_id"] = "test-run"
    joblib.dump(model, tmp_path / "failure_model.joblib")
    service = FailureRiskService(tmp_path)
    app.dependency_overrides[get_failure_service] = lambda: service
    try:
        client = TestClient(app)
        row = {key: evaluation[0][key] for key in ("timestamp", "test_id", "protocol", "component", "retry_count", "rssi", "latency_ms", "throughput_mbps")}
        result = client.post("/predict/failure-risk", json={"logs": [row]})
        assert result.status_code == 200
        prediction = result.json()["predictions"][0]
        assert prediction["run_id"] == "test-run"
        assert prediction["failure_risk"] == probabilities(model, [row])[0][0]
        assert client.post("/predict/failure-risk", json={"logs": [{**row, "status": "fail"}]}).status_code == 422
        assert client.post("/predict/failure-risk", json={"logs": [{**row, "retry_count": -1}]}).status_code == 422
        assert client.post("/predict/failure-risk", json={"logs": [{**row, "timestamp": train[-1]["timestamp"]}]}).status_code == 422
        assert client.post("/predict/failure-risk", json={"logs": []}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_missing_model_returns_503(tmp_path, monkeypatch):
    import wireless_log_triage.api as api
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    assert TestClient(app).post("/predict/failure-risk", json={"logs": [{
        "timestamp": "2025-01-02T00:00:00+00:00", "test_id": "new", "protocol": "wifi",
        "component": "MAC", "retry_count": 0, "rssi": -55, "latency_ms": 20, "throughput_mbps": 180,
    }]}).status_code == 503


def test_heldout_report_cannot_be_overwritten(tmp_path):
    from wireless_log_triage.prediction import train_and_evaluate_prediction
    (tmp_path / "failure_evaluation.json").write_text("{}")
    with pytest.raises(FileExistsError, match="do not re-tune"):
        train_and_evaluate_prediction(tmp_path)
