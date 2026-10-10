from wireless_log_triage.dataset import FAILURES, generate


def test_dataset_is_deterministic_and_holdout_has_all_causes():
    training, evaluation = generate()
    assert (len(training), len(evaluation)) == (900, 300)
    assert training == generate()[0]
    assert set(FAILURES).issubset({row["failure_label"] for row in evaluation})
    assert {row["test_id"] for row in training}.isdisjoint({row["test_id"] for row in evaluation})
    assert max(row["timestamp"] for row in training) < min(row["timestamp"] for row in evaluation)
    assert any(row["injected_anomaly"] for row in evaluation)
