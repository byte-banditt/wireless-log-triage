"""Independent, chronological failure-risk model; no clustering dependencies."""

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import read_jsonl

FEATURES = ("protocol", "component", "retry_count", "rssi", "latency_ms", "throughput_mbps")
STATUSES = {"pass", "fail", "timeout", "retry"}
THRESHOLD = 0.5  # Fixed before inspecting the held-out window.


def labels(rows):
    if any(row["status"] not in STATUSES for row in rows):
        raise ValueError("status must be pass, fail, timeout, or retry")
    return np.array([int(row["status"] != "pass") for row in rows])


def validate_split(train, evaluation):
    if not train or not evaluation:
        raise ValueError("both time windows must contain records")
    for rows in (train, evaluation):
        times = [datetime.fromisoformat(row["timestamp"]) for row in rows]
        if any(value.tzinfo is None for value in times) or times != sorted(times):
            raise ValueError("timestamps must be timezone-aware and chronological")
        if set(labels(rows)) != {0, 1}:
            raise ValueError("each window must contain both pass and failure outcomes")
        for row in rows:
            feature_row(row, 0.5)
    if max(datetime.fromisoformat(r["timestamp"]) for r in train) >= min(datetime.fromisoformat(r["timestamp"]) for r in evaluation):
        raise ValueError("training window must end before evaluation starts")
    ids = [row["test_id"] for row in train + evaluation]
    if len(ids) != len(set(ids)):
        raise ValueError("test_id must be unique across both windows")


def feature_row(row, rate):
    result = {key: row[key] for key in FEATURES}
    for key in FEATURES[2:]:
        result[key] = float(result[key])
        if not np.isfinite(result[key]):
            raise ValueError(f"{key} must be finite")
    if result["protocol"] not in ("wifi", "bluetooth", "cellular") or not result["component"]:
        raise ValueError("protocol and component must be valid")
    if any(result[key] < 0 for key in ("retry_count", "latency_ms", "throughput_mbps")):
        raise ValueError("retry_count, latency_ms, and throughput_mbps must be nonnegative")
    result["historical_failure_rate"] = float(rate)
    return result


class HistoricalRates:
    """Unsmoothed component/protocol frequency; unseen pairs use global history."""

    def __init__(self):
        self.counts = defaultdict(lambda: [0, 0])
        self.total = 0
        self.failures = 0

    def probability(self, row):
        failures, count = self.counts.get((row["component"], row["protocol"]), (0, 0))
        return failures / count if count else (self.failures / self.total if self.total else 0.5)

    def update(self, row):
        failed = int(labels([row])[0])
        counts = self.counts[(row["component"], row["protocol"])]
        counts[0] += failed
        counts[1] += 1
        self.failures += failed
        self.total += 1

    def __getstate__(self):
        return {"counts": dict(self.counts), "total": self.total, "failures": self.failures}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.counts = defaultdict(lambda: [0, 0], self.counts)


def training_features(rows):
    """Only strictly earlier timestamps contribute, including within training."""
    rates = HistoricalRates()
    features = []
    pending = []
    previous = None
    for row in rows:
        timestamp = datetime.fromisoformat(row["timestamp"])
        if timestamp != previous:
            for prior in pending:
                rates.update(prior)
            pending = []
        features.append(feature_row(row, rates.probability(row)))
        pending.append(row)
        previous = timestamp
    for prior in pending:
        rates.update(prior)
    return features, rates


def fit_predictor(train):
    features, rates = training_features(train)
    classifier = make_pipeline(
        DictVectorizer(sparse=False), StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    )
    classifier.fit(features, labels(train))
    return {"classifier": classifier, "rates": rates, "threshold": THRESHOLD,
            "train_end": max(datetime.fromisoformat(row["timestamp"]) for row in train).isoformat()}


def probabilities(model, rows):
    baseline = np.array([model["rates"].probability(row) for row in rows])
    features = [feature_row(row, rate) for row, rate in zip(rows, baseline)]
    return model["classifier"].predict_proba(features)[:, 1], baseline


def score_predictions(truth, probability):
    predicted = probability >= THRESHOLD
    bins = []
    ece = 0.0
    for index in range(10):
        mask = (probability >= index / 10) & ((probability < (index + 1) / 10) if index < 9 else (probability <= 1))
        count = int(mask.sum())
        mean_probability = float(probability[mask].mean()) if count else None
        observed = float(truth[mask].mean()) if count else None
        if count:
            ece += count / len(truth) * abs(mean_probability - observed)
        bins.append({"lower": index / 10, "upper": (index + 1) / 10, "count": count,
                     "mean_probability": mean_probability, "failure_fraction": observed})
    return {"pr_auc_average_precision": float(average_precision_score(truth, probability)),
            "precision": float(precision_score(truth, predicted, zero_division=0)),
            "recall": float(recall_score(truth, predicted, zero_division=0)),
            "brier_score": float(brier_score_loss(truth, probability)), "ece_10_bins": float(ece)}, bins


def uncertainty(truth, model_probability, baseline_probability, rows, draws=1000):
    """Paired time-batch bootstrap of fixed predictions, not model selection."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.get("batch_id", f"block-{index // 20}")].append(index)
    blocks = [np.array(indices) for indices in groups.values()]
    random = np.random.default_rng(42)
    values = []
    for _ in range(draws):
        indices = np.concatenate([blocks[i] for i in random.integers(len(blocks), size=len(blocks))])
        if len(np.unique(truth[indices])) < 2:
            continue
        model_ap = average_precision_score(truth[indices], model_probability[indices])
        baseline_ap = average_precision_score(truth[indices], baseline_probability[indices])
        values.append((model_ap, baseline_ap, model_ap - baseline_ap))
    if not values:
        raise ValueError("bootstrap produced no two-class samples")
    bounds = np.quantile(np.asarray(values), [0.025, 0.975], axis=0)
    return {"method": "paired time-batch bootstrap of frozen holdout predictions",
            "draws": draws, "valid_draws": len(values), "blocks": len(blocks),
            "classifier_pr_auc_95pct": bounds[:, 0].tolist(),
            "baseline_pr_auc_95pct": bounds[:, 1].tolist(),
            "paired_pr_auc_delta_95pct": bounds[:, 2].tolist()}


def train_and_evaluate_prediction(data_dir: Path):
    import mlflow
    import sklearn

    report_path = data_dir / "failure_evaluation.json"
    if report_path.exists():
        raise FileExistsError("held-out failure evaluation already exists; do not re-tune on this window")
    train = read_jsonl(data_dir / "train.jsonl")
    evaluation = read_jsonl(data_dir / "eval.jsonl")
    validate_split(train, evaluation)
    model = fit_predictor(train)
    # One held-out prediction pass for each method, with frozen training history.
    classifier_probability, baseline_probability = probabilities(model, evaluation)
    truth = labels(evaluation)
    classifier_metrics, classifier_bins = score_predictions(truth, classifier_probability)
    baseline_metrics, baseline_bins = score_predictions(truth, baseline_probability)
    intervals = uncertainty(truth, classifier_probability, baseline_probability, evaluation)
    uri = f"sqlite:///{(data_dir / 'mlflow.db').resolve().as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("wireless-log-triage")
    hashes = {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in ("train.jsonl", "eval.jsonl")}
    with mlflow.start_run(run_name="held-out-failure-prediction") as run:
        report = {"run_id": run.info.run_id, "tracking_uri": uri,
                  "experiment": "wireless-log-triage", "run_name": "held-out-failure-prediction",
                  "train_rows": len(train), "eval_rows": len(evaluation), "eval_failures": int(truth.sum()),
                  "eval_passes": int((truth == 0).sum()), "threshold": THRESHOLD,
                  "target": "status != pass (fail, timeout, retry)",
                  "features": [*FEATURES, "historical_failure_rate"], "train_end": model["train_end"],
                  "eval_start": evaluation[0]["timestamp"], "input_sha256": hashes,
                  "metrics": {"classifier": classifier_metrics, "historical_rate_baseline": baseline_metrics},
                  "calibration": {"classifier": classifier_bins, "historical_rate_baseline": baseline_bins},
                  "uncertainty": intervals, "sklearn_version": sklearn.__version__,
                  "limitations": ["Single synthetic held-out window; not real hardware validation.",
                                   "One held-out synthetic window does not establish stable deployment PR-AUC; inspect sample counts and block intervals.",
                                   "Per-run numeric telemetry must be available before the outcome decision; next-run forecasting is not validated."]}
        mlflow.log_params({"classifier": "LogisticRegression", "C": 1.0, "threshold": THRESHOLD,
                           "train_rows": len(train), "eval_rows": len(evaluation), "split": "existing chronological windows",
                           "history": "strictly earlier training timestamps; frozen on eval", **hashes})
        mlflow.log_metrics({f"failure_{method}_{key}": value for method, metrics in report["metrics"].items() for key, value in metrics.items()})
        for key in ("classifier_pr_auc_95pct", "baseline_pr_auc_95pct", "paired_pr_auc_delta_95pct"):
            mlflow.log_metrics({f"failure_{key}_lower": intervals[key][0], f"failure_{key}_upper": intervals[key][1]})
        model["run_id"] = run.info.run_id
        joblib.dump(model, data_dir / "failure_model.joblib")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        predictions_path = data_dir / "failure_predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as file:
            for row, label, predicted, baseline in zip(evaluation, truth, classifier_probability, baseline_probability):
                file.write(json.dumps({"test_id": row["test_id"], "timestamp": row["timestamp"], "failure": int(label),
                                       "classifier_probability": float(predicted), "baseline_probability": float(baseline)}) + "\n")
        for path in (report_path, predictions_path, data_dir / "failure_model.joblib"):
            mlflow.log_artifact(str(path), artifact_path="failure_prediction")
    return report


class FailureRiskService:
    def __init__(self, data_dir):
        self.model = joblib.load(data_dir / "failure_model.joblib")

    def predict(self, rows):
        if any(datetime.fromisoformat(row["timestamp"]) <= datetime.fromisoformat(self.model["train_end"]) for row in rows):
            raise ValueError("prediction timestamp must be after the training window")
        if not rows:
            return []
        predicted, baseline = probabilities(self.model, rows)
        return [{"test_id": row["test_id"], "failure_risk": float(risk), "baseline_failure_risk": float(rate),
                 "predicted_failure": bool(risk >= self.model["threshold"]), "threshold": self.model["threshold"],
                 "run_id": self.model["run_id"]} for row, risk, rate in zip(rows, predicted, baseline)]
