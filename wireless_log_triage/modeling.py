import copy
import json
from collections import Counter, deque
from pathlib import Path
from uuid import uuid4

import hdbscan
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, precision_score, recall_score, silhouette_score

from .config import MODEL_NAME
from .dataset import read_jsonl

METRICS = ("rssi", "latency_ms", "throughput_mbps")


class RollingDetector:
    def __init__(self, window=30, threshold=3.0):
        self.window = window
        self.threshold = threshold
        self.history = {}
        self.last_timestamp = {}

    def observe(self, row):
        timestamp = row.get("timestamp")
        last = self.last_timestamp.get(row["protocol"])
        if timestamp is not None and last is not None and timestamp < last:
            raise ValueError("logs must be ingested in timestamp order per protocol")
        series = self.history.setdefault(row["protocol"], {metric: deque(maxlen=self.window) for metric in METRICS})
        scores = {}
        for metric in METRICS:
            history = series[metric]
            if len(history) < self.window:
                scores[metric] = 0.0
            else:
                deviation = np.std(history, ddof=1)
                scores[metric] = abs(float(row[metric]) - float(np.mean(history))) / max(float(deviation), 1e-6)
        for metric in METRICS:
            series[metric].append(float(row[metric]))
        if timestamp is not None:
            self.last_timestamp[row["protocol"]] = timestamp
        largest = max(scores, key=scores.get)
        return {"anomalous": scores[largest] > self.threshold, "score": round(scores[largest], 6), "metric": largest}


def fit_detector(rows):
    detector = RollingDetector()
    for row in rows:
        if not row["injected_anomaly"]:
            detector.observe(row)
    return detector


def embed(encoder, rows):
    return np.asarray(encoder.encode([row["message"] for row in rows], normalize_embeddings=True, show_progress_bar=False))


def _classification_metrics(labels, predicted):
    return {
        "precision_macro": float(precision_score(labels, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predicted, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predicted, average="macro", zero_division=0)),
    }


def _cluster_metrics(name, train_labels, train_clusters, eval_labels, eval_clusters, eval_vectors):
    mapping = {}
    for cluster in sorted(set(train_clusters)):
        if cluster != -1:
            mapping[int(cluster)] = Counter(label for label, assigned in zip(train_labels, train_clusters) if assigned == cluster).most_common(1)[0][0]
    predictions = [mapping.get(int(cluster), "unknown") for cluster in eval_clusters]
    metrics = _classification_metrics(eval_labels, predictions)
    metrics["ari"] = float(adjusted_rand_score(eval_labels, eval_clusters))
    metrics["noise_fraction"] = float(np.mean(np.asarray(eval_clusters) == -1))
    valid = np.asarray(eval_clusters) != -1
    if 1 < len(set(np.asarray(eval_clusters)[valid])) < int(valid.sum()):
        metrics["silhouette"] = float(silhouette_score(eval_vectors[valid], np.asarray(eval_clusters)[valid], metric="cosine"))
    return mapping, {f"{name}_{key}": value for key, value in metrics.items()}


def train_and_evaluate(data_dir: Path, model_name=MODEL_NAME):
    import mlflow

    train = read_jsonl(data_dir / "train.jsonl")
    evaluation = read_jsonl(data_dir / "eval.jsonl")
    train_failures = [row for row in train if row["status"] != "pass"]
    eval_failures = [row for row in evaluation if row["status"] != "pass"]
    encoder = SentenceTransformer(model_name)
    train_vectors = embed(encoder, train_failures)
    eval_vectors = embed(encoder, eval_failures)
    density = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=3, prediction_data=True)
    density.fit(train_vectors)
    density_eval, _ = hdbscan.approximate_predict(density, eval_vectors)
    baseline = KMeans(n_clusters=9, n_init=10, random_state=42)
    baseline.fit(train_vectors)
    baseline_eval = baseline.predict(eval_vectors)
    truth_train = [row["failure_label"] for row in train_failures]
    truth_eval = [row["failure_label"] for row in eval_failures]
    density_map, density_metrics = _cluster_metrics("hdbscan", truth_train, density.labels_, truth_eval, density_eval, eval_vectors)
    baseline_map, baseline_metrics = _cluster_metrics("kmeans", truth_train, baseline.labels_, truth_eval, baseline_eval, eval_vectors)
    detector = fit_detector(train)
    eval_detector = copy.deepcopy(detector)
    anomaly_predictions = [eval_detector.observe(row)["anomalous"] for row in evaluation]
    anomaly_truth = [row["injected_anomaly"] for row in evaluation]
    metrics = {
        **density_metrics,
        **baseline_metrics,
        "anomaly_precision": float(precision_score(anomaly_truth, anomaly_predictions, zero_division=0)),
        "anomaly_recall": float(recall_score(anomaly_truth, anomaly_predictions, zero_division=0)),
        "anomaly_f1": float(f1_score(anomaly_truth, anomaly_predictions, zero_division=0)),
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"hdbscan": density, "kmeans": baseline, "hdbscan_map": density_map, "kmeans_map": baseline_map, "detector": detector, "model_name": model_name, "index_id": uuid4().hex[:12]}, data_dir / "models.joblib")
    mlflow.set_tracking_uri(f"sqlite:///{(data_dir / 'mlflow.db').resolve().as_posix()}")
    mlflow.set_experiment("wireless-log-triage")
    with mlflow.start_run(run_name="held-out-evaluation") as run:
        mlflow.log_params({"seed": 42, "train_rows": len(train), "eval_rows": len(evaluation), "embedding_model": model_name, "hdbscan_min_cluster_size": 8, "hdbscan_min_samples": 3, "kmeans_clusters": 9, "anomaly_window": 30, "anomaly_z_threshold": 3.0})
        mlflow.log_metrics(metrics)
        report = {"run_id": run.info.run_id, "train_rows": len(train), "eval_rows": len(evaluation), "eval_failures": len(eval_failures), "metrics": metrics}
        (data_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(data_dir / "evaluation.json"))
    return report
