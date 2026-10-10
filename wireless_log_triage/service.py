from collections import Counter
from pathlib import Path
from threading import Lock

import chromadb
import hdbscan
import joblib
from sentence_transformers import SentenceTransformer

from .modeling import embed
from .triage import grounded_triage


class TriageService:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        models_path = data_dir / "models.joblib"
        if not models_path.exists():
            raise FileNotFoundError(f"Run 'wireless-triage prepare' first: {models_path}")
        self.models = joblib.load(models_path)
        self.encoder = SentenceTransformer(self.models["model_name"])
        collection_name = "wireless_logs_" + self.models["index_id"]
        self.collection = chromadb.PersistentClient(path=str(data_dir / "chroma")).get_or_create_collection(collection_name)
        self.state_path = data_dir / f"detector_state_{self.models['index_id']}.joblib"
        self.detector = joblib.load(self.state_path) if self.state_path.exists() else self.models["detector"]
        self._lock = Lock()

    def ingest(self, batch_id, logs):
        with self._lock:
            return self._ingest(batch_id, logs)

    def _ingest(self, batch_id, logs):
        if not logs:
            return []
        logs = sorted(logs, key=lambda row: row["timestamp"])
        ids = [row["test_id"] for row in logs]
        if len(ids) != len(set(ids)) or self.collection.get(ids=ids)["ids"]:
            raise ValueError("test_id must be unique across ingested logs")
        vectors = embed(self.encoder, logs)
        failed_indices = [index for index, row in enumerate(logs) if row["status"] != "pass"]
        assigned = {}
        if failed_indices:
            labels, _ = hdbscan.approximate_predict(self.models["hdbscan"], vectors[failed_indices])
            assigned = dict(zip(failed_indices, labels))
        metadata = []
        responses = []
        for index, row in enumerate(logs):
            cluster = int(assigned.get(index, -1))
            anomaly = self.detector.observe(row)
            item = {**row, "batch_id": batch_id, "cluster": cluster, "failure_pattern": self.models["hdbscan_map"].get(cluster, "unknown"), "anomalous": anomaly["anomalous"], "anomaly_score": anomaly["score"], "anomaly_metric": anomaly["metric"]}
            item["error_code"] = item.get("error_code") or ""
            metadata.append({key: value for key, value in item.items() if key != "message"})
            responses.append(item)
        self.collection.add(ids=ids, embeddings=vectors.tolist(), documents=[row["message"] for row in logs], metadatas=metadata)
        joblib.dump(self.detector, self.state_path)
        return responses

    def _rows(self, where=None):
        result = self.collection.get(where=where, include=["documents", "metadatas"])
        return [{**metadata, "test_id": test_id, "message": document} for test_id, metadata, document in zip(result["ids"], result["metadatas"], result["documents"])]

    def clusters(self):
        rows = [row for row in self._rows() if row["status"] != "pass"]
        counts = Counter((row["cluster"], row["failure_pattern"]) for row in rows)
        return [{"cluster": cluster, "failure_pattern": pattern, "count": count, "anomaly_count": sum(row["anomalous"] for row in rows if row["cluster"] == cluster), "test_ids": [row["test_id"] for row in rows if row["cluster"] == cluster]} for (cluster, pattern), count in sorted(counts.items())]

    def anomalies(self):
        return self._rows(where={"anomalous": True})

    def triage(self, batch_id):
        rows = self._rows(where={"batch_id": batch_id})
        if not rows:
            raise KeyError(batch_id)
        return grounded_triage(rows)
