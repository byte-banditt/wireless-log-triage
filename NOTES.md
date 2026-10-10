# Interview notes

## 1. Synthetic data and holdout

The generator creates timestamped WiFi, Bluetooth, and cellular runs with nine labeled failure causes, plus passing runs. Error text has multiple paraphrases, shared context phrases, and occasional misleading error codes; numeric shifts overlap normal variation. A separate later time window forms the evaluation set, so no evaluation row is used to fit a model or choose its threshold. `injected_anomaly` is an independent ground-truth marker for deliberately altered RSSI, latency, or throughput; ordinary failures can still have unusual metrics.

## 2. Failure clustering and held-out scoring

`BAAI/bge-small-en-v1.5` embeds short error messages in CVE-RAG's BGE family, so paraphrases can meet without exact token matches. The small variant lowers RAM use on a Windows laptop; the dataset is synthetic, so larger embeddings are not assumed to help. HDBSCAN discovers dense groups and marks uncertain logs as noise; `approximate_predict` assigns held-out logs without fitting on them. KMeans is a comparison with nine clusters, using the known category count as an explicit advantage; training labels alone name clusters, and held-out precision, recall, F1, ARI, and valid silhouette go to MLflow.

## 3. Metric anomalies

A rolling z-score compares each metric with the previous 30 runs of the same radio protocol, which keeps WiFi throughput from being judged against Bluetooth throughput. It catches sudden changes in RSSI, latency, or throughput and needs no labeled anomalies to fit a complex model. Training history excludes injected training anomalies; evaluation runs in timestamp order and updates history after each prediction, and ingestion rejects older timestamps. The threshold is fixed at 3.0 before evaluation; held-out precision and recall measure how often the injected changes are caught.

## 4. Grounded triage and API

Chroma stores each ingested line with its embedding and structured metadata; cluster and anomaly endpoints read that same index. The triage endpoint checks for CVE-RAG's `qwen3:8b` Ollama model, then sends only failed lines from one batch and asks for structured, cited findings. Returned citations must match supplied `test_id` values, and the component must occur in cited lines; absent or invalid citations produce `insufficient evidence`. This checks citation identity, not every claim's factual truth, so a human should inspect the cited lines before acting. An unreachable Ollama endpoint returns HTTP 503 rather than presenting a synthetic answer as model output.

## 5. Packaging, CI, and deployment

The CPU Docker image copies the fitted model and downloads BGE at build time, following CVE-RAG's slim Python and port 8080 conventions. GitHub Actions runs unit tests on each push and pull request without downloading model weights. `deploy.ps1` builds the image with Cloud Build and deploys the API to Cloud Run; it limits concurrency to one because rolling history is stateful. Cloud Run's local Chroma directory is ephemeral, so this first deploy demonstrates the service rather than durable storage and needs a reachable Ollama URL.

## 6. Actual evaluation run

MLflow run `7c2af9e132764853a13ecf3477e2543a` scored 300 held-out rows, including 225 failed rows. HDBSCAN macro F1 was `0.8395698858468625`, while KMeans macro F1 was `0.5797979797979798`; HDBSCAN ARI was `0.2841960080833475`, so the clusters do not perfectly recover the nine generator labels. Rolling anomaly precision was `0.9090909090909091` and recall was `0.7894736842105263` against injected anomaly labels. These numbers describe this synthetic seed and exact run only, not real wireless hardware performance.

## 7. Publishing the benchmark

The repository tracks the synthetic train and holdout rows plus `data/evaluation.json`, while keeping the local MLflow SQLite database and fitted model out of Git. I checked that all 15 exported metrics exactly match the finished MLflow run; readers can reproduce a new local run with `wireless-triage prepare`. The README reports both macro F1 and ARI because they rank HDBSCAN and KMeans differently on this dataset. This project has no message broker, so it cannot supply an HTTP failure rate during broker outages; such a number must come from a separate measured load test.
