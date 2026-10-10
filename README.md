# Wireless Log Triage

Software-only wireless test-log triage demo for WiFi, Bluetooth, and cellular validation. It generates synthetic labeled logs, embeds failure text with BGE-small, compares HDBSCAN with KMeans, detects numeric anomalies, stores ingested logs in Chroma, and asks Ollama for cited triage. Design explanations live in [NOTES.md](NOTES.md).

## Windows setup

Use Python 3.10 or 3.11 in PowerShell. No WSL, radio hardware, or C/C++ build is required; install dependencies from wheels.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu"
python -m pip install -r requirements.txt pytest httpx
python -m pip install --no-deps -e .
wireless-triage prepare
```

`prepare` writes `data/train.jsonl`, `data/eval.jsonl`, `data/models.joblib`, and an actual MLflow run in `data/mlflow.db`. Its stdout and `data/evaluation.json` contain the run ID and unrounded held-out scores. Inspect with `mlflow ui --backend-store-uri sqlite:///data/mlflow.db`.

## Held-out benchmark

The committed [evaluation report](data/evaluation.json) records MLflow run `7c2af9e132764853a13ecf3477e2543a`: 900 training rows and 300 held-out rows (225 failures). The metrics below are the run's unrounded values, not estimates. Re-run `wireless-triage prepare` to regenerate the dataset, model, report, and local MLflow run.

| Method | Metric | Value |
| --- | --- | ---: |
| HDBSCAN | Macro F1 | 0.8395698858468625 |
| HDBSCAN | ARI | 0.2841960080833475 |
| KMeans | Macro F1 | 0.5797979797979798 |
| KMeans | ARI | 0.5686269381492283 |
| Rolling z-score | Anomaly precision | 0.9090909090909091 |
| Rolling z-score | Anomaly recall | 0.7894736842105263 |

The comparison is mixed: HDBSCAN has higher mapped macro F1, while KMeans has higher ARI. See [NOTES.md](NOTES.md) for why both are reported.

For grounded triage, run Ollama locally with `qwen3:8b` (`ollama pull qwen3:8b`). Set `TRIAGE_OLLAMA_URL` if Ollama runs elsewhere. Clustering and anomaly endpoints work without Ollama.

```powershell
wireless-triage demo
uvicorn app:app --host 127.0.0.1 --port 8080
```

The demo ingests held-out logs and prints cluster and anomaly results, then a cited triage response when Ollama is available. `GET /docs` shows the request schema. Example:

```powershell
$body = @{ batch_id = 'manual-1'; logs = @(@{ timestamp = '2025-01-02T00:00:00+00:00'; test_id = 'MANUAL-1'; protocol = 'wifi'; component = 'MAC'; status = 'timeout'; error_code = 'E_TIMEOUT'; message = 'association handshake timed out after response wait'; rssi = -55; latency_ms = 26; throughput_mbps = 176; retry_count = 2 }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://localhost:8080/ingest -ContentType application/json -Body $body
Invoke-RestMethod http://localhost:8080/clusters
Invoke-RestMethod http://localhost:8080/triage/manual-1
Invoke-RestMethod http://localhost:8080/anomalies
```

## Docker and Cloud Run

After `prepare`, build the CPU image with `docker build -t wireless-log-triage .`. The Dockerfile follows CVE-RAG's slim Python, CPU PyTorch, port 8080, and preloaded BGE layout. Run locally with `docker run -p 8080:8080 -e TRIAGE_OLLAMA_URL=http://host.docker.internal:11434 wireless-log-triage`.

For GCP, install and authenticate `gcloud`, then run `./deploy.ps1 -Project PROJECT_ID -Region us-central1 -OllamaUrl https://YOUR-REACHABLE-OLLAMA`. Cloud Run hosts the API; triage requires an Ollama endpoint reachable from it. Chroma data in the container filesystem is ephemeral, so this Phase 1 deployment is a demo, not durable production storage.

## Limitations

- All labels and scores come from one synthetic seed and its held-out time window. They do not establish performance on real RF hardware or production test logs.
- HDBSCAN can mark failures as noise; its held-out ARI is `0.2841960080833475`. Cluster names are mapped using training labels, not inferred root causes.
- Grounding validates cited test IDs and components, not every LLM claim. Ollama must be reachable; otherwise `GET /triage/{batch_id}` returns HTTP 503.
- The local MLflow SQLite database, fitted model, and Chroma index are generated artifacts, not committed. Re-run `wireless-triage prepare`; Cloud Run storage remains ephemeral.
- This service has no message broker or broker-outage load test. Its HTTP failure rate during broker outages is not measured here; no rate should be inferred from these ML benchmarks.
