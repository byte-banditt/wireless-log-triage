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
| Failure-risk logistic regression | PR-AUC (average precision) | 0.9076171103262939 |
| Failure-risk logistic regression | Precision (threshold 0.5) | 0.791970802919708 |
| Failure-risk logistic regression | Recall (threshold 0.5) | 0.9644444444444444 |
| Failure-risk logistic regression | Brier score (lower is better) | 0.15289729719811837 |
| Failure-risk logistic regression | ECE, 10 equal-width bins (lower is better) | 0.0520332635995658 |
| Component/protocol historical lookup | PR-AUC (average precision) | 0.8854525018702433 |
| Component/protocol historical lookup | Precision (threshold 0.5) | 0.7833935018050542 |
| Component/protocol historical lookup | Recall (threshold 0.5) | 0.9644444444444444 |
| Component/protocol historical lookup | Brier score (lower is better) | 0.1538093828322181 |
| Component/protocol historical lookup | ECE, 10 equal-width bins (lower is better) | 0.014834531172789572 |

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


## Failure-risk prediction (independent of clustering/anomaly detection)

`wireless-triage predict-train` reads the existing `data/train.jsonl` and `data/eval.jsonl`; it does not run `prepare`, generate rows, make a random split, or modify the clustering/anomaly models. The measured run used 900 earlier training records and 300 later evaluation records (225 failures, 75 passes). The original six benchmark rows above remain unchanged; the added rows come directly from [the new evaluation report](data/failure_evaluation.json).

The schema has no literal `status="failure"`: this model uses the existing failure definition `status != "pass"`, namely `fail`, `timeout`, or `retry`. This predicts an attempt's non-pass outcome, not whether all retries ultimately fail. It uses only protocol, component, retry_count, RSSI, latency, throughput, and historical component/protocol failure rate. Status, message, error_code, failure_label, injected_anomaly, test ID, and batch ID are excluded from classifier features. Timestamp is used only for chronology and the inference cutoff.

Historical rates for training examples use strictly earlier training timestamps; rows at the same timestamp do not see each other's outcomes. An unseen pair falls back to the earlier global rate, or a fixed 0.5 prior before any history exists. Evaluation and API inference use the final training-only lookup, frozen without incorporating evaluation labels. The lookup itself is the explicit simple baseline. Logistic regression uses one-hot categorical features, standardized features, fixed C=1, and a fixed 0.5 decision threshold. No hyperparameter or threshold search, probability recalibration, or repeated model fitting on the evaluation window was performed. PR-AUC here means scikit-learn average precision, not trapezoidal interpolation of the PR curve.

**Outcome:** classifier ranking is better than the lookup on this window; recall is identical. Its Brier score is slightly lower, but its ECE is higher, so calibration is not uniformly improved. The full report includes all ten reliability bins (mean probability, observed failure fraction, and sample count). No overall superiority or real-hardware performance is claimed.

### Actual execution and MLflow evidence

The captured [training stdout](data/failure_training.stdout.json), [per-row frozen predictions](data/failure_predictions.jsonl), and [MLflow read-back verification](data/failure_logging_verification.json) contain actual executed results. Input hashes are in the report. Only one evaluation prediction pass per method was made; paired bootstrap intervals were computed from those frozen outputs, without retraining or threshold selection.

Verification: 14 pytest tests passed, including the existing suite, history/label-leakage checks, serialization, and endpoint validation. A separate [API smoke response](data/failure_api_smoke.json) returned HTTP 200 using the actual trained artifact and a new input, without reevaluating the held-out window.

```powershell
# After normal setup / prepare, reuse its existing chronological data:
wireless-triage predict-train
# Prediction report is intentionally write-once. For a genuinely new experiment,
# use new train/eval windows in a separate TRIAGE_DATA_DIR, not this reused holdout.
mlflow ui --backend-store-uri sqlite:///data/mlflow.db --port 5000
```

The new run uses the exact SQLite backend and `wireless-log-triage` experiment configured by the existing `prepare` code, with run name `held-out-failure-prediction`. A fresh Git clone does not contain the original ignored MLflow database. To make its recorded clustering/anomaly metrics visible alongside this run without rerunning or changing that pipeline, the committed `evaluation.json` was imported into a separate **archived / NOT REEXECUTED** run. This import copies recorded values unchanged; it is not a new validation of the old metrics. Its provenance and original source run ID are tagged explicitly; see [import stdout](data/historical_import.stdout.json). The helper skips importing if the original run or an archive already exists:

```powershell
python scripts/import_historical_metrics.py
```

Measured failure run: [`89c3d0844766415f93284e27f66579bc`](http://localhost:5000/#/experiments/1/runs/89c3d0844766415f93284e27f66579bc). Historical archive: [`73ddb8e3c26b4606b855c9323d6b0ef2`](http://localhost:5000/#/experiments/1/runs/73ddb8e3c26b4606b855c9323d6b0ef2); original recorded source ID `7c2af9e132764853a13ecf3477e2543a`. Links require the local UI and this generated database.

### Sample size and uncertainty

**900 training / 300 evaluation records are too small to claim a stable deployment PR-AUC.** There are only 75 negative outcomes and 15 held-out time batches, all from one synthetic seed. The measured 95% intervals below use 1,000 paired time-batch bootstrap draws; they describe this window and cannot establish robustness to new seeds, protocol mixtures, hardware, or future time drift. The block size follows the existing batch IDs (20 records); it was not selected from evaluation results.

- Classifier PR-AUC: `[0.8887992163205952, 0.9248572025862576]`.
- Lookup PR-AUC: `[0.8682181533195735, 0.9048471133340356]`.
- Paired classifier-minus-lookup PR-AUC: `[0.00029095074227597495, 0.04262571806907043]`.

Regeneration recommendation: use harness **`--count 10000` per protocol**, its supported maximum, initially with `--max-retries 0` to get one record per independent mock test rather than correlated retry attempts. These are proposed collection counts, not executed benchmark results. Run from the separate harness repository:

```powershell
./build/wireless-test-harness.exe --count 10000 --protocol wifi --fail-rate 0.25 --seed 43 --max-retries 0 --out wifi-10000.jsonl
./build/wireless-test-harness.exe --count 10000 --protocol bluetooth --fail-rate 0.25 --seed 44 --max-retries 0 --out bluetooth-10000.jsonl
./build/wireless-test-harness.exe --count 10000 --protocol cellular --fail-rate 0.25 --seed 45 --max-retries 0 --out cellular-10000.jsonl
```

Retain an earlier 75% training window and a later 25% evaluation window, as in `prepare`, never a random split. This plans 2,500 held-out tests per protocol, not a guarantee of stable PR-AUC. Keep complete tests/retry groups in one window if retries are enabled, namespace the harness test IDs by protocol when combining files (each invocation restarts IDs), and keep the new windows separate from the committed 900/300 data. Repeat on genuinely new seeds/windows before making a resume claim; do not use this evaluated window for model selection.

`prepare` currently generates Python synthetic records; it does not import harness output. For the proposed harness experiment, explicitly convert the outputs into chronological `train.jsonl` / `eval.jsonl` in a **new** `TRIAGE_DATA_DIR` before `predict-train`, rather than calling `prepare` and overwriting them. The two generators share a record schema but have different outcome/telemetry distributions; the reported Python-generator scores are not harness validation. No harness source change was needed or made.

### API

`POST /predict/failure-risk` uses its own model dependency and does not initialize the Chroma/BGE service. Supply telemetry known before the failure decision and a timestamp strictly after the model's training window. Outcome fields are rejected, unknown components use the training global-rate fallback, nonfinite telemetry is rejected, and a missing model returns HTTP 503 with the training command. Each response exposes both classifier and lookup risk, the fixed threshold, and the measured run ID.

```json
{
  "logs": [{
    "timestamp": "2025-01-02T00:00:00+00:00",
    "test_id": "FUTURE-1",
    "protocol": "wifi",
    "component": "MAC",
    "retry_count": 1,
    "rssi": -55,
    "latency_ms": 24,
    "throughput_mbps": 180
  }]
}
```

**Prediction timing limitation:** the existing JSONL files contain completed-run telemetry. A later held-out time window verifies temporal generalization of per-run classification, not next-run forecasting from a previous run. RSSI/latency/throughput may encode already-occurring failures; they must be available at the intended decision time for a prospective risk use. This work does not validate that availability or claim advance failure warning. Such a claim needs decision-time telemetry and outcome alignment, not just higher held-out PR-AUC.

`data/failure_model.joblib` is a generated, ignored artifact, independent of `models.joblib`. The existing Dockerfile packages the clustering artifact only: when using prediction in Docker, also mount the locally trained failure artifact read-only at `/data/failure_model.joblib`, or add it to your deployment artifact bundle. Training is not run inside the API.

New files: `wireless_log_triage/prediction.py`, `tests/test_prediction.py`, `scripts/import_historical_metrics.py`, and the failure evaluation/prediction/stdout/read-back reports under `data/`. Existing `api.py` and `cli.py` add the endpoint and `predict-train` command. HDBSCAN/KMeans/z-score implementation and `data/evaluation.json` are untouched.
