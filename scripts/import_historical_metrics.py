"""Restore visibility of a committed report; never pretend to re-execute it."""

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    path = args.data_dir / "evaluation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(f"sqlite:///{(args.data_dir / 'mlflow.db').resolve().as_posix()}")
    experiment = mlflow.set_experiment("wireless-log-triage")
    client = MlflowClient()
    original = client.search_runs([experiment.experiment_id], filter_string=f"attributes.run_id = '{report['run_id']}'")
    archived = client.search_runs([experiment.experiment_id], filter_string=f"tags.archived_source_run_id = '{report['run_id']}'")
    if original or archived:
        print(json.dumps({"already_present": (original or archived)[0].info.run_id}, indent=2))
        return
    with mlflow.start_run(run_name="archived-held-out-evaluation-NOT-REEXECUTED") as run:
        mlflow.set_tags({"archived_source_run_id": report["run_id"],
                         "provenance": "Imported committed evaluation.json; NOT re-executed or revalidated",
                         "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        mlflow.log_params({key: report[key] for key in ("train_rows", "eval_rows", "eval_failures")})
        mlflow.log_metrics(report["metrics"])
        mlflow.log_artifact(str(path))
        print(json.dumps({"archived_run_id": run.info.run_id, "original_run_id": report["run_id"],
                          "provenance": "Imported committed report; NOT re-executed or revalidated",
                          "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
