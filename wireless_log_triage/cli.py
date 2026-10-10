import argparse
import json

from .config import DATA_DIR
from .dataset import generate, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Wireless log triage demo")
    parser.add_argument("command", choices=("prepare", "demo"))
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        from .modeling import train_and_evaluate

        train, evaluation = generate()
        write_jsonl(DATA_DIR / "train.jsonl", train)
        write_jsonl(DATA_DIR / "eval.jsonl", evaluation)
        print(json.dumps(train_and_evaluate(DATA_DIR), indent=2))
        return

    from .service import TriageService

    service = TriageService(DATA_DIR)
    evaluation = read_jsonl(DATA_DIR / "eval.jsonl")
    fields = ("timestamp", "test_id", "protocol", "component", "status", "error_code", "message", "rssi", "latency_ms", "throughput_mbps", "retry_count")
    sample = [{key: row[key] for key in fields} for row in evaluation[:100]]
    cause = next(row["failure_label"] for row in evaluation[100:] if row["status"] != "pass")
    matched = [{key: row[key] for key in fields} for row in evaluation[100:] if row["failure_label"] == cause][:8]
    if not service.collection.get(where={"batch_id": "eval-sample"})["ids"]:
        service.ingest("eval-sample", sample)
    if not service.collection.get(where={"batch_id": "demo-cause"})["ids"]:
        service.ingest("demo-cause", matched)
    result = {"ingested": len(sample) + len(matched), "clusters": service.clusters(), "anomalies": service.anomalies()}
    try:
        result["triage"] = service.triage("demo-cause")
    except RuntimeError as error:
        result["triage_error"] = str(error)
    print(json.dumps(result, indent=2))
