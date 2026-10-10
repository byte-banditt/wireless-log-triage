import json

import requests

from .config import OLLAMA_MODEL, OLLAMA_URL


def grounded_triage(rows, model=OLLAMA_MODEL, url=OLLAMA_URL):
    failures = [row for row in rows if row["status"] != "pass"]
    if not failures:
        return {"summary": "insufficient evidence", "likely_root_cause": "insufficient evidence", "affected_component": None, "citations": [], "model": model}
    known = {row["test_id"] for row in failures}
    context = [{key: row[key] for key in ("timestamp", "test_id", "protocol", "component", "status", "error_code", "message", "cluster", "rssi", "latency_ms", "throughput_mbps", "retry_count")} for row in failures]
    prompt = (
        "You are triaging wireless test logs. Use only the supplied log lines. "
        "Return JSON with keys summary, likely_root_cause, affected_component, citations. "
        "citations must be a list of exact test_id values supporting the cause. "
        "If logs do not support one likely cause, say 'insufficient evidence' in both summary and likely_root_cause, with empty citations. "
        "Never infer hardware measurements or causes absent from the lines.\n"
        f"LOG LINES:\n{json.dumps(context)}"
    )
    try:
        available = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
        available.raise_for_status()
        names = {entry.get("name") for entry in available.json().get("models", [])}
        if model not in names:
            raise RuntimeError(f"Ollama model unavailable: {model}")
        response = requests.post(f"{url.rstrip('/')}/api/generate", json={"model": model, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0}}, timeout=120)
        response.raise_for_status()
        answer = json.loads(response.json()["response"])
    except (requests.RequestException, ValueError, KeyError) as error:
        raise RuntimeError(f"Ollama triage unavailable: {error}") from error
    citations = answer.get("citations", [])
    if not isinstance(citations, list) or not citations or any(citation not in known for citation in citations):
        return {"summary": "insufficient evidence", "likely_root_cause": "insufficient evidence", "affected_component": None, "citations": [], "model": model}
    component = answer.get("affected_component")
    cited_rows = [row for row in failures if row["test_id"] in citations]
    if component not in {row["component"] for row in cited_rows}:
        component = None
    return {"summary": str(answer.get("summary", "insufficient evidence")), "likely_root_cause": str(answer.get("likely_root_cause", "insufficient evidence")), "affected_component": component, "citations": citations, "model": model}
