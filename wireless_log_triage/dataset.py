import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROTOCOLS = ("wifi", "bluetooth", "cellular")
FAILURES = {
    "handshake_timeout": ("MAC", "E_TIMEOUT", ("association handshake timed out after response wait", "peer did not finish handshake before timeout", "connection setup stalled awaiting peer")),
    "signal_degradation": ("PHY", "E_SIGNAL", ("received signal weak with unstable link margin", "radio link quality degraded during transfer", "low RSSI coincided with dropped frames")),
    "retry_storm": ("MAC", "E_RETRY", ("retransmission count surged while link stalled", "packet retries exceeded expected window", "repeated frames caused transfer delay")),
    "firmware_crash": ("FW", "E_FW_RESET", ("firmware watchdog reset after radio command", "controller crashed and restarted during test", "radio firmware lost heartbeat")),
    "auth_failure": ("SEC", "E_AUTH", ("peer rejected credentials during authentication", "security negotiation failed on key exchange", "authentication challenge was denied")),
    "throughput_collapse": ("MAC", "E_RATE", ("data rate collapsed under active traffic", "transfer throughput fell below target", "payload delivery slowed unexpectedly")),
    "rrc_reconfiguration": ("RRC", "E_RRC", ("RRC reconfiguration was rejected by network", "cellular control plane failed reconfiguration", "RRC state transition did not complete")),
    "interference": ("PHY", "E_INTERFERENCE", ("channel interference raised noise floor", "cochannel traffic disrupted radio frames", "spectrum congestion correlated with frame loss")),
    "power_state": ("PM", "E_POWER", ("wake transition failed during power save", "radio remained asleep after resume request", "power state transition delayed packet delivery")),
}
BASELINES = {
    "wifi": (-54.0, 24.0, 180.0),
    "bluetooth": (-66.0, 37.0, 2.0),
    "cellular": (-82.0, 65.0, 55.0),
}


def generate(seed=42, train_count=900, eval_count=300):
    randomizer = random.Random(seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    total = train_count + eval_count
    categories = tuple(FAILURES)
    for index in range(total):
        failure = index % 4 != 0
        category = categories[(index // 4) % len(categories)] if failure else "pass"
        protocol = "cellular" if category == "rrc_reconfiguration" else PROTOCOLS[index % len(PROTOCOLS)]
        baseline_rssi, baseline_latency, baseline_throughput = BASELINES[protocol]
        if failure:
            component, error_code, templates = FAILURES[category]
            message = randomizer.choice(templates)
            status = randomizer.choice(("fail", "fail", "timeout", "retry"))
            if randomizer.random() < 0.15:
                error_code = randomizer.choice(tuple(item[1] for item in FAILURES.values()))
            message += randomizer.choice(("", " after retry", " during roaming", " on test bench"))
        else:
            component = randomizer.choice(("PHY", "MAC", "RRC", "SEC"))
            error_code = ""
            message = "test completed within expected radio limits"
            status = "pass"
        rssi = baseline_rssi + randomizer.gauss(0, 3)
        latency = max(1.0, baseline_latency + randomizer.gauss(0, baseline_latency * 0.09))
        throughput = max(0.01, baseline_throughput + randomizer.gauss(0, baseline_throughput * 0.08))
        retry_count = max(0, int(randomizer.gauss(1.5, 1)))
        if category == "signal_degradation":
            rssi -= randomizer.uniform(2, 8)
        elif category == "retry_storm":
            retry_count += randomizer.randint(3, 9)
        elif category == "throughput_collapse":
            throughput *= randomizer.uniform(0.6, 0.95)
        injected_anomaly = randomizer.random() < 0.12
        if injected_anomaly:
            kind = randomizer.choice(("rssi", "latency_ms", "throughput_mbps"))
            if kind == "rssi":
                rssi -= randomizer.uniform(17, 27)
            elif kind == "latency_ms":
                latency *= randomizer.uniform(2.5, 4)
            else:
                throughput *= randomizer.uniform(0.04, 0.22)
        rows.append({
            "timestamp": (start + timedelta(minutes=index)).isoformat(),
            "test_id": f"T{index:05d}",
            "batch_id": f"B{index // 20:04d}",
            "protocol": protocol,
            "component": component,
            "status": status,
            "error_code": error_code,
            "message": message,
            "rssi": round(rssi, 3),
            "latency_ms": round(latency, 3),
            "throughput_mbps": round(throughput, 3),
            "retry_count": retry_count,
            "failure_label": category,
            "injected_anomaly": injected_anomaly,
        })
    return rows[:train_count], rows[train_count:]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]
