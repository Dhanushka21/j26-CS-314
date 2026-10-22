"""STEP 15: load + latency test of the running API (NFR1 < 100 ms, NFR4 >= 10,000 events/day).

  1. Terminal 1:  uvicorn c2_detection.api.main:app --port 8002
  2. Terminal 2:  python c2_detection/tools/load_test.py --events 10000 --concurrency 7

Replays real TEST flows for the 7 devices in contracts/devices.json (each device gets its own
attack/benign stream, in order, so the per-device windows fill up like real traffic).
Creates results/load_test.json and results/load_test_latency.png
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from contracts.models import DEVICES  # noqa: E402
from src.data import read_features  # noqa: E402

# one traffic stream per device (benign + a mix of attacks)
STREAMS = ["Benign_test.pcap.csv", "TCP_IP-DDoS-SYN_test.pcap.csv", "TCP_IP-DoS-UDP_test.pcap.csv",
           "MQTT-DDoS-Connect_Flood_test.pcap.csv", "Recon-Port_Scan_test.pcap.csv",
           "ARP_Spoofing_test.pcap.csv", "TCP_IP-DoS-ICMP_test.pcap.csv"]


async def device_worker(client, url, device, rows, queue, lat, errors, api_key):
    headers = {"X-API-Key": api_key} if api_key else {}
    i = 0
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        body = {"session_id": f"load-{device}-{i}", "device_id": device,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "cert_valid": True, "behavior_score": 0.9, "access_decision": "PERMIT",
                "features": [float(v) for v in rows[i % len(rows)]]}
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=body, headers=headers)
            r.raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
        i += 1


async def run(a):
    devices = list(DEVICES)[: a.concurrency]
    data = {d: read_features(C.TEST_DIR / STREAMS[i % len(STREAMS)])[:2000] for i, d in enumerate(devices)}
    queue = asyncio.Queue()
    for _ in range(a.events):
        queue.put_nowait(1)
    lat, errors = [], []
    async with httpx.AsyncClient(timeout=10) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[device_worker(client, a.url, d, data[d], queue, lat, errors, a.api_key)
                               for d in devices])
        total = time.perf_counter() - t0
    return np.array(lat), errors, total, devices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=10_000)
    ap.add_argument("--concurrency", type=int, default=7, help="parallel devices (max 7)")
    ap.add_argument("--url", default="http://127.0.0.1:8002/classify")
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args()
    a.concurrency = min(a.concurrency, len(DEVICES))

    lat, errors, total, devices = asyncio.run(run(a))
    res = {
        "events_sent": a.events, "succeeded": int(len(lat)), "failed": len(errors),
        "success_rate_pct": round(100 * len(lat) / a.events, 3),
        "devices": devices, "wall_time_s": round(total, 1),
        "throughput_events_per_s": round(len(lat) / total, 1),
        "capacity_events_per_day": int(len(lat) / total * 86400),
        "latency_ms": {"mean": round(float(lat.mean()), 2), "p50": round(float(np.percentile(lat, 50)), 2),
                       "p95": round(float(np.percentile(lat, 95)), 2), "p99": round(float(np.percentile(lat, 99)), 2),
                       "max": round(float(lat.max()), 2)},
        "nfr1_p95_under_100ms": bool(np.percentile(lat, 95) < C.MAX_LATENCY_MS),
        "nfr4_10k_events_per_day": bool(len(lat) / total * 86400 >= 10_000),
    }
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "load_test.json").write_text(json.dumps(res, indent=2))

    plt.figure(figsize=(8, 4))
    plt.hist(lat, bins=60)
    for q, ls in ((50, "-"), (95, "--"), (99, ":")):
        plt.axvline(np.percentile(lat, q), ls=ls, c="k", label=f"p{q} = {np.percentile(lat, q):.1f} ms")
    plt.axvline(C.MAX_LATENCY_MS, c="red", label="NFR1 limit 100 ms")
    plt.xlabel("end-to-end latency per event (ms)")
    plt.ylabel("events")
    plt.title(f"C2 API load test: {len(lat):,} events, {a.concurrency} devices in parallel")
    plt.legend()
    plt.tight_layout()
    plt.savefig(C.RESULTS_DIR / "load_test_latency.png", dpi=150)

    print(json.dumps(res, indent=2))
    if errors:
        print("first errors:", errors[:3])


if __name__ == "__main__":
    main()