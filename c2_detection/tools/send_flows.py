"""Demo / integration client: replays real TEST rows to the running API as C1ToC2 payloads.

  1. start the API:   uvicorn c2_detection.api.main:app --port 8002
  2. other terminal:  python c2_detection/tools/send_flows.py --file TCP_IP-DoS-UDP_test.pcap.csv --n 20
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                          # noqa: E402
from src.data import label_from_filename, read_features    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="file name inside data/raw/CSV/Test/")
    ap.add_argument("--n", type=int, default=20, help="number of flows to send")
    ap.add_argument("--device", default="ventilator_01")
    ap.add_argument("--url", default="http://127.0.0.1:8002/classify")
    a = ap.parse_args()

    path = C.TEST_DIR / a.file
    truth = label_from_filename(path.name)
    rows = read_features(path)[: a.n]
    correct = 0
    with httpx.Client(timeout=5) as client:
        for i, feats in enumerate(rows):
            body = {"session_id": f"demo-{i}", "device_id": a.device,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "cert_valid": True, "behavior_score": 0.9, "access_decision": "PERMIT",
                    "features": [float(v) for v in feats]}
            t0 = time.perf_counter()
            r = client.post(a.url, json=body)
            ms = (time.perf_counter() - t0) * 1000
            out = r.json()
            ok = out["predicted_class"] == truth
            correct += ok
            print(f"{i:>3}  {'OK ' if ok else 'XX '} pred={out['predicted_class']:<26} "
                  f"conf={out['confidence']:.3f}  {ms:5.1f} ms (server {r.headers['X-Process-Time-ms']} ms)")
    print(f"\ntrue class = {truth}    correct {correct}/{len(rows)}")


if __name__ == "__main__":
    main()