"""Send the group's own sample payloads (contracts/samples) to the running Component 3 service.

  python send_detection.py            # service on http://127.0.0.1:8003
  python send_detection.py --url http://127.0.0.1:8003/analyze
"""
import argparse

import requests

from src.contract_loader import sample

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8003/analyze")
url = ap.parse_args().url

# 1) Component 2 -> 3: a ventilator under DDoS-SYN (life-critical device)
vent = sample("c2_to_c3")
# 2) Same attack on a non-critical device
bp = {**vent, "device_id": "bp_monitor_01"}
# 3) Component 1 -> 3: identity compromise
c1 = sample("c1_to_c3")

for name, payload in [("ventilator_01 (C2->C3)", vent), ("bp_monitor_01 (C2->C3)", bp), ("identity (C1->C3)", c1)]:
    r = requests.post(url, json=payload, timeout=10)
    print(name, "->", r.status_code)
    print("   ", r.json())
