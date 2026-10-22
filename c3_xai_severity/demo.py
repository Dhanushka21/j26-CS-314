"""Live demo: sends 8 scenarios through the running Component 3 and prints what happens.

Start first (3 terminals, venv active, inside c3_xai_severity):
  A: uvicorn mock_component4:app --port 5000
  B: uvicorn src.app:create_app --factory --port 8003
Then, in a third terminal:  python demo.py
(The detector is the stand-in model, so the SHAP feature names are illustrative until you train on the real CSV.)
"""
import sys
import time

import requests

from src.contract_loader import CLASSES, sample
from train_standin import synthetic

URL = "http://127.0.0.1:8003"
ROWS = synthetic(n_per_class=1, seed=7).set_index("label")

SCENARIOS = [
    ("Ventilator under SYN flood (life-critical)", "ventilator_01", "TCP_IP-DDoS-SYN", 0.97),
    ("Blood-pressure monitor under SYN flood", "bp_monitor_01", "TCP_IP-DDoS-SYN", 0.96),
    ("Patient server, single-source DoS", "patient_server_01", "TCP_IP-DoS-SYN", 0.91),
    ("Admin server, OS scan", "admin_server_01", "Recon-OS_Scan", 0.88),
    ("BP monitor, ping sweep (mild attack)", "bp_monitor_01", "Recon-Ping_Sweep", 0.84),
    ("Insulin pump, SAME mild ping sweep", "insulin_pump_01", "Recon-Ping_Sweep", 0.84),
    ("Normal traffic (Benign)", "bp_monitor_01", "Benign", 0.99),
]


def c2_payload(device, cls, conf):
    base = sample("c2_to_c3")
    probs = {c: 0.0 for c in CLASSES}
    probs[cls] = conf
    probs["Benign" if cls != "Benign" else "TCP_IP-DoS-SYN"] = round(1 - conf, 2)
    base.update(device_id=device, predicted_class=cls, confidence=conf, probabilities=probs,
                features=[float(v) for v in ROWS.loc[cls]])
    return base


def show(title, payload):
    r = requests.post(f"{URL}/analyze", json=payload, timeout=10)
    out = r.json()
    print(f"\n{'=' * 78}\n{title}\n{'-' * 78}")
    if out.get("status") == "skipped_benign":
        print("  Component 3 : Benign traffic, nothing sent to Component 4")
        return
    a = out["alert"]
    flag = "YES (life-critical device)" if a["critical_protection_flag"] else "no"
    print(f"  Device      : {a['device_id']}")
    print(f"  Detected    : {a['attack_type']} (confidence {a['confidence']})")
    print(f"  SEVERITY    : {a['severity'].upper()}   protection flag: {flag}")
    for f in a["shap_top_features"]:
        print(f"  SHAP        : {f['feature']:<16} {f['shap_value']:+.4f}")
    print(f"  Why         : {a['explanation_summary']}")
    c4 = out.get("component4_response") or {}
    print(f"  Component 4 : {c4.get('action', 'no reply (' + str(out['forward_error']) + ')')}")
    print(f"  Time        : {out['processing_ms']} ms")


def main():
    try:
        requests.get(f"{URL}/health", timeout=3).raise_for_status()
    except requests.RequestException:
        sys.exit(f"Component 3 is not running on {URL}. Start it first (see the top of this file).")
    for title, dev, cls, conf in SCENARIOS:
        show(title, c2_payload(dev, cls, conf))
        time.sleep(0.4)
    show("Zero-Trust layer reports a stolen identity (Component 1)", sample("c1_to_c3"))
    print(f"\n{'=' * 78}\nDone. Watch the mock Component 4 terminal: it printed one line per alert.")


if __name__ == "__main__":
    main()
