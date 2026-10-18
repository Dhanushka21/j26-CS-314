"""Send one test alert per row of Dulsara's mapping table, plus a malformed one.

  python send_test_alerts.py --url http://<HIS_IP>:5000/api/v1/alert
"""
import argparse

import requests

BASE = {
    "device_id": "Test-Device", "timestamp": "2026-10-18T10:00:00Z", "attack_type": "TCP_IP-DDoS-SYN",
    "confidence": 0.98,
    "shap_top_features": [{"feature": "flow_duration", "contribution": 0.31}],
    "explanation_summary": "integration test",
}
CASES = [
    ("Critical", True, "rate-limit only, NO isolation"),
    ("Critical", False, "rate-limit + block"),
    ("High", False, "rate-limit + redirect to honeypot"),
    ("Medium", False, "rate-limit"),
    ("Low", False, "monitor only"),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000/api/v1/alert")
    url = ap.parse_args().url
    for sev, flag, expected in CASES:
        r = requests.post(url, json={**BASE, "severity": sev, "critical_protection_flag": flag}, timeout=5)
        print(f"{sev:8} flag={flag!s:5} expect: {expected:35} -> HTTP {r.status_code} {r.text[:80]}")
    r = requests.post(url, json={"severity": "Critical"}, timeout=5)
    print(f"malformed payload -> HTTP {r.status_code} (check he rejects or handles it cleanly)")
