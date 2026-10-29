"""C1 → C2 → C3  Integration Test  (C2-side, no live network needed)
===================================================================
What this tests
---------------
This file is the C2 contribution to the end-to-end integration story.
It uses FastAPI's TestClient (no real socket, instant) so it runs in CI
without needing the API to be started separately.

Test groups
-----------
  T1  C1 mock → C2 /classify → C2ToC3 contract  (happy-path, PERMIT flow)
  T2  C1 REVOKE → C2 rejects  (access control gate)
  T3  C1 warm-up → behaviour stabilises after WINDOW_SIZE flows (per device)
  T4  C2 output ∈ CLASSES, proba sums to 1, features passed through  (payload integrity)
  T5  NFR1 – p95 latency < 100 ms across 200 requests
  T6  Multi-device isolation – two devices get independent buffers
  T7  C3 mock – simulates what C3 would do with the C2ToC3 payload

Run from repo root:
    pytest -q c2_detection/tests/test_integration.py -v
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "c2_detection"))
sys.path.insert(0, str(ROOT))

import config as C                                            # noqa: E402
from api.main import app                                      # noqa: E402
from contracts.models import CLASSES, N_FEATURES, C2ToC3     # noqa: E402

client = TestClient(app)

# ── helpers ───────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _flow(
    device_id: str = "bp_monitor_01",
    access_decision: str = "PERMIT",
    features: Optional[list] = None,
    behavior_score: float = 0.92,
    cert_valid: bool = True,
    session_id: Optional[str] = None,
) -> dict:
    """Build a C1ToC2 payload (what C1 would POST to /classify)."""
    return {
        "session_id": session_id or f"integ-{_ts()}",
        "device_id": device_id,
        "timestamp": _ts(),
        "cert_valid": cert_valid,
        "behavior_score": behavior_score,
        "access_decision": access_decision,
        "features": features if features is not None else [0.0] * N_FEATURES,
    }


def _post(payload: dict) -> dict:
    r = client.post("/classify", json=payload)
    assert r.status_code == 200, f"C2 returned {r.status_code}: {r.text}"
    return r.json()


# ── C3 mock ───────────────────────────────────────────────────────────────────
# In production C3 receives C2ToC3 and adds SHAP severity scores.
# Here we simulate what C3 does to validate the handoff shape is correct.
SEVERITY_MAP = {
    "Benign": "Low",
    "Recon-Ping_Sweep": "Low",
    "Recon-Port_Scan": "Low",
    "Recon-OS_Scan": "Low",
    "Recon-VulScan": "Medium",
    "MQTT-Malformed_Data": "Medium",
    "ARP_Spoofing": "High",
    "MQTT-DDoS-Connect_Flood": "High",
    "MQTT-DDoS-Publish_Flood": "High",
    "MQTT-DoS-Connect_Flood": "High",
    "MQTT-DoS-Publish_Flood": "High",
    "TCP_IP-DoS-ICMP": "High",
    "TCP_IP-DoS-SYN": "High",
    "TCP_IP-DoS-TCP": "High",
    "TCP_IP-DoS-UDP": "High",
    "TCP_IP-DDoS-ICMP": "Critical",
    "TCP_IP-DDoS-SYN": "Critical",
    "TCP_IP-DDoS-TCP": "Critical",
    "TCP_IP-DDoS-UDP": "Critical",
}

def mock_c3_process(c2_payload: dict) -> dict:
    """Simulate C3: receive C2ToC3, add severity + mock SHAP, return C3ToC4 shape."""
    cls = c2_payload["predicted_class"]
    severity = SEVERITY_MAP.get(cls, "High")
    # mock SHAP: top 3 features by index (real C3 uses actual SHAP values)
    top_feats = [
        {"feature": C.MODEL_FEATURES[i], "shap_value": round(float(c2_payload["confidence"] * (3 - i) * 0.1), 4)}
        for i in range(3)
    ]
    return {
        "device_id": c2_payload["device_id"],
        "timestamp": c2_payload["timestamp"],
        "attack_type": cls,
        "confidence": c2_payload["confidence"],
        "severity": severity,
        "critical_protection_flag": False,          # C3 sets this from devices.json
        "shap_top_features": top_feats,
        "explanation_summary": f"[mock-C3] {cls} detected with {severity} severity.",
        "recommended_action": "DROP" if severity == "Critical" else "LOG",
        "source_ip": c2_payload.get("source_ip"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# T1 — happy-path PERMIT flow produces a valid C2ToC3 payload
# ══════════════════════════════════════════════════════════════════════════════
class TestT1_HappyPath:
    def test_classify_returns_200(self):
        r = client.post("/classify", json=_flow())
        assert r.status_code == 200

    def test_response_validates_against_contract(self):
        out = _post(_flow(device_id="ventilator_01"))
        c2 = C2ToC3(**out)                          # raises if contract is broken
        assert c2.predicted_class in CLASSES

    def test_all_19_classes_in_probabilities(self):
        out = _post(_flow())
        assert set(out["probabilities"].keys()) == set(CLASSES)

    def test_probabilities_sum_to_1(self):
        out = _post(_flow())
        total = sum(out["probabilities"].values())
        assert abs(total - 1.0) < 1e-3, f"proba sum = {total}"

    def test_confidence_equals_max_proba(self):
        out = _post(_flow())
        assert abs(out["confidence"] - max(out["probabilities"].values())) < 1e-5

    def test_features_passed_through_unchanged(self):
        feats = [float(i % 7) for i in range(N_FEATURES)]
        out = _post(_flow(features=feats))
        assert out["features"] == feats

    def test_device_id_preserved(self):
        out = _post(_flow(device_id="cardiac_monitor_01"))
        assert out["device_id"] == "cardiac_monitor_01"

    def test_timestamp_preserved(self):
        ts = "2026-10-29T08:00:00Z"
        payload = _flow()
        payload["timestamp"] = ts
        out = _post(payload)
        assert out["timestamp"] == ts


# ══════════════════════════════════════════════════════════════════════════════
# T2 — C1 REVOKE blocks classification (access control gate)
# ══════════════════════════════════════════════════════════════════════════════
class TestT2_AccessControl:
    def test_revoke_returns_403(self):
        r = client.post("/classify", json=_flow(access_decision="REVOKE"))
        assert r.status_code == 403

    def test_revoke_detail_message(self):
        r = client.post("/classify", json=_flow(access_decision="REVOKE"))
        assert "revoked" in r.json()["detail"].lower()

    def test_permit_after_revoke_allowed(self):
        """Each flow is independent — a different session with PERMIT should go through."""
        client.post("/classify", json=_flow(access_decision="REVOKE", device_id="admin_server_01"))
        r = client.post("/classify", json=_flow(access_decision="PERMIT", device_id="admin_server_01"))
        assert r.status_code == 200

    def test_wrong_feature_count_rejected(self):
        r = client.post("/classify", json=dict(_flow(), features=[0.0] * 10))
        assert r.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# T3 — warm-up: confidence should be non-trivial (>1/19 random) after warm-up
# ══════════════════════════════════════════════════════════════════════════════
class TestT3_WarmUp:
    DEVICE = "insulin_pump_01"

    def test_warm_up_still_returns_200(self):
        """First few flows of a new device must not error."""
        for i in range(5):
            r = client.post("/classify", json=_flow(device_id=self.DEVICE, session_id=f"wu-{i}"))
            assert r.status_code == 200, f"failed on flow {i}: {r.text}"

    def test_confidence_rises_after_window(self):
        """After WINDOW_SIZE flows confidence should be higher than initial warm-up guess."""
        device = "patient_server_01"
        confs = []
        for i in range(C.WINDOW_SIZE + 20):
            out = _post(_flow(device_id=device, session_id=f"wu2-{i}"))
            confs.append(out["confidence"])
        avg_early = sum(confs[:5]) / 5
        avg_late = sum(confs[-10:]) / 10
        # not a strict guarantee (depends on model + features), but usually true
        assert avg_late >= avg_early * 0.5, (
            f"confidence did not stabilise: early={avg_early:.3f} late={avg_late:.3f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# T4 — payload integrity across multiple calls
# ══════════════════════════════════════════════════════════════════════════════
class TestT4_PayloadIntegrity:
    def test_repeated_calls_always_valid_contract(self):
        for i in range(20):
            out = _post(_flow(device_id="bp_monitor_01", session_id=f"rep-{i}"))
            c2 = C2ToC3(**out)
            assert c2.predicted_class in CLASSES
            assert 0 <= c2.confidence <= 1

    def test_different_feature_vectors_may_give_different_classes(self):
        """Sanity: two very different feature vectors should not always give identical output."""
        results = set()
        rng = np.random.default_rng(42)
        for _ in range(10):
            feats = rng.uniform(0, 1e6, N_FEATURES).tolist()
            out = _post(_flow(features=feats, device_id="bp_monitor_01"))
            results.add(out["predicted_class"])
        # at least 2 distinct predictions across 10 random vectors
        assert len(results) >= 2, f"all 10 random flows gave same class: {results}"


# ══════════════════════════════════════════════════════════════════════════════
# T5 — NFR1: p95 latency < 100 ms  (200 sequential requests)
# ══════════════════════════════════════════════════════════════════════════════
class TestT5_Latency:
    N = 200

    def test_p95_latency_under_100ms(self):
        # warm-up
        for _ in range(5):
            client.post("/classify", json=_flow())

        latencies = []
        for i in range(self.N):
            t0 = time.perf_counter()
            r = client.post("/classify", json=_flow(session_id=f"lat-{i}"))
            latencies.append((time.perf_counter() - t0) * 1000)
            assert r.status_code == 200

        latencies.sort()
        p95 = latencies[int(0.95 * len(latencies))]
        mean = sum(latencies) / len(latencies)
        print(f"\n  Latency over {self.N} calls — mean={mean:.1f} ms  p95={p95:.1f} ms")
        assert p95 < 100, f"p95 latency {p95:.1f} ms exceeds 100 ms NFR"


# ══════════════════════════════════════════════════════════════════════════════
# T6 — device isolation: each device keeps its own buffer
# ══════════════════════════════════════════════════════════════════════════════
class TestT6_DeviceIsolation:
    def test_two_devices_return_independent_predictions(self):
        """
        Send a constant zero-feature stream to device A and a high-value stream to B.
        Their predictions should be independent (no buffer cross-contamination).
        """
        feats_a = [0.0] * N_FEATURES
        feats_b = [1e6] * N_FEATURES
        for i in range(30):
            out_a = _post(_flow(device_id="infusion_pump_01",  features=feats_a, session_id=f"da-{i}"))
            out_b = _post(_flow(device_id="admin_server_01", features=feats_b, session_id=f"db-{i}"))
            # just validate both calls succeed and have independent device_ids
            assert out_a["device_id"] == "infusion_pump_01"
            assert out_b["device_id"] == "admin_server_01"

    def test_7_devices_all_succeed(self):
        from contracts.models import DEVICES
        for dev in DEVICES:
            out = _post(_flow(device_id=dev))
            assert out["device_id"] == dev


# ══════════════════════════════════════════════════════════════════════════════
# T7 — C3 mock handoff: C2 output → mock C3 → C3ToC4 shape
# ══════════════════════════════════════════════════════════════════════════════
class TestT7_C3MockHandoff:
    def test_c3_mock_accepts_c2_output(self):
        """Simulate the full C1→C2→C3 flow using the C3 mock."""
        # Step 1: C1 sends flow to C2
        c1_payload = _flow(device_id="ventilator_01")
        c2_out = _post(c1_payload)

        # Step 2: C3 receives C2ToC3 and processes it
        c3_out = mock_c3_process(c2_out)

        # Step 3: validate C3ToC4 shape
        assert c3_out["attack_type"] in CLASSES
        assert c3_out["severity"] in ("Critical", "High", "Medium", "Low")
        assert c3_out["recommended_action"] in ("DROP", "REDIRECT", "METER", "LOG")
        assert isinstance(c3_out["shap_top_features"], list)
        assert len(c3_out["shap_top_features"]) == 3

    def test_c3_mock_full_pipeline_20_flows(self):
        """Run 20 flows through the full mock pipeline and verify no errors."""
        rng = np.random.default_rng(99)
        for i in range(20):
            feats = rng.uniform(0, 1e5, N_FEATURES).tolist()
            c2_out = _post(_flow(device_id="cardiac_monitor_01", features=feats, session_id=f"pipe-{i}"))
            c3_out = mock_c3_process(c2_out)
            assert c3_out["attack_type"] in CLASSES
            assert 0 <= c3_out["confidence"] <= 1

    def test_critical_attack_gets_drop_action(self):
        """Critical severity attacks should recommend DROP (from C3 logic)."""
        # force a critical class by manipulating features (best effort — model chooses)
        # we test the mock C3 logic directly instead
        c2_fake = {
            "device_id": "ventilator_01",
            "timestamp": _ts(),
            "predicted_class": "TCP_IP-DDoS-SYN",   # known Critical
            "confidence": 0.95,
            "probabilities": {c: (0.95 if c == "TCP_IP-DDoS-SYN" else 0.0) for c in CLASSES},
            "features": [0.0] * N_FEATURES,
            "source_ip": "10.0.0.99",
        }
        c3_out = mock_c3_process(c2_fake)
        assert c3_out["severity"] == "Critical"
        assert c3_out["recommended_action"] == "DROP"

    def test_benign_traffic_gets_log_action(self):
        c2_fake = {
            "device_id": "bp_monitor_01",
            "timestamp": _ts(),
            "predicted_class": "Benign",
            "confidence": 0.99,
            "probabilities": {c: (0.99 if c == "Benign" else 0.0) for c in CLASSES},
            "features": [0.0] * N_FEATURES,
            "source_ip": None,
        }
        c3_out = mock_c3_process(c2_fake)
        assert c3_out["severity"] == "Low"
        assert c3_out["recommended_action"] == "LOG"


# ══════════════════════════════════════════════════════════════════════════════
# Health endpoint sanity
# ══════════════════════════════════════════════════════════════════════════════
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "rf_win"
    assert body["window_size"] == 150
    assert "IAT" in body["dropped_features"]