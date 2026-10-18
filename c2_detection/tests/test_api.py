"""API contract + latency tests.   Run from repo root:   pytest -q c2_detection/tests"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "c2_detection"))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from contracts.models import CLASSES, C2ToC3  # noqa: E402

client = TestClient(app)
SAMPLE = json.loads((ROOT / "contracts" / "samples" / "c1_to_c2.json").read_text())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_classify_matches_contract():
    r = client.post("/classify", json=SAMPLE)
    assert r.status_code == 200, r.text
    out = C2ToC3(**r.json())                         # validates against the shared contract
    assert out.predicted_class in CLASSES
    assert set(out.probabilities) == set(CLASSES)     # all 19 classes present (payload completeness)
    assert abs(sum(out.probabilities.values()) - 1) < 1e-3
    assert len(out.features) == 45
    assert out.device_id == SAMPLE["device_id"]
    assert out.timestamp == SAMPLE["timestamp"]


def test_wrong_feature_count_rejected():
    assert client.post("/classify", json=dict(SAMPLE, features=[0.0] * 10)).status_code == 422


def test_revoked_session_rejected():
    assert client.post("/classify", json=dict(SAMPLE, access_decision="REVOKE")).status_code == 403


def test_latency_under_100ms():
    client.post("/classify", json=SAMPLE)            # warm
    t0 = time.perf_counter()
    for _ in range(50):
        client.post("/classify", json=SAMPLE)
    avg_ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"avg latency {avg_ms:.1f} ms")
    assert avg_ms < 100