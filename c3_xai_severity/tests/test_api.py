import joblib
import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from train_standin import CLASSES, synthetic
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    df = synthetic(n_per_class=40)
    X, y = df.drop(columns=["label"]), df["label"]
    model = RandomForestClassifier(n_estimators=15, max_depth=8, random_state=0).fit(X, y)
    path = tmp_path_factory.mktemp("m") / "detector.joblib"
    joblib.dump(model, path)
    client = TestClient(create_app(model_path=str(path), forward=False, severity_mode="rule"))
    return client, df


def payload(df, cls, device, conf=0.95):
    row = df[df["label"] == cls].iloc[0].drop("label")
    return {"device_id": device, "timestamp": "2026-10-18T10:00:00Z",
            "feature_vector": [float(v) for v in row], "predicted_class": cls, "confidence": conf}


def test_health(setup):
    client, _ = setup
    assert client.get("/health").json()["features"] == 46


def test_alert_has_agreed_schema(setup):
    client, df = setup
    r = client.post("/api/v1/detection", json=payload(df, "TCP_IP-DoS-SYN", "Wearable-07")).json()
    a = r["alert"]
    assert set(a) == {"device_id", "timestamp", "attack_type", "confidence", "severity",
                      "critical_protection_flag", "shap_top_features", "explanation_summary"}
    assert "recommended_action" not in a
    assert a["severity"] == "High" and a["critical_protection_flag"] is False
    assert len(a["shap_top_features"]) == 3
    assert set(a["shap_top_features"][0]) == {"feature", "contribution"}


def test_life_critical_device_gets_critical_flag(setup):
    client, df = setup
    a = client.post("/api/v1/detection", json=payload(df, "Recon-Ping_Sweep", "Vent-01")).json()["alert"]
    assert a["severity"] == "Critical" and a["critical_protection_flag"] is True


def test_benign_is_skipped(setup):
    client, df = setup
    r = client.post("/api/v1/detection", json=payload(df, "Benign", "Vent-01")).json()
    assert r["status"] == "skipped_benign" and "alert" not in r


def test_bad_inputs_are_rejected(setup):
    client, df = setup
    bad = payload(df, "TCP_IP-DoS-SYN", "Vent-01")
    bad["feature_vector"] = bad["feature_vector"][:10]
    assert client.post("/api/v1/detection", json=bad).status_code == 422
    bad2 = payload(df, "TCP_IP-DoS-SYN", "Vent-01", conf=1.7)
    assert client.post("/api/v1/detection", json=bad2).status_code == 422
    bad3 = payload(df, "TCP_IP-DoS-SYN", "Vent-01")
    bad3["predicted_class"] = "Made_Up_Attack"
    assert client.post("/api/v1/detection", json=bad3).status_code == 422


def test_latency_under_100ms(setup):
    client, df = setup
    p = payload(df, "TCP_IP-DDoS-SYN", "Wearable-07")
    client.post("/api/v1/detection", json=p)  # warm-up
    times = [client.post("/api/v1/detection", json=p).json()["processing_ms"] for _ in range(20)]
    print("processing_ms:", sorted(times))
    assert sorted(times)[int(len(times) * 0.9)] < 100
