import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.app import create_app
from src.contract_loader import FEATURES, sample
from train_standin import synthetic

ALERT_FIELDS = {"device_id", "timestamp", "attack_type", "confidence", "severity", "critical_protection_flag",
                "shap_top_features", "explanation_summary", "source_ip"}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    df = synthetic(n_per_class=40)
    model = RandomForestClassifier(n_estimators=15, max_depth=8, random_state=0).fit(df[FEATURES], df["label"])
    path = tmp_path_factory.mktemp("m") / "detector.joblib"
    joblib.dump(model, path)
    return TestClient(create_app(model_path=str(path), forward=False, severity_mode="rule"))


def c2(device="bp_monitor_01", cls="TCP_IP-DoS-SYN", conf=0.95, **extra):
    p = sample("c2_to_c3")
    p.update(device_id=device, predicted_class=cls, confidence=conf, **extra)
    return p


def test_health(setup):
    assert setup.get("/health").json()["features"] == 45


def test_group_sample_c2_to_c3_works(setup):
    r = setup.post("/analyze", json=sample("c2_to_c3")).json()     # ventilator_01, DDoS-SYN
    a = r["alert"]
    assert r["status"] == "processed"
    assert a["severity"] == "Critical" and a["critical_protection_flag"] is True
    assert a["source_ip"] == "10.0.0.5"


def test_alert_has_agreed_schema(setup):
    a = setup.post("/analyze", json=c2()).json()["alert"]
    assert set(a) == ALERT_FIELDS
    assert "recommended_action" not in a
    assert a["severity"] == "High" and a["critical_protection_flag"] is False
    assert len(a["shap_top_features"]) == 3
    assert set(a["shap_top_features"][0]) == {"feature", "shap_value"}
    assert all(f["feature"] in FEATURES for f in a["shap_top_features"])


def test_life_critical_device_gets_critical_flag(setup):
    a = setup.post("/analyze", json=c2("insulin_pump_01", "Recon-Ping_Sweep")).json()["alert"]
    assert a["severity"] == "Critical" and a["critical_protection_flag"] is True


def test_cardiac_monitor_is_not_life_critical_per_contracts(setup):
    a = setup.post("/analyze", json=c2("cardiac_monitor_01", "Recon-Ping_Sweep")).json()["alert"]
    assert a["severity"] == "Low" and a["critical_protection_flag"] is False


def test_benign_is_skipped(setup):
    r = setup.post("/analyze", json=c2(cls="Benign")).json()
    assert r["status"] == "skipped_benign" and "alert" not in r


def test_identity_compromise_from_c1(setup):
    r = setup.post("/analyze", json=sample("c1_to_c3")).json()     # bp_monitor_01
    a = r["alert"]
    assert a["attack_type"] == "IDENTITY_COMPROMISE" and a["source_ip"] == "10.0.0.7"
    assert a["shap_top_features"] == [] and a["critical_protection_flag"] is False
    life = setup.post("/analyze", json={**sample("c1_to_c3"), "device_id": "ventilator_01"}).json()["alert"]
    assert life["severity"] == "Critical" and life["critical_protection_flag"] is True


def test_bad_inputs_are_rejected(setup):
    short = c2()
    short["features"] = short["features"][:10]
    assert setup.post("/analyze", json=short).status_code == 422
    assert setup.post("/analyze", json=c2(conf=1.7)).status_code == 422
    assert setup.post("/analyze", json=c2(cls="Made_Up_Attack")).status_code == 422
    assert setup.post("/analyze", json={"hello": "world"}).status_code == 422


def test_latency_under_100ms(setup):
    p = c2("bp_monitor_01", "TCP_IP-DDoS-SYN")
    setup.post("/analyze", json=p)  # warm-up
    times = sorted(setup.post("/analyze", json=p).json()["processing_ms"] for _ in range(20))
    print("processing_ms:", times)
    assert times[int(len(times) * 0.9)] < 100
