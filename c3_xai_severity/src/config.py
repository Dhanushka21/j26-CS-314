"""Central settings for Component 3. Override any value with an environment variable."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the detector model lives. In development this is the stand-in model
# trained by train_standin.py. For integration, point it at Component 2's model.
MODEL_PATH = os.environ.get("C3_MODEL_PATH", os.path.join(BASE_DIR, "models", "detector.joblib"))
SEVERITY_MODEL_PATH = os.environ.get("C3_SEVERITY_MODEL_PATH", os.path.join(BASE_DIR, "models", "severity.joblib"))

# Component 4 (Dulsara) endpoint. He will share the real IP before the demo.
COMPONENT4_URL = os.environ.get("C3_COMPONENT4_URL", "http://127.0.0.1:5000/api/v1/alert")
COMPONENT4_TIMEOUT_S = float(os.environ.get("C3_COMPONENT4_TIMEOUT_S", "2.0"))

# Our own service port. Must not be 5000 (Component 4 uses it).
PORT = int(os.environ.get("C3_PORT", "8000"))

TOP_K_FEATURES = int(os.environ.get("C3_TOP_K", "3"))

# "rule" = rule-based taxonomy, "ml" = trained severity classifier (falls back to rule if no model file)
SEVERITY_MODE = os.environ.get("C3_SEVERITY_MODE", "rule")

# Optional policy: escalate High -> Critical when detection confidence is at least this value.
# Leave as None until the team and supervisor agree on it.
_esc = os.environ.get("C3_ESCALATE_CONF", "")
ESCALATE_CONF = float(_esc) if _esc else None

# CICIoMT2024 CSVs have flow features and a label, but (as far as we know) no device column.
# So device types are assigned here. Replace with your simulated hospital's real device list.
DEVICE_REGISTRY = {
    "IV-Pump-01": "infusion_pump",
    "IV-Pump-04": "infusion_pump",
    "Vent-01": "ventilator",
    "Cardiac-Mon-02": "cardiac_monitor",
    "ECG-03": "ecg_machine",
    "Wearable-07": "wearable_sensor",
    "BP-Cuff-05": "bp_monitor",
}

LIFE_CRITICAL_TYPES = {"infusion_pump", "ventilator", "cardiac_monitor", "ecg_machine", "icd"}

# Fail-safe choice: if a device_id is not in the registry, should it be treated as life-critical?
# True is safer for patients but weakens mitigation on unknown devices. Decide with your team.
UNKNOWN_DEVICE_IS_CRITICAL = os.environ.get("C3_UNKNOWN_CRITICAL", "false").lower() == "true"

BENIGN_LABEL = "Benign"
