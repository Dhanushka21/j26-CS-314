"""Central settings for Component 3. Override any value with an environment variable."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Detector model used for SHAP. In development this is the stand-in trained by train_standin.py.
# For integration, point it at Component 2's model (same 45 features, same 19 classes).
MODEL_PATH = os.environ.get("C3_MODEL_PATH", os.path.join(BASE_DIR, "models", "detector.joblib"))
SEVERITY_MODEL_PATH = os.environ.get("C3_SEVERITY_MODEL_PATH", os.path.join(BASE_DIR, "models", "severity.joblib"))

# Component 4 (Dulsara). He confirmed port 5000 and /api/v1/alert; the contracts README still says
# 8004 /alert, so ask Dhanushka to update it. Replace 127.0.0.1 with his IP before the demo.
COMPONENT4_URL = os.environ.get("C3_COMPONENT4_URL", "http://127.0.0.1:5000/api/v1/alert")
COMPONENT4_TIMEOUT_S = float(os.environ.get("C3_COMPONENT4_TIMEOUT_S", "2.0"))

# Our own port, from the contracts README table (C3 = 8003, POST /analyze).
PORT = int(os.environ.get("C3_PORT", "8003"))

TOP_K_FEATURES = int(os.environ.get("C3_TOP_K", "3"))

# "rule" = rule-based taxonomy, "ml" = trained severity classifier (falls back to rule if no model file)
SEVERITY_MODE = os.environ.get("C3_SEVERITY_MODE", "rule")

# Optional policy: escalate High -> Critical when detection confidence is at least this value.
# Leave unset until the team and supervisor agree on it.
_esc = os.environ.get("C3_ESCALATE_CONF", "")
ESCALATE_CONF = float(_esc) if _esc else None

# Fail-safe choice for device ids that are not in contracts/devices.json.
UNKNOWN_DEVICE_IS_CRITICAL = os.environ.get("C3_UNKNOWN_CRITICAL", "false").lower() == "true"

# C1 -> C3 identity-compromise events carry no attack class or confidence. OPEN DECISION for the group:
# the base severity used for them (life-critical devices are still forced to Critical).
IDENTITY_SEVERITY = os.environ.get("C3_IDENTITY_SEVERITY", "High")

BENIGN_LABEL = "Benign"
