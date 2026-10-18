"""Severity decision: rule-based or ML, always followed by the life-critical override."""
import os
from typing import Tuple

import joblib
import pandas as pd

from . import config, taxonomy


def ml_features(attack_type: str, confidence: float, shap_mag: float, margin: float,
                life_critical: bool) -> pd.DataFrame:
    """One row in the same column layout used by train_severity.py.
    margin = top probability minus second probability from Component 2 (low = the detector was torn)."""
    row = {f"attack__{a}": 1 if a == attack_type else 0 for a in taxonomy.ATTACK_SEVERITY}
    row.update({"confidence": confidence, "shap_magnitude": shap_mag, "prob_margin": margin,
                "life_critical": int(life_critical)})
    return pd.DataFrame([row])


class SeverityEngine:
    def __init__(self, mode: str = None, model_path: str = None):
        self.mode = mode or config.SEVERITY_MODE
        self.model = None
        path = model_path or config.SEVERITY_MODEL_PATH
        if self.mode == "ml" and os.path.exists(path):
            self.model = joblib.load(path)
        elif self.mode == "ml":
            self.mode = "rule"  # no trained model yet: fall back so the service still works

    def decide(self, attack_type: str, confidence: float, shap_mag: float, device_id: str,
               margin: float = 1.0) -> Tuple[str, bool]:
        life_critical = taxonomy.device_is_life_critical(device_id)
        if self.mode == "ml" and self.model is not None:
            X = ml_features(attack_type, confidence, shap_mag, margin, life_critical)
            X = X.reindex(columns=self.model.feature_names_in_, fill_value=0)
            severity = str(self.model.predict(X)[0])
        else:
            severity = taxonomy.rule_severity(attack_type, confidence, config.ESCALATE_CONF)
        return taxonomy.apply_override(severity, life_critical)  # fail-safe always runs last

    def decide_identity(self, device_id: str) -> Tuple[str, bool]:
        """C1 identity-compromise events: base severity is a group decision (config.IDENTITY_SEVERITY)."""
        return taxonomy.apply_override(config.IDENTITY_SEVERITY, taxonomy.device_is_life_critical(device_id))
