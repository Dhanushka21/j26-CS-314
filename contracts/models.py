"""Shared data contracts for IoMT Shield (J-26-CS-314). Pydantic v2.
Every component imports these models. Do NOT change field names without telling all 4 members."""
import json
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

_HERE = Path(__file__).parent
CLASSES: list[str] = json.loads((_HERE / "classes.json").read_text())["classes"]
FEATURES: list[str] = json.loads((_HERE / "features.json").read_text())["features"]
DEVICES: dict = {d["device_id"]: d for d in json.loads((_HERE / "devices.json").read_text())["devices"]}
N_FEATURES = len(FEATURES)  # 45

Severity = Literal["Critical", "High", "Medium", "Low"]
Action = Literal["DROP", "REDIRECT", "METER", "LOG"]


class _FeatureVectorMixin(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def _len(cls, v):
        if len(v) != N_FEATURES:
            raise ValueError(f"features must have {N_FEATURES} values in features.json order")
        return v


# C1 -> C2 : verified session / flow handed to detection
class C1ToC2(_FeatureVectorMixin):
    session_id: str
    device_id: str
    timestamp: str  # ISO 8601, e.g. 2026-10-14T10:00:00Z
    cert_valid: bool
    behavior_score: float = Field(ge=0, le=1)
    access_decision: Literal["PERMIT", "RESTRICT", "REVOKE"]


# C2 -> C3 : classification result
class C2ToC3(_FeatureVectorMixin):
    device_id: str
    timestamp: str
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    source_ip: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.predicted_class not in CLASSES:
            raise ValueError(f"unknown class {self.predicted_class}")
        bad = [k for k in self.probabilities if k not in CLASSES]
        if bad:
            raise ValueError(f"unknown classes in probabilities: {bad}")
        return self


# C1 -> C3 : identity compromise event
class C1ToC3(BaseModel):
    device_id: str
    timestamp: str
    alert_type: Literal["IDENTITY_COMPROMISE"]
    cert_valid: bool
    behavior_score: float = Field(ge=0, le=1)
    source_ip: str


class ShapFeature(BaseModel):
    feature: str
    shap_value: float


# C3 -> C4 : severity-annotated alert (the 9 fields in the C4 proposal, plus optional source_ip)
class C3ToC4(BaseModel):
    device_id: str
    timestamp: str
    attack_type: str
    confidence: float = Field(ge=0, le=1)
    severity: Severity
    critical_protection_flag: bool
    shap_top_features: list[ShapFeature]
    explanation_summary: str
    recommended_action: Action
    source_ip: Optional[str] = None  # needed by C4 for source-IP matching rules


# C1 -> C4 : lateral movement (bypasses C3, handled as Critical)
class C1ToC4(BaseModel):
    alert_type: Literal["LATERAL_MOVEMENT"]
    source_ip: str
    attempted_destination: str
    device_name: str
    timestamp: Optional[str] = None
