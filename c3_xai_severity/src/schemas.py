from typing import List, Literal

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """What Component 2 (AI detection) sends us, one flow per request."""
    device_id: str
    timestamp: str
    feature_vector: List[float]
    predicted_class: str
    confidence: float = Field(ge=0, le=1)


class ShapFeature(BaseModel):
    feature: str
    contribution: float


class Alert(BaseModel):
    """What we send to Component 4 (SDN). Matches the schema Dulsara accepted."""
    device_id: str
    timestamp: str
    attack_type: str
    confidence: float
    severity: Literal["Critical", "High", "Medium", "Low"]
    critical_protection_flag: bool
    shap_top_features: List[ShapFeature]
    explanation_summary: str
