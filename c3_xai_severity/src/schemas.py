"""Our outgoing alert. Incoming payloads use the group's C2ToC3 / C1ToC3 models directly.

contracts.models.C3ToC4 still lists `recommended_action` as required. Component 3 and Component 4 agreed to
drop it (Component 4 decides the action), so our Alert is C3ToC4 minus that field. tests/test_contracts.py
checks the two stay in step.
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from .contract_loader import ShapFeature, Severity


class Alert(BaseModel):
    device_id: str
    timestamp: str
    attack_type: str
    confidence: float = Field(ge=0, le=1)
    severity: Severity
    critical_protection_flag: bool
    shap_top_features: List[ShapFeature]
    explanation_summary: str
    source_ip: Optional[str] = None
