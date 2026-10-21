"""
C1 Function 2 - graduated decision engine with clinical safety constraint.

Takes the credential result, the behavioural score from Function 1, and the
device's criticality tier, and produces one access decision.

The constraint: a device on the life-critical register is never fully isolated
by this component. Where the band mapping would revoke, the decision becomes
CONTAIN - the device keeps its single required clinical path, every other path
is removed, and clinical engineering is alerted.

`constraint_applied` records whether the constraint changed the outcome. That
flag is what makes the ablation in Objective 6 countable: run the identical
pipeline with `safety_constraint=False` and count the life-critical sessions
that would have been severed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[1]
_DEVICES = json.loads((_ROOT / "contracts" / "devices.json").read_text())

#: device_id -> {protection_priority, life_critical}
REGISTER: dict[str, dict] = {d["device_id"]: d for d in _DEVICES["devices"]}

Band = Literal["PERMIT", "RESTRICT", "REVOKE", "CONTAIN"]

# Thresholds come from the measured trade-off curve in out/tradeoff_*.csv.
# 0.15 is the lowest threshold that still detects 100% of spoofing windows.
PERMIT_ABOVE = 0.50
REVOKE_BELOW = 0.15


@dataclass(frozen=True)
class Decision:
    device_id: str
    band: Band
    constraint_applied: bool
    reasons: tuple[str, ...]
    retained_path: str | None = None
    escalated: bool = False

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "access_decision": self.band,
            "constraint_applied": self.constraint_applied,
            "reasons": list(self.reasons),
            "retained_path": self.retained_path,
            "escalated": self.escalated,
        }


@dataclass
class DecisionEngine:
    """FR5-FR9. Set `safety_constraint=False` to run the ablation baseline."""

    safety_constraint: bool = True
    permit_above: float = PERMIT_ABOVE
    revoke_below: float = REVOKE_BELOW
    register: dict[str, dict] = field(default_factory=lambda: REGISTER)

    # -- FR5: criticality resolution ----------------------------------------
    def resolve_tier(self, device_id: str) -> bool:
        """True if the device is on the life-critical register.

        An unknown device is treated as not life-critical: the constraint must
        never be granted by default, or an attacker could claim protection by
        presenting an unregistered identity.
        """
        entry = self.register.get(device_id)
        return bool(entry and entry.get("life_critical", False))

    # -- FR6: band mapping ---------------------------------------------------
    def _band(self, cert_valid: bool, score: float) -> tuple[Band, list[str]]:
        if not cert_valid:
            return "REVOKE", ["certificate invalid"]
        if score >= self.permit_above:
            return "PERMIT", [f"behaviour score {score:.2f} within baseline"]
        if score >= self.revoke_below:
            return "RESTRICT", [f"behaviour score {score:.2f} below normal range"]
        return "REVOKE", [f"behaviour score {score:.2f} inconsistent with device baseline"]

    # -- FR6 + FR7 + FR8 + FR9 ----------------------------------------------
    def decide(self, device_id: str, cert_valid: bool, score: float) -> Decision:
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0,1], got {score}")

        band, reasons = self._band(cert_valid, score)
        life_critical = self.resolve_tier(device_id)
        constraint_applied = False
        retained_path = None
        escalated = False

        # FR7 / FR8: a life-critical device is never fully isolated here.
        if band == "REVOKE" and life_critical and self.safety_constraint:
            band = "CONTAIN"
            constraint_applied = True
            retained_path = f"{device_id} -> patient_server_01:443"
            escalated = True  # FR9
            reasons.append("life-critical: containment applied instead of revocation")
            reasons.append("escalated to clinical engineering")

        return Decision(
            device_id=device_id,
            band=band,
            constraint_applied=constraint_applied,
            reasons=tuple(reasons),
            retained_path=retained_path,
            escalated=escalated,
        )
