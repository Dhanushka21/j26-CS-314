"""
C1 end-to-end: Function 1 (behavioural score) -> Function 2 (access decision).

Emits payloads validated against the shared team contracts:
  C1ToC2  every verified session handed to detection
  C1ToC3  identity-compromise alerts
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from contracts.models import C1ToC2, C1ToC3  # noqa: E402
from baseline import FEATURES  # noqa: E402
from decision import DecisionEngine  # noqa: E402

DEVICE_MAP: dict[str, dict] = json.loads(
    (_ROOT / "config" / "device_map.json").read_text())["map"]
_T0 = datetime(2026, 10, 20, 10, 0, 0, tzinfo=timezone.utc)


def load_baselines(path: Path | None = None) -> dict:
    return joblib.load(path or _ROOT / "models" / "device_baselines.joblib")


def to_contract_device(capture_device: str) -> str:
    entry = DEVICE_MAP.get(capture_device)
    return entry["device_id"] if entry else capture_device


def _ts(i: int) -> str:
    return (_T0 + timedelta(seconds=30 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(windows: pd.DataFrame, baselines: dict, engine: DecisionEngine,
        claimed_device: str | None = None, cert_valid: bool = True,
        source_ip: str = "10.0.0.7") -> list[dict]:
    """Score each window against the baseline of the device it claims to be.

    `claimed_device` overrides the capture's own label - that is the
    impersonation case: traffic from somewhere else claiming to be this device.
    """
    results: list[dict] = []
    for i, (_, row) in enumerate(windows.iterrows()):
        capture_device = claimed_device or row["device"]
        if capture_device not in baselines:
            continue
        score = float(baselines[capture_device].score(pd.DataFrame([row]))[0])

        device_id = to_contract_device(capture_device)
        decision = engine.decide(device_id, cert_valid, score)

        payload = C1ToC2(
            session_id=f"sess-{i:04d}",
            device_id=device_id,
            timestamp=_ts(i),
            cert_valid=cert_valid,
            behavior_score=round(score, 4),
            access_decision=decision.band if decision.band != "CONTAIN" else "RESTRICT",
            features=[float(row[f]) for f in FEATURES],
        )

        alert = None
        if cert_valid and decision.band in ("REVOKE", "CONTAIN"):
            alert = C1ToC3(
                device_id=device_id,
                timestamp=_ts(i),
                alert_type="IDENTITY_COMPROMISE",
                cert_valid=True,
                behavior_score=round(score, 4),
                source_ip=source_ip,
            ).model_dump()

        results.append({
            "session_id": payload.session_id,
            "device_id": device_id,
            "capture_device": capture_device,
            "score": score,
            "decision": decision.band,
            "constraint_applied": decision.constraint_applied,
            "escalated": decision.escalated,
            "reasons": list(decision.reasons),
            "c1_to_c2": payload.model_dump(),
            "c1_to_c3": alert,
        })
    return results
