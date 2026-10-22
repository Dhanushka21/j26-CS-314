"""Component 3 service (contracts README: C3 = port 8003, POST /analyze).

/analyze accepts either:
  - C2ToC3 (detection result from Component 2), or
  - C1ToC3 (identity-compromise event from Component 1, recognised by its `alert_type` field).

Run from the project folder:  uvicorn src.app:create_app --factory --port 8003
"""
import json
import logging
import time

import joblib
import requests
from fastapi import Body, FastAPI, HTTPException
from pydantic import ValidationError

from . import config
from .contract_loader import C1ToC3, C2ToC3, N_FEATURES
from .explain import Explainer, summarize
from .schemas import Alert
from .severity import SeverityEngine

log = logging.getLogger("component3")


def _bad_request(e: ValidationError):
    return HTTPException(422, json.loads(e.json(include_url=False)))


def create_app(model_path: str = None, forward_url: str = None, forward: bool = True,
               severity_mode: str = None) -> FastAPI:
    app = FastAPI(title="Component 3: XAI and Severity")

    model = joblib.load(model_path or config.MODEL_PATH)
    explainer = Explainer(model, top_k=config.TOP_K_FEATURES)
    engine = SeverityEngine(mode=severity_mode)
    target = forward_url or config.COMPONENT4_URL

    def send(alert: Alert):
        """Returns (forwarded, error, Component 4's reply body if it sent JSON)."""
        if not forward:
            return False, None, None
        try:
            r = requests.post(target, json=alert.model_dump(exclude_none=True), timeout=config.COMPONENT4_TIMEOUT_S)
            try:
                body = r.json()
            except ValueError:
                body = None
            return r.ok, (None if r.ok else f"component4 returned {r.status_code}"), body
        except requests.RequestException as e:
            log.warning("could not reach Component 4: %s", e)
            return False, str(e), None

    def respond(alert: Alert, t0: float):
        forwarded, err, c4_body = send(alert)
        return {"status": "processed", "processing_ms": round((time.perf_counter() - t0) * 1000, 2),
                "forwarded": forwarded, "forward_error": err, "component4_response": c4_body,
                "alert": alert.model_dump(exclude_none=True)}

    @app.get("/health")
    def health():
        return {"status": "ok", "features": N_FEATURES, "severity_mode": engine.mode}

    @app.post("/analyze")
    def analyze(body: dict = Body(...)):
        t0 = time.perf_counter()

        if "alert_type" in body:                                   # C1 -> C3 identity compromise
            try:
                c1 = C1ToC3(**body)
            except ValidationError as e:
                raise _bad_request(e)
            severity, flag = engine.decide_identity(c1.device_id)
            alert = Alert(
                device_id=c1.device_id, timestamp=c1.timestamp, attack_type=c1.alert_type,
                confidence=1.0, severity=severity, critical_protection_flag=flag,
                shap_top_features=[],        # no ML model behind this alert, so no SHAP values are invented
                explanation_summary=(f"Identity compromise reported by the Zero-Trust layer "
                                     f"(certificate valid: {c1.cert_valid}, behaviour score: {c1.behavior_score})."),
                source_ip=c1.source_ip,
            )
            return respond(alert, t0)

        try:                                                       # C2 -> C3 detection result
            d = C2ToC3(**body)
        except ValidationError as e:
            raise _bad_request(e)

        if d.predicted_class == config.BENIGN_LABEL:
            return {"status": "skipped_benign"}                    # Component 4 wants attacks only
        if d.predicted_class not in explainer.classes:
            raise HTTPException(422, f"detector model has no class '{d.predicted_class}'")

        top, shap_mag = explainer.analyse(d.features, d.predicted_class)
        probs = sorted(d.probabilities.values(), reverse=True)
        margin = (probs[0] - probs[1]) if len(probs) > 1 else 1.0
        severity, flag = engine.decide(d.predicted_class, d.confidence, shap_mag, d.device_id, margin)

        alert = Alert(
            device_id=d.device_id, timestamp=d.timestamp, attack_type=d.predicted_class,
            confidence=d.confidence, severity=severity, critical_protection_flag=flag,
            shap_top_features=top, explanation_summary=summarize(d.predicted_class, top),
            source_ip=d.source_ip,
        )
        return respond(alert, t0)

    return app
