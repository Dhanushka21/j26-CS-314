"""Component 3 service.

Run:  uvicorn src.app:create_app --factory --port 8000
"""
import logging
import time

import joblib
import requests
from fastapi import FastAPI, HTTPException

from . import config
from .explain import Explainer, summarize
from .schemas import Alert, Detection
from .severity import SeverityEngine

log = logging.getLogger("component3")


def create_app(model_path: str = None, forward_url: str = None, forward: bool = True,
               severity_mode: str = None) -> FastAPI:
    app = FastAPI(title="Component 3: XAI and Severity")

    model = joblib.load(model_path or config.MODEL_PATH)
    explainer = Explainer(model, top_k=config.TOP_K_FEATURES)
    engine = SeverityEngine(mode=severity_mode)
    n_features = len(explainer.names)
    target = forward_url or config.COMPONENT4_URL

    @app.get("/health")
    def health():
        return {"status": "ok", "features": n_features, "severity_mode": engine.mode}

    @app.post("/api/v1/detection")
    def detection(d: Detection):
        t0 = time.perf_counter()

        if d.predicted_class == config.BENIGN_LABEL:
            return {"status": "skipped_benign"}        # Component 4 wants attacks only
        if len(d.feature_vector) != n_features:
            raise HTTPException(422, f"feature_vector must have {n_features} values, got {len(d.feature_vector)}")
        if d.predicted_class not in explainer.classes:
            raise HTTPException(422, f"unknown predicted_class '{d.predicted_class}'")

        top = explainer.top_features(d.feature_vector, d.predicted_class)
        shap_mag = explainer.shap_magnitude(d.feature_vector, d.predicted_class)
        severity, flag = engine.decide(d.predicted_class, d.confidence, shap_mag, d.device_id)

        alert = Alert(
            device_id=d.device_id, timestamp=d.timestamp, attack_type=d.predicted_class,
            confidence=d.confidence, severity=severity, critical_protection_flag=flag,
            shap_top_features=top, explanation_summary=summarize(d.predicted_class, top),
        )
        processing_ms = round((time.perf_counter() - t0) * 1000, 2)

        forwarded, forward_error = False, None
        if forward:
            try:
                r = requests.post(target, json=alert.model_dump(), timeout=config.COMPONENT4_TIMEOUT_S)
                forwarded = r.ok
                if not r.ok:
                    forward_error = f"component4 returned {r.status_code}"
            except requests.RequestException as e:
                forward_error = str(e)
                log.warning("could not reach Component 4: %s", e)

        return {"status": "processed", "processing_ms": processing_ms, "forwarded": forwarded,
                "forward_error": forward_error, "alert": alert.model_dump()}

    return app
