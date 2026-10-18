"""C2 Detection REST API  (port 8002)

  POST /classify   body = C1ToC2  ->  returns C2ToC3     (contracts/models.py)
  GET  /health     status + which model is served

Run from repo root:
  uvicorn c2_detection.api.main:app --port 8002
Swagger UI (try it in the browser):  http://127.0.0.1:8002/docs

Optional security (NFR2): set env C2_API_KEY=<secret>  -> callers must send header  X-API-Key: <secret>
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # c2_detection/ on path
import config as C  # noqa: E402  (also puts the repo root on the path for `contracts`)

import numpy as np  # noqa: E402
from fastapi import FastAPI, Header, HTTPException, Request  # noqa: E402

from api.predictor import Predictor  # noqa: E402
from contracts.models import C1ToC2, C2ToC3  # noqa: E402

app = FastAPI(title="IoMT Shield - C2 Detection", version="0.1.0")
predictor = Predictor()
API_KEY = os.getenv("C2_API_KEY")


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Every response carries X-Process-Time-ms -> proof for the < 100 ms requirement (NFR1)."""
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.kind, "window_size": C.WINDOW_SIZE,
            "dropped_features": C.DROP_FEATURES}


@app.post("/classify", response_model=C2ToC3)
def classify(flow: C1ToC2, x_api_key: str | None = Header(default=None)):
    # 1. security
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")
    # 2. C1 already revoked this session -> do not classify
    if flow.access_decision == "REVOKE":
        raise HTTPException(status_code=403, detail="session revoked by C1")

    # 3. predict
    proba = predictor.predict(flow.device_id, flow.features)
    idx = int(np.argmax(proba))

    # 4. build the C2 -> C3 contract
    return C2ToC3(
        device_id=flow.device_id,
        timestamp=flow.timestamp,
        predicted_class=C.CLASSES[idx],
        confidence=float(proba[idx]),
        probabilities={c: round(float(p), 6) for c, p in zip(C.CLASSES, proba)},
        features=flow.features,     # raw 45 values in features.json order (C3 needs them for SHAP)
        source_ip=None,             # C1ToC2 has no source_ip yet (comes from the Mininet topology)
    )