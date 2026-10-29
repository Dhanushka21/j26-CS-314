# C2 – AI Multi-Class IoMT Attack Detection

**Owner:** T.D. Shanthasiri (IT23304956)  
**Dataset:** CICIoMT2024 (Wi-Fi & MQTT) — 19 classes, 45 features  
**Best model:** Sliding-Window Random Forest (`rf_win150`) — 150-flow window → 484 window statistics

---

## Results (test set, IAT removed)

| Model | Accuracy | Weighted F1 | Macro F1 | Latency |
|---|---:|---:|---:|---:|
| Random Forest (no window) | 0.755 | 0.751 | 0.682 | ~4 ms |
| MLP | 0.751 | 0.747 | 0.659 | ~0.2 ms |
| LSTM (window=10) | 0.823 | 0.817 | 0.775 | ~1 ms |
| **RF Window (w=150) ✅ served** | **0.911** | **0.907** | **0.836** | ~10 ms |

5-fold CV: weighted F1 = **0.9532 ± 0.0071**  
Baseline (Yapa et al. 2025): 73.3 % → Target NFR3: 90 % weighted F1 ✅  
Load test (10,000 events): p95 latency = **99.09 ms** ✅ (NFR1: < 100 ms)

### Key finding — IAT removal
`IAT` encodes the capture timestamp, not attack behaviour.  
With IAT: RF reaches 0.969 weighted F1 (data leakage).  
Without IAT: RF drops to 0.751 — proving the leakage.  
All models drop IAT via `config.DROP_FEATURES = ["IAT"]`.

### How the window model works
Each device keeps a rolling buffer of the last 150 flows. Per-buffer, 11 statistics
(last, mean, std, min, max, delta — long + short window of 10) computed across
44 model features → **484-feature vector** → RF classifier.  
Before buffer fills (< 150 flows), warm-up model (`rf_win10`, window=10) is used.

---

## Quick start

### 1. Install dependencies
```bash
pip install -r c2_detection/requirements.txt
```

### 2. Download trained models
Trained model files are NOT in git (too large). Ask owner (IT23304956) for Drive link.

Place at:


### 3. Start the API
```bash
uvicorn c2_detection.api.main:app --port 8002 --reload
```
Swagger UI: http://127.0.0.1:8002/docs

### 4. Run tests
```bash
pytest c2_detection/tests/ -v
```

---

## API reference (for C1 / C3)

### POST /classify


**Request (C1ToC2):**
```json
{
  "session_id": "sess-0001",
  "device_id": "bp_monitor_01",
  "timestamp": "2026-10-29T08:00:00Z",
  "cert_valid": true,
  "behavior_score": 0.91,
  "access_decision": "PERMIT",
  "features": [0.0, "...45 values in features.json order"]
}
```

**Response (C2ToC3):**
```json
{
  "device_id": "bp_monitor_01",
  "timestamp": "2026-10-29T08:00:00Z",
  "predicted_class": "Benign",
  "confidence": 0.94,
  "probabilities": { "Benign": 0.94, "...19 classes" },
  "features": [0.0, "...45 raw values for C3 SHAP"],
  "source_ip": null
}
```

- `features` = raw 45 values — C3 must drop IAT (`config.MODEL_FEATURE_IDX`) before SHAP
- Buffer is **per `device_id`** — first ~150 flows use warm-up model
- `access_decision == "REVOKE"` → **403**

### GET /health
```json
{ "status": "ok", "model": "rf_win", "window_size": 150, "dropped_features": ["IAT"] }
```

---

## Reproduce experiments

```bash
python c2_detection/prepare_data.py
python c2_detection/train_rf.py
python c2_detection/train_window_rf.py
python c2_detection/train_nn.py --model mlp
python c2_detection/train_nn.py --model lstm
python c2_detection/compare.py
python c2_detection/cross_validate.py
pytest c2_detection/tests/ -v
```

Trained models not in git — ask owner for Drive link.