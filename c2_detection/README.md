# C2 – AI Multi-Class Attack Detection

**Owner:** T.D. Shanthasiri (IT23304956)
RF vs MLP vs LSTM on CICIoMT2024 (Wi-Fi & MQTT, 19 classes). Best model served by FastAPI.

## Results (test set, IAT removed)
| Model | Accuracy | Weighted F1 | Macro F1 | Latency |
|---|---:|---:|---:|---:|
| Random Forest | 0.755 | 0.751 | 0.682 | ~4 ms |
| MLP | 0.751 | 0.747 | 0.659 | ~0.2 ms |
| **LSTM (served)** | **0.823** | **0.817** | **0.775** | ~1 ms |

Baseline (Yapa et al. 2025): 73.3 %. Target: 90 % weighted F1.

**IAT finding:** `IAT` encodes the capture time, not attack behaviour. With IAT, RF reaches 0.969 weighted F1,
but without it RF drops to 0.751, so IAT is dropped from all models (`config.DROP_FEATURES`).

## API (for C1 / C3)
```
uvicorn c2_detection.api.main:app --port 8002
POST http://<host>:8002/classify   body: C1ToC2   ->   returns: C2ToC3
GET  http://<host>:8002/health
Swagger: http://127.0.0.1:8002/docs
```
- `features` in the response = the **raw 45 values** (features.json order). C3 must scale them with
  `models/scaler.joblib` + `src.preprocess.transform()` and drop IAT (`config.MODEL_FEATURE_IDX`) before SHAP.
- LSTM keeps the last 10 flows **per `device_id`**; the first ~10 flows of a device are less accurate.
- `access_decision == "REVOKE"` -> 403. Optional API key: env `C2_API_KEY` + header `X-API-Key`.

## Reproduce
```
pip install -r c2_detection/requirements.txt
# put CICIoMT2024 "WIFI & MQTT/attacks/CSV" at c2_detection/data/raw/CSV  (Train/ 51 files, Test/ 21 files)
python c2_detection/tools/explore_data.py
python c2_detection/prepare_data.py
python c2_detection/train_rf.py
python c2_detection/train_nn.py --model mlp
python c2_detection/train_nn.py --model lstm
python c2_detection/compare.py
pytest -q c2_detection/tests
```
Trained models (`models/*.pt`, `*.joblib`) are not in git; ask the owner for the Drive link.