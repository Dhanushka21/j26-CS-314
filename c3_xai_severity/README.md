# Component 3: XAI and Severity (IT4010 J-26-CS-314)

Flow: Component 2 -> POST /api/v1/detection (port 8000) -> SHAP + severity + life-critical override -> POST to Component 4 (port 5000).

## Steps
1. `pip install -r requirements.txt`
2. `python train_standin.py` (synthetic data; add `--csv file.csv --label-col <name>` for real CICIoMT2024)
3. `python -m pytest -q -s` (11 tests, includes latency and the safety override)
4. `uvicorn mock_component4:app --port 5000` (terminal A)
5. `uvicorn src.app:create_app --factory --port 8000` (terminal B)
6. Send a detection to http://127.0.0.1:8000/api/v1/detection (see tests/test_api.py `payload` for the shape)
7. ML severity: `python train_severity.py`, then run the service with `C3_SEVERITY_MODE=ml`
8. Integrate: set `C3_COMPONENT4_URL=http://<DULSARA_IP>:5000/api/v1/alert`; run `python send_test_alerts.py --url <same>`
9. Swap in Component 2's real model: set `C3_MODEL_PATH` (tree models work as-is; CNN/LSTM needs DeepExplainer in src/explain.py)

## Must verify
- Label spellings in src/taxonomy.py against the CSV and Component 2's output
- Attack -> severity table (a proposal) with your supervisor
- Device registry in src/config.py matches your simulated hospital
