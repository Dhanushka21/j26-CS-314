# Component 3: XAI and Severity (IT4010 J-26-CS-314)

Keep this folder inside the group repo, next to `contracts/`. It loads classes, features, devices and the
Pydantic models from there, so nothing is copied.

Flow: Component 2 (or Component 1) -> `POST /analyze` (port 8003) -> SHAP + severity + life-critical override
-> `POST` to Component 4 (default `http://127.0.0.1:5000/api/v1/alert`).

## Run (from this folder, venv active)
1. `pip install -r requirements.txt`
2. `python train_standin.py` (synthetic; or `--csv file.csv --label-col <name>` for CICIoMT2024)
3. `python -m pytest -q -s` (20 tests)
4. Terminal A: `uvicorn mock_component4:app --port 5000`
5. Terminal B: `uvicorn src.app:create_app --factory --port 8003`
6. Terminal C: `python send_detection.py` (sends the group's sample payloads)
7. ML severity: `python train_severity.py`, then start B with `C3_SEVERITY_MODE=ml`
8. Real Component 4: `C3_COMPONENT4_URL=http://<DULSARA_IP>:5000/api/v1/alert`, then `python send_test_alerts.py --url <same>`
9. Component 2's real model: `C3_MODEL_PATH=<file>` (45 contract features in contract order; tree models work as-is)

## Open decisions for the group
- `contracts/models.py` C3ToC4 still requires `recommended_action`; C3 and C4 agreed to drop it.
- Contracts README says C4 is `POST /alert` on 8004; Dulsara confirmed 5000 `/api/v1/alert`.
- Base severity for C1 identity-compromise events (`C3_IDENTITY_SEVERITY`, default High) and the confidence value sent for them (1.0).
- Attack -> severity table in `src/taxonomy.py` (agree with supervisor).
