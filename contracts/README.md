# Contracts (shared JSON formats)

Single source of truth for data exchanged between components. **Change a field = tell all 4 members + update `models.py`, `samples/`.**

## Flow
```
C1 --(c1_to_c2)--> C2 --(c2_to_c3)--> C3 --(c3_to_c4)--> C4
C1 --(c1_to_c3: identity compromise)--> C3
C1 --(c1_to_c4: lateral movement)------> C4   (bypasses C3)
```

## Endpoints (proposed - confirm in group)
| Component | Owner | Port | Endpoint |
|---|---|---|---|
| C1 Zero-Trust | Dhanushka | 8001 | `POST /verify` |
| C2 Detection | Shanthasiri | 8002 | `POST /classify` (accepts `C1ToC2`, returns `C2ToC3`) |
| C3 XAI/Severity | Sasangith | 8003 | `POST /analyze` (accepts `C2ToC3` or `C1ToC3`) |
| C4 SDN Mitigation | Dulsara | 8004 | `POST /alert` (accepts `C3ToC4` or `C1ToC4`) |

(C4 proposal says Flask port 5000; if Dulsara keeps 5000, just update this table.)

## Files
- `models.py` - Pydantic models (import these)
- `classes.json` - 19 class names, fixed order (from CICIoMT2024 file names)
- `features.json` - 45 flow features, fixed CSV column order
- `devices.json` - device ids, protection priority, life-critical flag (from C4 proposal)
- `samples/*.json` - example payload per contract

## Rules
1. Timestamps: ISO 8601 UTC (`2026-10-14T10:00:00Z`).
2. `features` always 45 floats in `features.json` order.
3. Class names exactly as in `classes.json`. `severity`: Critical/High/Medium/Low. `recommended_action`: DROP/REDIRECT/METER/LOG.
4. `device_id` exactly as in `devices.json`. Life-critical (ventilator, insulin pump, infusion pump): C3 forces Critical, C4 caps action at METER.
5. CICIoMT2024 CSVs have **no device/source column**, so `device_id` / `source_ip` come from the simulated topology (Mininet), not the dataset.

## Usage
```python
from contracts.models import C2ToC3
payload = C2ToC3(**data)          # validates
payload.model_dump()              # back to dict
```
Run from repo root so `contracts` is importable (`pip install pydantic`).
