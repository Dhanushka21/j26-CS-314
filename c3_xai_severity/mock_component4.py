"""Stand-in for Dulsara's Component 4 so you can test before you have his IP.

  uvicorn mock_component4:app --port 5000

It applies the severity -> action mapping he confirmed, so you can check what he should do.
"""
from fastapi import FastAPI

app = FastAPI(title="Mock Component 4")
RECEIVED = []


def action_for(severity: str, flag: bool) -> str:
    if severity == "Critical" and flag:
        return "rate-limit inbound only (NO isolation)"
    if severity == "Critical":
        return "rate-limit + block"
    if severity == "High":
        return "rate-limit + redirect to honeypot"
    if severity == "Medium":
        return "rate-limit"
    return "monitor only"


@app.post("/api/v1/alert")
def alert(payload: dict):
    RECEIVED.append(payload)
    act = action_for(payload.get("severity"), bool(payload.get("critical_protection_flag")))
    print(f"[C4 mock] {payload.get('device_id')} {payload.get('attack_type')} "
          f"sev={payload.get('severity')} flag={payload.get('critical_protection_flag')} -> {act}")
    return {"received": True, "action": act}


@app.get("/received")
def received():
    return RECEIVED
