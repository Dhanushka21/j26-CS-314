import requests
from train_standin import synthetic

df = synthetic(n_per_class=2)
row = df[df.label == "TCP_IP-DDoS-SYN"].iloc[0].drop("label")

for device in ["Wearable-07", "Vent-01"]:
    r = requests.post("http://127.0.0.1:8000/api/v1/detection", json={
        "device_id": device,
        "timestamp": "2026-10-18T10:00:00Z",
        "feature_vector": [float(v) for v in row],
        "predicted_class": "TCP_IP-DDoS-SYN",
        "confidence": 0.98,
    })
    print(device, r.json())