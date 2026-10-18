"""Loads the best trained model (models/best_model.json) and classifies ONE flow at a time.

- Input  : the 45 raw features of one flow (features.json order), as sent by C1.
- Scaling: the SAME scaler.joblib + transform() used in training (Step 6).
- Features: IAT is dropped exactly like in training (config.MODEL_FEATURE_IDX).
- LSTM   : needs the last 10 flows, so a rolling buffer is kept per device_id.
           Until a device has sent 10 flows, the window is left-padded with its first flow.
"""
import json
from collections import defaultdict, deque

import joblib
import numpy as np
import torch

import config as C
from src.nets import LSTMClassifier, MLP
from src.preprocess import transform


class Predictor:
    def __init__(self):
        info = json.loads((C.MODELS_DIR / "best_model.json").read_text())
        self.kind = info["model"]                                   # "rf" | "mlp" | "lstm"
        self.window = C.WINDOW_SIZE
        self.idx = C.MODEL_FEATURE_IDX
        self.scaler = joblib.load(C.MODELS_DIR / "scaler.joblib")
        self.buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))

        n_in, n_out = len(self.idx), len(C.CLASSES)
        if self.kind == "rf":
            self.model = joblib.load(C.MODELS_DIR / "rf.joblib")
            self.model.n_jobs = 1
        else:
            net = MLP(n_in, n_out) if self.kind == "mlp" else LSTMClassifier(n_in, n_out)
            net.load_state_dict(torch.load(C.MODELS_DIR / f"{self.kind}.pt", map_location="cpu"))
            self.model = net.eval()
        # warm-up: the very first PyTorch call is slow (~1 s); do it now, not on the first request
        self.predict("_warmup", [0.0] * C.N_FEATURES)
        self.buffers.pop("_warmup", None)
        print(f"[C2] serving {self.kind.upper()}  (weighted-F1={info['weighted_f1']:.4f})")

    def predict(self, device_id: str, features: list[float]) -> np.ndarray:
        """Return 19 class probabilities for the newest flow of `device_id`."""
        x = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        buf = self.buffers[device_id]
        buf.append(x)

        if self.kind == "lstm":
            seq = list(buf)
            seq = [seq[0]] * (self.window - len(seq)) + seq           # left-pad to 10 flows
            inp = transform(np.stack(seq)[None, ...], self.scaler)    # (1, 10, 45)
            inp = inp[:, :, self.idx]                                 # (1, 10, 44)
        else:
            inp = transform(x[None, :], self.scaler)[:, self.idx]     # (1, 44)

        if self.kind == "rf":
            return self.model.predict_proba(inp)[0]
        with torch.no_grad():
            return torch.softmax(self.model(torch.from_numpy(inp)), dim=1)[0].numpy()