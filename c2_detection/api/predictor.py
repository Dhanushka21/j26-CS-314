"""Loads the best trained model (models/best_model.json) and classifies ONE flow at a time.

- Input   : the 45 raw features of one flow (features.json order), as sent by C1.
- Per device a rolling buffer of the most recent flows is kept (device_id).
- "rf_win" (final model, Step 13):
      buffer has >= W flows (W = 150) -> window statistics of the last W flows -> rf_win<W>
      buffer has <  W flows (warm-up)  -> window statistics of the last 10 flows -> rf_win10
      (fewer than 10 flows: the first flow is repeated to fill 10)
- "lstm" / "mlp" / "rf" (Step 10 models) are still supported.
"""
import json
from collections import defaultdict, deque

import joblib
import numpy as np
import torch

import config as C
from src.nets import LSTMClassifier, MLP
from src.preprocess import transform
from src.window_features import SHORT, window_stats


def _pad(flows: list, n: int) -> np.ndarray:
    """Left-pad a list of flows to length n by repeating the first one."""
    return np.stack([flows[0]] * (n - len(flows)) + flows)


class Predictor:
    def __init__(self):
        info = json.loads((C.MODELS_DIR / "best_model.json").read_text())
        self.kind = info["model"]                         # "rf_win" | "lstm" | "mlp" | "rf"
        self.window = int(info.get("window", C.WINDOW_SIZE))
        self.idx = C.MODEL_FEATURE_IDX
        self.buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))

        if self.kind == "rf_win":
            self.model = joblib.load(C.MODELS_DIR / f"rf_win{self.window}.joblib")
            self.warm_model = joblib.load(C.MODELS_DIR / f"rf_win{SHORT}.joblib")
            self.model.n_jobs = self.warm_model.n_jobs = 1
        else:
            self.scaler = joblib.load(C.MODELS_DIR / "scaler.joblib")
            n_in, n_out = len(self.idx), len(C.CLASSES)
            if self.kind == "rf":
                self.model = joblib.load(C.MODELS_DIR / "rf.joblib")
                self.model.n_jobs = 1
            else:
                net = MLP(n_in, n_out) if self.kind == "mlp" else LSTMClassifier(n_in, n_out)
                net.load_state_dict(torch.load(C.MODELS_DIR / f"{self.kind}.pt", map_location="cpu"))
                self.model = net.eval()

        # warm-up call so the first real request is not slow
        self.predict("_warmup", [0.0] * C.N_FEATURES)
        self.buffers.pop("_warmup", None)
        print(f"[C2] serving {self.kind.upper()} window={self.window}  (weighted-F1={info['weighted_f1']:.4f})")

    def predict(self, device_id: str, features: list[float]) -> np.ndarray:
        """Return 19 class probabilities for the newest flow of `device_id`."""
        x = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        buf = self.buffers[device_id]
        buf.append(x)
        flows = list(buf)

        if self.kind == "rf_win":
            if len(flows) >= self.window:                       # full long window
                stats = window_stats(np.stack(flows[-self.window:]))
                return self.model.predict_proba(stats[None, :])[0]
            short = flows[-SHORT:]                              # warm-up: short window model
            stats = window_stats(_pad(short, SHORT))
            return self.warm_model.predict_proba(stats[None, :])[0]

        if self.kind == "lstm":
            seq = _pad(flows[-C.WINDOW_SIZE:], C.WINDOW_SIZE)
            inp = transform(seq[None, ...], self.scaler)[:, :, self.idx]   # (1, 10, 44)
        else:
            inp = transform(x[None, :], self.scaler)[:, self.idx]          # (1, 44)

        if self.kind == "rf":
            return self.model.predict_proba(inp)[0]
        with torch.no_grad():
            return torch.softmax(self.model(torch.from_numpy(inp)), dim=1)[0].numpy()