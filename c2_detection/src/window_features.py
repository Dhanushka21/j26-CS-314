"""Window-statistics features (Step 12).

Instead of feeding raw flows to the model, describe the last `w` flows of a device with
statistics per feature. This gives a tree model the sequence context (rate ramps, burstiness)
that a single flow cannot show.

Per window (IAT removed -> 44 features, signed-log first):
  long window  (w flows) : last, mean, std, min, max, delta(last - first)   -> 6 x 44
  short window (10 flows): mean, std, min, max, delta                       -> 5 x 44
  total = 11 x 44 = 484 values

Two implementations that give IDENTICAL numbers:
  file_window_stats()  - fast, pandas rolling over a whole CSV file (training)
  window_stats()       - numpy on one buffer of flows (API, one request at a time)
"""
import numpy as np
import pandas as pd

import config as C
from src.preprocess import signed_log

SHORT = 10


def _prep(x: np.ndarray) -> np.ndarray:
    """raw (n, 45) -> signed-log of the 44 model features."""
    return signed_log(np.asarray(x, dtype=np.float64)[:, C.MODEL_FEATURE_IDX])


def file_window_stats(x_raw: np.ndarray, w: int) -> np.ndarray:
    """All windows of one file at once. Row i describes flows [i, i+w). Returns (n-w+1, 484).
    Each statistic is computed and converted to float32 one at a time to keep RAM low."""
    df = pd.DataFrame(_prep(x_raw))
    long_, short = df.rolling(w), df.rolling(SHORT)
    makers = [lambda: df, long_.mean, long_.std, long_.min, long_.max, lambda: df - df.shift(w - 1),
              short.mean, short.std, short.min, short.max, lambda: df - df.shift(SHORT - 1)]
    parts = [m().to_numpy(dtype=np.float32)[w - 1:] for m in makers]
    return np.nan_to_num(np.concatenate(parts, axis=1))


def window_stats(flows_raw: np.ndarray) -> np.ndarray:
    """One window (w, 45) -> (484,). Same maths as file_window_stats (std uses ddof=1)."""
    a = _prep(flows_raw)
    s = a[-SHORT:]
    feats = [a[-1], a.mean(0), a.std(0, ddof=1), a.min(0), a.max(0), a[-1] - a[0],
             s.mean(0), s.std(0, ddof=1), s.min(0), s.max(0), s[-1] - s[0]]
    return np.nan_to_num(np.concatenate(feats)).astype(np.float32)