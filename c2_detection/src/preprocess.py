"""Normalisation + SMOTE for C2.

The SAME transform() is used in training AND later in the API, so a live flow
is scaled exactly like the training data.
"""
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Normalisation
# ---------------------------------------------------------------------------
def signed_log(x: np.ndarray) -> np.ndarray:
    """log(1 + |x|) keeping the sign.
    Rate / Header_Length / IAT go up to millions; log squashes them so one
    huge feature does not dominate the others."""
    return np.sign(x) * np.log1p(np.abs(x))


def fit_scaler(x_windows: np.ndarray) -> StandardScaler:
    """Learn mean/std of every feature from the TRAIN windows only."""
    flat = x_windows.reshape(-1, x_windows.shape[-1])        # (n*10, 45)
    return StandardScaler().fit(signed_log(flat))


def transform(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """signed_log + standardise. Works for rows (n, 45) or windows (n, 10, 45)."""
    shape = x.shape
    out = scaler.transform(signed_log(x.reshape(-1, shape[-1])))
    return out.reshape(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. SMOTE (train set only!)
# ---------------------------------------------------------------------------
def smote_windows(x: np.ndarray, y: np.ndarray, target: int, seed: int):
    """Oversample every class that has fewer than `target` windows up to `target`.

    SMOTE works on 2-D data, so each window (10, 45) is flattened to 450 values,
    SMOTE creates new windows between neighbouring windows of the same class,
    and the result is reshaped back to (n, 10, 45).
    Classes already >= target are not touched."""
    n, w, f = x.shape
    flat = x.reshape(n, w * f)
    counts = np.bincount(y)

    strategy = {c: target for c, k in enumerate(counts) if 0 < k < target}
    if not strategy:
        return x, y

    # SMOTE needs at least 2 samples (it interpolates between neighbours).
    # Classes with < 3 windows are just duplicated instead.
    tiny = {c: t for c, t in strategy.items() if counts[c] < 3}
    normal = {c: t for c, t in strategy.items() if c not in tiny}
    if tiny:
        flat, y = RandomOverSampler(sampling_strategy=tiny, random_state=seed).fit_resample(flat, y)
    if normal:
        k = int(min(5, min(counts[c] for c in normal) - 1))   # k_neighbors must be < class size
        flat, y = SMOTE(sampling_strategy=normal, k_neighbors=k,
                        random_state=seed).fit_resample(flat, y)
    return flat.reshape(-1, w, f).astype(np.float32), y