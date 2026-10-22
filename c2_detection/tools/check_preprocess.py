"""STEP 5 check: scaling + SMOTE on the TEST windows (quick demo only).
The real run on TRAIN data happens in Step 6 (prepare_data.py).

Run from repo root:   python c2_detection/tools/check_preprocess.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                                # noqa: E402
from src.data import build_split                                  # noqa: E402
from src.preprocess import fit_scaler, transform, smote_windows   # noqa: E402


def main():
    X, y = build_split(C.TEST_DIR, C.WINDOW_SIZE, C.TEST_STRIDE, C.MAX_WINDOWS_PER_CLASS_TEST, C.SEED)

    # ---- 1. scaling ----
    i = C.FEATURES.index("Rate")
    print(f"\nRate BEFORE : min={X[..., i].min():,.0f}  max={X[..., i].max():,.0f}  mean={X[..., i].mean():,.0f}")
    scaler = fit_scaler(X)
    Xs = transform(X, scaler)
    print(f"Rate AFTER  : min={Xs[..., i].min():.2f}  max={Xs[..., i].max():.2f}  mean={Xs[..., i].mean():.2f}")
    flat = Xs.reshape(-1, Xs.shape[-1])
    print(f"All features after scaling: mean~{flat.mean():.3f}  std~{flat.std():.3f}  (expected ~0 and ~1)")
    print("shape unchanged:", X.shape == Xs.shape)

    # ---- 2. SMOTE ----
    target = 2000
    before = np.bincount(y, minlength=len(C.CLASSES))
    Xb, yb = smote_windows(Xs, y, target=target, seed=C.SEED)
    after = np.bincount(yb, minlength=len(C.CLASSES))
    print(f"\nSMOTE (target={target})")
    for name, b, a in zip(C.CLASSES, before, after):
        print(f"  {name:<26} {b:>6} -> {a:>6}")
    print("new X shape:", Xb.shape)


if __name__ == "__main__":
    main()