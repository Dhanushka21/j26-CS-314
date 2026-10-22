"""STEP 6: CICIoMT2024 CSVs -> scaled, balanced sliding windows, saved to disk.

Run from repo root:   python c2_detection/prepare_data.py

Pipeline:
  1. build TRAIN windows (official Train/ files)   - stride 5, cap 10,000 per class
  2. build TEST  windows (official Test/ files)    - stride 10, cap 5,000 per class
  3. split 10% of TRAIN off as VALIDATION (stratified)
  4. fit scaler on TRAIN only -> apply to train / val / test
  5. SMOTE on TRAIN only
  6. save  data/processed/dataset.npz, data/processed/meta.json, models/scaler.joblib

All three models use these SAME windows:
  LSTM -> the full window (10, 45);  MLP / RF -> the LAST flow of each window (45,)
"""
import json
import time

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

import config as C
from src.data import build_split
from src.preprocess import fit_scaler, smote_windows, transform


def counts(y):
    return np.bincount(y, minlength=len(C.CLASSES))


def main():
    t0 = time.time()
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1 + 2. windows
    print("== 1. TRAIN windows")
    x_tr, y_tr = build_split(C.TRAIN_DIR, C.WINDOW_SIZE, C.TRAIN_STRIDE,
                             C.MAX_WINDOWS_PER_CLASS_TRAIN, C.SEED)
    print("\n== 2. TEST windows (never used for training)")
    x_te, y_te = build_split(C.TEST_DIR, C.WINDOW_SIZE, C.TEST_STRIDE,
                             C.MAX_WINDOWS_PER_CLASS_TEST, C.SEED)

    # 3. validation split (stratified = same class mix in train and val)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_tr, y_tr, test_size=C.VAL_FRACTION, stratify=y_tr, random_state=C.SEED)
    print(f"\n== 3. train={len(x_tr):,}  val={len(x_val):,}  test={len(x_te):,}")

    # 4. scaling - fit on TRAIN only
    scaler = fit_scaler(x_tr)
    joblib.dump(scaler, C.MODELS_DIR / "scaler.joblib")
    x_tr, x_val, x_te = transform(x_tr, scaler), transform(x_val, scaler), transform(x_te, scaler)
    print("== 4. scaler fitted on train, saved to models/scaler.joblib")

    # 5. SMOTE - TRAIN only
    print("== 5. SMOTE (train only)")
    before = counts(y_tr)
    x_tr, y_tr = smote_windows(x_tr, y_tr, C.SMOTE_TARGET_PER_CLASS, C.SEED)
    after = counts(y_tr)
    print(f"  {'class':<26} {'before':>7} {'after':>7} {'val':>6} {'test':>6}")
    for name, b, a, v, t in zip(C.CLASSES, before, after, counts(y_val), counts(y_te)):
        print(f"  {name:<26} {b:>7} {a:>7} {v:>6} {t:>6}")

    # 6. save
    np.savez(C.PROCESSED_DIR / "dataset.npz",
             X_train=x_tr, y_train=y_tr, X_val=x_val, y_val=y_val, X_test=x_te, y_test=y_te)
    meta = {
        "window_size": C.WINDOW_SIZE, "train_stride": C.TRAIN_STRIDE, "test_stride": C.TEST_STRIDE,
        "classes": C.CLASSES, "features": C.FEATURES,
        "train_before_smote": before.tolist(), "train_after_smote": after.tolist(),
        "val": counts(y_val).tolist(), "test": counts(y_te).tolist(),
        "shapes": {"train": list(x_tr.shape), "val": list(x_val.shape), "test": list(x_te.shape)},
    }
    (C.PROCESSED_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    # small copy in results/ so it can be committed and used in the report
    (C.RESULTS_DIR / "dataset_summary.json").write_text(json.dumps(meta, indent=2))

    print(f"\n== 6. saved  train={x_tr.shape}  val={x_val.shape}  test={x_te.shape}")
    print(f"   total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()